"""GPU execution entrypoint for EXP-2026-NVS-001 (plan v1.4).

Runs in the real GPU/vLLM environment. `extract_hidden_states` and
`RealGenerator.generate()` are implemented against vLLM 0.29's real, verified
API -- see `_VLLMHiddenStateEngine` below for exactly what was verified and
how, on an A40 GPU, originally with `meta-llama/Meta-Llama-3.1-8B-Instruct`
(confirmed 2026-09-12) and reconfirmed with the current model, `Qwen/
Qwen2.5-7B-Instruct` (v1.4 U8, 2026-09-13, after Rule 3 failed twice under
Llama -- see `MODEL_NAME`'s comment for the decision record). vLLM's Qwen2
decoder layer (`vllm/model_executor/models/qwen2.py`, `Qwen2DecoderLayer.
forward`) returns the identical `(hidden_states, residual)` shape as Llama's,
so the same hook mechanism applies unchanged.

Two things this script deliberately does NOT do, both corrections to an
earlier draft:

1. It never silently substitutes synthetic data for real calibration/pilot
   output. If real hidden-state extraction cannot run (no GPU/vLLM, or the
   model/engine fails to load), `extract_hidden_states` raises a loud
   `RuntimeError` rather than writing fabricated numbers to
   `configs/W_pca128.npy` / `results/phase1_pilot_*` and reporting them as
   real SHA-256 hashes. For a pre-registered experiment, a loud failure here
   is much cheaper than a quiet fabrication.

2. It does not compute a Phase 1 mixed-rate / Gate-3-style verdict during
   collection. Every Type A response that isn't Type C/D comes back labeled
   `"pending_type_a"` (§5.2's positive-determination method is deliberately
   undecided until real generation samples are reviewed -- see
   `eval_hallucination.classify_type_a`). "Mixed rate" is defined in terms of
   positive vs. negative samples per prompt (§5.4); with no positive rule
   yet, that quantity cannot be computed, let alone gated on (Rule 3, §8).
   Collection and labeling are therefore two separate steps here: this
   script collects and saves everything (text, hidden states, kappa,
   diagnostic signals) with `pending_type_a` left as `pending_type_a`; a
   later, separate pass reviews real samples, encodes the decided rule in
   `classify_type_a`, and only then computes mixed_rate/M against the saved
   records (no re-generation needed).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

# vLLM's V1 engine always runs the model in a spawned subprocess (even for
# tensor_parallel_size=1) and ships `apply_model` callables to it over an
# RPC channel. msgpack (the default) cannot serialize function objects at
# all; the documented escape hatch is VLLM_ALLOW_INSECURE_SERIALIZATION=1,
# which falls back to pickle. This process never receives untrusted input
# over that channel (it's a local parent/child pair this script itself
# spawns), so enabling it here is safe. Must be set before `vllm` is
# imported/used.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
# This environment's CUDA toolkit (nvcc 12.4) predates the flag
# (`--compress-mode=size`) that flashinfer's JIT sampler build emits for the
# installed torch/vLLM (cu130) pairing, so the JIT compile fails outright
# (confirmed 2026-09-12: `ninja -v` on the cached build reproduces
# `nvcc fatal: Unknown option '--compress-mode=size'`). Falling back to
# vLLM's non-flashinfer sampler avoids that build entirely; it changes which
# kernel does argmax/sampling, not model correctness.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

try:
    import torch
except ImportError:
    torch = None

try:
    import vllm
    from vllm import LLM, SamplingParams, TokensPrompt
except ImportError:
    vllm = None
    LLM = None
    SamplingParams = None
    TokensPrompt = None

import collect_trajectories
from collect_trajectories import LAYERS, GeneratedSample, TrajectoryCollector, collect
from eval_hallucination import EvalConfig, PromptRecord, PromptType
from train_w64 import SOURCE_DIMENSION, run_calibration

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
RESULTS = REPO_ROOT / "results"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # §2.1, v1.4 U8 (changed from
# meta-llama/Meta-Llama-3.1-8B-Instruct: Rule 3 (mixed_rate >= 30%) failed
# twice under Llama -- v1.3-worded pilot (69% abstain, 800 samples) and a
# v1.4-worded + t=1.0 diagnostic (94.5% abstain, 400 samples) -- so per U7 no
# further prompt/parameter tuning is permitted; the only remaining options
# were change model or stop the experiment. See docs/EXP-2026-NVS-001_v1.4.md
# changelog (U8) for the decision record.
STOP_TOKEN_STRINGS = ("<|im_end|>", "<|endoftext|>")  # Qwen2.5 chat-template
# turn-end / true EOS tokens (verified against the real tokenizer 2026-09-13),
# replacing Llama-3.1's <|eot_id|>/<|eom_id|>.


def compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def log_gpu_environment(output_path: Path = RESULTS / "env_info.json") -> dict[str, Any]:
    """Records hardware/library versions for FREEZE_MANIFEST.md / Appendix B item 1."""
    env_info: dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "python_version": sys.version,
        "torch_version": torch.__version__ if torch else None,
        "cuda_available": bool(torch and torch.cuda.is_available()),
        "vllm_version": getattr(vllm, "__version__", None) if vllm else None,
        "gpu_devices": [],
    }
    if torch and torch.cuda.is_available():
        env_info["cuda_version"] = torch.version.cuda
        env_info["gpu_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(env_info, indent=2) + "\n", encoding="utf-8")
    print(f"[+] recorded environment info: {output_path}")
    return env_info


def _gpu_available() -> bool:
    return torch is not None and torch.cuda.is_available() and vllm is not None


# ---------------------------------------------------------------------------
# vLLM hidden-state extraction
#
# vLLM's `generate()` returns only text/token ids; it does not expose
# per-layer intermediate hidden states as part of its public,
# generate()-facing API. Investigated and rejected: the EAGLE-3
# speculative-decoding machinery (`LlamaModel`/`EagleModelMixin`'s
# `aux_hidden_state_layers` + `_maybe_add_hidden_state`, vLLM 0.29,
# `vllm/model_executor/models/{llama,interfaces}.py`) computes exactly the
# hidden states we want at exactly this layer-index convention, but wiring
# it up requires the GPU model runner's `use_aux_hidden_state_outputs` flag
# (set only via the EAGLE3 speculative-decoding config path,
# `vllm/v1/worker/gpu_model_runner.py`); driving that path just to read out
# hidden states -- and fighting its assumption that a draft model consumes
# them -- is more surface area to get wrong than is justified here.
#
# What IS used instead, verified end-to-end against a plain HF Transformers
# `output_hidden_states=True` forward pass on this exact GPU (max cosine
# distance 8e-4 at layer 24, the deepest/most kernel-divergence-prone layer;
# essentially exact at layer 8 -- consistent with ordinary
# vLLM-vs-HF-Transformers kernel-implementation differences accumulating
# depth-wise, not a bug):
#
#   - `LLM.apply_model(func)` (public, documented) runs `func` on the actual
#     live `nn.Module` inside vLLM's model-execution worker and returns its
#     result.
#   - A standard `torch.nn.Module.register_forward_hook` on
#     `model.model.layers[L-1]` (the L-th decoder block, 0-indexed) captures
#     that layer's output on every forward call. vLLM's `LlamaDecoderLayer`
#     returns `(hidden_states, residual)`, NOT the final residual-stream
#     value directly (an internal fusion of the following RMSNorm); the
#     actual hidden state at that point is `hidden_states + residual` --
#     confirmed by reading vLLM's own EAGLE-3 aux-hidden-state code, which
#     computes it exactly this way, and cross-checked numerically above.
#   - `enforce_eager=True` disables CUDA-graph capture, so the hook fires on
#     every forward call. A replayed CUDA graph re-executes only captured
#     GPU ops, not the surrounding Python (hooks included) -- with graphs
#     enabled, decode-step hidden states after the first would silently go
#     uncaptured.
#   - vLLM's V1 engine always executes the model in a spawned subprocess
#     (confirmed 2026-09-12, even at tensor_parallel_size=1), so state a
#     hook writes is invisible to the caller unless it is (a) retrieved via
#     another `apply_model` call and (b) stored somewhere that persists
#     across separate `apply_model` invocations. `apply_model` hands back
#     the SAME persistent model object every call, so hook-registration
#     state, a recording on/off flag, and the capture buffer itself are all
#     stashed as plain attributes on that object (`_nvs_*`) rather than in
#     module-level globals (which would live in the wrong process) or a
#     closure (which a fresh `apply_model` call cannot reach back into).
#   - Only ever one request in flight per `generate()` call (this class
#     never batches multiple prompts together), so hook-call order equals
#     token-position order -- no request-interleaving correlation problem to
#     solve for a continuous-batching engine.
#   - `SamplingParams(max_tokens=1, temperature=0.0)` forces a single
#     deterministic prefill forward pass over an entire given token sequence
#     (prompt, or prompt+response) rather than a real decode loop. This is
#     used for the actual hidden-state readout in both call sites below:
#       * Phase 0 calibration: prefill the calibration prompt text (chat-
#         formatted) once; every position's hidden state is captured in that
#         one prefill.
#       * Phase 1 (and any future phase) generation: `generate()` samples a
#         real response first (temperature/top_p/top_k/seed exactly per
#         `configs/generation_config.json`); a second, separate prefill call
#         then re-runs prompt+that exact response as one sequence to read
#         off every response token's own hidden state unambiguously. A live
#         decode loop's per-step hidden states are off by one position
#         relative to "the hidden state AT each generated token" (each
#         decode step's forward pass consumes the PREVIOUS token to predict
#         the NEXT one, and never runs a step for the last generated token's
#         own position, since nothing more needs to be predicted from it) --
#         re-deriving them via one clean prefill over the known, now-fixed
#         sequence sidesteps that off-by-one rather than trying to patch it
#         up from decode-step order.
# ---------------------------------------------------------------------------


def _install_hidden_state_hooks(model: Any, layers: tuple[int, ...] = LAYERS) -> int:
    """Runs inside the vLLM worker process via `apply_model`. Idempotent --
    safe to call again on the same model object."""
    if getattr(model, "_nvs_hooks_installed", False):
        return len(model.model.layers)

    model._nvs_recording = False
    model._nvs_buffer = {}

    def _make_hook(layer_idx: int):
        def _hook(_module, _inputs, output):
            if not model._nvs_recording:
                return
            hidden_states, residual = output
            value = hidden_states + residual if residual is not None else hidden_states
            model._nvs_buffer.setdefault(layer_idx, []).append(
                value.detach().to(torch.float32).cpu().clone()
            )

        return _hook

    inner = model.model
    n_layers = len(inner.layers)
    for layer in layers:
        if not (1 <= layer <= n_layers):
            raise ValueError(f"layer {layer} out of range for a {n_layers}-layer model")
        inner.layers[layer - 1].register_forward_hook(_make_hook(layer))

    model._nvs_hooks_installed = True
    return n_layers


def _set_recording(model: Any, value: bool) -> bool:
    model._nvs_recording = value
    if value:
        model._nvs_buffer = {}
    return True


def _fetch_hidden_state_buffer(model: Any) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for layer, chunks in model._nvs_buffer.items():
        out[layer] = torch.cat(chunks, dim=0).numpy()
    model._nvs_buffer = {}
    return out


class _VLLMHiddenStateEngine:
    """Owns one vLLM `LLM` instance for `MODEL_NAME` and extracts exact
    per-token hidden states via the verified mechanism documented above."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        *,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 2048,
    ):
        if not _gpu_available():
            raise RuntimeError(
                "_VLLMHiddenStateEngine requires a CUDA GPU with vllm installed; "
                "neither is available in this environment"
            )

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = LLM(
            model=model_name,
            dtype="bfloat16",
            enforce_eager=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            # Automatic prefix caching (vLLM's default) reuses KV cache --
            # and skips the forward pass entirely -- for any token span that
            # matches a previously-computed prefix. Across many calibration/
            # pilot calls sharing the same chat-template boilerplate, this
            # silently drops hook-captured hidden states for the reused
            # prefix positions (confirmed 2026-09-12: a real calibration run
            # captured 29 rows where 45 were expected). Disabled so every
            # call's hidden states are always freshly computed.
            enable_prefix_caching=False,
        )
        n_layers = self.llm.apply_model(_install_hidden_state_hooks)
        if not n_layers or n_layers[0] < max(LAYERS):
            raise RuntimeError(f"failed to install hidden-state hooks on {model_name}: {n_layers!r}")

        self._stop_token_ids = self._resolve_stop_token_ids()

    def _resolve_stop_token_ids(self) -> list[int]:
        """§2.1 stop condition: EOS tokens only (model-specific; see
        `STOP_TOKEN_STRINGS`)."""
        ids = []
        for tok_str in STOP_TOKEN_STRINGS:
            tok_id = self.tokenizer.convert_tokens_to_ids(tok_str)
            if tok_id is not None and tok_id != self.tokenizer.unk_token_id:
                ids.append(tok_id)
        if not ids:
            raise RuntimeError(
                f"could not resolve stop tokens {STOP_TOKEN_STRINGS} "
                f"from {MODEL_NAME}'s tokenizer"
            )
        return ids

    def _chat_token_ids(self, text: str) -> list[int]:
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
        )
        # This tokenizer's apply_chat_template(tokenize=True) returns a
        # BatchEncoding (dict-like), not a plain list; list(BatchEncoding)
        # yields its string keys ("input_ids", ...), not token ids.
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        return list(ids)

    def _capture_layers(self, token_ids: list[int], layers: tuple[int, ...]) -> dict[int, np.ndarray]:
        self.llm.apply_model(lambda m: _set_recording(m, True))
        try:
            sp = SamplingParams(max_tokens=1, temperature=0.0, seed=0)
            self.llm.generate([TokensPrompt(prompt_token_ids=token_ids)], sp, use_tqdm=False)
        finally:
            self.llm.apply_model(lambda m: _set_recording(m, False))

        buffers = self.llm.apply_model(_fetch_hidden_state_buffer)
        captured = buffers[0]

        out: dict[int, np.ndarray] = {}
        for layer in layers:
            arr = captured.get(layer)
            if arr is None:
                raise RuntimeError(f"no hidden states captured for layer {layer}")
            if arr.shape[0] != len(token_ids):
                raise RuntimeError(
                    f"captured {arr.shape[0]} hidden-state rows for layer {layer}, "
                    f"expected {len(token_ids)} (input sequence length) -- hook "
                    f"capture is out of sync with the input sequence"
                )
            out[layer] = arr.astype(np.float32)
        return out

    def encode_texts(self, prompt_texts: list[str], layers: tuple[int, ...]) -> list[dict[int, np.ndarray]]:
        """Chat-formats each text as a user turn (identical formatting to
        what real generation uses right before sampling a response) and
        returns per-layer hidden states for ALL of that formatted prompt's
        tokens -- used as-is by Phase 0 calibration (§2.3: hidden states
        "of the 200 [calibration] prompts", not of generated responses)."""
        results = []
        total = len(prompt_texts)
        for i, text in enumerate(prompt_texts, start=1):
            results.append(self._capture_layers(self._chat_token_ids(text), layers))
            if i % 10 == 0 or i == total:
                print(f"    [calibration] extracted {i}/{total}")
        return results

    def generate_and_extract(
        self,
        prompt_text: str,
        seed: int,
        sampling_config: dict,
        layers: tuple[int, ...] = LAYERS,
    ) -> tuple[str, int, dict[int, np.ndarray]]:
        """§2.1 real generation, then hidden-state readout for exactly the
        generated response tokens (see module-level design notes above for
        why this needs a second prefill call rather than reusing decode-step
        hidden states)."""
        prompt_ids = self._chat_token_ids(prompt_text)

        sp = SamplingParams(
            temperature=sampling_config["temperature"],
            top_p=sampling_config["top_p"],
            top_k=sampling_config["top_k"],
            max_tokens=sampling_config["max_new_tokens"],
            seed=seed,
            stop_token_ids=self._stop_token_ids,
        )
        outputs = self.llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)], sp, use_tqdm=False)
        completion = outputs[0].outputs[0]
        response_token_ids = list(completion.token_ids)
        response_text = completion.text
        token_count = len(response_token_ids)

        if token_count == 0:
            # Degenerate: the model emitted a stop token immediately. Class E
            # (token_count < MIN_SEQUENCE_LENGTH) handles this downstream;
            # just report empty hidden states rather than running a
            # prefill over an empty response span.
            empty = {layer: np.zeros((0, SOURCE_DIMENSION), dtype=np.float32) for layer in layers}
            return response_text, token_count, empty

        full_ids = prompt_ids + response_token_ids
        full_hidden = self._capture_layers(full_ids, layers)
        response_hidden = {layer: full_hidden[layer][len(prompt_ids):] for layer in layers}
        return response_text, token_count, response_hidden


_ENGINE: Optional[_VLLMHiddenStateEngine] = None


def _get_engine() -> _VLLMHiddenStateEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = _VLLMHiddenStateEngine(MODEL_NAME)
    return _ENGINE


def _load_generation_config(path: Path = CONFIGS / "generation_config.json") -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["sampling"]


def extract_hidden_states(prompt_texts: list[str], *, layers: tuple[int, ...] = LAYERS) -> list[dict[int, np.ndarray]]:
    """Runs `MODEL_NAME` via vLLM (plan §2.1) and returns, per
    prompt, a {layer: (T, 4096)} dict of hidden states. See
    `_VLLMHiddenStateEngine` above for the verified extraction mechanism."""
    engine = _get_engine()
    return engine.encode_texts(prompt_texts, layers=layers)


def execute_phase0_calibration(
    calibration_prompts_path: Path = CONFIGS / "calibration_prompts_v1.json",
) -> dict[str, str]:
    """§2.3: extracts Layer 16 hidden states over the calibration corpus and
    fits (mu, W_128). Writes to the REAL frozen paths -- only ever called
    with real extracted hidden states; never given a synthetic fallback."""
    print("\n=== Phase 0: PCA calibration ===")
    if not calibration_prompts_path.exists():
        raise FileNotFoundError(calibration_prompts_path)

    calibration_prompts = json.loads(calibration_prompts_path.read_text(encoding="utf-8"))
    print(f"[+] loaded {len(calibration_prompts)} calibration prompts")

    # Real extraction only -- raises loudly (RuntimeError/ValueError) rather
    # than falling back to synthetic data: writing fabricated numbers to
    # configs/W_pca128.npy and reporting them as real hashes would corrupt
    # the pre-registered artifact.
    per_prompt_states = extract_hidden_states([p["text"] for p in calibration_prompts], layers=(16,))
    all_hidden_states = np.vstack([states[16] for states in per_prompt_states])
    assert all_hidden_states.shape[1] == SOURCE_DIMENSION

    w_path, mu_path, _diag_path = run_calibration(all_hidden_states)

    hashes = {
        "W_pca128_sha256": compute_file_sha256(w_path),
        "mu_pca_sha256": compute_file_sha256(mu_path),
    }
    print(f"[+] W_pca128.npy sha256: {hashes['W_pca128_sha256']}")
    print(f"[+] mu_pca.npy   sha256: {hashes['mu_pca_sha256']}")
    return hashes


class RealGenerator:
    """Wraps the vLLM engine in the `Generator` shape `collect_trajectories.
    collect()` expects: real stochastic generation (§2.1 sampling config)
    plus real per-layer hidden-state extraction (see module-level design
    notes above)."""

    def __init__(self, sampling_config: dict, *, total: int | None = None):
        self._sampling_config = sampling_config
        self._total = total
        self._count = 0

    def generate(self, prompt_id: str, sample_index: int, prompt_text: str, seed: int) -> GeneratedSample:
        engine = _get_engine()
        text, token_count, hidden_states = engine.generate_and_extract(
            prompt_text, seed, self._sampling_config, layers=LAYERS
        )
        self._count += 1
        if self._count % 10 == 0 or self._count == self._total:
            suffix = f"/{self._total}" if self._total else ""
            print(f"    [pilot] generated {self._count}{suffix} ({prompt_id}#{sample_index}, T={token_count})")
        return GeneratedSample(
            prompt_id=prompt_id,
            sample_index=sample_index,
            text=text,
            token_count=token_count,
            hidden_states=hidden_states,
        )


MIXED_RATE_GATE = 0.30  # §5.4 / §8 Rule 3


def compute_mixed_rate_stats(records: list[dict]) -> dict[str, Any]:
    """§5.4: mixed rate = fraction of prompts whose (up to 8) samples contain
    both a positive (`type_a_positive`) and a negative (`type_c_abstain` or
    `type_d_hedge`) outcome. `class_e_indeterminate` samples are excluded
    from the positive/negative tally (indeterminate, not a negative) but the
    prompt itself still counts in the denominator (every prompt attempted).

    Denominator is the number of distinct prompts actually present in
    `records`, not a caller-supplied total -- correct for both the intended
    100-prompt Phase 1 pilot and any smaller/adjustable diagnostic run.
    """
    by_prompt: dict[str, dict[str, int]] = {}
    for r in records:
        counts = by_prompt.setdefault(r["prompt_id"], {"positive": 0, "negative": 0, "indeterminate": 0})
        if r["label"] == "type_a_positive":
            counts["positive"] += 1
        elif r["label"] in ("type_c_abstain", "type_d_hedge"):
            counts["negative"] += 1
        else:  # class_e_indeterminate
            counts["indeterminate"] += 1

    total_prompts = len(by_prompt)
    mixed_prompt_ids = [
        pid for pid, c in by_prompt.items() if c["positive"] >= 1 and c["negative"] >= 1
    ]
    mixed_rate = len(mixed_prompt_ids) / total_prompts if total_prompts else 0.0

    mixed_pos = sum(by_prompt[pid]["positive"] for pid in mixed_prompt_ids)
    mixed_neg = sum(by_prompt[pid]["negative"] for pid in mixed_prompt_ids)
    pos_rate_in_mixed = mixed_pos / (mixed_pos + mixed_neg) if (mixed_pos + mixed_neg) else 0.0

    return {
        "total_prompts": total_prompts,
        "mixed_prompt_count": len(mixed_prompt_ids),
        "mixed_rate": mixed_rate,
        "pos_rate_in_mixed": pos_rate_in_mixed,
        "gate_threshold": MIXED_RATE_GATE,
        "rule_3_pass": mixed_rate >= MIXED_RATE_GATE,
    }


def execute_phase1_pilot(
    prompts_path: Path = CONFIGS / "prompts_v1.json",
    num_pilot_prompts: int = 100,
    samples_per_prompt: int = 8,
    output_path: Path = RESULTS / "phase1_pilot_trajectories.json",
    sampling_overrides: Optional[dict] = None,
) -> Path:
    """§5.4 Phase 1: collects `num_pilot_prompts` x `samples_per_prompt` real
    generations, labels them (v1.4 §5.2's rule is fully resolved -- no
    pending/deferred labels remain), and computes the mixed-rate / Rule 3
    gate verdict (§8) directly against the saved records.

    `sampling_overrides` merges into the frozen sampling config from
    `configs/generation_config.json` (e.g. `{"temperature": 1.0}`) -- for
    ad-hoc diagnostic runs only; a real frozen Phase 1 run should leave this
    unset and use the frozen config as-is.
    """
    print(f"\n=== Phase 1: pilot collection ({num_pilot_prompts} prompts x {samples_per_prompt} samples) ===")
    if not prompts_path.exists():
        raise FileNotFoundError(prompts_path)

    all_prompts_raw = json.loads(prompts_path.read_text(encoding="utf-8"))
    pilot_prompts = [
        PromptRecord(prompt_id=p["prompt_id"], prompt_type=PromptType.TYPE_A, template=p["template"], text=p["text"])
        for p in all_prompts_raw[:num_pilot_prompts]
    ]
    print(f"[+] using {len(pilot_prompts)} of {len(all_prompts_raw)} Type A prompts")

    eval_config = EvalConfig(
        refusal_patterns_path=CONFIGS / "refusal_patterns_en.json",
        typeB_reference_path=CONFIGS / "typeB_reference.json",
    )
    collector = TrajectoryCollector(
        eval_config=eval_config,
        # Re-read as module attributes (not bound as this function's own
        # defaults) so tests can monkeypatch collect_trajectories.
        # DEFAULT_W_PATH / DEFAULT_MU_PATH to a guaranteed-missing path
        # without depending on whether the real frozen artifact happens to
        # exist yet.
        w_path=collect_trajectories.DEFAULT_W_PATH,
        mu_path=collect_trajectories.DEFAULT_MU_PATH,
    )  # real W/mu only -- raises if not frozen yet

    sampling_config = _load_generation_config()
    if sampling_overrides:
        sampling_config = {**sampling_config, **sampling_overrides}
        print(f"[!] sampling overrides in effect (diagnostic run, NOT the frozen config): {sampling_overrides}")
    generator = RealGenerator(sampling_config, total=len(pilot_prompts) * samples_per_prompt)

    records = collect(pilot_prompts, generator, collector, samples_per_prompt=samples_per_prompt)

    label_counts: dict[str, int] = {}
    for r in records:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

    mixed_stats = compute_mixed_rate_stats(records)

    serializable = [
        {
            "prompt_id": r["prompt_id"],
            "sample_index": r["sample_index"],
            "seed": r["seed"],
            "text": r["text"],
            "token_count": r["token_count"],
            "label": r["label"],
            "label_details": r["label_details"],
        }
        for r in records
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "sampling_config": sampling_config,
                "label_counts": label_counts,
                "mixed_rate_stats": mixed_stats,
                "records": serializable,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[+] collected {len(records)} generations, label counts: {label_counts}")
    print(
        f"[+] mixed_rate={mixed_stats['mixed_rate']:.3f} "
        f"({mixed_stats['mixed_prompt_count']}/{mixed_stats['total_prompts']} prompts), "
        f"pos_rate_in_mixed={mixed_stats['pos_rate_in_mixed']:.3f}, "
        f"Rule 3 (>= {MIXED_RATE_GATE:.0%}): "
        f"{'PASS' if mixed_stats['rule_3_pass'] else 'FAIL'}"
    )
    print(f"[+] saved: {output_path}")
    return output_path


def execute_phase2_collection(
    prompts_path: Path = CONFIGS / "prompts_v1_phase2.json",
    samples_per_prompt: int = 8,
    output_path: Path = RESULTS / "phase2_trajectories.json",
    ablation_ids_path: Path = CONFIGS / "ablation_prompt_ids.json",
    ablation_raw_dir: Path = RESULTS / "ablation_raw_hidden_states",
    projected_dir: Path = RESULTS / "phase2_projected",
) -> Path:
    """§5.4 Phase 2: collects real generations over the FULL Phase 2 prompt
    pool (the disjoint second batch, §5.4/U6 -- Phase 1's prompts are never
    reused here). For prompts in `ablation_ids_path` (§2.3 F1, frozen BEFORE
    this call per §2.3's "generation begins" cutoff), also persists raw
    (unprojected) Layer-16 hidden states for §6.3 ABL-1.

    Each sample's per-layer 128D projected trajectories (needed by Phase 3
    for the Layer 8/16/24 AUROC comparison, BL-NORM, and ABL-2's PC4-67) are
    saved as one compressed .npz per sample under `projected_dir` rather than
    inlined into the JSON -- ~3944 samples x 3 layers x up to 200x128 floats
    would make the JSON file multi-gigabyte and slow to parse. The main JSON
    stays lightweight (text/label/kappa summary + a path reference).

    Does not compute AUROC/Gate-3 here -- that is Phase 3's job, run
    separately against this saved file.
    """
    print(f"\n=== Phase 2: main collection ({samples_per_prompt} samples/prompt) ===")
    if not prompts_path.exists():
        raise FileNotFoundError(prompts_path)
    if not ablation_ids_path.exists():
        raise FileNotFoundError(ablation_ids_path)

    all_prompts_raw = json.loads(prompts_path.read_text(encoding="utf-8"))
    phase2_prompts = [
        PromptRecord(prompt_id=p["prompt_id"], prompt_type=PromptType.TYPE_A, template=p["template"], text=p["text"])
        for p in all_prompts_raw
    ]
    ablation_ids = set(json.loads(ablation_ids_path.read_text(encoding="utf-8"))["prompt_ids"])
    print(f"[+] using {len(phase2_prompts)} Phase 2 prompts, {len(ablation_ids)} flagged for raw-hidden-state ablation")

    eval_config = EvalConfig(
        refusal_patterns_path=CONFIGS / "refusal_patterns_en.json",
        typeB_reference_path=CONFIGS / "typeB_reference.json",
    )
    collector = TrajectoryCollector(
        eval_config=eval_config,
        w_path=collect_trajectories.DEFAULT_W_PATH,
        mu_path=collect_trajectories.DEFAULT_MU_PATH,
        ablation_prompt_ids=ablation_ids,
        ablation_raw_dir=ablation_raw_dir,
    )

    sampling_config = _load_generation_config()
    generator = RealGenerator(sampling_config, total=len(phase2_prompts) * samples_per_prompt)

    records = collect(phase2_prompts, generator, collector, samples_per_prompt=samples_per_prompt)

    label_counts: dict[str, int] = {}
    for r in records:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

    projected_dir.mkdir(parents=True, exist_ok=True)
    serializable = []
    for r in records:
        proj_fname = f"{r['prompt_id']}__{r['sample_index']}.npz"
        np.savez_compressed(
            projected_dir / proj_fname,
            **{k: v.astype(np.float32) for k, v in r["projected_layers"].items()},
        )
        serializable.append(
            {
                "prompt_id": r["prompt_id"],
                "sample_index": r["sample_index"],
                "seed": r["seed"],
                "text": r["text"],
                "token_count": r["token_count"],
                "label": r["label"],
                "label_details": r["label_details"],
                "projected_path": str((projected_dir / proj_fname).resolve().relative_to(REPO_ROOT)),
                "kappa": None
                if r["kappa"] is None
                else {"kappa": r["kappa"].kappa.tolist(), "valid_mask": r["kappa"].valid_mask.tolist()},
                "raw_ablation_path": r["raw_ablation_path"],
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"label_counts": label_counts, "records": serializable}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"[+] collected {len(records)} generations, label counts: {label_counts}")
    print(f"[+] saved: {output_path}")
    return output_path


if __name__ == "__main__":
    print("=== EXP-2026-NVS-001: GPU pipeline entrypoint ===")
    log_gpu_environment()
    execute_phase0_calibration()
    execute_phase1_pilot()
