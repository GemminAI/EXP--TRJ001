"""GPU execution entrypoint for EXP-2026-NVS-001 (plan v1.3).

Meant to run in the real GPU/vLLM environment this repository's local
development so far has not had access to. Two things this script
deliberately does NOT do, both corrections to an earlier draft:

1. It never silently substitutes synthetic data for real calibration/pilot
   output. If real hidden-state extraction isn't wired up yet,
   `extract_hidden_states` raises `NotImplementedError` rather than writing
   fabricated numbers to `configs/W_pca128.npy` / `results/phase1_pilot_*`
   and reporting real-looking SHA-256 hashes for them. For a pre-registered
   experiment, a loud failure here is much cheaper than a quiet fabrication.

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

`extract_hidden_states` is the one real integration point this environment
cannot fill in (no GPU/vLLM here, and vLLM's actual per-token intermediate
hidden-state API was never verified in this session -- see
`collect_trajectories.VLLMGenerator`). Implement it in the GPU environment
against vLLM's actual, verified API before running this for real.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import vllm
except ImportError:
    vllm = None

from collect_trajectories import LAYERS, GeneratedSample, TrajectoryCollector, collect
from eval_hallucination import EvalConfig, PromptRecord, PromptType
from train_w64 import SOURCE_DIMENSION, run_calibration

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
RESULTS = REPO_ROOT / "results"


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


def extract_hidden_states(prompt_texts: list[str], *, layers: tuple[int, ...] = LAYERS) -> list[dict[int, np.ndarray]]:
    """Runs Meta-Llama-3.1-8B-Instruct via vLLM (plan §2.1) and returns, per
    prompt, a {layer: (T, 4096)} dict of hidden states.

    NOT IMPLEMENTED HERE. This is the one piece this development environment
    genuinely cannot write: no GPU, no vLLM installed, and vLLM's actual
    per-token intermediate hidden-state retrieval API (as opposed to HF
    Transformers' `output_hidden_states=True`) was never verified anywhere in
    this session. Implement this against vLLM's real, checked API in the GPU
    environment -- do not guess an API surface here and ship it unverified.
    """
    raise NotImplementedError(
        "extract_hidden_states: implement against vLLM's real hidden-state "
        "API in the GPU execution environment; not available here."
    )


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

    # Real extraction only -- raises NotImplementedError until wired up on
    # the actual GPU host (see extract_hidden_states docstring). No
    # synthetic-data fallback: writing fabricated numbers to
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


def execute_phase1_pilot(
    prompts_path: Path = CONFIGS / "prompts_v1.json",
    num_pilot_prompts: int = 100,
    samples_per_prompt: int = 8,
    output_path: Path = RESULTS / "phase1_pilot_trajectories.json",
) -> Path:
    """§5.4 Phase 1: collects 100 x 8 real generations and saves everything.

    Does NOT compute mixed_rate, pos_rate, or the Rule-3 gate verdict here --
    see module docstring for why that is a separate, later step (it needs
    the §5.2 Type A rule decided first, from real samples this collection
    itself produces).
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
    collector = TrajectoryCollector(eval_config=eval_config)  # real W/mu only -- raises if not frozen yet

    class RealGenerator:
        """Wraps `extract_hidden_states` in the `Generator` shape
        `collect_trajectories.collect()` expects. Text generation itself
        (not just hidden-state extraction) also needs the real vLLM call --
        left alongside `extract_hidden_states` as not implemented here."""

        def generate(self, prompt_id: str, sample_index: int, prompt_text: str, seed: int) -> GeneratedSample:
            raise NotImplementedError(
                "RealGenerator.generate: implement real vLLM generation + "
                "hidden-state extraction in the GPU environment"
            )

    records = collect(pilot_prompts, RealGenerator(), collector, samples_per_prompt=samples_per_prompt)

    label_counts: dict[str, int] = {}
    for r in records:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

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
        json.dumps({"label_counts": label_counts, "records": serializable}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"[+] collected {len(records)} generations, label counts: {label_counts}")
    print(
        "[!] mixed_rate / M NOT computed here -- review the 'pending_type_a' "
        "records' text above, decide the §5.2 Type A positive-determination "
        "rule, fill it into eval_hallucination.classify_type_a, then re-run "
        "labeling over this saved file (no re-generation needed) before "
        "computing the Rule 3 gate."
    )
    print(f"[+] saved: {output_path}")
    return output_path


if __name__ == "__main__":
    print("=== EXP-2026-NVS-001: GPU pipeline entrypoint ===")
    log_gpu_environment()
    execute_phase0_calibration()
    execute_phase1_pilot()
