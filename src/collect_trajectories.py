"""Trajectory collection pipeline for EXP-2026-NVS-001 (plan v1.3 §5, "採取スクリプト").

Orchestrates: prompt -> model generation (`Generator` interface) -> per-layer
hidden-state extraction -> projection via the frozen Layer-16 PCA basis
(mu, W_128) applied identically to Layers 8/16/24 as one shared observation
axis (confirmed 2026-09-12: the plan's single frozen basis is deliberately
used as a common coordinate system across layers, to compare layer dynamics
in the same space -- not a separate per-layer PCA fit) -> kappa(t) on Layer
16 PC1-64 -> hallucination-label evaluation.

Real model execution (vLLM, plan §2.1) needs a GPU this environment does not
have. `Generator` is an interface so the rest of the pipeline is testable now
via `MockGenerator`; a real `VLLMGenerator` can be dropped in later without
touching anything else here.

Label-evaluation note: for a Type A prompt whose response is not Type C/D,
`eval_hallucination.evaluate()` deliberately raises `TypeADecisionPending`
(§5.2's positive-determination method is still undecided, per instruction
2026-09-12). This collector does NOT treat that as a collection failure --
it catches it, stores the raw diagnostic text/signals, and marks the record
`label: "pending_type_a"` so collection is never blocked on a labeling
decision that has not been made yet; re-labeling later needs no
re-generation.

PCA-weights note: unlike an earlier draft, this module does NOT silently
fall back to an identity/random projection when `configs/W_pca128.npy` /
`configs/mu_pca.npy` are missing -- for a pre-registered experiment, running
collection without the real frozen projection must fail loudly, not
silently produce meaningless data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np

from compute_kappa import MIN_SEQUENCE_LENGTH, KappaResult, compute_kappa
from eval_hallucination import (
    EvalConfig,
    GenerationRecord,
    PromptRecord,
    PromptType,
    TypeADecisionPending,
    evaluate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_W_PATH = REPO_ROOT / "configs" / "W_pca128.npy"
DEFAULT_MU_PATH = REPO_ROOT / "configs" / "mu_pca.npy"
LAYERS = (8, 16, 24)  # §2.2: saved at collection time; 16 is primary
PRIMARY_LAYER = 16
PENDING_TYPE_A_LABEL = "pending_type_a"


def deterministic_seed(prompt_id: str, sample_index: int) -> int:
    """§2.1: `seed = hash(prompt_id, sample_index) % 2**32`. Python's builtin
    `hash()` is salted per-process (PYTHONHASHSEED) and is not reproducible
    across runs/machines, so a stable hash (MD5) is used instead -- this is
    what a frozen, logged seed requires in practice."""
    digest = hashlib.md5(f"{prompt_id}_{sample_index}".encode("utf-8")).hexdigest()
    return int(digest, 16) % (2**32)


@dataclass(frozen=True)
class GeneratedSample:
    """One model generation, before projection/labeling."""

    prompt_id: str
    sample_index: int
    text: str
    token_count: int
    hidden_states: dict[int, np.ndarray]  # layer -> (T, 4096) float32
    generation_error: bool = False


class Generator(Protocol):
    """Interface the collection loop drives -- real (vLLM) or mock."""

    def generate(
        self, prompt_id: str, sample_index: int, prompt_text: str, seed: int
    ) -> GeneratedSample: ...


class MockGenerator:
    """Synthetic generator for pipeline testing without a GPU/vLLM.

    Produces plausible-shaped (T, 4096) hidden states and placeholder text --
    NOT real model output. Never treat its records as experiment data.
    """

    def __init__(self, *, min_tokens: int = 25, max_tokens: int = 60, hidden_dim: int = 4096):
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens
        self._hidden_dim = hidden_dim

    def generate(
        self, prompt_id: str, sample_index: int, prompt_text: str, seed: int
    ) -> GeneratedSample:
        rng = np.random.default_rng(seed)
        token_count = int(rng.integers(self._min_tokens, self._max_tokens))
        text = f"[mock response to {prompt_id}#{sample_index}] {prompt_text[:40]}"
        hidden_states = {
            layer: rng.standard_normal((token_count, self._hidden_dim)).astype(np.float32)
            for layer in LAYERS
        }
        return GeneratedSample(
            prompt_id=prompt_id,
            sample_index=sample_index,
            text=text,
            token_count=token_count,
            hidden_states=hidden_states,
        )


class VLLMGenerator:
    """Real generation via vLLM (plan §2.1).

    NOT implemented here -- this environment has no GPU/vLLM. Whether vLLM
    exposes a standard per-token intermediate hidden-state retrieval API (as
    opposed to HF Transformers' `output_hidden_states=True`) needs to be
    verified in the real execution environment; this class is only the
    integration point, not a working implementation.
    """

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise NotImplementedError(
            "VLLMGenerator is not implemented in this environment (no GPU/vLLM "
            "available here). Implement generate() in the real execution "
            "environment against the actual vLLM hidden-state extraction API."
        )

    def generate(
        self, prompt_id: str, sample_index: int, prompt_text: str, seed: int
    ) -> GeneratedSample:
        raise NotImplementedError


class TrajectoryCollector:
    def __init__(
        self,
        *,
        eval_config: EvalConfig,
        w_path: Path = DEFAULT_W_PATH,
        mu_path: Path = DEFAULT_MU_PATH,
    ):
        if not w_path.exists() or not mu_path.exists():
            raise FileNotFoundError(
                f"frozen projection not found ({w_path}, {mu_path}) -- run "
                f"src/train_w64.py against real calibration hidden states "
                f"first; this collector refuses to fall back to a placeholder "
                f"projection for a pre-registered experiment"
            )
        self.W = np.load(w_path)  # (4096, 128)
        self.mu = np.load(mu_path)  # (4096,)
        self._eval_config = eval_config
        self._reference = (
            json.loads(eval_config.typeB_reference_path.read_text(encoding="utf-8"))
            if eval_config.typeB_reference_path.exists()
            else None
        )

    def project(self, hidden_states_4096: np.ndarray) -> np.ndarray:
        """`z = (h - mu) @ W`, applied identically across Layers 8/16/24 --
        the plan's single frozen Layer-16-fit basis used as one shared
        observation axis (confirmed 2026-09-12), not a separate per-layer fit."""
        return (hidden_states_4096 - self.mu) @ self.W

    def process_sample(self, sample: GeneratedSample, prompt: PromptRecord) -> dict:
        gen = GenerationRecord(
            prompt_id=sample.prompt_id,
            sample_index=sample.sample_index,
            seed=deterministic_seed(sample.prompt_id, sample.sample_index),
            text=sample.text,
            token_count=sample.token_count,
            generation_error=sample.generation_error,
        )

        try:
            eval_result = evaluate(prompt, gen, self._eval_config, reference=self._reference)
            label = eval_result.label.value
            label_details: dict[str, Any] = asdict(eval_result)
        except TypeADecisionPending as exc:
            # Deliberately not a collection failure -- see module docstring.
            label = PENDING_TYPE_A_LABEL
            label_details = {"pending_reason": str(exc)}

        projected_layers: dict[str, np.ndarray] = {}
        kappa_result: Optional[KappaResult] = None
        for layer, h_4096 in sample.hidden_states.items():
            proj_128 = self.project(h_4096)
            projected_layers[f"layer_{layer}_128d"] = proj_128
            if layer == PRIMARY_LAYER and sample.token_count >= MIN_SEQUENCE_LENGTH:
                kappa_result = compute_kappa(proj_128[:, :64])

        return {
            "prompt_id": sample.prompt_id,
            "sample_index": sample.sample_index,
            "seed": gen.seed,
            "text": sample.text,
            "token_count": sample.token_count,
            "label": label,
            "label_details": label_details,
            "projected_layers": projected_layers,
            "kappa": kappa_result,
        }


def collect(
    prompts: list[PromptRecord],
    generator: Generator,
    collector: TrajectoryCollector,
    *,
    samples_per_prompt: int = 8,
) -> list[dict]:
    records = []
    for prompt in prompts:
        for sample_index in range(samples_per_prompt):
            seed = deterministic_seed(prompt.prompt_id, sample_index)
            sample = generator.generate(prompt.prompt_id, sample_index, prompt.text, seed)
            records.append(collector.process_sample(sample, prompt))
    return records


def _demo() -> None:
    """Mock end-to-end smoke run -- NOT real experiment data. Uses the
    synthetic (W, mu) from `train_w64.py`'s self-test path, not the real
    frozen projection (which does not exist yet -- no real calibration run
    has happened)."""
    import json

    selftest_dir = REPO_ROOT / "data" / "_train_w64_selftest"
    if not (selftest_dir / "W_pca128.npy").exists():
        print("[!] run `python src/train_w64.py` first to produce a self-test (W, mu)")
        return

    prompts_raw = json.loads((REPO_ROOT / "configs" / "prompts_v1.json").read_text(encoding="utf-8"))[:2]
    prompts = [
        PromptRecord(
            prompt_id=p["prompt_id"],
            prompt_type=PromptType.TYPE_A,
            template=p["template"],
            text=p["text"],
        )
        for p in prompts_raw
    ]

    eval_config = EvalConfig(
        refusal_patterns_path=REPO_ROOT / "configs" / "refusal_patterns_en.json",
        typeB_reference_path=REPO_ROOT / "configs" / "typeB_reference.json",
    )
    collector = TrajectoryCollector(
        eval_config=eval_config,
        w_path=selftest_dir / "W_pca128.npy",
        mu_path=selftest_dir / "mu_pca.npy",
    )
    records = collect(prompts, MockGenerator(), collector, samples_per_prompt=2)
    print(f"[+] collected {len(records)} mock records")
    for r in records:
        kappa_shape = r["kappa"].kappa.shape if r["kappa"] is not None else None
        print(f"    {r['prompt_id']}#{r['sample_index']}: label={r['label']} kappa_shape={kappa_shape}")


if __name__ == "__main__":
    _demo()
