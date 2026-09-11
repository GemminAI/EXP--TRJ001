"""Generates the remaining Phase 0 config files (plan v1.3 §2.1, §7.4).

Only two files are written here:
  - configs/generation_config.json  (FREEZE_MANIFEST row 10)
  - configs/mask_protocol.json      (§2.5 checklist: "マスク手続き定義")

`configs/data_splits.json` and `configs/ablation_prompt_ids.json` are
DELIBERATELY NOT generated here, even though they were requested alongside
the other two. Both are explicitly Phase-1-completion artifacts, not Phase 0:

  - `freeze/FREEZE_MANIFEST.md`'s own structure puts them in a separate
    "Phase 1 後に確定する項目" table, fixed by a *different* tag
    (`phase1-complete`), not in the main Phase 0 artifact table (rows 1-12).
  - Plan §5.4: the actual Phase 2 prompt count M is computed from Phase 1's
    *measured* mixed_rate and pos_rate_in_mixed -- unknown until Phase 1
    (real model generation) has actually run.
  - Plan §2.3: the 200 ablation prompt IDs are drawn from the Phase 2 Test
    split "immediately after the split is generated, before Phase 2
    generation starts" -- i.e. after Phase 1, not before it.
  - Plan §2.3, verbatim: "生成開始後の選定は禁止" (no post-hoc selection).

Generating either file now, before Phase 1 has run, would not just be early
-- it would fabricate a split/ablation-ID population the plan's own
procedure hasn't determined yet.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATION_CONFIG_PATH = REPO_ROOT / "configs" / "generation_config.json"
MASK_PROTOCOL_PATH = REPO_ROOT / "configs" / "mask_protocol.json"


def build_generation_config() -> dict:
    """§2.1. Fields left `null` are Appendix B item 1 ("vLLM の具体バージョンと
    モデルのコミットハッシュ") -- explicitly unconfirmed in the plan itself,
    to be filled in at the real freeze commit, not guessed here."""
    return {
        "_notice": (
            "model_commit_hash and inference_engine_version are Appendix B "
            "item 1 -- explicitly unconfirmed pre-freeze items in plan v1.3. "
            "Fill them in from the real execution environment before "
            "v1.3-frozen; do not invent values here."
        ),
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "model_revision": "main",
        "model_commit_hash": None,
        "quantization": "none (bfloat16)",
        "inference_engine": "vLLM",
        "inference_engine_version": None,
        "inference_engine_min_version": "0.6.0",
        "generation_language": "en",
        "sampling": {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "max_new_tokens": 200,
        },
        "seed_formula": "seed = hash(prompt_id, sample_index) % 2**32",
        "stop_tokens": ["<|eot_id|>", "<|eom_id|>"],
        "layers_saved": [8, 16, 24],
        "primary_layer": 16,
    }


def build_mask_protocol() -> dict:
    """§7.4. Defines the masking PROCEDURE and its frozen parameters --
    not precomputed per-sample masks (trajectory length T varies per sample
    and isn't known until generation happens; masks are derived
    deterministically from `mask_seed` plus each sample's own
    (prompt_id, sample_index) at Phase 4 analysis time, per this spec)."""
    return {
        "mask_seed": 20260912,
        "interpolation": {
            "levels": [0.20, 0.35, 0.50],
            "description": (
                "For each trajectory, randomly mask this fraction of its "
                "INTERNAL tokens (the same positions kappa is defined on: "
                "excluding the two endpoints t=1, t=T). Mask count = "
                "floor(level * (T - 2)). All six methods (M0-M5, plan §7.2) "
                "are evaluated against the SAME mask pattern for a given "
                "(prompt_id, sample_index, level) -- masks are derived "
                "deterministically from mask_seed plus that sample's own "
                "identity, never redrawn per method."
            ),
        },
        "extrapolation": {
            "fraction": 0.20,
            "description": (
                "Remove the trailing 20% of each trajectory; predict those "
                "positions from the leading 80%."
            ),
        },
        "metric": "MSE across all 64 dimensions (plan §7.4)",
        "hyperparameter_selection_split": "validation",
        "reporting_split": "test",
    }


def run() -> None:
    GENERATION_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    GENERATION_CONFIG_PATH.write_text(
        json.dumps(build_generation_config(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    MASK_PROTOCOL_PATH.write_text(
        json.dumps(build_mask_protocol(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {GENERATION_CONFIG_PATH}")
    print(f"wrote {MASK_PROTOCOL_PATH}")
    print(
        "NOT generated (Phase-1-completion artifacts, see module docstring): "
        "configs/data_splits.json, configs/ablation_prompt_ids.json"
    )


if __name__ == "__main__":
    run()
