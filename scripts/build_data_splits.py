"""Build configs/data_splits.json for EXP-TRJ002a.

This split is NOT a frozen pre-registered artifact of EXP--TRJ001 -- no such
split exists in this repository (FREEZE_MANIFEST.md is entirely unfilled,
Phase 1/2/3 never ran). It is a newly created, seeded stratified split over
EXP--TRJ001's Type A prompt pool (configs/prompts_v1.json, 1000 prompts,
5 templates x 200), built specifically for the TRJ002a projection benchmark
per the counts given in its instruction prompt: 296 train / 99 val / 98 test
(493 total), stratified by template so every split keeps roughly even
template representation.

TRJ002a's label scheme (type_a_positive vs type_c_abstain/type_d_hedge) is
only defined for Type A responses (eval_hallucination.Label), so only Type A
prompts are used here -- Type B prompts (configs/prompts_typeB_v1.json) are
out of scope for this split.
"""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = REPO_ROOT / "configs" / "prompts_v1.json"
OUT_PATH = REPO_ROOT / "configs" / "data_splits.json"

SEED = 20260914
N_TRAIN, N_VAL, N_TEST = 296, 99, 98
N_TOTAL = N_TRAIN + N_VAL + N_TEST  # 493


def main() -> None:
    prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    ids = [p["prompt_id"] for p in prompts]
    templates = [p["template"] for p in prompts]

    # Step 1: draw a 493-prompt pool out of the 1000, stratified by template.
    pool_ids, _, pool_templates, _ = train_test_split(
        ids,
        templates,
        train_size=N_TOTAL,
        stratify=templates,
        random_state=SEED,
    )

    # Step 2: split the pool into train / val / test, stratified by template.
    train_ids, rest_ids, train_t, rest_t = train_test_split(
        pool_ids,
        pool_templates,
        train_size=N_TRAIN,
        stratify=pool_templates,
        random_state=SEED,
    )
    val_ids, test_ids, _, _ = train_test_split(
        rest_ids,
        rest_t,
        train_size=N_VAL,
        stratify=rest_t,
        random_state=SEED,
    )

    assert len(train_ids) == N_TRAIN
    assert len(val_ids) == N_VAL
    assert len(test_ids) == N_TEST
    assert not (set(train_ids) & set(val_ids) & set(test_ids))

    out = {
        "_notice": (
            "Newly created for EXP-TRJ002a -- NOT a frozen EXP--TRJ001 "
            "pre-registered split (no such split exists; FREEZE_MANIFEST.md "
            "is unfilled and Phase 1-3 never ran on this repo). Seeded "
            "stratified split over configs/prompts_v1.json (Type A only)."
        ),
        "seed": SEED,
        "source_pool": "configs/prompts_v1.json",
        "n_samples_per_prompt": 8,
        "splits": {
            "train": sorted(train_ids),
            "val": sorted(val_ids),
            "test": sorted(test_ids),
        },
        "counts": {"train": N_TRAIN, "val": N_VAL, "test": N_TEST},
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT_PATH} : train={N_TRAIN} val={N_VAL} test={N_TEST}")


if __name__ == "__main__":
    main()
