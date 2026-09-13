"""Creates and freezes the Phase 2 data split (plan v1.4 §2.4).

Prompt-unit split: every sample from the same prompt_id lands in the same
split. Train 60% / Val 20% / Test 20%, `split_seed = 20260912` (frozen).

Scope note (v1.4 U6/U9): this splits ONLY the Phase 2 prompt pool
(`configs/prompts_v1_phase2.json`, the disjoint second batch generated after
Phase 1 fully consumed the original 1,000-prompt pool -- see
`generate_type_a_prompts_phase2.py`). Phase 1's prompts/samples are excluded
entirely, per U6's data-isolation requirement.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = REPO_ROOT / "configs" / "prompts_v1_phase2.json"
OUT_PATH = REPO_ROOT / "configs" / "data_splits.json"

SPLIT_SEED = 20260912  # §2.4, frozen
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
# remainder (0.2) is test


def create_splits(prompt_ids: list[str], seed: int = SPLIT_SEED) -> dict[str, list[str]]:
    ids = list(prompt_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = round(n * TRAIN_FRAC)
    n_val = round(n * VAL_FRAC)

    return {
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train : n_train + n_val]),
        "test": sorted(ids[n_train + n_val :]),
    }


def main() -> None:
    prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    prompt_ids = [p["prompt_id"] for p in prompts]
    splits = create_splits(prompt_ids)

    out = {
        "split_seed": SPLIT_SEED,
        "source": str(PROMPTS_PATH.relative_to(REPO_ROOT)),
        "total_prompts": len(prompt_ids),
        "counts": {k: len(v) for k, v in splits.items()},
        **splits,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[+] wrote {OUT_PATH}")
    print(f"[+] counts: {out['counts']} (total {out['total_prompts']})")


if __name__ == "__main__":
    main()
