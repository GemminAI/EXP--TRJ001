"""Freezes the ablation prompt ID set (plan v1.4 §2.3/§6.3): selected from
the Phase 2 Test split, before Phase 2 generation begins.

Deviation from the pre-registered target (flagged, not silently absorbed):
v1.3/v1.4 specify 200 ablation prompts. Because Phase 1's v1.4 re-run
consumed the ENTIRE original 1,000-prompt Type A pool (a consequence of the
explicit "Phase 1 full re-run, 1,000 x 8" instruction, not of this script),
a second, disjoint prompt batch had to be generated for Phase 2
(`generate_type_a_prompts_phase2.py`), yielding only 493 clean prompts ->
Test split (20%) = 98 prompts, short of 200. Rather than generating a much
larger pool solely to backfill this secondary artifact, or silently reporting
a fabricated "200" that doesn't exist, this freezes ALL available Test-split
prompts (98) and records the shortfall explicitly in the output file.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_PATH = REPO_ROOT / "configs" / "data_splits.json"
OUT_PATH = REPO_ROOT / "configs" / "ablation_prompt_ids.json"

TARGET_COUNT = 200  # v1.3/v1.4 §2.3/§6.3 pre-registered target


def main() -> None:
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    test_ids = splits["test"]

    ablation_ids = sorted(test_ids)[:TARGET_COUNT]  # no-op cap; documents intent
    shortfall = TARGET_COUNT - len(ablation_ids)

    out = {
        "target_count": TARGET_COUNT,
        "actual_count": len(ablation_ids),
        "shortfall": max(shortfall, 0),
        "shortfall_reason": (
            None
            if shortfall <= 0
            else (
                "Phase 1's v1.4 full re-run (1,000 x 8) consumed the entire "
                "original Type A prompt pool, leaving no independent inventory "
                "for Phase 2; a second, disjoint 493-prompt batch was generated "
                "instead of the originally-planned ~900, so the Test split "
                "(20%) yields only 98 prompts, not the pre-registered 200. "
                "See generate_type_a_prompts_phase2.py and "
                "create_data_splits.py."
            )
        ),
        "source_split": "test",
        "prompt_ids": ablation_ids,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[+] wrote {OUT_PATH}: {len(ablation_ids)}/{TARGET_COUNT} ablation prompt IDs frozen")
    if shortfall > 0:
        print(f"[!] shortfall of {shortfall} vs the pre-registered target -- see shortfall_reason in the file")


if __name__ == "__main__":
    main()
