"""Wikipedia collision check for Phase 2 Type A candidates (plan v1.4 §5.4).

Same method as `check_entity_collisions.py` (Wikipedia API only, T2), applied
to `configs/prompts_v1_phase2_candidates.json` (the second, disjoint batch --
see `generate_type_a_prompts_phase2.py`). Unlike the Phase 1 freeze, this
keeps EVERY clean candidate (no fixed per-domain cap) since the goal is
simply enough independent inventory to cover M (~444, per §5.4's formula
from the real Phase 1 mixed_rate/pos_rate_in_mixed), not a frozen balanced
1,000-prompt set.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

from check_entity_collisions import USER_AGENT, check_batch

REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_PATH = REPO_ROOT / "configs" / "prompts_v1_phase2_candidates.json"
OUT_PROMPTS_PATH = REPO_ROOT / "configs" / "prompts_v1_phase2.json"
LOG_PATH = REPO_ROOT / "results" / "collision_check_log_phase2.md"

BATCH_SIZE = 50


def run() -> None:
    candidates = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))

    by_domain: dict[str, list[dict]] = {}
    for c in candidates:
        by_domain.setdefault(c["template"], []).append(c)

    collisions: list[dict] = []
    final: list[dict] = []
    counts: dict[str, dict[str, int]] = {}

    with httpx.Client(headers={"User-Agent": USER_AGENT}) as client:
        for domain, items in by_domain.items():
            entities = [c["entity"] for c in items]
            collision_map: dict[str, bool] = {}
            for i in range(0, len(entities), BATCH_SIZE):
                batch = entities[i : i + BATCH_SIZE]
                collision_map.update(check_batch(client, batch))

            domain_collisions = []
            kept = 0
            for c in items:
                if collision_map.get(c["entity"], False):
                    domain_collisions.append(c["entity"])
                    continue
                final.append(c)
                kept += 1

            collisions.extend(domain_collisions)
            counts[domain] = {
                "checked": len(entities),
                "collisions": len(domain_collisions),
                "kept": kept,
            }

    prompts = [
        {
            "prompt_id": f"type_a_p2_{c['template']}_{i + 1:04d}",
            "prompt_type": "type_a",
            "template": c["template"],
            "text": c["text"],
            "reference_key": None,
        }
        for i, c in enumerate(final)
    ]
    OUT_PROMPTS_PATH.write_text(json.dumps(prompts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    _write_log(counts, collisions, len(prompts))
    print(f"wrote {len(prompts)} clean Phase 2 Type A prompts to {OUT_PROMPTS_PATH}")
    print(f"log: {LOG_PATH}")


def _write_log(counts: dict[str, dict[str, int]], collisions: list[str], total_kept: int) -> None:
    now = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Collision Check Log — Phase 2 Type A prompts (plan v1.4 §5.4)",
        "",
        f"Run (UTC): {now}",
        "",
        "## Method",
        "",
        "Same as the Phase 1 batch (`results/collision_check_log.md`): Wikipedia API only, "
        "`action=query`, `redirects=1`, batched <=50 titles/request. No second search-API "
        "pass (T2). Residual risk statement applies identically -- see the Phase 1 log.",
        "",
        f"**Total clean prompts kept: {total_kept}**",
        "",
        "## Per-domain results",
        "",
        "| domain | checked | collisions | kept |",
        "|---|---|---|---|",
    ]
    for domain, c in counts.items():
        lines.append(f"| {domain} | {c['checked']} | {c['collisions']} | {c['kept']} |")

    lines += ["", "## Dropped (collided) candidates", ""]
    if collisions:
        for entity in collisions:
            lines.append(f"- {entity}")
    else:
        lines.append("(none)")
    lines.append("")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
