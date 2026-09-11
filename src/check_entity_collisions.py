"""Wikipedia collision check for Type A candidates (plan v1.3 §5.3, T2).

Reads `configs/prompts_v1_candidates.json` (from generate_type_a_prompts.py),
checks each candidate's entity name against the real Wikipedia API, drops any
that resolve to an existing page (including redirects), and freezes exactly
200-per-domain / 1,000-total clean prompts into `configs/prompts_v1.json`.
Writes a full log -- including the residual-risk statement required by the
plan (Wikipedia-only check, no second search-API pass) -- to
`results/collision_check_log.md`.

Rate-limiting / retry-on-429 pattern and the "missing"-flag existence check
are reused from the organization's existing Wikipedia API client
(`GemminAI/experiments/EXP-Ubuntu/run/stage1_ingest.py`), batched here
(<=50 titles per request, the MediaWiki API's own limit) since ~1,100
candidates would otherwise mean ~1,100 separate requests.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_PATH = REPO_ROOT / "configs" / "prompts_v1_candidates.json"
OUT_PROMPTS_PATH = REPO_ROOT / "configs" / "prompts_v1.json"
LOG_PATH = REPO_ROOT / "results" / "collision_check_log.md"

API = "https://en.wikipedia.org/w/api.php"
BATCH_SIZE = 50
TARGET_PER_DOMAIN = 200
# Wikimedia's API etiquette requires a descriptive User-Agent (blank/default
# UAs get a 403); see https://meta.wikimedia.org/wiki/User-Agent_policy
USER_AGENT = (
    "EXP-2026-NVS-001-collision-check/1.0 "
    "(https://github.com/GemminAI/EXP--TRJ001; research/pre-registration experiment)"
)


def polite_get(client: httpx.Client, params: dict, *, retries: int = 6) -> httpx.Response:
    """Reused pattern from stage1_ingest.py: 0.4s politeness delay, exponential
    backoff on 429 honoring Retry-After."""
    delay = 1.0
    for attempt in range(retries):
        time.sleep(0.4)
        resp = client.get(API, params=params, timeout=30)
        if resp.status_code == 429:
            wait = float(resp.headers.get("retry-after", delay))
            print(f"    429 rate-limited, backing off {wait:.1f}s (attempt {attempt + 1}/{retries})")
            time.sleep(wait)
            delay *= 2
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp


def check_batch(client: httpx.Client, titles: list[str]) -> dict[str, bool]:
    """Returns {title: collides_with_real_page}."""
    params = {
        "action": "query",
        "format": "json",
        "titles": "|".join(titles),
        "redirects": 1,
    }
    resp = polite_get(client, params)
    data = resp.json()
    query = data.get("query", {})

    # Map normalization / redirects back to the originally-queried title.
    canonical_to_original = {t: t for t in titles}
    for norm in query.get("normalized", []):
        canonical_to_original[norm["to"]] = norm["from"]
    for redir in query.get("redirects", []):
        # redirect target -> whatever the (possibly normalized) source was
        src = redir["from"]
        original = canonical_to_original.get(src, src)
        canonical_to_original[redir["to"]] = original

    result = {t: False for t in titles}
    for page in query.get("pages", {}).values():
        title_seen = page.get("title")
        original = canonical_to_original.get(title_seen, title_seen)
        if original not in result:
            continue
        result[original] = "missing" not in page
    return result


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
            kept = 0
            checked = 0
            domain_collisions = []
            entities = [c["entity"] for c in items]
            collision_map: dict[str, bool] = {}
            for i in range(0, len(entities), BATCH_SIZE):
                batch = entities[i : i + BATCH_SIZE]
                collision_map.update(check_batch(client, batch))
                checked += len(batch)

            for c in items:
                if kept >= TARGET_PER_DOMAIN:
                    break
                if collision_map.get(c["entity"], False):
                    domain_collisions.append(c["entity"])
                    continue
                final.append(c)
                kept += 1

            if kept < TARGET_PER_DOMAIN:
                raise RuntimeError(
                    f"domain {domain!r}: only {kept}/{TARGET_PER_DOMAIN} clean candidates "
                    f"after dropping {len(domain_collisions)} collisions out of {checked} "
                    f"checked -- generate more candidates (see generate_type_a_prompts.py) "
                    f"rather than shipping an under-filled domain"
                )

            collisions.extend(domain_collisions)
            counts[domain] = {
                "checked": checked,
                "collisions": len(domain_collisions),
                "kept": kept,
            }

    prompts = [
        {
            "prompt_id": f"type_a_{c['template']}_{i + 1:04d}",
            "prompt_type": "type_a",
            "template": c["template"],
            "text": c["text"],
            "reference_key": None,
        }
        for i, c in enumerate(final)
    ]
    OUT_PROMPTS_PATH.write_text(json.dumps(prompts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    _write_log(counts, collisions)
    print(f"wrote {len(prompts)} clean Type A prompts to {OUT_PROMPTS_PATH}")
    print(f"log: {LOG_PATH}")


def _write_log(counts: dict[str, dict[str, int]], collisions: list[str]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Collision Check Log — Type A prompts (plan v1.3 §5.3, T2)",
        "",
        f"Run (UTC): {now}",
        "",
        "## Method",
        "",
        "Wikipedia API only (`action=query`, `redirects=1`, batched <=50 titles/request). "
        "A candidate is dropped if its exact title resolves to any existing Wikipedia page "
        "(including via redirect). **No second search-API pass was performed** (T2: no "
        "existing Google Custom Search or equivalent implementation was found in the "
        "organization; adding one was judged out of scope for this experiment).",
        "",
        "## Residual risk (do not read as \"100% verified\")",
        "",
        "This check only catches collisions Wikipedia's own title/redirect resolution "
        "surfaces. It will not catch: real entities absent from English Wikipedia (minor "
        "place names, non-English-language notable figures, unindexed/very recent works), "
        "near-miss title variants not covered by redirects, or entities documented only on "
        "other sites. This residual risk is accepted per plan v1.3 §5.3 and is not resolved "
        "by this log -- it is recorded here for the record.",
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
