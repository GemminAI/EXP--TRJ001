"""Generate Phase 2 Type A prompt candidates (plan v1.4, §5.4): a SECOND,
disjoint batch of fictional-entity prompts, since the full 1,000-prompt v1.4
pool (`configs/prompts_v1.json`) was entirely consumed as Phase 1 pilot data
(all 1,000 prompts x 8 samples, per the v1.4 Phase 1 re-run) -- leaving zero
independent inventory for Phase 2, contrary to the original design (which
assumed only 100 of 1,000 would be used for Phase 1).

Reuses the same entity-generation functions, lexical fragments, and
templates as `generate_type_a_prompts.py` (so entities are constructed
identically) but with a NEW seed AND explicit exclusion of every entity
already considered in the first batch (`configs/prompts_v1_candidates.json`,
1,100 candidates spanning both the 1,000 kept and ~100 dropped for Wikipedia
collisions) -- so no entity can appear in both Phase 1 and Phase 2, per
§5.3/G2's prohibition on reusing Phase-1 prompts in Phase 2.
"""

from __future__ import annotations

import json
import random
from itertools import product
from pathlib import Path

from generate_type_a_prompts import (
    ADJ,
    CHEM_FRAG_A,
    CHEM_SUFFIX,
    FIELD,
    FIRST_NAMES,
    FRAG_A,
    FRAG_B,
    NOUN,
    PLACE_KIND,
    TEMPLATES,
    TOPIC,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIRST_BATCH_CANDIDATES_PATH = REPO_ROOT / "configs" / "prompts_v1_candidates.json"
OUT_PATH = REPO_ROOT / "configs" / "prompts_v1_phase2_candidates.json"

PROMPT_GEN_SEED_PHASE2 = 20260913  # distinct from PROMPT_GEN_SEED (20260912)
CANDIDATES_PER_DOMAIN = 100  # target ~500 total; chemicals' pool is the
# tightest (324 total combos, 220 already used in batch 1 -> 104 remain), so
# this domain may fall short of 100 if collisions eat into the remainder --
# handled by raising an error rather than silently shipping fewer, so a
# shortfall is visible rather than silently absorbed (matches
# check_entity_collisions.py's own convention).


def _load_used_entities() -> set[str]:
    if not FIRST_BATCH_CANDIDATES_PATH.exists():
        raise FileNotFoundError(
            f"{FIRST_BATCH_CANDIDATES_PATH} not found -- cannot verify "
            f"disjointness from Phase 1's entity pool"
        )
    candidates = json.loads(FIRST_BATCH_CANDIDATES_PATH.read_text(encoding="utf-8"))
    return {c["entity"] for c in candidates}


def _unique_samples_excluding(rng: random.Random, pool: list[str], n: int, exclude: set[str]) -> list[str]:
    available = [x for x in pool if x not in exclude]
    if len(available) < n:
        raise ValueError(
            f"pool too small after excluding {len(pool) - len(available)} "
            f"already-used entities: need {n}, have {len(available)}"
        )
    rng.shuffle(available)
    return available[:n]


def gen_places(rng: random.Random, n: int, exclude: set[str]) -> list[str]:
    combos = [a + b for a, b in product(FRAG_A, FRAG_B)]
    rng.shuffle(combos)
    out: list[str] = []
    for name in combos:
        if len(out) >= n:
            break
        kind = rng.choice(PLACE_KIND)
        candidate = f"{name} {kind}".strip() if kind else name
        if candidate in exclude or candidate in out:
            continue
        out.append(candidate)
    if len(out) < n:
        raise ValueError(f"could not generate {n} unique excluded places, got {len(out)}")
    return out


def gen_persons(rng: random.Random, n: int, exclude: set[str]) -> list[str]:
    surnames = [a + b for a, b in product(FRAG_A, FRAG_B)]
    rng.shuffle(surnames)
    first = list(FIRST_NAMES)
    rng.shuffle(first)
    out = []
    i = 0
    attempts = 0
    while len(out) < n and attempts < 10 * n + len(surnames):
        fn = first[i % len(first)]
        sn = surnames[i % len(surnames)]
        candidate = f"{fn} {sn}"
        i += 1
        attempts += 1
        if candidate in exclude or candidate in out:
            continue
        out.append(candidate)
    if len(out) < n:
        raise ValueError(f"could not generate {n} unique excluded persons, got {len(out)}")
    return out


def gen_movies(rng: random.Random, n: int, exclude: set[str]) -> list[str]:
    pattern_a = [f"{adj} {noun}" for adj, noun in product(ADJ, NOUN)]
    places = [a + b for a, b in product(FRAG_A, FRAG_B)]
    pattern_b = [f"The {noun} of {place}" for noun, place in product(NOUN, places[:40])]
    combined = pattern_a + pattern_b
    return _unique_samples_excluding(rng, combined, n, exclude)


def gen_papers(rng: random.Random, n: int, exclude: set[str]) -> list[str]:
    combos = [f"{topic} in {field}" for topic, field in product(TOPIC, FIELD)]
    return _unique_samples_excluding(rng, combos, n, exclude)


def gen_chemicals(rng: random.Random, n: int, exclude: set[str]) -> list[str]:
    combos = [a + b for a, b in product(CHEM_FRAG_A, CHEM_SUFFIX)]
    combos = [name[0].upper() + name[1:] for name in combos]
    return _unique_samples_excluding(rng, combos, n, exclude)


DOMAIN_GENERATORS = {
    "fictional_movie": gen_movies,
    "fictional_person": gen_persons,
    "fictional_paper": gen_papers,
    "fictional_chemical": gen_chemicals,
    "fictional_location": gen_places,
}


def generate(n_per_domain: int = CANDIDATES_PER_DOMAIN, seed: int = PROMPT_GEN_SEED_PHASE2) -> list[dict]:
    used_entities = _load_used_entities()
    rng = random.Random(seed)
    candidates = []
    for domain, generator in DOMAIN_GENERATORS.items():
        entities = generator(rng, n_per_domain, used_entities)
        overlap = set(entities) & used_entities
        assert not overlap, f"{domain}: {len(overlap)} entities overlap with batch 1: {overlap}"
        phrasing = TEMPLATES[domain]
        for i, entity in enumerate(entities):
            text = phrasing[i % len(phrasing)].format(x=entity)
            candidates.append(
                {
                    "candidate_id": f"cand_p2_{domain}_{i + 1:04d}",
                    "template": domain,
                    "entity": entity,
                    "text": text,
                }
            )
    return candidates


def main() -> None:
    candidates = generate()
    OUT_PATH.write_text(json.dumps(candidates, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    by_domain: dict[str, int] = {}
    for c in candidates:
        by_domain[c["template"]] = by_domain.get(c["template"], 0) + 1
    print(f"wrote {len(candidates)} Phase 2 candidates to {OUT_PATH}")
    for domain, count in by_domain.items():
        print(f"  {domain}: {count}")


if __name__ == "__main__":
    main()
