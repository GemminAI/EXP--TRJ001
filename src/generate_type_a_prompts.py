"""Generate Type A (fictional-entity) prompt candidates for EXP-2026-NVS-001 (plan v1.3 §5.3).

Produces MORE than the frozen target (200/domain, 1,000 total) so that
`check_entity_collisions.py` can discard any candidate that turns out to
collide with a real Wikipedia entity and still land on exactly 200 clean
prompts per domain. This script itself does not check for collisions and
does not write the frozen `configs/prompts_v1.json` -- that is
`check_entity_collisions.py`'s job, per §5.3's separation of "generate" from
"collision-check and freeze".

All entity names are built by combining invented lexical fragments (never
real dictionary words) specifically to keep the true collision rate low --
the Wikipedia check is still authoritative and mandatory (§5.3, T2).

Determinism: a fixed seed (`PROMPT_GEN_SEED`) makes regeneration
reproducible; the actual frozen artifact is the output JSON (hashed into
`freeze/FREEZE_MANIFEST.md`), not this script's seed.
"""

from __future__ import annotations

import json
import random
from itertools import product
from pathlib import Path

PROMPT_GEN_SEED = 20260912
CANDIDATES_PER_DOMAIN = 220  # 200 frozen target + 20 buffer for collision drops

OUT_PATH = Path(__file__).resolve().parents[1] / "configs" / "prompts_v1_candidates.json"

# --- invented lexical fragments (never real words) -------------------------

FRAG_A = [
    "Kal", "Ves", "Thal", "Mor", "Bren", "Sol", "Ashen", "Dun", "Fen", "Or",
    "Skol", "Rav", "Ely", "Gorn", "Ist", "Quor", "Nyth", "Zeph", "Halden",
    "Corvi", "Ilan", "Ulric", "Vesk", "Tarn", "Wren", "Ombra", "Calder",
    "Ferrin", "Junip", "Lasho",
]
FRAG_B = [
    "ith", "orne", "holt", "mere", "dale", "reach", "wick", "ven", "gard",
    "thorn", "moor", "stead", "crest", "fell", "ridge", "hollow", "spire",
    "vale", "brook", "haven",
]

FIRST_NAMES = [
    "Elena", "Marcus", "Ines", "Dorian", "Sable", "Renata", "Callum", "Wren",
    "Ivo", "Petra", "Soren", "Talia", "Faris", "Nadia", "Lucian", "Esme",
    "Rowan", "Vera", "Anselm", "Nyla", "Tobias", "Miriel", "Godric", "Selin",
    "Iona", "Bastian", "Yulia", "Cormac", "Adele", "Ronan",
]

ADJ = [
    "Hollow", "Iron", "Silent", "Crimson", "Last", "Broken", "Endless",
    "Forgotten", "Distant", "Velvet", "Wandering", "Quiet", "Frozen",
    "Golden", "Midnight", "Pale", "Amber", "Restless", "Faint", "Deep",
]
NOUN = [
    "Shadow", "Season", "Harbor", "Signal", "Garden", "Horizon", "Archive",
    "Current", "Passage", "Echo", "Compass", "Ledger", "Tide", "Refuge",
    "Orbit", "Verdict", "Static", "Threshold", "Lantern", "Cipher",
]

TOPIC = [
    "Latent Drift Correction", "Sparse Curvature Estimation",
    "Cross-Modal Attenuation", "Recursive Boundary Inference",
    "Temporal Manifold Realignment", "Adaptive Noise Disentanglement",
    "Nonlinear Phase Recovery", "Stochastic Gradient Folding",
    "Hierarchical Context Bleaching", "Anisotropic Feature Collapse",
    "Residual Topology Smoothing", "Discrete Saliency Reweighting",
    "Piecewise Embedding Contraction", "Asymmetric Drift Regularization",
    "Latent Frame Interleaving", "Bounded Entropy Backpropagation",
    "Sequential Manifold Pruning", "Cascaded Signal Rebinding",
    "Marginal Trajectory Flattening", "Sparse Attention Redistribution",
]
FIELD = [
    "Sequential Encoders", "Distributed Sensor Arrays",
    "Synthetic Cohort Studies", "Low-Resource Dialogue Systems",
    "Multi-Agent Simulation", "Embedded Vision Pipelines",
    "Sparse Tensor Networks", "Federated Query Systems",
    "Neuromorphic Substrates", "Adaptive Control Loops",
    "Streaming Recommendation Systems", "Procedural Audio Synthesis",
    "Decentralized Sensor Fusion", "Robotic Grasp Planning",
    "Cross-Lingual Retrieval", "Time-Series Anomaly Detection",
    "Graph-Structured Memory", "Mixed-Precision Training",
    "Edge-Device Inference", "Weakly Supervised Segmentation",
]

CHEM_FRAG_A = [
    "Xylo", "Bromo", "Fluor", "Cerv", "Thal", "Nitro", "Sulfa", "Chloro",
    "Vinyl", "Acet", "Pyra", "Meth", "Ethox", "Carbo", "Phospha", "Silor",
    "Quinal", "Tresin",
]
CHEM_SUFFIX = [
    "therin", "zolamide", "furanate", "oxysilane", "chromide", "peptin",
    "valinate", "siloxide", "nitrile", "sulfonate", "carbamide", "phenolate",
    "triazole", "glycoside", "aminate", "oxindole", "cresylate", "phosphonium",
]

PLACE_KIND = ["Valley", "Province", "Harbor", "Basin", "Highlands", "Crossing", "Reach", ""]

TEMPLATES = {
    # v1.4 §5.3 (2026-09-13): rewritten from confirmatory phrasing
    # ("Can you tell me about...?", "Who is...?") to descriptive phrasing
    # that presupposes the entity exists and asks for its content directly
    # (plot summary / achievements / paper's claims / synthesis-properties /
    # history-population). No explicit "don't say you don't know"
    # instruction is added -- the presupposition itself is the manipulation
    # under test, not an instruction not to abstain.
    "fictional_movie": [
        'Summarize the plot of the movie "{x}".',
        'Describe the plot and main characters of "{x}".',
        'Write a brief synopsis of "{x}", including its ending.',
        'What happens in "{x}"? Summarize the story from beginning to end.',
    ],
    "fictional_person": [
        "Describe {x}'s main accomplishments and career.",
        "Summarize {x}'s life and the work they are known for.",
        "What is {x} best known for? Describe their most notable achievements.",
        "Give an overview of {x}'s career and contributions to their field.",
    ],
    "fictional_paper": [
        'Summarize the main claims and findings of the paper "{x}".',
        'Describe the methodology used in "{x}" and what its key results were.',
        'What are the main contributions of "{x}"? Summarize them.',
        'Summarize the argument and conclusions of "{x}".',
    ],
    "fictional_chemical": [
        "Describe the synthesis and chemical properties of {x}.",
        "Summarize how {x} is synthesized and what it is typically used for.",
        "Describe the structure, reactivity, and applications of {x}.",
        "What are the chemical properties of {x}, and how is it produced?",
    ],
    "fictional_location": [
        "Describe the history and population of {x}.",
        "Summarize the history and geography of {x}.",
        "Describe what {x} is known for and give an overview of its population.",
        "Give an overview of {x}'s history and demographics.",
    ],
}


def _unique_samples(rng: random.Random, pool: list, n: int) -> list:
    if len(pool) < n:
        raise ValueError(f"pool too small: need {n}, have {len(pool)}")
    pool = list(pool)
    rng.shuffle(pool)
    return pool[:n]


def gen_places(rng: random.Random, n: int) -> list[str]:
    combos = [a + b for a, b in product(FRAG_A, FRAG_B)]
    base = _unique_samples(rng, combos, n)
    out = []
    for name in base:
        kind = rng.choice(PLACE_KIND)
        out.append(f"{name} {kind}".strip() if kind else name)
    return out


def gen_persons(rng: random.Random, n: int) -> list[str]:
    surnames = [a + b for a, b in product(FRAG_A, FRAG_B)]
    rng.shuffle(surnames)
    first = list(FIRST_NAMES)
    rng.shuffle(first)
    out = []
    i = 0
    while len(out) < n:
        fn = first[i % len(first)]
        sn = surnames[i % len(surnames)]
        out.append(f"{fn} {sn}")
        i += 1
    return out[:n]


def gen_movies(rng: random.Random, n: int) -> list[str]:
    pattern_a = [f"{adj} {noun}" for adj, noun in product(ADJ, NOUN)]
    places = [a + b for a, b in product(FRAG_A, FRAG_B)]
    pattern_b = [f"The {noun} of {place}" for noun, place in product(NOUN, places[:40])]
    combined = pattern_a + pattern_b
    return _unique_samples(rng, combined, n)


def gen_papers(rng: random.Random, n: int) -> list[str]:
    combos = [f"{topic} in {field}" for topic, field in product(TOPIC, FIELD)]
    return _unique_samples(rng, combos, n)


def gen_chemicals(rng: random.Random, n: int) -> list[str]:
    combos = [a + b for a, b in product(CHEM_FRAG_A, CHEM_SUFFIX)]
    return [name[0].upper() + name[1:] for name in _unique_samples(rng, combos, n)]


DOMAIN_GENERATORS = {
    "fictional_movie": gen_movies,
    "fictional_person": gen_persons,
    "fictional_paper": gen_papers,
    "fictional_chemical": gen_chemicals,
    "fictional_location": gen_places,
}


def generate(n_per_domain: int = CANDIDATES_PER_DOMAIN, seed: int = PROMPT_GEN_SEED) -> list[dict]:
    rng = random.Random(seed)
    candidates = []
    for domain, generator in DOMAIN_GENERATORS.items():
        entities = generator(rng, n_per_domain)
        phrasing = TEMPLATES[domain]
        for i, entity in enumerate(entities):
            text = phrasing[i % len(phrasing)].format(x=entity)
            candidates.append(
                {
                    "candidate_id": f"cand_{domain}_{i + 1:04d}",
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
    print(f"wrote {len(candidates)} candidates to {OUT_PATH}")
    for domain, count in by_domain.items():
        print(f"  {domain}: {count}")


if __name__ == "__main__":
    main()
