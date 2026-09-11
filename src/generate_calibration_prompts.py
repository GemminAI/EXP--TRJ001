"""Generate the calibration corpus for PCA fitting (plan v1.3 §2.3).

`configs/calibration_prompts_v1.json`: 200 prompts "completely non-overlapping
with this experiment" (§2.3), used only to compute Layer 16 hidden-state
statistics for the frozen (mu, W_128) projection -- never evaluated for
hallucination, never part of Type A/B. They are therefore deliberately
generic, real-topic, non-fictional, non-single-fact-QA prompts: the opposite
shape of both Type A (fictional entities) and Type B (single-fact QA), so
there is no content overlap with either by construction.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from random import Random

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "configs" / "calibration_prompts_v1.json"

CALIBRATION_SEED = 20260912
TARGET_COUNT = 200

TOPICS = [
    "photosynthesis", "supply and demand", "the water cycle", "compound interest",
    "blockchain technology", "the immune system", "machine learning", "black holes",
    "the French Revolution", "plate tectonics", "quantum computing", "inflation",
    "the scientific method", "renewable energy", "the human genome",
    "climate change", "artificial neural networks", "the stock market",
    "ancient Roman architecture", "the Great Barrier Reef", "vaccination",
    "cryptography", "the printing press", "urban planning", "coral bleaching",
    "the Industrial Revolution", "solar panels", "genetic engineering",
    "the internet's history", "space exploration", "electric vehicles",
    "the periodic table", "sleep cycles", "nutrition labels", "wildfire ecology",
    "the stock exchange", "cybersecurity", "3D printing", "ocean currents",
    "the nervous system", "supply chains",
]

TASKS = [
    "learn a new language", "start a vegetable garden", "improve my public speaking",
    "save money for retirement", "train for a marathon", "write a resume",
    "reduce food waste at home", "learn to play the guitar", "start meditating",
    "organize a small home office", "plan a road trip", "improve my sleep habits",
    "start freelancing", "learn to cook Italian food", "build a personal budget",
    "prepare for a job interview", "learn basic woodworking", "start composting",
    "improve my time management", "learn to swim as an adult",
]

PAIRS = [
    ("a virus", "a bacterium"), ("weather", "climate"), ("a comet", "an asteroid"),
    ("a crocodile", "an alligator"), ("HTTP", "HTTPS"), ("a stock", "a bond"),
    ("a republic", "a democracy"), ("RAM", "storage"), ("a hurricane", "a typhoon"),
    ("mitosis", "meiosis"), ("a virus", "malware"), ("a lake", "a pond"),
    ("an omnivore", "a herbivore"), ("civil law", "criminal law"),
    ("a planet", "a dwarf planet"), ("weather forecasting", "climate modeling"),
    ("a museum", "a gallery"), ("a recession", "a depression"),
    ("an island", "a peninsula"), ("a font", "a typeface"),
]

PHENOMENA = [
    "the sky appear blue", "leaves change color in autumn", "ice float on water",
    "tides rise and fall", "metal rust over time", "bread rise when baked",
    "mirrors reverse images left-to-right but not top-to-bottom",
    "the seasons change", "thunder follow lightning", "glass appear transparent",
    "hot air rise", "soap remove grease", "bicycles stay upright while moving",
    "the moon appear to change shape", "fruit ripen faster near bananas",
    "carbonated drinks go flat", "static electricity build up in dry weather",
    "the ocean appear blue", "echoes occur in large rooms",
    "milk curdle when it goes sour",
]

TEMPLATES = {
    "explain_a_concept": ("Explain how {x} works, in terms a curious beginner would understand.", TOPICS),
    "opinion": ("What is your perspective on {x}? Explain your reasoning.", TOPICS),
    "summarize": ("Summarize the main ideas behind {x} in a few sentences.", TOPICS),
    "howto": ("How do I {x}? Give a step-by-step outline.", TASKS),
    "advice": ("What advice would you give someone who wants to {x}?", TASKS),
    "compare": ("What is the difference between {x} and {y}?", PAIRS),
    "reasoning": ("Why does {x}?", PHENOMENA),
    "creative": ("Write a short, engaging paragraph about {x}.", TOPICS),
}


def _fill(template: str, item) -> str:
    if isinstance(item, tuple):
        return template.format(x=item[0], y=item[1])
    return template.format(x=item)


def generate(n: int = TARGET_COUNT, seed: int = CALIBRATION_SEED) -> list[dict]:
    rng = Random(seed)
    pool: list[tuple[str, str]] = []
    for category, (template, items) in TEMPLATES.items():
        for item in items:
            pool.append((category, _fill(template, item)))
    rng.shuffle(pool)

    seen_text: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for category, text in pool:
        if text in seen_text:
            continue
        seen_text.add(text)
        deduped.append((category, text))

    if len(deduped) < n:
        raise ValueError(f"only {len(deduped)} unique calibration prompts available, need {n}")

    selected = deduped[:n]
    return [
        {"prompt_id": f"calib_{i + 1:04d}", "category": category, "text": text}
        for i, (category, text) in enumerate(selected)
    ]


def main() -> None:
    prompts = generate()
    OUT_PATH.write_text(json.dumps(prompts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(prompts)} calibration prompts to {OUT_PATH}")


if __name__ == "__main__":
    main()
