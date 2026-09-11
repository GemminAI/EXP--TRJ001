"""Author Type B (real-entity factual) prompts for EXP-2026-NVS-001 (plan v1.3 §5.3, T1).

Unlike Type A, these cannot be procedurally generated: each item is a
single-fact, real-world claim that must actually be true, so the underlying
facts are curated here as data, not synthesized.

Domains mirror Type A's five templates (movie / person / paper_book /
chemical / location), 40 items each, 200 total -- a budget independent from
Type A's 1,000, per plan v1.3 T1.

Question form is deliberately uniform per domain and answer-typed (a single
year, proper name, chemical formula, or capital city) so that
`eval_hallucination.py`'s normalize-and-exact-match Type B check
(lowercase + strip symbols + strip whitespace) can grade it without
ambiguity.

IMPORTANT -- provenance / what this script does NOT do:
Facts below were authored from general knowledge, chosen specifically for
being extremely well-established/canonical (classic films, foundational
historical figures, canonical novels, standard chemical formulas,
uncontroversial national capitals) to minimize error risk. They have NOT
been cross-checked against a live source (Wikipedia/Wikidata) item-by-item
in this pass. Plan Appendix B item 2 ("Type B の正解集合スナップショット
日時と数値許容誤差規則") is therefore still explicitly open -- this script
does not resolve it. `snapshot_date` is left null and
`verification_status` is set accordingly; do not treat this reference set
as frozen-quality until a real verification pass (and a numeric-tolerance
decision) has been done.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PROMPTS_PATH = REPO_ROOT / "configs" / "prompts_typeB_v1.json"
OUT_REFERENCE_PATH = REPO_ROOT / "configs" / "typeB_reference.json"

# --- domain: movie -- release year -----------------------------------------
MOVIES = [
    ("Jaws", "1975"), ("Star Wars", "1977"), ("The Godfather", "1972"),
    ("Titanic", "1997"), ("Jurassic Park", "1993"), ("The Matrix", "1999"),
    ("Forrest Gump", "1994"), ("Pulp Fiction", "1994"),
    ("The Shawshank Redemption", "1994"), ("The Lion King", "1994"),
    ("Toy Story", "1995"), ("Casablanca", "1942"), ("Gone with the Wind", "1939"),
    ("The Wizard of Oz", "1939"), ("Psycho", "1960"),
    ("2001: A Space Odyssey", "1968"), ("E.T. the Extra-Terrestrial", "1982"),
    ("Back to the Future", "1985"), ("Rocky", "1976"), ("Grease", "1978"),
    ("The Sound of Music", "1965"), ("Schindler's List", "1993"),
    ("Braveheart", "1995"), ("Gladiator", "2000"), ("The Dark Knight", "2008"),
    ("Avatar", "2009"), ("Inception", "2010"), ("Frozen", "2013"),
    ("Interstellar", "2014"), ("La La Land", "2016"), ("Parasite", "2019"),
    ("Avengers: Endgame", "2019"), ("Home Alone", "1990"), ("Aladdin", "1992"),
    ("The Lord of the Rings: The Fellowship of the Ring", "2001"),
    ("Harry Potter and the Sorcerer's Stone", "2001"), ("Spider-Man", "2002"),
    ("Finding Nemo", "2003"), ("The Sixth Sense", "1999"), ("Jurassic World", "2015"),
]

# --- domain: person -- birth year -------------------------------------------
PERSONS = [
    ("Albert Einstein", "1879"), ("Charles Darwin", "1809"),
    ("Abraham Lincoln", "1809"), ("Wolfgang Amadeus Mozart", "1756"),
    ("Ludwig van Beethoven", "1770"), ("Napoleon Bonaparte", "1769"),
    ("Mahatma Gandhi", "1869"), ("Martin Luther King Jr.", "1929"),
    ("Nelson Mandela", "1918"), ("Marie Curie", "1867"),
    ("Winston Churchill", "1874"), ("Franklin D. Roosevelt", "1882"),
    ("Thomas Edison", "1847"), ("Nikola Tesla", "1856"),
    ("Charles Dickens", "1812"), ("Mark Twain", "1835"),
    ("William Shakespeare", "1564"), ("Leonardo da Vinci", "1452"),
    ("Vincent van Gogh", "1853"), ("Pablo Picasso", "1881"),
    ("Michael Jackson", "1958"), ("Elvis Presley", "1935"),
    ("John Lennon", "1940"), ("Walt Disney", "1901"), ("Steve Jobs", "1955"),
    ("Bill Gates", "1955"), ("Sigmund Freud", "1856"), ("Karl Marx", "1818"),
    ("Vladimir Lenin", "1870"), ("Queen Elizabeth II", "1926"),
    ("Barack Obama", "1961"), ("Muhammad Ali", "1942"),
    ("Neil Armstrong", "1930"), ("Stephen Hawking", "1942"),
    ("Galileo Galilei", "1564"), ("Charlie Chaplin", "1889"),
    ("Amelia Earhart", "1897"), ("Queen Victoria", "1819"),
    ("Thomas Jefferson", "1743"), ("George Washington", "1732"),
]

# --- domain: paper_book -- author -------------------------------------------
BOOKS = [
    ("Pride and Prejudice", "Jane Austen"), ("Moby-Dick", "Herman Melville"),
    ("War and Peace", "Leo Tolstoy"), ("Crime and Punishment", "Fyodor Dostoevsky"),
    ("1984", "George Orwell"), ("Animal Farm", "George Orwell"),
    ("To Kill a Mockingbird", "Harper Lee"),
    ("The Great Gatsby", "F. Scott Fitzgerald"), ("Frankenstein", "Mary Shelley"),
    ("Dracula", "Bram Stoker"),
    ("The Adventures of Huckleberry Finn", "Mark Twain"),
    ("The Adventures of Tom Sawyer", "Mark Twain"),
    ("Great Expectations", "Charles Dickens"), ("Oliver Twist", "Charles Dickens"),
    ("A Tale of Two Cities", "Charles Dickens"), ("Jane Eyre", "Charlotte Bronte"),
    ("Wuthering Heights", "Emily Bronte"), ("The Catcher in the Rye", "J.D. Salinger"),
    ("Brave New World", "Aldous Huxley"), ("Fahrenheit 451", "Ray Bradbury"),
    ("The Hobbit", "J.R.R. Tolkien"), ("The Lord of the Rings", "J.R.R. Tolkien"),
    ("Harry Potter and the Philosopher's Stone", "J.K. Rowling"),
    ("The Lion, the Witch and the Wardrobe", "C.S. Lewis"),
    ("Don Quixote", "Miguel de Cervantes"), ("The Odyssey", "Homer"),
    ("Hamlet", "William Shakespeare"), ("Romeo and Juliet", "William Shakespeare"),
    ("Macbeth", "William Shakespeare"), ("Les Miserables", "Victor Hugo"),
    ("The Hunchback of Notre-Dame", "Victor Hugo"), ("Anna Karenina", "Leo Tolstoy"),
    ("The Picture of Dorian Gray", "Oscar Wilde"),
    ("The Adventures of Sherlock Holmes", "Arthur Conan Doyle"),
    ("Alice's Adventures in Wonderland", "Lewis Carroll"),
    ("The Old Man and the Sea", "Ernest Hemingway"),
    ("One Hundred Years of Solitude", "Gabriel Garcia Marquez"),
    ("The Little Prince", "Antoine de Saint-Exupery"),
    ("Robinson Crusoe", "Daniel Defoe"), ("Gulliver's Travels", "Jonathan Swift"),
]

# --- domain: chemical -- formula (accepted_answers may list variants) ------
CHEMICALS = [
    ("water", ["H2O"]), ("table salt (sodium chloride)", ["NaCl"]),
    ("carbon dioxide", ["CO2"]), ("oxygen gas", ["O2"]), ("nitrogen gas", ["N2"]),
    ("hydrogen gas", ["H2"]), ("methane", ["CH4"]), ("ammonia", ["NH3"]),
    ("glucose", ["C6H12O6"]), ("ethanol", ["C2H6O", "C2H5OH"]),
    ("sulfuric acid", ["H2SO4"]), ("hydrochloric acid", ["HCl"]),
    ("sodium hydroxide", ["NaOH"]), ("calcium carbonate", ["CaCO3"]),
    ("carbon monoxide", ["CO"]), ("ozone", ["O3"]),
    ("hydrogen peroxide", ["H2O2"]), ("nitrous oxide", ["N2O"]),
    ("sulfur dioxide", ["SO2"]), ("potassium chloride", ["KCl"]),
    ("baking soda (sodium bicarbonate)", ["NaHCO3"]),
    ("acetic acid", ["CH3COOH", "C2H4O2"]),
    ("silicon dioxide (quartz)", ["SiO2"]), ("quicklime (calcium oxide)", ["CaO"]),
    ("magnesium oxide", ["MgO"]), ("nitric acid", ["HNO3"]),
    ("phosphoric acid", ["H3PO4"]), ("potassium hydroxide", ["KOH"]),
    ("copper sulfate", ["CuSO4"]), ("iron oxide (rust)", ["Fe2O3"]),
    ("zinc oxide", ["ZnO"]), ("calcium hydroxide", ["Ca(OH)2"]),
    ("washing soda (sodium carbonate)", ["Na2CO3"]),
    ("aluminum oxide", ["Al2O3"]), ("hydrogen sulfide", ["H2S"]),
    ("chlorine gas", ["Cl2"]), ("benzene", ["C6H6"]), ("propane", ["C3H8"]),
    ("butane", ["C4H10"]), ("table sugar (sucrose)", ["C12H22O11"]),
]

# --- domain: location -- capital city ---------------------------------------
CAPITALS = [
    ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"),
    ("Spain", "Madrid"), ("the United Kingdom", "London"), ("Japan", "Tokyo"),
    ("China", "Beijing"), ("Russia", "Moscow"), ("Canada", "Ottawa"),
    ("Australia", "Canberra"), ("Brazil", "Brasilia"), ("Egypt", "Cairo"),
    ("India", "New Delhi"), ("Mexico", "Mexico City"),
    ("Argentina", "Buenos Aires"), ("South Korea", "Seoul"), ("Turkey", "Ankara"),
    ("Greece", "Athens"), ("Portugal", "Lisbon"), ("the Netherlands", "Amsterdam"),
    ("Switzerland", "Bern"), ("Sweden", "Stockholm"), ("Norway", "Oslo"),
    ("Denmark", "Copenhagen"), ("Finland", "Helsinki"), ("Poland", "Warsaw"),
    ("Austria", "Vienna"), ("Belgium", "Brussels"), ("Ireland", "Dublin"),
    ("Thailand", "Bangkok"), ("Indonesia", "Jakarta"), ("Vietnam", "Hanoi"),
    ("the Philippines", "Manila"), ("Kenya", "Nairobi"), ("Nigeria", "Abuja"),
    ("Saudi Arabia", "Riyadh"), ("New Zealand", "Wellington"), ("Chile", "Santiago"),
    ("Peru", "Lima"), ("Colombia", "Bogota"),
]

DOMAIN_SPECS = {
    "movie": {
        "items": MOVIES,
        "question": 'In what year was the film "{q}" released?',
        "answers": lambda a: [a],
    },
    "person": {
        "items": PERSONS,
        "question": "In what year was {q} born?",
        "answers": lambda a: [a],
    },
    "paper_book": {
        "items": BOOKS,
        "question": 'Who wrote the novel "{q}"?',
        "answers": lambda a: [a],
    },
    "chemical": {
        "items": CHEMICALS,
        "question": "What is the chemical formula of {q}?",
        "answers": lambda a: list(a),
    },
    "location": {
        "items": CAPITALS,
        "question": "What is the capital of {q}?",
        "answers": lambda a: [a],
    },
}


def build():
    prompts = []
    reference_items = {}
    for domain, spec in DOMAIN_SPECS.items():
        items = spec["items"]
        assert len(items) == 40, f"{domain}: expected 40 items, got {len(items)}"
        for i, (subject, answer) in enumerate(items):
            key = f"type_b_{domain}_{i + 1:03d}"
            prompts.append(
                {
                    "prompt_id": key,
                    "prompt_type": "type_b",
                    "template": domain,
                    "text": spec["question"].format(q=subject),
                    "reference_key": key,
                }
            )
            reference_items[key] = {
                "accepted_answers": spec["answers"](answer),
                "subject": subject,
            }
    return prompts, reference_items


def main():
    prompts, reference_items = build()
    assert len(prompts) == 200, f"expected 200 Type B prompts, got {len(prompts)}"

    OUT_PROMPTS_PATH.write_text(
        json.dumps(prompts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    reference = {
        "_notice": (
            "Facts authored from general knowledge (src/generate_type_b_prompts.py), "
            "chosen for being extremely well-established/canonical to minimize error "
            "risk -- NOT yet cross-checked item-by-item against a live source. Plan "
            "Appendix B item 2 (snapshot date + numeric-tolerance rule) is still open; "
            "do not treat this as frozen-quality until a verification pass is done."
        ),
        "snapshot_date": None,
        "verification_status": "unverified_pending_appendix_b_item_2",
        "items": reference_items,
    }
    OUT_REFERENCE_PATH.write_text(
        json.dumps(reference, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"wrote {len(prompts)} Type B prompts to {OUT_PROMPTS_PATH}")
    print(f"wrote reference set ({len(reference_items)} items) to {OUT_REFERENCE_PATH}")


if __name__ == "__main__":
    main()
