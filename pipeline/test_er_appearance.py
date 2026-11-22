"""
ER: reading appearance however it is worded - synonyms, colour families,
gender, hair - and only ruling a merge out on confident details. Phrasings
are generic, not taken from any one case.

    python -m pytest pipeline/test_er_appearance.py
    python pipeline/test_er_appearance.py        # without pytest
"""
import logging
import sys
from pathlib import Path

_root = Path(__file__).parent.parent
for _p in [str(_root / "agents"), str(_root / "memory"), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
logging.disable(logging.WARNING)

from entity_resolution import EntityResolutionPipeline as ER  # noqa: E402


def _cmp(a, b):
    return ER._compare_appearance(ER._parse_appearance(a), ER._parse_appearance(b))


# the same person, described differently
SAME = [
    ("man in a grey hoodie and black joggers, white sneakers", "male, charcoal sweatshirt, black trousers, white trainers"),
    ("woman, mustard tunic, maroon shawl", "lady in a yellow kurta with a burgundy stole"),
    ("red baseball cap, navy jacket", "Individual wearing a red cap and a dark blue jacket"),
    ("tall man, black hoodie", "black sweatshirt, tall"),
    ("older woman with grey hair, green bag", "elderly lady, gray hair, carrying a green handbag"),
    ("grey hoodie, black pants", "dark jacket, black jeans"),            # a witness's looser wording
]
# different people
DIFFERENT = [
    ("man in a black hoodie", "woman in a black hoodie"),
    ("red cap, black jacket", "blue cap, black jacket"),
    ("grey hair, white kurta", "black hair, white kurta"),
    ("tall man, green shirt", "short man, green shirt"),
]


def test_paraphrases_match_without_conflict():
    for a, b in SAME:
        score, conflict = _cmp(a, b)
        assert not conflict and score > 0, (a, b, score, conflict)


def test_different_people_conflict():
    for a, b in DIFFERENT:
        assert _cmp(a, b)[1], (a, b)


def test_hedged_gender_is_not_a_veto():
    assert not _cmp("possibly a woman, black hoodie", "man in a black hoodie")[1]


def test_hair_colour_does_not_leak_onto_clothes():
    p = ER._parse_appearance("older woman, grey hair, mustard kurta")
    assert p["hair"] == "grey" and p["top"] == ("yellow", "kurta") and p["gender"] == "female"


def test_unknown_words_are_ignored_not_guessed():
    p = ER._parse_appearance("person in a fluorescent poncho")
    assert not any(k in p for k in ("top", "head", "bottom", "feet", "gender"))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {t.__name__}: {exc}")
    sys.exit(1 if failed else 0)
