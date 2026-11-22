"""
ER: voice-print group support (GROUP_SUPPORT_* in entity_resolution.py).

A recording whose print falls just under the pairwise bar joins a group of
recordings already merged on voice prints when it agrees with the whole group
and with no other voice. Built on case 9's numbers: the intercepted call
scored 0.79 / 0.74 / 0.70 against the suspect's three merged calls (bar 0.80
on 2 s of speech) and 0.37 or less against the witness.

    python -m pytest pipeline/test_er_voice_group.py
    python pipeline/test_er_voice_group.py        # without pytest
"""
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

_root = Path(__file__).parent.parent
for _p in [str(_root / "agents"), str(_root / "memory"), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
logging.disable(logging.WARNING)

from entity_resolution import resolve_entities  # noqa: E402

BASE = datetime(2024, 1, 15, 10, 0, 0)


def _prints(gram):
    """Unit vectors whose pairwise cosine similarities are `gram`."""
    g = np.array(gram, dtype=float)
    w, v = np.linalg.eigh(g)
    assert w.min() > -1e-9, "similarities not realisable"
    return (v * np.sqrt(np.clip(w, 0, None))).tolist()


def _entities(gram, sources=None, speech=2.0, must_not_merge=None):
    vecs = _prints(gram)
    obs = []
    for i, vec in enumerate(vecs):
        src = (sources or [f"rec{j}" for j in range(len(vecs))])[i]
        obs.append({"obs_id": f"O{i + 1}", "entity": f"{src}_SPEAKER_{i:02d}", "role": "unknown",
                    "modality": "audio", "source": src, "location": "ATM", "content": "statement",
                    "timestamp": (BASE + timedelta(seconds=30 * i)).isoformat(), "time_offset": 30 * i,
                    "confidence": 0.8, "identity_embedding": vec, "voice_print_speech_sec": speech})
    case = {"case_id": "T", "fir": {}, "observations": obs}
    result = resolve_entities(case, llm_enabled=False,
                              human_constraints={"must_not_merge": must_not_merge or []})
    return sorted(sorted(e["sources"]) for e in result["canonical_entities"]), result


# A, B, C: the suspect's three merged calls; D: the intercepted call; E: the witness
CASE9 = [
    [1.00, 0.81, 0.74, 0.70, 0.32],
    [0.81, 1.00, 0.81, 0.74, 0.28],
    [0.74, 0.81, 1.00, 0.79, 0.40],
    [0.70, 0.74, 0.79, 1.00, 0.37],
    [0.32, 0.28, 0.40, 0.37, 1.00],
]


def test_call_agreeing_with_whole_group_joins_it():
    groups, result = _entities(CASE9)
    assert groups == [["O1", "O2", "O3", "O4"], ["O5"]]
    joins = [l for l in result["identity_links"] if l["decision"] == "merged_group_support"]
    assert len(joins) == 1 and joins[0]["alias_1"].startswith("rec3")


def test_without_group_support_pairwise_bar_alone_leaves_it_out():
    # D against a single print at 0.79 - no group to agree with
    groups, _ = _entities([[1.0, 0.79], [0.79, 1.0]])
    assert groups == [["O1"], ["O2"]]


def test_close_to_a_voice_outside_the_group_stays_apart():
    # D is also 0.66 to the witness: not clearly the group's voice
    gram = [row[:] for row in CASE9]
    gram[3][4] = gram[4][3] = 0.66
    groups, _ = _entities(gram)
    assert ["O4"] in groups


def test_weak_against_one_member_stays_apart():
    gram = [row[:] for row in CASE9]
    gram[3][0] = gram[0][3] = 0.55
    groups, _ = _entities(gram)
    assert ["O4"] in groups


def test_same_recording_as_a_member_stays_apart():
    # D is a second speaker in A's recording: provably a different person
    groups, _ = _entities(CASE9, sources=["rec0", "rec1", "rec2", "rec0", "rec4"])
    assert ["O4"] in groups


def test_must_not_merge_blocks_the_join():
    groups, _ = _entities(CASE9, must_not_merge=[["rec3_SPEAKER_03", "rec0_SPEAKER_00"]])
    assert ["O4"] in groups


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
