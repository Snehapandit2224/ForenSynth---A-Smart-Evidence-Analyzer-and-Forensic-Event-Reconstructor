"""
ER: identical names far apart in time (SAME_ALIAS_MAX_GAP_SEC).

The same name in one case is the same person over a much longer span than
two different labels are linked over - an email at 10:05 and the same
person's FIR statement at 11:25 used to stay two people under the 1-hour
limit meant for different labels.

    python -m pytest pipeline/test_er_same_name.py
    python pipeline/test_er_same_name.py        # without pytest
"""
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

_root = Path(__file__).parent.parent
for _p in [str(_root / "agents"), str(_root / "memory"), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
logging.disable(logging.WARNING)

from entity_resolution import resolve_entities  # noqa: E402

BASE = datetime(2025, 3, 14, 10, 0, 0)


def _obs(obs_id, entity, minutes, modality="text", **extra):
    ts = BASE + timedelta(minutes=minutes)
    o = {"obs_id": obs_id, "entity": entity, "role": "unknown", "modality": modality,
         "source": f"{modality}_{obs_id}", "location": "MG Road ATM", "content": "statement",
         "timestamp": ts.isoformat(), "time_offset": int(minutes * 60), "confidence": 0.8}
    o.update(extra)
    return o


def _entities(observations):
    result = resolve_entities({"case_id": "T", "fir": {}, "observations": observations}, llm_enabled=False)
    return sorted(sorted(e["sources"]) for e in result["canonical_entities"])


def test_same_name_80_minutes_apart_is_one_person():
    # Kavya's email at 10:05 and her FIR statement at 11:25
    assert _entities([_obs("O1", "Kavya Rao", 5), _obs("O2", "Kavya Rao", 85)]) == [["O1", "O2"]]


def test_same_name_chain_links_up():
    # each sighting within the gap of the previous one
    assert _entities([_obs("O1", "Kavya Rao", 0), _obs("O2", "Kavya Rao", 50),
                      _obs("O3", "Kavya Rao", 100)]) == [["O1", "O2", "O3"]]


def test_same_name_beyond_seven_days_stays_apart():
    assert len(_entities([_obs("O1", "Kavya Rao", 0), _obs("O2", "Kavya Rao", 8 * 24 * 60)])) == 2


def test_full_name_two_days_apart_is_one_person():
    assert _entities([_obs("O1", "Kavya Rao", 0), _obs("O2", "Kavya Rao", 2 * 24 * 60)]) == [["O1", "O2"]]


def test_single_first_name_two_days_apart_needs_a_second_signal():
    assert len(_entities([_obs("O1", "Ramesh", 0), _obs("O2", "Ramesh", 2 * 24 * 60)])) == 2
    same_phone = [_obs("O1", "Ramesh", 0, contact_ids=["+91 98450 12345"]),
                  _obs("O2", "Ramesh", 2 * 24 * 60, contact_ids=["098450-12345"])]
    assert _entities(same_phone) == [["O1", "O2"]]


def test_stated_name_links_a_speaker_to_the_named_person():
    obs = [_obs("O1", "phone_record_SPEAKER_00", 0, modality="audio", self_identified_as="Kavya Rao"),
           _obs("O2", "Kavya Rao", 30)]
    assert _entities(obs) == [["O1", "O2"]]


def test_same_phone_links_a_sender_to_a_caller():
    obs = [_obs("O1", "report_20", 0, contact_ids=["+91 98450 12345"]),
           _obs("O2", "interview_SPEAKER_01", 40, modality="audio", contact_ids=["98450 12345"])]
    assert _entities(obs) == [["O1", "O2"]]


def test_two_labels_on_one_camera_at_once_are_two_people():
    obs = [_obs("O1", "Person_14", 0, modality="video", source="camera_1", appearance="black hoodie, red cap"),
           _obs("O2", "Person_15", 0, modality="video", source="camera_1", appearance="black hoodie, red cap")]
    assert len(_entities(obs)) == 2


def test_undiarized_shared_label_is_not_merged_by_name():
    # one label for a whole recording covers every speaker on it
    obs = [_obs("O1", "call_Speaker_Unknown", 0, modality="audio", extraction_status="undiarized"),
           _obs("O2", "call_Speaker_Unknown", 2, modality="audio", extraction_status="undiarized")]
    assert len(_entities(obs)) == 2


def test_different_labels_80_minutes_apart_stay_apart():
    # the longer gap is only for identical names, never for different labels
    assert len(_entities([_obs("O1", "Person_14", 5, modality="video"),
                          _obs("O2", "Speaker_C", 85, modality="audio")])) == 2


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
