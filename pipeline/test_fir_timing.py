"""
FIR claims: timed by the incident window (not the filing time), the report to
the police as its own observation, ASSAULT as an act, and Timeline links from
a window-timed claim to the evidence moment with the same act. Synthetic
inputs - no real case.

    python -m pytest pipeline/test_fir_timing.py
    python pipeline/test_fir_timing.py        # without pytest
"""
import logging
import sys
from pathlib import Path

_root = Path(__file__).parent.parent
for _p in [str(_root / "agents"), str(_root / "agents" / "field_agents" / "text agent"), str(_root / "memory"), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
logging.disable(logging.WARNING)

from fir_agent import place_fir_claims  # noqa: E402
from shared import extract_action_tags  # noqa: E402
from timeline_agent import TimelineAgent, TimelineEvent  # noqa: E402
from shared import parse_timestamp  # noqa: E402

FIR_TEXT = ("3. (b) Information received at P.S.: Date 15/01/2024, Time 11:20 hrs\n"
            "I saw one man pacing near the ATM. He went up to the card reader. He assaulted me.")
SUMMARY = {"incident_datetime": "2024-01-15T10:00:00", "incident_datetime_end": "2024-01-15T10:06:00",
           "informant": "Ramesh", "date_of_report": "2024-01-15T11:40:00"}


def _obs(i, entity, action, ts="2024-01-15T11:40:00"):
    return {"entity": entity, "action": action, "timestamp": ts, "time_basis": "document_date",
            "content": f"{entity} {action}", "evidence": "", "_order": i}


def test_scene_claims_are_placed_in_the_window_in_order():
    obs = [_obs(1, "man", "LOITER"), _obs(2, "man", "TAMPER"), _obs(3, "man", "ASSAULT")]
    out = place_fir_claims(obs, SUMMARY, FIR_TEXT)
    times = [o["timestamp"] for o in out if o["entity"] == "man"]
    assert times == sorted(times) and all("2024-01-15T10:00:00" < t < "2024-01-15T10:06:00" for t in times), times
    assert all(o["time_basis"] == "window" and o["claim_window"] == ["2024-01-15T10:00:00", "2024-01-15T10:06:00"]
               for o in out if o["entity"] == "man")


def test_a_time_stated_inside_the_window_is_kept():
    out = place_fir_claims([_obs(1, "man", "ENTER", "2024-01-15T10:04:00")], SUMMARY, FIR_TEXT)
    assert out[0]["timestamp"] == "2024-01-15T10:04:00" and out[0]["time_basis"] == "document_date"


def test_report_to_police_is_its_own_observation_at_information_received():
    out = place_fir_claims([_obs(1, "man", "LOITER")], SUMMARY, FIR_TEXT)
    reports = [o for o in out if o["action"] == "REPORT"]
    assert len(reports) == 1 and reports[0]["entity"] == "Ramesh" and reports[0]["timestamp"] == "2024-01-15T11:20:00"


def test_an_extracted_report_is_retimed_not_duplicated():
    out = place_fir_claims([_obs(1, "Ramesh", "REPORT")], SUMMARY, FIR_TEXT)
    assert [o["timestamp"] for o in out if o["action"] == "REPORT"] == ["2024-01-15T11:20:00"]


def test_after_the_fact_statements_keep_their_time():
    out = place_fir_claims([_obs(1, "bank", "COMMUNICATE")], SUMMARY, FIR_TEXT)
    assert out[0]["timestamp"] == "2024-01-15T11:40:00"


def test_assault_is_an_act():
    assert "ASSAULT" in extract_action_tags("One of them attacked me when I came near the booth")
    assert "STEAL" not in extract_action_tags("He assaulted me when I questioned him")


def _event(eid, entity, ts, tags, window=None, modality="video"):
    t = parse_timestamp(ts)
    w = (parse_timestamp(window[0]), parse_timestamp(window[1])) if window else None
    return TimelineEvent(event_id=eid, obs_ids=[eid], timestamp=ts, ts_epoch=t, location="atm", entity_id=entity,
                         primary_alias=entity, aliases=[entity], modality=modality, content="", confidence=0.7,
                         action_tags=tags, claim_window=w)


WINDOW = ("2024-01-15T10:00:00", "2024-01-15T10:06:00")


def test_window_claim_links_to_the_same_act_inside_the_window():
    claim = _event("F", "fir_man", "2024-01-15T10:03:00", ["TAMPER"], WINDOW, "text")
    tamper = _event("V", "cam_person", "2024-01-15T10:05:10", ["TAMPER"])
    other = _event("X", "cam_person", "2024-01-15T10:04:00", ["ENTER"])
    TimelineAgent._link_same_moment([claim, tamper, other])
    assert claim.same_moment_as == ["V"] and other.same_moment_as == []


def test_window_claim_joins_only_the_nearest_of_two_matching_events():
    claim = _event("F", "fir_man", "2024-01-15T10:04:30", ["EXIT"], WINDOW, "text")
    e1 = _event("A", "cam_person", "2024-01-15T10:05:20", ["EXIT"])
    e2 = _event("B", "cam_person2", "2024-01-15T10:05:50", ["EXIT"])
    TimelineAgent._link_same_moment([claim, e1, e2])
    assert "A" in claim.same_moment_as and "B" not in claim.same_moment_as


def test_nothing_outside_the_window_is_linked():
    claim = _event("F", "fir_man", "2024-01-15T10:03:00", ["TAMPER"], WINDOW, "text")
    late = _event("V", "cam_person", "2024-01-15T10:09:00", ["TAMPER"])
    TimelineAgent._link_same_moment([claim, late])
    assert claim.same_moment_as == []


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
