"""
How an event is known and what backs it up: observed / reported / FIR claim,
corroboration across a moment, the confidence cap for an uncorroborated
report, the Critique notes (uncorroborated report, result reported before the
act, missing steps), and the FIR's identity hand-off to its observations.
Synthetic inputs - no real case.

    python -m pytest pipeline/test_reported_evidence.py
    python pipeline/test_reported_evidence.py        # without pytest
"""
import logging
import sys
from pathlib import Path

_root = Path(__file__).parent.parent
for _p in [str(_root / "agents"), str(_root / "agents" / "field_agents" / "text agent"), str(_root / "memory"), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
logging.disable(logging.WARNING)

import critique_agent as C  # noqa: E402
from fir_agent import attach_fir_identities  # noqa: E402
from shared import parse_timestamp  # noqa: E402
from timeline_agent import UNCORROBORATED_CONFIDENCE_CAP, TimelineAgent, TimelineEvent, _basis  # noqa: E402


# ── basis ─────────────────────────────────────────────────────────────────────
def test_basis_of_an_event():
    assert _basis([{"fir": True}]) == "claim"
    assert _basis([{"reported": True}]) == "reported"
    assert _basis([{"reported": True}, {"fir": True}]) == "reported"
    assert _basis([{"reported": True}, {}]) == "observed"


def _event(eid, entity, ts, tags, basis, sources, links=()):
    return TimelineEvent(event_id=eid, obs_ids=[eid], timestamp=ts, ts_epoch=parse_timestamp(ts), location="atm",
                         entity_id=entity, primary_alias=entity, aliases=[entity], modality="text", content="",
                         confidence=0.86, action_tags=tags, basis=basis, sources=list(sources),
                         same_moment_as=list(links))


# ── corroboration + confidence cap ───────────────────────────────────────────
def test_a_lone_report_is_uncorroborated_and_capped():
    e = _event("A", "email_person", "2024-01-15T10:05:54", ["EXIT"], "reported", ["O8#1"])
    TimelineAgent._assess_corroboration([e])
    assert e.corroboration == 0 and e.confidence == UNCORROBORATED_CONFIDENCE_CAP


def test_a_report_linked_to_cctv_is_corroborated_and_keeps_its_confidence():
    rep = _event("A", "email_person", "2024-01-15T10:05:54", ["EXIT"], "reported", ["O8#1"], ["B"])
    cam = _event("B", "Person_94", "2024-01-15T10:05:50", ["EXIT"], "observed", ["video:camera_3"], ["A"])
    TimelineAgent._assess_corroboration([rep, cam])
    assert rep.corroboration == 1 and rep.corroborated_by == ["video:camera_3"] and rep.confidence == 0.86


def test_the_fir_never_corroborates_itself():
    claim = _event("A", "fir_man", "2024-01-15T10:03:00", ["ASSAULT"], "claim", [], [])
    TimelineAgent._assess_corroboration([claim])
    assert claim.corroboration == 0 and claim.confidence == UNCORROBORATED_CONFIDENCE_CAP


def test_an_observed_event_is_never_capped():
    cam = _event("B", "Person_94", "2024-01-15T10:05:50", ["EXIT"], "observed", ["video:camera_3"])
    TimelineAgent._assess_corroboration([cam])
    assert cam.confidence == 0.86


# ── Critique ─────────────────────────────────────────────────────────────────
def test_uncorroborated_report_is_named():
    tl = {"events": [
        {"event_id": "A", "entity_id": "e7", "primary_alias": "unknown_people_1", "action_tags": ["EXIT"],
         "basis": "reported", "corroboration": 0, "timestamp": "2024-01-15T10:05:54"},
        {"event_id": "B", "entity_id": "e1", "primary_alias": "Person_94", "action_tags": ["EXIT"],
         "basis": "observed", "corroboration": 0, "timestamp": "2024-01-15T10:05:50"},
        {"event_id": "C", "entity_id": "e2", "primary_alias": "Kamala", "action_tags": ["REPORT"],
         "basis": "reported", "corroboration": 0, "timestamp": "2024-01-15T10:07:10"}]}
    notes = C._check_uncorroborated(tl)
    assert [n.affected_events for n in notes] == [["A"]] and "appears only in reports" in notes[0].detail


def _o(oid, entity, ts, action, **extra):
    o = {"obs_id": oid, "entity": entity, "modality": extra.pop("modality", "text"), "action": action,
         "timestamp": ts, "time_basis": "explicit", "content": "", "evidence": extra.pop("evidence", "")}
    o.update(extra)
    return o


def test_result_reported_before_the_first_entry_is_noted():
    obs = [_o("R", "unknown_person", "2024-01-15T10:00:00", "TAMPER", tags=["result"], reported_by="staff",
              evidence="something has been attached to the card slot"),
           _o("E", "Person_94", "2024-01-15T10:04:08", "ENTER", modality="video", time_source="verified_clock")]
    notes = C._check_result_before_act(obs)
    assert len(notes) == 1 and "larger than both records' timing uncertainty" in notes[0].detail


def test_a_few_seconds_early_is_order_unclear():
    obs = [_o("R", "unknown_person", "2024-01-15T10:04:00", "TAMPER", tags=["result"], reported_by="staff"),
           _o("E", "Person_94", "2024-01-15T10:04:08", "ENTER", modality="video", time_source="verified_clock")]
    notes = C._check_result_before_act(obs)
    assert len(notes) == 1 and "order is unclear" in notes[0].detail


def test_result_after_an_entry_is_fine():
    obs = [_o("E", "Person_94", "2024-01-15T10:04:08", "ENTER", modality="video", time_source="verified_clock"),
           _o("R", "unknown_person", "2024-01-15T10:06:00", "TAMPER", tags=["result"], reported_by="staff")]
    assert not C._check_result_before_act(obs)


def test_missing_step_names_every_fitting_act_and_points_to_other_sources():
    seq = [{"event_id": "A", "entity_id": "e1", "action_tags": ["ENTER"], "ts_epoch": 100.0},
           {"event_id": "B", "entity_id": "e1", "action_tags": ["EXIT"], "ts_epoch": 200.0}]
    plain = C._check_missing_links({"e1": seq}, seq)
    assert plain and "STEAL missing" not in plain[0].detail and "no evidence shows which" in plain[0].detail
    log = {"event_id": "L", "entity_id": "atm_log", "primary_alias": "ATM log", "action_tags": ["TAMPER"], "ts_epoch": 150.0}
    pointed = C._check_missing_links({"e1": seq}, seq + [log])
    assert "another source shows TAMPER" in pointed[0].detail


# ── FIR identity hand-off ────────────────────────────────────────────────────
RECORD = {"claims": {
    "accused": [{"name": None, "description": "young man, pink cap, face mask, grey hooded sweatshirt", "evidence": "x"}],
    "witnesses": [{"name": "Kamala Iyer", "description": "grey hair, mustard kurta, maroon shawl",
                   "evidence": "Smt. Kamala Iyer, aged about 60, phone 98450 62137, was waiting outside"}],
    "victims": []}, "anchors": {"informant": {"name": "Ramesh Kulkarni", "phone": "98450 XXXXX"}}}


def test_single_unnamed_accused_description_reaches_its_observations():
    obs = [{"entity": "unknown_man_1@FIR", "role_hint": "suspect", "appearance": ""},
           {"entity": "unknown_bystander_1@FIR", "role_hint": "witness", "appearance": ""}]
    out = attach_fir_identities(obs, RECORD)
    assert out[0]["appearance"].startswith("young man, pink cap") and not out[1]["appearance"]


def test_named_witness_gets_description_and_phone():
    out = attach_fir_identities([{"entity": "Kamala Iyer", "role_hint": "witness", "appearance": ""}], RECORD)
    assert "mustard kurta" in out[0]["appearance"] and out[0]["contact_ids"] == ["98450 62137"]


def test_two_unnamed_accused_are_not_guessed():
    rec = {"claims": {"accused": [{"name": None, "description": "red cap"}, {"name": None, "description": "blue cap"}],
                      "witnesses": [], "victims": []}, "anchors": {}}
    out = attach_fir_identities([{"entity": "unknown_person_1@FIR", "role_hint": "suspect", "appearance": ""}], rec)
    assert not out[0]["appearance"]


def test_a_masked_phone_is_not_used():
    out = attach_fir_identities([{"entity": "Ramesh Kulkarni", "role_hint": "victim", "appearance": ""}], RECORD)
    assert "contact_ids" not in out[0]


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
