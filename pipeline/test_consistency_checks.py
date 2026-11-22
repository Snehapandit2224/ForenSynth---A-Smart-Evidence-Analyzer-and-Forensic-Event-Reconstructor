"""
Critique: evidence-consistency checks (act order, claims about the scene),
the integrity flag, and the denial guard - on small synthetic cases, not on
any real case, so they test the rules rather than one case's numbers.

    python -m pytest pipeline/test_consistency_checks.py
    python pipeline/test_consistency_checks.py        # without pytest
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

import critique_agent as C  # noqa: E402

BASE = datetime(2025, 6, 2, 14, 0, 0)


def _o(obs_id, entity, sec, modality="video", action=None, source=None, **extra):
    o = {"obs_id": obs_id, "entity": entity, "modality": modality, "source": source or f"{modality}_{obs_id}",
         "timestamp": (BASE + timedelta(seconds=sec)).isoformat(), "content": extra.pop("content", ""),
         "time_source": "verified_clock" if modality == "video" else None}
    if modality != "video":
        o["time_basis"] = "explicit"
    if action:
        o["action"] = action
    o.update(extra)
    return o


def _er(*groups):
    return {"canonical_entities": [{"entity_id": f"entity_{i}", "sources": list(g)} for i, g in enumerate(groups, 1)]}


def _checks(issues, name):
    return [i for i in issues if i.check == name]


# ── act order ────────────────────────────────────────────────────────────────
def test_same_exit_on_two_cameras_far_apart_is_flagged():
    obs = [_o("A", "P1", 0, action="ENTER", source="cam_1"), _o("B", "P1", 60, action="EXIT", source="cam_1"),
           _o("C", "P1", 95, action="EXIT", source="cam_2")]
    assert len(_checks(C._check_act_sequence(_er("ABC"), obs), "act_sequence_inconsistent")) == 1


def test_exit_reentry_exit_is_fine():
    obs = [_o("A", "P1", 0, action="EXIT", source="cam_1"), _o("B", "P1", 40, action="ENTER", source="cam_2"),
           _o("C", "P1", 90, action="EXIT", source="cam_2")]
    assert not C._check_act_sequence(_er("ABC"), obs)


def test_same_exit_seconds_apart_is_one_act():
    obs = [_o("A", "P1", 0, action="EXIT", source="cam_1"), _o("B", "P1", 6, action="EXIT", source="cam_2")]
    assert not C._check_act_sequence(_er("AB"), obs)


def test_act_inside_after_leaving_is_flagged():
    obs = [_o("A", "R", 0, "text", "ENTER"), _o("B", "R", 30, "text", "EXIT"), _o("C", "R", 31, "text", "TAMPER")]
    assert len(_checks(C._check_act_sequence(_er("ABC"), obs), "act_sequence_inconsistent")) == 1


def test_untrusted_times_are_not_used():
    obs = [_o("A", "R", 0, "text", "EXIT", time_basis="unknown"), _o("B", "R", 40, "text", "EXIT", source="other")]
    assert not C._check_act_sequence(_er("AB"), obs)


# ── claims about the scene ───────────────────────────────────────────────────
def test_no_one_saw_against_an_onlooker():
    obs = [_o("A", "R", 0, "text", "TAMPER", evidence="Done. No one saw."),
           _o("B", "W", 50, "video", "OBSERVE", content="Bystander watches the booth")]
    assert len(_checks(C._check_claims_vs_evidence(_er("A", "B"), obs), "claim_contradicted")) == 1


def test_no_one_saw_with_nobody_watching_is_fine():
    obs = [_o("A", "R", 0, "text", "TAMPER", evidence="Done. No one saw."), _o("B", "R2", 50, "video", "EXIT")]
    assert not C._check_claims_vs_evidence(_er("A", "B"), obs)


def test_headcount_two_against_one_on_camera():
    obs = [_o("A", "E", 0, "text", "EXIT", evidence="Two individuals left together quickly."),
           _o("B", "P1", 20, "video", "EXIT")]
    assert len(_checks(C._check_claims_vs_evidence(_er("A", "B"), obs), "headcount_mismatch")) == 1


def test_headcount_two_matches_a_group_on_camera():
    obs = [_o("A", "E", 0, "text", "EXIT", evidence="Two individuals left together quickly."),
           _o("B", "P1", 20, "video", "FLEE", co_entities=["P2"])]
    assert not C._check_claims_vs_evidence(_er("A", "B"), obs)


def test_only_him_against_two_on_camera():
    obs = [_o("A", "W", 0, "audio", "EXIT", evidence="It was only him, nobody else with him, he left fast"),
           _o("B", "P1", 30, "video", "EXIT", co_entities=["P2"])]
    assert len(_checks(C._check_claims_vs_evidence(_er("A", "B"), obs), "headcount_mismatch")) == 1


# ── integrity ────────────────────────────────────────────────────────────────
def test_corrupted_record_is_flagged():
    obs = [_o("A", "S", 0, "text", content="[CORRUPTED] - File hash mismatch; content integrity unverified."),
           _o("B", "S", 5, "text", content="Partial view of the door, then I left")]
    flagged = C._check_evidence_integrity(obs)
    assert [i.check for i in flagged] == ["evidence_integrity"] and "A" in flagged[0].detail


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
