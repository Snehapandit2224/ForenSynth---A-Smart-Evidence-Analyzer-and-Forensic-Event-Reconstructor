#!/usr/bin/env python3
"""
ForenSynth - Ground Truth Evaluation Layer
Compares a pipeline-reconstructed timeline against the case generator's
ground truth: event recall, precision and F1, action-tag accuracy,
temporal error, entity accuracy, and confidence calibration.

Matching - by PROVENANCE, not by guessing:
  The generator records which ground-truth event every observation came
  from (observations[].event_ref in the full case file), and every pipeline
  event lists the observations it was built from (obs_ids). So each
  pipeline event "is about" the ground-truth event most of its observations
  came from (its majority event).

  For each ground-truth event:
    matched      - some pipeline event's majority event is this one. If
                   several are, the one with the most of its observations
                   is the match; the rest are duplicates (fragments of one
                   moment), which cost precision.
    covered      - no pipeline event is mainly about it, but one includes
                   an observation of it (merged into a neighbouring event by
                   the Timeline's clustering): 0.5 recall credit.
    unobservable - none of its observations survived into the case (the
                   generator's missing-modality noise dropped them all).
                   Nothing could recover it; reported, and also excluded in
                   event_recall_observable.
    missed       - its observations are in the case, but in no event.

  (This used to pair each ground-truth event with the nearest-in-time
  pipeline event of the same person, found by alias with a substring
  fallback - which paired different actions of one person, and under name
  variants matched "Kumar" to two different people. That alias matching is
  kept only as a fallback for ground truth without event_ref.)

Metrics on the matched pairs:
  entity_accuracy     - is the pipeline event's ENTITY (ER's group) really
                        the event's person? Each ER entity is identified
                        with the person most of its observations belong to;
                        a pair is correct if that person is the event's.
                        A wrong merge makes the minority person's events
                        wrong. (This used to be circular: pairs were only
                        formed when the two already resolved to the same
                        person, so it was 100% for every case, including
                        ones where ER fused two people. Wrong SPLITS don't
                        show here - each fragment is still the right person;
                        ER's split/merge rates are in evaluate_er.py.)
  action_tag_accuracy - do the pipeline event's action tags include a tag
                        for the ground-truth action (explicit table below,
                        not substring matching)?
  temporal_accuracy_sec - mean absolute timestamp error in seconds (an
                        ERROR: lower is better; the key keeps its old name).
                        temporal_error_median_sec alongside.
  confidence_calibration - correlation between the pipeline's confidence
                        and being right (action AND entity correct).

  recall    = (matched + 0.5 * covered) / ground-truth events
  precision = matched / pipeline events
  f1        = harmonic mean. PASS is F1 >= 0.75, PARTIAL >= 0.55, else FAIL.
  precision_linked / f1_linked - the same, not counting repeats the timeline
            links to the matched event as one moment (same_moment_as), nor
            events made only of denials. Both versions are reported; the
            verdict uses the raw F1.

Usage:
    python pipeline/evaluate.py --case CASE_ATM_001 \\
        --ground-truth cases/CASE_ATM_001.json \\
        --timeline output/timelines/CASE_ATM_001_timeline_V2.json
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

PASS_THRESHOLD = 0.75
PARTIAL_THRESHOLD = 0.55

# Ground-truth action -> the Timeline action tags that describe it.
# A witness observing an exit is often tagged by what was seen (EXIT).
ACTION_TAGS: Dict[str, Set[str]] = {
    "approach_atm": {"APPROACH"}, "approach_office": {"APPROACH"},
    "enter_atm": {"ENTER"}, "enter_office": {"ENTER"},
    "withdraw_cash": {"WITHDRAW"},
    "exit_atm": {"EXIT"}, "exit_office": {"EXIT"},
    "wait_outside": {"WATCH", "LOITER"}, "loiter_near_atm": {"LOITER", "WATCH"},
    "tamper_atm": {"TAMPER"},
    "flee_scene": {"FLEE"},
    "steal_items": {"STEAL"}, "steal_data": {"STEAL"},
    "navigate_to_target": {"NAVIGATE"},
    "perform_legit_work": {"WORK"},
    "observe_exit": {"OBSERVE", "EXIT"},
    "report_incident": {"REPORT"},
    "initiate_communication": {"COMMUNICATE"}, "respond_to_communication": {"COMMUNICATE"},
    "exchange_information": {"COMMUNICATE"}, "coordinate_activity": {"COMMUNICATE"},
    "confirm_plan": {"CONFIRM", "COMMUNICATE"},
    "intercept_communication": {"INTERCEPT"},
    "flag_suspicious_comm": {"REPORT", "INTERCEPT"},
}


def _parse_ts(ts: str) -> Optional[datetime]:
    """Naive datetime (aware times converted to UTC), or None."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


def _delta(a: Optional[datetime], b: Optional[datetime]) -> Optional[float]:
    return None if a is None or b is None else abs((a - b).total_seconds())


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _action_match(gt_action: str, action_tags: List[str]) -> bool:
    tags = {(t or "").upper() for t in action_tags or []}
    if not gt_action or not tags:
        return False
    expected = ACTION_TAGS.get(gt_action)
    if expected is not None:
        return bool(tags & expected)
    # an action not in the table: its leading verb as a whole tag
    return gt_action.split("_")[0].upper() in tags


# ── Fallback matching (ground truth without observation -> event links) ──────

def _resolve_entity_fuzzy(alias: str, entity_mapping: Dict[str, str]) -> Optional[str]:
    if not alias:
        return None
    if alias in entity_mapping:
        return entity_mapping[alias]
    a = _norm(alias)
    for gt_alias, gt_eid in entity_mapping.items():
        if _norm(gt_alias) == a:
            return gt_eid
    for gt_alias, gt_eid in entity_mapping.items():
        g = _norm(gt_alias)
        if g and (g in a or a in g):
            return gt_eid
    return None


def _match_by_alias_and_time(gt_events, pipeline_events, entity_mapping):
    """Legacy: nearest-in-time pipeline event of the same (alias-resolved)
    person. Only used when the ground truth has no event_ref links."""
    used, pairs, unmatched = set(), [], []
    p_ent = []
    for p in pipeline_events:
        cands = [p.get("primary_alias", "")] + list(p.get("aliases") or [])
        p_ent.append(next((e for e in (_resolve_entity_fuzzy(a, entity_mapping) for a in cands) if e), None))
    for g in sorted(gt_events, key=lambda e: e.get("timestamp") or ""):
        g_ts = _parse_ts(g.get("timestamp", ""))
        best = None
        for i, p in enumerate(pipeline_events):
            if i in used or p_ent[i] != g.get("entity_id"):
                continue
            d = _delta(_parse_ts(p.get("timestamp", "")), g_ts)
            d = float("inf") if d is None else d
            if best is None or d < best[1]:
                best = (i, d)
        if best is None:
            unmatched.append(g.get("event_id", "?"))
            continue
        used.add(best[0])
        pairs.append((g, pipeline_events[best[0]]))
    return pairs, unmatched


# ── Provenance matching ───────────────────────────────────────────────────────

def _match_by_provenance(gt_events, pipeline_events, obs_to_gt_event):
    """(pairs, covered_ids, missed_ids, unobservable_ids, duplicates) where
    duplicates = [(index of a repeat event, index of the matched event)]."""
    observed = set(obs_to_gt_event.values())
    majority, counts = [], []
    for p in pipeline_events:
        c = Counter(obs_to_gt_event[o] for o in p.get("obs_ids") or [] if o in obs_to_gt_event)
        counts.append(c)
        # majority event; ties go to the earliest ground-truth event id order
        majority.append(max(sorted(c), key=lambda e: c[e]) if c else None)

    pairs, covered, missed, unobservable = [], set(), [], []
    duplicates = []
    for g in sorted(gt_events, key=lambda e: e.get("timestamp") or ""):
        gid = g.get("event_id")
        mine = [i for i, m in enumerate(majority) if m == gid]
        if mine:
            g_ts = _parse_ts(g.get("timestamp", ""))
            best = max(mine, key=lambda i: (counts[i][gid],
                                            -(_delta(_parse_ts(pipeline_events[i].get("timestamp", "")), g_ts)
                                              or 0.0)))
            pairs.append((g, pipeline_events[best]))
            duplicates += [(i, best) for i in mine if i != best]
        elif any(gid in c for c in counts):
            covered.add(gid)
        elif gid not in observed:
            unobservable.append(gid)
        else:
            missed.append(gid)
    return pairs, covered, missed, unobservable, duplicates


def _er_entity_identity(pipeline_events, obs_truth) -> Dict[str, str]:
    """ER entity id -> the person most of its observations belong to."""
    votes: Dict[str, Counter] = {}
    for p in pipeline_events:
        c = votes.setdefault(p.get("entity_id", ""), Counter())
        c.update(obs_truth[o] for o in p.get("obs_ids") or [] if obs_truth.get(o))
    return {eid: max(sorted(c), key=lambda e: c[e]) for eid, c in votes.items() if c}


def _correlation(xs: List[float], ys: List[float]) -> Optional[float]:
    """Pearson correlation; None if undefined (constant input or <2 samples)."""
    n = len(xs)
    if n < 2:
        return None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov   = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x ** 0.5 * var_y ** 0.5)


def evaluate_case(
    case_id: str,
    ground_truth_path: str,
    timeline_v3_path: str,
    output_dir: str = "output/evaluation",
) -> dict:
    """
    Compare a pipeline-reconstructed timeline against case-generator ground
    truth. NOTE: despite the parameter name (kept for interface
    compatibility), pass the case's actual FINAL timeline file.

    Returns a metrics dict and also writes it to
    {output_dir}/{case_id}_evaluation.json.
    """
    result: Dict[str, Any] = {
        "case_id": case_id,
        "n_ground_truth_events": None, "n_pipeline_events": None, "n_matched": None,
        "n_covered": None, "n_missed": None, "n_unobservable": None, "n_duplicates": None,
        "n_duplicates_linked": None, "n_denial_events": None,
        "precision_linked": None, "f1_linked": None,
        "event_recall": None, "event_recall_observable": None, "action_tag_accuracy": None,
        "temporal_accuracy_sec": None, "temporal_error_median_sec": None, "entity_accuracy": None,
        "precision": None, "f1_score": None,
        "confidence_calibration": {"correlation": None, "n_samples": 0},
        "matching": None, "result": "ERROR", "status": "failed",
    }
    try:
        gt_data = json.loads(Path(ground_truth_path).read_text(encoding="utf-8"))
        tl_data = json.loads(Path(timeline_v3_path).read_text(encoding="utf-8"))

        gt = gt_data.get("ground_truth", {})
        gt_events = gt.get("events", [])
        entity_mapping = gt.get("entity_mapping", {})
        pipeline_events = tl_data.get("events", [])
        gt_obs = gt_data.get("observations", [])
        obs_to_gt_event = {o["obs_id"]: o["event_ref"] for o in gt_obs if o.get("obs_id") and o.get("event_ref")}
        # each observation's true person: exact when the generator recorded
        # it (realism flags), else through its alias
        obs_entity = gt.get("obs_entity") or {}
        obs_truth = {o["obs_id"]: obs_entity.get(o["obs_id"]) or entity_mapping.get(o.get("entity", ""))
                     for o in gt_obs if o.get("obs_id")}

        if obs_to_gt_event:
            matching = "provenance"
            pairs, covered, missed, unobservable, duplicates = _match_by_provenance(
                gt_events, pipeline_events, obs_to_gt_event)
        else:
            matching = "alias_time_fallback"
            pairs, missed = _match_by_alias_and_time(gt_events, pipeline_events, entity_mapping)
            covered, unobservable, duplicates = set(), [], []
        n_dup = len(duplicates)

        # Repeats the timeline itself presents as one moment (same-moment
        # links, possibly through other events) aren't separate claims; nor
        # is an event made only of denials - a contradicting statement about
        # the act, not another copy of it. precision_linked leaves both out.
        denial_obs = {o["obs_id"] for o in gt_obs if "contradiction" in (o.get("noise_tags") or [])}
        by_id = {p.get("event_id"): i for i, p in enumerate(pipeline_events)}
        parent = list(range(len(pipeline_events)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i, p in enumerate(pipeline_events):
            for other in p.get("same_moment_as") or []:
                if other in by_id:
                    parent[find(i)] = find(by_id[other])
        is_denial = lambda i: bool(pipeline_events[i].get("obs_ids")) and             all(o in denial_obs for o in pipeline_events[i]["obs_ids"])
        n_denial = sum(1 for i, _ in duplicates if is_denial(i))
        n_linked = sum(1 for i, best in duplicates if not is_denial(i) and find(i) == find(best))

        identity = _er_entity_identity(pipeline_events, obs_truth)
        rows = []
        for g, p in pairs:
            g_ts, p_ts = _parse_ts(g.get("timestamp", "")), _parse_ts(p.get("timestamp", ""))
            rows.append({
                "gt_event_id":          g.get("event_id", "?"),
                "pipeline_event_id":    p.get("event_id", "?"),
                "gt_action":            g.get("action", ""),
                "pipeline_action_tags": p.get("action_tags", []),
                "action_match":         _action_match(g.get("action", ""), p.get("action_tags", [])),
                "gt_entity_id":         g.get("entity_id", ""),
                "pipeline_entity":      p.get("entity_id", ""),
                "pipeline_entity_is":   identity.get(p.get("entity_id", "")),
                "entity_match":         identity.get(p.get("entity_id", "")) == g.get("entity_id"),
                "temporal_delta_sec":   _delta(p_ts, g_ts),
                "pipeline_confidence":  p.get("confidence"),
            })

        n_gt, n_pipeline, n_matched = len(gt_events), len(pipeline_events), len(rows)
        n_covered = len(covered)
        n_observable = n_gt - len(unobservable)
        recall     = ((n_matched + 0.5 * n_covered) / n_gt) if n_gt else 0.0
        recall_obs = ((n_matched + 0.5 * n_covered) / n_observable) if n_observable else 0.0
        precision  = (n_matched / n_pipeline) if n_pipeline else 0.0
        f1         = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        n_claims   = n_pipeline - n_linked - n_denial
        precision_linked = (n_matched / n_claims) if n_claims else 0.0
        f1_linked  = (2 * precision_linked * recall / (precision_linked + recall))             if (precision_linked + recall) else 0.0

        action_acc = (sum(r["action_match"] for r in rows) / n_matched) if n_matched else 0.0
        entity_acc = (sum(r["entity_match"] for r in rows) / n_matched) if n_matched else 0.0

        deltas = [r["temporal_delta_sec"] for r in rows if r["temporal_delta_sec"] is not None]

        conf_pairs = [(r["pipeline_confidence"], 1.0 if (r["action_match"] and r["entity_match"]) else 0.0)
                      for r in rows if r["pipeline_confidence"] is not None]
        confidence_corr = _correlation([c for c, _ in conf_pairs], [c for _, c in conf_pairs])

        verdict = "PASS" if f1 >= PASS_THRESHOLD else "PARTIAL" if f1 >= PARTIAL_THRESHOLD else "FAIL"

        result.update({
            "n_ground_truth_events":     n_gt,
            "n_pipeline_events":         n_pipeline,
            "n_matched":                 n_matched,
            "n_covered":                 n_covered,
            "n_missed":                  len(missed),
            "n_unobservable":            len(unobservable),
            "n_duplicates":              n_dup,
            "n_duplicates_linked":       n_linked,
            "n_denial_events":           n_denial,
            "precision_linked":          round(precision_linked, 4),
            "f1_linked":                 round(f1_linked, 4),
            "event_recall":              round(recall, 4),
            "event_recall_observable":   round(recall_obs, 4),
            "action_tag_accuracy":       round(action_acc, 4),
            "temporal_accuracy_sec":     round(sum(deltas) / len(deltas), 2) if deltas else None,
            "temporal_error_median_sec": round(median(deltas), 2) if deltas else None,
            "entity_accuracy":           round(entity_acc, 4),
            "precision":                 round(precision, 4),
            "f1_score":                  round(f1, 4),
            "confidence_calibration": {
                "correlation": None if confidence_corr is None else round(confidence_corr, 4),
                "n_samples":   len(conf_pairs),
            },
            "matching":                 matching,
            "result":                   verdict,
            "matched_pairs":            rows,
            "covered_gt_event_ids":     sorted(covered),
            "unmatched_gt_event_ids":   missed,
            "unobservable_gt_event_ids": unobservable,
            "status":                   "complete",
        })
    except Exception:
        log.exception("Evaluation failed for %s", case_id)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_id}_evaluation.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Evaluation saved -> %s (F1=%s, result=%s)", out_path, result.get("f1_score"), result.get("result"))
    return result


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="Evaluate a pipeline timeline against ground truth.")
    ap.add_argument("--case", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--timeline", required=True)
    ap.add_argument("--output-dir", default="output/evaluation")
    args = ap.parse_args()
    r = evaluate_case(args.case, args.ground_truth, args.timeline, args.output_dir)
    print(json.dumps({k: v for k, v in r.items() if k not in ("matched_pairs",)}, indent=2))
