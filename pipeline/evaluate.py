#!/usr/bin/env python3
"""
ForenSynth - Ground Truth Evaluation Layer
Compares a pipeline-reconstructed timeline against the case generator's
ground truth to measure event recall, action-tag accuracy, temporal
accuracy, entity accuracy, an overall F1 score, and confidence calibration.

Matching algorithm:
  1. For each ground-truth event, look at pipeline events whose alias(es)
     fuzzy-resolve (via ground_truth.entity_mapping) to the same entity_id.
     "Fuzzy" means: exact alias match first, then a case-insensitive
     substring match (so e.g. "UNRESOLVED_person_14" still resolves).
  2. Among those same-entity candidates, pick the one with the smallest
     timestamp delta (each pipeline event can only be used once).
  3. A ground-truth event with no same-entity pipeline event at all is
     unmatched (an event the pipeline never reconstructed).

Action-tag accuracy and entity accuracy are then computed independently
on top of that matched set: action match is a fuzzy keyword check between
the ground truth's snake_case action (e.g. "approach_atm") and the
pipeline's action_tags (e.g. "APPROACH"); entity accuracy re-checks with
the STRICT (non-substring) alias resolver, so a pair that only matched
via a loose substring alias overlap can still be marked entity-incorrect.

Usage:
    python pipeline/evaluate.py --case CASE_ATM_001 \\
        --ground-truth GENERATOR_FIXED/cases_atm/CASE_ATM_001.json \\
        --timeline output/timelines/CASE_ATM_001_timeline_V2.json
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

PASS_THRESHOLD = 0.75
PARTIAL_THRESHOLD = 0.55


def _parse_ts(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", ""))
    except ValueError:
        return None


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _resolve_entity_strict(alias: str, entity_mapping: Dict[str, str]) -> Optional[str]:
    """Exact (case-insensitive) alias -> ground-truth entity_id lookup only."""
    if not alias:
        return None
    if alias in entity_mapping:
        return entity_mapping[alias]
    a = _norm(alias)
    for gt_alias, gt_eid in entity_mapping.items():
        if _norm(gt_alias) == a:
            return gt_eid
    return None


def _resolve_entity_fuzzy(alias: str, entity_mapping: Dict[str, str]) -> Optional[str]:
    """Strict match first, then a partial/substring alias match -- e.g.
    'UNRESOLVED_person_14' or a slightly-mangled alias still resolves."""
    hit = _resolve_entity_strict(alias, entity_mapping)
    if hit:
        return hit
    a = _norm(alias)
    if not a:
        return None
    for gt_alias, gt_eid in entity_mapping.items():
        g = _norm(gt_alias)
        if g and (g in a or a in g):
            return gt_eid
    return None


def _pipeline_event_entity(ev: Dict[str, Any], entity_mapping: Dict[str, str],
                            strict: bool = False) -> Optional[str]:
    """Try every alias a pipeline event carries (primary + known aliases),
    plus its raw entity_id with an 'UNRESOLVED_' prefix stripped."""
    resolver = _resolve_entity_strict if strict else _resolve_entity_fuzzy
    candidates = [ev.get("primary_alias", "")] + list(ev.get("aliases") or [])
    eid = ev.get("entity_id", "") or ""
    if eid.startswith("UNRESOLVED_"):
        candidates.append(eid[len("UNRESOLVED_"):])
    for alias in candidates:
        hit = resolver(alias, entity_mapping)
        if hit:
            return hit
    return None


def _fuzzy_action_match(gt_action: str, action_tags: List[str]) -> bool:
    """gt_action is a snake_case phrase like 'approach_atm'; action_tags are
    short pipeline keywords like 'APPROACH'. Match if either contains the other."""
    if not gt_action or not action_tags:
        return False
    gt_norm = gt_action.replace("_", " ").upper()
    gt_verb = gt_action.split("_")[0].upper()
    for tag in action_tags:
        t = (tag or "").upper()
        if not t:
            continue
        if t in gt_norm or gt_verb in t or t in gt_verb:
            return True
    return False


def _match_events(
    gt_events: List[Dict[str, Any]],
    pipeline_events: List[Dict[str, Any]],
    entity_mapping: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Greedy match: each ground-truth event (chronological order) is paired
    with the nearest-in-time, same-(fuzzy)-entity pipeline event that hasn't
    already been used. Returns (matched_pairs, unmatched_gt_event_ids)."""
    used = set()
    pairs = []
    unmatched = []

    p_entities = [_pipeline_event_entity(p, entity_mapping) for p in pipeline_events]

    for g in sorted(gt_events, key=lambda e: e.get("timestamp") or ""):
        g_ts = _parse_ts(g.get("timestamp", ""))
        best_idx, best_delta = None, None
        for i, p in enumerate(pipeline_events):
            if i in used or p_entities[i] != g.get("entity_id"):
                continue
            p_ts = _parse_ts(p.get("timestamp", ""))
            delta = float("inf") if (g_ts is None or p_ts is None) else abs((p_ts - g_ts).total_seconds())
            if best_delta is None or delta < best_delta:
                best_idx, best_delta = i, delta

        if best_idx is None:
            unmatched.append(g.get("event_id", "?"))
            continue

        used.add(best_idx)
        p = pipeline_events[best_idx]
        p_ts = _parse_ts(p.get("timestamp", ""))
        pairs.append({
            "gt_event_id":                 g.get("event_id", "?"),
            "pipeline_event_id":           p.get("event_id", "?"),
            "gt_action":                   g.get("action", ""),
            "pipeline_action_tags":        p.get("action_tags", []),
            "action_match":                _fuzzy_action_match(g.get("action", ""), p.get("action_tags", [])),
            "gt_entity_id":                g.get("entity_id", ""),
            "pipeline_entity_strict_match": _pipeline_event_entity(p, entity_mapping, strict=True) == g.get("entity_id"),
            "temporal_delta_sec":          None if (g_ts is None or p_ts is None) else abs((p_ts - g_ts).total_seconds()),
            "pipeline_confidence":         p.get("confidence"),
        })

    return pairs, unmatched


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
    compatibility with the spec), pass the case's actual FINAL timeline
    file -- most cases converge at V1 or V2 and never reach V3, and using
    a stale/mismatched V3 file would silently compare against the wrong data.

    Returns a metrics dict and also writes it to
    {output_dir}/{case_id}_evaluation.json.
    """
    result: Dict[str, Any] = {
        "case_id": case_id,
        "n_ground_truth_events": None, "n_pipeline_events": None, "n_matched": None,
        "event_recall": None, "action_tag_accuracy": None,
        "temporal_accuracy_sec": None, "entity_accuracy": None,
        "precision": None, "f1_score": None,
        "confidence_calibration": {"correlation": None, "n_samples": 0},
        "result": "ERROR", "status": "failed",
    }
    try:
        gt_data = json.loads(Path(ground_truth_path).read_text(encoding="utf-8"))
        tl_data = json.loads(Path(timeline_v3_path).read_text(encoding="utf-8"))

        gt = gt_data.get("ground_truth", {})
        gt_events = gt.get("events", [])
        entity_mapping = gt.get("entity_mapping", {})
        pipeline_events = tl_data.get("events", [])

        pairs, unmatched = _match_events(gt_events, pipeline_events, entity_mapping)

        n_gt, n_pipeline, n_matched = len(gt_events), len(pipeline_events), len(pairs)

        recall    = (n_matched / n_gt) if n_gt else 0.0
        precision = (n_matched / n_pipeline) if n_pipeline else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        action_acc = (sum(1 for p in pairs if p["action_match"]) / n_matched) if n_matched else 0.0
        entity_acc = (sum(1 for p in pairs if p["pipeline_entity_strict_match"]) / n_matched) if n_matched else 0.0

        temporal_deltas = [p["temporal_delta_sec"] for p in pairs if p["temporal_delta_sec"] is not None]
        temporal_acc = (sum(temporal_deltas) / len(temporal_deltas)) if temporal_deltas else None

        conf_pairs = [(p["pipeline_confidence"], 1.0 if (p["action_match"] and p["pipeline_entity_strict_match"]) else 0.0)
                      for p in pairs if p["pipeline_confidence"] is not None]
        confidences  = [c for c, _ in conf_pairs]
        correctness  = [c for _, c in conf_pairs]
        confidence_corr = _correlation(confidences, correctness)

        if f1 >= PASS_THRESHOLD:
            verdict = "PASS"
        elif f1 >= PARTIAL_THRESHOLD:
            verdict = "PARTIAL"
        else:
            verdict = "FAIL"

        result.update({
            "n_ground_truth_events": n_gt,
            "n_pipeline_events":     n_pipeline,
            "n_matched":             n_matched,
            "event_recall":          round(recall, 4),
            "action_tag_accuracy":   round(action_acc, 4),
            "temporal_accuracy_sec": None if temporal_acc is None else round(temporal_acc, 2),
            "entity_accuracy":       round(entity_acc, 4),
            "precision":             round(precision, 4),
            "f1_score":              round(f1, 4),
            "confidence_calibration": {
                "correlation": None if confidence_corr is None else round(confidence_corr, 4),
                "n_samples":   len(confidences),
            },
            "result":                 verdict,
            "matched_pairs":          pairs,
            "unmatched_gt_event_ids": unmatched,
            "status":                 "complete",
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
