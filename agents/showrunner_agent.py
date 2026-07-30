"""
ForenSynth – showrunner_agent.py  (merged V2)

Merges the best of two showrunner designs:

  New notebook (LangGraph):
    - Explicit convergence detection (score-delta, budget, no-resolvable-gaps)
    - Belief update validation (clamp, floor-guard, renormalize)
    - investigator_flags with deduplication
    - iter_log — full per-iteration audit trail
    - output_case classification: CLEAR_WINNER / PARTIAL / AMBIGUOUS
    - _inject_belief_state into Timeline Agent payload

  Current version:
    - No LangGraph dependency (plain Python)
    - Memory store integration (forensynth.db)
    - ER constraint injection (must_not_merge → re_run_er)
    - Multi-stage routing: re_run_er / re_run_timeline / human_review / no_action
    - Synthetic alias filtering (Alt_, UNRESOLVED, self-pairs)
    - Empty constraint check (skip re_run_er if nothing actionable)

Position in pipeline:
    Critique Agent → ShowrunnerDecision → (ER v2 | Timeline V2 | human_review)

Usage:
    from showrunner_agent import run_showrunner
    decision = run_showrunner(payload)
    # decision["action"]         → re_run_er | re_run_timeline | human_review | no_action
    # decision["output_case"]    → CLEAR_WINNER | PARTIAL | AMBIGUOUS
    # decision["iter_log"]       → full per-iteration audit trail
    # decision["investigator_flags"] → deduplicated unresolvable gaps
"""
from __future__ import annotations

import copy
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("forensynth.showrunner")

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_TIMELINE_VERSION      = 3      # V1 → V2 → V3 → stop
MAX_ER_VERSION            = 3
MAX_BELIEF_DELTA          = 0.25
BELIEF_FLOOR              = 0.05
BELIEF_CEILING            = 0.95
CONVERGENCE_DELTA         = 0.03   # avg confidence delta below this = flatlined
CLEAR_WINNER_GAP          = 0.10   # top TL score must beat next-best by this
PARTIAL_MIN_SCORE         = 0.60

# ── Routing sets (support both old gap_type and new check field names) ────────

ER_LEVEL_GAPS = {
    "G4_unresolved_entity", "G7_role_count_mismatch",
    "G1_missing_linking_event", "role_count_mismatch",
}
TIMELINE_LEVEL_GAPS = {
    "G2_temporal_impossibility", "G3_low_confidence_event", "G6_unresolved_temporal",
    "causal_cycle", "orphaned_high_confidence_event", "llm_coherence",
    "classification_structural_sanity", "missing_linking_event",
}
HUMAN_REVIEW_GAPS = {
    "G5_narrative_break", "narrative_break",
}


# ── Field name helpers ────────────────────────────────────────────────────────

def _gap_type(g: Dict) -> str:
    return g.get("gap_type") or g.get("check") or g.get("issue_type", "")

def _action(g: Dict) -> str:
    return g.get("showrunner_action", "")

def _resolvable(g: Dict) -> bool:
    if "resolvable" in g:
        return bool(g["resolvable"])
    return g.get("severity", "") in ("blocking", "warning")

def _sev_val(g: Dict) -> float:
    """Convert severity to float for comparison."""
    s = g.get("severity", 0)
    if isinstance(s, str):
        return {"blocking": 1.0, "warning": 0.6, "info": 0.2}.get(s, 0.0)
    try:
        return float(s)
    except Exception:
        return 0.0


# ── Belief state helpers (from new notebook) ──────────────────────────────────

def _validate_belief_deltas(
    deltas: Dict[str, Dict[str, float]],
    belief_state: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Clamp deltas to MAX_BELIEF_DELTA, enforce BELIEF_FLOOR.
    Returns validated delta dict.
    """
    validated: Dict[str, Dict[str, float]] = {}
    for entity_id, candidate_deltas in deltas.items():
        cands   = belief_state.get(entity_id, {})
        clamped: Dict[str, float] = {}
        for candidate, delta in candidate_deltas.items():
            if abs(delta) > MAX_BELIEF_DELTA:
                delta = MAX_BELIEF_DELTA * (1 if delta > 0 else -1)
                log.warning("Delta for %s/%s clamped to ±%.2f",
                            entity_id, candidate, MAX_BELIEF_DELTA)
            current_p = cands.get(candidate, 0.0)
            if current_p + delta < BELIEF_FLOOR:
                delta = BELIEF_FLOOR - current_p
            clamped[candidate] = round(delta, 4)
        if clamped:
            validated[entity_id] = clamped
    return validated


def _apply_belief_update(
    belief_state: Dict[str, Dict[str, float]],
    validated_deltas: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Apply validated deltas, enforce floor/ceiling, renormalize each cluster.
    """
    new_bs = copy.deepcopy(belief_state)
    for entity_id, deltas in validated_deltas.items():
        if entity_id not in new_bs:
            continue
        for candidate, delta in deltas.items():
            if candidate not in new_bs[entity_id]:
                continue
            new_p = new_bs[entity_id][candidate] + delta
            new_bs[entity_id][candidate] = max(BELIEF_FLOOR,
                                               min(BELIEF_CEILING, new_p))
        total = sum(new_bs[entity_id].values())
        if total > 0:
            new_bs[entity_id] = {
                k: round(v / total, 4)
                for k, v in new_bs[entity_id].items()
            }
    return new_bs


def _inject_belief_state(
    base_payload: Dict[str, Any],
    belief_state: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """
    Update canonical_entities confidence_score and primary_alias
    with the current top-probability candidate from belief_state.
    This makes the Timeline Agent use the revised entity probabilities.
    """
    payload = copy.deepcopy(base_payload)
    entities = payload.get("entity_resolved", {}).get("canonical_entities", [])
    for ent in entities:
        eid  = ent.get("entity_id", "")
        cands = belief_state.get(eid, {})
        if not cands:
            continue
        top_alias = max(cands, key=cands.get)
        top_conf  = cands[top_alias]
        ent["primary_alias"]    = top_alias
        ent["confidence_score"] = top_conf
    return payload


def _avg_confidence(timeline: Dict) -> float:
    events = timeline.get("events", [])
    if not events:
        return 0.0
    return sum(e.get("confidence", 0.0) for e in events) / len(events)


def _classify_output(all_timelines: List[Dict], final_tl: Dict) -> Tuple[str, str]:
    """
    CLEAR_WINNER / PARTIAL / AMBIGUOUS based on comparing
    avg event confidence across all timeline versions.
    """
    scores      = sorted([_avg_confidence(tl) for tl in all_timelines], reverse=True)
    final_score = _avg_confidence(final_tl)
    has_conflicts = any(
        e.get("conflict_flag", False) for e in final_tl.get("events", [])
    )

    if len(scores) >= 2:
        gap = scores[0] - scores[1]
        if gap > CLEAR_WINNER_GAP and not has_conflicts:
            return "CLEAR_WINNER", (
                f"Top timeline score {scores[0]:.2f} beats next-best "
                f"{scores[1]:.2f} by {gap:.2f} with no conflicts."
            )
        if final_score >= PARTIAL_MIN_SCORE:
            return "PARTIAL", (
                f"Best score {final_score:.2f} exceeds threshold. "
                + ("Has conflict-flagged events." if has_conflicts
                   else "Gap to next-best is narrow.")
            )
        return "AMBIGUOUS", (
            f"Top score {final_score:.2f} is low or scores too close "
            f"(gap={gap:.2f}). Investigator review recommended."
        )

    # Single timeline
    if final_score >= 0.75 and not has_conflicts:
        return "CLEAR_WINNER", f"Single timeline, high confidence ({final_score:.2f}), no conflicts."
    if final_score >= PARTIAL_MIN_SCORE:
        return "PARTIAL", f"Single timeline, moderate confidence ({final_score:.2f})."
    return "AMBIGUOUS", f"Single timeline, low confidence ({final_score:.2f})."


# ── Decision builder ──────────────────────────────────────────────────────────

def _make_decision(
    case_id: str,
    action: str,
    input_tl_version: str,
    input_crit_version: str,
    output_tl_version: str,
    er_constraints: Optional[Dict] = None,
    belief_updates: Optional[Dict] = None,
    reasoning: str = "",
    issues_addressed: Optional[List[str]] = None,
    issues_deferred: Optional[List[str]] = None,
    output_case: str = "AMBIGUOUS",
    iter_log: Optional[List] = None,
    investigator_flags: Optional[List] = None,
    all_timelines: Optional[List] = None,
) -> Dict[str, Any]:
    return {
        "case_id":             case_id,
        "action":              action,
        "input_tl_version":    input_tl_version,
        "input_crit_version":  input_crit_version,
        "output_tl_version":   output_tl_version,
        "er_constraints":      er_constraints or {},
        "belief_updates":      belief_updates or {},
        "reasoning":           reasoning,
        "issues_addressed":    issues_addressed or [],
        "issues_deferred":     issues_deferred or [],
        "output_case":         output_case,
        "iter_log":            iter_log or [],
        "investigator_flags":  investigator_flags or [],
        "all_timelines":       all_timelines or [],
        "generated_at":        datetime.now(timezone.utc).isoformat(),
        "status":              "complete",
    }


# ── Core ShowrunnerAgent ──────────────────────────────────────────────────────

class ShowrunnerAgent:
    """
    Merged Showrunner Agent — plain Python, no LangGraph.

    Incorporates:
    - Convergence detection (score-delta, budget, no-resolvable-gaps)
    - Belief update validation + application + renormalization
    - investigator_flags with deduplication
    - iter_log per-iteration audit trail
    - output_case classification
    - _inject_belief_state for Timeline Agent
    - ER constraint injection (must_not_merge)
    - Memory store integration
    - Synthetic alias filtering
    - Multi-stage routing
    """

    def decide(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        critique     = payload.get("critique", {})
        timeline     = payload.get("timeline", {})
        prev_constr  = payload.get("previous_constraints", {})

        case_id      = critique.get("case_id", "UNKNOWN")
        tl_version   = critique.get("timeline_version", "V1")
        crit_version = critique.get("critique_version", "C1")
        recommended  = critique.get("recommended_action", "FINALIZE")
        gaps         = critique.get("gaps", []) or critique.get("issues", [])
        belief_upd   = critique.get("belief_updates", {})
        iteration    = int(crit_version.lstrip("C")) if crit_version else 1

        log.info("[Showrunner] case=%s  tl=%s  crit=%s  recommended=%s  gaps=%d",
                 case_id, tl_version, crit_version, recommended, len(gaps))

        # ── Build iter_log entry ──────────────────────────────────────────
        resolvable_count = sum(1 for g in gaps if _resolvable(g))
        iter_entry = {
            "iteration":          iteration,
            "timeline_version":   tl_version,
            "critique_version":   crit_version,
            "gaps_detected":      len(gaps),
            "resolvable_gaps":    resolvable_count,
            "unresolvable_gaps":  len(gaps) - resolvable_count,
            "recommended_action": recommended,
            "overall_score":      critique.get("overall_score", 0.0),
            "narrative_summary":  critique.get("narrative_summary", ""),
        }
        iter_log = list(prev_constr.get("iter_log", [])) + [iter_entry]

        # ── Deduplicated investigator_flags ───────────────────────────────
        prev_flags  = list(prev_constr.get("investigator_flags", []))
        unres_ids   = set(critique.get("unresolvable_gaps", []))
        for gap in gaps:
            gap_id = gap.get("gap_id") or gap.get("check") or gap.get("issue_type", "")
            if gap_id not in unres_ids and _resolvable(gap):
                continue
            key = (_gap_type(gap), gap.get("affected_entity","") or gap.get("event_id",""))
            if not any(
                (f.get("gap_type",""), f.get("affected_entity","")) == key
                for f in prev_flags
            ):
                prev_flags.append({
                    "gap_type":        _gap_type(gap),
                    "affected_entity": gap.get("affected_entity","") or gap.get("event_id",""),
                    "detail":          gap.get("detail","") or gap.get("root_cause",""),
                    "fix_hint":        gap.get("fix_hint","") or gap.get("showrunner_action",""),
                    "iteration":       iteration,
                })
        investigator_flags = prev_flags

        # ── Validate + apply belief updates ───────────────────────────────
        belief_state   = prev_constr.get("belief_state", {})
        validated      = _validate_belief_deltas(belief_upd, belief_state)
        new_belief     = _apply_belief_update(belief_state, validated) if validated else belief_state

        # ── All timelines for output_case classification ──────────────────
        all_timelines  = list(prev_constr.get("all_timelines", [])) + [timeline]
        output_case, case_reason = _classify_output(all_timelines, timeline)

        # ── Convergence checks ────────────────────────────────────────────
        prev_avg  = prev_constr.get("prev_avg_confidence", 0.0)
        curr_avg  = _avg_confidence(timeline)
        score_flat = abs(curr_avg - prev_avg) < CONVERGENCE_DELTA and iteration > 1

        if score_flat:
            log.info("[Showrunner] CONVERGENCE — score flatlined (delta=%.4f) — finalizing",
                     abs(curr_avg - prev_avg))
            return _make_decision(
                case_id, "no_action", tl_version, crit_version, tl_version,
                reasoning=(
                    f"Timeline confidence flatlined (delta={abs(curr_avg-prev_avg):.4f} "
                    f"< {CONVERGENCE_DELTA}) — further revision won't improve the result. "
                    f"Output case: {output_case}. {case_reason}"
                ),
                output_case=output_case,
                iter_log=iter_log,
                investigator_flags=investigator_flags,
                all_timelines=all_timelines,
            )

        # ── No-action cases ───────────────────────────────────────────────
        if recommended == "FINALIZE" or not gaps:
            log.info("[Showrunner] FINALIZE — timeline clean")
            return _make_decision(
                case_id, "no_action", tl_version, crit_version, tl_version,
                reasoning=f"Critique found no actionable gaps. Output case: {output_case}. {case_reason}",
                output_case=output_case,
                iter_log=iter_log,
                investigator_flags=investigator_flags,
                all_timelines=all_timelines,
            )

        # ── Version ceiling ───────────────────────────────────────────────
        next_tl = self._next_tl(tl_version)
        if next_tl is None:
            log.info("[Showrunner] VERSION CEILING — human_review")
            return _make_decision(
                case_id, "human_review", tl_version, crit_version, tl_version,
                reasoning=f"Timeline at V{MAX_TIMELINE_VERSION} — max revisions reached.",
                output_case=output_case,
                iter_log=iter_log,
                investigator_flags=investigator_flags,
                all_timelines=all_timelines,
                issues_deferred=[_gap_type(g) for g in gaps],
            )

        # ── Recurrence check ──────────────────────────────────────────────
        prev_gap_types = set(prev_constr.get("previous_gap_types", []))
        if isinstance(prev_constr.get("previous_gap_types"), list):
            prev_gap_types |= set(prev_constr["previous_gap_types"])
        current_gap_types = {_gap_type(g) for g in gaps}
        recurring = current_gap_types & prev_gap_types

        if recurring and tl_version not in ("V1", "V2"):
            log.info("[Showrunner] RECURRENCE %s — human_review", recurring)
            return _make_decision(
                case_id, "human_review", tl_version, crit_version, tl_version,
                reasoning=(
                    f"Gap type(s) {recurring} recurred after previous fix. "
                    "Automated revision cannot resolve — escalating."
                ),
                output_case=output_case,
                iter_log=iter_log,
                investigator_flags=investigator_flags,
                all_timelines=all_timelines,
                issues_deferred=[_gap_type(g) for g in gaps],
            )

        # ── Classify gaps ─────────────────────────────────────────────────
        er_gaps    = [g for g in gaps if _action(g) == "re_run_er"
                      or _gap_type(g) in ER_LEVEL_GAPS]
        tl_gaps    = [g for g in gaps if _action(g) == "re_run_timeline"
                      or _gap_type(g) in TIMELINE_LEVEL_GAPS]
        human_gaps = [g for g in gaps if _action(g) == "human_review"
                      or _gap_type(g) in HUMAN_REVIEW_GAPS]
        tl_gaps    = [g for g in tl_gaps    if g not in er_gaps]
        human_gaps = [g for g in human_gaps if g not in er_gaps and g not in tl_gaps]

        resolvable_tl = [g for g in tl_gaps if _resolvable(g)]

        # G1 recheck (old critique format)
        g1_recheck = [
            g for g in gaps
            if _gap_type(g) == "G1_missing_linking_event"
            and "RECHECK_OBSERVATIONS" in g.get("fix_hint", "")
        ]
        if g1_recheck and not er_gaps:
            er_gaps = g1_recheck

        # ── ER-level action ───────────────────────────────────────────────
        if er_gaps:
            constraints = self._build_er_constraints(er_gaps, gaps, timeline, prev_constr)
            next_er_ver = self._next_er(prev_constr)

            if next_er_ver is None:
                return _make_decision(
                    case_id, "human_review", tl_version, crit_version, tl_version,
                    reasoning="ER re-run limit reached — manual review of entity clustering required.",
                    output_case=output_case, iter_log=iter_log,
                    investigator_flags=investigator_flags, all_timelines=all_timelines,
                    issues_deferred=[_gap_type(g) for g in er_gaps],
                )

            if (not constraints.get("must_not_merge") and
                    not constraints.get("must_merge") and
                    not constraints.get("soft_hints")):
                log.info("[Showrunner] ER constraints empty — human_review")
                return _make_decision(
                    case_id, "human_review", tl_version, crit_version, tl_version,
                    reasoning=(
                        "ER-level issues detected but no actionable constraints could be "
                        "derived (all candidate aliases were synthetic placeholders — "
                        "entity identity is genuinely ambiguous). Manual review required."
                    ),
                    output_case=output_case, iter_log=iter_log,
                    investigator_flags=investigator_flags, all_timelines=all_timelines,
                    issues_deferred=[_gap_type(g) for g in er_gaps],
                )

            log.info("[Showrunner] re_run_er — v%d, must_not_merge=%d",
                     next_er_ver, len(constraints.get("must_not_merge", [])))

            return _make_decision(
                case_id, "re_run_er", tl_version, crit_version, next_tl,
                er_constraints={
                    **constraints,
                    "er_version":          next_er_ver,
                    "previous_gap_types":  list(current_gap_types),
                    "iter_log":            iter_log,
                    "investigator_flags":  investigator_flags,
                    "all_timelines":       all_timelines,
                    "belief_state":        new_belief,
                    "prev_avg_confidence": curr_avg,
                },
                belief_updates=new_belief,
                reasoning=self._build_reasoning(er_gaps, "re_run_er"),
                issues_addressed=[_gap_type(g) for g in er_gaps],
                issues_deferred=[_gap_type(g) for g in tl_gaps + human_gaps],
                output_case=output_case, iter_log=iter_log,
                investigator_flags=investigator_flags, all_timelines=all_timelines,
            )

        # ── Timeline-level action ─────────────────────────────────────────
        if resolvable_tl:
            log.info("[Showrunner] re_run_timeline → %s", next_tl)
            return _make_decision(
                case_id, "re_run_timeline", tl_version, crit_version, next_tl,
                er_constraints={
                    "must_not_merge":      [],
                    "must_merge":          [],
                    "soft_hints":          {},
                    "previous_gap_types":  list(current_gap_types),
                    "iter_log":            iter_log,
                    "investigator_flags":  investigator_flags,
                    "all_timelines":       all_timelines,
                    "belief_state":        new_belief,
                    "prev_avg_confidence": curr_avg,
                },
                belief_updates=new_belief,
                reasoning=self._build_reasoning(resolvable_tl, "re_run_timeline"),
                issues_addressed=[_gap_type(g) for g in resolvable_tl],
                issues_deferred=[_gap_type(g) for g in human_gaps +
                                 [g for g in tl_gaps if not _resolvable(g)]],
                output_case=output_case, iter_log=iter_log,
                investigator_flags=investigator_flags, all_timelines=all_timelines,
            )

        # ── All unresolvable → human review ──────────────────────────────
        log.info("[Showrunner] SURFACE_TO_INVESTIGATOR — no resolvable gaps")
        return _make_decision(
            case_id, "human_review", tl_version, crit_version, tl_version,
            reasoning=(
                f"All detected gaps are unresolvable through automated revision. "
                f"Gap types: {sorted(current_gap_types)}. Escalating."
            ),
            output_case=output_case, iter_log=iter_log,
            investigator_flags=investigator_flags, all_timelines=all_timelines,
            issues_deferred=[_gap_type(g) for g in gaps],
        )

    # ── ER constraint builder ─────────────────────────────────────────────────

    def _build_er_constraints(
        self, er_gaps, all_gaps, timeline, prev_constr
    ) -> Dict[str, Any]:
        must_not_merge: List[List[str]] = []
        must_merge:     List[List[str]] = []
        soft_hints:     Dict[str, float] = {}

        # From explicit must_not_merge on issues
        for issue in all_gaps:
            for pair in issue.get("must_not_merge", []):
                if not (isinstance(pair, list) and len(pair) == 2):
                    continue
                a, b = pair[0], pair[1]
                if a == b:           continue
                if "Alt_" in a:      continue
                if "Alt_" in b:      continue
                if "UNRESOLVED" in a: continue
                if "UNRESOLVED" in b: continue
                if pair not in must_not_merge:
                    must_not_merge.append(pair)

        # From G4 gaps
        for gap in er_gaps:
            if _gap_type(gap) in ("G4_unresolved_entity", "unresolved_entity"):
                current = gap.get("current_candidate", "")
                alt     = gap.get("alternative_candidate", "")
                if not current or not alt:          continue
                if current == alt:                  continue
                if "Alt_" in current or "Alt_" in alt: continue
                if "UNRESOLVED" in current or "UNRESOLVED" in alt: continue
                if [current, alt] not in must_not_merge:
                    must_not_merge.append([current, alt])
                sev = _sev_val(gap)
                soft_hints[current] = soft_hints.get(current, 0.0) - (sev * 0.2)
                soft_hints[alt]     = soft_hints.get(alt, 0.0)     + (sev * 0.2)

        # From G7 gaps — conflict-flagged aliases
        g7 = [g for g in er_gaps if _gap_type(g) in ("G7_role_count_mismatch", "role_count_mismatch")]
        if g7:
            by_entity: Dict[str, List[str]] = {}
            for ev in timeline.get("events", []):
                if ev.get("conflict_flag") and ev.get("primary_alias"):
                    by_entity.setdefault(ev.get("entity_id", ""), []).append(ev["primary_alias"])
            for eid, aliases in by_entity.items():
                if len(aliases) >= 2:
                    for i in range(len(aliases)):
                        for j in range(i + 1, len(aliases)):
                            pair = [aliases[i], aliases[j]]
                            if pair not in must_not_merge:
                                must_not_merge.append(pair)

        # Accumulate from previous constraints
        for pair in prev_constr.get("must_not_merge", []):
            if pair not in must_not_merge:
                must_not_merge.append(pair)
        for pair in prev_constr.get("must_merge", []):
            if pair not in must_merge:
                must_merge.append(pair)
        for alias, delta in prev_constr.get("soft_hints", {}).items():
            soft_hints[alias] = soft_hints.get(alias, 0.0) + delta

        return {
            "must_not_merge": must_not_merge,
            "must_merge":     must_merge,
            "soft_hints":     soft_hints,
        }

    # ── Version helpers ───────────────────────────────────────────────────────

    def _next_tl(self, current: str) -> Optional[str]:
        try:
            n = int(current.lstrip("V"))
        except Exception:
            return None
        return None if n >= MAX_TIMELINE_VERSION else f"V{n+1}"

    def _next_er(self, prev_constr: Dict) -> Optional[int]:
        current = prev_constr.get("er_version", 1) if prev_constr else 1
        return None if current >= MAX_ER_VERSION else current + 1

    # ── Reasoning builder ─────────────────────────────────────────────────────

    def _build_reasoning(self, gaps: List[Dict], action: str) -> str:
        if not gaps:
            return ""
        top   = max(gaps, key=_sev_val)
        types = list({_gap_type(g) for g in gaps})
        n     = len(gaps)
        sev_str = top.get("severity", str(round(_sev_val(top), 2)))
        root    = top.get("root_cause") or top.get("detail", "")

        if action == "re_run_er":
            return (
                f"Detected {n} ER-level gap(s) ({', '.join(types)}). "
                f"Highest: {_gap_type(top)} (severity={sev_str}). "
                f"Root: {root[:100]}. "
                "Injecting ER constraints and re-running entity resolution."
            )
        return (
            f"Detected {n} timeline-level gap(s) ({', '.join(types)}). "
            f"Highest: {_gap_type(top)} (severity={sev_str}). "
            "Applying belief updates and rebuilding timeline."
        )


# ── Public API ────────────────────────────────────────────────────────────────

def run_showrunner(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point called by pipeline.py and memory store.

    Args:
        payload: from memory_store.load_for_showrunner()
                   {critique, timeline, observations, previous_constraints}

    Returns decision dict with keys:
        action              → no_action | re_run_er | re_run_timeline | human_review
        output_tl_version   → V1 | V2 | V3
        output_case         → CLEAR_WINNER | PARTIAL | AMBIGUOUS
        er_constraints      → {must_not_merge, must_merge, soft_hints, er_version, ...}
        belief_updates      → updated belief state
        iter_log            → per-iteration audit trail
        investigator_flags  → deduplicated unresolvable gaps
        reasoning           → human-readable explanation
    """
    return ShowrunnerAgent().decide(payload)


def run_showrunner_from_files(
    critique_path: str,
    timeline_path: str,
    prev_constraints_path: str = None,
    output_dir: str = None,
) -> Dict[str, Any]:
    """Convenience: run Showrunner from JSON files."""
    import json
    from pathlib import Path as _Path

    with open(critique_path,  encoding="utf-8") as f: critique = json.load(f)
    with open(timeline_path,  encoding="utf-8") as f: timeline = json.load(f)

    prev_constr = {}
    if prev_constraints_path and _Path(prev_constraints_path).exists():
        with open(prev_constraints_path, encoding="utf-8") as f:
            prev_constr = json.load(f)

    payload = {
        "critique":             critique,
        "timeline":             timeline,
        "observations":         [],
        "previous_constraints": prev_constr,
    }
    result = run_showrunner(payload)

    if output_dir:
        import json as _json
        case_id  = critique.get("case_id", "UNKNOWN")
        crit_ver = critique.get("critique_version", "C1")
        out_path = _Path(output_dir) / f"{case_id}_showrunner_{crit_ver}.json"
        _Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_path.write_text(_json.dumps(result, indent=2, ensure_ascii=False))
        log.info("Showrunner saved -> %s", out_path)

    return result