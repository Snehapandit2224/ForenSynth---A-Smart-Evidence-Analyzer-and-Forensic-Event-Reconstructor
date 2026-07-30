"""
ForenSynth - critique_agent.py  (merged V2)

Merges the best of two critique designs:

  New notebook (ForenSynth-X+):
    - Causal cycle detection (blocking, names weakest edge)
    - Orphaned high-confidence events (warning)
    - LLM coherence with timestamp verification
    - LLM edge dependency ratio (info/transparency)
    - Classification structural sanity
    - Clean severity model: blocking / warning / info
    - Warning-only triggers REVISE on first pass only
    - edges_to_exclude() as the revision primitive

  Original notebook:
    - G1 missing linking event (re_run_er signal)
    - G5 narrative break
    - G7 role count mismatch
    - G3 low confidence events
    - Belief state tracking
    - Memory store integration
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
_root = _Path(__file__).parent.parent
for _p in [str(_root / "agents"), str(_root / "memory"), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from dotenv import load_dotenv
load_dotenv(dotenv_path=_Path(__file__).parent.parent / ".env")

log = logging.getLogger("forensynth.critique_agent")

SEVERITY_LEVELS                = ("info", "warning", "blocking")
ORPHAN_CONFIDENCE_THRESHOLD    = 0.75
LLM_EDGE_DEPENDENCY_WARN_RATIO = 0.50
MIN_LINKS_FOR_DEPENDENCY_CHECK = 2
LOW_CONF_THRESHOLD             = 0.55
NARRATIVE_BREAK_SEC            = 600
MAX_BELIEF_DELTA               = 0.25
BELIEF_FLOOR                   = 0.05

CAUSAL_ACTION_RULES = [
    ("APPROACH","ENTER"),("APPROACH","TAMPER"),("APPROACH","WATCH"),
    ("ENTER","WITHDRAW"),("ENTER","TAMPER"),("ENTER","STEAL"),
    ("ENTER","NAVIGATE"),("ENTER","WORK"),("NAVIGATE","STEAL"),
    # NOTE: LOITER removed — it is optional in ATM cases (suspect may exit directly)
    # ("WITHDRAW","LOITER"),  # optional
    # ("LOITER","EXIT"),      # optional
    ("WITHDRAW","EXIT"),("TAMPER","EXIT"),
    ("STEAL","EXIT"),("WATCH","FLEE"),
    ("EXIT","FLEE"),("EXIT","OBSERVE"),("FLEE","OBSERVE"),
    ("OBSERVE","REPORT"),("COMMUNICATE","CONFIRM"),("CONFIRM","COMMUNICATE"),
    ("INTERCEPT","REPORT"),
]
_MUST_PRECEDE: Dict[str,set] = {}
for _pre,_dep in CAUSAL_ACTION_RULES:
    _MUST_PRECEDE.setdefault(_pre,set()).add(_dep)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class CritiqueIssue:
    check:             str
    severity:          str
    detail:            str
    flagged_edges:     List[Tuple[str,str]] = field(default_factory=list)
    must_not_merge:    List[List[str]]      = field(default_factory=list)
    showrunner_action: str                  = ""
    affected_events:   List[str]            = field(default_factory=list)

    def to_dict(self) -> Dict[str,Any]:
        return {
            "check":             self.check,
            "severity":          self.severity,
            "detail":            self.detail,
            "flagged_edges":     self.flagged_edges,
            "must_not_merge":    self.must_not_merge,
            "showrunner_action": self.showrunner_action,
            "affected_events":   self.affected_events,
            "issue_type":        self.check,
            "description":       self.detail,
            "recommendation":    self.showrunner_action or self._default_rec(),
            "event_id":          self.affected_events[0] if self.affected_events else "",
            "affected_obs":      [],
        }

    def _default_rec(self) -> str:
        if self.severity == "blocking": return "Revision required."
        if self.severity == "warning":  return "Review recommended."
        return "Informational."


@dataclass
class CritiqueResult:
    case_id:           str
    timeline_version:  str
    critique_version:  str
    revision_number:   int
    verdict:           str
    issues:            List[CritiqueIssue] = field(default_factory=list)
    checks_run:        List[str]           = field(default_factory=list)
    llm_used:          bool                = False
    reason:            str                 = ""
    belief_updates:    Dict[str,Dict[str,float]] = field(default_factory=dict)
    overall_score:     float               = 1.0
    requires_revision: bool                = False
    revision_target:   str                 = ""
    narrative_summary: str                 = ""

    def edges_to_exclude(self) -> List[Tuple[str,str]]:
        out = []
        for i in self.issues:
            if i.severity == "blocking" or (i.severity == "warning" and self.revision_number == 0):
                out.extend(i.flagged_edges)
        return out

    def er_constraints(self) -> Dict[str,Any]:
        pairs = []
        for i in self.issues:
            for p in i.must_not_merge:
                if p not in pairs: pairs.append(p)
        return {"must_not_merge": pairs, "must_merge": [], "soft_hints": {}}

    def to_dict(self) -> Dict[str,Any]:
        return {
            "case_id":           self.case_id,
            "timeline_version":  self.timeline_version,
            "critique_version":  self.critique_version,
            "revision_number":   self.revision_number,
            "verdict":           self.verdict,
            "reason":            self.reason,
            "checks_run":        self.checks_run,
            "llm_used":          self.llm_used,
            "overall_score":     round(self.overall_score, 4),
            "requires_revision": self.requires_revision,
            "revision_target":   self.revision_target,
            "narrative_summary": self.narrative_summary,
            "belief_updates":    self.belief_updates,
            "issues":            [i.to_dict() for i in self.issues],
            "gaps":              [i.to_dict() for i in self.issues],
            "edges_to_exclude":  self.edges_to_exclude(),
            "er_constraints":    self.er_constraints(),
            "unresolvable_gaps": [
                i.check for i in self.issues
                if i.severity == "info" or
                (i.severity == "warning" and self.revision_number > 0)
            ],
            "recommended_action": (
                "RECHECK_OBSERVATIONS" if self.revision_target == "er"
                else "REBUILD_WITH_BELIEF_UPDATE" if self.revision_target == "timeline"
                else "FINALIZE"
            ),
        }


# ── LLM coherence client ──────────────────────────────────────────────────────

_COHERENCE_SYSTEM = """You are reviewing a reconstructed forensic timeline for genuine STRUCTURAL contradictions in its own causal/temporal claims.

Only flag a structural impossibility in how the reconstruction itself has ordered or connected events — for example: an effect claimed to happen before its stated cause, or the same person physically in two different locations at the exact same instant.

Do NOT flag any of the following:
- A suspect's DENIAL or STATEMENT contradicting their own observed ACTIONS. Example: "I did not go inside the ATM booth" followed by CCTV of them inside — this is a suspect lying, significant forensic evidence, not a defect.
- Disagreement between DIFFERENT witnesses about the same event — normal evidence uncertainty.
- A witness statement differing from physical evidence — investigative content, not a timeline defect.
- Contradictions between text messages and video of the same person — cross-modal evidence, not structural errors.
- Events that are uncertain, low-confidence, or vaguely described.

Only flag if event A is claimed to CAUSE or PRECEDE event B, but timestamps show B actually occurred before A.

Respond ONLY with JSON:
{"coherent": true|false, "issues": [{"description": "...", "related_event_ids": ["EVT_..."]}]}"""


class CritiqueLLM:
    def __init__(self, model: str = "llama-3.1-8b-instant", timeout: float = 20.0):
        self._model   = model
        self._timeout = timeout
        self._ok      = False
        self._client  = None
        self.calls_made = 0
        key = os.environ.get("Timeline_Key","") or os.environ.get("GROQ_API_KEY","")
        if not key:
            log.info("CritiqueLLM: no API key — coherence check disabled.")
            return
        try:
            from groq import Groq
            self._client = Groq(api_key=key)
            self._ok = True
            log.info("CritiqueLLM: ready (model=%s)", self._model)
        except Exception as exc:
            log.warning("CritiqueLLM init failed (%s)", exc)

    def available(self) -> bool: return self._ok

    def check_coherence(self, lines: List[str]) -> Optional[Dict]:
        if not self._ok or not lines: return None
        user = "Timeline sequence:\n" + "\n".join(f"{i+1}. {l}" for i,l in enumerate(lines))
        try:
            self.calls_made += 1
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role":"system","content":_COHERENCE_SYSTEM},
                          {"role":"user","content":user}],
                temperature=0.0, max_tokens=512, timeout=self._timeout,
            )
            text = resp.choices[0].message.content or ""
            text = re.sub(r"^```(json)?","",text.strip()).rstrip("`").strip()
            result = json.loads(text)
            return result if isinstance(result,dict) and "coherent" in result else None
        except Exception as exc:
            log.warning("CritiqueLLM call failed (%s)", exc)
            self._ok = False
            return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _iso_to_epoch(ts: str) -> float:
    if not ts: return 0.0
    try: return datetime.fromisoformat(ts.replace("Z","+00:00")).timestamp()
    except: return 0.0

def _enrich_events(events: List[Dict]) -> List[Dict]:
    out = []
    for ev in events:
        if not ev.get("ts_epoch"):
            ev = {**ev, "ts_epoch": _iso_to_epoch(ev.get("timestamp",""))}
        out.append(ev)
    return out

def _get_ts_map(timeline: Dict) -> Dict[str,float]:
    return {n["id"]: n.get("ts_epoch",0.0)
            for n in timeline.get("timeline_graph",{}).get("nodes",[])}

def _build_entity_sequences(events: List[Dict]) -> Dict[str,List[Dict]]:
    by_entity: Dict[str,List] = {}
    for ev in events:
        by_entity.setdefault(ev.get("entity_id","UNKNOWN"),[]).append(ev)
    for eid in by_entity:
        by_entity[eid].sort(key=lambda e: e.get("ts_epoch",0.0) or 0.0)
    return by_entity

def _llm_edges_only(related: List[str], links: List[Dict]) -> List[Tuple[str,str]]:
    rel = set(related)
    return [(l["source"],l["target"]) for l in links
            if l["source"] in rel and l["target"] in rel
            and l.get("label","").startswith("local LLM:")]

def _verify_ordering(flagged: List[Tuple[str,str]], ts_map: Dict[str,float]) -> bool:
    for src,tgt in flagged:
        s,t = ts_map.get(src), ts_map.get(tgt)
        if s and t and s > t: return True
    return False


# ── Checks — new notebook ────────────────────────────────────────────────────

def _check_causal_cycles(timeline: Dict) -> Optional[CritiqueIssue]:
    edges = timeline.get("timeline_graph",{}).get("edges",[])
    G = nx.DiGraph()
    for e in edges:
        if e.get("relation") == "BEFORE":
            G.add_edge(e["source"],e["target"],confidence=e.get("confidence",0.0))
    if nx.is_directed_acyclic_graph(G): return None
    cycle = nx.find_cycle(G)
    weakest = min(cycle, key=lambda uv: G[uv[0]][uv[1]]["confidence"])
    cycle_str = " -> ".join([u for u,v in cycle]+[cycle[0][0]])
    return CritiqueIssue(
        check="causal_cycle", severity="blocking",
        detail=(f"Causal cycle: {cycle_str}. Logical impossibility. "
                f"Weakest edge ({weakest[0]}->{weakest[1]}, "
                f"conf={G[weakest[0]][weakest[1]]['confidence']:.2f}) flagged."),
        flagged_edges=[weakest], showrunner_action="re_run_timeline",
    )

def _check_orphaned(timeline: Dict) -> Optional[CritiqueIssue]:
    events = timeline.get("events",[])
    edges  = timeline.get("timeline_graph",{}).get("edges",[])
    connected = {e["source"] for e in edges} | {e["target"] for e in edges}
    orphans = [ev for ev in events
               if ev.get("confidence",0.0) >= ORPHAN_CONFIDENCE_THRESHOLD
               and ev["event_id"] not in connected]
    if not orphans: return None
    ids = ", ".join(ev["event_id"] for ev in orphans)
    return CritiqueIssue(
        check="orphaned_high_confidence_event", severity="warning",
        detail=(f"{len(orphans)} well-supported event(s) have no graph connections: {ids}. "
                "Usually a narrative gap worth investigator attention."),
        affected_events=[ev["event_id"] for ev in orphans],
        showrunner_action="re_run_timeline",
    )

def _check_llm_dependency(timeline: Dict) -> Optional[CritiqueIssue]:
    links = timeline.get("causal_links",[])
    if len(links) < MIN_LINKS_FOR_DEPENDENCY_CHECK: return None
    llm_n = sum(1 for l in links if l.get("label","").startswith("local LLM:"))
    ratio = llm_n / len(links)
    if ratio < LLM_EDGE_DEPENDENCY_WARN_RATIO: return None
    return CritiqueIssue(
        check="llm_edge_dependency_ratio", severity="info",
        detail=(f"{llm_n}/{len(links)} causal links ({ratio:.0%}) rely on LLM inference. "
                "Transparency note — not an error."),
    )

def _check_classification(timeline: Dict) -> Optional[CritiqueIssue]:
    valid = {"CLEAR","PARTIAL","AMBIGUOUS"}
    cls   = timeline.get("output_classification")
    rsn   = timeline.get("output_classification_reason","")
    if cls not in valid:
        return CritiqueIssue(
            check="classification_structural_sanity", severity="warning",
            detail=f"output_classification='{cls}' not in {sorted(valid)} — upstream bug.",
        )
    if not rsn or "avg_confidence" not in rsn:
        return CritiqueIssue(
            check="classification_structural_sanity", severity="warning",
            detail="output_classification_reason missing expected metrics.",
        )
    return None

def _check_llm_coherence(timeline: Dict, llm: CritiqueLLM,
                          revision_number: int) -> Optional[CritiqueIssue]:
    narrative = timeline.get("narrative",[])
    if not narrative or not llm.available(): return None
    lines = [f"{n.get('timestamp','?')} | {n.get('actor','?')}: "
             f"{n.get('action','')} [{n.get('event_id','?')}] "
             f"(conf={n.get('confidence',0):.2f})" for n in narrative]
    result = llm.check_coherence(lines)
    if result is None or result.get("coherent",True): return None
    raw = result.get("issues",[])
    if not raw: return None
    links      = timeline.get("causal_links",[])
    ts_map     = _get_ts_map(timeline)
    flagged    = list(set(e for i in raw for e in _llm_edges_only(i.get("related_event_ids",[]),links)))
    descs      = [i.get("description","") for i in raw]
    if flagged and not _verify_ordering(flagged, ts_map):
        return CritiqueIssue(
            check="llm_coherence", severity="info",
            detail=("LLM flagged: " + "; ".join(descs) +
                    " (timestamps checked — no ordering contradiction found. Informational only.)"),
        )
    severity = "warning" if flagged else "info"
    detail   = "LLM coherence flagged: " + "; ".join(descs)
    if not flagged:
        detail += " (no LLM-derived causal edge connects these events — informational only.)"
    return CritiqueIssue(
        check="llm_coherence", severity=severity, detail=detail,
        flagged_edges=flagged,
        showrunner_action="re_run_timeline" if flagged else "",
    )


# ── Checks — original ─────────────────────────────────────────────────────────

def _check_missing_links(entity_sequences: Dict) -> List[CritiqueIssue]:
    issues = []
    for entity_id, seq in entity_sequences.items():
        for i in range(len(seq)-1):
            ev_a, ev_b = seq[i], seq[i+1]
            tags_a: List[str] = ev_a.get("action_tags",[])
            tags_b: List[str] = ev_b.get("action_tags",[])
            all_inter: set = set()
            for mid in seq[i+1:]: all_inter.update(mid.get("action_tags",[]))
            for tag_a in tags_a:
                for req in _MUST_PRECEDE.get(tag_a,set()):
                    if req in (set(tags_b)|all_inter): continue
                    matching = [tb for tb in tags_b if tb in _MUST_PRECEDE.get(req,set())]
                    if not matching: continue
                    ts_a = ev_a.get("ts_epoch",0.0) or 0.0
                    ts_b = ev_b.get("ts_epoch",0.0) or 0.0
                    window = ts_b-ts_a if ts_b>ts_a else 0.0
                    issues.append(CritiqueIssue(
                        check="missing_linking_event", severity="warning",
                        detail=(f"Rule chain '{tag_a}->{req}->{matching[0]}' broken for "
                                f"{entity_id}: '{req}' missing between "
                                f"{ev_a['event_id']} and {ev_b['event_id']}"
                                +(f" ({window:.0f}s)." if window else ".")+
                                " ER may have missed an observation cluster."),
                        affected_events=[ev_a["event_id"],ev_b["event_id"]],
                        showrunner_action="re_run_timeline",  # timeline rebuild, not ER
                    ))
                    break
    return issues

def _check_narrative_breaks(timeline: Dict, ts_map: Dict[str,float]) -> List[CritiqueIssue]:
    narrative = timeline.get("narrative",[])
    by_actor: Dict[str,List] = {}
    for line in narrative:
        by_actor.setdefault(line.get("actor","UNKNOWN"),[]).append(line)
    issues = []
    for actor, lines in by_actor.items():
        for i in range(len(lines)-1):
            la,lb = lines[i],lines[i+1]
            ts_a = ts_map.get(la.get("event_id",""),0.0) or _iso_to_epoch(la.get("timestamp",""))
            ts_b = ts_map.get(lb.get("event_id",""),0.0) or _iso_to_epoch(lb.get("timestamp",""))
            if ts_a<=0 or ts_b<=0: continue
            gap = ts_b-ts_a
            if gap<=NARRATIVE_BREAK_SEC: continue
            issues.append(CritiqueIssue(
                check="narrative_break", severity="warning",
                detail=(f"{actor} has a {gap/60:.0f}-min gap between "
                        f"{la.get('event_id')} and {lb.get('event_id')}. "
                        "No observations cover this window."),
                affected_events=[la.get("event_id",""),lb.get("event_id","")],
                showrunner_action="human_review",
            ))
    return issues

def _check_role_count(timeline: Dict) -> List[CritiqueIssue]:
    return [
        CritiqueIssue(
            check="role_count_mismatch", severity="warning",
            detail=(f"FIR role count mismatch: {c.get('detail','')} "
                    "Indicates ER over-merged suspects or missing observations."),
            showrunner_action="re_run_er",
        )
        for c in timeline.get("conflicts_summary",[])
        if c.get("conflict_type") == "role_count_mismatch"
    ]

def _check_low_confidence(timeline: Dict) -> List[CritiqueIssue]:
    issues = []
    for u in timeline.get("uncertainties",[]):
        score = u.get("uncertainty_score",0.0)
        if score < 0.45: continue
        ev_id = u.get("event_id","")
        issues.append(CritiqueIssue(
            check="low_confidence_event", severity="info",
            detail=(f"Event {ev_id} confidence={1-score:.2f}. "
                    f"Reasons: {'; '.join(u.get('reasons',[]))}"),
            affected_events=[ev_id],
        ))
    return issues


# ── Belief state ──────────────────────────────────────────────────────────────

def build_belief_state(canonical_entities: List[Dict]) -> Dict[str,Dict[str,float]]:
    belief: Dict[str,Dict[str,float]] = {}
    for ent in canonical_entities:
        eid   = ent.get("entity_id","")
        cands = ent.get("candidate_mentions",[])
        if not cands:
            belief[eid] = {ent.get("primary_alias",eid): 1.0}
            continue
        scores: Dict[str,float] = {}
        for c in cands:
            name  = c.get("alias") or c.get("primary_alias","")
            score = float(c.get("confidence",0.5))
            if name: scores[name] = max(scores.get(name,0.0),score)
        total = sum(scores.values()) or 1.0
        belief[eid] = {n:round(s/total,4) for n,s in scores.items()}
    return belief

def infer_belief_state(events: List[Dict]) -> Dict[str,Dict[str,float]]:
    entities: Dict[str,Dict] = {}
    for ev in events:
        eid   = ev.get("entity_id","")
        alias = ev.get("primary_alias",eid)
        if eid not in entities: entities[eid] = {"alias":alias,"conflict":False}
        if ev.get("conflict_flag"): entities[eid]["conflict"] = True
    belief: Dict[str,Dict[str,float]] = {}
    for eid,info in entities.items():
        belief[eid] = ({info["alias"]:0.70, f"Alt_{eid}":0.30}
                       if info["conflict"] else {info["alias"]:1.0})
    return belief


# ── Main function ─────────────────────────────────────────────────────────────

def critique_timeline(
    timeline: Dict[str,Any],
    er_result: Dict[str,Any] = None,
    revision_number: int = 0,
    critique_version: str = "C1",
    llm: Optional[CritiqueLLM] = None,
) -> CritiqueResult:
    case_id = timeline.get("case_id","UNKNOWN")
    tl_ver  = timeline.get("timeline_version","V1")
    log.info("Critique — case=%s tl=%s crit=%s revision=%d",
             case_id, tl_ver, critique_version, revision_number)

    events = _enrich_events(timeline.get("events",[]))
    timeline = {**timeline, "events": events}
    ts_map           = _get_ts_map(timeline)
    entity_sequences = _build_entity_sequences(events)

    if llm is None: llm = CritiqueLLM()

    issues: List[CritiqueIssue] = []
    checks_run: List[str] = []

    # New notebook checks
    for fn, name in [
        (_check_causal_cycles,  "causal_cycle"),
        (_check_orphaned,       "orphaned_high_confidence_event"),
        (_check_llm_dependency, "llm_edge_dependency_ratio"),
        (_check_classification, "classification_structural_sanity"),
    ]:
        checks_run.append(name)
        r = fn(timeline)
        if r: issues.append(r)

    # Original checks
    for name, results in [
        ("missing_linking_event", _check_missing_links(entity_sequences)),
        ("narrative_break",       _check_narrative_breaks(timeline, ts_map)),
        ("role_count_mismatch",   _check_role_count(timeline)),
        ("low_confidence_event",  _check_low_confidence(timeline)),
    ]:
        checks_run.append(name)
        if isinstance(results, list): issues.extend(results)
        elif results: issues.append(results)

    # LLM coherence (one call)
    checks_run.append("llm_coherence")
    coherence = _check_llm_coherence(timeline, llm, revision_number)
    if coherence: issues.append(coherence)

    # Verdict
    blocking  = [i for i in issues if i.severity == "blocking"]
    warn_r0   = [i for i in issues if i.severity == "warning" and revision_number == 0]
    warn_rest = [i for i in issues if i.severity == "warning" and revision_number > 0]
    er_signals = [i for i in issues if i.showrunner_action == "re_run_er"
                  and i.severity in ("blocking","warning")]

    if blocking or warn_r0 or er_signals:
        verdict = "REVISE"
        parts = []
        if blocking:  parts.append(f"{len(blocking)} blocking")
        if warn_r0:   parts.append(f"{len(warn_r0)} warning(s) on first pass")
        if er_signals and not (blocking or warn_r0):
            parts.append(f"{len(er_signals)} ER-level signal(s)")
        reason = ", ".join(parts)
    else:
        verdict = "ACCEPT"
        reason  = "no blocking issues" + (
            f" ({len(warn_rest)} warning(s) not re-triggering after prior revision)"
            if warn_rest else "")

    # Revision target: ER takes priority over timeline
    revision_target = ""
    if verdict == "REVISE":
        if any(i.showrunner_action == "re_run_er" for i in issues):
            revision_target = "er"
        elif any(i.showrunner_action == "re_run_timeline" for i in issues):
            revision_target = "timeline"

    # Score
    n_b = len(blocking)
    n_w = len(warn_r0) + len(warn_rest)
    n_i = sum(1 for i in issues if i.severity == "info")
    overall_score = max(0.0, 1.0 - (n_b*0.30 + n_w*0.10 + n_i*0.02))

    # Belief updates
    canonical    = (er_result or {}).get("canonical_entities",[])
    belief_state = build_belief_state(canonical) if canonical else infer_belief_state(events)
    belief_updates: Dict[str,Dict[str,float]] = {}

    # Narrative summary
    if not issues:
        narrative_summary = f"Case {case_id} (pass {revision_number+1}): no issues. ACCEPT."
    else:
        top = issues[0]
        narrative_summary = (
            f"Case {case_id} (pass {revision_number+1}): {verdict}. "
            f"{n_b} blocking, {n_w} warning(s), {n_i} info. "
            f"Top: {top.check} — {top.detail[:100]}"
        )

    report = CritiqueResult(
        case_id=case_id, timeline_version=tl_ver,
        critique_version=critique_version, revision_number=revision_number,
        verdict=verdict, issues=issues, checks_run=checks_run,
        llm_used=llm.calls_made > 0, reason=reason,
        belief_updates=belief_updates,
        overall_score=round(overall_score,4),
        requires_revision=(verdict == "REVISE"),
        revision_target=revision_target,
        narrative_summary=narrative_summary,
    )

    log.info("Critique done — verdict=%s score=%.2f revision_target=%s",
             verdict, overall_score, revision_target)
    return report


# ── Public API ────────────────────────────────────────────────────────────────

def run_critique_agent(
    payload: Dict[str,Any],
    critique_version: str = "C1",
    iteration: int = 1,
) -> Dict[str,Any]:
    timeline  = payload.get("timeline",{})
    er_result = payload.get("er_result",{})

    report = critique_timeline(
        timeline=timeline, er_result=er_result,
        revision_number=iteration-1,
        critique_version=critique_version,
        llm=CritiqueLLM(),
    )

    ev_to_obs = {ev["event_id"]: ev.get("obs_ids",[])
                 for ev in timeline.get("events",[])}

    result = report.to_dict()
    for issue_dict in result.get("issues",[]):
        obs = []
        for eid in issue_dict.get("affected_events",[]):
            obs.extend(ev_to_obs.get(eid,[]))
        issue_dict["affected_obs"] = list(dict.fromkeys(obs))

    result["gaps"] = result["issues"]
    return result


def run_critique_agent_from_files(
    timeline_path: str,
    er_path: str = None,
    critique_version: str = "C1",
    output_dir: str = None,
) -> Dict[str,Any]:
    with open(timeline_path, encoding="utf-8") as f:
        timeline = json.load(f)
    er = {}
    if er_path and _Path(er_path).exists():
        with open(er_path, encoding="utf-8") as f:
            er = json.load(f)
    payload = {"timeline":timeline,"er_result":er,"observations":[],"graph":None}
    result  = run_critique_agent(payload, critique_version=critique_version)
    if output_dir:
        case_id  = timeline.get("case_id","UNKNOWN")
        out_path = _Path(output_dir) / f"{case_id}_critique_{critique_version}.json"
        _Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        log.info("Critique saved -> %s", out_path)
    return result