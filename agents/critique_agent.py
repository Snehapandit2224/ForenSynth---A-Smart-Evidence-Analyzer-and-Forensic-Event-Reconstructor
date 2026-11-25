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
            # A REVISE verdict with no automatic target (e.g. a narrative
            # break) needs a person, not FINALIZE - FINALIZE here made the
            # Showrunner close the case on issues it was meant to escalate.
            "recommended_action": (
                "RECHECK_OBSERVATIONS" if self.revision_target == "er"
                else "REBUILD_WITH_BELIEF_UPDATE" if self.revision_target == "timeline"
                else "SURFACE_TO_INVESTIGATOR" if self.verdict == "REVISE"
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
    def __init__(self, model: str = "openai/gpt-oss-20b", timeout: float = 20.0,
                 enabled: bool = True):
        self._model   = model
        self._timeout = timeout
        self._ok      = False
        self._client  = None
        self.calls_made = 0
        if not enabled:
            log.info("CritiqueLLM: disabled (--no-llm) - coherence check skipped.")
            return
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
        if not self._ok or not lines or self._client is None: return None
        user = "Timeline sequence:\n" + "\n".join(f"{i+1}. {l}" for i,l in enumerate(lines))
        try:
            self.calls_made += 1
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role":"system","content":_COHERENCE_SYSTEM},
                          {"role":"user","content":user}],
                # gpt-oss-20b is a reasoning model: hidden reasoning tokens
                # count against max_tokens. At 512 with default effort it
                # could spend the whole budget reasoning and return an empty
                # answer ("Expecting value: line 1 column 1"), which also
                # switched the check off for the rest of the run (_ok=False).
                temperature=0.0, max_tokens=1024, timeout=self._timeout,
                reasoning_effort="low", response_format={"type": "json_object"},
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
    # Info, not a revision trigger: rebuilding the timeline from the same
    # evidence cannot connect an event nothing links to. It is surfaced to
    # the investigator (unresolvable_gaps -> investigator_flags) instead.
    return CritiqueIssue(
        check="orphaned_high_confidence_event", severity="info",
        detail=(f"{len(orphans)} well-supported event(s) have no graph connections: {ids}. "
                "Usually a narrative gap worth investigator attention."),
        affected_events=[ev["event_id"] for ev in orphans],
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

def _check_missing_links(entity_sequences: Dict, events: Optional[List[Dict]] = None) -> List[CritiqueIssue]:
    """A step that should sit between two of one person's acts but isn't
    recorded (entered ... left, nothing in between). Every act that would
    fit is named - the evidence doesn't say WHICH one happened, so naming
    one ("STEAL missing") asserted something nothing showed. When another
    source shows one of those acts inside the gap (an ATM log's card-reader
    fault, an FIR claim), it is pointed to instead of calling the step
    missing."""
    issues = []
    for entity_id, seq in entity_sequences.items():
        for i in range(len(seq)-1):
            ev_a, ev_b = seq[i], seq[i+1]
            tags_a: List[str] = ev_a.get("action_tags",[])
            tags_b: List[str] = ev_b.get("action_tags",[])
            all_inter: set = set()
            for mid in seq[i+1:]: all_inter.update(mid.get("action_tags",[]))
            for tag_a in sorted(tags_a):
                bridging = sorted(req for req in _MUST_PRECEDE.get(tag_a, set())
                                  if req not in (set(tags_b) | all_inter)
                                  and any(tb in _MUST_PRECEDE.get(req, set()) for tb in tags_b))
                if not bridging:
                    continue
                ts_a = ev_a.get("ts_epoch",0.0) or 0.0
                ts_b = ev_b.get("ts_epoch",0.0) or 0.0
                window = ts_b-ts_a if ts_b>ts_a else 0.0
                then = sorted(tb for tb in tags_b if any(tb in _MUST_PRECEDE.get(r, set()) for r in bridging))[0]
                elsewhere = [e for e in (events or []) if e.get("entity_id") != entity_id
                             and ts_a <= (e.get("ts_epoch") or 0.0) <= ts_b
                             and set(e.get("action_tags") or []) & set(bridging)]
                # Info: a missing observation can't be produced by
                # re-running the timeline (it used to route there and
                # loop until the score flatlined). Surfaced to the
                # investigator via unresolvable_gaps instead.
                if elsewhere:
                    e = elsewhere[0]
                    detail = (f"{entity_id}: nothing of its own recorded between {tag_a} ({ev_a['event_id']}) and "
                              f"{then} ({ev_b['event_id']})" + (f", {window:.0f}s" if window else "") +
                              f"; another source shows {'/'.join(sorted(set(e.get('action_tags') or []) & set(bridging)))} "
                              f"in that gap ({e.get('event_id')}, {e.get('primary_alias') or e.get('entity_id')}) - "
                              "possibly the same act, by this person; identity not established.")
                else:
                    detail = (f"{entity_id}: nothing recorded between {tag_a} ({ev_a['event_id']}) and {then} "
                              f"({ev_b['event_id']})" + (f", {window:.0f}s" if window else "") +
                              f" - one of {'/'.join(bridging)} would be expected there; no evidence shows which, "
                              "or whether anything happened.")
                issues.append(CritiqueIssue(
                    check="missing_linking_event", severity="info", detail=detail,
                    affected_events=[ev_a["event_id"],ev_b["event_id"]],
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

def _check_fir_unverified(er_result: Optional[Dict]) -> List[CritiqueIssue]:
    """FIR head-counts the evidence doesn't line up with (ER's
    fir_role_check):
      more_than_fir  - more PROVABLY different people of a role than the FIR
                       names: the FIR is contradicted -> human review.
      possible_split - more entities than the FIR names, but not provably
                       different people (most likely one person's aliases
                       left apart) -> info.
      unverified     - nobody independent shows that role -> info.
    (A shortfall, "mismatch", is ER's role_count_mismatch conflict, checked
    in _check_role_count.)"""
    issues = []
    for c in (er_result or {}).get("fir_role_check", []):
        role, exp, got = c.get("role"), c.get("expected_count"), c.get("resolved_count")
        status = c.get("status")
        if status == "unverified":
            issues.append(CritiqueIssue(
                check="fir_role_unverified", severity="info",
                detail=(f"FIR names {exp} '{role}' role(s), but no independent evidence "
                        "(FIR narrative excluded) shows any.")))
        elif status == "more_than_fir":
            issues.append(CritiqueIssue(
                check="fir_role_excess", severity="warning",
                detail=(f"FIR names {exp} '{role}' role(s), but the evidence shows at least "
                        f"{c.get('provably_distinct', got)} provably different people in that role "
                        "(conflicting appearance, voices or recordings). The FIR's account is "
                        "contradicted."),
                showrunner_action="human_review"))
        elif status == "possible_split":
            issues.append(CritiqueIssue(
                check="fir_role_possible_split", severity="info",
                detail=(f"FIR names {exp} '{role}' role(s); the evidence resolves to {got} "
                        "entities with that role, but nothing proves they are different people - "
                        "most likely aliases of one person with no identity evidence linking them.")))
    return issues


def _check_entity_contradictions(er_result: Optional[Dict], observations: List[Dict]) -> List[CritiqueIssue]:
    """One ER entity whose own observations contradict each other - a likely
    WRONG MERGE (two people fused), which no automated re-run can untangle:
      - descriptions that never agree (every pair conflicts: red cap vs blue
        cap, tall vs short),
      - voice prints that differ (cosine below 0.5),
      - conflicting roles stated by the observations themselves (not hints).
    Blocking, for an investigator."""
    if not er_result or not observations:
        return []
    from entity_resolution import EntityResolutionPipeline as _ER, IDENTITY_CANDIDATE_MIN
    obs_by_id = {str(o.get("obs_id")): o for o in observations}
    issues = []
    for ent in er_result.get("canonical_entities", []):
        members = [obs_by_id[o] for o in ent.get("sources", []) if o in obs_by_id]
        reasons = []
        by_alias: Dict[str, List[Dict]] = {}
        for o in members:
            parsed = _ER._parse_appearance(o.get("appearance") or "")
            if any(k != "extra" for k in parsed):
                by_alias.setdefault(o.get("entity", ""), []).append(parsed)
        aliases = sorted(by_alias)
        for i, a in enumerate(aliases):
            for b in aliases[i + 1:]:
                if all(_ER._compare_appearance(pa, pb)[1] for pa in by_alias[a] for pb in by_alias[b]):
                    reasons.append(f"descriptions of '{a}' and '{b}' contradict each other")
        prints = [(o.get("entity", ""), o["identity_embedding"]) for o in members if o.get("identity_embedding")]
        for i, (a, va) in enumerate(prints):
            for b, vb in prints[i + 1:]:
                if len(va) == len(vb) and sum(x * y for x, y in zip(va, vb)) < IDENTITY_CANDIDATE_MIN:
                    reasons.append(f"voices of '{a}' and '{b}' differ")
                    break
        if ent.get("role_conflict") and ent.get("role_source") == "observation":
            reasons.append(f"its observations state conflicting roles ({', '.join(ent.get('roles', []))})")
        if reasons:
            issues.append(CritiqueIssue(
                check="entity_internal_contradiction", severity="blocking",
                detail=(f"Entity {ent.get('entity_id')} ({ent.get('primary_alias')}) may be two people "
                        "fused into one: " + "; ".join(dict.fromkeys(reasons)) + "."),
                showrunner_action="human_review"))
    return issues



# ── Checks from field-agent extras and the FIR's claims ──────────────────────
# Investigator notes (info): they point at something to look at, none of
# them is something a re-run could fix.

_PHYSICAL_ACTS = {"APPROACH", "ENTER", "EXIT", "TAMPER", "WITHDRAW", "FLEE", "STEAL",
                  "LOITER", "WATCH", "NAVIGATE"}
STATEMENT_WINDOW_SEC = 900


def _acts(o: Dict) -> set:
    from shared import extract_action_tags
    action = str(o.get("action") or "").upper()
    if action and action not in ("REPORT", "COMMUNICATE", "CONFIRM"):
        return {action}             # OBSERVE included: the act watched is someone else's
    # no act, or only how it was conveyed (REPORT...): read the words too
    found = set(extract_action_tags(o.get("evidence") or o.get("content") or ""))
    return found | ({action} if action else set())


def _check_statements_vs_evidence(er_result: Optional[Dict], observations: List[Dict]) -> List[CritiqueIssue]:
    """A denial or alibi (the extractors' `tags`) while other evidence of the
    SAME entity shows them physically doing something around that time - a
    statement contradicted by the evidence. Investigative value, not a
    reconstruction defect (see the coherence prompt), so info."""
    entity_of = {o: e.get("entity_id") for e in (er_result or {}).get("canonical_entities", [])
                 for o in e.get("sources", [])}
    by_entity: Dict[str, List[Dict]] = {}
    for o in observations:
        if str(o.get("obs_id")) in entity_of:
            by_entity.setdefault(entity_of[str(o["obs_id"])], []).append(o)
    issues = []
    for o in observations:
        said = {"denial", "alibi"} & set(o.get("tags") or [])
        ent = entity_of.get(str(o.get("obs_id")))
        if not said or not ent:
            continue
        t = o.get("time_offset") if o.get("timestamp") else None
        contra = [x for x in by_entity.get(ent, [])
                  if x is not o and not ({"denial", "alibi"} & set(x.get("tags") or []))
                  and x.get("modality") in ("video", "audio") and _acts(x) & _PHYSICAL_ACTS
                  and (t is None or not x.get("timestamp")
                       or abs((x.get("time_offset") or 0) - t) <= STATEMENT_WINDOW_SEC)]
        if contra:
            x = contra[0]
            issues.append(CritiqueIssue(
                check="statement_contradicted", severity="info",
                detail=(f"{o.get('entity')} gives a {'/'.join(sorted(said))} "
                        f"(\"{(o.get('evidence') or o.get('content') or '')[:90]}\"), but {x.get('modality')} "
                        f"evidence of the same person shows {'/'.join(sorted(_acts(x) & _PHYSICAL_ACTS))} "
                        f"(\"{(x.get('content') or '')[:70]}\")."),
                affected_events=[]))
    return issues


# ── Evidence consistency: what the records say against each other ────────────

# One act is a moment of a few seconds; two records of "the same" exit further
# apart than this, both on trustworthy clocks, are two exits or clocks that
# disagree.
REPEAT_MIN_GAP_SEC = 20
# A claim about who was there (how many left, that no one saw) is checked
# against the footage and other records within this window.
CLAIM_WINDOW_SEC = 120
_SCENE_ACTS = {"ENTER", "EXIT", "FLEE", "WITHDRAW", "TAMPER"}
_MOVE_FAMILY = {"EXIT": "leave", "FLEE": "leave", "ENTER": "enter", "APPROACH": "approach"}
_COUNT_WORDS = {"two": 2, "both": 2, "three": 3, "four": 4, "five": 5, "2": 2, "3": 3, "4": 4, "5": 5}
_COUNT_CLAIM = re.compile(
    r"\b(two|both|three|four|five|[2-5])\s+(?:of them|individuals|people|persons|men|women|guys|"
    r"suspects|males|females|figures)\b", re.I)
_SINGLE_CLAIM = re.compile(r"\b(?:only (?:him|her|one)|(?:he|she) was alone|by (?:him|her)self|"
                           r"nobody else|no one else|no-one else)\b", re.I)
_UNSEEN_CLAIM = re.compile(r"\b(?:no one|nobody|no-one)\b[^.?!]{0,20}?\b(?:saw|seen|noticed|watching|around)\b",
                           re.I)


def _epoch(o: Dict) -> Optional[float]:
    """The pipeline's own reading of a timestamp: "10:00:58Z", "15:30:58+05:30"
    and a plain "10:00:58" are one instant (shared.parse_timestamp). Reading
    them separately put a "+05:30" record hours away from a plain one."""
    from shared import parse_timestamp
    t = parse_timestamp(o.get("timestamp") or "")
    return t if t > 0 else None


def _span(o: Dict, t: float) -> Tuple[float, float]:
    """The stretch of time a claim is about: an FIR claim timed only by the
    incident window ("between 10:00 and 10:06") is about the whole window -
    its timestamp is a placement; anything else, CLAIM_WINDOW_SEC either side."""
    window = o.get("claim_window")
    if o.get("time_basis") == "window" and isinstance(window, (list, tuple)) and len(window) == 2:
        try:
            from shared import parse_timestamp
            lo, hi = (parse_timestamp(str(w)) for w in window)
            if lo > 0 and hi >= lo:
                return lo, hi
        except ValueError:
            pass
    return t - CLAIM_WINDOW_SEC, t + CLAIM_WINDOW_SEC


def _said(o: Dict) -> str:
    return f"{o.get('evidence') or ''} {o.get('content') or ''}"


def _by_entity(er_result: Optional[Dict], observations: List[Dict]) -> Dict[str, List[Dict]]:
    entity_of = {o: e.get("entity_id") for e in (er_result or {}).get("canonical_entities", [])
                 for o in e.get("sources", [])}
    out: Dict[str, List[Dict]] = {}
    for o in observations:
        if str(o.get("obs_id")) in entity_of:
            out.setdefault(entity_of[str(o["obs_id"])], []).append(o)
    return out


def _check_act_sequence(er_result: Optional[Dict], observations: List[Dict]) -> List[CritiqueIssue]:
    """One person's own records, in time order, against the order acts must
    happen in: in before out, no second exit without re-entering, nothing
    done inside after leaving. Catches the same exit filmed twice half a
    minute apart (two exits, or cameras whose clocks disagree) and a message
    stamped "done, leaving" before the one reporting the act inside.
    Only records with a trustworthy time (time_reliability >= 0.7) and no
    denial count; a stated time of a forwarded message is still named."""
    from shared import time_reliability
    issues = []
    for ent, members in _by_entity(er_result, observations).items():
        # a message's own sent time (time_basis document_date) is when the
        # sender reported the act "now" - good enough to order messages
        # the FIR's account is a claim, not a record of the act: its "came out
        # at about 10:05" against the camera's 10:05:20 is not a second exit
        steps = sorted(((_epoch(o), o) for o in _independent(members)
                        if _epoch(o) is not None
                        and (time_reliability(o) >= 0.7
                             or (o.get("modality") == "text" and o.get("time_basis") == "document_date"))
                        and not ({"denial", "alibi"} & set(o.get("tags") or []))
                        # a witness telling what someone else did is timed by
                        # the witness's estimate, often a minute off - not a
                        # record of when the act happened
                        and not (o.get("reported_by") and o.get("reported_by") != o.get("entity"))
                        and _acts(o) & _SCENE_ACTS), key=lambda s: s[0])
        state, last = None, {}
        for t, o in steps:
            # fleeing is a way of leaving: a second flight counts as a second exit
            acts = {"EXIT" if a == "FLEE" else a for a in _acts(o) & _SCENE_ACTS}
            problem = None
            if "EXIT" in acts:
                prev = last.get("EXIT")
                if state == "out" and prev and prev[1].get("source") != o.get("source") \
                        and t - prev[0] >= REPEAT_MIN_GAP_SEC:
                    problem = (f"leaves twice with no re-entry recorded in between: {prev[1].get('obs_id')} "
                               f"({prev[1].get('source')}, {prev[1].get('timestamp')}) and {o.get('obs_id')} "
                               f"({o.get('source')}, {o.get('timestamp')}), {int(t - prev[0])}s apart - "
                               "either a re-entry is missing or the two sources' clocks disagree")
                state, last["EXIT"] = "out", (t, o)
            elif "ENTER" in acts:
                prev = last.get("ENTER")
                if state == "in" and prev and prev[1].get("source") != o.get("source") \
                        and t - prev[0] >= REPEAT_MIN_GAP_SEC:
                    problem = (f"enters twice with no exit recorded in between: {prev[1].get('obs_id')} "
                               f"({prev[1].get('timestamp')}) and {o.get('obs_id')} ({o.get('timestamp')})")
                state, last["ENTER"] = "in", (t, o)
            elif state == "out" and "EXIT" in last:
                prev = last["EXIT"][1]
                problem = (f"{'/'.join(sorted(acts))} recorded at {o.get('timestamp')} ({o.get('obs_id')}) "
                           f"after leaving at {prev.get('timestamp')} ({prev.get('obs_id')}), with no re-entry - "
                           "one of the two timestamps is wrong (a forwarded or late-logged message?)")
            if problem:
                issues.append(CritiqueIssue(
                    check="act_sequence_inconsistent", severity="info",
                    detail=f"{o.get('entity')} ({ent}) {problem}.",
                    affected_events=[]))
    return issues


def _check_claims_vs_evidence(er_result: Optional[Dict], observations: List[Dict]) -> List[CritiqueIssue]:
    """Claims ABOUT the scene checked against other records of it:
      - "no one saw" while someone else is recorded watching (OBSERVE) or
        reporting it around then,
      - a head-count ("two individuals left together", "only him") against
        how many different people the footage shows doing that act then.
    Investigator notes: the claim may be a lie, a mistake, or about another
    moment."""
    entity_of = {o: e.get("entity_id") for e in (er_result or {}).get("canonical_entities", [])
                 for o in e.get("sources", [])}
    who = lambda o: entity_of.get(str(o.get("obs_id")), o.get("entity"))
    timed = [(o, _epoch(o)) for o in observations if _epoch(o) is not None]
    issues = []
    counted: set = set()
    for o, t in timed:
        text = _said(o)
        lo, hi = _span(o, t)
        if _UNSEEN_CLAIM.search(text):
            onlookers = [x for x, tx in timed if who(x) != who(o) and lo <= tx <= hi
                         and "OBSERVE" in _acts(x)
                         and not ({"denial", "alibi"} & set(x.get("tags") or []))]
            if onlookers:
                x = onlookers[0]
                issues.append(CritiqueIssue(
                    check="claim_contradicted", severity="info",
                    detail=(f"{o.get('entity')} claims no one saw (\"{(o.get('evidence') or o.get('content'))[:80]}\", "
                            f"{o.get('timestamp')}), but {x.get('modality')} {x.get('obs_id')} records "
                            f"{x.get('entity')} watching or reporting at {x.get('timestamp')} "
                            f"(\"{(x.get('content') or '')[:70]}\")."),
                    affected_events=[]))
        m = _COUNT_CLAIM.search(text)
        claimed = _COUNT_WORDS[m.group(1).lower()] if m else (1 if _SINGLE_CLAIM.search(text) else None)
        families = {_MOVE_FAMILY[a] for a in _acts(o) if a in _MOVE_FAMILY}
        if claimed is None or not families or o.get("modality") == "video":
            continue
        shown = [x for x, tx in timed if x.get("modality") == "video" and lo <= tx <= hi
                 and families & {_MOVE_FAMILY[a] for a in _acts(x) if a in _MOVE_FAMILY}]
        if not shown:
            continue
        people = set()
        for x in shown:
            people.add(who(x))
            people.update(x.get("co_entities") or [])
        # one document is one claim: an email split into two observations
        # ("two individuals left" / "one carrying a bag") is reported once
        doc = (o.get("modality"), str(o.get("source_doc") or o.get("audio_file") or o.get("obs_id")).split("#")[0])
        if len(people) != claimed and doc not in counted:
            counted.add(doc)
            issues.append(CritiqueIssue(
                check="headcount_mismatch", severity="info",
                detail=(f"{o.get('modality')} {o.get('obs_id')} ({o.get('entity')}) says {claimed} "
                        f"{'person' if claimed == 1 else 'people'} ({'/'.join(sorted(families))}): "
                        f"\"{(o.get('evidence') or o.get('content'))[:80]}\"; the footage around then shows "
                        f"{len(people)} ({', '.join(x.get('obs_id') for x in shown)})."),
                affected_events=[]))
    return issues


# Acts that leave something behind a later report can find: a device on the
# card slot, cash gone, a lock broken.
_RESULT_ACTS = {"TAMPER", "STEAL", "WITHDRAW"}
# How far apart two records' clocks can be, by how their time was known
# (shared.time_reliability): a camera's verified clock vs a log, a message's
# sent time, a witness's estimate.
def _clock_slack(reliability: float) -> float:
    return 15.0 if reliability >= 0.95 else 60.0 if reliability >= 0.7 else 300.0 if reliability >= 0.5 else 900.0


def _check_result_before_act(observations: List[Dict]) -> List[CritiqueIssue]:
    """A RESULT of an act reported before anyone could have done it: "something
    is attached to the card slot" at 10:04:00 when the first recorded entry is
    10:04:08. Either it was done earlier (an earlier visit, someone else) or
    one of the two times is off. Across people - unlike the act-order check -
    and weighed by how far each time can be trusted: beyond both clocks'
    slack it is an inconsistency; within it, the order is unclear."""
    from shared import time_reliability
    results = [o for o in observations if "result" in (o.get("tags") or [])
               and _acts(o) & _RESULT_ACTS and _epoch(o) is not None]
    entries = [o for o in _independent(observations) if "ENTER" in _acts(o) and _epoch(o) is not None
               and not ({"denial", "alibi"} & set(o.get("tags") or []))
               and not (o.get("reported_by") and o.get("reported_by") != o.get("entity"))]
    if not results or not entries:
        return []
    first = min(entries, key=_epoch)
    issues = []
    for r in results:
        gap = _epoch(first) - _epoch(r)
        if gap <= 0:
            continue
        slack = _clock_slack(time_reliability(r)) + _clock_slack(time_reliability(first))
        verdict = ("the gap is larger than both records' timing uncertainty" if gap > slack else
                   f"the gap is within the two records' timing uncertainty (about {slack:.0f}s), so the order is unclear")
        issues.append(CritiqueIssue(
            check="result_before_act", severity="info",
            detail=(f"\"{(r.get('evidence') or r.get('content') or '')[:90]}\" ({r.get('modality')} {r.get('obs_id')}, "
                    f"{r.get('timestamp')}) reports {'/'.join(sorted(_acts(r) & _RESULT_ACTS))} as already done, "
                    f"{gap:.0f}s before the first recorded entry ({first.get('entity')}, {first.get('modality')} "
                    f"{first.get('obs_id')}, {first.get('timestamp')}); {verdict}. Either it was done earlier - an "
                    "earlier visit, or someone else - or one of the two times is off."),
            affected_events=[]))
    return issues


_FRAME_ACTS = {"COMMUNICATE", "CONFIRM", "REPORT"}


def _check_uncorroborated(timeline: Dict) -> List[CritiqueIssue]:
    """Events only a report or the FIR states, with no other independent
    source behind them (the Timeline's basis / corroboration): one email's
    "two individuals left", the FIR's assault. Kept in the timeline - they
    are evidence - but named for the investigator, with whether the people
    in them appear anywhere other than in reports."""
    events = timeline.get("events", []) or []
    seen_directly = {e.get("entity_id") for e in events if e.get("basis") == "observed"}
    issues = []
    for e in events:
        acts = set(e.get("action_tags") or []) - _FRAME_ACTS
        if e.get("basis") not in ("reported", "claim") or e.get("corroboration", 0) or e.get("denial") or not acts:
            continue
        who = e.get("primary_alias") or e.get("entity_id")
        source = "claimed only by the FIR" if e.get("basis") == "claim" else "reported by one source"
        unknown = "" if e.get("entity_id") in seen_directly else \
            f"; {who} appears only in reports - not linked to anyone seen or heard directly"
        issues.append(CritiqueIssue(
            check="uncorroborated_report", severity="info",
            detail=(f"{who} {'/'.join(sorted(acts))} at {e.get('timestamp')} ({e.get('event_id')}) is {source}; "
                    f"no other independent source shows it{unknown}."),
            affected_events=[e.get("event_id")]))
    return issues


def _independent(observations: List[Dict]) -> List[Dict]:
    return [o for o in observations if not o.get("fir_derived")]


def _check_fir_items(fir: Optional[Dict], observations: List[Dict]) -> List[CritiqueIssue]:
    """Weapons / property the FIR claims against what the evidence's tags
    show (and the other way round)."""
    if not fir:
        return []
    issues = []
    obs = _independent(observations)
    for label, claimed, tags in (("weapon", fir.get("weapons") or [], {"weapon"}),
                                 ("property", fir.get("property") or [], {"property", "money"})):
        seen = [o for o in obs if tags & set(o.get("tags") or [])]
        if claimed and not seen:
            issues.append(CritiqueIssue(
                check=f"fir_{label}_uncorroborated", severity="info",
                detail=f"The FIR claims {label}(s) {claimed}; no independent evidence mentions any."))
        elif seen and not claimed and ("weapons" in fir if label == "weapon" else "property" in fir):
            issues.append(CritiqueIssue(
                check=f"evidence_{label}_not_in_fir", severity="info",
                detail=(f"{len(seen)} observation(s) mention a {label} (e.g. "
                        f"\"{(seen[0].get('evidence') or seen[0].get('content') or '')[:80]}\"); the FIR lists none.")))
    return issues


def _name_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z]+", (text or "").lower()) if len(w) > 2}


def _check_fir_people(fir: Optional[Dict], observations: List[Dict]) -> List[CritiqueIssue]:
    """Everyone the FIR names - accused, witnesses, victims - should appear in
    independent evidence: by name (an observation's actor or the people it
    mentions), or for an unnamed accused by a matching description."""
    if not fir:
        return []
    from entity_resolution import EntityResolutionPipeline as _ER
    obs = _independent(observations)
    named = set()
    for o in obs:
        named |= _name_words(o.get("entity", ""))
        for m in o.get("mentioned_entities") or []:
            named |= _name_words(m)
    descriptions = [_ER._parse_appearance(o.get("appearance") or "") for o in obs if o.get("appearance")]
    people = [("accused", p.get("name"), p.get("description")) for p in fir.get("accused_people") or []]
    people += [(r, n, None) for r in ("witness", "victim") for n in fir.get(f"{r}es" if r == "witness" else f"{r}s") or []]
    issues = []
    for role, name, description in people:
        if name and _name_words(name):
            if not _name_words(name) & named:
                issues.append(CritiqueIssue(
                    check="fir_person_unverified", severity="info",
                    detail=f"The FIR names {role} '{name}'; no independent evidence names or mentions them."))
        elif description:
            pd = _ER._parse_appearance(description)
            if any(k != "extra" for k in pd) and not any(
                    (lambda r: not r[1] and r[0] >= 0.4)(_ER._compare_appearance(pd, d)) for d in descriptions):
                issues.append(CritiqueIssue(
                    check="fir_person_unverified", severity="info",
                    detail=f"The FIR describes an unnamed {role} ('{description}'); no evidence shows anyone matching."))
    return issues


FIR_LOCATION_MIN_SIMILARITY = 0.35


def _check_fir_location(fir: Optional[Dict], observations: List[Dict]) -> List[CritiqueIssue]:
    """The FIR's incident location against where the evidence was recorded."""
    place = str((fir or {}).get("location") or "").strip()
    if not place or place.lower() in ("unspecified", "multiple sources"):
        return []
    from shared import semantic_location_similarity
    locations = {str(o.get("location") or "").strip() for o in _independent(observations)}
    locations = {l for l in locations if l and l.lower() not in ("unspecified",)}
    if not locations:
        return []
    best = max(semantic_location_similarity(place.lower(), l.lower()) for l in locations)
    if best >= FIR_LOCATION_MIN_SIMILARITY:
        return []
    return [CritiqueIssue(
        check="fir_location_mismatch", severity="info",
        detail=(f"The FIR places the incident at '{place}'; none of the {len(locations)} evidence locations "
                f"resembles it (best similarity {best:.2f})."))]


def _check_fir_delay(fir: Optional[Dict]) -> List[CritiqueIssue]:
    delay = (fir or {}).get("delay_in_reporting")
    return [CritiqueIssue(check="fir_reporting_delay", severity="info",
                          detail=f"The FIR records a delay in reporting: {delay}.")] if delay else []


def _check_video_quality(observations: List[Dict]) -> List[CritiqueIssue]:
    """What the video agent itself flagged: footage problems, an unreadable or
    unverified clock, people it couldn't identify, checks it left open."""
    issues = []
    for o in observations:
        if o.get("modality") != "video":
            continue
        problems = list(o.get("video_flags") or [])
        if o.get("footage_issue"):
            problems.append(f"footage issue: {o['footage_issue']}")
        if isinstance(o.get("footage_quality"), (int, float)) and o["footage_quality"] < 0.3:
            problems.append(f"low footage quality ({o['footage_quality']:.2f})")
        if o.get("unresolved_checks"):
            problems.append(f"unresolved checks: {', '.join(map(str, o['unresolved_checks']))}")
        if problems:
            issues.append(CritiqueIssue(
                check="video_evidence_quality", severity="info",
                detail=f"Video observation {o.get('obs_id')} ({o.get('entity')}): " + "; ".join(problems) + "."))
    return issues

# A record whose content is unavailable or unverified: deleted, corrupted,
# redacted, metadata-only, truncated, unsent. Matched on the bracketed status
# marker exports put in place of the body, or flagged by the text agent.
_INTEGRITY_MARKER = re.compile(
    r"^\s*\[(deleted|undelivered|corrupted|redacted|metadata only|forwarded\W+original deleted|partial|draft)\b"
    r"|hash mismatch|integrity (?:unverified|check failed)|checksum (?:mismatch|failed)", re.I)


def _check_evidence_integrity(observations: List[Dict]) -> List[CritiqueIssue]:
    """Records whose content cannot be trusted as written. They are reported
    rather than silently used or dropped: a corrupted message is still a
    record that something was sent."""
    issues = []
    for o in observations:
        if o.get("extraction_status") == "integrity_failed" or _INTEGRITY_MARKER.search(str(o.get("content") or "")):
            issues.append(CritiqueIssue(
                check="evidence_integrity", severity="warning",
                detail=f"{o.get('modality', 'evidence')} record {o.get('obs_id')} ({o.get('entity')}, "
                       f"{o.get('timestamp') or 'undated'}) has unavailable or unverified content: "
                       f"\"{str(o.get('content'))[:120]}\". Do not treat it as evidence of what it would say; "
                       f"verify against the original source."))
    return issues


FIR_WINDOW_MARGIN_SEC = 60
FIR_WINDOW_MIN_INSIDE = 0.5


def _check_fir_time(fir: Optional[Dict], observations: List[Dict]) -> List[CritiqueIssue]:
    """The FIR's incident window against the independent evidence: if most
    timed observations (FIR narrative excluded) fall outside it, the FIR's
    timing is contradicted - for an investigator."""
    window = (fir or {}).get("time_window")
    if not (isinstance(window, (list, tuple)) and len(window) == 2) or window[1] <= window[0]:
        return []
    start, end = window[0] - FIR_WINDOW_MARGIN_SEC, window[1] + FIR_WINDOW_MARGIN_SEC
    offsets = [o.get("time_offset") for o in observations
               if o.get("timestamp") and not o.get("fir_derived") and isinstance(o.get("time_offset"), (int, float))]
    if len(offsets) < 3:
        return []
    inside = sum(1 for t in offsets if start <= t <= end) / len(offsets)
    if inside >= FIR_WINDOW_MIN_INSIDE:
        return []
    return [CritiqueIssue(
        check="fir_time_contradiction", severity="warning",
        detail=(f"Only {inside:.0%} of the {len(offsets)} timed observations fall inside the FIR's "
                f"incident window ({window[0]}-{window[1]}s from the case start). The FIR's timing is "
                "contradicted by the evidence."),
        showrunner_action="human_review")]

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
    er_result: Optional[Dict[str,Any]] = None,
    revision_number: int = 0,
    critique_version: str = "C1",
    llm: Optional[CritiqueLLM] = None,
    observations: Optional[List[Dict[str,Any]]] = None,
    fir: Optional[Dict[str,Any]] = None,
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
        ("missing_linking_event", _check_missing_links(entity_sequences, events)),
        ("narrative_break",       _check_narrative_breaks(timeline, ts_map)),
        ("role_count_mismatch",   _check_role_count(timeline)),
        ("low_confidence_event",  _check_low_confidence(timeline)),
        ("fir_role_unverified",   _check_fir_unverified(er_result)),
        ("entity_internal_contradiction", _check_entity_contradictions(er_result, observations or [])),
        ("fir_time_contradiction", _check_fir_time(fir, observations or [])),
        ("statement_contradicted", _check_statements_vs_evidence(er_result, observations or [])),
        ("act_sequence_inconsistent", _check_act_sequence(er_result, observations or [])),
        ("claims_vs_evidence",    _check_claims_vs_evidence(er_result, observations or [])),
        ("result_before_act",     _check_result_before_act(observations or [])),
        ("uncorroborated_report", _check_uncorroborated(timeline)),
        ("fir_items",             _check_fir_items(fir, observations or [])),
        ("fir_person_unverified", _check_fir_people(fir, observations or [])),
        ("fir_location_mismatch", _check_fir_location(fir, observations or [])),
        ("fir_reporting_delay",   _check_fir_delay(fir)),
        ("video_evidence_quality", _check_video_quality(observations or [])),
        ("evidence_integrity",     _check_evidence_integrity(observations or [])),
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
    llm_enabled: bool = True,
) -> Dict[str,Any]:
    timeline  = payload.get("timeline",{})
    er_result = payload.get("er_result",{})

    report = critique_timeline(
        timeline=timeline, er_result=er_result,
        revision_number=iteration-1,
        critique_version=critique_version,
        llm=CritiqueLLM(enabled=llm_enabled),
        observations=payload.get("observations"),
        fir=payload.get("fir"),
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
    er_path: Optional[str] = None,
    critique_version: str = "C1",
    output_dir: Optional[str] = None,
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