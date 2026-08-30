#!/usr/bin/env python3
"""
scene_planner.py  —  Visual Scene Planner for ForenSynth
Transforms Timeline Agent V3 JSON + Critique C3 JSON into render-safe scene specs.

Rules applied:
  1. Group events with identical timestamps into a single simultaneous scene
  2. Classify each event: physical_event | statement | audio_report | text_report
  3. Separate source_entity (who produced the evidence) from participants (who acted)
  4. Flag action_tag mismatches vs content
  5. Flag location conflicts (e.g. "office" in an ATM case)
  6. Mark unresolved entities with uncertainty
  7. Attach critique gaps to affected scenes
"""
import json
from pathlib import Path
from collections import defaultdict

BASE = Path("/sessions/gallant-practical-heisenberg/mnt/outputs")

# ── Loaders ───────────────────────────────────────────────────────────────────
def load(name):
    return json.loads((BASE / name).read_text())

# ── Heuristics ────────────────────────────────────────────────────────────────
STATEMENT_KEYWORDS = [
    "did not", "i did not", "did not go", "wasn't", "was not",
    "never", "i never", "denied", "deny", "claim", "alleged",
    "said that", "stated that", "testified"
]
CONFLICT_KEYWORDS = ["office", "workplace", "building", "corridor", "hallway"]
CASE_CONTEXT = "ATM"  # for conflict detection

TAMPER_KEYWORDS = ["fiddle", "fiddling", "tamper", "tampering", "manipulat",
                   "insert", "card slot", "card reader", "skimm", "device"]

ACTION_TAG_CORRECTIONS = {
    "WITHDRAW": {
        "keywords": TAMPER_KEYWORDS,
        "corrected": "TAMPER_WITH_CARD_READER",
        "note": "Action tag 'WITHDRAW' does not match content describing card-slot manipulation."
    }
}

def classify_event(ev):
    """Return (event_class, visualizable, source_only)."""
    mod     = ev.get("modality", "")
    content = (ev.get("content") or "").lower()
    alias   = (ev.get("primary_alias") or "").lower()
    role    = (ev.get("role") or "").lower()
    tags    = [t.lower() for t in (ev.get("action_tags") or [])]

    # Text/email/document source — entity is always a source, not a scene actor
    if mod == "text" or "email" in alias or "doc" in alias:
        return "text_report", False, True

    # Audio source
    if mod == "audio" or "speaker" in alias:
        return "audio_report", False, True

    # Denial / statement check on any modality
    if any(k in content for k in STATEMENT_KEYWORDS):
        return "statement", False, False

    # Default: physical event
    return "physical_event", True, False


def detect_action_issues(ev):
    """Return list of {tag, corrected, note} for mismatched action tags."""
    content = (ev.get("content") or "").lower()
    issues  = []
    for tag, rule in ACTION_TAG_CORRECTIONS.items():
        if tag in (ev.get("action_tags") or []):
            if any(k in content for k in rule["keywords"]):
                issues.append({
                    "original_tag": tag,
                    "corrected_tag": rule["corrected"],
                    "note": rule["note"]
                })
    return issues


def detect_location_conflict(ev):
    """Return conflict dict if content references wrong context location."""
    content = (ev.get("content") or "").lower()
    for kw in CONFLICT_KEYWORDS:
        if kw in content:
            return {
                "conflict": True,
                "content_mentions": kw,
                "case_context": CASE_CONTEXT,
                "note": f"Content mentions '{kw}' but case context is {CASE_CONTEXT}. Human review required."
            }
    return {"conflict": False}


def resolve_participants(ev, event_class, source_only):
    """
    Return (participants, evidence_sources).
    source_only → entity goes to evidence_sources, participants = [unknown subjects]
    """
    alias   = ev.get("primary_alias", "?")
    eid     = ev.get("entity_id", "?")
    role    = ev.get("role", "unknown")
    content = (ev.get("content") or "").lower()

    if source_only:
        # Entity is a witness/document — extract implied subjects from content
        subjects = []
        if "two men" in content or "two people" in content:
            subjects = [
                {"entity_id": "UNKNOWN_SUBJECT_1", "role": "subject", "action": "flee", "resolved": False},
                {"entity_id": "UNKNOWN_SUBJECT_2", "role": "subject", "action": "flee", "resolved": False},
            ]
        elif "someone" in content or "individual" in content or "person" in content:
            subjects = [
                {"entity_id": "UNKNOWN_SUBJECT_1", "role": "subject", "action": "flee", "resolved": False}
            ]
        evidence = [{
            "source_id": alias,
            "entity_id": eid,
            "modality":  ev.get("modality","?"),
            "role":      "witness" if "witness" in role else "reporter",
            "content":   ev.get("content","")[:120],
        }]
        return subjects, evidence

    # Physical event / statement — entity IS a participant
    participant_role = "observer" if "bystander" in role or "observe" in " ".join(
        ev.get("action_tags") or []).lower() else "subject"

    participant = {
        "entity_id": eid,
        "primary_alias": alias,
        "role": participant_role,
        "action": (ev.get("action_tags") or ["UNKNOWN"])[0],
        "resolved": False,  # all entities unresolved per timeline warning
        "confidence": ev.get("confidence", 0),
    }
    return [participant], []


# ── Main Planner ──────────────────────────────────────────────────────────────
def build_scene_specs(tl_path, cr_path):
    tl = load(tl_path)
    cr = load(cr_path)
    events   = tl["events"]
    causal   = tl.get("causal_links", [])
    gaps     = cr.get("gaps", [])

    # Map event_id → gaps
    gap_map = defaultdict(list)
    for g in gaps:
        for eid in g.get("affected_events", []):
            gap_map[eid].append({
                "gap_id":   g.get("gap_id","?"),
                "gap_type": g.get("gap_type","?"),
                "severity": g.get("severity", 0),
                "note":     g.get("narrative_label",""),
            })

    # Soften causal links — label them as possible sequences
    causal_soft = []
    for lnk in causal:
        causal_soft.append({
            "source": lnk["source"],
            "target": lnk["target"],
            "relationship": "POSSIBLE_SEQUENCE",  # never assert definite causality
            "confidence":   lnk.get("confidence", 0.70),
            "original_label": lnk.get("label",""),
            "note": "Causal relationship is unverified; rendered as possible temporal sequence."
        })

    # Group events by timestamp
    ts_groups = defaultdict(list)
    for ev in events:
        ts = (ev.get("timestamp") or "")[:19]
        ts_groups[ts].append(ev)

    scenes = []
    scene_num = 0

    for ts in sorted(ts_groups.keys()):
        group = ts_groups[ts]
        scene_num += 1
        simultaneous = len(group) > 1

        if simultaneous:
            # ── Simultaneous group ─────────────────────────────────────────
            all_participants = []
            all_sources      = []
            all_conflicts    = []
            all_gaps         = []
            all_action_issues= []
            event_ids        = []
            avg_conf         = sum(e.get("confidence",0) for e in group) / len(group)

            for ev in group:
                event_ids.append(ev.get("event_id","?"))
                ec, vis, src_only = classify_event(ev)
                parts, srcs       = resolve_participants(ev, ec, src_only)
                loc_conflict      = detect_location_conflict(ev)
                act_issues        = detect_action_issues(ev)
                ev_gaps           = gap_map.get(ev.get("event_id",""), [])

                all_participants.extend(parts)
                all_sources.extend(srcs)
                if loc_conflict["conflict"]:
                    all_conflicts.append({**loc_conflict, "from_event": ev.get("event_id","?")})
                all_gaps.extend(ev_gaps)
                all_action_issues.extend(act_issues)

            # Deduplicate participants by entity_id
            seen = set()
            deduped = []
            for p in all_participants:
                if p["entity_id"] not in seen:
                    seen.add(p["entity_id"])
                    deduped.append(p)

            scenes.append({
                "scene_id":       f"SCENE_{scene_num:03d}",
                "scene_type":     "simultaneous_group",
                "event_ids":      event_ids,
                "timestamp":      ts,
                "location":       group[0].get("location","?"),
                "participants":   deduped,
                "evidence_sources": all_sources,
                "location_conflicts": all_conflicts,
                "action_issues":  all_action_issues,
                "critique_gaps":  all_gaps,
                "confidence":     avg_conf,
                "status":         "partially_corroborated" if all_sources else "observed",
                "requires_human_review": bool(all_conflicts or all_gaps),
                "causal_links":   [c for c in causal_soft
                                   if c["source"] in event_ids or c["target"] in event_ids],
            })

        else:
            # ── Single event ───────────────────────────────────────────────
            ev = group[0]
            ev_id       = ev.get("event_id","?")
            ec, vis, src_only = classify_event(ev)
            parts, srcs = resolve_participants(ev, ec, src_only)
            loc_conflict= detect_location_conflict(ev)
            act_issues  = detect_action_issues(ev)
            ev_gaps     = gap_map.get(ev_id, [])

            # Correct display action tag if mismatched
            display_action = (ev.get("action_tags") or ["UNKNOWN"])[0]
            if act_issues:
                display_action = act_issues[0]["corrected_tag"]

            scenes.append({
                "scene_id":       f"SCENE_{scene_num:03d}",
                "scene_type":     ec,
                "event_ids":      [ev_id],
                "timestamp":      ts,
                "location":       ev.get("location","?"),
                "participants":   parts,
                "evidence_sources": srcs,
                "display_action": display_action,
                "raw_content":    ev.get("content",""),
                "modality":       ev.get("modality","?"),
                "location_conflicts": [loc_conflict] if loc_conflict["conflict"] else [],
                "action_issues":  act_issues,
                "critique_gaps":  ev_gaps,
                "confidence":     ev.get("confidence", 0),
                "visualizable":   vis,
                "status":         "observed" if ec=="physical_event" else
                                  "statement" if ec=="statement" else "reported",
                "requires_human_review": bool(loc_conflict["conflict"] or ev_gaps or act_issues),
                "causal_links":   [c for c in causal_soft
                                   if c["source"]==ev_id or c["target"]==ev_id],
            })

    return {
        "case_id":         tl["case_id"],
        "tl_version":      "V3",
        "critique_version":"C3",
        "total_scenes":    len(scenes),
        "avg_confidence":  sum(s["confidence"] for s in scenes)/max(len(scenes),1),
        "classification":  tl.get("output_classification","?"),
        "scenes":          scenes,
        "unresolved_entities": tl.get("unresolved_entities",[]),
        "showrunner_verdict": "human_review",
    }


if __name__ == "__main__":
    spec = build_scene_specs(
        "CASE_ATM_001_timeline_V3.json",
        "CASE_ATM_001_critique_C3.json"
    )
    out = BASE / "CASE_ATM_001_scene_spec.json"
    out.write_text(json.dumps(spec, indent=2))
    print(f"Scene spec written: {len(spec['scenes'])} scenes -> {out}")
    for s in spec["scenes"]:
        flags = []
        if s.get("location_conflicts"):  flags.append("CONFLICT")
        if s.get("action_issues"):       flags.append("ACTION-MISMATCH")
        if s.get("critique_gaps"):       flags.append(f"{len(s['critique_gaps'])} GAPS")
        print(f"  {s['scene_id']}  {s['scene_type']:20s}  conf={s['confidence']:.2f}  "
              f"participants={len(s['participants'])}  sources={len(s['evidence_sources'])}"
              + (f"  [{', '.join(flags)}]" if flags else ""))