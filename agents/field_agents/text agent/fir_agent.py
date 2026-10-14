#!/usr/bin/env python3
"""
ForenSynth - FIR Field Agent

Reads one First Information Report (English text - a filled-in form or a
plain narrative complaint) and produces two things:

1. An FIR RECORD (<case_id>_fir.json, and the `fir_records` table in the
   memory store) - stored separately from observations because the FIR is
   the case's partial ground truth. Every field carries a verbatim
   `evidence` quote, a `verified` flag and a confidence, and is split into:

     anchors - facts about the REPORT itself: FIR number, police station,
               date/time it was lodged, who the informant is. Reliable.
     claims  - what the informant ALLEGES happened: incident time and place,
               offence, accused, victims, witnesses, property, weapons.
               These are the hypothesis the rest of the pipeline tests
               against independent evidence - not facts.

2. FIR-DERIVED OBSERVATIONS (<case_id>__fir_obs_only.json) - the
   informant's narrative is itself evidence, so it is run through the same
   extractor as every other text (text_agent.extract_observations) with
   `source: "fir"` and `fir_derived: true`. Anything downstream that counts
   corroboration for an FIR claim must skip fir_derived observations, or
   the FIR ends up confirming itself.

People extracted as accused/victims/witnesses whose evidence quote can't be
found in the FIR are DROPPED (listed under `dropped`) - a hallucinated
accused is the worst failure this agent could have. Scalar fields with an
unverified quote are kept but flagged `verified: false` and their
confidence halved.

Usage:
    python agents/field_agent/fir_agent.py --case-id CASE_FIR_001 --input samples/fir/FIR_001_chain_snatching.txt
    python agents/field_agent/fir_agent.py --case-id CASE_FIR_001 --input fir.txt --db ./forensynth.db
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[3] / "memory"))
from llm_extract import (GROQ_MODEL, GroqJSON, clamp_conf, strip_honorific,  # noqa: E402
                         ungrounded_words, verify_quote)
from text_agent import _doc, _parse_dt, build_case, extract_observations  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
log = logging.getLogger("forensynth.fir_agent")

ANCHOR_FIELDS = ("fir_number", "police_station", "district", "date_of_report")
CLAIM_SCALARS = ("incident_location", "offence", "description")
PERSON_LISTS = ("accused", "victims", "witnesses")
ITEM_LISTS = ("property", "weapons")

_SYSTEM_PROMPT = """You are extracting the fields of an Indian First Information Report (FIR) /
police complaint written in English. It may be a filled-in FIR form or a free narrative.

Rules:
- Extract only what the text states. Never guess names, dates or places. Use null / [] when absent.
- EVERY field you fill must have "evidence": a short VERBATIM quote (max ~250 chars) copied
  exactly from the text that supports it.
- Dates/times as ISO-8601 "YYYY-MM-DDTHH:MM:SS" (date only -> "YYYY-MM-DDT00:00:00").
  Indian forms usually write dates as DD/MM/YYYY.
- "date_of_report" = when the FIR was lodged/registered (not when the crime happened).
- "incident_time": the alleged time of OCCURRENCE.
  "start"/"end" are the BOUNDS of when it could have happened: a single stated time -> same value
  for both; a range ("between 9 and 9:30 pm") -> both ends. If the offence was not witnessed but
  DISCOVERED later, start = the last time things were known to be normal and end = the time of
  discovery (e.g. house locked 19/04 06:00, found broken into 21/04 22:30 -> start 19/04 06:00,
  end 21/04 22:30). "best_estimate" = a narrower time the text points to inside that window (e.g. a
  watchman's sighting at 2 am), else null. Resolve weekday names and "yesterday"/"last night"
  against dates stated elsewhere in the text.
- "offence": REQUIRED whenever any crime is described - short crime type in plain words
  (e.g. "chain snatching", "house burglary"); evidence = the phrase that describes the act.
  "legal_sections": sections/acts exactly as cited (e.g. "BNS 304(2)", "IPC 379"), else [].
- "description": 1-3 sentence neutral summary of the allegation (third person). Its evidence is
  the single most central quote.
- "informant": the person lodging the report. "relation_to_incident": victim | witness |
  relative_of_victim | other.
- People in accused / victims / witnesses: one entry per person - "two men" is TWO entries, even if
  they are only described together. Unknown accused still get an entry with name null and whatever
  description is given ("man in red helmet"). A "description" uses ONLY details the text states
  about that person - never add appearance, build or age that isn't written. Include the
  informant in victims or witnesses too if they are one, with "is_informant": true.
- "confidence" 0-1 per field: how clearly and specifically the text states it.

Return ONLY this JSON object:
{"fir_number": F, "police_station": F, "district": F, "date_of_report": F,
 "informant": {"name": ..., "father_or_husband_name": ..., "age": ..., "address": ...,
               "phone": ..., "relation_to_incident": ..., "evidence": ..., "confidence": ...},
 "incident_time": {"start": ..., "end": ..., "best_estimate": ..., "evidence": ..., "confidence": ...},
 "incident_location": F, "offence": F,
 "legal_sections": {"value": [...], "evidence": ..., "confidence": ...},
 "description": F,
 "accused":   [P], "victims": [P], "witnesses": [P],
 "property":  [{"item": ..., "value": ..., "evidence": ..., "confidence": ...}],
 "weapons":   [{"item": ..., "evidence": ..., "confidence": ...}],
 "delay_in_reporting": {"value": ..., "evidence": ..., "confidence": ...}}
where F = {"value": ..., "evidence": ..., "confidence": ...}
and   P = {"name": ..., "description": ..., "address": ..., "is_informant": false,
           "evidence": ..., "confidence": ...}"""


# =============================================================================
# Regex fallback (no LLM) - labelled form fields only, never the narrative
# =============================================================================

_LABELS = {
    "fir_number": r"FIR\s*(?:No\.?|Number)\s*[:\-.]?\s*([A-Za-z0-9/\-]+)",
    # form fields often share a line, separated by 2+ spaces - stop there
    "police_station": r"(?:Police\s*Station|P\.?\s*S\.?)\s*[:\-.]\s*([^\n,;]+?)(?:\s{2,}|$)",
    "district": r"District\s*[:\-.]\s*([^\n,;]+?)(?:\s{2,}|$)",
    "date_of_report": r"(?:Date\s*(?:and\s*Time\s*)?of\s*(?:Report|Registration|FIR)|Date\s*of\s*information\s*received)"
                      r"[^:\n]*[:\-]\s*([^\n]+)",
}


def _regex_fallback(text: str) -> dict:
    out: dict = {}
    for field, pat in _LABELS.items():
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            out[field] = {"value": m.group(1).strip(), "evidence": m.group(0).strip(), "confidence": 0.7}
    # "Complainant / Informant:" heading followed (same or next lines) by "Name: ..."
    m = re.search(r"(?:Complainant|Informant)[^\n]*?\n?(?:[^\n]*\n){0,2}?[^\n]*?\bName\s*[:\-]\s*([^\n,]+)",
                  text, flags=re.IGNORECASE)
    if m:
        out["informant"] = {"name": m.group(1).strip(), "evidence": m.group(1).strip(), "confidence": 0.7}
    return out


# =============================================================================
# Validation
# =============================================================================

def _field(raw, text: str, dropped: list, name: str, as_datetime: bool = False) -> dict | None:
    if isinstance(raw, (str, int, float, list)):
        raw = {"value": raw}   # model sometimes skips the {value, evidence} wrapper
    if not isinstance(raw, dict) or raw.get("value") in (None, "", []):
        return None
    ev = str(raw.get("evidence") or "").strip()
    ok = verify_quote(ev, text)
    value = raw["value"]
    if as_datetime:
        dt = _parse_dt(value)
        value = dt.isoformat() if dt else str(value)
    conf = clamp_conf(raw.get("confidence"), 0.6)
    if not ok:
        dropped.append({"field": name, "reason": "evidence quote not found in FIR text (kept, flagged)"})
    return {"value": value, "evidence": ev, "verified": ok, "confidence": conf if ok else round(conf / 2, 3)}


def _people(raw_list, text: str, dropped: list, name: str) -> list[dict]:
    out = []
    for p in raw_list or []:
        if not isinstance(p, dict):
            continue
        ev = str(p.get("evidence") or "").strip()
        label = p.get("name") or p.get("description") or "?"
        if not verify_quote(ev, text):
            dropped.append({"field": name, "entry": label, "reason": "evidence quote not found in FIR text (dropped)"})
            continue
        desc = p.get("description")
        invented = ungrounded_words(desc, text)
        if invented:
            # Description isn't a verbatim quote, so check its words: any
            # detail not in the FIR (e.g. "tall") -> fall back to the quote.
            dropped.append({"field": name, "entry": label,
                            "reason": f"description had details not in FIR {invented} - replaced by evidence quote"})
            desc = ev
        out.append({
            "name": strip_honorific(str(p["name"])) if p.get("name") else None,
            "description": desc,
            "address": p.get("address"),
            "is_informant": bool(p.get("is_informant")),
            "evidence": ev, "verified": True,
            "confidence": clamp_conf(p.get("confidence"), 0.6),
        })
    return out


def _items(raw_list, text: str, dropped: list, name: str) -> list[dict]:
    out = []
    for it in raw_list or []:
        if not isinstance(it, dict) or not it.get("item"):
            continue
        ev = str(it.get("evidence") or "").strip()
        if not verify_quote(ev, text):
            dropped.append({"field": name, "entry": it.get("item"), "reason": "evidence quote not found (dropped)"})
            continue
        out.append({**{k: it.get(k) for k in ("item", "value")}, "evidence": ev, "verified": True,
                    "confidence": clamp_conf(it.get("confidence"), 0.6)})
    return out


def build_fir_record(case_id: str, text: str, raw: dict, extractor: str, source_file: str) -> dict:
    dropped: list = []
    anchors = {f: _field(raw.get(f), text, dropped, f, as_datetime=(f == "date_of_report")) for f in ANCHOR_FIELDS}

    inf = raw.get("informant")
    if isinstance(inf, dict) and inf.get("name"):
        ok = verify_quote(inf.get("evidence"), text)
        if not ok:
            dropped.append({"field": "informant", "reason": "evidence quote not found in FIR text (kept, flagged)"})
        conf = clamp_conf(inf.get("confidence"), 0.6)
        anchors["informant"] = {
            **{k: inf.get(k) for k in ("father_or_husband_name", "age", "address", "phone",
                                       "relation_to_incident")},
            "name": strip_honorific(inf["name"]),
            "evidence": str(inf.get("evidence") or ""), "verified": ok,
            "confidence": conf if ok else round(conf / 2, 3),
        }
    else:
        anchors["informant"] = None

    claims: dict = {}
    it = raw.get("incident_time")
    if isinstance(it, dict) and (it.get("start") or it.get("end")):
        start, end = _parse_dt(it.get("start")), _parse_dt(it.get("end"))
        start, end = start or end, end or start
        if start and end and start > end:
            start, end = end, start
        best = _parse_dt(it.get("best_estimate"))
        if best and start and not (start <= best <= end):
            best = None   # an estimate outside its own window is self-contradictory
        ok = verify_quote(it.get("evidence"), text)
        if not ok:
            dropped.append({"field": "incident_time", "reason": "evidence quote not found (kept, flagged)"})
        conf = clamp_conf(it.get("confidence"), 0.6)
        claims["incident_time"] = {
            "start": start.isoformat() if start else None, "end": end.isoformat() if end else None,
            "best_estimate": best.isoformat() if best else None,
            "evidence": str(it.get("evidence") or ""), "verified": ok,
            "confidence": conf if ok else round(conf / 2, 3),
        }
    else:
        claims["incident_time"] = None
    for f in CLAIM_SCALARS:
        claims[f] = _field(raw.get(f), text, dropped, f)
    ls = raw.get("legal_sections")
    claims["legal_sections"] = (_field(ls, text, dropped, "legal_sections")
                                if isinstance(ls, dict) and ls.get("value") else None)
    claims["delay_in_reporting"] = _field(raw.get("delay_in_reporting"), text, dropped, "delay_in_reporting")
    for f in PERSON_LISTS:
        claims[f] = _people(raw.get(f), text, dropped, f)
    for f in ITEM_LISTS:
        claims[f] = _items(raw.get(f), text, dropped, f)

    return {
        "case_id": case_id,
        "fir_id": f"{case_id}_FIR",
        "source_file": source_file,
        "extracted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "extractor": extractor,
        "language": "en",
        "anchors": anchors,
        "claims": claims,
        "dropped": dropped,
        "raw_text": text,
    }


# =============================================================================
# FIR record -> the case-level `fir` summary existing agents already read
# =============================================================================

def _val(f):
    return f.get("value") if isinstance(f, dict) else None


def fir_summary(record: dict) -> dict:
    """Compact `fir` block for cases.fir_json. Keeps every key ER already
    reads (crime_type, location, description, roles, time_window) so
    entity_resolution.py's role-count check works unchanged, plus the new
    informant / accused / incident_datetime fields agreed for the schema."""
    a, c = record["anchors"], record["claims"]
    roles = {"suspect": len(c["accused"]), "victim": len(c["victims"]), "witness": len(c["witnesses"])}
    inf = a.get("informant") or {}
    rel = (inf.get("relation_to_incident") or "").lower()
    listed_informant = any(p["is_informant"] for p in c["victims"] + c["witnesses"])
    if inf.get("name") and not listed_informant and rel in ("victim", "witness"):
        roles[rel] += 1
    it = c.get("incident_time") or {}
    return {
        "crime_type": _val(c["offence"]) or "unspecified",
        "location": _val(c["incident_location"]) or "unspecified",
        "description": _val(c["description"]) or "",
        "time_window": [0, 0],          # relative offsets are filled in by orchestrator.py on merge
        "roles": {k: v for k, v in roles.items() if v},
        "incident_datetime": it.get("start"),
        "incident_datetime_end": it.get("end"),
        "incident_best_estimate": it.get("best_estimate"),
        "informant": inf.get("name"),
        "accused": [p["name"] or p["description"] for p in c["accused"]],
        "fir_number": _val(a["fir_number"]),
        "police_station": _val(a["police_station"]),
        "date_of_report": _val(a["date_of_report"]),
        "fir_record_id": record["fir_id"],
    }


# =============================================================================
# Main entry
# =============================================================================

def process_fir(case_id: str, text: str, source_file: str, llm: GroqJSON) -> tuple[dict, dict]:
    raw = llm.ask(_SYSTEM_PROMPT, f"--- FIR TEXT START ---\n{text}\n--- FIR TEXT END ---", max_tokens=8192)
    extractor = GROQ_MODEL if raw is not None else "regex_fallback"
    if raw is None:
        log.warning("LLM unavailable - FIR record built from labelled form fields only.")
        raw = _regex_fallback(text)
    record = build_fir_record(case_id, text, raw, extractor, source_file)
    summary = fir_summary(record)

    # Informant narrative -> observations through the shared text extractor.
    # The FIR's own incident/report date is the reference for relative times.
    informant = summary.get("informant")
    ref = _parse_dt(summary.get("incident_datetime")) or _parse_dt(summary.get("date_of_report"))
    doc = _doc(f"{case_id}_FIR", text, author=informant, sent_at=_parse_dt(summary.get("date_of_report")),
               doc_type_hint="fir", source="fir", location=summary.get("location"))
    obs = extract_observations([doc], llm, reference_date=ref, source_override="fir")
    for o in obs:
        o["fir_derived"] = True
    if not obs:
        raise RuntimeError("No observations could be extracted from the FIR narrative.")
    case = build_case(f"{case_id}__fir", obs, "FIR-derived observations", fir=summary)
    return record, case


def save_fir_to_memory(record: dict, summary: dict, db_path: str) -> None:
    from memory_store import ForenSynthMemory
    mem = ForenSynthMemory(db_path)
    mem.save_fir_record(record, summary)
    log.info("Saved FIR record %s to memory store (%s)", record["fir_id"], mem.db_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="FIR text -> separate FIR record + FIR-derived observations")
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--input", required=True, type=Path, help="FIR as a .txt file (English)")
    ap.add_argument("--output", default="cases")
    ap.add_argument("--db", help="Also save the FIR record to this memory-store DB")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--model", default=GROQ_MODEL)
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"[ERROR] Missing FIR file: {args.input}")
    text = args.input.read_text(encoding="utf-8").strip()
    llm = GroqJSON(model=args.model, enabled=not args.no_llm, max_calls=10)
    record, case = process_fir(args.case_id, text, args.input.name, llm)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / f"{args.case_id}_fir.json"
    obs_path = out_dir / f"{args.case_id}__fir_obs_only.json"
    rec_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    obs_path.write_text(json.dumps(case, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.db:
        save_fir_to_memory(record, case["fir"], args.db)

    s = case["fir"]
    print(f"FIR {s['fir_number'] or '?'} @ {s['police_station'] or '?'} | offence: {s['crime_type']} | "
          f"incident: {s['incident_datetime'] or '?'} | informant: {s['informant'] or '?'}")
    print(f"accused: {s['accused'] or 'none named'} | roles: {s['roles']} | "
          f"dropped (unverified): {len(record['dropped'])} | extractor: {record['extractor']}")
    print(f"{len(case['observations'])} FIR-derived observations")
    print(f"Saved -> {rec_path}\n         {obs_path}")


if __name__ == "__main__":
    main()
