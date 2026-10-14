#!/usr/bin/env python3
"""
ForenSynth - Text Field Agent (generic, LLM-first)

Turns ANY text evidence - witness statements, chat/SMS exports, emails,
officer notes, social posts, the narrative part of an FIR - into a
ForenSynth obs_only case file, the same schema entity_resolution.py /
run_case.py already consume.

Pipeline per document:
  1. load       - files (.txt/.md/.eml/.json) or the Enron email corpus,
                  normalised to {text, author, sent_at, doc_type_hint, ...}
  2. extract    - LLM (Groq) reads the document and returns discrete,
                  forensically relevant events: who did/said what, when,
                  where, a role HINT, crime-relevant tags, and a verbatim
                  evidence quote for each
  3. verify     - every quote is checked against the source text
                  (llm_extract.verify_quote); events whose quote isn't
                  really there are dropped, never passed downstream
  4. bookkeep   - obs_id numbering, absolute timestamp resolution and
                  time_offset are plain code, not LLM

Role is deliberately NOT asserted: `role` stays "unknown" and the LLM's
judgement goes in `role_hint` / `role_hint_confidence` (decided with the
team: field agents give hints, ER/Timeline/Critique reason it out).
agents/field_agent/orchestrator.py --adopt-role-hints can promote confident
hints into `role` at intake if you want that.

With no GROQ_API_KEY (or --no-llm) the agent still runs: one observation
per paragraph/message, spaCy NER for mentioned people, heuristic
confidence - the old behaviour, clearly tagged `extraction: "fallback"`.

Usage:
    python agents/field_agent/text_agent.py --case-id CASE_TXT_001 --input samples/statement.txt
    python agents/field_agent/text_agent.py --case-id CASE_TXT_002 --input chats.json mail.eml
    python agents/field_agent/text_agent.py --case-id CASE_ENRON_001          # Enron pilot (default)
"""
from __future__ import annotations

import argparse
import email
import email.policy
import hashlib
import itertools
import json
import logging
import re
import sys
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_extract import GROQ_MODEL, GroqJSON, chunk_text, clamp_conf, strip_honorific, verify_quote  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
log = logging.getLogger("forensynth.text_agent")

ROLE_HINTS = {"suspect", "witness", "victim", "unknown"}
TIME_BASES = {"explicit", "relative", "document_date", "unknown"}
EVENT_TAGS = {
    "threat", "admission", "denial", "alibi", "weapon", "money", "injury",
    "property", "movement", "communication", "identification", "planning",
    "coercion", "contact",
}
DOC_TYPES = {
    "email", "chat", "sms", "witness_statement", "fir", "police_note",
    "social_post", "letter", "transcript", "other",
}
# Timing certainty feeds confidence lightly: an event pinned to an explicit
# time is more useful to Timeline than one that only inherits the date the
# document was written. Kept mild on purpose - content clarity dominates.
_TIME_BASIS_FACTOR = {"explicit": 1.0, "relative": 0.95, "document_date": 0.85, "unknown": 0.75}


# =============================================================================
# 1. Loading
# =============================================================================

def _to_naive(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


def _parse_dt(value) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return _to_naive(value)
    if isinstance(value, (int, float)):
        return _to_naive(datetime.fromtimestamp(value, tz=timezone.utc))
    s = str(value).strip()
    try:
        return _to_naive(datetime.fromisoformat(s.replace("Z", "+00:00")))
    except ValueError:
        pass
    try:
        return _to_naive(parsedate_to_datetime(s))   # RFC-2822 email dates
    except (TypeError, ValueError):
        pass
    for fmt in ("%d/%m/%Y %H:%M", "%d/%m/%Y", "%d-%m-%Y %H:%M", "%d-%m-%Y", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _doc(doc_id, text, author=None, sent_at=None, doc_type_hint=None, source=None, location=None) -> dict:
    return {
        "doc_id": doc_id, "text": text.strip(), "author": author,
        "sent_at": _parse_dt(sent_at), "doc_type_hint": doc_type_hint,
        "source": source or doc_type_hint or "text_document", "location": location,
    }


_QUOTE_MARKERS = [
    re.compile(r"-----Original Message-----", re.IGNORECASE),
    re.compile(r"^-{2,}\s*Forwarded by", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^On .{0,80} wrote:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^>.*", re.MULTILINE),
    # Outlook/Lotus Notes export style: "someone@x.com on 12/01/2000 09:57:30 AM"
    re.compile(r"^\S+@\S+\s+on\s+\d{1,2}/\d{1,2}/\d{2,4}", re.IGNORECASE | re.MULTILINE),
]


def strip_quoted_reply(body: str) -> str:
    """Keep only the new content of a reply, dropping quoted thread history."""
    cut = len(body)
    for pat in _QUOTE_MARKERS:
        m = pat.search(body)
        if m:
            cut = min(cut, m.start())
    return body[:cut].strip()


def alias_for(address: str) -> str:
    """Stable pseudonymous alias for an email address (same 'email_<hash>'
    convention as before) so ER treats repeat senders as one entity."""
    h = hashlib.sha1(address.strip().lower().encode()).hexdigest()[:6]
    return f"email_{h}"


def _is_conversation(recs: list[dict]) -> bool:
    """Short timestamped messages with senders (chat/SMS export) - read
    them together, since a message like "done. she fell down" only means
    something next to the ones before it."""
    return (len(recs) >= 2 and all(r["author"] and r["sent_at"] for r in recs)
            and sum(len(r["text"]) for r in recs) / len(recs) < 600)


def _conversation_doc(stem: str, recs: list[dict]) -> dict:
    recs = sorted(recs, key=lambda r: r["sent_at"])
    lines = [f"[{r['sent_at'].strftime('%Y-%m-%d %H:%M')}] {r['author']}: {r['text']}" for r in recs]
    d = _doc(stem, "\n".join(lines), author=None, sent_at=recs[0]["sent_at"],
             doc_type_hint=recs[0]["doc_type"] or "chat", source=recs[0]["source"] or "chat",
             location=recs[0]["location"])
    d["is_transcript"] = True
    d["messages"] = [(r["sent_at"], r["author"], r["text"]) for r in recs]
    return d


def _records_to_docs(stem: str, recs: list[dict]) -> list[dict]:
    if _is_conversation(recs):
        return [_conversation_doc(stem, recs)]
    return [_doc(f"{stem}#{i}", r["text"], author=r["author"], sent_at=r["sent_at"],
                 doc_type_hint=r["doc_type"], source=r["source"], location=r["location"])
            for i, r in enumerate(recs, start=1)]


# Chat exports saved as plain text: WhatsApp Android "14/03/2025, 19:58 - Name: msg",
# WhatsApp iOS "[14/03/2025, 19:58:00] Name: msg", generic "[2025-03-14 19:58] Name: msg";
# 12-hour times ("7:58 pm") too. Numeric dates are read day-first (Indian exports).
_CHAT_PREFIX = re.compile(
    r"^\[?(\d{1,4}[/.-]\d{1,2}[/.-]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]\.?m\.?)?)\]?\s*(?:-\s+)?",
    re.IGNORECASE)
_CHAT_DATE_FMTS = ("%d/%m/%Y", "%d/%m/%y", "%Y/%m/%d")
_CHAT_TIME_FMTS = ("%H:%M:%S", "%H:%M", "%I:%M:%S %p", "%I:%M %p", "%I:%M:%S%p", "%I:%M%p")


def _chat_dt(date: str, time_: str) -> datetime | None:
    date = re.sub(r"[.-]", "/", date)
    time_ = re.sub(r"\s+", " ", time_.replace(".", "")).strip().upper()
    for df in _CHAT_DATE_FMTS:
        for tf in _CHAT_TIME_FMTS:
            try:
                return datetime.strptime(f"{date} {time_}", f"{df} {tf}")
            except ValueError:
                continue
    return None


def parse_chat_txt(text: str) -> list[dict] | None:
    """Plain-text chat export -> message records (same shape the .json
    loader builds), or None if the text doesn't look like one. Lines without
    a timestamp continue the previous message; timestamped lines without a
    "Sender: " part are system notices and are skipped."""
    text = text.replace("‎", "").replace(" ", " ")
    recs: list[dict] = []
    stamped = total = 0
    skipping = False
    for line in text.splitlines():
        if not line.strip():
            continue
        total += 1
        m = _CHAT_PREFIX.match(line)
        if not m:
            if recs and not skipping:
                recs[-1]["text"] += "\n" + line.strip()
            continue
        stamped += 1
        sender, sep, msg = line[m.end():].partition(": ")
        skipping = not sep or not sender.strip() or len(sender) > 60
        if skipping:
            continue
        recs.append({"text": msg.strip(), "author": sender.strip(), "sent_at": _chat_dt(m.group(1), m.group(2)),
                     "doc_type": "chat", "source": "chat_export", "location": None})
    if len(recs) >= 2 and stamped * 2 >= total:
        return [r for r in recs if r["text"]]
    return None


def load_file(path: Path) -> list[dict]:
    """.txt/.md -> one document; .eml -> one email; .json -> a list of
    {text, author?, timestamp?, doc_type?, location?} records (or
    {"documents": [...]}) - the shape a chat/SMS export is easy to put in."""
    suffix = path.suffix.lower()
    if suffix == ".eml":
        msg = email.message_from_bytes(path.read_bytes(), policy=email.policy.default)
        body = msg.get_body(preferencelist=("plain",))
        text = strip_quoted_reply(body.get_content() if body else "")
        sender = str(msg.get("From", "") or "")
        return [_doc(path.stem, f"Subject: {msg.get('Subject', '')}\n\n{text}",
                     author=alias_for(sender) if sender else None, sent_at=msg.get("Date"),
                     doc_type_hint="email", source="email")]
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data.get("documents", data) if isinstance(data, dict) else data
        if isinstance(records, dict):
            records = [records]
        recs = []
        for r in records:
            text = r.get("text") or r.get("content") or r.get("body") or r.get("message") or ""
            if str(text).strip():
                recs.append({"text": str(text).strip(), "author": r.get("author") or r.get("from") or r.get("sender"),
                             "sent_at": _parse_dt(r.get("timestamp") or r.get("date") or r.get("sent_at")),
                             "doc_type": r.get("doc_type"), "source": r.get("source"), "location": r.get("location")})
        return _records_to_docs(path.stem, recs)
    text = path.read_text(encoding="utf-8-sig")
    recs = parse_chat_txt(text) if suffix == ".txt" else None
    if recs:
        return _records_to_docs(path.stem, recs)
    return [_doc(path.stem, text)]


# --- Enron pilot loader (kept so `orchestrator.py --run-agents` still works with no inputs) ---

VALID_YEAR_RANGE = (1998, 2003)   # Enron header parsing yields some garbage years
_AUTOMATED_SENDER = re.compile(
    r"noreply|no-reply|postmaster|mailer-daemon|webmaster|notification|"
    r"alert|autoreply|donotreply|distribution|announce|listserv", re.IGNORECASE)
_AUTOMATED_SUBJECT = re.compile(
    r"\bnotice\b|\bposted\b|\breport\b|\bconfirmation\b|\balert\b|"
    r"\bdigest\b|\bbulletin\b|\bauto[- ]?generated\b|\bmentions\b|\bclippings?\b", re.IGNORECASE)
_WIRE_SERVICE_BODY = re.compile(
    r"\bReuters\b|\bAFX News\b|\bDow Jones\b|\bBloomberg\b|\bPR Newswire\b|"
    r"\bBusiness Wire\b|\bCopyright \d{4}\b|\bWall Street Journal\b|\bAssociated Press\b")
_CONVERSATIONAL_MARKERS = re.compile(r"\b(i|you|we|your|my)\b|\?|'(m|re|ve|ll|d|t)\b", re.IGNORECASE)


def _normalize_subject(subject: str) -> str:
    return re.sub(r"^(re|fwd?):\s*", "", subject.strip(), flags=re.IGNORECASE).strip().lower()


def find_thread(rows: list, min_messages: int = 4, max_messages: int = 8) -> list:
    """Pick the most conversational multi-sender thread from an Enron slice
    (skips broadcasts, automated notices and forwarded wire articles)."""
    threads: dict[str, list] = {}
    seen_copies: set = set()
    for row in rows:
        if not row.get("from") or not row.get("body") or not row.get("date"):
            continue
        if (_AUTOMATED_SENDER.search(row["from"]) or _AUTOMATED_SUBJECT.search(row.get("subject") or "")
                or _WIRE_SERVICE_BODY.search(row["body"][:1000])):
            continue
        dt = _parse_dt(row["date"])
        if dt is None or not (VALID_YEAR_RANGE[0] <= dt.year <= VALID_YEAR_RANGE[1]):
            continue
        copy_key = (row["from"], str(row["date"]), row["body"])   # same mail, many mailboxes
        if copy_key in seen_copies:
            continue
        seen_copies.add(copy_key)
        key = _normalize_subject(row.get("subject") or "")
        if key:
            threads.setdefault(key, []).append({**row, "_dt": dt})

    def distinct_bodies(msgs):
        return len({re.sub(r"\s+", " ", m["body"][:200]).strip() for m in msgs})

    candidates = [m for m in threads.values()
                  if min_messages <= len(m) <= max_messages
                  and len({x["from"] for x in m}) >= 2
                  and distinct_bodies(m) >= max(2, len(m) // 2)]
    if not candidates:
        raise RuntimeError("No suitable thread found - re-run with a larger --sample-size.")
    candidates.sort(key=lambda msgs: (
        sum(len(_CONVERSATIONAL_MARKERS.findall(strip_quoted_reply(m["body"]))) for m in msgs),
        sum(1 for m in msgs if len(strip_quoted_reply(m["body"])) >= 40)), reverse=True)
    return candidates[0]


def load_enron(sample_size: int) -> list[dict]:
    from datasets import load_dataset
    print(f"Streaming corbt/enron-emails, scanning up to {sample_size} rows...")
    ds = load_dataset("corbt/enron-emails", split="train", streaming=True)
    thread = sorted(find_thread(list(itertools.islice(ds, sample_size))), key=lambda r: r["_dt"])
    docs = []
    for i, msg in enumerate(thread, start=1):
        body = strip_quoted_reply(msg["body"])
        if body:
            docs.append(_doc(f"enron#{i}", f"Subject: {msg.get('subject', '')}\n\n{body}",
                             author=alias_for(msg["from"]), sent_at=msg["_dt"],
                             doc_type_hint="email", source="email_log",
                             location="remote - email server log"))
    print(f"Thread: '{thread[0].get('subject', '').strip()}' ({len(docs)} messages)")
    return docs


# =============================================================================
# 2. LLM extraction
# =============================================================================

_SYSTEM_PROMPT = f"""You are a forensic evidence analyst. You read one piece of text evidence
(it may be an email, chat log, SMS, witness statement, FIR / police complaint, officer's note,
social media post, letter or transcript) and extract every discrete, forensically relevant
event or statement in it as a structured observation.

Rules:
- Extract only what the text says. Never invent people, times, places or actions.
- Every observation MUST include "evidence": a short VERBATIM quote (copied exactly, max ~250
  characters) from the text that supports it. If you cannot quote it, do not output it.
- One observation = one action, statement, or perceived fact (split compound sentences).
- "actor" is who did / said the thing. Use one consistent canonical name per person across the
  document (list them in "people"). If the author/narrator ("I", "me") is the actor and an author
  identifier is given, use that identifier exactly. Unnamed people get a stable descriptive label,
  e.g. "unknown_man_1" (and describe them in "people").
  In a message transcript (lines like "[YYYY-MM-DD HH:MM] Sender: text") the sender is the actor
  of what they say, and that line's bracketed time is when the message was sent.
- "content": one neutral sentence describing the event, third person, past tense. For a chat
  message that only reports or plans something, describe the message (e.g. "Rakesh messaged that
  ..."), and set event_time to the message time.
- "role_hint": suspect | witness | victim | unknown - your best reading of the actor's role in the
  alleged incident, with "role_hint_confidence" 0-1 and a brief "role_reason". This is a HINT for
  investigators, not a finding: use "unknown" when the text does not support a call.
- "event_time": ISO-8601 "YYYY-MM-DDTHH:MM:SS" if the time can be determined, else null.
  Resolve relative expressions ("around 9 pm", "ten minutes later", "yesterday", "on Sunday night")
  against dates stated elsewhere in the text, else the document date / reference date. If the
  day and month are given but not the year ("Friday 14th March"), take the year from other dates
  in the text or the reference date. "time_text": the verbatim time expression you used (e.g.
  "around 8:30 or 8:40 in the night"), else null. "time_basis": explicit (stated outright) |
  relative (computed from another time) | document_date (only the document's own date is known)
  | unknown.
- "location": where it happened, as written, else null.
- "tags": any of {sorted(EVENT_TAGS)}.
- "confidence" 0-1: how clear, specific and first-hand the statement is (hearsay, vague or
  hedged text scores lower; precise first-hand detail scores higher).
- Skip greetings, signatures, boilerplate and anything with no bearing on what happened.

Return ONLY a JSON object:
{{"doc_type": one of {sorted(DOC_TYPES)},
  "people": [{{"name": "...", "aliases": ["..."], "description": "..."}}],
  "observations": [{{"actor": "...", "role_hint": "...", "role_hint_confidence": 0.0,
     "role_reason": "...", "content": "...", "evidence": "...", "event_time": null,
     "time_text": null, "time_basis": "...", "location": null, "mentioned_people": ["..."], "tags": [],
     "confidence": 0.0}}]}}"""


def _user_prompt(doc: dict, chunk: str, part: str, reference_date: datetime | None) -> str:
    meta = {
        "document_id": doc["doc_id"],
        "author_identifier": doc.get("author"),
        "document_date": doc["sent_at"].isoformat() if doc.get("sent_at") else None,
        "reference_date": reference_date.isoformat() if reference_date else None,
        "declared_type": doc.get("doc_type_hint"),
        "part": part,
    }
    return f"Document metadata: {json.dumps(meta)}\n\n--- TEXT START ---\n{chunk}\n--- TEXT END ---"


def _alias_map(people: list) -> dict[str, str]:
    """name/alias (lowercased) -> canonical name, from the LLM's people list."""
    out: dict[str, str] = {}
    for p in people or []:
        if not isinstance(p, dict) or not p.get("name"):
            continue
        canon = str(p["name"]).strip()
        for a in [canon, *(p.get("aliases") or [])]:
            if a and str(a).strip():
                out[str(a).strip().lower()] = canon
    return out


_PM_WORDS = re.compile(r"\b(p\.?m\.?|night|evening|afternoon|tonight)\b", re.IGNORECASE)
_AM_WORDS = re.compile(r"\b(a\.?m\.?|morning|dawn)\b", re.IGNORECASE)


def _fix_meridiem(ts: datetime | None, time_text: str | None) -> datetime | None:
    """Deterministic AM/PM check against the verbatim time phrase - the
    model occasionally writes 08:30 for "8:30 in the night". Only fires
    when the phrase is unambiguous (pm-words xor am-words)."""
    if ts is None or not time_text:
        return ts
    pm, am = bool(_PM_WORDS.search(time_text)), bool(_AM_WORDS.search(time_text))
    if pm and not am and 1 <= ts.hour < 12 and not re.search(r"\b(late|mid)night\b", time_text, re.I):
        return ts.replace(hour=ts.hour + 12)
    if am and not pm and ts.hour >= 12:
        return ts.replace(hour=ts.hour - 12)
    return ts


def _clean_llm_obs(raw: dict, doc: dict, chunk: str, aliases: dict[str, str]) -> dict | None:
    """Validate one LLM observation. Returns None (dropped) if its evidence
    quote can't be found in the source chunk."""
    evidence = str(raw.get("evidence") or "").strip()
    if not verify_quote(evidence, chunk):
        log.info("  dropped unverified event: %.80s", raw.get("content"))
        return None
    actor = str(raw.get("actor") or "").strip() or doc.get("author") or "unknown_person"
    actor = strip_honorific(aliases.get(actor.lower(), actor))
    role = str(raw.get("role_hint") or "unknown").strip().lower()
    basis = str(raw.get("time_basis") or "unknown").strip().lower()
    return {
        "actor": actor,
        "role_hint": role if role in ROLE_HINTS else "unknown",
        "role_hint_confidence": clamp_conf(raw.get("role_hint_confidence"), 0.0),
        "role_reason": str(raw.get("role_reason") or "")[:300],
        "content": str(raw.get("content") or evidence)[:2000],
        "evidence": evidence[:500],
        "event_time": _fix_meridiem(_parse_dt(raw.get("event_time")), raw.get("time_text")),
        "time_basis": basis if basis in TIME_BASES else "unknown",
        "location": raw.get("location") or None,
        "mentioned_people": [strip_honorific(aliases.get(str(m).strip().lower(), str(m).strip()))
                             for m in (raw.get("mentioned_people") or []) if str(m).strip()],
        "tags": sorted({str(t).lower() for t in (raw.get("tags") or [])} & EVENT_TAGS),
        "confidence": clamp_conf(raw.get("confidence"), 0.6),
    }


def extract_llm(doc: dict, llm: GroqJSON, reference_date: datetime | None) -> tuple[list[dict], str | None] | None:
    """LLM path for one document. None means "LLM unavailable/failed, use
    fallback"; an empty list means the LLM read it and found nothing relevant."""
    chunks = chunk_text(doc["text"])
    events, doc_type = [], None
    for i, chunk in enumerate(chunks, start=1):
        part = f"{i} of {len(chunks)}"
        result = llm.ask(_SYSTEM_PROMPT, _user_prompt(doc, chunk, part, reference_date), max_tokens=8192)
        if result is None:
            return None
        doc_type = doc_type or str(result.get("doc_type") or "").lower() or None
        aliases = _alias_map(result.get("people"))
        raw_obs = result.get("observations") or []
        kept = [c for c in (_clean_llm_obs(r, doc, chunk, aliases) for r in raw_obs if isinstance(r, dict)) if c]
        log.info("  %s part %s: %d event(s) extracted, %d verified", doc["doc_id"], part, len(raw_obs), len(kept))
        events.extend(kept)
    return events, (doc_type if doc_type in DOC_TYPES else None)


# =============================================================================
# 2b. No-LLM fallback
# =============================================================================

_NLP = None


def _people_ner(text: str) -> list[str]:
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm")
        except Exception as exc:   # spaCy or model missing - fallback still runs
            log.warning("spaCy unavailable (%s) - no NER in fallback mode.", exc)
            _NLP = False
    if not _NLP:
        return []
    seen, people = set(), []
    for ent in _NLP(text[:3000]).ents:
        name = ent.text.strip()
        if ent.label_ == "PERSON" and len(name) > 1 and name.lower() not in seen:
            seen.add(name.lower())
            people.append(name)
    return people


def _heuristic_confidence(content: str) -> float:
    n = len(content.split())
    return 0.6 if n < 6 else 0.7 if n < 20 else 0.75


def attach_message_times(doc: dict, events: list[dict]) -> None:
    """Chat transcript: when the LLM gives an event no time of its own (or
    just the document date, i.e. the FIRST message's time), use the time of
    the message its evidence quote comes from - otherwise a whole chat
    collapses onto one timestamp."""
    for ev in events:
        if ev["event_time"] is not None and ev["time_basis"] != "document_date":
            continue
        for sent_at, _author, msg in doc["messages"]:
            if verify_quote(ev["evidence"], msg):
                ev["event_time"], ev["time_basis"] = sent_at, "explicit"
                break


def extract_fallback(doc: dict) -> list[dict]:
    """One observation per paragraph (per message for a transcript) - no
    understanding, just structure."""
    out = []
    if doc.get("is_transcript"):
        for sent_at, author, msg in doc["messages"]:
            out.append({
                "actor": author, "role_hint": "unknown", "role_hint_confidence": 0.0,
                "role_reason": "", "content": msg, "evidence": msg,
                "event_time": sent_at, "time_basis": "explicit",
                "location": None, "mentioned_people": _people_ner(msg), "tags": [],
                "confidence": _heuristic_confidence(msg),
            })
        return out
    paras = [p.strip() for p in re.split(r"\n\s*\n", doc["text"]) if len(p.strip()) >= 20]
    for p in paras or [doc["text"]]:
        people = _people_ner(p)
        out.append({
            "actor": doc.get("author") or (people[0] if people else "unknown_author"),
            "role_hint": "unknown", "role_hint_confidence": 0.0, "role_reason": "",
            "content": p[:2000], "evidence": p[:500], "event_time": None,
            "time_basis": "document_date" if doc.get("sent_at") else "unknown",
            "location": None, "mentioned_people": people, "tags": [],
            "confidence": _heuristic_confidence(p),
        })
    return out


# =============================================================================
# 3. Assemble obs_only case
# =============================================================================

def extract_observations(docs: list[dict], llm: GroqJSON | None,
                         reference_date: datetime | None = None,
                         source_override: str | None = None) -> list[dict]:
    """Documents -> observation dicts (obs_id/time_offset assigned by build_case).
    Exposed separately so fir_agent.py reuses the exact same extractor."""
    observations, order = [], 0
    for doc in docs:
        extraction = "fallback"
        events, doc_type = None, None
        if llm is not None and llm.available:
            res = extract_llm(doc, llm, reference_date)
            if res is not None:
                events, doc_type = res
                extraction = "llm"
        if events is None:
            events = extract_fallback(doc)
        elif doc.get("is_transcript"):
            attach_message_times(doc, events)
        source = source_override or doc_type or doc.get("doc_type_hint") or doc["source"]

        for ev in events:
            order += 1
            ts, basis = ev["event_time"], ev["time_basis"]
            if ts is None:
                if doc.get("sent_at"):
                    ts, basis = doc["sent_at"], "document_date"
                else:
                    # No usable date anywhere: leave the timestamp empty - the
                    # pipeline's own "missing" (shared.parse_timestamp -> 0,
                    # Timeline scores it neutral) - rather than invent one.
                    basis = "unknown"
            observations.append({
                "entity": ev["actor"],
                "role": "unknown",
                "role_hint": ev["role_hint"],
                "role_hint_confidence": ev["role_hint_confidence"],
                "role_reason": ev["role_reason"],
                "modality": "text",
                "source": source,
                "source_doc": doc["doc_id"],
                "reported_by": doc.get("author"),
                "location": ev["location"] or doc.get("location") or "unspecified",
                "content": ev["content"],
                "evidence": ev["evidence"],
                "mentioned_entities": ev["mentioned_people"],
                "tags": ev["tags"],
                "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S") if ts else "",
                "time_basis": basis,
                "confidence": round(ev["confidence"] * _TIME_BASIS_FACTOR[basis], 3),
                "extraction": extraction,
                "_order": order,
            })
    return observations


def build_case(case_id: str, observations: list[dict], description: str, fir: dict | None = None) -> dict:
    if not observations:
        raise RuntimeError("No observations extracted from the input text.")
    # Dated observations chronologically, undated ones after them in reading order.
    observations = sorted(observations, key=lambda o: (not o["timestamp"], o["timestamp"], o["_order"]))
    dated = [o for o in observations if o["timestamp"]]
    base = datetime.fromisoformat(dated[0]["timestamp"]) if dated else None
    out = []
    for i, o in enumerate(observations, start=1):
        o = {k: v for k, v in o.items() if k != "_order"}
        o["obs_id"] = f"O{i}"
        o["time_offset"] = (int((datetime.fromisoformat(o["timestamp"]) - base).total_seconds())
                            if o["timestamp"] else 0)
        out.append(o)
    return {
        "case_id": case_id,
        "domain": "Text_Evidence",
        "template": "LLM_Text_Extraction",
        "fir": fir or {
            "crime_type": "unknown - no FIR attached",
            "location": "unspecified",
            "time_window": [0, max(o["time_offset"] for o in out)],
            "description": description,
            "roles": {},
        },
        "observations": out,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Any text evidence -> ForenSynth obs_only case")
    ap.add_argument("--case-id", default="CASE_ENRON_001")
    ap.add_argument("--input", nargs="+", type=Path,
                    help="Text evidence files (.txt/.md/.eml/.json). Omit to run the Enron pilot.")
    ap.add_argument("--output", default="cases")
    ap.add_argument("--reference-date", help="ISO date to resolve relative times against when a "
                                             "document has no date of its own (e.g. the FIR incident date)")
    ap.add_argument("--sample-size", type=int, default=5000, help="Enron rows to scan")
    ap.add_argument("--no-llm", action="store_true", help="Skip the LLM, use the paragraph fallback")
    ap.add_argument("--model", default=GROQ_MODEL)
    args = ap.parse_args()

    if args.input:
        missing = [p for p in args.input if not p.exists()]
        if missing:
            sys.exit(f"[ERROR] Missing input file(s): {missing}")
        docs = [d for p in args.input for d in load_file(p)]
        label = ", ".join(p.name for p in args.input)
    else:
        docs = load_enron(args.sample_size)
        label = "Enron email thread"

    llm = GroqJSON(model=args.model, enabled=not args.no_llm, max_calls=max(20, 2 * len(docs)))
    ref = _parse_dt(args.reference_date) if args.reference_date else None
    obs = extract_observations(docs, llm, reference_date=ref)
    case = build_case(args.case_id, obs, f"Observations extracted from {len(docs)} text document(s): {label}.")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.case_id}_obs_only.json"
    out_path.write_text(json.dumps(case, indent=2, ensure_ascii=False), encoding="utf-8")

    n = len(case["observations"])
    by_mode = sum(1 for o in case["observations"] if o["extraction"] == "llm")
    print(f"{n} observations from {len(docs)} document(s) "
          f"({by_mode} LLM-extracted, {n - by_mode} fallback, {llm.calls_made} LLM call(s)), "
          f"{len({o['entity'] for o in case['observations']})} distinct actors")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
