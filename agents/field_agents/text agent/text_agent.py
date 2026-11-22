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

FIR: one input can be the case's First Information Report (detected by the
FIR form header or a complaint letter to a police station, or marked with --fir). It is handled
by the same agent, in addition to the steps above:
  * its fields go into a separate FIR RECORD (<case_id>_fir.json; the
    `fir_records` table) - the case's PARTIAL ground truth: anchors (FIR no.,
    station, report date, informant) are reliable, claims (incident time and
    place, offence, accused, witnesses) are hypotheses to test
  * its summary becomes the case's `fir` block
  * its narrative becomes observations tagged source "fir", fir_derived: true
    (never count those as corroborating the FIR)
  * its incident date is the reference date for every other document

With no GROQ_API_KEY (or --no-llm) the agent still runs: one observation
per paragraph/message, spaCy NER for mentioned people, heuristic
confidence - the old behaviour, clearly tagged `extraction: "fallback"`;
the FIR record is then built from labelled form fields only.

Usage (run from the repo root):
    python "agents/field_agents/text agent/text_agent.py" --case-id CASE_FIR_001         --input samples/fir/FIR_001_chain_snatching.txt samples/fir/FIR_001_witness_statement.txt                 samples/fir/FIR_001_chat.json
    python "agents/field_agents/text agent/text_agent.py" --case-id CASE_FIR_002         --input samples/fir/FIR_002_house_burglary.txt                         # FIR only
    python "agents/field_agents/text agent/text_agent.py" --case-id CASE_TXT_002 --input chats.json mail.eml
    python "agents/field_agents/text agent/text_agent.py" --case-id CASE_ENRON_001   # Enron pilot (no inputs)
"""
from __future__ import annotations

import argparse
import email
import email.policy
import hashlib
import itertools
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_extract import (GROQ_MODEL, GroqJSON, as_text, chunk_text, clamp_conf,  # noqa: E402
                         strip_honorific, ungrounded_words, verify_quote)

# The text agent's default model. gpt-oss-20b (GROQ_MODEL, still the default
# for the other agents) kept writing FIR testimony from the informant's side
# ("Kavya saw a man go inside") instead of naming the man as the actor, and
# its output varied run to run; gpt-oss-120b followed the testimony rule
# consistently on the same FIR. Override with --model or TEXT_AGENT_MODEL.
TEXT_MODEL = os.environ.get("TEXT_AGENT_MODEL", "openai/gpt-oss-120b")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
log = logging.getLogger("forensynth.text_agent")

ROLE_HINTS = {"suspect", "witness", "victim", "unknown"}
# The Timeline's canonical action tags (agents/shared.py ACTION_TAG_VOCABULARY):
# the extractor names the kind of act, so the Timeline doesn't have to guess
# it from phrasing.
# A record whose body is unavailable or unverified (deleted, corrupted,
# redacted, metadata-only, truncated, unsent) - the status marker exports put
# in place of the text.
_INTEGRITY_MARKER = re.compile(
    r"^\s*\[(deleted|undelivered|corrupted|redacted|metadata only|forwarded\W+original deleted|partial|draft)\b"
    r"|hash mismatch|integrity (?:unverified|check failed)|checksum (?:mismatch|failed)", re.I)

# First-person statements that an act did not happen, or not then / not
# there - checked on the verbatim quote, whatever the LLM tagged.
_DENIAL_OR_ALIBI = re.compile(
    r"\b(?:I|we)\b[^.?!]{0,40}?\b(?:never|didn't|did not|wasn't|was not|weren't|were not|haven't|have not)\b"
    r"|\b(?:left|departed|gone|went home)\b[^.?!]{0,30}?\blong before\b"
    r"|\b(?:I|we) (?:was|were) (?:at home|elsewhere|not there|somewhere else)\b", re.I)

ACTIONS = ("APPROACH", "ENTER", "WITHDRAW", "TAMPER", "LOITER", "EXIT", "FLEE", "STEAL", "NAVIGATE",
           "WORK", "COMMUNICATE", "CONFIRM", "WATCH", "OBSERVE", "REPORT", "INTERCEPT")
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
    if isinstance(value, (dict, list, tuple)):     # {"value": "..."} from some models
        value = as_text(value)
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


_CLOSINGS = {"regards", "best regards", "kind regards", "warm regards", "thanks", "thank you",
             "many thanks", "sincerely", "yours sincerely", "yours faithfully", "yours truly",
             "cheers", "best", "best wishes"}
_NAME_WORD = r"(?:[A-Z][a-z]+(?:-[A-Z][a-z]+)?|[A-Z]\.)"
_HONORIFIC = r"(?:(?:Sri|Shri|Smt|Kum|Mr|Mrs|Ms|Dr)\.?\s+)?"
_SIGNATURE_LINE = re.compile(rf"(?:[-–—~]+\s*)?({_HONORIFIC}{_NAME_WORD}(?:\s+{_NAME_WORD}){{0,3}})")


def signature_name(text: str) -> str | None:
    """The name a message is signed with, read by code (never the LLM), so
    the link between a sender alias and a person is verbatim in the text:
    the LAST line must be just a name, either dash-led ("- Kavya Rao") or
    right after a closing line ("Regards," / "Kavya Rao")."""
    lines = [ln.strip() for ln in str(text or "").strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    last, prev = lines[-1], lines[-2].lower().rstrip(",.!")
    m = _SIGNATURE_LINE.fullmatch(last)
    if not m:
        return None
    name = m.group(1)
    if name.lower().rstrip(".") in _CLOSINGS:
        return None
    if not (last[0] in "-–—~" or prev in _CLOSINGS):
        return None
    return strip_honorific(name)


def _doc(doc_id, text, author=None, sent_at=None, doc_type_hint=None, source=None, location=None) -> dict:
    signed = signature_name(text) if author else None
    return {
        "doc_id": doc_id, "text": text.strip(), "author": author,
        "sent_at": _parse_dt(sent_at), "doc_type_hint": doc_type_hint,
        "source": source or doc_type_hint or "text_document", "location": location,
        # the sender alias and the name it signs with: one person, proven by the text
        "signed_as": signed if signed and signed.lower() != str(author).lower() else None,
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
    # Seconds kept when the export has them: minute-only times made messages
    # sent 10:02:31 / 10:02:48 / 10:02:49 look simultaneous.
    fmt = lambda t: t.strftime("%Y-%m-%d %H:%M:%S" if t.second else "%Y-%m-%d %H:%M")
    lines = [f"[{fmt(r['sent_at'])}] {r['author']}: {r['text']}" for r in recs]
    d = _doc(stem, "\n".join(lines), author=None, sent_at=recs[0]["sent_at"],
             doc_type_hint=recs[0]["doc_type"] or "chat", source=recs[0]["source"] or "chat",
             location=recs[0]["location"])
    d["is_transcript"] = True
    d["messages"] = [(r["sent_at"], r["author"], r["text"]) for r in recs]
    # per sender: the one name their messages are signed with (none if they disagree)
    signed: dict[str, set] = {}
    for r in recs:
        if (n := signature_name(r["text"])) and n.lower() != str(r["author"]).lower():
            signed.setdefault(r["author"], set()).add(n)
    d["signed_by_sender"] = {a: next(iter(ns)) for a, ns in signed.items() if len(ns) == 1}
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


TEXT_EXTENSIONS = {".txt", ".md", ".eml", ".json"}


def expand_inputs(paths: list[Path]) -> list[Path]:
    """Files as given; folders searched recursively for text evidence.
    Audio manifests (*manifest*.json) are skipped, so a whole case folder
    can be passed without its recordings' metadata being read as statements."""
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.is_file()
                            and f.suffix.lower() in TEXT_EXTENSIONS
                            and "manifest" not in f.name.lower())
        else:
            files.append(p)
    return files


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
  e.g. "unknown_man_1" (and describe them in "people"). A person's "description" holds ONLY
  physical details the text states - clothing, colours, build, age, distinguishing marks (e.g.
  "black hoodie, red cap, tall") - or "" if none are stated. Never add details.
  In a message transcript (lines like "[YYYY-MM-DD HH:MM] Sender: text") the sender is the actor
  of what they say, and that line's bracketed time is when the message was sent - EXCEPT in an
  audio_transcript (next rule).
- A recorded call or interview (declared_type "audio_transcript"), an FIR (declared_type "fir")
  and a witness statement are testimony. Each thing the speaker describes someone else doing is
  an event whose actor is THE PERSON WHO DID IT (their name, or an unnamed label such as
  "unknown_man_1"). Example: "I saw a man going inside the kiosk" -> actor "unknown_man_1",
  content "unknown_man_1 went inside the ATM kiosk.", evidence "I saw a man going inside the
  kiosk". The speaker is the actor only of what they did themselves ("I was walking on the
  footpath" -> actor = the speaker). Questions an interviewer asks are not events - skip them.
  The chat rule below ("X messaged that ...") does not apply to testimony.
- "content": one neutral sentence describing the event, third person, past tense. For a chat
  message that only reports or plans something, describe the message (e.g. "Rakesh messaged that
  ..."), and set event_time to the message time.
- "role_hint": suspect | witness | victim | unknown - your best reading of the actor's role in the
  alleged incident, with "role_hint_confidence" 0-1 and a brief "role_reason". This is a HINT for
  investigators, not a finding: use "unknown" when the text does not support a call.
  A role is the person's part in the INCIDENT, not in the record: telling, reporting or describing something - including one's own actions - does not make anyone a witness. A witness saw the incident without taking part; someone whose own acts are entering, withdrawing, tampering, keeping lookout or leaving is not a witness because they describe them.
- "event_time": ISO-8601 "YYYY-MM-DDTHH:MM:SS" if the time can be determined, else null.
  Resolve relative expressions ("around 9 pm", "ten minutes later", "yesterday", "on Sunday night")
  against dates stated elsewhere in the text, else the document date / reference date. If the
  day and month are given but not the year ("Friday 14th March"), take the year from other dates
  in the text or the reference date. "time_text": the verbatim time expression you used (e.g.
  "around 8:30 or 8:40 in the night"), else null. "time_basis": explicit (stated outright) |
  relative (computed from another time) | document_date (only the document's own date is known)
  | unknown.
- In an FIR the document date is when the FIR was REGISTERED, usually after the incident. An
  incident event with no time of its own takes the incident time narrated before it, with
  time_basis "relative" (a man loitering, entering and leaving after "at about 10 o'clock" ->
  about 10:00). Skip the complaint formalities (no connection to the parties, giving this
  complaint, asking police to act, signatures, the officer registering the case): the FIR record
  already holds them. A report made during the incident (e.g. calling 112) is an event.
- "location": where it happened, as written, else null.
- "tags": any of {sorted(EVENT_TAGS)}.
- "action": the kind of act, one of {list(ACTIONS)} (WATCH = keeping lookout, LOITER = hanging
  around, OBSERVE = seeing something happen, REPORT = telling police/authorities), or null if
  none fits. A denial ("I never went inside") gets null, not the act it denies. A message that
  announces or reports an act gets the act it describes, not COMMUNICATE: "Entered." -> ENTER,
  "Cash out now." -> WITHDRAW, "Done here. Moving." -> EXIT, "Still waiting." -> LOITER,
  "Reader fitted." -> TAMPER. COMMUNICATE is only for contact that is itself the point
  ("Call me", "Ping me when you're there"). In such a message the actor is the sender: an
  object or device it names ("Reader fitted.", "Card working.") is never the actor, and is not
  listed in "people".
- "confidence" 0-1: how clear, specific and first-hand the statement is (hearsay, vague or
  hedged text scores lower; precise first-hand detail scores higher).
- Skip greetings, signatures, boilerplate and anything with no bearing on what happened.

Return ONLY a JSON object:
{{"doc_type": one of {sorted(DOC_TYPES)},
  "people": [{{"name": "...", "aliases": ["..."], "description": "..."}}],
  "observations": [{{"actor": "...", "role_hint": "...", "role_hint_confidence": 0.0,
     "role_reason": "...", "content": "...", "evidence": "...", "event_time": null,
     "time_text": null, "time_basis": "...", "location": null, "mentioned_people": ["..."], "tags": [],
     "action": null, "confidence": 0.0}}]}}"""


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


def _descriptions(people: list, chunk: str) -> dict[str, str]:
    """canonical name -> the LLM's description of that person (clothing,
    build...), passed on as `appearance` for ER's identity matching. A
    description using any word the text doesn't contain is dropped: an
    invented "black hoodie" would become false identity evidence."""
    out = {}
    for p in people or []:
        if isinstance(p, dict) and p.get("name"):
            desc = as_text(p.get("description")).strip()
            if desc and ungrounded_words(desc, chunk):
                log.info("  dropped ungrounded description for %s: %.60s", p["name"], desc)
                desc = ""
            out[str(p["name"]).strip()] = desc
    return out


_UNNAMED = re.compile(r"^(unknown|unidentified|unnamed)", re.IGNORECASE)


def scoped_actor(actor: str, doc_id: str) -> str:
    """Unnamed people are labelled per document ("unknown_man_1"), so the
    same label in two documents is usually two different people - but ER
    treats an identical alias as one person. Make the label document-scoped;
    linking them is ER's job, on evidence (e.g. matching appearance)."""
    return f"{actor}@{doc_id}" if _UNNAMED.match(actor or "") else actor


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


def _clean_llm_obs(raw: dict, doc: dict, chunk: str, aliases: dict[str, str],
                   descriptions: dict[str, str] | None = None) -> dict | None:
    """Validate one LLM observation. Returns None (dropped) if its evidence
    quote can't be found in the source chunk."""
    evidence = str(raw.get("evidence") or "").strip()
    if not verify_quote(evidence, chunk):
        log.info("  dropped unverified event: %.80s", raw.get("content"))
        return None
    actor = str(raw.get("actor") or "").strip() or doc.get("author") or "unknown_person"
    canon = aliases.get(actor.lower(), actor)
    actor = strip_honorific(canon)
    # A short message from a known sender with no named subject is the
    # sender's own act: "Reader fitted." is not done by someone called
    # Reader, "At the corner, about to move." not by an unknown person.
    author = doc.get("author")
    if (author and actor != author and _is_short_message(doc.get("text", ""))
            and (actor.lower().startswith("unknown") or actor.lower() not in aliases)):
        actor = author
    role = str(raw.get("role_hint") or "unknown").strip().lower()
    basis = str(raw.get("time_basis") or "unknown").strip().lower()
    tags = {str(t).lower() for t in (raw.get("tags") or [])} & EVENT_TAGS
    if _DENIAL_OR_ALIBI.search(evidence):
        tags.add("denial")
    event_time = _fix_meridiem(_parse_dt(raw.get("event_time")), raw.get("time_text"))
    time_text = as_text(raw.get("time_text")).strip()
    if event_time is not None and basis == "explicit" and not (time_text and verify_quote(time_text, chunk)):
        # "Explicit" means stated in the text. Without a stated time the model
        # is rounding the document's own timestamp (an email sent 10:02:31
        # came back as "explicit 10:02:00", so four emails looked
        # simultaneous) or guessing - fall back to the document date instead.
        log.info("  explicit time %s not stated in the text (%r) - using the document date: %.60s",
                 event_time.isoformat(), time_text, raw.get("content"))
        event_time, basis = None, "unknown"
    return {
        "actor": actor,
        "role_hint": role if role in ROLE_HINTS else "unknown",
        "role_hint_confidence": clamp_conf(raw.get("role_hint_confidence"), 0.0),
        "role_reason": str(raw.get("role_reason") or "")[:300],
        "content": str(raw.get("content") or evidence)[:2000],
        "evidence": evidence[:500],
        "event_time": event_time,
        "time_basis": basis if basis in TIME_BASES else "unknown",
        "location": raw.get("location") or None,
        "mentioned_people": [strip_honorific(aliases.get(str(m).strip().lower(), str(m).strip()))
                             for m in (raw.get("mentioned_people") or []) if str(m).strip()],
        "tags": sorted(tags),
        "confidence": clamp_conf(raw.get("confidence"), 0.6),
        # a denial or alibi states the act did NOT happen (or not then, or not
        # there): it never carries the act's tag ("I departed long before the
        # time mentioned" came back as WITHDRAW, then EXIT)
        "action": None if tags & {"denial", "alibi"} else
                  a if (a := str(raw.get("action") or "").strip().upper()) in ACTIONS else None,
        "actor_description": (descriptions or {}).get(canon, ""),
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
        descriptions = _descriptions(result.get("people"), chunk)
        raw_obs = result.get("observations") or []
        kept = [c for c in (_clean_llm_obs(r, doc, chunk, aliases, descriptions)
                            for r in raw_obs if isinstance(r, dict)) if c]
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
        given = ev["event_time"] if ev["time_basis"] != "document_date" else None
        for sent_at, _author, msg in doc["messages"]:
            if verify_quote(ev["evidence"], msg):
                # The message's own time wins when the model gave none, or
                # gave that same time rounded to the minute.
                if given is None or abs((given - sent_at).total_seconds()) < 60:
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
# 3. FIR - the case's partial ground truth (any input detected as an FIR)
# =============================================================================
#
# An FIR yields two things:
#   * an FIR RECORD (<case_id>_fir.json, `fir_records` table) kept apart from
#     observations. Every field has a verbatim `evidence` quote, a `verified`
#     flag and a confidence, split into
#       anchors - facts about the REPORT (FIR no., station, date lodged,
#                 informant). Reliable.
#       claims  - what the informant ALLEGES (incident time/place, offence,
#                 accused, victims, witnesses, property, weapons). Hypotheses
#                 the rest of the pipeline tests - not facts.
#   * FIR-DERIVED OBSERVATIONS: the narrative goes through the same extractor
#     as every other document, tagged source "fir" + fir_derived: true.
#     Anything counting corroboration for an FIR claim must skip those, or the
#     FIR ends up confirming itself.
# Accused/victims/witnesses whose quote isn't in the FIR are DROPPED (listed in
# `dropped`); scalar fields with an unverified quote are kept, flagged and
# their confidence halved.

# What makes a document an FIR itself (not one that merely cites "FIR No. ..."):
# the FIR form header, or a complaint letter addressed to a police station.
_FIR_FORM = re.compile(r"FIRST\s+INFORMATION\s+REPORT|Section\s*173\s*B\.?N\.?S\.?S|Section\s*154\s*Cr\.?\s*P\.?\s*C",
                       re.IGNORECASE)
_TO_POLICE = re.compile(r"Police\s+Station|Station\s+House\s+Officer|\bS\.?H\.?O\b|Inspector\s+of\s+Police",
                        re.IGNORECASE)
_COMPLAINT_SUBJECT = re.compile(r"^\s*(?:Sub(?:ject)?|Re)\s*[:.\-].*\bcomplain", re.IGNORECASE | re.MULTILINE)


def is_fir(path: Path, text: str) -> bool:
    """Auto-detect an FIR among the inputs: the FIR form, or a written complaint
    to the police (addressed to a police station, subject 'complaint ...').
    Anything else can be marked explicitly with --fir."""
    if path.suffix.lower() not in (".txt", ".md"):
        return False
    head = text[:800]
    return bool(_FIR_FORM.search(head) or (_TO_POLICE.search(head) and _COMPLAINT_SUBJECT.search(head)))


ANCHOR_FIELDS = ("fir_number", "police_station", "district", "date_of_report")
CLAIM_SCALARS = ("incident_location", "offence", "description")
PERSON_LISTS = ("accused", "victims", "witnesses")
ITEM_LISTS = ("property", "weapons")

_FIR_SYSTEM_PROMPT = """You are extracting the fields of an Indian First Information Report (FIR) /
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
    if isinstance(inf, dict):
        # Some models wrap each sub-field as {"value": ..., "evidence": ...};
        # the record keeps plain values (fir_summary lower()s the relation).
        inf = {k: (v.get("value") if isinstance(v, dict) and k != "evidence" else v)
               for k, v in inf.items()}
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
    if isinstance(it, dict):
        # a model may wrap each value as {"value": ...}
        it = {k: (as_text(v) if k in ("start", "end", "best_estimate") else v) for k, v in it.items()}
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
        # people / items / delay the Critique checks against the evidence
        "accused_people": [{"name": p["name"], "description": p["description"]} for p in c["accused"]],
        "witnesses": [p["name"] or p["description"] for p in c["witnesses"]],
        "victims": [p["name"] or p["description"] for p in c["victims"]],
        "weapons": [i.get("item") for i in c.get("weapons", []) if i.get("item")],
        "property": [i.get("item") for i in c.get("property", []) if i.get("item")],
        "delay_in_reporting": _val(c.get("delay_in_reporting")),
        "fir_number": _val(a["fir_number"]),
        "police_station": _val(a["police_station"]),
        "date_of_report": _val(a["date_of_report"]),
        "fir_record_id": record["fir_id"],
    }


def process_fir(case_id: str, text: str, source_file: str, llm: GroqJSON) -> tuple[dict, dict, list[dict]]:
    """FIR text -> (record, case-level `fir` summary, fir_derived observations)."""
    raw = llm.ask(_FIR_SYSTEM_PROMPT, f"--- FIR TEXT START ---\n{text}\n--- FIR TEXT END ---", max_tokens=8192)
    extractor = llm.model if raw is not None else "regex_fallback"
    if raw is None:
        log.warning("LLM unavailable - FIR record built from labelled form fields only.")
        raw = _regex_fallback(text)
    record = build_fir_record(case_id, text, raw, extractor, source_file)
    summary = fir_summary(record)

    # Informant narrative -> observations through the shared extractor, with the
    # FIR's own incident/report date as the reference for relative times.
    ref = _parse_dt(summary.get("incident_datetime")) or _parse_dt(summary.get("date_of_report"))
    doc = _doc(f"{case_id}_FIR", text, author=summary.get("informant"),
               sent_at=_parse_dt(summary.get("date_of_report")),
               doc_type_hint="fir", source="fir", location=summary.get("location"))
    doc["incident_at"] = _parse_dt(summary.get("incident_datetime"))
    doc["incident_end"] = _parse_dt(summary.get("incident_datetime_end"))
    obs = extract_observations([doc], llm, reference_date=ref, source_override="fir")
    for o in obs:
        o["fir_derived"] = True
    if not obs:
        raise RuntimeError("No observations could be extracted from the FIR narrative.")
    return record, summary, obs


# =============================================================================
# 4. Assemble obs_only case
# =============================================================================

# Standard FIR complaint / registration wording - not incident events; the FIR
# record already holds who reported and when. Matched on an event's evidence quote.
_FIR_FORMALITY = re.compile(
    r"no connection with|giving (?:this|the) complaint|request you to (?:take|register)|"
    r"take (?:necessary |legal )?action against|registered the case|took up (?:the )?investigation|"
    r"signature of|reasons? for delay|came to the police station|"
    r"lodg(?:e|ed|ing) (?:this|the|a|my) complaint|"
    # form header fields quoted as if they were events (registration time, GD entry)
    r"date and time of (?:fir|report|registration)|information received at|general diary", re.I)


def _same_person(a: str | None, b: str | None) -> bool:
    norm = lambda x: re.sub(r"\s+", " ", (strip_honorific(x) or "").lower()).strip()
    return bool(a and b) and norm(a) == norm(b)


def _fir_event_time(ev: dict, ts, basis: str, last_ts, doc: dict):
    """An FIR's document date is its REGISTRATION time, often hours after the
    incident. An event in the informant's account with no time of its own (or
    only that document date) carries forward the last incident time stated
    before it in the FIR ("at about 10 o'clock ... then I saw ..."), else the
    FIR's incident time - not the registration time, which would put the
    whole incident an hour late and split its people from other evidence.
    Only the informant's own complaint (action REPORT, actor = informant)
    keeps the registration time; anything else the model put ON the
    registration time (the security staff telling the informant something,
    a passer-by seeing them leave) is an incident event and is re-anchored
    too. Returns (ts, basis, last_ts)."""
    registered = doc.get("sent_at")
    own_complaint = ev.get("action") == "REPORT" and _same_person(ev.get("actor"), doc.get("author"))
    if ts is not None and basis in ("explicit", "relative"):
        if not (registered and ts == registered and not own_complaint):
            return ts, basis, ts
    elif own_complaint:
        return ts, basis, last_ts
    anchor = last_ts or doc.get("incident_at")
    if anchor is None:
        return ts, basis, last_ts
    return anchor, "relative", last_ts


def _known_as(actor: str, doc: dict) -> list[str]:
    """The actor's other proven name: a sender alias <-> the name its message
    is signed with (see signature_name). ER treats it like the same name."""
    pairs = dict(doc.get("signed_by_sender") or {})
    if doc.get("author") and doc.get("signed_as"):
        pairs[doc["author"]] = doc["signed_as"]
    a = str(actor or "").strip().lower()
    for alias, name in pairs.items():
        if a == str(alias).lower():
            return [name]
        if a == name.lower():
            return [alias]
    return []


def extract_observations(docs: list[dict], llm: GroqJSON | None,
                         reference_date: datetime | None = None,
                         source_override: str | None = None) -> list[dict]:
    """Documents -> observation dicts (obs_id/time_offset assigned by build_case).
    The FIR narrative goes through this same extractor (see process_fir)."""
    observations, order = [], 0
    for doc in docs:
        if _INTEGRITY_MARKER.search(doc["text"]):
            # unavailable / unverified content: one record that it exists, flagged,
            # never read for events (the LLM found none and it was silently dropped)
            order += 1
            ts = doc.get("sent_at")
            observations.append({
                "entity": scoped_actor(doc.get("author") or "unknown_sender", doc["doc_id"]),
                "role": "unknown", "role_hint": "unknown", "role_hint_confidence": 0.0, "role_reason": "",
                "modality": "text", "source": source_override or doc.get("doc_type_hint") or doc["source"],
                "source_doc": doc["doc_id"], "reported_by": doc.get("author"),
                "location": doc.get("location") or "unspecified",
                "content": doc["text"].strip()[:500], "evidence": doc["text"].strip()[:500],
                "mentioned_entities": [], "appearance": "", "tags": [],
                "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S") if ts else "",
                "time_basis": "explicit" if ts else "unknown",
                "confidence": 0.1, "extraction": "integrity_failed", "extraction_status": "integrity_failed",
                "_order": order,
            })
            continue
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
        author_contacts = _author_contacts(doc["text"])
        is_fir = doc.get("doc_type_hint") == "fir"
        last_ts = None                # FIR: the last incident time stated so far

        if is_fir:
            kept = [ev for ev in events if not _FIR_FORMALITY.search(ev.get("evidence") or "")]
            if len(kept) < len(events):
                log.info("  dropped %d FIR complaint formality line(s)", len(events) - len(kept))
            events = kept
        for ev in events:
            order += 1
            # the sender's number / address identifies the author - only on
            # the author's own acts, never on people they describe
            own = author_contacts if author_contacts and ev["actor"] == doc.get("author") else []
            ts, basis = ev["event_time"], ev["time_basis"]
            if is_fir:
                ts, basis, last_ts = _fir_event_time(ev, ts, basis, last_ts, doc)
            if ts is None:
                if doc.get("sent_at"):
                    ts, basis = doc["sent_at"], "document_date"
                else:
                    # No usable date anywhere: leave the timestamp empty - the
                    # pipeline's own "missing" (shared.parse_timestamp -> 0,
                    # Timeline scores it neutral) - rather than invent one.
                    basis = "unknown"
            observations.append({
                "entity": scoped_actor(ev["actor"], doc["doc_id"]),
                "role": "unknown",
                "role_hint": ev["role_hint"],
                "role_hint_confidence": ev["role_hint_confidence"],
                "role_reason": ev["role_reason"],
                "modality": "text",
                "source": source,
                "source_doc": doc["doc_id"],
                "reported_by": doc.get("author"),
                **({"known_as": k} if (k := _known_as(ev["actor"], doc)) else {}),
                "location": ev["location"] or doc.get("location") or "unspecified",
                "content": ev["content"],
                "evidence": ev["evidence"],
                "mentioned_entities": ev["mentioned_people"],
                "appearance": ev.get("actor_description", ""),
                **({"action": ev["action"]} if ev.get("action") else {}),
                "tags": ev["tags"],
                "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S") if ts else "",
                "time_basis": basis,
                "confidence": round(ev["confidence"] * _TIME_BASIS_FACTOR[basis], 3),
                "extraction": extraction,
                **({"contact_ids": own} if own else {}),
                "_order": order,
            })
        if is_fir and extraction == "llm":   # fallback paragraphs are form sections, not events
            _spread_fir_times([o for o in observations if o["source_doc"] == doc["doc_id"]],
                              doc.get("incident_end"))
    return observations


def _is_short_message(text: str) -> bool:
    """A chat-style message: at most 20 words once From/To/Subject headers
    are set aside."""
    body = [ln for ln in (text or "").splitlines()
            if ln.strip() and not re.match(r"(?i)^\s*(from|to|cc|bcc|subject|date|sender|sent by)\s*:", ln)]
    return len(" ".join(body).split()) <= 20


_CONTACT = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|\+?\d[\d ()-]{8,}\d")


def _author_contacts(text: str) -> list[str]:
    """Phone numbers / email addresses that are the AUTHOR's: on a From: /
    Sender: header line, or in the signature (the last three non-empty
    lines). Numbers elsewhere in the body may be anyone's."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    others = re.compile(r"(?i)^(to|cc|bcc|recipient|subject|date)\s*:")
    body = [ln for ln in lines if not others.match(ln) and not re.match(r"(?i)^(from|sender|sent by)\s*:", ln)]
    # a signature only below a body: in a two-line message the number may be anyone's
    signature = body[-3:] if len(body) >= 4 else []
    own = [ln for ln in lines if re.match(r"(?i)^(from|sender|sent by)\s*:", ln)] + signature
    found = []
    for ln in own:
        for m in _CONTACT.findall(ln):
            if m not in found:
                found.append(m)
    return found


FIR_SPREAD_MAX_STEP = timedelta(seconds=60)


def _spread_fir_times(obs: list[dict], window_end: datetime | None) -> None:
    """An FIR narrates the incident in order ("hanging around ... then going
    inside ... then coming out") but usually gives one time for all of it.
    Several events on the SAME timestamp read as simultaneous, and the
    Timeline merges one person's same-moment observations into a single
    event - loiter, enter and exit would collapse into one. So events sharing
    a timestamp are spaced in narrative order from that time towards the NEXT
    time the FIR states (e.g. "at around 10:10 I called 112"), or the end of
    its incident window if none follows - at most FIR_SPREAD_MAX_STEP apart,
    since the window's end is often a later act (the 112 call), not the end
    of these events. Marked time_basis "relative" with a time_note. With no
    later time to bound them, they are left as they are."""
    groups: dict[str, list[dict]] = {}
    for o in obs:
        if o["timestamp"]:
            groups.setdefault(o["timestamp"], []).append(o)
    stated = sorted(t for t in (_parse_dt(k) for k in groups) if t)
    for ts_str, group in groups.items():
        start = _parse_dt(ts_str)
        if len(group) < 2 or start is None:
            continue
        later = [t for t in stated if t > start]
        if window_end and window_end > start:
            later.append(window_end)
        if not later:
            continue
        bound = min(later)
        group.sort(key=lambda o: o["_order"])
        step = min((bound - start) / len(group), FIR_SPREAD_MAX_STEP)
        for i, o in enumerate(group):
            old = o["time_basis"]
            o["timestamp"] = (start + step * i).strftime("%Y-%m-%dT%H:%M:%S")
            o["time_basis"] = "relative"
            o["time_note"] = (f"interpolated: event {i + 1} of {len(group)} narrated in order from "
                              f"{ts_str}, {int(step.total_seconds())} s apart, before the next "
                              f"stated time {bound.strftime('%Y-%m-%dT%H:%M:%S')}")
            o["confidence"] = round(o["confidence"] / _TIME_BASIS_FACTOR[old]
                                    * _TIME_BASIS_FACTOR["relative"], 3)


def _fir_window(fir: dict, base: datetime | None, obs: list[dict]) -> list[int]:
    """FIR incident window as offsets from the case's earliest timestamp."""
    def off(iso):
        dt = _parse_dt(iso)
        return max(0, int((dt - base).total_seconds())) if dt and base else None
    start, end = off(fir.get("incident_datetime")), off(fir.get("incident_datetime_end"))
    if start is not None and end is not None:
        return [start, end]
    return [0, max(o["time_offset"] for o in obs)]


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
    if fir:
        fir = {**fir, "time_window": _fir_window(fir, base, out)}
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
    ap = argparse.ArgumentParser(description="Any text evidence (FIR included) -> ForenSynth obs_only case")
    ap.add_argument("--case-id", default="CASE_ENRON_001")
    ap.add_argument("--input", nargs="+", type=Path, default=[],
                    help="Text evidence files (.txt/.md/.eml/.json), or folders of them. An FIR among "
                         "them is detected automatically. Omit (and --fir) to run the Enron pilot.")
    ap.add_argument("--fir", type=Path, help="Mark this file as the FIR (instead of auto-detecting)")
    ap.add_argument("--no-fir-detect", action="store_true", help="Treat every --input as a plain document")
    ap.add_argument("--output", default="cases")
    ap.add_argument("--db", help="Also save the case (and FIR record) to this memory-store DB")
    ap.add_argument("--reference-date", help="ISO date to resolve relative times against when a document "
                                             "has no date of its own (default: the FIR's incident date)")
    ap.add_argument("--sample-size", type=int, default=5000, help="Enron rows to scan")
    ap.add_argument("--no-llm", action="store_true", help="Skip the LLM, use the paragraph/regex fallback")
    ap.add_argument("--model", default=TEXT_MODEL, help=f"Groq model (default {TEXT_MODEL})")
    args = ap.parse_args()

    paths = list(args.input) + ([args.fir] if args.fir and args.fir not in args.input else [])
    missing = [p for p in paths if not p.exists()]
    if missing:
        sys.exit(f"[ERROR] Missing input file(s): {missing}")
    files = expand_inputs(args.input)
    if args.input and not files and not args.fir:
        sys.exit(f"[ERROR] No {'/'.join(sorted(TEXT_EXTENSIONS))} files in: {args.input}")

    # Split the FIR off from the other documents. One FIR per case.
    fir_path, doc_paths = args.fir, []
    for p in files:
        if args.fir and p.resolve() == args.fir.resolve():
            continue
        if not args.no_fir_detect and not args.fir and is_fir(p, p.read_text(encoding="utf-8-sig")):
            if fir_path is None:
                fir_path = p
                continue
            log.warning("%s also looks like an FIR - only %s is used as the FIR; reading it as a plain "
                        "document.", p.name, fir_path.name)
        doc_paths.append(p)

    if doc_paths or fir_path:
        docs = [d for p in doc_paths for d in load_file(p)]
        label = ", ".join(p.name for p in ([fir_path] if fir_path else []) + doc_paths)
    else:
        docs = load_enron(args.sample_size)
        label = "Enron email thread"

    llm = GroqJSON(model=args.model, enabled=not args.no_llm, max_calls=max(20, 2 * len(docs) + 10))
    record = summary = None
    obs: list[dict] = []
    if fir_path:
        log.info("FIR: %s", fir_path.name)
        record, summary, obs = process_fir(args.case_id, fir_path.read_text(encoding="utf-8-sig").strip(),
                                           fir_path.name, llm)
    ref = (_parse_dt(args.reference_date) if args.reference_date else None) or (
        summary and (_parse_dt(summary.get("incident_datetime")) or _parse_dt(summary.get("date_of_report"))))
    if docs:
        obs += extract_observations(docs, llm, reference_date=ref or None)
    for i, o in enumerate(obs):      # FIR first, then documents, in reading order
        o["_order"] = i
    case = build_case(args.case_id, obs, f"Observations extracted from {len(docs) + bool(fir_path)} text "
                                         f"document(s): {label}.", fir=summary)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.case_id}_obs_only.json"
    out_path.write_text(json.dumps(case, indent=2, ensure_ascii=False), encoding="utf-8")
    if record:
        rec_path = out_dir / f"{args.case_id}_fir.json"
        rec_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.db:
        sys.path.insert(0, str(Path(__file__).parents[3] / "memory"))
        from memory_store import ForenSynthMemory
        mem = ForenSynthMemory(args.db)
        mem.save_case(case)
        if record:
            mem.save_fir_record(record, case["fir"])

    if summary:
        s = case["fir"]
        print(f"FIR {s['fir_number'] or '?'} @ {s['police_station'] or '?'} | offence: {s['crime_type']} | "
              f"incident: {s['incident_datetime'] or '?'} | informant: {s['informant'] or '?'}")
        print(f"accused: {s['accused'] or 'none named'} | roles: {s['roles']} | "
              f"dropped (unverified): {len(record['dropped'])} | extractor: {record['extractor']}")
    n = len(case["observations"])
    by_mode = sum(1 for o in case["observations"] if o["extraction"] == "llm")
    n_fir = sum(1 for o in case["observations"] if o.get("fir_derived"))
    print(f"{n} observations ({n_fir} FIR-derived) from {len(docs) + bool(fir_path)} document(s) "
          f"({by_mode} LLM-extracted, {n - by_mode} fallback, {llm.calls_made} LLM call(s)), "
          f"{len({o['entity'] for o in case['observations']})} distinct actors")
    print(f"Saved -> {out_path}" + (f"\n         {rec_path}" if record else ""))


if __name__ == "__main__":
    main()
