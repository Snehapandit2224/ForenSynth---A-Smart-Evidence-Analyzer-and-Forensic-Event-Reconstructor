#!/usr/bin/env python3
"""
ForenSynth - render_text.py
Turn generated case files (CASE_ATM_XXX.json) into real TEXT evidence documents + an FIR,
the text counterpart of render_media.py (which does audio/video and leaves text alone).

  text observations -> documents the text agent reads as-is:
      email_log        -> emails_<sender>.json     mailbox export, one per sender
      sms_record       -> sms_<sender>.json        SMS export, one per sender
      incident_report  -> incident_register.txt    bank security desk register (free text)
  the case's `fir` + ground truth -> FIR_<case>.txt
      form-style FIR in the same layout as samples/fir/FIR_001_chain_snatching.txt, written
      from the informant's point of view (a witness), so it is PARTIAL ground truth:
      approximate times, only what that witness could have seen, hearsay for the rest.

  Evidence Generation/Cases/Case_009/Text/      (next to Audio/ and Video/; --out media -> media/CASE_ATM_009/text/)
    FIR_CASE_ATM_009.txt
    emails_report_20.json ...
    incident_register.txt
    text_manifest.json             file -> obs_ids it carries, informant -> entity_id, ...

With --run, each case is pushed through the field-agent orchestrator:
    FIR + docs -> text_agent.py (one agent: FIR record + fir_derived observations + document observations)
    merged case -> <case>/Text/agent_output/<case>_TXT_obs_only.json and the memory store

Consistency rules:
  * Senders keep the generator's text alias (report_20, email_86 ...) as their identity, so
    ground_truth.entity_mapping still scores the text agent's output.
  * The informant gets a fictional human name (deterministic per case). If the informant also
    wrote a text document, it is signed with that name - a real cross-document link for ER.
    text_manifest.json lists name -> entity_id so evaluation can extend entity_mapping.
  * Accused are described with the SAME appearance render_media.py gives them on video.
  * Generator annotations such as "(forwarded)" become document features (Fwd: subject,
    "forwarded to security desk"), not literal text. Noise content (corrupted records,
    off-domain lines) is kept on purpose - the agents should cope with it.

Usage:
    python GENERATOR_FIXED/render_text.py                          # every case, documents only
    python GENERATOR_FIXED/render_text.py --case CASE_ATM_009      # one case
    python GENERATOR_FIXED/render_text.py --case CASE_ATM_009 --run          # + FIR/text agents
    python GENERATOR_FIXED/render_text.py --run --no-llm           # agents in fallback mode

--run needs GROQ_API_KEY for LLM extraction (else the agents fall back to heuristics).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_media import ROOT, _rng, alias_to_entity, build_characters  # noqa: E402

ORCHESTRATOR = ROOT / "agents" / "field_agents" / "text agent" / "orchestrator.py"
TEXT_SOURCES = {"email_log", "sms_record", "incident_report"}
FWD_RE = re.compile(r"\s*\(forwarded\)\s*", re.I)
TAG_RE = re.compile(r"\s*\([^()]*\)\s*")

BANKS = {"HDFC": "HDFC Bank", "SBI": "State Bank of India", "ICICI": "ICICI Bank",
         "AXIS": "Axis Bank", "CANARA": "Canara Bank"}
FIRST_M = ["Ramesh", "Suresh", "Prakash", "Venkatesh", "Naveen", "Harish", "Arun", "Girish", "Mahesh", "Deepak"]
FIRST_F = ["Lakshmi", "Shobha", "Anitha", "Kavya", "Deepa", "Rekha", "Pooja", "Sowmya", "Nirmala", "Bhavya"]
SURNAMES = ["Gowda", "Rao", "Shetty", "Murthy", "Hegde", "Kulkarni", "Nayak", "Reddy", "Bhat", "Naidu"]
FATHERS = ["Krishnappa", "Shivanna", "Narayan", "Basavaraj", "Ramaiah", "Puttaswamy", "Gopal", "Lingaiah"]
PINCODES = {"Indiranagar": "560038", "Koramangala": "560034", "MG Road": "560001",
            "Jayanagar": "560041", "Whitefield": "560066"}


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
def clean(content: str) -> tuple[str, bool]:
    """Generator content -> (document text, was_forwarded)."""
    fwd = bool(FWD_RE.search(content))
    text = TAG_RE.sub(" ", FWD_RE.sub(" ", content)).strip()
    return re.sub(r"\s+([.,!?])", r"\1", text), fwd


def place(location: str) -> tuple[str, str]:
    """'Indiranagar HDFC ATM' -> ('Indiranagar', 'HDFC Bank'). Unknown bank -> 'bank'."""
    words = location.split()
    bank = next((BANKS[w.upper()] for w in words if w.upper() in BANKS), None)
    area = []
    for w in words:
        if w.upper() in BANKS or w.upper() == "ATM":
            break
        area.append(w)
    return " ".join(area) or location, bank or "bank"


def hm(dt: datetime) -> str:
    return dt.strftime("%H:%M")


def floor5(dt: datetime) -> datetime:
    return dt.replace(minute=dt.minute - dt.minute % 5, second=0)


def ceil5(dt: datetime) -> datetime:
    f = floor5(dt)
    return f if f == dt else f + timedelta(minutes=5)


def clock_words(dt: datetime) -> str:
    h = dt.hour % 12 or 12
    part = "in the morning" if dt.hour < 12 else "in the afternoon" if dt.hour < 17 else "in the evening"
    return f"{h}:{dt.minute:02d} {part}" if dt.minute else f"{h} o'clock {part}"


def sections(crime: str, n_suspects: int) -> str:
    c, out = crime.lower(), []
    if "robbery" in c:
        out.append("Section 309(4) (robbery)")
    if "theft" in c or "cash machine" in c:
        out.append("Section 303(2) (theft)")
    if "skimming" in c:
        out.append("Section 318(4) (cheating)")
    if "assault" in c:
        out.append("Section 115(2) (voluntarily causing hurt)")
    if not out:
        out.append("Section 303(2) (theft)")
    if n_suspects > 1:
        out.append("Section 3(5) (common intention)")
    acts = "Bharatiya Nyaya Sanhita, 2023 - " + ", ".join(out)
    if "skimming" in c:
        acts += "; Information Technology Act, 2000 - Section 66C"
    return acts


# ----------------------------------------------------------------------------------------------
# People
# ----------------------------------------------------------------------------------------------
def pick_informant(case: dict) -> str | None:
    """The witness who reported the incident, else one who saw it, else any witness."""
    ents = case["ground_truth"]["entities"]
    evs = case["ground_truth"]["events"]
    witnesses = [e["entity_id"] for e in ents if e["role"] == "witness"]
    for action in ("report_incident", "observe_exit"):
        for ev in evs:
            if ev["action"] == action and ev["entity_id"] in witnesses:
                return ev["entity_id"]
    return witnesses[0] if witnesses else None


def make_person(case_id: str, eid: str, gender: str) -> dict:
    r = _rng(case_id, eid, "name")
    first = r.choice(FIRST_M if gender == "male" else FIRST_F)
    return {"name": f"{first} {r.choice(SURNAMES)}", "father": f"{r.choice(FATHERS)}",
            "age": r.randint(24, 58), "house": r.randint(3, 220), "cross": r.randint(1, 14),
            "phone": f"9{r.randint(100, 999)}{r.randint(1, 9)} XXXXX", "gender": gender}


# ----------------------------------------------------------------------------------------------
# Text documents from text observations
# ----------------------------------------------------------------------------------------------
def render_documents(case: dict, out: Path, signer: dict) -> list[dict]:
    """signer: text alias -> name, for documents written by the (named) informant."""
    cid = case["case_id"]
    text_obs = sorted((o for o in case["observations"] if o["modality"] == "text"),
                      key=lambda o: o["timestamp"])
    area, bank = place(case["fir"]["location"])
    docs: list[dict] = []

    def add(fname, doc_type, obs_list):
        docs.append({"file": fname, "doc_type": doc_type, "obs_ids": [o["obs_id"] for o in obs_list],
                     "senders": sorted({o["entity"] for o in obs_list})})

    by_sender: dict[tuple[str, str], list] = {}
    for o in text_obs:
        if o["source"] in ("email_log", "sms_record"):
            by_sender.setdefault((o["source"], o["entity"]), []).append(o)
        elif o["source"] not in TEXT_SOURCES:
            print(f"  [text] {o['obs_id']} unknown text source {o['source']!r} - filed as incident register entry")

    # emails / sms: JSON exports the text agent loads directly (author + timestamp per record)
    for (source, sender), obs_list in sorted(by_sender.items()):
        records = []
        for o in obs_list:
            body, fwd = clean(o["content"])
            if sender in signer and not body.startswith("[CORRUPTED]"):
                body += f"\n\n- {signer[sender]}"
            if source == "email_log":
                subject = ("Fwd: " if fwd else "") + ("update" if len(obs_list) > 1 else "ATM")
                if fwd:
                    body = f"---------- Forwarded message ----------\n{body}"
                records.append({"from": sender, "timestamp": o["timestamp"], "doc_type": "email",
                                "source": "email_log", "location": o["location"],
                                "text": f"Subject: {subject}\n\n{body}"})
            else:
                records.append({"sender": sender, "timestamp": o["timestamp"], "doc_type": "sms",
                                "source": "sms_record", "location": o["location"], "text": body})
        kind = "emails" if source == "email_log" else "sms"
        export = {"description": f"{'Mailbox' if kind == 'emails' else 'SMS'} export for sender {sender}, "
                                 f"recovered during investigation of {cid} (synthetic).",
                  "documents": records}
        fname = f"{sender}.json" if sender.startswith(kind[:3]) else f"{kind}_{sender}.json"
        (out / fname).write_text(json.dumps(export, indent=2, ensure_ascii=False), encoding="utf-8")
        add(fname, "email" if kind == "emails" else "sms", obs_list)

    # incident register: one free-text document per case, explicit time per entry
    reg = [o for o in text_obs if o["source"] not in ("email_log", "sms_record")]
    if reg:
        day = datetime.fromisoformat(reg[0]["timestamp"])
        lines = [f"{bank.upper()} - SECURITY OPERATIONS DESK",
                 f"Incident Register - {case['fir']['location']} ({area})",
                 f"Date: {day.strftime('%d/%m/%Y')} ({day.strftime('%A')})",
                 "Maintained by: duty security officer, bank security office", "",
                 "Messages and complaints logged at the desk (time = time received):", ""]
        for o in reg:
            body, fwd = clean(o["content"])
            if o["entity"] in signer:
                body += f" - {signer[o['entity']]}"
            how = "message forwarded to the desk" if fwd else "message received"
            lines.append(f"{datetime.fromisoformat(o['timestamp']).strftime('%H:%M:%S')}  "
                         f"Sender ID {o['entity']} ({how}): \"{body}\"")
        lines += ["", "Entries copied from the desk system without alteration."]
        (out / "incident_register.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        add("incident_register.txt", "incident_report", reg)
    return docs


# ----------------------------------------------------------------------------------------------
# FIR
# ----------------------------------------------------------------------------------------------
SEEN = {   # what a passer-by can see from outside the kiosk -> first-person narrative
    "approach_atm": "{who} walking up to the ATM kiosk",
    "loiter_near_atm": "{who} hanging around outside the ATM kiosk for some time and looking up and down the road",
    "enter_atm": "{who} going inside the ATM kiosk",
    "wait_outside": "{who} standing outside the kiosk door with a phone, watching the road as if keeping watch",
    "exit_atm": "{who} coming out of the ATM kiosk",
    "flee_scene": "{who} walking away very fast from the ATM, almost running",
}


def render_fir(case: dict, chars: dict, informant_eid: str | None, person: dict, out: Path) -> dict:
    cid, fir = case["case_id"], case["fir"]
    gt = case["ground_truth"]
    evs = sorted(gt["events"], key=lambda e: e["timestamp"])
    r = _rng(cid, "fir")
    area, bank = place(fir["location"])
    t0 = datetime.fromisoformat(evs[0]["timestamp"])
    t_last = datetime.fromisoformat(evs[-1]["timestamp"])
    suspects = [e["entity_id"] for e in gt["entities"] if e["role"] == "suspect"]
    other_witnesses = [e["entity_id"] for e in gt["entities"]
                       if e["role"] == "witness" and e["entity_id"] != informant_eid]

    # Informant's own sightings: their events + suspect actions visible from outside in the
    # few minutes before them. Times are what a person remembers: rounded to 5 minutes.
    mine = [e for e in evs if e["entity_id"] == informant_eid]
    seen_from = (datetime.fromisoformat(mine[0]["timestamp"]) - timedelta(minutes=3)) if mine else t0
    seen = [e for e in evs if e["entity_id"] in suspects and e["action"] in SEEN
            and datetime.fromisoformat(e["timestamp"]) >= seen_from]
    if mine and not seen:
        seen = [e for e in evs if e["entity_id"] in suspects and e["action"] in ("exit_atm", "flee_scene")]
    win_start = floor5(datetime.fromisoformat(seen[0]["timestamp"]) if seen else t0)
    win_end = ceil5(datetime.fromisoformat(mine[-1]["timestamp"]) if mine else t_last)
    if win_end <= win_start:
        win_end = win_start + timedelta(minutes=5)
    reported = ceil5(t_last + timedelta(minutes=r.randint(50, 150)))
    received = reported - timedelta(minutes=r.choice([10, 15, 20]))

    # Accused: unknown persons, described as they look on CCTV.
    ordinal = ["first", "second", "third", "fourth"]
    label = {eid: ("a man" if i == 0 else "another man") if chars[eid]["gender"] == "male"
             else ("a woman" if i == 0 else "another woman") for i, eid in enumerate(suspects)}
    accused_lines = [f"   ({chr(97 + i)}) Unknown person {i + 1}: {chars[eid]['look']}."
                     for i, eid in enumerate(suspects)]

    # Narrative
    who = person["name"]
    pin = PINCODES.get(area, "5600XX")
    opener = {
        "observe_exit": "I was walking on the footpath",
        "report_incident": "I was passing by on the road",
    }.get(mine[0]["action"] if mine else "", "I was near the ATM")
    para1 = (f"I, {who}, residing at the above address, state that on {t0.strftime('%d/%m/%Y')} at about "
             f"{clock_words(win_start)} {opener} near the {fir['location']} in {area}.")
    sights, used, named = [], set(), []
    for e in seen:
        eid = e["entity_id"]
        if (eid, e["action"]) in used:
            continue
        used.add((eid, e["action"]))
        if eid not in named:
            who_txt = label[eid]
            named.append(eid)
        elif len(named) == 1:
            who_txt = "the same " + ("man" if chars[eid]["gender"] == "male" else "woman")
        else:
            who_txt = f"the {ordinal[named.index(eid)]} " + ("man" if chars[eid]["gender"] == "male" else "woman")
        sights.append(SEEN[e["action"]].format(who=who_txt))
    para2 = ("I saw " + "; then I saw ".join(sights) + "." if sights else
             "I saw some persons near the ATM kiosk behaving in a suspicious manner.")
    leavers = {e["entity_id"] for e in evs if e["action"] in ("exit_atm", "flee_scene")} or set(suspects)
    para2 += (f" The {'two' if len(leavers) == 2 else len(leavers)} persons left together in a hurry."
              if len(leavers) > 1 else
              f" {chars[next(iter(leavers))]['he'].capitalize()} left in a hurry.")
    if len(suspects) == 1:
        desc = f"{chars[suspects[0]]['he'].capitalize()} was {chars[suspects[0]]['look']}."
    else:
        desc = " ".join(f"The {ordinal[i]} person was {chars[eid]['look']}." for i, eid in enumerate(suspects))
    seen = "him" if len(suspects) == 1 and chars[suspects[0]]["gender"] == "male" else            "her" if len(suspects) == 1 else "them"
    para3 = f"I could see {seen} clearly from the footpath. {desc}" if desc else ""
    acts = {e["action"] for e in evs}
    report = next((e for e in mine if e["action"] == "report_incident"), None)
    para4 = []
    if report:
        para4.append(f"At around {clock_words(floor5(datetime.fromisoformat(report['timestamp'])))} "
                     f"I called the police control room (112) and informed the bank security.")
    if acts & {"tamper_atm", "withdraw_cash"}:
        heard = []
        if "tamper_atm" in acts:
            heard.append("something had been fixed on the card slot of the machine")
        if "withdraw_cash" in acts:
            heard.append("cash had been taken out of the machine")
        para4.append(f"Later the {bank} security staff who came to the spot told me that "
                     + " and that ".join(heard) + ".")
    if other_witnesses:
        para4.append(f"One more person who was passing by also saw {seen if suspects else 'them'} going away.")
    if informant_eid:
        para4.append(f"I have no connection with the {bank} or with these persons; "
                     "I am giving this complaint as an eyewitness.")
    para4.append("I request you to take action against the above unknown "
                 + ("person." if len(suspects) == 1 else "persons."))

    delay = reported - t_last
    delay_txt = (f"Informant waited at the spot for the {bank} officials to arrive and check the ATM, "
                 "and then came to the police station." if delay > timedelta(hours=1) else "No delay.")
    property_txt = ("Cash withdrawn from the ATM; amount to be confirmed by the bank."
                    if "withdraw_cash" in acts else "Not yet known; to be confirmed by the bank.")
    officer = _rng(cid, "officer")
    officer_name = f"{officer.choice(FIRST_M)[0]}. {officer.choice(SURNAMES)}"
    relation = "Husband's" if person["gender"] == "female" and person["age"] > 30 else "Father's"

    text = f"""FIRST INFORMATION REPORT
(Under Section 173 B.N.S.S.)

1. District: Bengaluru City        P.S.: {area} Police Station        Year: {t0.year}
   FIR No.: {r.randint(100, 899):04d}/{t0.year}              Date and Time of FIR: {reported.strftime('%d/%m/%Y %H:%M')}

2. Acts and Sections: {sections(fir['crime_type'], len(suspects))}

3. Occurrence of offence:
   (a) Day: {t0.strftime('%A')}        Date: {t0.strftime('%d/%m/%Y')}        Time: between about {hm(win_start)} and {hm(win_end)} hrs
   (b) Information received at P.S.: Date {received.strftime('%d/%m/%Y')}, Time {hm(received)} hrs
   (c) General Diary Reference: Entry No. {r.randint(12, 60)}, Time {hm(received)} hrs

4. Type of information: Written

5. Place of occurrence: {fir['location']}, {area}, Bengaluru. Distance from P.S.: approx. {r.randint(5, 30) / 10} km.

6. Complainant / Informant:
   (a) Name: {"Sri" if person["gender"] == "male" else "Smt."} {who}
   (b) {relation} name: Sri {person['father']}
   (c) Age: {person['age']} years
   (d) Address: No. {person['house']}, {person['cross']}th Cross, {area}, Bengaluru - {pin}
   (e) Phone: {person['phone']}

7. Details of known / suspected / unknown accused with full particulars:
   {len(suspects)} unknown person(s):
{chr(10).join(accused_lines) if accused_lines else "   Not known."}

8. Reasons for delay in reporting by the complainant/informant: {delay_txt}

9. Particulars of properties stolen: {property_txt}

10. Total value of property stolen: Not yet known.

12. First Information contents:

{para1}

{para2}

{para3}

{" ".join(para4)}

Signature of complainant: (signed) {who}

13. Action taken: Since the above information reveals commission of offence(s) u/s as
mentioned at item No. 2, registered the case and took up the investigation.

Signature of Officer in charge: (signed) {officer_name}, PSI, {area} P.S.
"""
    text = re.sub(r"\n{3,}", "\n\n", text)
    fname = f"FIR_{cid}.txt"
    (out / fname).write_text(text, encoding="utf-8")
    return {"file": fname, "informant": who, "informant_entity_id": informant_eid,
            "claimed_window": [win_start.isoformat(), win_end.isoformat()],
            "reported_at": reported.isoformat(), "accused_entity_ids": suspects}


def bank_informant(case: dict) -> tuple[dict, None]:
    """No witness in the ground truth: the branch manager reports after reviewing CCTV.
    Not a ground-truth entity, so informant_entity_id stays None."""
    return make_person(case["case_id"], "branch_manager", "male"), None


# ----------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------
EVIDENCE_CASES = ROOT / "Evidence Generation" / "Cases"


def case_text_dir(out_root: Path, cid: str) -> Path:
    """Evidence Generation layout (Cases/Case_009/{Audio,Video,Text}) when out_root is that
    Cases folder, else the render_media layout (<out>/CASE_ATM_009/text)."""
    if out_root.name == "Cases":
        return out_root / f"Case_{cid.rsplit('_', 1)[-1]}" / "Text"
    return out_root / cid / "text"


def render_case(path: Path, out_root: Path) -> tuple[str, Path, dict]:
    case = json.loads(path.read_text(encoding="utf-8"))
    cid = case["case_id"]
    out = case_text_dir(out_root, cid)
    out.mkdir(parents=True, exist_ok=True)
    chars = build_characters(case)
    informant_eid = pick_informant(case)
    if informant_eid:
        person = make_person(cid, informant_eid, chars[informant_eid]["gender"])
    else:
        person, informant_eid = bank_informant(case)

    # The informant signs their own texts with their name (aliases for their text identity).
    text_alias = next((e.get("aliases", {}).get("text") for e in case["ground_truth"]["entities"]
                       if e["entity_id"] == informant_eid), None)
    signer = {text_alias: person["name"]} if text_alias else {}

    docs = render_documents(case, out, signer)
    fir = render_fir(case, chars, informant_eid, person, out)
    a2e = alias_to_entity(case)
    manifest = {
        "case_id": cid,
        "source_case": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "fir": fir,
        "documents": docs,
        "text_obs_without_document": [o["obs_id"] for o in case["observations"]
                                      if o["modality"] == "text"
                                      and not any(o["obs_id"] in d["obs_ids"] for d in docs)],
        # extra aliases the agents will produce that the generator's entity_mapping lacks
        "entity_mapping_additions": {person["name"]: informant_eid} if informant_eid else {},
        "sender_entities": {s: a2e.get(s) for d in docs for s in d["senders"]},
    }
    (out / "text_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"=== {cid}: FIR (informant {person['name']} -> {informant_eid}) + {len(docs)} text document(s) "
          f"covering {sum(len(d['obs_ids']) for d in docs)} text observation(s) -> {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return cid, out, manifest


def run_agents(cid: str, out: Path, manifest: dict, args) -> int:
    """FIR + documents -> orchestrator.py -> text_agent.py (one run), merge, memory store."""
    case_id = f"{cid}{args.suffix}"
    cmd = [sys.executable, str(ORCHESTRATOR), "--case-id", case_id,
           "--fir", str(out / manifest["fir"]["file"]),
           "--staging", str(out / "agent_output" / "_staging"),
           "--output", str(out / "agent_output"), "--db", args.db]
    if manifest["documents"]:
        cmd += ["--text-inputs", *[str(out / d["file"]) for d in manifest["documents"]]]
    if args.adopt_role_hints:
        cmd.append("--adopt-role-hints")
    env = None
    if args.no_llm:
        import os
        env = {**os.environ, "GROQ_API_KEY": ""}   # agents fall back to regex/heuristics
    print(f"  [run] {case_id}: text_agent (FIR + documents) via orchestrator ...")
    rc = subprocess.run(cmd, cwd=str(ROOT), env=env).returncode
    print(f"  [run] {case_id}: {'OK' if rc == 0 else f'FAILED (exit {rc})'}")
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", action="append", help="case ID (CASE_ATM_009) or path; repeatable")
    default_dir = ROOT / "cases"
    ap.add_argument("--cases-dir", default=str(default_dir))
    ap.add_argument("--out", default=str(EVIDENCE_CASES if EVIDENCE_CASES.is_dir() else ROOT / "media"),
                    help="default: 'Evidence Generation/Cases' (-> Case_009/Text/), else media/")
    ap.add_argument("--run", action="store_true", help="push FIR + documents through the text agent")
    ap.add_argument("--suffix", default="_TXT",
                    help="appended to the case ID for the agent run, so the generator's own "
                         "CASE_ATM_XXX entry in the memory store is not touched (default _TXT)")
    ap.add_argument("--db", default=str(ROOT / "forensynth.db"))
    ap.add_argument("--no-llm", action="store_true", help="with --run: no Groq calls, fallback extraction")
    ap.add_argument("--adopt-role-hints", action="store_true", help="passed through to the orchestrator")
    args = ap.parse_args()

    if args.case:
        paths = [Path(c) if c.endswith(".json") else Path(args.cases_dir) / f"{c}.json" for c in args.case]
    else:
        paths = [Path(p) for p in sorted(glob.glob(str(Path(args.cases_dir) / "CASE_ATM_*.json")))
                 if not p.endswith("_obs_only.json")]
    if not paths:
        raise SystemExit(f"No cases found in {args.cases_dir}")
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"Missing case file(s): {missing}")

    failed = []
    for p in paths:
        cid, out, manifest = render_case(p.resolve(), Path(args.out))
        if args.run and run_agents(cid, out, manifest, args) != 0:
            failed.append(cid)
    print(f"\nDone. Output in {args.out}" + (f"  |  agent runs failed: {failed}" if failed else ""))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
