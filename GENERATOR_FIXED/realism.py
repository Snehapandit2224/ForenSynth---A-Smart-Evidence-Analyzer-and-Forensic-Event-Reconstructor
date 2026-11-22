"""
ForenSynth-X+ Realism layer
Makes a generated case look like what the REAL field agents (audio, video,
text, FIR) hand the pipeline, instead of what perfect agents would.

The base generator writes observations as if every agent were flawless:
one clean alias per person per modality, the true role on every
observation, a timestamp on everything, and the speaker of a statement
always being its actor. Real agents produce none of that, and the pipeline
can score well on the clean cases for reasons that disappear on real
evidence (e.g. ER keeping a suspect and a witness apart only because the
roles were handed to it). Each flag below reproduces one real-agent
behaviour, so the downstream agents (ER, Timeline, Critique, Showrunner) can
be tested on realistic input with a known answer.

Runs AFTER the case is fully generated and validated, on its own random
stream (seeded from the generator seed + case id), so a case generated with
no flags is byte-for-byte what it always was, and the same seed + flags
always gives the same case.

Everything a flag changes is recorded in ground_truth:
    flags        - the settings used
    obs_entity   - obs_id -> true entity, for EVERY observation (the alias
                   alone no longer identifies the person, e.g. an undiarized
                   recording's shared label)
    entity_mapping - alias -> entity for every unambiguous alias
    appearance   - entity -> its true look
    names        - entity -> its full name (alias variants)
    fir_errors   - which FIR claims are false, with the true values
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

from config import get_observation_location
from noise import _CONTRADICTION_TEXT_INTEGRITY, _CONTRADICTION_VIDEO_QUALITY

_NOT_DENIALS = set(_CONTRADICTION_VIDEO_QUALITY) | set(_CONTRADICTION_TEXT_INTEGRITY)

# ── Configuration ─────────────────────────────────────────────────────────────

_DIFFICULTY_SCALE = {"easy": 0.5, "medium": 1.0, "hard": 1.5}


@dataclass
class RealismConfig:
    """Which real-agent behaviours to reproduce. All off by default."""
    appearance: bool = False           # people have a look; video/witnesses describe it
    appearance_noise: float = 0.2      # chance a description is partial or slightly off
    alias_variants: bool = False       # text names people inconsistently ("Ramesh" / "R. Kumar")
    voice_prints: bool = False         # audio carries per-speaker identity embeddings
    hide_roles: bool = False           # role "unknown" + an imperfect role_hint
    per_source_labels: bool = False    # a new label per camera / recording
    reported_events: bool = False      # witnesses describe what a suspect did
    reported_event_rate: float = 0.6   # chance a witnessable suspect action gets reported
    undiarized_audio: float = 0.0      # chance a recording has no speaker separation
    timestamp_gaps: float = 0.0        # chance an observation has no timestamp
    mixed_timezones: bool = False      # some sources write "Z" / "+05:30" times
    fir_errors: bool = False           # the FIR gets some facts wrong
    action_labels: bool = False        # the extractor names each event's action (with mistakes)
    agent_metadata: bool = False       # time provenance (time_source / time_basis), denial tags
    difficulty: str = "medium"         # scales every rate above
    # derived from difficulty (not CLI flags)
    wrong_hint_rate: float = 0.10
    low_hint_rate: float = 0.10
    tz_source_rate: float = 0.5
    fir_error_count: int = 1
    action_error_rate: float = 0.10    # wrong action label
    action_missing_rate: float = 0.10  # no action label

    def enabled(self) -> bool:
        return any((self.appearance, self.alias_variants, self.voice_prints, self.hide_roles,
                    self.per_source_labels, self.reported_events, self.undiarized_audio > 0,
                    self.timestamp_gaps > 0, self.mixed_timezones, self.fir_errors,
                    self.action_labels, self.agent_metadata))

    def scaled(self) -> "RealismConfig":
        """A copy with every rate scaled by the difficulty."""
        k = _DIFFICULTY_SCALE.get(self.difficulty, 1.0)
        cap = lambda x: round(min(0.9, x * k), 3)
        c = RealismConfig(**asdict(self))
        c.appearance_noise = cap(self.appearance_noise)
        c.reported_event_rate = cap(self.reported_event_rate)
        c.undiarized_audio = cap(self.undiarized_audio)
        c.timestamp_gaps = cap(self.timestamp_gaps)
        c.wrong_hint_rate = cap(self.wrong_hint_rate)
        c.low_hint_rate = cap(self.low_hint_rate)
        c.tz_source_rate = cap(self.tz_source_rate)
        c.action_error_rate = cap(self.action_error_rate)
        c.action_missing_rate = cap(self.action_missing_rate)
        c.fir_error_count = 2 if self.difficulty == "hard" else 1
        return c

    @classmethod
    def realistic(cls, difficulty: str = "medium") -> "RealismConfig":
        """--realistic: every behaviour on at its standard rate."""
        return cls(appearance=True, alias_variants=True, voice_prints=True, hide_roles=True,
                   per_source_labels=True, reported_events=True, undiarized_audio=0.3,
                   timestamp_gaps=0.15, mixed_timezones=True, fir_errors=True,
                   action_labels=True, agent_metadata=True, difficulty=difficulty)


# ── Vocabularies ──────────────────────────────────────────────────────────────

# Words chosen to be ones ER's appearance matcher knows (agents/entity_resolution.py)
_COLOURS = ["black", "white", "grey", "red", "blue", "green", "yellow", "brown", "maroon", "navy", "khaki"]
_TOPS = ["hoodie", "jacket", "shirt", "t-shirt", "kurta", "sweater"]
_HEADWEAR = ["cap", "helmet", "scarf", None, None]
_BUILDS = ["tall", "short", "thin", "stocky"]
# a witness's imprecision: the colour they report instead
_NEAR_COLOUR = {
    "black": ["navy", "dark"], "navy": ["black", "blue"], "blue": ["navy"], "red": ["maroon"],
    "maroon": ["red", "brown"], "grey": ["white", "dark"], "white": ["grey"], "green": ["khaki"],
    "khaki": ["brown", "green"], "brown": ["maroon", "khaki"], "yellow": ["white"],
}

_FIRST_NAMES = ["Ramesh", "Suresh", "Arjun", "Vikram", "Rahul", "Anil", "Deepak", "Manoj", "Kiran",
                "Priya", "Kavitha", "Lakshmi", "Meena", "Anjali", "Sunita", "Farhan", "Imran", "Joseph"]
_SURNAMES = ["Kumar", "Sharma", "Reddy", "Nair", "Iyer", "Patel", "Rao", "Khan", "Das", "Menon"]

# The canonical action an extractor names for each generator action (the
# Timeline's ACTION_TAG_VOCABULARY in agents/shared.py).
_ACTION_LABEL = {
    "approach_atm": "APPROACH", "approach_office": "APPROACH", "enter_atm": "ENTER",
    "enter_office": "ENTER", "withdraw_cash": "WITHDRAW", "exit_atm": "EXIT", "exit_office": "EXIT",
    "wait_outside": "WATCH", "loiter_near_atm": "LOITER", "tamper_atm": "TAMPER", "flee_scene": "FLEE",
    "steal_items": "STEAL", "steal_data": "STEAL", "navigate_to_target": "NAVIGATE",
    "perform_legit_work": "WORK", "observe_exit": "OBSERVE", "report_incident": "REPORT",
    "initiate_communication": "COMMUNICATE", "respond_to_communication": "COMMUNICATE",
    "exchange_information": "COMMUNICATE", "coordinate_activity": "COMMUNICATE",
    "confirm_plan": "CONFIRM", "intercept_communication": "INTERCEPT", "flag_suspicious_comm": "REPORT",
}

# What a witness says about a suspect action ({who} = "a man in a black hoodie").
# Only actions a bystander can see; communications aren't witnessable.
_REPORTED_PHRASES: dict[str, list[str]] = {
    "approach_atm": ["{who} walked up to the ATM.", "I saw {who} heading towards the ATM."],
    "enter_atm": ["{who} went inside the ATM booth.", "I saw {who} step into the booth."],
    "withdraw_cash": ["{who} was at the machine taking out cash.", "{who} was using the ATM."],
    "exit_atm": ["{who} came out of the booth fast.", "Then {who} came out of the ATM quickly."],
    "wait_outside": ["{who} was standing outside the booth, watching.", "{who} kept waiting near the entrance."],
    "loiter_near_atm": ["{who} was hanging around near the ATM for a long time.",
                        "{who} kept loitering by the ATM."],
    "tamper_atm": ["{who} was fiddling with the card slot.", "I saw {who} fixing something on the machine."],
    "flee_scene": ["{who} ran off towards the main road.", "{who} rushed away from there."],
    "approach_office": ["{who} walked up to the office building.", "I saw {who} coming towards the office."],
    "enter_office": ["{who} went into the office.", "I saw {who} entering the building."],
    "exit_office": ["{who} left the office in a hurry.", "{who} came out of the building quickly."],
    "navigate_to_target": ["{who} went towards the accounts section.", "{who} walked to the back rooms."],
    "steal_items": ["{who} took something and put it in a bag.", "I saw {who} picking up items from a desk."],
    "steal_data": ["{who} was copying files at a computer.", "{who} plugged something into a computer."],
    "perform_legit_work": ["{who} was working at a desk.", "{who} seemed to be doing normal work."],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _describe(look: dict, rng: random.Random, noise: float, partial_ok: bool = True) -> str:
    """A description of a look, as a witness or camera agent would give it:
    with probability `noise` it is partial or gets a colour slightly wrong."""
    colour, top, hcolour, head, build = (look["colour"], look["top"], look.get("head_colour"),
                                         look.get("head"), look["build"])
    if rng.random() < noise:
        if rng.random() < 0.5:
            colour = rng.choice(_NEAR_COLOUR.get(colour, [colour]))
        elif partial_ok:
            build = head = None   # "a man in a black hoodie" - nothing else
    parts = [build] if build else []
    parts.append(f"{colour} {top}")
    if head:
        parts.append(f"{hcolour} {head}" if hcolour else head)
    return ", ".join(parts)


def _who(description: str | None) -> str:
    """"tall, black hoodie, red cap" -> "a tall man in a black hoodie and a red cap"."""
    if not description:
        return "a man"
    parts = description.split(", ")
    build = parts[0] if parts[0] in _BUILDS else ""
    clothes = parts[1:] if build else parts
    return f"a {build + ' ' if build else ''}man in a {' and a '.join(clothes)}"


def _vector(rng: random.Random, dim: int) -> list[float]:
    v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _with_tz(ts: str, style: str) -> str:
    """The same instant (the pipeline reads naive times as UTC) written as an
    aware timestamp: "...Z" or the "+05:30" local equivalent."""
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return ts
    if style == "Z":
        return dt.isoformat(timespec="seconds") + "Z"
    return (dt + timedelta(hours=5, minutes=30)).isoformat(timespec="seconds") + "+05:30"


# ── Main transform ────────────────────────────────────────────────────────────

def apply_realism(case: dict, cfg: RealismConfig, rng: random.Random) -> dict:
    cfg = cfg.scaled()
    gt = case["ground_truth"]
    mapping0: dict[str, str] = dict(gt["entity_mapping"])
    entities = {e["entity_id"]: e for e in gt["entities"]}
    events = {e["event_id"]: e for e in gt["events"]}
    domain = case["domain"]
    obs_list: list[dict] = [dict(o) for o in case["observations"]]
    for o in obs_list:
        o["_true"] = mapping0[o["entity"]]
        o["_kind"] = "own"

    # 1. Looks ────────────────────────────────────────────────────────────────
    looks: dict[str, dict] = {}
    if cfg.appearance or cfg.reported_events:
        shared = None
        for eid in sorted(entities):
            look = {"colour": rng.choice(_COLOURS), "top": rng.choice(_TOPS),
                    "head": rng.choice(_HEADWEAR), "build": rng.choice(_BUILDS)}
            look["head_colour"] = rng.choice(_COLOURS) if look["head"] in ("cap", "scarf") else None
            # Look-alikes on purpose: suspects often dress alike (dark tops),
            # so a colour + garment match alone must not prove identity.
            if entities[eid]["role"] == "suspect":
                if shared and rng.random() < 0.35:
                    look["colour"], look["top"] = shared
                shared = shared or (look["colour"], look["top"])
            looks[eid] = look

    # 2. Names (alias variants) ───────────────────────────────────────────────
    names: dict[str, tuple[str, str]] = {}
    if cfg.alias_variants:
        surname_pool = list(_SURNAMES)
        trap = rng.choice(surname_pool)   # two people sharing a surname
        for i, eid in enumerate(sorted(entities)):
            last = trap if i < 2 else rng.choice(surname_pool)
            first = rng.choice([n for n in _FIRST_NAMES if n not in {f for f, _ in names.values()}])
            names[eid] = (first, last)

    def name_variant(eid: str) -> str:
        first, last = names[eid]
        return rng.choices([f"{first} {last}", first, last, f"{first[0]}. {last}"],
                           weights=[0.4, 0.3, 0.15, 0.15])[0]

    # 3. Reported events ──────────────────────────────────────────────────────
    if cfg.reported_events:
        witnesses = [eid for eid, e in entities.items()
                     if e["role"] == "witness" and ({"audio", "text"} & set(e["aliases"]))]
        # each witness reports through a channel they actually have: an
        # existing recording / document of theirs, else a fresh statement
        channels = {}
        for w in witnesses:
            mods = [m for m in ("audio", "text") if m in entities[w]["aliases"]]
            mod = mods[0]
            own_sources = [o["source"] for o in obs_list if o["_true"] == w and o["modality"] == mod]
            channels[w] = (mod, own_sources[0] if own_sources else
                           ("witness_statement" if mod == "audio" else "complaint_register"))
        next_id = max(int(o["obs_id"].lstrip("O")) for o in obs_list if o["obs_id"][1:].isdigit()) + 1
        unnamed: dict[tuple[str, str], str] = {}
        for ev in sorted(events.values(), key=lambda e: e["time_offset"]):
            phrases = _REPORTED_PHRASES.get(ev["action"])
            if not witnesses or ev["role"] != "suspect" or not phrases or rng.random() >= cfg.reported_event_rate:
                continue
            w = rng.choice(witnesses)
            mod, src = channels[w]
            # the actor is the suspect, labelled the way an extractor labels
            # an unnamed person - per recording/document; sometimes named
            if cfg.alias_variants and rng.random() < 0.3:
                actor = name_variant(ev["entity_id"])
            else:
                key = (src, ev["entity_id"])
                if key not in unnamed:
                    k = 1 + sum(1 for s, _ in unnamed if s == src)
                    unnamed[key] = f"unknown_man_{k}@{src}"
                actor = unnamed[key]
            desc = _describe(looks[ev["entity_id"]], rng, cfg.appearance_noise) \
                if cfg.appearance and rng.random() < 0.8 else None
            delay = rng.randint(-60, 60)   # a witness's sense of time
            ts = datetime.fromisoformat(ev["timestamp"]) + timedelta(seconds=delay)
            obs_list.append({
                "obs_id": f"O{next_id}", "event_ref": ev["event_id"], "entity": actor,
                "role": "suspect", "modality": mod, "source": src,
                "location": get_observation_location(domain, mod, src),
                "content": (lambda t: t[0].upper() + t[1:])(rng.choice(phrases).format(who=_who(desc))),
                "timestamp": ts.isoformat(), "time_offset": max(0, ev["time_offset"] + delay),
                "confidence": round(rng.uniform(0.55, 0.8), 3), "noise_tags": ["reported_event"],
                **({"appearance": desc} if desc else {}),
                "_true": ev["entity_id"], "_kind": "reported", "_witness": w,
            })
            next_id += 1

    # 4. Labels: per-source, undiarized, name variants ────────────────────────
    # one draw per recording (source), not per observation
    undiarized_sources = {src for src in sorted({o["source"] for o in obs_list if o["modality"] == "audio"})
                          if cfg.undiarized_audio > 0 and rng.random() < cfg.undiarized_audio}
    speaker_no: dict[tuple[str, str], int] = {}
    person_no: dict[tuple[str, str], int] = {}
    used_person: dict[str, set] = {}

    def own_label(eid: str, mod: str, src: str) -> str:
        """The label a real agent gives this person in this source."""
        if mod == "audio":
            if src in undiarized_sources:
                return f"{src}_Speaker_Unknown"
            if cfg.per_source_labels:
                key = (src, eid)
                if key not in speaker_no:
                    speaker_no[key] = sum(1 for s, _ in speaker_no if s == src)
                return f"{src}_SPEAKER_{speaker_no[key]:02d}"
        if mod == "video" and cfg.per_source_labels:
            key = (src, eid)
            if key not in person_no:
                taken = used_person.setdefault(src, set())
                n = rng.choice([i for i in range(1, 100) if i not in taken])
                taken.add(n)
                person_no[key] = n
            return f"{src}_Person_{person_no[key]:02d}"
        if mod == "text" and cfg.alias_variants:
            return name_variant(eid)
        return entities[eid]["aliases"].get(mod, "")

    for o in sorted(obs_list, key=lambda x: (x["time_offset"], x["obs_id"])):
        if o["_kind"] == "own":
            o["entity"] = own_label(o["_true"], o["modality"], o["source"])
        else:
            o["reported_by"] = own_label(o["_witness"], o["modality"], o["source"])
        if o["modality"] == "audio" and o["source"] in undiarized_sources:
            if o["_kind"] == "own":
                o["extraction_status"] = "undiarized"
                o["confidence"] = round(o["confidence"] * 0.7, 3)

    # 5. Voice prints ─────────────────────────────────────────────────────────
    if cfg.voice_prints:
        base = {eid: _vector(rng, 64) for eid in sorted(entities)}
        prints: dict[tuple[str, str], list[float]] = {}
        for o in obs_list:
            if o["modality"] != "audio" or o["_kind"] != "own" or o["source"] in undiarized_sources:
                continue
            key = (o["_true"], o["source"])
            if key not in prints:   # one voice print per speaker per recording, with noise
                v = [x + rng.gauss(0.0, 0.05) for x in base[o["_true"]]]
                n = math.sqrt(sum(x * x for x in v)) or 1.0
                prints[key] = [round(x / n, 4) for x in v]
            o["identity_embedding"] = prints[key]

    # 6. Appearance on video (the video agent's people registry) ──────────────
    if cfg.appearance:
        registry: dict[tuple[str, str], str] = {}
        for o in obs_list:
            if o["modality"] == "video" and o["_kind"] == "own":
                key = (o["_true"], o["source"])
                if key not in registry:   # one description per camera, like the registry
                    registry[key] = _describe(looks[o["_true"]], rng, cfg.appearance_noise)
                o["appearance"] = registry[key]

    # 7. Roles hidden, hints given ────────────────────────────────────────────
    if cfg.hide_roles:
        for o in obs_list:
            true_role = entities[o["_true"]]["role"]
            r = rng.random()
            if r < cfg.wrong_hint_rate:
                hint, conf = rng.choice([x for x in ("suspect", "witness", "victim") if x != true_role]), \
                    rng.uniform(0.7, 0.9)
            elif r < cfg.wrong_hint_rate + cfg.low_hint_rate:
                hint, conf = rng.choice([true_role, "unknown"]), rng.uniform(0.2, 0.6)
            else:
                hint, conf = true_role, rng.uniform(0.7, 0.95)
            o["role"] = "unknown"
            o["role_hint"], o["role_hint_confidence"] = hint, round(conf, 3)

    # 8. Missing timestamps ───────────────────────────────────────────────────
    if cfg.timestamp_gaps > 0:
        for o in obs_list:
            if o["modality"] != "audio" and rng.random() < cfg.timestamp_gaps:
                o["timestamp"], o["time_offset"] = "", 0   # an unread clock: no time at all
                o.setdefault("noise_tags", []).append("timestamp_missing")

    # 9. Mixed timezone formats ───────────────────────────────────────────────
    if cfg.mixed_timezones:
        styles = {}
        for src in sorted({o["source"] for o in obs_list}):
            if rng.random() < cfg.tz_source_rate:
                styles[src] = rng.choice(["Z", "+05:30"])
        for o in obs_list:
            if o["source"] in styles and o["timestamp"]:
                o["timestamp"] = _with_tz(o["timestamp"], styles[o["source"]])

    # 9b. Action labels, as the text/audio extractors give them ──────────────
    if cfg.action_labels:
        labels = sorted(set(_ACTION_LABEL.values()))
        for o in obs_list:
            if "contradiction" in o.get("noise_tags", []):
                continue          # a denial names no act
            true_label = _ACTION_LABEL.get(events.get(o.get("event_ref"), {}).get("action", ""))
            r = rng.random()
            if not true_label or r < cfg.action_missing_rate:
                continue
            o["action"] = rng.choice([x for x in labels if x != true_label])                 if r < cfg.action_missing_rate + cfg.action_error_rate else true_label

    # 9c. What the agents say about their own output ──────────────────────
    #  video:       time_source - the agent saw the burned-in clock tick
    #               ("verified_clock") unless the clock was unreadable or
    #               written in another format (the mixed-timezone sources)
    #  text/audio:  time_basis - a person's own message or speech is timed
    #               by its record ("explicit"); a witness's report of someone
    #               else by the witness's estimate ("relative" / "document_date")
    #  denials:     tagged "denial", as the extractors tag them
    if cfg.agent_metadata:
        for o in obs_list:
            if o["modality"] == "video":
                unread = not o["timestamp"] or "+" in o["timestamp"] or o["timestamp"].endswith("Z")
                o["time_source"] = "overlay_first+offset(unverified)" if unread else "verified_clock"
            else:
                o["time_basis"] = ("unknown" if not o["timestamp"] else
                                   rng.choice(["relative", "document_date"]) if o.get("reported_by") else "explicit")
            # only actual denials: the generator's other "contradictions" are
            # footage / file-integrity problems ("Footage inconclusive...")
            if "contradiction" in o.get("noise_tags", []) and o["content"] not in _NOT_DENIALS:
                o["tags"] = ["denial"]

    # 10. FIR errors ──────────────────────────────────────────────────────────
    fir_errors = []
    if cfg.fir_errors:
        fir = case["fir"]
        kinds = rng.sample(["suspect_count", "witness_count", "time_window"], k=cfg.fir_error_count)
        for kind in kinds:
            if kind in ("suspect_count", "witness_count"):
                role = kind.split("_")[0]
                true_n = fir.get("roles", {}).get(role, 0)
                if not true_n:
                    continue
                wrong = true_n + 1 if true_n == 1 or rng.random() < 0.5 else true_n - 1
                fir["roles"][role] = wrong
                fir_errors.append({"type": kind, "fir_value": wrong, "true_value": true_n})
            else:
                shift = rng.choice([-1, 1]) * rng.randint(300, 900)
                true_w = list(fir["time_window"])
                fir["time_window"] = [max(0, t + shift) for t in true_w]
                fir_errors.append({"type": kind, "fir_value": fir["time_window"], "true_value": true_w})

    # 11. Ground truth ────────────────────────────────────────────────────────
    obs_list.sort(key=lambda o: (o["time_offset"], o["obs_id"]))
    obs_entity = {o["obs_id"]: o["_true"] for o in obs_list}
    by_alias: dict[str, set] = {}
    for o in obs_list:
        by_alias.setdefault(o["entity"], set()).add(o["_true"])
    # an alias that names one person maps to them; a shared label
    # (an undiarized recording) names nobody in particular
    gt["entity_mapping"] = {a: next(iter(ids)) for a, ids in sorted(by_alias.items()) if len(ids) == 1}
    gt["obs_entity"] = obs_entity
    gt["flags"] = {k: v for k, v in asdict(cfg).items()}
    if looks:
        gt["appearance"] = {eid: _describe(look, random.Random(0), 0.0) for eid, look in looks.items()}
    if names:
        gt["names"] = {eid: f"{f} {l}" for eid, (f, l) in names.items()}
    if fir_errors:
        gt["fir_errors"] = fir_errors
    for e in gt["entities"]:
        e["all_aliases"] = sorted(a for a, ids in by_alias.items() if e["entity_id"] in ids)

    for o in obs_list:
        for k in ("_true", "_kind", "_witness"):
            o.pop(k, None)
    case["observations"] = obs_list
    return case
