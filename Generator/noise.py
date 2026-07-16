"""
ForenSynth-X+ Noise
Applies all five noise types to clean observations, producing the
final noisy observation list visible to the downstream reasoning system.

Fixes applied vs Copilot version:
  1. Contradiction count formula fixed — previously always produced 1
     contradiction regardless of n due to int(n*0.075) < 1 for n < 14.
     Now uses round() and a sensible minimum, making the rate meaningful.
  2. Video contradictions are now action-specific refutations (same bank
     as audio) rather than generic camera-noise stubs. Generic camera-noise
     is still used as a second contradiction *type* (footage quality) but
     the primary contradiction now carries semantic meaning.
  3. Text contradictions are now action-specific denial messages rather
     than always being "[DELETED]" stubs, which stripped all meaning.
  4. Temporal noise is now domain-scaled proportionally to the observed
     event span, so Office_Theft events (which span hours) receive
     noise in the minutes range rather than 8 seconds.
  5. Final results are returned sorted by timestamp rather than by obs_id,
     better reflecting real evidence streams that arrive in temporal order
     with sensor jitter — forcing temporal reasoning in downstream agents.
"""

import random
from datetime import datetime, timedelta

from config import NoiseConfig
from observations import CleanObservation


# ---------------------------------------------------------------------------
# Semantic variation phrase banks (paraphrase templates)
# ---------------------------------------------------------------------------

_SEMANTIC_VARIANTS: dict[str, list[str]] = {
    "Person enters ATM booth.": [
        "An individual was observed entering the ATM enclosure.",
        "Someone stepped into the ATM cabin.",
        "A person went inside the cash machine booth.",
        "The subject was seen entering the ATM kiosk.",
        "An unidentified individual entered the ATM enclosure.",
        "Footage shows a person stepping into the ATM lobby.",
        "A figure was recorded entering the cash machine booth.",
    ],
    "Person exits ATM booth.": [
        "An individual was seen leaving the ATM.",
        "Someone walked out of the ATM enclosure.",
        "A person exited the cash machine area.",
        "The subject departed the ATM booth.",
        "Footage shows an individual leaving the ATM kiosk.",
        "A person was observed walking out of the cash machine cabin.",
        "The individual exited the ATM lobby and moved away.",
    ],
    "Person interacts with ATM machine.": [
        "Individual seen operating the ATM terminal.",
        "Suspect conducts transaction at the machine.",
        "Person appears to use the ATM.",
        "An individual was recorded interfacing with the ATM panel.",
        "Subject observed pressing buttons on the ATM terminal.",
        "A person was seen conducting activity at the cash machine.",
        "Footage captures an individual interacting with the ATM keypad.",
    ],
    "Bystander observes individuals leaving ATM.": [
        "Witness saw people departing from ATM area.",
        "A passerby noticed persons exiting ATM vicinity.",
        "Onlooker reported seeing individuals leave ATM.",
        "A bystander observed multiple persons leaving the ATM kiosk.",
        "Witness statement describes people walking away from the cash machine.",
        "An observer noted individuals departing the ATM booth area.",
        "A member of the public reported seeing persons leave the ATM.",
    ],
    "Person seen pacing near ATM for extended period.": [
        "Individual observed loitering outside ATM for some time.",
        "Suspect was noted lingering near the ATM kiosk.",
        "A person was seen repeatedly walking back and forth near the ATM.",
        "Subject appeared to be waiting near the cash machine for an extended duration.",
        "Footage captures an individual loitering in the ATM vicinity.",
        "A person stood near the ATM for an unusually long time before approaching.",
    ],
    "Person appears to tamper with ATM card slot.": [
        "Individual noticed fiddling with the ATM card reader.",
        "Suspect observed manipulating card entry slot.",
        "Someone seen handling the card slot area of ATM.",
        "A person was recorded interfering with the ATM card reader mechanism.",
        "Subject appeared to attach something to the card insertion slot.",
        "Footage shows an individual inspecting and touching the card reader.",
        "A suspect was observed making unexplained contact with the ATM card panel.",
    ],
    "Person enters office premises after hours.": [
        "Individual observed accessing office building outside working hours.",
        "Suspect seen entering office after closing time.",
        "A person was recorded accessing the building after business hours.",
        "Subject entered the premises at an hour when the office was officially closed.",
        "Footage shows an individual using an access card after working hours.",
        "A person was seen entering the building well past closing time.",
        "The subject accessed the office outside their sanctioned working window.",
    ],
    "Person transfers files to external storage device.": [
        "Individual caught copying data to USB drive.",
        "Suspect observed transferring documents to removable storage.",
        "A person was seen connecting an external drive and copying files.",
        "Subject recorded transferring data from a workstation to a USB device.",
        "Footage captures an individual inserting a storage device and initiating a transfer.",
        "A suspect was observed exporting files to an external media device.",
        "The individual was seen copying large volumes of data to a removable drive.",
    ],
    "Suspect sends initial message.": [
        "First suspect initiates contact via messaging app.",
        "Primary individual sends opening communication.",
        "The accused was observed sending the first message in the exchange.",
        "Subject initiated digital contact with a co-conspirator.",
        "The first communication in the chain was sent by the primary suspect.",
        "An initial message was dispatched by the accused via an encrypted channel.",
        "Logs confirm the suspect sent the opening message in the communication thread.",
    ],
    "Individual sends first message to co-conspirator.": [
        "Primary suspect initiates contact with associate.",
        "First message in the chain sent by the accused.",
        "The subject dispatched an opening communication to a known associate.",
        "Logs show the accused initiated the exchange with a co-conspirator.",
        "The first outbound message in the thread was sent by the primary individual.",
        "Subject made first digital contact with the second party in the scheme.",
    ],
    "Coded messages exchanged between parties.": [
        "Parties exchanged ambiguous messages with possible hidden meaning.",
        "Communication records show indirect language consistent with coded exchange.",
        "A series of messages using non-literal language was detected between the accused.",
        "Logs reveal an exchange of messages that appear to use coded terminology.",
        "The two parties communicated using language that may carry concealed operational meaning.",
        "Intercepted messages between the subjects contain language consistent with coded coordination.",
    ],
}

_GENERIC_VARIANTS = [
    "Activity noted in the vicinity of the incident.",
    "Unidentified entity observed performing relevant action.",
    "Partial observation recorded; details unclear.",
    "Movement detected matching described behaviour pattern.",
]


def _paraphrase(content: str, rng: random.Random, modality: str = "video") -> str:
    """Return a semantically varied version of the content string."""
    if content in _SEMANTIC_VARIANTS:
        return rng.choice(_SEMANTIC_VARIANTS[content])
    if modality == "video":
        qualifiers = [
            " (partially obscured)",
            " (observed from distance)",
            " (reported by bystander)",
            " (captured on footage)",
        ]
        return content.rstrip(".") + rng.choice(qualifiers) + "."
    return content


# ---------------------------------------------------------------------------
# Contradiction content banks
# ---------------------------------------------------------------------------

_CONTRADICTIONS: dict[str, list[str]] = {
    "enter_atm": [
        "I never entered the ATM, I was outside the whole time.",
        "I did not go inside that booth at any point.",
        "I was nowhere near the ATM enclosure.",
        "I only walked past, I never stepped inside.",
        "I can confirm I did not enter the ATM that day.",
    ],
    "withdraw_cash": [
        "No transaction was made from my end.",
        "I did not withdraw any money from that machine.",
        "I have no record of conducting a transaction there.",
        "My card was not used at that ATM.",
        "I did not touch the machine, let alone make a withdrawal.",
    ],
    "exit_atm": [
        "I left the area well before any incident occurred.",
        "I was not present when whatever happened took place.",
        "I departed that location long before the time mentioned.",
        "I had already left the ATM before anything suspicious occurred.",
        "I exited that area without incident, well before the alleged time.",
    ],
    "steal_items": [
        "I did not take anything from the office.",
        "Nothing was removed by me from those premises.",
        "I have no knowledge of any missing items.",
        "I left everything as I found it.",
        "I categorically deny taking any property from that location.",
    ],
    "steal_data": [
        "I never accessed any restricted files.",
        "No data was transferred by me to any external device.",
        "I did not touch the server or any confidential folders.",
        "I have no access to those files and made no attempt to obtain them.",
        "My credentials were not used to access or copy any restricted data.",
    ],
    "tamper_atm": [
        "I didn't touch the machine at all.",
        "I never interfered with the card reader in any way.",
        "I was not near enough to the ATM to tamper with anything.",
        "I did not attach any device to that machine.",
        "I categorically deny touching or modifying that ATM.",
    ],
    "approach_atm": [
        "I wasn't anywhere near the ATM.",
        "I did not approach that machine on the date in question.",
        "I had no reason to go near that ATM and I did not.",
        "My movements that day did not take me anywhere near that kiosk.",
        "I was not in the vicinity of the ATM at the relevant time.",
    ],
    "initiate_communication": [
        "I never sent any such message.",
        "No communication was initiated by me on that channel.",
        "I did not make contact with anyone regarding this matter.",
        "That message did not come from me.",
        "I have no record of initiating any such exchange.",
    ],
    "confirm_plan": [
        "There was no plan discussed.",
        "I did not confirm or agree to anything of that nature.",
        "No such arrangement was made with my knowledge or consent.",
        "I was not party to any plan being discussed.",
        "I categorically deny confirming any coordinated activity.",
    ],
    "navigate_to_target": [
        "I did not go to any restricted area that day.",
        "I had no reason to visit the server room or accounts floor.",
        "My movements were confined to my designated work area.",
        "I did not enter any zone beyond my authorised access.",
        "I was not navigating to any specific target location.",
    ],
    "coordinate_activity": [
        "I was not coordinating with anyone.",
        "No instructions were exchanged between me and any other party.",
        "There was no coordination on my part regarding this incident.",
        "I acted alone and independently at all times.",
        "I deny communicating with any associate about timing or location.",
    ],
    "flee_scene": [
        "I did not flee. I left calmly and normally.",
        "My departure was unhurried and unrelated to any incident.",
        "I was not running or moving with urgency when I left.",
        "I left the area in an orderly manner with no reason to rush.",
        "I deny leaving the scene hastily or in response to any event.",
    ],
    "wait_outside": [
        "I was not stationed near the entrance at any point.",
        "I did not act as a lookout or stand guard for anyone.",
        "I was not loitering outside the ATM or any other location.",
        "I had no role in watching or monitoring any area.",
        "I deny being positioned near the ATM entrance during the incident.",
    ],
    "loiter_near_atm": [
        "I was not pacing near the ATM at any point.",
        "I did not loiter in that area before entering.",
        "I arrived and transacted immediately — no waiting.",
        "I had no reason to linger near the machine and I did not.",
        "I deny being observed near the ATM for any extended period.",
    ],
    "approach_office": [
        "I did not approach the office building outside my authorised hours.",
        "I was not present near the premises at the time mentioned.",
        "My attendance at the building that day was not as alleged.",
        "I deny approaching the office at the time stated.",
        "I was elsewhere when the alleged approach took place.",
    ],
    "enter_office": [
        "I am a registered employee and entered the building during my normal shift.",
        "My badge scan shows I entered during authorised hours only.",
        "I did not access any area outside my permitted zone.",
        "My entry was routine and within my designated access window.",
        "I deny entering the building outside my authorised schedule.",
    ],
    "perform_legit_work": [
        "I was not at my workstation during the period in question.",
        "I was in a meeting for most of that period — my calendar confirms it.",
        "I did not complete any of the tasks attributed to me that day.",
        "I was absent from the office floor during the relevant period.",
        "I deny performing any work activities at the time indicated.",
    ],
    "observe_exit": [
        "I did not see anyone leave the ATM or office during that time.",
        "I was not in a position to observe any exits from that location.",
        "I cannot confirm having seen anyone depart the premises.",
        "I was not paying attention to anyone leaving the area.",
        "I deny witnessing the departures described.",
    ],
    "report_incident": [
        "I did not file any report or complaint regarding this incident.",
        "I made no contact with authorities about what I may have seen.",
        "I did not approach any officer or security personnel that day.",
        "I have not provided any statement or complaint related to this matter.",
        "I deny having reported any incident to any official.",
    ],
    "exchange_information": [
        "No information was exchanged by me with any party.",
        "I did not share any details, plans, or locations with anyone.",
        "That exchange of information did not involve me.",
        "I have no record of communicating operational details to any person.",
        "I deny sharing any information of the kind described.",
    ],
    "respond_to_communication": [
        "I did not respond to any such message or call.",
        "No reply was sent by me to that number or address.",
        "I have no record of responding to the communication described.",
        "That response did not originate from me.",
        "I deny having replied to any party regarding this matter.",
    ],
    "flag_suspicious_comm": [
        "I did not receive any unusual messages on that date.",
        "No suspicious communication was flagged by me.",
        "I deny having received or forwarded any such message.",
        "My device records show no such incoming communication.",
        "I was not aware of and did not act on any suspicious exchange.",
    ],
    "intercept_communication": [
        "I did not intercept any communication from those parties.",
        "No such interception was performed through my systems.",
        "I deny having access to the communication channel described.",
        "My logs show no interception event at the stated time.",
        "I was not involved in monitoring or capturing any such exchange.",
    ],
}

# Modality-specific fallback contradictions
_CONTRADICTION_VIDEO_QUALITY = [
    "Camera angle unclear; individual's identity uncertain.",
    "Footage inconclusive due to lighting conditions.",
    "Visual feed interrupted; action not fully captured.",
    "Recording quality insufficient to confirm the individual's identity.",
    "Camera obstructed at the relevant timestamp; sequence unclear.",
    "Frame rate drop during this segment renders the footage unreliable.",
    "Subject's face not visible in available footage.",
    "Video timestamp metadata inconsistent with reported incident time.",
]

_CONTRADICTION_TEXT_INTEGRITY = [
    "[DELETED] — Message removed by sender.",
    "[UNDELIVERED] — Message not received by recipient.",
    "[CORRUPTED] — File hash mismatch; content integrity unverified.",
    "[REDACTED] — Content withheld pending legal review.",
    "[METADATA ONLY] — Message body unavailable; delivery record retained.",
    "[FORWARDED — ORIGINAL DELETED] — Forwarded copy only; original thread erased.",
    "[PARTIAL] — Log entry truncated; remainder of message not recovered.",
    "[DRAFT] — Message composed but send status unconfirmed.",
]


def _get_action_denial(action: str, rng: random.Random) -> str | None:
    """Return a denial statement for the given action, or None if unknown."""
    options = _CONTRADICTIONS.get(action)
    return rng.choice(options) if options else None


def apply_noise(
    clean_observations: list[CleanObservation],
    cfg: NoiseConfig,
    rng: random.Random,
    event_action_map: dict[str, str] | None = None,
    time_window_sec: int | None = None,
) -> list[dict]:
    """
    Apply all noise types to the clean observation list.

    Args:
        clean_observations: Output of expand_events_to_observations().
        cfg: Noise configuration parameters.
        rng: Seeded random instance.
        event_action_map: Dict mapping event_id -> action string.
            Pass ``{e.event_id: e.action for e in events}``.
        time_window_sec: Total case time window duration in seconds.
            Used to scale temporal noise proportionally across domains.

    Returns:
        List of final observation dicts sorted by noisy timestamp.
    """
    indexed = list(enumerate(clean_observations))
    rng.shuffle(indexed)

    n = len(indexed)
    missing_count = int(n * cfg.missing_modality_rate)

    # FIX: use round() so contradiction_rate has a real effect for small n.
    # Old: max(1, int(n*0.075)) = 1 for n<14. New: max(1, round(n*0.075)).
    if n == 0 or cfg.contradiction_rate <= 0:
        contradiction_count = 0
    else:
        contradiction_count = min(n, max(1, round(n * cfg.contradiction_rate)))

    missing_indices: set[int] = {i for i, _ in rng.sample(indexed, k=min(missing_count, n))}
    remaining_after_missing = [pair for pair in indexed if pair[0] not in missing_indices]
    contradiction_pool = rng.sample(
        remaining_after_missing,
        k=min(contradiction_count, len(remaining_after_missing)),
    )
    contradiction_indices: set[int] = {i for i, _ in contradiction_pool}

    # FIX: domain-aware temporal noise scaling.
    # Scale noise to 0.5–3% of the case time window, keeping it perceptible
    # but not destructive for all domains (not just ATM).
    if time_window_sec is not None and time_window_sec > 0:
        scaled_min = max(cfg.temporal_noise_min_sec, int(time_window_sec * 0.005))
        scaled_max = max(cfg.temporal_noise_max_sec, int(time_window_sec * 0.03))
        scaled_max = min(scaled_max, 3600)  # cap at 1 hour
    else:
        scaled_min = cfg.temporal_noise_min_sec
        scaled_max = cfg.temporal_noise_max_sec

    results: list[dict] = []

    for original_index, obs in sorted(indexed, key=lambda x: x[0]):
        if original_index in missing_indices:
            continue

        noise_tags: list[str] = []
        content = obs.content
        timestamp = obs.timestamp
        time_offset = obs.time_offset
        confidence = obs.confidence

        # --- Contradiction injection (FIXED) ---
        # All modalities now get action-specific contradictions where possible.
        if original_index in contradiction_indices:
            action = (
                event_action_map.get(obs.event_ref)
                if event_action_map
                else _infer_action_from_event_ref(obs)
            )
            denial = _get_action_denial(action or "", rng) if action else None

            if obs.modality == "audio":
                if denial:
                    content = denial
                    noise_tags.append("contradiction")
                    confidence = max(0.10, confidence - rng.uniform(0.15, 0.30))

            elif obs.modality == "video":
                # 60% action denial (as written refutation), 40% footage quality challenge
                if denial and rng.random() < 0.60:
                    content = denial
                else:
                    content = rng.choice(_CONTRADICTION_VIDEO_QUALITY)
                noise_tags.append("contradiction")
                confidence = max(0.10, confidence - rng.uniform(0.20, 0.40))

            elif obs.modality == "text":
                # 60% action denial message, 40% integrity challenge stub
                if denial and rng.random() < 0.60:
                    content = denial
                else:
                    content = rng.choice(_CONTRADICTION_TEXT_INTEGRITY)
                noise_tags.append("contradiction")
                confidence = max(0.10, confidence - rng.uniform(0.15, 0.30))

        # --- Temporal noise (domain-scaled) ---
        if rng.random() < cfg.temporal_noise_rate:
            shift = rng.randint(scaled_min, scaled_max)
            direction = rng.choice([-1, 1])
            shift_sec = direction * shift
            try:
                dt = datetime.fromisoformat(timestamp)
                dt_noisy = dt + timedelta(seconds=shift_sec)
                timestamp = dt_noisy.isoformat()
                time_offset = max(0, time_offset + shift_sec)
                noise_tags.append("temporal_noise")
            except ValueError:
                pass

        # --- Semantic variation ---
        if rng.random() < cfg.semantic_variation_rate and "contradiction" not in noise_tags:
            new_content = _paraphrase(content, rng, modality=obs.modality)
            if new_content != content:
                content = new_content
                noise_tags.append("semantic_variation")

        final = obs.to_final_dict(noise_tags=noise_tags)
        final["content"] = content
        final["timestamp"] = timestamp
        final["time_offset"] = time_offset
        final["confidence"] = round(confidence, 3)

        results.append(final)

    # FIX: sort by noisy timestamp — agents see evidence in approximate temporal
    # order rather than perfect ground-truth event order (obs_id sequence).
    results.sort(key=lambda x: x["timestamp"])
    return results


def _infer_action_from_event_ref(obs: CleanObservation) -> str:
    """
    Best-effort action inference from obs content for contradiction lookup.
    """
    content_lower = obs.content.lower()
    keyword_action_map = {
        "enter":       "enter_atm",
        "inside":      "enter_atm",
        "withdraw":    "withdraw_cash",
        "transaction": "withdraw_cash",
        "exit":        "exit_atm",
        "leave":       "exit_atm",
        "steal":       "steal_items",
        "transfer":    "steal_data",
        "tamper":      "tamper_atm",
        "approach":    "approach_atm",
        "message":     "initiate_communication",
        "plan":        "confirm_plan",
        "files":       "steal_items",
        "loiter":      "loiter_near_atm",
        "pacing":      "loiter_near_atm",
        "coordinate":  "coordinate_activity",
        "navigate":    "navigate_to_target",
        "server":      "navigate_to_target",
        "fleet":       "flee_scene",
        "run":         "flee_scene",
    }
    for keyword, action in keyword_action_map.items():
        if keyword in content_lower:
            return action
    return "unknown_action"
