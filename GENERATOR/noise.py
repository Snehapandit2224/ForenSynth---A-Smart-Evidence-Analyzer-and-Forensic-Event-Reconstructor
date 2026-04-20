"""
ForenSynth-X+ Noise
Applies all five noise types to clean observations, producing the
final noisy observation list visible to the downstream reasoning system.
"""

import random
from copy import deepcopy
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
    """Return a semantically varied version of the content string.

    Qualifier-based fallback is only applied to video observations —
    appending camera qualifiers to spoken dialogue or raw message text
    would corrupt those evidence types.
    """
    if content in _SEMANTIC_VARIANTS:
        return rng.choice(_SEMANTIC_VARIANTS[content])
    # Qualifier fallback: video only
    if modality == "video":
        qualifiers = [
            " (partially obscured)",
            " (observed from distance)",
            " (reported by bystander)",
            " (captured on footage)",
        ]
        return content.rstrip(".") + rng.choice(qualifiers) + "."
    # For audio/text, return unchanged — no meaningful paraphrase available
    return content


# ---------------------------------------------------------------------------
# Contradiction content bank
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
}

_CONTRADICTION_TEXT_OVERRIDE = [
    "[DELETED] — Message removed by sender.",
    "[UNDELIVERED] — Message not received by recipient.",
    "[CORRUPTED] — File hash mismatch; content integrity unverified.",
    "[REDACTED] — Content withheld pending legal review.",
    "[METADATA ONLY] — Message body unavailable; delivery record retained.",
    "[FORWARDED — ORIGINAL DELETED] — Forwarded copy only; original thread erased.",
    "[PARTIAL] — Log entry truncated; remainder of message not recovered.",
    "[DRAFT] — Message composed but send status unconfirmed.",
]

_CONTRADICTION_VIDEO_OVERRIDE = [
    "Camera angle unclear; individual's identity uncertain.",
    "Footage inconclusive due to lighting conditions.",
    "Visual feed interrupted; action not fully captured.",
    "Recording quality insufficient to confirm the individual's identity.",
    "Camera obstructed at the relevant timestamp; sequence unclear.",
    "Frame rate drop during this segment renders the footage unreliable.",
    "Subject's face not visible in available footage.",
    "Video timestamp metadata inconsistent with reported incident time.",
]


def apply_noise(
    clean_observations: list[CleanObservation],
    cfg: NoiseConfig,
    rng: random.Random,
    event_action_map: dict[str, str] | None = None,
) -> list[dict]:
    """
    Apply all noise types to the clean observation list.

    Args:
        clean_observations: Output of expand_events_to_observations().
        cfg: Noise configuration parameters.
        rng: Seeded random instance.
        event_action_map: Optional dict mapping event_id → action string.
            When provided, contradiction injection uses the true action instead
            of a content-keyword heuristic. Pass ``{e.event_id: e.action for e in events}``.

    Returns:
        List of final observation dicts (public-facing, no canonical_entity).
    """
    # Work on a shuffled copy to ensure noise targets aren't always the first N
    indexed = list(enumerate(clean_observations))
    rng.shuffle(indexed)

    # Track which indices receive which noise types
    n = len(indexed)
    missing_count = int(n * cfg.missing_modality_rate)
    contradiction_count = max(1, int(n * cfg.contradiction_rate))

    missing_indices: set[int] = {i for i, _ in rng.sample(indexed, k=min(missing_count, n))}
    remaining_after_missing = [pair for pair in indexed if pair[0] not in missing_indices]
    contradiction_pool = rng.sample(
        remaining_after_missing,
        k=min(contradiction_count, len(remaining_after_missing))
    )
    contradiction_indices: set[int] = {i for i, _ in contradiction_pool}

    results: list[dict] = []

    for original_index, obs in sorted(indexed, key=lambda x: x[0]):
        # --- Drop missing ---
        if original_index in missing_indices:
            continue

        # Deep copy so we don't mutate the clean observation
        noise_tags: list[str] = []

        content = obs.content
        timestamp = obs.timestamp
        time_offset = obs.time_offset
        confidence = obs.confidence

        # --- Contradiction injection ---
        if original_index in contradiction_indices:
            if obs.modality == "audio":
                # Use true action from event_action_map if provided; else fallback to heuristic
                if event_action_map and obs.event_ref in event_action_map:
                    action = event_action_map[obs.event_ref]
                else:
                    action = _infer_action_from_event_ref(obs)
                contradiction_options = _CONTRADICTIONS.get(action)
                if contradiction_options:
                    content = rng.choice(contradiction_options)
                    noise_tags.append("contradiction")
                    confidence = max(0.10, confidence - rng.uniform(0.15, 0.30))
            elif obs.modality == "video":
                content = rng.choice(_CONTRADICTION_VIDEO_OVERRIDE)
                noise_tags.append("contradiction")
                confidence = max(0.10, confidence - rng.uniform(0.20, 0.40))
            elif obs.modality == "text":
                content = rng.choice(_CONTRADICTION_TEXT_OVERRIDE)
                noise_tags.append("contradiction")
                confidence = max(0.10, confidence - rng.uniform(0.15, 0.30))

        # --- Temporal noise ---
        if rng.random() < cfg.temporal_noise_rate:
            shift = rng.randint(cfg.temporal_noise_min_sec, cfg.temporal_noise_max_sec)
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
        # Only video gets qualifier-based fallback paraphrase; audio and text
        # are left as-is when no _SEMANTIC_VARIANTS entry is found, to avoid
        # appending camera qualifiers to spoken dialogue or message bodies.
        if rng.random() < cfg.semantic_variation_rate and "contradiction" not in noise_tags:
            new_content = _paraphrase(content, rng, modality=obs.modality)
            if new_content != content:
                content = new_content
                noise_tags.append("semantic_variation")

        # Build final observation dict (no canonical_entity)
        final = obs.to_final_dict(noise_tags=noise_tags)
        final["content"] = content
        final["timestamp"] = timestamp
        final["time_offset"] = time_offset
        final["confidence"] = round(confidence, 3)

        results.append(final)

    # Restore original ordering by obs_id numeric suffix
    results.sort(key=lambda x: int(x["obs_id"][1:]))
    return results


def _infer_action_from_event_ref(obs: CleanObservation) -> str:
    """
    Best-effort action inference from obs content for contradiction lookup.
    In a full system this would use the event_ref → event lookup.
    We use content keyword matching as a fallback.
    """
    content_lower = obs.content.lower()
    keyword_action_map = {
        "enter":     "enter_atm",
        "inside":    "enter_atm",
        "withdraw":  "withdraw_cash",
        "transaction": "withdraw_cash",
        "exit":      "exit_atm",
        "leave":     "exit_atm",
        "steal":     "steal_items",
        "transfer":  "steal_data",
        "tamper":    "tamper_atm",
        "approach":  "approach_atm",
        "message":   "initiate_communication",
        "plan":      "confirm_plan",
        "files":     "steal_items",
    }
    for keyword, action in keyword_action_map.items():
        if keyword in content_lower:
            return action
    return "unknown_action"
