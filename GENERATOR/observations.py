"""
ForenSynth-X+ Observations
Expands ground truth events into clean, modality-aware observations
(before noise injection).

Modality content rules:
    video  — Raw visual description of what a camera physically captures.
              No inferences about intent, encryption, digital channels, etc.
              Third-person, present-tense camera-observation language.
    audio  — The actual spoken words / call transcript content captured
              by the audio source. First-person dialogue.
    text   — The literal raw content of the message, SMS, email, or log
              entry. Not a description of it — the message itself.
"""

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from config import SOURCE_LABELS, get_observation_location
from entities import CanonicalEntity
from templates import Template, EventSlot
from timeline import GroundTruthEvent


@dataclass
class CleanObservation:
    """
    A single observation derived from a ground truth event.
    Contains canonical_entity — this is stripped before final output.

    Attributes:
        obs_id: Sequential observation label ("O1", "O2", …).
        event_ref: The event_id this observation corresponds to.
        entity: Alias label (visible in final output).
        canonical_entity: True entity_id (HIDDEN — ground truth only).
        role: Role of the observing/observed entity.
        modality: "video" | "audio" | "text".
        source: Source device/channel label.
        content: Raw evidence content, modality-appropriate.
        timestamp: ISO 8601 datetime string.
        time_offset: Seconds from time-window start.
        confidence: Simulated detection confidence [0.0, 1.0].
    """
    obs_id: str
    event_ref: str
    entity: str
    canonical_entity: str
    role: str
    modality: str
    source: str
    location: str
    content: str
    timestamp: str
    time_offset: int
    confidence: float

    def to_final_dict(self, noise_tags: list[str] | None = None) -> dict:
        """Return public-facing dict (no canonical_entity)."""
        return {
            "obs_id": self.obs_id,
            "event_ref": self.event_ref,
            "entity": self.entity,
            "role": self.role,
            "modality": self.modality,
            "source": self.source,
            "location": self.location,
            "content": self.content,
            "timestamp": self.timestamp,
            "time_offset": self.time_offset,
            "confidence": round(self.confidence, 3),
            "noise_tags": noise_tags or [],
        }


def _get_slot_for_event(
    template: Template, slot_id: str
) -> EventSlot | None:
    for slot in template.slots:
        if slot.slot_id == slot_id:
            return slot
    return None


# ---------------------------------------------------------------------------
# Audio content — spoken dialogue transcripts
# ---------------------------------------------------------------------------

def _build_audio_content(entity: CanonicalEntity, action: str, rng: random.Random) -> str:
    """Return a realistic spoken/call-record transcript for the action."""
    suspect_phrases = {
        "approach_atm": [
            "Yeah I'm almost at the machine.",
            "I can see the ATM now.",
            "Heading to the kiosk, nearly there.",
            "I'm close, give me two minutes.",
            "ATM is just ahead, I can see it.",
            "On my way to the machine now.",
            "Almost at the spot, stay on the line.",
            "I can see the booth from here.",
        ],
        "enter_atm": [
            "Yeah I just got inside the ATM.",
            "I'm in, it's clear.",
            "Inside now, no one else here.",
            "Stepped in, booth is empty.",
            "I'm inside the booth, door's shut.",
            "In the enclosure now, all good.",
            "Just entered, it's quiet in here.",
            "I'm inside, give me a moment.",
        ],
        "withdraw_cash": [
            "It's working, getting the money.",
            "Machine's responding fine.",
            "Transaction going through now.",
            "Card's in, processing.",
            "It accepted it, cash is coming out.",
            "Machine is dispensing, stand by.",
            "Got it, the money's out.",
            "Transaction complete, I have the cash.",
        ],
        "exit_atm": [
            "Done, I'm coming out.",
            "Let's move, quick.",
            "Leaving now, walk fast.",
            "I'm out, don't wait up.",
            "Exiting the booth, meet me outside.",
            "Out of the ATM, heading your way.",
            "Done in here, moving now.",
            "I've left the booth, where are you?",
        ],
        "wait_outside": [
            "I'm outside, watching the entrance.",
            "No one around, keep going.",
            "All clear out here, take your time.",
            "Standing by the entrance, no one nearby.",
            "I'm at the door, nothing suspicious.",
            "Outside and watching, you're good.",
            "No one's coming, carry on.",
            "I've got eyes on the entrance.",
        ],
        "flee_scene": [
            "Move move move, go!",
            "Let's get out of here.",
            "Someone's coming, run!",
            "Go now, don't look back.",
            "Leave everything, just walk fast.",
            "We need to leave right now.",
            "Security's nearby, get out!",
            "Split up, meet at the spot.",
        ],
        "loiter_near_atm": [
            "Just standing around, nothing yet.",
            "Waiting for the right moment.",
            "I'm near the machine, no one here yet.",
            "Still waiting, be patient.",
            "Hanging around, keeping an eye out.",
            "No one's come yet, still waiting.",
            "I'm nearby, timing it.",
            "Almost ready, just watching.",
        ],
        "tamper_atm": [
            "Device is attached.",
            "Skimmer's in place.",
            "It's fitted, looks normal from outside.",
            "Done, you can't tell it's there.",
            "Card reader's set, let's go.",
            "Attachment's secure, no one noticed.",
            "It's on there, looks factory.",
            "All set, the reader's rigged.",
        ],
        "enter_office": [
            "I'm inside the building.",
            "Access card worked fine.",
            "In the lobby now, no one at the desk.",
            "Badge scanned okay, I'm through.",
            "Inside the office, lights are off.",
            "Got in without any issue.",
            "Entry done, heading up now.",
            "I'm in the building, going to the floor.",
        ],
        "navigate_to_target": [
            "Heading to the accounts floor now.",
            "Almost at the server room.",
            "On my way to the restricted section.",
            "Taking the stairs to avoid cameras.",
            "Nearly at the target floor.",
            "Walking through the corridor now.",
            "Passing the main hall, almost there.",
            "I can see the door to the server room.",
        ],
        "steal_items": [
            "Got the files.",
            "Transferring now.",
            "Documents are in the bag.",
            "I have what we need.",
            "Grabbed everything on the list.",
            "Items secured, heading out.",
            "I've got the folders, let's go.",
            "Everything's packed, moving out.",
        ],
        "steal_data": [
            "USB is in, copying.",
            "Data's transferring to the drive.",
            "Transfer at sixty percent.",
            "Almost done copying, stay on the line.",
            "Files are on the drive, pulling it out.",
            "Copy complete, removing the device.",
            "Data's secure on the USB, leaving now.",
            "Transfer finished, wiping the logs.",
        ],
        "exit_office": [
            "Leaving now, see you outside.",
            "I'm out, no one saw me.",
            "Exiting via the stairwell.",
            "Out of the building, heading to the car.",
            "Left the floor, walking to the exit.",
            "I'm through the lobby, nearly out.",
            "Out now, everything went clean.",
            "Leaving the premises, no alarms.",
        ],
        "perform_legit_work": [
            "Just finishing up the report.",
            "In the meeting right now.",
            "Sending the last few emails.",
            "Wrapping up for the day.",
            "Still at my desk, almost done.",
            "On a call with the client.",
            "Filing the last document.",
            "Just catching up on some backlog.",
        ],
        "initiate_communication": [
            "Hey, it's me. You ready?",
            "We need to talk. Call me back.",
            "It's set for tonight, you know what to do.",
            "Are you in position? Let me know.",
            "I'm reaching out as planned.",
            "Check in when you get this.",
            "We're good to move, confirm when ready.",
            "This is the call you were expecting.",
        ],
        "exchange_information": [
            "The timing is confirmed.",
            "You know where to be.",
            "Details are as discussed, nothing has changed.",
            "Location is the same, stick to the plan.",
            "I'm sending you the address now.",
            "Time is fixed, don't be late.",
            "The package is ready on your end.",
            "Everything is confirmed, we proceed tonight.",
        ],
        "confirm_plan": [
            "We're good to go.",
            "Everything is set.",
            "Confirmed, I'll be there.",
            "All parties are ready.",
            "Plan is locked, no changes.",
            "Understood, proceeding as agreed.",
            "I'm ready on my end.",
            "We're aligned, let's move.",
        ],
        "respond_to_communication": [
            "Got your message. Understood.",
            "I'm ready on my end.",
            "Received, I'll follow through.",
            "Message got through, I'm on it.",
            "Confirmed, I'll be in position.",
            "Copy that, standing by.",
            "Understood, no questions.",
            "All clear, proceeding as told.",
        ],
        "coordinate_activity": [
            "Meet at the spot in 20.",
            "Stick to the plan.",
            "You handle your part, I'll handle mine.",
            "Timings are as agreed, don't deviate.",
            "I'll signal when it's clear.",
            "Coordinate with the others, everyone needs to know.",
            "We move together, no one breaks early.",
            "Keep your phone on, I'll update you.",
        ],
        "observe_exit": [
            "Two people just rushed out of the ATM.",
            "I saw them leave the office.",
            "They came out fast and walked off quickly.",
            "A couple of individuals just left the building.",
            "Someone just ran out of the booth.",
            "I noticed them leave in a hurry.",
            "They exited and separated immediately.",
            "Both of them left at once through the side door.",
        ],
        "report_incident": [
            "I need to report something suspicious.",
            "I want to file a complaint.",
            "I witnessed something that didn't look right.",
            "Something happened near the ATM I should report.",
            "I'd like to speak to an officer about what I saw.",
            "I saw something unusual and wanted to let you know.",
            "There was an incident here and I think you should know.",
            "I'm calling to report a suspicious person near the ATM.",
        ],
        "flag_suspicious_comm": [
            "I received a strange message.",
            "Someone sent me something odd.",
            "A message came through that felt off.",
            "I got a communication that seemed suspicious.",
            "This message doesn't look like it was meant for me.",
            "Someone forwarded me something I think you should see.",
            "I received a text that may be related to criminal activity.",
            "A message arrived on my phone I can't explain.",
        ],
        "intercept_communication": [
            "This message doesn't look right.",
            "I think this was meant for someone else.",
            "I intercepted something that looked suspicious.",
            "A communication came through I believe was misdirected.",
            "I received a message I was clearly not meant to see.",
            "This exchange looks like it's planning something illegal.",
            "I've picked up a communication that concerns me.",
            "Something came through our system that shouldn't have.",
        ],
    }
    phrases = suspect_phrases.get(action, [f"Regarding the {action.replace('_', ' ')}."])
    return rng.choice(phrases)


# ---------------------------------------------------------------------------
# Text content — raw message / SMS / email / log body
# ---------------------------------------------------------------------------

_TEXT_CONTENT: dict[str, list[str]] = {
    "approach_atm": [
        "Almost there. One min.",
        "On my way to the spot now.",
        "Walking up to it now. Clear?",
        "Heading there. Any activity?",
        "At the corner, about to move.",
    ],
    "enter_atm": [
        "Inside. Booth is empty.",
        "In. Door shut. No one here.",
        "Entered. Starting now.",
        "I'm in the booth. All good.",
        "Booth clear. Proceeding.",
    ],
    "withdraw_cash": [
        "Card working. Cash out now.",
        "Transaction done. Got it all.",
        "Machine accepted. Stand by.",
        "Done. Moving to step two.",
        "It went through. Full amount.",
    ],
    "exit_atm": [
        "Out now. Walk.",
        "Leaving. Meet at point B.",
        "Done here. Moving.",
        "Exited. Don't wait for me.",
        "Out of the booth. Heading your way.",
    ],
    "wait_outside": [
        "Outside. All clear. Go ahead.",
        "Watching the entrance. No one coming.",
        "Standing by. You're good.",
        "Nothing suspicious. Continue.",
        "Door clear. No security visible.",
    ],
    "flee_scene": [
        "Leave now. Don't look back.",
        "Go. Someone's watching.",
        "Abort. Get out fast.",
        "Split. Meet at fallback.",
        "Security nearby. Move.",
    ],
    "loiter_near_atm": [
        "Still waiting. Not yet.",
        "Timing not right. Holding.",
        "Few more minutes. Stay ready.",
        "Not clear yet. Stand by.",
        "Almost. Give it 5 more.",
    ],
    "tamper_atm": [
        "Device placed. Looks stock.",
        "Reader fitted. No one saw.",
        "Done. Collecting tomorrow.",
        "Skimmer set. Walk away normal.",
        "All fitted. Head out casual.",
    ],
    "enter_office": [
        "Badge worked. I'm in.",
        "Inside. Lobby empty.",
        "Got in. Heading up.",
        "Entry clear. No guard at desk.",
        "Access granted. Moving to floor.",
    ],
    "navigate_to_target": [
        "On the floor. Almost at the room.",
        "Corridor clear. Moving.",
        "Taking stairs. No cameras this way.",
        "At the door. Give me 2 mins.",
        "Server room in sight.",
    ],
    "steal_items": [
        "Got the folders. In the bag.",
        "Files secured. Leaving now.",
        "Everything's packed. Moving out.",
        "Took what was on the list.",
        "Documents with me. Heading to exit.",
    ],
    "steal_data": [
        "Copy at 80%. Almost done.",
        "USB full. Pulling out.",
        "Transfer complete. Wiping history.",
        "Files copied. Removing drive.",
        "Done. 4.2GB on the stick.",
    ],
    "exit_office": [
        "Out. Clean exit.",
        "Left through stairwell. No issues.",
        "I'm outside. No alarms.",
        "Exited clean. Heading to car.",
        "Left the premises. No one saw me.",
    ],
    "perform_legit_work": [
        "Working late. Finishing the Henderson report.",
        "Still at my desk. Catching up on emails.",
        "On a client call. Back in 30.",
        "Filing the Q3 audit docs.",
        "Wrapping up. Will send by EOD.",
    ],
    "initiate_communication": [
        "It's me. You ready for tonight?",
        "Call me when you see this.",
        "We move as planned. Confirm.",
        "Reaching out as discussed. Reply.",
        "You know what to do. Ready?",
    ],
    "exchange_information": [
        "Location: MG Road exit, south side. Time: 21:30.",
        "Signal is three flashes. Respond with two.",
        "Package at the usual drop. Pick up by 10.",
        "Meeting point changed. New address to follow.",
        "Target leaves at 9. You have a 15-minute window.",
    ],
    "confirm_plan": [
        "Confirmed. Moving tonight.",
        "All set. Don't contact again until after.",
        "Good to go. Everyone's ready.",
        "Plan locked. No changes.",
        "Proceed as discussed. Ready on my end.",
    ],
    "respond_to_communication": [
        "Got it. Will be there.",
        "Understood. In position by 9.",
        "Confirmed. Standing by.",
        "Received. No questions.",
        "Copy. Proceeding as instructed.",
    ],
    "coordinate_activity": [
        "You take the west side. I'll handle east.",
        "Meet at the parking bay in 20.",
        "Keep your phone on. I'll update timing.",
        "Stick to your part. I'll handle mine.",
        "If anything changes, abort and wait for my signal.",
    ],
    "observe_exit": [
        "Two men just ran out of the ATM. Looked suspicious.",
        "Saw someone leave in a hurry. Looked back twice.",
        "Person exited ATM very fast. Didn't look right.",
        "Two individuals left together quickly. One carrying a bag.",
        "Saw a man rush out and walk away fast. Seemed nervous.",
    ],
    "report_incident": [
        "I want to report suspicious activity at the ATM on Brigade Road. Two men behaved oddly and left in a hurry.",
        "Filing complaint: saw someone tamper with the card slot at the ATM near Koramangala.",
        "Reporting: I received a suspicious message I believe is related to criminal activity.",
        "Complaint submitted: witnessed unauthorised access to the server room after hours.",
        "Writing to report: I saw a person remove files from the restricted section of the office.",
    ],
    "flag_suspicious_comm": [
        "I received a message not meant for me. Contents look like a plan for something illegal.",
        "Someone sent me coordinates and a time. I think it was misdirected.",
        "Got a message saying 'confirm your position'. I don't know the sender.",
        "Received an SMS with what looks like instructions for a meeting. Forwarding to you.",
        "A message came in from an unknown number referencing a 'drop'. Reporting it.",
    ],
    "intercept_communication": [
        "[INTERCEPTED — Network Monitoring System]\nFrom: +91-XXXXXXXXXX\nTo: +91-XXXXXXXXXX\nContent: Ready at 9. Go on my signal.",
        "[FLAGGED — Cyber Cell]\nMessage chain between two unregistered numbers: 'Location confirmed. Move at 21:15.'",
        "[CAPTURED — PBX Intercept]\nCall recording excerpt: 'Package is at the south exit. Pick it up now.'",
        "[SYSTEM LOG — Firewall Alert]\nOutbound connection to unregistered IP. Data volume: 4.1GB. Duration: 8 min.",
        "[INTERCEPTED — Active Surveillance]\nSMS: 'All clear. Proceed as planned. No changes.'",
    ],
}

_TEXT_CONTENT_FALLBACK = [
    "Message content unavailable — log entry incomplete.",
    "Record extracted. Content redacted pending review.",
    "Communication log entry. Details under analysis.",
]


def _build_text_content(action: str, rng: random.Random) -> str:
    """Return the literal raw text/message/log content for this action."""
    options = _TEXT_CONTENT.get(action, _TEXT_CONTENT_FALLBACK)
    return rng.choice(options)


# ---------------------------------------------------------------------------
# Main expansion function
# ---------------------------------------------------------------------------

def expand_events_to_observations(
    events: list[GroundTruthEvent],
    entities: list[CanonicalEntity],
    template: Template,
    rng: random.Random,
    domain: str = "unknown",
) -> list[CleanObservation]:
    """
    Convert each ground truth event into one or more clean observations
    across available modalities for the entity involved.

    Modality rules enforced here:
        video  — uses description_templates from the template slot (camera-capture language).
                 Templates must describe only what is physically visible.
        audio  — uses _build_audio_content (spoken dialogue transcript).
        text   — uses _build_text_content (raw message/SMS/email/log body).

    Args:
        events: Ordered list of ground truth events.
        entities: All canonical entities for this case.
        template: The template used (for slot content templates).
        rng: Seeded random instance.
        domain: Domain string used for location lookup (e.g. "ATM_Robbery").

    Returns:
        Ordered list of CleanObservation objects.
    """
    entity_map = {e.entity_id: e for e in entities}
    observations: list[CleanObservation] = []
    obs_counter = 1

    for event in events:
        entity = entity_map.get(event.entity_id)
        if not entity:
            continue

        slot = _get_slot_for_event(template, event.slot_id)
        available_modalities = list(entity.aliases.keys())

        for modality in available_modalities:
            alias = entity.aliases[modality]
            source = rng.choice(SOURCE_LABELS[modality])

            # --- Content by modality ---
            if modality == "audio":
                # Actual spoken dialogue / call transcript
                content = _build_audio_content(entity, event.action, rng)

            elif modality == "text":
                # Literal raw message, SMS, email, or log body
                content = _build_text_content(event.action, rng)

            elif modality == "video":
                # What the camera physically sees — use visual description_templates
                if slot and slot.description_templates:
                    content = rng.choice(slot.description_templates)
                else:
                    content = f"{event.action.replace('_', ' ').capitalize()}."
            else:
                # Fallback for any future modality
                if slot and slot.description_templates:
                    content = rng.choice(slot.description_templates)
                else:
                    content = f"{event.action.replace('_', ' ').capitalize()}."

            # Confidence by role + modality
            base_conf = {
                ("suspect", "video"): (0.75, 0.95),
                ("suspect", "audio"): (0.60, 0.85),
                ("suspect", "text"):  (0.65, 0.90),
                ("witness", "video"): (0.40, 0.70),
                ("witness", "audio"): (0.70, 0.90),
                ("witness", "text"):  (0.65, 0.85),
            }.get((entity.role, modality), (0.50, 0.80))

            confidence = rng.uniform(*base_conf)

            # Small timestamp jitter per modality (clean layer — minimal)
            clean_jitter = rng.randint(0, 3)
            event_dt = datetime.fromisoformat(event.timestamp)
            obs_dt = event_dt + timedelta(seconds=clean_jitter)

            # Resolve spatial location from domain + modality + source
            location = get_observation_location(domain, modality, source)

            observations.append(CleanObservation(
                obs_id=f"O{obs_counter}",
                event_ref=event.event_id,
                entity=alias,
                canonical_entity=entity.entity_id,
                role=entity.role,
                modality=modality,
                source=source,
                location=location,
                content=content,
                timestamp=obs_dt.isoformat(),
                time_offset=event.time_offset + clean_jitter,
                confidence=confidence,
            ))
            obs_counter += 1

    return observations
