"""Prompt for the vision model. The model only reports what it sees; the agent maps that to fields."""
import json

from .tools import tool_catalogue

from .config import CAMERA_VIEW_TO_SOURCE, VIDEO_QUALITY_BANK, plural_sentences

SYSTEM = """You analyse one CCTV clip for a forensic evidence dataset and report only what the camera
physically shows. You never infer intent, guilt, or anything not visible.

RULES
1. List one event per person per distinct case-relevant action (approaching the ATM, entering,
   using the machine, touching the card slot area, waiting/lingering, exiting, hurrying away,
   watching others leave, reporting to someone or on a phone). People acting together are the
   exception: see rule 7. Ignore background pedestrians and traffic that do nothing case-relevant.
   One continuous movement is ONE event, named by where it ends: a person who walks up to the
   booth and goes straight in is one enter_atm event, not approach_atm + enter_atm; someone who
   comes out and keeps walking is one exit_atm event. Report two events for one person only when
   they clearly stop, wait or do something else in between.
2. For each event choose the action key and the index of the closest vocabulary sentence from the
   VOCABULARY. Pick the sentence that describes the visible action most literally. Prefer neutral
   subject words (Person / Individual / Subject) unless several actors are on screen and an ordinal
   ("First individual", "Second person") is needed. Use "free_content" only if no sentence fits.
3. Read the burned-in overlay exactly as displayed: the camera label and the date/time. If digits are
   unreadable or change inconsistently, return null rather than guessing.
4. Identity: describe each person by clothing, headwear, bag, build. Set "identifiable" to false if
   clothing and face cannot be made out (back view, occluded, too small, too blurry). Only set
   "registry_match" when the person clearly matches a KNOWN PEOPLE entry on clothing AND accessories.
5. footage_issue: set one key if the footage itself prevents seeing an action clearly, otherwise null.
6. certainty = how sure you are the action happened as labelled; visibility = how clearly the
   person and action can be seen. Both 0 to 1. Low-quality, occluded or distant views get low values.
7. Groups: if two or more people perform the same case-relevant action together (same moment, side
   by side, or clearly as a group), report ONE event for all of them, not one event per person.
   Set "group": true, list every member in "people" with the lead first, set "person" to the lead
   (the first one fully visible), and choose a plural / multi-actor sentence. The plural sentences
   are listed under PLURAL SENTENCES; they start with "Two ..." or "Both ...", e.g. "Two individuals
   are recorded leaving the scene side by side." They may sit under a different action key than the
   single-person ones: two people leaving together is flee_scene (hurried or not), because exit_atm
   only has single-person sentences. If no plural sentence exists for the action, keep the closest
   sentence. A person acting alone never gets a plural sentence: one person hurrying away is
   exit_atm. A person who only watches, without doing the action, stays a separate witness event:
   a witness stays behind or goes their own way. Someone who waits or looks on and then walks off
   WITH the person they were looking at is a member of the group, never a witness. When one person
   comes out of the booth, joins another who is outside, and they move off together, that whole
   sequence is ONE group leaving event; do not add a separate exit event for the one who came out.
   If you do report a witness watching someone leave, set "stays_behind": true on that witness
   event, and only after checking in the frames that they stay put or go their own way.

HOW YOU WORK
You have a budget of at most THREE replies for this clip. Each reply is ONE JSON object.
  Reply 1: either your final answer
             {"thought": "<short reason>", "final": <OUTPUT FORMAT object>}
           or ONE batch of tool calls, run in the order given (up to 8):
             {"thought": "<short reason>", "tools": [{"tool": "<name>", "args": {...}}, ...]}
  Reply 2: your final answer. No more tools: all tool results come back together before it.
  Reply 3: only if your final answer fails the field-rule checks. The problems are sent back and
           you give the corrected final answer, still without tools.
Tools are only available in reply 1, so put every tool you need in that one batch.

Plan the batch so it settles everything at once:
  a. Watch the clip. Note where the overlay is and what happens.
  b. Clock: the CLOCK CHECK below says whether OCR has already checked the clock. If it is
     VERIFIED or FROZEN, the clock is settled: skip clock tools (FROZEN means clock_verified=false
     and overlay timestamps null). Only if it is UNCLEAR: read the burned-in clock yourself at 2-3
     times spread across the clip and put those readings into check_clock, adding crop_overlay on
     the overlay corner at the same times. Only a CONSISTENT result counts as verified. If the
     clock is unreadable or INCONSISTENT, set clock_verified=false and report overlay timestamps as
     null. Never guess digits.
  c. If an action is unclear (hands at the card slot, someone exiting), add get_frames with a crop,
     or rewatch the segment at a higher fps.
  d. If two or more people are involved, settle rule 7 before writing events: add get_frames at
     3-4 times across the middle and second half of the clip to see where each person goes. If
     they move off together in the same direction it is ONE group event, even if one of them
     stood looking at the other first.
  e. If someone may have been seen before, add known_people and compare with the reference stills
     before setting registry_match. If in doubt, leave it null.
  f. End the batch with validate_draft on your draft answer (it runs after check_clock, so the
     clock checks apply). Fix what it reports in reply 2.
If the clip is clear and there is no clock to verify, answer final in reply 1."""

OUTPUT_FORMAT = {
    "overlay_camera_label": "text of the camera label in the overlay, or null",
    "overlay_timestamp_first": "YYYY-MM-DD HH:MM:SS shown on the first frame, or null",
    "camera_view": "one of: " + ", ".join(list(CAMERA_VIEW_TO_SOURCE) + ["unknown"]),
    "footage_issue": "null or one of: " + ", ".join(VIDEO_QUALITY_BANK),
    "image_quality": "0-1",
    "clock_verified": "true only if check_clock returned CONSISTENT",
    "clock_note": "one line on how the clock was read, or why it could not be",
    "events": [{
        "person": "clothing / headwear / bag / build (the lead person if group)",
        "identifiable": "true or false",
        "registry_match": "a KNOWN PEOPLE label, or null",
        "action": "an action key from VOCABULARY",
        "template_index": "integer index of the chosen sentence, or null",
        "free_content": "null, or one literal camera-language sentence if nothing fits",
        "start": "MM:SS into the clip when the action starts",
        "overlay_timestamp_at_start": "YYYY-MM-DD HH:MM:SS shown when the action starts, or null",
        "certainty": "0-1",
        "visibility": "0-1",
        "ref_time": "seconds into the clip where this person is seen most clearly",
        "bbox": "[x, y, w, h] 0-1 fractions around the person at ref_time, or null",
        "group": "true if two or more people do this action together (rule 7), otherwise false",
        "stays_behind": "only for a witness watching someone leave: true if you checked that this "
                        "person stays behind or goes their own way; otherwise null",
        "people": "null, or for a group every member with the lead first: "
                  "[{person, identifiable, registry_match, ref_time, bbox}, ...]",
    }],
}

CAMERA_VIEWS = {
    "exterior_above_entrance": "outside, mounted above the kiosk door, looking down at door and pavement",
    "exterior_street_side":    "outside, side-on view of the kiosk from the street",
    "wide_street_frontage":    "across the street, wide angle, whole kiosk frontage, people small",
    "lobby_doorway":           "at the lobby doorway, framing the entry/exit point",
    "interior_card_reader":    "inside the kiosk, looking at the keypad and card reader",
}


def clock_check_text(clock):
    """What the OCR pre-check (tools.ClipTools.ocr_clock) found, for the model."""
    status = (clock or {}).get("status")
    if status == "consistent":
        return (f"VERIFIED by OCR: the clock ticks with the video; clock at first frame = "
                f"{clock['clock_at_first_frame']} ({clock['corner']} corner). Set clock_verified=true.")
    if status == "frozen":
        return ("FROZEN: OCR shows the clock stuck on one time while the video moves on. Set "
                "clock_verified=false and report overlay timestamps as null.")
    return f"UNCLEAR ({(clock or {}).get('note', 'not checked')}): verify the clock yourself (step b)."


def budget_text(max_calls):
    """The reply budget, when it is smaller than the three replies the system prompt describes."""
    if max_calls <= 1:
        return ("BUDGET: you have only ONE reply for this clip. Tools are NOT available: give your final "
                "answer straight away.")
    if max_calls == 2:
        return ("BUDGET: you have TWO replies; there is no third reply to fix a failed answer, so use "
                "validate_draft in your tool batch.")
    return None


def build_user_prompt(vocab, registry_people, clock=None, max_calls=3):
    vocab_view = {a: {"role": v["role"], "sentences": dict(enumerate(v["templates"]))} for a, v in vocab.items()}
    known = {label: p["appearance"] for label, p in registry_people.items() if p.get("identifiable", True)}
    return "\n\n".join([
        "VOCABULARY (action -> numbered sentences):\n" + json.dumps(vocab_view, ensure_ascii=False),
        "PLURAL SENTENCES (action -> sentence numbers, for group events only):\n"
        + json.dumps(plural_sentences(vocab), ensure_ascii=False),
        "CAMERA VIEWS:\n" + json.dumps(CAMERA_VIEWS, ensure_ascii=False),
        "KNOWN PEOPLE (seen in earlier clips of this case):\n" + (json.dumps(known, ensure_ascii=False) if known else "none yet"),
        "CLOCK CHECK:\n" + clock_check_text(clock),
        "TOOLS:\n" + tool_catalogue(),
        "OUTPUT FORMAT (the value of \"final\"):\n" + json.dumps(OUTPUT_FORMAT, ensure_ascii=False),
        *([budget_text(max_calls)] if budget_text(max_calls) else []),
        "Start with step a. Reply with one JSON object (reply 1).",
    ])
