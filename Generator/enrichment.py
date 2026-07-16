"""
ForenSynth-X+ LLM Enrichment (Cohere)
Single-call enrichment that rewrites all natural language surfaces
in one batched Cohere API request.

What gets rewritten (one API call per case):
    - fir.description  : formal Indian police register language
    - fir.location     : expanded Bengaluru address-style string
    - observations[].content : richer, modality-aware natural language
    - observations[].source  : expanded source label descriptor

What is NEVER touched:
    ground_truth (entities, events, entity_mapping), timestamps,
    time_offset, confidence, noise_tags, obs_id, event_ref,
    entity, role, modality — and all schema structural fields.

Usage:
    python main.py --enrich
    python main.py --enrich --api-key co-...
    # or set COHERE_API_KEY env variable

Model:
    Default: command-a-03-2025 (Cohere flagship, early 2026).
    Override via enrich_case(model="...") if needed.

Design:
    One prompt → one JSON response → apply to deep copy of case.
    On any failure, returns original case unchanged with a warning.
"""

import copy
import json
import urllib.error
import urllib.request

_COHERE_URL = "https://api.cohere.com/v2/chat"
# command-r-plus was retired 2025-09-15.
# command-a-03-2025 is the current Cohere flagship (as of early 2026).
# Pinned date-suffixed aliases (e.g. command-r-plus-08-2024) are also safe
# to use if you need the older model family specifically.
_DEFAULT_MODEL = "command-a-03-2025"

# ---------------------------------------------------------------------------
# Prompt context per domain+template
# ---------------------------------------------------------------------------

_PROMPT_CONTEXT: dict[str, str] = {
    "ATM_Robbery:Entry_Action_Exit": (
        "a lone individual who entered an ATM booth and conducted a suspicious transaction"
    ),
    "ATM_Robbery:Entry_Suspicious_Exit": (
        "an individual who loitered near an ATM and tampered with the card slot — suspected skimming"
    ),
    "ATM_Robbery:MultiActor_Entry_Action_Exit": (
        "two individuals acting in coordination at an ATM — one inside, one standing watch"
    ),
    "Office_Theft:Entry_HiddenAction_Exit": (
        "an unknown individual who gained unauthorised access after hours and removed documents"
    ),
    "Office_Theft:Entry_LegitAction_HiddenAction_Exit": (
        "a registered employee who performed normal duties then exfiltrated data — insider threat"
    ),
    "Communication:Communication_Action_Consequence": (
        "a single individual who planned and confirmed an illegal act via digital channels"
    ),
    "Communication:MultiActor_Communication": (
        "multiple individuals coordinating criminal activity across encrypted platforms"
    ),
}

# Domain-aware source label hints.
# Each domain maps raw source labels to realistic descriptors that reflect
# what that source actually represents in that specific crime context.
_SOURCE_CONTEXT: dict[str, dict[str, str]] = {
    "ATM_Robbery": {
        "camera_1":           "fixed CCTV camera mounted at the ATM booth entrance",
        "camera_2":           "secondary CCTV camera covering the ATM kiosk exterior",
        "camera_3":           "wide-angle CCTV camera surveilling the ATM street frontage",
        "cctv_entrance":      "entrance-facing CCTV unit at the ATM lobby door",
        "cctv_atm":           "ATM-integrated camera directly facing the card reader and keypad",
        "mic_booth":          "audio capture unit mounted inside the ATM enclosure",
        "phone_record":       "call log recovered from a phone found near the ATM",
        "witness_statement":  "verbal statement recorded from a bystander at the ATM location",
        "intercepted_call":   "intercepted call placed in the vicinity of the ATM",
        "email_log":          "email log flagged in relation to the ATM incident",
        "sms_record":         "SMS exchange associated with the ATM robbery",
        "complaint_register": "formal complaint entry filed at the local police station",
        "incident_report":    "bank security incident report submitted post-event",
    },
    "Office_Theft": {
        "camera_1":           "CCTV camera covering the main office corridor",
        "camera_2":           "secondary CCTV camera positioned near the server room entrance",
        "camera_3":           "third-floor CCTV unit covering the accounts department area",
        "cctv_entrance":      "lobby-mounted CCTV camera at the office building main entrance",
        "cctv_atm":           "internal CCTV unit covering the cash handling area",
        "mic_booth":          "audio recording unit in the security monitoring room",
        "phone_record":       "call log extracted from a device seized on the office premises",
        "witness_statement":  "written statement provided by a staff member or colleague",
        "intercepted_call":   "internal call intercepted via the office PBX system",
        "email_log":          "corporate email server log flagged by IT security audit",
        "sms_record":         "SMS record associated with the accused employee",
        "complaint_register": "HR complaint register entry filed by a colleague",
        "incident_report":    "formal security incident report submitted by the facilities team",
    },
    "Communication": {
        "camera_1":           "CCTV camera at the location where the suspect was traced",
        "camera_2":           "secondary surveillance camera near the suspect known address",
        "camera_3":           "public CCTV unit in the area where the device was active",
        "cctv_entrance":      "entrance camera at the premises linked to the communication",
        "cctv_atm":           "ATM-area CCTV capturing movement near the suspect location",
        "mic_booth":          "audio interception unit deployed at a monitored location",
        "phone_record":       "intercepted outbound call record from an unregistered SIM",
        "witness_statement":  "verbal account from a third party who received the communication",
        "intercepted_call":   "call intercepted during active network surveillance",
        "email_log":          "corporate or personal email log flagged by cyber forensics",
        "sms_record":         "SMS chain extracted from a seized or monitored device",
        "complaint_register": "complaint filed by a recipient of the suspicious communication",
        "incident_report":    "cyber cell incident report initiated after the tip-off",
    },
}

# Fallback for any unlisted domain
_SOURCE_CONTEXT_FALLBACK: dict[str, str] = {
    "camera_1":           "fixed CCTV camera at the scene",
    "camera_2":           "secondary surveillance camera at the location",
    "camera_3":           "third surveillance camera covering the area",
    "cctv_entrance":      "CCTV camera at the building entrance",
    "cctv_atm":           "ATM-mounted surveillance camera",
    "mic_booth":          "audio capture unit at the location",
    "phone_record":       "recorded phone call log",
    "witness_statement":  "witness verbal statement",
    "intercepted_call":   "intercepted telephone call",
    "email_log":          "email communication log",
    "sms_record":         "SMS message record",
    "complaint_register": "official complaint register entry",
    "incident_report":    "formal incident report",
}


def _get_source_hints(domain: str) -> str:
    """Return formatted source hint string for the given domain."""
    ctx = _SOURCE_CONTEXT.get(domain, _SOURCE_CONTEXT_FALLBACK)
    return "\n".join(f'  "{k}" -> {v}' for k, v in ctx.items())


# ---------------------------------------------------------------------------
# Core API call
# ---------------------------------------------------------------------------

def _cohere_chat(
    prompt: str,
    api_key: str,
    model: str = _DEFAULT_MODEL,
    max_tokens: int = 1000,
) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }).encode("utf-8")

    req = urllib.request.Request(
        _COHERE_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["message"]["content"][0]["text"].strip()


# ---------------------------------------------------------------------------
# Single-call enrichment
# ---------------------------------------------------------------------------

def enrich_case(
    case: dict,
    api_key: str,
    model: str = _DEFAULT_MODEL,
) -> dict:
    """
    Enrich all natural language surfaces in a case file using a
    single Cohere API call.

    Rewrites:
        fir.description, fir.location,
        observations[].content, observations[].source

    Never touches:
        ground_truth, timestamps, time_offset, confidence,
        noise_tags, obs_id, event_ref, entity, role, modality.

    Args:
        case:    Fully assembled case dict from ForenSynthGenerator.
        api_key: Cohere API key (co-...).
        model:   Cohere model (default: command-a-03-2025).

    Returns:
        Deep copy of case with enriched fields.
        On any failure, returns the original case unchanged.
    """
    fir = case["fir"]
    domain = case["domain"]
    template = case["template"]
    context_key = f"{domain}:{template}"
    crime_context = _PROMPT_CONTEXT.get(context_key, f"a {fir['crime_type']} incident")

    observations = case.get("observations", [])

    # Build compact obs list — only what the model needs to rewrite
    obs_input = [
        {
            "id": obs["obs_id"],
            "modality": obs["modality"],
            "role": obs["role"],
            "source": obs["source"],
            "location": obs["location"],
            "content": obs["content"],
            "has_contradiction": "contradiction" in obs.get("noise_tags", []),
        }
        for obs in observations
    ]

    source_hints = _get_source_hints(domain)

    prompt = (
        f"You are enriching a synthetic forensic case file. "
        f"Return ONLY a single JSON object — no explanation, no markdown fences.\n\n"
        f"Case context:\n"
        f"  Domain          : {domain}\n"
        f"  Template        : {template}\n"
        f"  Crime type      : {fir['crime_type']}\n"
        f"  Current location: {fir['location']}\n"
        f"  Nature          : {crime_context}\n"
        f"  Suspects        : {fir['roles']['suspect']}\n"
        f"  Witnesses       : {fir['roles']['witness']}\n"
        f"  Current FIR description (reference only): {fir['description']}\n\n"
        f"Source label hints:\n{source_hints}\n\n"
        f"Return this exact JSON structure:\n"
        f"{{\n"
        f'  "fir_description": "...",\n'
        f'  "fir_location": "...",\n'
        f'  "observations": [{{"id": "O1", "content": "...", "source": "...", "location": "..."}}, ...]\n'
        f"}}\n\n"
        f"Rules for fir_description:\n"
        f"- 2 to 3 sentences, formal Indian police register\n"
        f"- Use: complainant, accused, deponent, corroborate\n"
        f"- Do NOT name individuals, do NOT contradict suspect/witness counts\n\n"
        f"Rules for fir_location:\n"
        f"- Realistic Bengaluru address-style string, under 15 words\n"
        f"- Do NOT invent a specific named bank branch or ATM brand\n\n"
        f"Rules for each observation content field (modality-specific):\n"
        f"- video: what the camera physically sees — third-person visual description of visible actions ONLY, under 25 words.\n"
        f"  Do NOT infer intent, encryption, app names, message content, or anything not directly visible on camera.\n"
        f"  Good: 'Person seen holding phone to ear.' Bad: 'Suspect sends message via encrypted channel.'\n"
        f"- audio: the actual spoken words captured — realistic first-person dialogue transcript, under 20 words.\n"
        f"  Write the words spoken, not a description of the call.\n"
        f"- text: the literal raw body of the message, SMS, email, or log entry, under 30 words.\n"
        f"  Write the actual message content, not a description of it.\n"
        f"  Good: 'It is set. Move at 21:30. Confirm when ready.' Bad: 'Coded messages exchanged between parties.'\n"
        f"- If has_contradiction is true: for audio/text preserve denial/contradiction tone; for video use footage-quality doubt language.\n"
        f"- source: expand the raw label into a specific realistic descriptor, under 12 words\n"
        f"- location: refine the spatial location into a precise realistic description, under 12 words\n"
        f"  e.g. ATM booth entrance exterior facing -> exterior CCTV mount above ATM booth door\n"
        f"  Do NOT invent building names or ATM brand names\n"
        f"- Keep the same number of observations in the same order\n"
        f"- Do NOT change id values\n\n"
        f"Observations to enrich:\n{json.dumps(obs_input, indent=2)}"
    )

    # Scale tokens: FIR (~150) + per-observation (~60 each)
    max_tokens = 200 + len(observations) * 75  # extra budget for location field

    try:
        raw = _cohere_chat(prompt, api_key, model=model, max_tokens=max_tokens)

        # Strip markdown fences if model adds them
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
        if clean.endswith("```"):
            clean = "\n".join(clean.split("\n")[:-1])
        clean = clean.strip()

        parsed = json.loads(clean)

        # Validate required keys
        if "fir_description" not in parsed or "observations" not in parsed:
            raise ValueError("Response missing required keys")

        enriched_description = parsed["fir_description"].strip()
        enriched_location = parsed.get("fir_location", "").strip()
        enriched_obs_list = parsed["observations"]

        if not enriched_description:
            raise ValueError("Empty fir_description in response")
        if not isinstance(enriched_obs_list, list):
            raise ValueError("observations is not a list")

        # Build obs lookup: id -> {content, source, location}
        obs_map: dict[str, dict] = {}
        for item in enriched_obs_list:
            oid = item.get("id")
            if oid:
                obs_map[oid] = {
                    "content":  item.get("content", "").strip(),
                    "source":   item.get("source", "").strip(),
                    "location": item.get("location", "").strip(),
                }

        # Apply to deep copy — never mutate original
        enriched_case = copy.deepcopy(case)
        enriched_case["fir"]["description"] = enriched_description
        if enriched_location:
            enriched_case["fir"]["location"] = enriched_location

        enriched_count = 0
        for obs in enriched_case["observations"]:
            update = obs_map.get(obs["obs_id"])
            if update:
                if update["content"]:
                    obs["content"] = update["content"]
                if update["source"]:
                    obs["source"] = update["source"]
                if update["location"]:
                    obs["location"] = update["location"]
                enriched_count += 1

        enriched_case["fir"]["enriched"] = True
        print(f"[enrich] Done — FIR + {enriched_count}/{len(observations)} observations enriched.")
        return enriched_case

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"[enrich] HTTP {e.code}: {body[:200]} — using original case")
        return case
    except urllib.error.URLError as e:
        print(f"[enrich] Network error: {e.reason} — using original case")
        return case
    except (KeyError, IndexError, json.JSONDecodeError, ValueError) as e:
        print(f"[enrich] Parse error: {e} — using original case")
        return case
    except Exception as e:
        print(f"[enrich] Unexpected error: {e} — using original case")
        return case
