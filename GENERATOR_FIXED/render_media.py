#!/usr/bin/env python3
"""
ForenSynth - render_media.py
Turn generated case files (cases/CASE_ATM_XXX.json) into real evidence media:

  audio observations -> WAV recordings (edge-tts voice + per-source acoustics via ffmpeg)
  video observations -> CCTV clips    (Veo text-to-video + CCTV degradation + clock overlay)
  text observations  -> left as text  (the text agent reads them directly)

What comes out matches what the field agents expect:

  media/CASE_ATM_003/
    audio/O2.wav ...           one recording per audio observation
    audio/clips.json           manifest for agents/field_agents/audio_agent.py --manifest
    video/prompts.json         the Veo prompt for every video observation
    video/raw/O1.mp4 ...       raw clips (from Veo, or ones you made by hand)
    video/O1.mp4 ...           final clips: degraded + "CAM02  2024-01-15 10:01:10" overlay,
                               ready for: python -m video_agent.run --case-id ... --video <dir>
    media_manifest.json        obs_id -> file, voice/character used, how it was made

Consistency rules (so ER has something real to resolve):
  * Every ground-truth entity gets ONE appearance and ONE voice, chosen deterministically from
    the case ID, and with a matching gender, so Person_99 and Speaker_J (same suspect_1) look
    and sound like the same man in every clip.
  * Video clips show the ground-truth EVENT behind the observation (event_ref). Spoken denials
    filed as video (contradiction noise) therefore still get footage of the real action.
  * Footage-quality sentences ("Subject's face not visible ...", "(poor image quality)",
    "(partially obscured)" ...) are turned into the matching visual problem.
  * Generator tags in brackets ("(per call log transcript)" ...) are not spoken.

Usage:
    # audio for every ATM case + Veo prompts (free; no video API calls)
    python GENERATOR_FIXED/render_media.py

    # one case, also generate the video with Veo (paid API, ~8 s per clip)
    python GENERATOR_FIXED/render_media.py --case CASE_ATM_003 --video veo

    # you made the raw clips yourself (e.g. in the Gemini/Flow UI from prompts.json)
    python GENERATOR_FIXED/render_media.py --case CASE_ATM_003 --video raw --raw-videos my_clips/CASE_ATM_003

Requires ffmpeg on PATH, `pip install edge-tts`, and for --video veo: google-genai + GEMINI_API_KEY.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIDEO_AGENT_DIR = ROOT / "agents" / "field_agents" / "video_agent"
sys.path.insert(0, str(VIDEO_AGENT_DIR))

AUDIO_PAD_S = 0.5     # room tone before/after each line
VIDEO_LEAD_S = 2      # the action starts ~2 s into the clip; clock starts this much earlier
DEFAULT_VEO_MODEL = "veo-3.1-fast-generate-preview"

# ----------------------------------------------------------------------------------------------
# Characters
# ----------------------------------------------------------------------------------------------
LOOK = {
    "male": {
        "age": ["in his early 20s", "in his late 20s", "in his 30s", "in his 40s", "in his 50s"],
        "build": ["slim", "stocky", "tall and lean", "heavy-set", "medium-built"],
        "top": ["a black hooded sweatshirt", "an olive-green bomber jacket", "a maroon checked shirt",
                "a grey zip-up jacket", "a navy-blue windcheater", "a white kurta", "a brown leather jacket",
                "a red half-sleeve t-shirt"],
        "bottom": ["grey track pants", "dark blue jeans", "khaki trousers", "black cargo pants", "beige chinos"],
        "shoes": ["white sneakers", "brown boots", "black sandals", "blue running shoes", "black formal shoes"],
        "extra": ["a black backpack", "a black baseball cap", "a blue surgical mask", "a grey beanie",
                  "dark sunglasses", "a sling bag across the chest", "a motorcycle helmet held in one hand"],
    },
    "female": {
        "age": ["in her early 20s", "in her late 20s", "in her 30s", "in her 40s", "in her 50s"],
        "build": ["slim", "short", "tall", "medium-built"],
        "top": ["a yellow salwar kameez with a dupatta", "a denim jacket over a white top", "a black kurti",
                "a maroon cardigan", "a green saree", "a grey hoodie"],
        "bottom": ["black leggings", "blue jeans", "white palazzo pants", "a long skirt"],
        "shoes": ["flat sandals", "white sneakers", "black flats"],
        "extra": ["a tote bag on her shoulder", "a blue surgical mask", "a cloth shopping bag",
                  "her hair in a long braid", "spectacles", "a scarf over her head"],
    },
}


def _rng(*parts) -> random.Random:
    return random.Random(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16))


def build_characters(case: dict) -> dict:
    """entity_id -> {gender, look, pronoun, aliases}. Deterministic per case."""
    chars = {}
    used = set()
    for ent in case["ground_truth"]["entities"]:
        r = _rng(case["case_id"], ent["entity_id"])
        gender = "male" if ent["role"] == "suspect" and r.random() < 0.8 else r.choice(["male", "female"])
        pool = LOOK[gender]
        for _ in range(20):                      # keep two people in one case from dressing alike
            top = r.choice(pool["top"])
            if top not in used:
                break
        used.add(top)
        look = (f"a {r.choice(pool['build'])} {'man' if gender == 'male' else 'woman'} {r.choice(pool['age'])}, "
                + (f"wearing {top} and {r.choice(pool['shoes'])}, " if re.search(r"saree|salwar", top)
                   else f"wearing {top}, {r.choice(pool['bottom'])} and {r.choice(pool['shoes'])}, ")
                + f"with {r.choice(pool['extra'])}")
        chars[ent["entity_id"]] = {
            "gender": gender, "look": look, "role": ent["role"], "aliases": ent.get("aliases", {}),
            "he": "he" if gender == "male" else "she", "his": "his" if gender == "male" else "her",
        }
    return chars


def alias_to_entity(case: dict) -> dict:
    out = {}
    for ent in case["ground_truth"]["entities"]:
        for alias in ent.get("aliases", {}).values():
            out[alias] = ent["entity_id"]
    return out


# ----------------------------------------------------------------------------------------------
# Text cleanup
# ----------------------------------------------------------------------------------------------
TAG_RE = re.compile(r"\s*\((?:[^()]*)\)\s*")


def spoken_text(content: str) -> str:
    """Drop generator annotations like '(per call log transcript)'; keep the sentence."""
    text = TAG_RE.sub(" ", content).strip()
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    text = re.sub(r"([!?])\.+", r"\1", text)
    if text and text[-1] not in ".!?":
        text += "."
    return text


def tags_in(content: str) -> list:
    return [t.strip("() ").lower() for t in re.findall(r"\([^()]*\)", content)]


# ----------------------------------------------------------------------------------------------
# AUDIO
# ----------------------------------------------------------------------------------------------
# source -> (ffmpeg voice chain, noise colour, noise amplitude)
AUDIO_FX = {
    "mic_booth": ("highpass=f=120,aecho=0.8:0.55:23|41:0.32|0.18", "pink", 0.006),
    "phone_record": ("highpass=f=300,lowpass=f=3400,acompressor=threshold=-18dB:ratio=4,"
                     "aresample=8000,aresample=16000", "brown", 0.010),
    "intercepted_call": ("highpass=f=350,lowpass=f=3100,acompressor=threshold=-20dB:ratio=6,aresample=8000,"
                         "aresample=16000,volume='if(between(mod(t,1.9),1.35,1.47),0.05,1)':eval=frame",
                         "white", 0.020),
    "witness_statement": ("highpass=f=90,lowpass=f=7000", "brown", 0.045),
}
DEFAULT_FX = ("highpass=f=100", "pink", 0.006)


async def _list_voices():
    import edge_tts
    return await edge_tts.list_voices()


def assign_voices(case: dict, chars: dict, voices: list) -> dict:
    """entity_id -> {voice, pitch}. Gender-matched to the character; distinct within the case."""
    pref = ["en-IN", "en-GB", "en-AU", "en-US", "en-ZA", "en-NZ", "en-IE", "en-SG"]
    by_gender = {"male": [], "female": []}
    for loc in pref:
        for v in voices:
            if v.get("Locale") == loc and v.get("Gender", "").lower() in by_gender:
                by_gender[v["Gender"].lower()].append(v["ShortName"])
    out, used = {}, set()
    for eid, c in chars.items():
        r = _rng(case["case_id"], eid, "voice")
        pool = by_gender[c["gender"]] or by_gender["male"] or by_gender["female"]
        local = [v for v in pool if v.startswith("en-IN")] or pool
        choices = [v for v in local if v not in used] or [v for v in pool if v not in used] or pool
        voice = choices[r.randrange(len(choices))]
        used.add(voice)
        out[eid] = {"voice": voice, "pitch": f"{r.randint(-8, 6):+d}Hz"}
    return out


def ff(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *map(str, args)], check=True)


async def _tts(text, voice, pitch, rate, path):
    import edge_tts
    await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(str(path))


def render_audio_obs(obs, voice, out_wav: Path):
    text = spoken_text(obs["content"])
    urgent = "!" in text
    rate = "+18%" if urgent else "+0%"
    chain, colour, amp = AUDIO_FX.get(obs["source"], DEFAULT_FX)
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "tts.mp3"
        asyncio.run(_tts(text, voice["voice"], voice["pitch"], rate, raw))
        pad_ms = int(AUDIO_PAD_S * 1000)
        fc = (f"[0:a]aformat=channel_layouts=mono,aresample=16000,{chain},"
              f"adelay={pad_ms},apad=pad_dur={AUDIO_PAD_S}[v];"
              f"[1:a]volume={amp}[n];[v][n]amix=inputs=2:duration=first:normalize=0,"
              f"alimiter=limit=0.95[o]")
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        ff("-i", raw, "-f", "lavfi", "-i", f"anoisesrc=color={colour}:amplitude=1:sample_rate=16000",
           "-filter_complex", fc, "-map", "[o]", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out_wav)
    return text, rate


# ----------------------------------------------------------------------------------------------
# VIDEO
# ----------------------------------------------------------------------------------------------
STYLE = ("Realistic security CCTV footage from a fixed, high-mounted surveillance camera; slight wide-angle "
         "distortion, muted desaturated colours, mild compression, overcast daytime around 10 AM. "
         "Indian city street in Bengaluru with a small glass-fronted ATM kiosk: one glass pull door, "
         "a single ATM machine inside, bank signage without readable text. "
         "Static camera: no pans, no zoom, no cuts. No music, no on-screen text, no timestamps, "
         "no captions, no logos.")
NEGATIVE = ("on-screen text, timestamp, subtitles, captions, watermark text, camera movement, cuts, music, "
            "cinematic lighting, close-up, slow motion, cartoon")

VIEW = {
    "camera_1": "The camera looks down at the kiosk entrance from a pole on the same footpath, door in the centre of frame.",
    "camera_2": "The camera is across the street, looking at the kiosk front; light traffic passes in the foreground.",
    "camera_3": "Wide-angle camera covering the whole street frontage, kiosk on the right third of the frame, footpath and road visible.",
    "cctv_entrance": "The camera is mounted above the kiosk door looking down at the entrance and the steps outside.",
    "cctv_atm": "Interior camera mounted in the ceiling corner of the kiosk, looking down at the ATM machine, keypad and card slot.",
}

ACTION = {
    "approach_atm": "{A} walks along the footpath toward the ATM kiosk and stops at its glass door, glancing around.",
    "loiter_near_atm": "{A} lingers outside the kiosk for a long time, pacing slowly and repeatedly looking at the door and up and down the street.",
    "enter_atm": "{A} pulls open the glass door and steps inside the kiosk; the door swings shut behind {him}.",
    "tamper_atm": "{A} stands inside the kiosk, leans close to the card slot and presses a thin device onto it, checking over {his} shoulder.",
    "withdraw_cash": "{A} stands inside the kiosk at the ATM, inserts a card, types on the keypad and takes a stack of cash from the dispenser.",
    "wait_outside": "{A} stands just outside the kiosk entrance, phone in hand, watching the street like a lookout.",
    "exit_atm": "{A} pushes the glass door open and leaves the kiosk quickly, walking out of frame.",
    "flee_scene": "{A} hurries away from the kiosk, almost jogging, and leaves the frame.",
    "observe_exit": "A bystander, {L}, is walking on the footpath, stops and turns to watch {S} hurry away from the kiosk.",
    "report_incident": "A bystander, {L}, stands near the kiosk, talks urgently on a mobile phone and points toward the ATM.",
}
JOINT_RE = re.compile(r"^(two|both|the two)\b", re.I)

# footage-quality sentences / tags -> (prompt addition, ffmpeg post filter)
QUALITY = [
    ("face not visible", "The person keeps the head down and the back to the camera the whole time; the face is never visible.", None),
    ("lighting conditions", "The scene is very dim and underexposed, figures are dark silhouettes.", "eq=brightness=-0.22:contrast=0.8"),
    ("camera angle unclear", "Awkward steep camera angle; the person is at the edge of frame and partly cut off.", None),
    ("feed interrupted", "", "FEED_CUT"),
    ("quality insufficient", "", "LOWRES"),
    ("camera obstructed", "A parked delivery van blocks most of the view of the kiosk.", None),
    ("frame rate drop", "", "FRAMEDROP"),
    ("poor image quality", "", "LOWRES"),
    ("partially obscured", "A pillar and a parked motorbike partly block the view of the person.", None),
    ("wide-angle view", "Very wide fisheye view; people appear small.", None),
    ("observed from distance", "The camera is far away; people appear small in the frame.", None),
]
POST = {
    "LOWRES": "scale=iw/4:-2,scale=iw*4:-2:flags=neighbor,noise=alls=18:allf=t,eq=saturation=0.6",
    "FRAMEDROP": "fps=3,fps=24",
    "FEED_CUT": "drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='between(t,3.2,5.6)'",
}


def is_denial(content: str) -> bool:
    return bool(re.match(r"^(I|I'm|I've|I'd|My)\b", content.strip()))


def video_prompt(case, obs, chars, a2e):
    events = {e["event_id"]: e for e in case["ground_truth"]["events"]}
    ev = events.get(obs.get("event_ref"))
    action = ev["action"] if ev else "approach_atm"
    actor_id = ev["entity_id"] if ev else a2e.get(obs["entity"])
    actor = chars.get(actor_id) or next(iter(chars.values()))
    suspects = [eid for eid, c in chars.items() if c["role"] == "suspect"]
    content = obs["content"]

    def desc(eid):
        return chars[eid]["look"]

    L = actor["look"]
    A = L[0].upper() + L[1:] + ","
    template = ACTION.get(action, "{A} is seen near the ATM kiosk.")
    if action == "observe_exit":
        s = ("two people (" + "; and ".join(desc(e) for e in suspects) + ")") if len(suspects) > 1 else (
            f"another person ({desc(suspects[0])})" if suspects else "two people")
        line = template.format(L=L, S=s)
    elif action == "flee_scene" and (JOINT_RE.match(content) or len(suspects) > 1):
        line = ("Two people hurry away from the kiosk side by side, almost jogging, and leave the frame. "
                + " ".join(f"Person {i + 1}: {desc(e)}." for i, e in enumerate(suspects)))
    else:
        line = template.format(A=A, L=L, him="him" if actor["he"] == "he" else "her", his=actor["his"])
    if action == "report_incident" and re.search(r"writ|notebook|paper", content, re.I):
        line = f"A bystander, {L}, stands near the kiosk writing in a small notebook while looking toward the ATM."
    if action == "wait_outside" and len(suspects) > 1:
        other = next((e for e in suspects if e != actor_id), None)
        if other:
            line += f" Through the glass, another person ({desc(other)}) is visible inside at the ATM."

    extra, post = [], []
    low = content.lower()
    for key, add, f in QUALITY:
        if key in low:
            if add:
                extra.append(add)
            if f:
                post.append(f)
    if not is_denial(content) and not extra and not post:
        extra.append(f"What the footage must clearly show: {spoken_text(content)}")

    prompt = " ".join([STYLE, VIEW.get(obs["source"], VIEW["camera_1"]), line, *extra,
                       "The action starts about two seconds into the 8-second clip."])
    return {"prompt": prompt, "negative_prompt": NEGATIVE, "action": action, "actor": actor_id,
            "post": post, "denial_rendered_as_action": is_denial(content)}


def _api_key():
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    for f in [ROOT / ".env", VIDEO_AGENT_DIR / ".env", VIDEO_AGENT_DIR / ".env.example"]:
        if f.exists():
            m = re.search(r"^GEMINI_API_KEY\s*=\s*(\S+)", f.read_text(encoding="utf-8"), re.M)
            if m:
                return m.group(1)
    raise SystemExit("GEMINI_API_KEY not set (env var or .env)")


def veo_generate(spec, out_mp4: Path, model: str):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=_api_key())
    full = dict(aspect_ratio="16:9", number_of_videos=1, duration_seconds=8,
                negative_prompt=spec["negative_prompt"], person_generation="allow_all")
    try:
        op = client.models.generate_videos(model=model, prompt=spec["prompt"],
                                           config=types.GenerateVideosConfig(**full))
    except Exception as exc:                       # some models reject optional fields
        print(f"    [veo] retrying with minimal config ({str(exc)[:120]})")
        op = client.models.generate_videos(model=model, prompt=spec["prompt"],
                                           config=types.GenerateVideosConfig(aspect_ratio="16:9"))
    while not op.done:
        time.sleep(10)
        op = client.operations.get(op)
    if op.error or not op.response or not op.response.generated_videos:
        raise RuntimeError(f"Veo returned no video: {op.error}")
    vid = op.response.generated_videos[0]
    client.files.download(file=vid.video)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    vid.video.save(str(out_mp4))


def finish_video(raw: Path, out: Path, obs, post):
    """CCTV degradation + burned-in camera label and ticking clock (video agent's own overlay)."""
    from video_agent.add_overlay import burn
    start = (datetime.fromisoformat(obs["timestamp"]) - timedelta(seconds=VIDEO_LEAD_S)).isoformat()
    src = raw
    with tempfile.TemporaryDirectory() as tmp:
        if post:
            src = Path(tmp) / "degraded.mp4"
            ff("-i", raw, "-vf", ",".join(POST[p] for p in post), "-an", "-c:v", "libx264", "-crf", "22",
               "-pix_fmt", "yuv420p", src)
        burn(src, out, obs["source"], start)
    return start


# ----------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------
def render_case(path: Path, args, voices):
    case = json.loads(path.read_text(encoding="utf-8"))
    cid = case["case_id"]
    out = Path(args.out) / cid
    chars = build_characters(case)
    a2e = alias_to_entity(case)
    voice_of = assign_voices(case, chars, voices) if voices else {}
    manifest = {"case_id": cid, "source_case": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                "characters": chars, "voices": voice_of, "observations": []}
    audio_rows, prompts = [], {}
    print(f"\n=== {cid}  ({len(case['observations'])} observations)")

    for obs in sorted(case["observations"], key=lambda o: o["time_offset"]):
        oid, mod = obs["obs_id"], obs["modality"]
        rec = {"obs_id": oid, "modality": mod, "entity": obs["entity"], "entity_id": a2e.get(obs["entity"]),
               "source": obs["source"], "timestamp": obs["timestamp"]}

        if mod == "audio" and args.audio:
            eid = a2e.get(obs["entity"])
            voice = voice_of.get(eid) or {"voice": "en-IN-PrabhatNeural", "pitch": "+0Hz"}
            wav = out / "audio" / f"{oid}.wav"
            if args.force or not wav.exists():
                text, rate = render_audio_obs(obs, voice, wav)
                print(f"  [audio] {oid} {obs['entity']:<11} {voice['voice']:<26} {obs['source']:<17} \"{text}\"")
            else:
                text, rate = spoken_text(obs["content"]), None
                print(f"  [audio] {oid} exists, skipped")
            start = obs["timestamp"]   # Whisper puts the first segment at ~0 s despite the pad
            audio_rows.append({"path": f"{oid}.wav", "start_time": start, "source": obs["source"],
                               "location": obs["location"]})
            rec.update(file=str(wav.relative_to(out)), spoken=text, voice=voice)

        elif mod == "video" and args.video != "none":
            spec = video_prompt(case, obs, chars, a2e)
            prompts[oid] = spec
            rec.update(prompt=spec["prompt"], post=spec["post"], action=spec["action"])
            raw = (Path(args.raw_videos) if args.raw_videos else out / "video" / "raw") / f"{oid}.mp4"
            final = out / "video" / f"{oid}.mp4"
            if args.video == "veo" and (args.force or not raw.exists()):
                print(f"  [video] {oid} generating with {args.veo_model} ...")
                try:
                    veo_generate(spec, raw, args.veo_model)
                except Exception as exc:
                    print(f"  [video] {oid} FAILED: {exc}")
                    rec["error"] = str(exc)
            if raw.exists() and args.video in ("veo", "raw"):
                if args.force or not final.exists() or final.stat().st_mtime < raw.stat().st_mtime:
                    rec["clock_start"] = finish_video(raw, final, obs, spec["post"])
                rec["file"] = str(final.relative_to(out))
                print(f"  [video] {oid} -> {final.relative_to(ROOT) if final.is_relative_to(ROOT) else final}")
            elif args.video in ("veo", "raw"):
                print(f"  [video] {oid} no raw clip at {raw}")
            else:
                print(f"  [video] {oid} prompt written ({spec['action']}"
                      f"{', degrade: ' + '+'.join(spec['post']) if spec['post'] else ''})")

        else:
            rec["note"] = "text observation: no media" if mod == "text" else "skipped"
        manifest["observations"].append(rec)

    out.mkdir(parents=True, exist_ok=True)
    if audio_rows:
        (out / "audio" / "clips.json").write_text(json.dumps(audio_rows, indent=2), encoding="utf-8")
    if prompts:
        (out / "video").mkdir(parents=True, exist_ok=True)
        (out / "video" / "prompts.json").write_text(json.dumps(prompts, indent=2), encoding="utf-8")
    (out / "media_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", action="append",
                    help="case ID (CASE_ATM_003) or path; repeatable. Default: every cases/CASE_ATM_*.json")
    ap.add_argument("--cases-dir", default=str(ROOT / "cases"))
    ap.add_argument("--out", default=str(ROOT / "media"))
    ap.add_argument("--no-audio", dest="audio", action="store_false")
    ap.add_argument("--video", choices=["prompts", "veo", "raw", "none"], default="prompts",
                    help="prompts: write Veo prompts only (free). veo: generate with the Veo API (paid). "
                         "raw: finish clips you already made (named O1.mp4 ...) in --raw-videos")
    ap.add_argument("--raw-videos", help="folder of raw clips for --video raw (default media/<case>/video/raw)")
    ap.add_argument("--veo-model", default=DEFAULT_VEO_MODEL)
    ap.add_argument("--list-veo-models", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-render files that already exist")
    args = ap.parse_args()

    if args.list_veo_models:
        from google import genai
        for m in genai.Client(api_key=_api_key()).models.list():
            if "veo" in m.name.lower():
                print(m.name)
        return
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found on PATH")

    if args.case:
        paths = []
        for c in args.case:
            p = Path(c)
            paths.append(p if p.suffix == ".json" else Path(args.cases_dir) / f"{c}.json")
    else:
        paths = [Path(p) for p in sorted(glob.glob(str(Path(args.cases_dir) / "CASE_ATM_*.json")))
                 if not p.endswith("_obs_only.json")]
    if args.raw_videos and len(paths) > 1:
        raise SystemExit("--raw-videos is per case; pass a single --case")

    voices = asyncio.run(_list_voices()) if args.audio else []
    for p in paths:
        render_case(p.resolve(), args, voices)
    print(f"\nDone. Output in {args.out}")


if __name__ == "__main__":
    main()
