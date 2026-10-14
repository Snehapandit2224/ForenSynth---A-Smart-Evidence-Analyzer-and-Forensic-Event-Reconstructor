# ForenSynth video agent (agentic)

Turns CCTV clips into observation records with the same fields as the case files:

```
obs_id, entity, role, modality, source, location, content, timestamp, time_offset, confidence
```

**Inputs: a video and a case ID. Nothing else.** No case file, FIR or ground truth is read.
Modality is always `video`.

## How it works

Gemini (a multimodal LLM) runs as an **agent**: it watches the clip, then calls local tools in a
loop until it is confident, and only then gives its answer.

| Tool | What it does |
|---|---|
| `crop_overlay` | Enlarged crops of the corner holding the clock and camera label, at chosen times, plus OCR text (Tesseract, if installed) |
| `check_clock` | Checks that clock readings at different times tick in step with the video. Only a consistent result counts as a verified clock, and timestamps are then computed from it by code, not from the model's own reading |
| `get_frames` | Full or cropped frames at chosen times (hands at the card slot, a face, the door) |
| `rewatch` | A short segment re-sent at a higher frame rate |
| `known_people` | People already seen in this case, with a reference still of each, for identity matching |
| `validate_draft` | Runs the field rules on a draft answer |

The final answer then has to pass the same field rules (a **validation gate**). If it fails, the
problems go back to the agent to fix, up to twice. Whatever still fails is flagged
`unresolved_checks` in the output log. Finally the agent's own code turns the answer into the
fields, using rules copied from the case generator. Typical clips take 3-6 tool calls.

Every step is recorded in `<case>_video_log.json` (`agent_trace`), so you can see what the agent
looked at and why.

## How each field is produced

| Field | How |
|---|---|
| `obs_id` | From the clip name (`O4.mp4` -> `O4`) or `--obs-id`. A second person/action in the same clip becomes `O4b`, `O4c`. Otherwise `V1`, `V2` ... People doing the same thing together (two people leaving side by side) are ONE observation, as in the generator: a plural sentence (`flee_scene`: "Two individuals are recorded leaving the scene side by side."), the lead person's `entity`, and the other members' labels under `co_entities` in the log. |
| `entity` | `Person_NN`, the generator's video alias format. A per-case registry keeps the same person on the same label across clips (the model matches clothing and accessories). People who can't be identified (back view, occluded) always get a new label. These labels are the agent's own: they won't equal the generator's aliases, which is what entity resolution is for. |
| `role` | From the action, as in the templates: `observe_exit` / `report_incident` -> `witness`, everything else -> `suspect`. |
| `modality` | Always `video`. |
| `domain` (file header) | From the case ID: `CASE_ATM_001` -> `ATM_Robbery`. |
| `source` | From the burned-in camera label (`CAM01` -> `camera_1`, `CAM03` -> `camera_3`, `ATM-INT` -> `cctv_atm`; edit `OVERLAY_LABEL_TO_SOURCE` in `config.py`). Falls back to the filename, then to the camera view the model recognises. |
| `location` | Looked up from `source` with the generator's own `OBSERVATION_LOCATIONS` table. |
| `content` | The model picks the closest sentence from the generator's ATM phrase banks (`atm_video_vocabulary.json`, extracted from `templates.py`). Using those exact phrases keeps the Timeline agent's action tagging (`ACTION_TAG_FRAGMENTS`) working. If the footage itself hides the action, content is one of the generator's footage-quality sentences instead, e.g. "Subject's face not visible in available footage." |
| `timestamp` | From the clock overlaid on the video. Preferred: the clock verified by `check_clock` + the action's offset into the clip. If the clock can't be verified, the model's own reading is used and flagged `clock_not_verified`. If the clock is frozen or garbled, the timestamp is left null and flagged `timestamp_unreadable`. |
| `time_offset` | Seconds from the generator's fixed base time `2024-01-15T10:00:00`, the same anchor every case uses (change with `--anchor`). |
| `confidence` | `0.45 x model certainty + 0.25 x visibility + 0.30 x measured image quality`, minus penalties for a footage problem, an unverified or unreadable clock, or an unidentifiable person; clamped to 0.10-0.99. Image quality is measured from the pixels (sharpness after removing grain, noise, flicker, darkness), not judged by the model. `evaluate.py --calibrate` can fit it to your case files. |

## Setup

```bash
pip install -r requirements.txt          # also needs ffmpeg + ffprobe on PATH
# optional but recommended: Tesseract OCR for reading the clock
#   Windows: https://github.com/UB-Mannheim/tesseract/wiki   Ubuntu: sudo apt install tesseract-ocr
cp .env.example .env                     # paste a free key from https://aistudio.google.com/apikey
```

The default model is `gemini-3.8-flash` (free tier, native video input). If Google renames it,
pass `--model <new-id>`.

## Run

One clip at a time (each run adds to the case's output file; rerunning a clip replaces its entry):

```bash
python -m video_agent.run --case-id CASE_ATM_001 --video clips/CASE_ATM_001/O1.mp4
python -m video_agent.run --case-id CASE_ATM_001 --video clips/CASE_ATM_001/O4.mp4
```

Or a whole folder of one case's clips at once (same result):

```bash
python -m video_agent.run --case-id CASE_ATM_001 --video clips/CASE_ATM_001
```

Name clips after the observation they show (`O1.mp4`, `O4.mp4` ...) or pass `--obs-id`. Feed a
case's clips in time order where possible: the people registry is built as clips arrive, so a
person is recognised in later clips once they've been seen.

Outputs in `runs/CASE_ATM_001/`:

- `CASE_ATM_001_video_obs.json` - `case_id`, `domain`, `observations` (same shape as `*_obs_only.json`, video only)
- `CASE_ATM_001_video_log.json` - per clip: raw model answer, quality metrics, how each field was decided, review flags
- `people_registry.json` and `people/` - who each `Person_NN` is (appearance, reference still) and where they appear
- `cache/` - model answers. Reruns reuse them, so the same clips give the same output without new API calls (`--no-cache` to force)

All cases, one folder per case:

```bash
for d in clips/CASE_ATM_*; do python -m video_agent.run --case-id "$(basename "$d")" --video "$d"; done
```

Each tool step is one request, so a clip takes about 4-7 requests. The free tier allows about 10
requests a minute; the agent waits and retries when it is rate limited. `--max-steps` caps the tool
calls per clip (default 8). Results are cached, so reruns cost nothing.

### Clips without an overlay

The agent reads time and camera from the clock and label burned into the footage. If your clips
don't have one, it still runs, but `timestamp` / `time_offset` come out null and `source` is
guessed from the camera view. Add the overlay first (data preparation; the agent itself still only
takes the video and case ID):

```bash
# whole case: start time and camera taken from each observation (clips named O1.mp4 ...)
python -m video_agent.add_overlay --case cases/cases_atm/CASE_ATM_001_obs_only.json --videos raw/CASE_ATM_001 --out clips/CASE_ATM_001
# one clip
python -m video_agent.add_overlay --video raw/O1.mp4 --out clips/CASE_ATM_001/O1.mp4 --camera camera_1 --start 2024-01-15T10:00:45
```

It draws `CAM01  2024-01-15 10:00:45` in the top-left (away from Veo's watermark), ticking with
the video.

### When the clock or camera label is garbled

Veo often mangles on-screen text. Such clips are flagged `timestamp_unreadable` or `source_unknown`
and get `timestamp`/`time_offset` = null rather than a guess. If you want to fix them by hand, an
optional overrides file supplies the values (cached answers are reused):

```json
{ "O6.mp4": { "start": "2024-01-15T10:05:52", "source": "camera_1" } }
```

```bash
python -m video_agent.run --case-id CASE_ATM_001 --video clips/CASE_ATM_001 --overrides overrides.json
```

`start` is the clock time at the first frame; the agent still adds the action's offset within the clip.

## Evaluate against the case files (testing only)

The agent never reads case files; this separate script does, to score the agent.

```bash
python -m video_agent.evaluate --cases cases/cases_atm --runs runs
```

Compares every extracted observation with the case's video observations (obs fields only):
source / location / role / action accuracy, entity consistency (a one-to-one label mapping is
inferred, e.g. `Person_94 -> Person_14`), timestamp error, and confidence error. Video observations
with no clip (for example spoken denials tagged as video, like CASE_ATM_001 O3) are listed as
missing; extra people the agent found (like the figure leaving in O6) are listed as extra.

`--calibrate` also fits a linear confidence map and saves `runs/calibration.json`; the next agent
run applies it.

## Offline test

```bash
python -m tests.test_offline
```

Builds five fake CCTV clips for CASE_ATM_001 with ffmpeg (O6 has a frozen clock, O17 is two people
leaving together), replays scripted
agent conversations with a mock model, and checks the fields. The tools run for real: OCR of the
overlay, the clock check (which catches the frozen clock), frame grabs, re-watch, reference
stills, and the validation gate sending a bad answer back. No API key needed.

## Files

```
video_agent/
  run.py          CLI
  agent.py        clip -> observations (field rules, timestamps, confidence, caching)
  loop.py         the agent loop and validation gate
  tools.py        the agent's tools + the field-rule validator
  prompt.py       instructions and tool list sent to the model
  backends.py     Gemini (video, images, text per turn) and a scripted mock
  quality.py      measured image quality
  registry.py     Person_NN labels across clips
  evaluate.py     comparison with *_obs_only.json + calibration
  config.py       location table, camera labels, footage-quality sentences, confidence weights
  atm_video_vocabulary.json   ATM phrase banks from GENERATOR_FIXED/templates.py
```
