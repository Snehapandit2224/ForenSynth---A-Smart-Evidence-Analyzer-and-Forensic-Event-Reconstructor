"""
Run the video agent on ONE clip. Change VIDEO below, then:

    python run_clip.py

The clip must sit in a folder named after its case and be named after its observation:

    clips/CASE_ATM_010/O17.mp4   ->  case CASE_ATM_010, observation O17

Results always go to runs/<case id>/ (one folder per case; each clip adds its own entry):
    runs/CASE_ATM_010/CASE_ATM_010_video_obs.json   the observations (obs_id O17 ...)
    runs/CASE_ATM_010/CASE_ATM_010_video_log.json   per clip ("clip": "O17.mp4"): what the agent did

If overrides_<case id>.json exists next to this file it is applied (see README).

The free Gemini tier allows about 20 requests a day per model (roughly two clips). When a model's
daily quota is used up, the next one in MODELS is tried automatically; the log file records which
model produced each clip's answer. Extra options are passed on, and an explicit --model switches
the fallback off, e.g.  python run_clip.py --model gemini-3.6-flash
"""
import re
import subprocess
import sys
from pathlib import Path

VIDEO = r"clips/CASE_ATM_010/O17.mp4"      # <-- change this to the clip you want to analyse

MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash"]   # tried in this order

ROOT = Path(__file__).resolve().parent


def main():
    video = Path(VIDEO) if Path(VIDEO).is_absolute() else ROOT / VIDEO
    if not video.is_file():
        raise SystemExit(f"video not found: {video}")
    case_id = video.parent.name.upper()
    if not re.match(r"^CASE_[A-Z]+_\d+$", case_id):
        raise SystemExit(f"put the clip in a folder named after its case (e.g. clips/CASE_ATM_010/), not '{video.parent.name}'")
    if not re.match(r"^O\d+$", video.stem, re.I):
        raise SystemExit(f"name the clip after its observation (e.g. O17.mp4), not '{video.name}'")

    cmd = [sys.executable, "-m", "video_agent.run", "--case-id", case_id, "--video", str(video)]
    overrides = ROOT / f"overrides_{case_id}.json"
    if overrides.exists():
        cmd += ["--overrides", str(overrides)]
    print(f"[run_clip] case {case_id}, clip {video.name} -> runs/{case_id}/", flush=True)
    if "--model" in sys.argv[1:]:
        raise SystemExit(subprocess.run(cmd + sys.argv[1:], cwd=ROOT).returncode)
    for model in MODELS:
        print(f"[run_clip] model {model}", flush=True)
        run = subprocess.run(cmd + ["--model", model] + sys.argv[1:], cwd=ROOT, stderr=subprocess.PIPE,
                             text=True, encoding="utf-8", errors="replace")
        if run.returncode == 0 or "RESOURCE_EXHAUSTED" not in run.stderr:
            sys.stderr.write(run.stderr)
            raise SystemExit(run.returncode)
        print(f"[run_clip] {model}: daily quota used up, trying the next model", flush=True)
    raise SystemExit("every model in MODELS is out of quota for today; try again tomorrow")


if __name__ == "__main__":
    main()
