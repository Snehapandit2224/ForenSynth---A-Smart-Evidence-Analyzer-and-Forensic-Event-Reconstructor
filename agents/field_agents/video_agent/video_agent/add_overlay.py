"""
Burn a CCTV-style overlay (camera label + ticking clock) onto clips that don't have one.
This is data preparation for your synthetic clips, run BEFORE the agent; the agent itself still
only takes the video and the case ID.

One clip:
  python -m video_agent.add_overlay --video raw\\O1.mp4 --out clips\\CASE_ATM_001\\O1.mp4 ^
         --camera camera_1 --start 2024-01-15T10:00:45

A whole case, using the timestamp and source of each observation in the case file
(clips must be named O1.mp4, O4.mp4 ...):
  python -m video_agent.add_overlay --case cases\\cases_atm\\CASE_ATM_001_obs_only.json ^
         --videos raw\\CASE_ATM_001 --out clips\\CASE_ATM_001
"""
import argparse
import calendar
import json
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

SOURCE_TO_LABEL = {"camera_1": "CAM01", "camera_2": "CAM02", "camera_3": "CAM03",
                   "cctv_entrance": "ENT", "cctv_atm": "ATM-INT"}

FONTS = ["C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/arial.ttf",
         "/System/Library/Fonts/Supplementary/Arial.ttf", "/Library/Fonts/Arial.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def _esc(path):
    """Path inside an ffmpeg filter: forward slashes, escaped drive colon (C\\:/...)."""
    return str(path).replace("\\", "/").replace(":", "\\:")


def burn(src, dst, camera, start, corner="top-left", fontsize=28):
    label = SOURCE_TO_LABEL.get(camera, camera)
    epoch = calendar.timegm(datetime.fromisoformat(start).timetuple())
    pos = {"top-left": "x=20:y=20", "top-right": "x=w-tw-20:y=20",
           "bottom-left": "x=20:y=h-th-20", "bottom-right": "x=w-tw-20:y=h-th-20"}[corner]
    font = next((f for f in FONTS if Path(f).exists()), None)
    with tempfile.TemporaryDirectory() as tmp:
        txt = Path(tmp) / "overlay.txt"
        txt.write_text(f"{label}  %{{pts:gmtime:{epoch}:%Y-%m-%d %H\\:%M\\:%S}}", encoding="utf-8")
        vf = (f"drawtext=textfile='{_esc(txt)}'" + (f":fontfile='{_esc(font)}'" if font else "")
              + f":{pos}:fontsize={fontsize}:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=6")
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf", vf,
                        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-an", str(dst)], check=True)
    print(f"[overlay] {Path(src).name} -> {dst}   {label} from {start}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="one clip")
    ap.add_argument("--camera", help="camera_1 | camera_2 | camera_3 | cctv_entrance | cctv_atm (or a label)")
    ap.add_argument("--start", help="clock time at the first frame, e.g. 2024-01-15T10:00:45")
    ap.add_argument("--case", help="*_obs_only.json: take start time and camera from each observation")
    ap.add_argument("--videos", help="folder of clips named after observations (with --case)")
    ap.add_argument("--out", required=True, help="output file (one clip) or folder (with --case)")
    ap.add_argument("--corner", default="top-left", choices=["top-left", "top-right", "bottom-left", "bottom-right"],
                    help="keep it away from any watermark (Veo's is bottom-right)")
    a = ap.parse_args()

    if a.case:
        if not a.videos:
            raise SystemExit("--case needs --videos")
        obs = {o["obs_id"]: o for o in json.loads(Path(a.case).read_text(encoding="utf-8"))["observations"]}
        clips = sorted(p for p in Path(a.videos).iterdir() if p.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv"))
        for clip in clips:
            m = re.match(r"^(O\d+)", clip.stem, re.I)
            o = obs.get(m.group(1).upper()) if m else None
            if not o or o["modality"] != "video":
                print(f"[overlay] {clip.name}: no matching video observation, skipped")
                continue
            burn(clip, Path(a.out) / f"{clip.stem}.mp4", o["source"], o["timestamp"], a.corner)
    elif a.video and a.camera and a.start:
        burn(a.video, a.out, a.camera, a.start, a.corner)
    else:
        raise SystemExit("give --video --camera --start, or --case --videos")


if __name__ == "__main__":
    main()
