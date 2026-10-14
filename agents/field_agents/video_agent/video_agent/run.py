"""
CLI for the agentic video extractor. Inputs: a video (or a folder of a case's videos) and the case ID. Nothing else is read.

  python -m video_agent.run --case-id CASE_ATM_001 --video clips/CASE_ATM_001          # whole case
  python -m video_agent.run --case-id CASE_ATM_001 --video clips/CASE_ATM_001/O4.mp4   # one clip

Writes runs/<case_id>/<case_id>_video_obs.json (observations, video only) and
runs/<case_id>/<case_id>_video_log.json (model answers, quality metrics, decisions, review flags).
One-clip runs add to / replace entries in those files, so you can feed clips one at a time.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from . import config as C
from .agent import FIELDS, VideoAgent, make_context

VIDEO_EXT = (".mp4", ".mov", ".webm", ".mkv")


def base_id(obs_id):
    return obs_id.rstrip("abcdefghijklmnopqrstuvwxyz")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case-id", required=True, help="e.g. CASE_ATM_001 (the domain is taken from it)")
    ap.add_argument("--video", required=True, help="a clip, or a folder with the case's clips (O1.mp4, O4.mp4 ...)")
    ap.add_argument("--obs-id", help="obs_id for a single clip (default: from the file name, e.g. O4.mp4 -> O4)")
    ap.add_argument("--anchor", default=C.DEFAULT_ANCHOR,
                    help="clock time where time_offset = 0 (default: the generator's base time)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--backend", default="gemini", choices=["gemini", "mock"])
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--fps", type=float, default=C.DEFAULT_FPS)
    ap.add_argument("--overrides", help='optional, only for clips whose on-screen clock/label is garbled: '
                                        '{"O6.mp4": {"start": "2024-01-15T10:05:52", "source": "camera_1"}}')
    ap.add_argument("--people", help='optional JSON seeding the registry: {"Person_14": "grey hoodie, red cap ..."}')
    ap.add_argument("--max-steps", type=int, default=8, help="tool calls the agent may make per clip")
    ap.add_argument("--no-cache", action="store_true", help="call the model again even if a cached answer exists")
    a = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    ctx = make_context(a.case_id, a.anchor)
    out_dir = Path(a.out) / ctx["case_id"]
    obs_path = out_dir / f"{ctx['case_id']}_video_obs.json"
    log_path = out_dir / f"{ctx['case_id']}_video_log.json"

    video = Path(a.video)
    if video.is_dir():
        clips = sorted(p for p in video.iterdir() if p.suffix.lower() in VIDEO_EXT)
        if not clips:
            raise SystemExit(f"no video clips in {video}")
        obs_ids = [VideoAgent.obs_id_for(c, i + 1) for i, c in enumerate(clips)]
        single = False
    elif video.is_file():
        clips = [video]
        prior = json.loads(obs_path.read_text())["observations"] if obs_path.exists() else []
        n = len({base_id(o["obs_id"]) for o in prior}) + 1
        obs_ids = [VideoAgent.obs_id_for(video, n, a.obs_id)]
        single = True
    else:
        raise SystemExit(f"{video} not found")

    overrides = json.loads(Path(a.overrides).read_text()) if a.overrides else {}
    agent = VideoAgent(ctx, out_dir, backend=a.backend, model=a.model, fps=a.fps, registry_seed=a.people,
                       overrides=overrides, use_cache=not a.no_cache, keep_registry=single,
                       max_steps=a.max_steps)
    observations, logs = agent.process_clips(clips, obs_ids)

    if single:   # merge with earlier clips of this case, replacing this clip's old entries
        new_ids = {base_id(o["obs_id"]) for o in observations}
        if obs_path.exists():
            observations = [o for o in json.loads(obs_path.read_text())["observations"]
                            if base_id(o["obs_id"]) not in new_ids] + observations
        if log_path.exists():
            logs = [l for l in json.loads(log_path.read_text())["clips"]
                    if not new_ids & {base_id(i) for i in l["obs_ids"]}] + logs
    observations.sort(key=lambda o: (o["timestamp"] is None, o["timestamp"] or "", o["obs_id"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"case_id": ctx["case_id"], "domain": ctx["domain"],
              "observations": [{k: o[k] for k in FIELDS} for o in observations]}
    obs_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log = {"case_id": ctx["case_id"], "updated_utc": datetime.now(timezone.utc).isoformat(),
           "anchor": ctx["anchor"].isoformat(), "model": agent.model_id, "clips": logs}
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    flagged = [d for l in logs for d in l["decisions"] if d["flags"]]
    print(f"\n[video] {len(result['observations'])} observations -> {obs_path}")
    if flagged:
        print(f"[video] {len(flagged)} need review (see {log_path.name}): "
              + ", ".join(f"{d['obs_id']}({'/'.join(d['flags'])})" for d in flagged))


if __name__ == "__main__":
    main()
