"""
Offline end-to-end test of the agentic extractor (no API key needed).

Builds five fake CCTV clips for CASE_ATM_001 with ffmpeg (ticking clock overlay; O6 has a frozen,
unreliable clock; O17 is two people leaving together, one group observation), then replays scripted agent conversations with the mock backend. The TOOLS run
for real: crops + OCR of the overlay, clock consistency check, frame grabs, re-watch, known-people
stills, and the validation gate that sends a bad final answer back.

  python -m tests.test_offline
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from video_agent.add_overlay import FONTS, _esc

ROOT = Path(__file__).resolve().parent.parent
CASE = ROOT / "tests" / "data" / "CASE_ATM_001_obs_only.json"
BASE_EPOCH = 1705312800   # 2024-01-15 10:00:00 as the overlay's "UTC" clock

CLIPS = {"O1.mp4": ("CAM01", 45, False, False), "O17.mp4": ("CAM01", 188, False, False),
         "O4.mp4": ("CAM03", 299, False, False),
         "O5.mp4": ("ATM-INT", 321, False, False), "O6.mp4": ("CAM01", 352, True, True)}


def make_clip(path, label, offset, degrade, frozen):
    photo = next(Path("/usr").rglob("china.jpg"), None)
    src = ["-loop", "1", "-i", str(photo)] if photo else ["-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=24"]
    txt = path.with_suffix(".txt")
    if frozen:   # clock stuck on one value: looks plausible, but does not tick
        txt.write_text(f"{label}  2024-01-15 10:05:52")
    else:
        txt.write_text(f"{label}  %{{pts:gmtime:{BASE_EPOCH + offset}:%Y-%m-%d %H\\:%M\\:%S}}")
    font = next((f for f in FONTS if Path(f).exists()), None)    # Windows ffmpeg has no default font
    vf = (f"scale=1280:720,setsar=1,drawtext=textfile='{_esc(txt)}'" + (f":fontfile='{_esc(font)}'" if font else "")
          + ":x=20:y=20:fontsize=30:fontcolor=white:"
          f"box=1:boxcolor=black@0.6:boxborderw=6")
    if degrade:
        vf += ",scale=-2:360,noise=alls=13:allf=t,eq=brightness='0.05*sin(2*PI*t*0.7)':eval=frame,scale=1280:720"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *src, "-t", "8", "-r", "24", "-vf", vf,
                    "-c:v", "libx264", "-crf", "37" if degrade else "22", "-pix_fmt", "yuv420p", str(path)], check=True)
    txt.unlink()


def clock(offset):
    s = 45 * 0 + offset
    return f"2024-01-15 10:{s // 60:02d}:{s % 60:02d}"


def tool(name, args, thought=""):
    return {"thought": thought or f"call {name}", "tool": name, "args": args}


def clock_steps(offset, corner="top-left"):
    return [tool("crop_overlay", {"corner": corner, "times": [0, 3, 6]}),
            tool("check_clock", {"readings": [{"t": t, "clock": clock(offset + t)} for t in (0, 3, 6)]})]


def final(answer):
    return {"thought": "done", "final": answer}


def ev(person, action, idx, start, cert, vis, match=None, identifiable=True, ref=1.0, bbox=None):
    return {"person": person, "identifiable": identifiable, "registry_match": match, "action": action,
            "template_index": idx, "free_content": None, "start": start, "overlay_timestamp_at_start": None,
            "certainty": cert, "visibility": vis, "ref_time": ref, "bbox": bbox}


def answer(label, view, events, quality=0.85, issue=None, verified=True):
    return {"overlay_camera_label": label, "overlay_timestamp_first": None, "camera_view": view,
            "footage_issue": issue, "image_quality": quality, "clock_verified": verified,
            "clock_note": "read with crop_overlay + check_clock", "events": events}


PLURAL_EXIT = "Two individuals are recorded leaving the scene side by side."
COMPANION = "woman, long olive raincoat, white sneakers, yellow tote bag"
MAN = "man, charcoal-grey hoodie, black track pants with white stripe, faded red cap, light-blue mask, black backpack on right shoulder"


def scripts(man):
    o1 = answer("CAM01", "exterior_above_entrance", [ev(MAN, "approach_atm", 1, "00:00", 0.9, 0.9, bbox=[0.1, 0.5, 0.1, 0.3])])
    o4 = answer("CAM03", "wide_street_frontage",
                [ev("man in grey hoodie, red cap, backpack, small in frame", "tamper_atm", 0, "00:01", 0.75, 0.6, match=man)], 0.8)
    o5 = answer("ATM-INT", "interior_card_reader",
                [ev("man in grey hoodie, red cap, blue mask, backpack on right shoulder", "exit_atm", 1, "00:00", 0.95, 0.95, match=man)], 0.9)
    o6_events = [{**ev("older woman, mustard salwar kameez, maroon dupatta, green bag", "observe_exit", 4, "00:01", 0.7, 0.6),
                  "stays_behind": True},     # a witness of someone leaving must be declared as staying behind
                 ev("person in dark clothing seen from behind", "exit_atm", 4, "00:02", 0.5, 0.2, identifiable=False)]
    o6_bad = answer("CAM01", "exterior_above_entrance", o6_events, 0.3, "face_not_visible", verified=None)
    o6 = answer("CAM01", "exterior_above_entrance", o6_events, 0.3, "face_not_visible", verified=False)
    # two people leave side by side: ONE group event, lead = the man already known from O1
    pair = {"group": True, "people": [{"person": MAN, "identifiable": True, "registry_match": man},
                                      {"person": COMPANION, "identifiable": True, "registry_match": None,
                                       "ref_time": 2.0, "bbox": [0.55, 0.3, 0.15, 0.6]}]}
    lead = ev(MAN, "flee_scene", 3, "00:00", 0.9, 0.85, match=man, bbox=[0.4, 0.3, 0.15, 0.6])
    o17 = answer("CAM01", "exterior_above_entrance", [{**lead, **pair}])
    o17_bad = answer("CAM01", "exterior_above_entrance", [{**lead, **pair, "action": "exit_atm", "template_index": 0}])
    # the mislabel this fixes: one exits, the companion reported as a witness without "stays_behind"
    o17_split = answer("CAM01", "exterior_above_entrance",
                       [ev(MAN, "exit_atm", 0, "00:00", 0.9, 0.85, match=man), ev(COMPANION, "observe_exit", 4, "00:01", 0.9, 0.9)])
    return {
        "O1.mp4": clock_steps(45) + [tool("validate_draft", {"draft": o1}), final(o1)],
        # singular exit sentence for a group -> gate sends it back; then the plural one
        "O17.mp4": clock_steps(188) + [tool("known_people", {}), tool("validate_draft", {"draft": o17_split}),
                                       final(o17_bad), final(o17)],
        "O4.mp4": clock_steps(299) + [tool("known_people", {}),
                                      tool("get_frames", {"times": [1, 3], "crop": [0.3, 0.3, 0.4, 0.5]}), final(o4)],
        "O5.mp4": clock_steps(321) + [tool("known_people", {}), tool("rewatch", {"start": 0, "end": 3, "fps": 10}), final(o5)],
        # frozen clock -> INCONSISTENT; first final forgets clock_verified -> gate sends it back
        "O6.mp4": [tool("crop_overlay", {"corner": "top-left", "times": [0, 4]}),
                   tool("check_clock", {"readings": [{"t": 0, "clock": "2024-01-15 10:05:52"},
                                                     {"t": 4, "clock": "2024-01-15 10:05:52"}]}),
                   final(o6_bad), final(o6)],
    }


def main():
    tmp = Path(tempfile.mkdtemp(prefix="video_agent_test_"))
    clips, runs, runs_folder = tmp / "clips", tmp / "runs", tmp / "runs_folder"
    clips.mkdir()
    for name, spec in CLIPS.items():
        make_clip(clips / name, *spec)
    (tmp / "overrides.json").write_text(json.dumps({"O6.mp4": {"start": "2024-01-15T10:05:52"}}))

    def run(video, out):
        subprocess.run([sys.executable, "-m", "video_agent.run", "--case-id", "CASE_ATM_001", "--video", str(video),
                        "--out", str(out), "--backend", "mock", "--overrides", str(tmp / "overrides.json")],
                       cwd=ROOT, check=True)

    def write_mocks(man):
        for name, script in scripts(man).items():
            (clips / name).with_suffix(".mock.json").write_text(json.dumps({"script": script}))

    # the registry label for the man is deterministic per case; learn it from a first O1 run
    write_mocks("Person_00")
    run(clips / "O1.mp4", runs)
    man = next(iter(json.loads((runs / "CASE_ATM_001" / "people_registry.json").read_text())))
    write_mocks(man)

    # one clip at a time, in the folder run's order; O4 twice (replace, not duplicate)
    for n in ("O17.mp4", "O4.mp4", "O5.mp4", "O6.mp4", "O4.mp4"):
        run(clips / n, runs)

    case_dir = runs / "CASE_ATM_001"
    out = json.loads((case_dir / "CASE_ATM_001_video_obs.json").read_text())
    log = json.loads((case_dir / "CASE_ATM_001_video_log.json").read_text())
    clips_log = {c["clip"]: c for c in log["clips"]}
    obs = {o["obs_id"]: o for o in out["observations"]}
    ref_keys = list(json.loads(CASE.read_text())["observations"][0].keys())

    assert sorted(obs) == ["O1", "O17", "O4", "O5", "O6", "O6b"], sorted(obs)
    for o in out["observations"]:
        assert list(o.keys()) == ref_keys and o["modality"] == "video"
    # tools really ran
    ocr_line = clips_log["O1.mp4"]["agent_trace"][0]["result"]
    print("\nOCR on O1 overlay:", ocr_line.splitlines()[1])
    if "OCR:" in ocr_line:
        assert "2024-01-15 10:00:45" in ocr_line, ocr_line
    assert clips_log["O1.mp4"]["verified_clock_at_first_frame"] == "2024-01-15T10:00:45"
    assert clips_log["O6.mp4"]["verified_clock_at_first_frame"] is None
    assert "INCONSISTENT" in clips_log["O6.mp4"]["agent_trace"][1]["result"]
    gate = [t for t in clips_log["O6.mp4"]["agent_trace"] if t.get("final")]
    assert gate[0]["issues"] and not gate[-1]["issues"], gate          # rejected once, then accepted
    assert "Known people" in clips_log["O4.mp4"]["agent_trace"][2]["result"]
    assert (case_dir / "people" / f"{man}.jpg").exists()
    # fields
    assert obs["O1"]["entity"] == obs["O4"]["entity"] == obs["O5"]["entity"] == man
    assert obs["O1"]["timestamp"] == "2024-01-15T10:00:45" and obs["O1"]["time_offset"] == 45
    assert obs["O4"]["timestamp"] == "2024-01-15T10:05:00" and obs["O4"]["source"] == "camera_3"
    assert obs["O5"]["location"] == "ATM booth interior, card reader and keypad area"
    assert obs["O6"]["role"] == "witness" and obs["O6b"]["role"] == "suspect"
    assert obs["O6b"]["entity"] not in (man, obs["O6"]["entity"])
    assert obs["O6b"]["content"] == "Subject's face not visible in available footage."
    assert obs["O6"]["timestamp"] == "2024-01-15T10:05:53"                 # from the override, not the frozen clock
    assert obs["O6"]["confidence"] < obs["O1"]["confidence"]
    # group event: two people leaving together -> exactly one observation, plural sentence, lead's entity
    assert clips_log["O17.mp4"]["obs_ids"] == ["O17"], clips_log["O17.mp4"]["obs_ids"]
    assert obs["O17"]["content"] == PLURAL_EXIT and obs["O17"]["role"] == "suspect"
    assert obs["O17"]["entity"] == man and obs["O17"]["source"] == "camera_1"
    assert obs["O17"]["timestamp"] == "2024-01-15T10:03:08" and obs["O17"]["time_offset"] == 188
    registry = json.loads((case_dir / "people_registry.json").read_text())
    in_o17 = [label for label, p in registry.items() if "O17" in p["seen_in"]]
    assert len(in_o17) == 2 and man in in_o17, in_o17                      # both people are in the registry
    companion = next(label for label in in_o17 if label != man)
    assert registry[companion]["appearance"] == COMPANION and (case_dir / "people" / f"{companion}.jpg").exists()
    assert clips_log["O17.mp4"]["decisions"][0]["co_entities"] == [companion]
    assert "stays_behind" in clips_log["O17.mp4"]["agent_trace"][3]["result"]   # exit + watcher draft is challenged
    gate = [t for t in clips_log["O17.mp4"]["agent_trace"] if t.get("final")]
    assert any("plural sentence" in i for i in gate[0]["issues"]) and not gate[-1]["issues"], gate

    run(clips, runs_folder)                                               # whole folder gives the same result
    folder = json.loads((runs_folder / "CASE_ATM_001" / "CASE_ATM_001_video_obs.json").read_text())
    assert folder["observations"] == out["observations"], "folder run differs from clip-by-clip run"
    print("\nextracted observations:\n" + json.dumps(out, indent=2))

    subprocess.run([sys.executable, "-m", "video_agent.evaluate", "--cases", str(CASE.parent), "--runs", str(runs)],
                   cwd=ROOT, check=True)
    print(f"\nALL CHECKS PASSED  (artifacts in {tmp})")


if __name__ == "__main__":
    main()
