"""
Offline end-to-end test of the agentic extractor (no API key needed).

Builds five fake CCTV clips for CASE_ATM_001 with ffmpeg (ticking clock overlay; O6 has a frozen,
unreliable clock; O17 is two people leaving together, one group observation), then replays scripted agent conversations with the mock backend. The TOOLS run
for real: crops + OCR of the overlay, clock consistency check, frame grabs, re-watch, known-people
stills, and the validation gate that sends a bad final answer back.

  python -m tests.test_offline
"""
import json
import shutil
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


def batch(*calls, thought="settle everything in one batch"):
    """Reply 1: every tool call at once (the agent gets two model requests per clip)."""
    return {"thought": thought, "tools": [{"tool": c["tool"], "args": c["args"]} for c in calls]}


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
        "O1.mp4": [batch(*clock_steps(45), tool("validate_draft", {"draft": o1})), final(o1)],
        # exit + watcher draft is challenged by validate_draft; reply 2 uses a singular sentence for the
        # group -> the gate sends it back; reply 3 (repair) is the plural group event
        "O17.mp4": [batch(*clock_steps(188), tool("known_people", {}), tool("validate_draft", {"draft": o17_split})),
                    final(o17_bad), final(o17)],
        "O4.mp4": [batch(*clock_steps(299), tool("known_people", {}),
                         tool("get_frames", {"times": [1, 3], "crop": [0.3, 0.3, 0.4, 0.5]}),
                         tool("rewatch", {"start": 0, "end": 3, "fps": 10})), final(o4)],
        # no tools: reply 1 is a final answer with a sentence number that does not exist -> the gate sends
        # it back, reply 2 fixes it (O5's clock is settled by the OCR pre-check, or the gate catches that too)
        "O5.mp4": [final({**o5, "events": [{**o5["events"][0], "template_index": 99}]}), final(o5)],
        # frozen clock -> INCONSISTENT; the draft that forgets clock_verified is flagged by validate_draft
        "O6.mp4": [batch(tool("crop_overlay", {"corner": "top-left", "times": [0, 4]}),
                         tool("check_clock", {"readings": [{"t": 0, "clock": "2024-01-15 10:05:52"},
                                                           {"t": 4, "clock": "2024-01-15 10:05:52"}]}),
                         tool("validate_draft", {"draft": o6_bad})),
                   final(o6)],
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
    assert "clock not verified" in clips_log["O6.mp4"]["agent_trace"][2]["result"]   # draft flagged
    assert not [t for t in clips_log["O6.mp4"]["agent_trace"] if t.get("final")][-1]["issues"]
    gate = [t for t in clips_log["O5.mp4"]["agent_trace"] if t.get("final")]
    assert len(gate) == 2 and gate[0]["issues"] and not gate[-1]["issues"], gate   # rejected once, then accepted
    # never more than three model requests per clip
    assert all(c["api_calls"] <= 3 for c in log["clips"]), {c["clip"]: c["api_calls"] for c in log["clips"]}
    assert clips_log["O1.mp4"]["api_calls"] == 2 and clips_log["O5.mp4"]["api_calls"] == 2
    assert "Known people" in clips_log["O4.mp4"]["agent_trace"][2]["result"]
    assert (case_dir / "people" / f"{man}.jpg").exists()
    # fields
    assert obs["O1"]["entity"] == obs["O4"]["entity"] == obs["O5"]["entity"] == man
    assert obs["O1"]["timestamp"] == "2024-01-15T10:00:45" and obs["O1"]["time_offset"] == 45
    assert obs["O4"]["timestamp"] == "2024-01-15T10:04:59" and obs["O4"]["source"] == "camera_3"
    assert obs["O5"]["location"] == "ATM booth interior, card reader and keypad area"
    assert obs["O6"]["role"] == "witness" and obs["O6b"]["role"] == "suspect"
    assert obs["O6b"]["entity"] not in (man, obs["O6"]["entity"])
    assert obs["O6b"]["content"] == "Subject's face not visible in available footage."
    assert obs["O6"]["timestamp"] == "2024-01-15T10:05:52"                 # from the override, not the frozen clock
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
    assert clips_log["O17.mp4"]["api_calls"] == 3                          # tools, final, repair

    # a folder run that dies on its last clip (O6: no model answer, like a quota error) keeps the clips
    # finished before it; the rerun takes those from the cache and completes the case
    o6_mock = (clips / "O6.mp4").with_suffix(".mock.json")
    o6_mock.rename(o6_mock.with_suffix(".bak"))
    failed = subprocess.run([sys.executable, "-m", "video_agent.run", "--case-id", "CASE_ATM_001", "--video", str(clips),
                             "--out", str(runs_folder), "--backend", "mock", "--overrides", str(tmp / "overrides.json")],
                            cwd=ROOT, capture_output=True, text=True)
    assert failed.returncode != 0, "the run without O6's answer should fail"
    partial = json.loads((runs_folder / "CASE_ATM_001" / "CASE_ATM_001_video_obs.json").read_text())
    assert sorted(o["obs_id"] for o in partial["observations"]) == ["O1", "O17", "O4", "O5"], partial
    o6_mock.with_suffix(".bak").rename(o6_mock)

    run(clips, runs_folder)                                               # whole folder gives the same result
    folder = json.loads((runs_folder / "CASE_ATM_001" / "CASE_ATM_001_video_obs.json").read_text())
    assert folder["observations"] == out["observations"], "folder run differs from clip-by-clip run"
    print("\nextracted observations:\n" + json.dumps(out, indent=2))

    # OCR clock pre-check (only where Tesseract is installed): verified before the model is asked,
    # and a frozen clock is caught even though it looks plausible
    if shutil.which("tesseract"):
        assert clips_log["O1.mp4"]["ocr_clock"]["status"] == "consistent", clips_log["O1.mp4"]["ocr_clock"]
        assert clips_log["O1.mp4"]["ocr_clock"]["clock_at_first_frame"] == "2024-01-15T10:00:45"
        assert clips_log["O6.mp4"]["ocr_clock"]["status"] == "frozen", clips_log["O6.mp4"]["ocr_clock"]

    # model failures: a reply that is valid JSON but no object must not crash, a clip the model gave no
    # answer for keeps its earlier result and is not cached, and a rerun with nothing in it clears the clip
    one = tmp / "clips_one"
    one.mkdir()
    shutil.copy(clips / "O1.mp4", one / "O1.mp4")
    runs_one = tmp / "runs_one"
    case_one = runs_one / "CASE_ATM_001"

    def run_one(script, *extra):
        (one / "O1.mock.json").write_text(json.dumps({"script": script}))
        r = subprocess.run([sys.executable, "-m", "video_agent.run", "--case-id", "CASE_ATM_001", "--video",
                            str(one / "O1.mp4"), "--out", str(runs_one), "--backend", "mock", *extra],
                           cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        return json.loads((case_one / "CASE_ATM_001_video_obs.json").read_text())["observations"], r.stdout

    good = answer("CAM01", "exterior_above_entrance", [ev(MAN, "approach_atm", 1, "00:00", 0.9, 0.9)])
    first, _ = run_one([final(good)])
    assert [o["obs_id"] for o in first] == ["O1"], first
    cached_before = sorted(p.name for p in (case_one / "cache").glob("*.json") if not p.name.startswith("ocr_"))
    kept, stdout = run_one(["not an object", ["a list"], "still not"], "--no-cache")
    assert kept == first, "a clip the model gave no answer for must keep its earlier result"
    assert "no answer from the model" in stdout, stdout
    cached_after = sorted(p.name for p in (case_one / "cache").glob("*.json") if not p.name.startswith("ocr_"))
    assert cached_after == cached_before, "a failed answer must not be cached"
    nothing = answer("CAM01", "exterior_above_entrance", [])
    cleared, _ = run_one([final(nothing)] * 3, "--no-cache")
    assert cleared == [], f"a rerun that finds nothing must clear the clip's old observations: {cleared}"
    # an action outside the vocabulary gets role "unknown", not a guessed "suspect"
    odd = answer("CAM01", "exterior_above_entrance", [{**ev(MAN, "dance_near_atm", None, "00:00", 0.9, 0.9),
                                                         "free_content": "Person dances near the ATM."}])
    odd_obs, _ = run_one([final(odd)] * 3, "--no-cache")
    assert [o["role"] for o in odd_obs] == ["unknown"], odd_obs

    subprocess.run([sys.executable, "-m", "video_agent.evaluate", "--cases", str(CASE.parent), "--runs", str(runs)],
                   cwd=ROOT, check=True)
    print(f"\nALL CHECKS PASSED  (artifacts in {tmp})")


if __name__ == "__main__":
    main()
