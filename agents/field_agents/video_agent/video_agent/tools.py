"""
Tools the agent can call while analysing a clip. Every tool runs locally (ffmpeg, Pillow,
optional Tesseract OCR) and returns text plus, where useful, images or a video segment that are
sent back to the model.

  crop_overlay   enlarged crops of a corner of the frame (where the clock / camera label is),
                 at chosen times, with OCR text when Tesseract is installed
  check_clock    deterministic: do clock readings at different times tick in step with the video?
                 If yes, records the clock time at the first frame (used for timestamps)
  get_frames     full frames, or a crop of them, at chosen times
  rewatch        a trimmed segment of the clip at a higher frame rate
  known_people   people already seen in this case: appearance + reference still
  validate_draft run the field rules on a draft answer and list problems
"""
import io
import json
import re
import shutil
import subprocess
import tempfile
from datetime import timedelta
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

from . import config as C

CORNERS = {   # (x, y, w, h) as fractions of the frame
    "top-left": (0.0, 0.0, 0.55, 0.14), "top-right": (0.45, 0.0, 0.55, 0.14),
    "bottom-left": (0.0, 0.86, 0.55, 0.14), "bottom-right": (0.45, 0.86, 0.55, 0.14),
}

TOOL_SPECS = {
    "crop_overlay": {"args": {"corner": "top-left | top-right | bottom-left | bottom-right",
                              "times": "list of seconds into the clip, up to 4"},
                     "use": "Read the burned-in clock / camera label up close at several times."},
    "check_clock": {"args": {"readings": "list of {\"t\": seconds into clip, \"clock\": \"YYYY-MM-DD HH:MM:SS\"}"},
                    "use": "Verify the clock ticks with the video; returns the clock time at the first frame."},
    "get_frames": {"args": {"times": "list of seconds, up to 4",
                            "crop": "optional [x, y, w, h] as 0-1 fractions of the frame"},
                   "use": "Look closely at people, hands, the card slot, or the door at given moments."},
    "rewatch": {"args": {"start": "seconds", "end": "seconds", "fps": "frames per second, up to 12"},
                "use": "Re-watch a short segment at a higher frame rate when an action is unclear."},
    "known_people": {"args": {}, "use": "See reference stills of people already seen in this case."},
    "validate_draft": {"args": {"draft": "your draft final answer"},
                       "use": "Check a draft against the field rules before finishing."},
}


def ffmpeg_frame(clip, t, out):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(clip),
                    "-frames:v", "1", str(out)], check=True)
    return Image.open(out).convert("RGB")


def jpeg(img, max_side=1024):
    img = img.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def crop_frac(img, box):
    x, y, w, h = [max(0.0, min(1.0, float(v))) for v in box]
    W, H = img.size
    return img.crop((int(x * W), int(y * H), max(int(x * W) + 2, int((x + w) * W)), max(int(y * H) + 2, int((y + h) * H))))


def ocr_overlay(img):
    """Tesseract on an overlay crop: try a few binarisations / layouts, keep the first that parses."""
    import numpy as np
    import pytesseract
    g = np.asarray(ImageOps.grayscale(img), dtype=np.float32)
    best = ""
    for invert_dark_text in (False, True):
        for thr in (160, 130, 190):
            mask = g < thr if invert_dark_text else g > thr        # text pixels
            b = Image.fromarray(np.where(mask, 0, 255).astype("uint8"))
            for psm in (6, 11):
                txt = " ".join(pytesseract.image_to_string(b, config=f"--psm {psm}").split())
                if CLOCK_RE.search(txt):
                    return txt
                best = best or txt
    return best


CLOCK_RE = re.compile(r"(\d{4})[-/.](\d{2})[-/.](\d{2})\D{1,3}(\d{2})[:.](\d{2})[:.](\d{2})")


class ClipTools:
    def __init__(self, clip, duration, vocab, registry, ctx, locations):
        self.clip = Path(clip)
        self.duration = duration
        self.vocab, self.registry, self.ctx, self.locations = vocab, registry, ctx, locations
        self.tmp = Path(tempfile.mkdtemp(prefix="va_tools_"))
        self.verified_clock = None          # set only by check_clock
        self.calls = []
        self.ocr = shutil.which("tesseract") is not None
        if self.ocr:
            try:
                import pytesseract  # noqa: F401
            except ImportError:
                self.ocr = False

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _t(self, t):
        return max(0.0, min(float(t), max(0.0, self.duration - 0.05)))

    def call(self, name, args):
        fn = getattr(self, "tool_" + name, None)
        if fn is None:
            return {"text": f"unknown tool '{name}'. Available: {', '.join(TOOL_SPECS)}"}
        try:
            result = fn(**(args or {}))
        except Exception as e:  # report tool errors to the model instead of crashing the run
            result = {"text": f"tool error: {e}"}
        self.calls.append({"tool": name, "args": args, "result": result.get("text")})
        return result

    # ------------------------------------------------------------------ tools
    def tool_crop_overlay(self, corner="top-left", times=(0,)):
        box = CORNERS.get(corner, CORNERS["top-left"])
        images, lines = [], []
        for i, t in enumerate(list(times)[:4]):
            t = self._t(t)
            img = crop_frac(ffmpeg_frame(self.clip, t, self.tmp / f"ov{i}.png"), box)
            big = img.resize((img.width * 3, img.height * 3), Image.LANCZOS).filter(ImageFilter.SHARPEN)
            images.append(jpeg(big, 1400))
            line = f"t={t:.1f}s"
            if self.ocr:
                txt = ocr_overlay(big)
                m = CLOCK_RE.search(txt)
                line += f"  OCR: {txt!r}" + (f"  parsed: {m.group(1)}-{m.group(2)}-{m.group(3)} "
                                             f"{m.group(4)}:{m.group(5)}:{m.group(6)}" if m else "  parsed: none")
            lines.append(line)
        note = "" if self.ocr else " (OCR not installed: read the crops yourself)"
        return {"text": f"Enlarged {corner} crops{note}:\n" + "\n".join(lines), "images": images}

    def tool_check_clock(self, readings=()):
        from .agent import parse_ts
        pts = [(float(r["t"]), parse_ts(r.get("clock"))) for r in readings if isinstance(r, dict)]
        pts = [(t, ts) for t, ts in pts if ts is not None]
        if len(pts) < 2:
            return {"text": "need at least 2 readable clock readings at different times"}
        starts = [ts - timedelta(seconds=t) for t, ts in pts]
        spread = (max(starts) - min(starts)).total_seconds()
        if spread <= 1.5 and (max(t for t, _ in pts) - min(t for t, _ in pts)) >= 1.0:
            start = min(starts) + (max(starts) - min(starts)) / 2
            start = start.replace(microsecond=0)
            self.verified_clock = start
            return {"text": f"CONSISTENT: clock ticks with the video. Clock at first frame = "
                            f"{start.isoformat()} (spread {spread:.1f}s)."}
        return {"text": f"INCONSISTENT: readings imply first-frame clock times spread over {spread:.0f}s, "
                        f"so the overlay clock is unreliable. Report overlay timestamps as null."}

    def tool_get_frames(self, times=(0,), crop=None):
        images, lines = [], []
        for i, t in enumerate(list(times)[:4]):
            t = self._t(t)
            img = ffmpeg_frame(self.clip, t, self.tmp / f"fr{i}.png")
            if crop:
                img = crop_frac(img, crop)
                if max(img.size) < 512:
                    s = 512 / max(img.size)
                    img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
            images.append(jpeg(img))
            lines.append(f"t={t:.1f}s")
        return {"text": "Frames " + ", ".join(lines) + (f" cropped to {crop}" if crop else ""), "images": images}

    def tool_rewatch(self, start=0, end=None, fps=8):
        start = self._t(start)
        end = self._t(end if end is not None else start + 3)
        if end <= start:
            end = min(self.duration, start + 2)
        out = self.tmp / f"seg_{start:.1f}_{end:.1f}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-to", f"{end:.2f}",
                        "-i", str(self.clip), "-an", "-c:v", "libx264", "-crf", "20", str(out)], check=True)
        fps = max(1.0, min(12.0, float(fps)))
        return {"text": f"Segment {start:.1f}-{end:.1f}s at {fps:g} fps (times in it start from 0).",
                "video": str(out), "fps": fps}

    def tool_known_people(self):
        people = {k: v for k, v in self.registry.people.items() if v.get("identifiable", True)}
        if not people:
            return {"text": "No people seen yet in this case."}
        images, lines = [], []
        for label, p in people.items():
            lines.append(f"{label}: {p['appearance']} (seen in {', '.join(p['seen_in']) or 'seed'})")
            if p.get("still") and Path(p["still"]).exists():
                images.append(Path(p["still"]).read_bytes())
                lines[-1] += f"  [reference still #{len(images)}]"
        return {"text": "Known people:\n" + "\n".join(lines), "images": images}

    def tool_validate_draft(self, draft=None):
        issues = validate(draft or {}, self.vocab, self.registry, self.ctx, self.verified_clock)
        return {"text": "No problems found." if not issues else "Problems:\n- " + "\n- ".join(issues)}


def validate(raw, vocab, registry, ctx, verified_clock):
    """Field-rule checks shared by the validate_draft tool and the final gate."""
    from .agent import mmss, parse_ts, source_from_label
    issues = []
    if not isinstance(raw, dict):
        return ["answer is not a JSON object"]
    if not source_from_label(raw.get("overlay_camera_label")) and raw.get("camera_view") not in C.CAMERA_VIEW_TO_SOURCE:
        issues.append("camera: overlay label not recognised and camera_view not one of the listed views")
    if raw.get("footage_issue") not in (None, *C.VIDEO_QUALITY_BANK):
        issues.append(f"footage_issue must be null or one of {list(C.VIDEO_QUALITY_BANK)}")
    if verified_clock is None and raw.get("clock_verified") is not False:
        issues.append("clock not verified: use crop_overlay at 2-3 times, then check_clock "
                      "(if it is unreadable or INCONSISTENT, say so with clock_verified=false)")
    events = raw.get("events")
    if not isinstance(events, list):
        return issues + ["events must be a list"]
    if not events and not raw.get("footage_issue"):
        issues.append("no events and no footage_issue: if nothing case-relevant is visible, set footage_issue")
    plural = C.plural_sentences(vocab)
    leaving = [i for i, ev in enumerate(events) if isinstance(ev, dict) and ev.get("action") in C.LEAVING_ACTIONS]
    for i, ev in enumerate(events):
        tag = f"event {i}"
        if ev.get("action") == "observe_exit" and leaving and ev.get("stays_behind") is not True:
            issues.append(f"{tag}: this person is reported as a witness watching event {leaving[0]} leave. Use "
                          f"get_frames in the second half of the clip to see where they go. If they walk off "
                          f"with the person they watched, merge both into ONE group event (rule 7: group=true, "
                          f"\"people\", a plural sentence). If they stay behind or go their own way, set "
                          f"\"stays_behind\": true on this event")
        a = ev.get("action")
        sentence = ev.get("free_content")
        if a not in vocab:
            issues.append(f"{tag}: action '{a}' not in VOCABULARY")
        else:
            idx = ev.get("template_index")
            if isinstance(idx, int) and 0 <= idx < len(vocab[a]["templates"]):
                sentence = vocab[a]["templates"][idx]
            elif not ev.get("free_content"):
                issues.append(f"{tag}: template_index must be 0..{len(vocab[a]['templates']) - 1} for {a}")
        people = ev.get("people") if isinstance(ev.get("people"), list) else []
        if ev.get("group"):
            if len(people) < 2:
                issues.append(f'{tag}: a group event needs "people" with at least 2 members (lead first)')
            if a in vocab and not C.is_plural(sentence):
                where = a if a in plural else C.GROUP_ACTION.get(a)
                if where in plural:
                    issues.append(f"{tag}: group event with a single-person sentence. Use a plural sentence: "
                                  f"action {where}, template_index one of {plural[where]}")
        elif C.is_plural(sentence):
            only_plural = a in vocab and len(plural.get(a, [])) == len(vocab[a]["templates"])
            issues.append(f"{tag}: plural sentence ('{sentence}') on a non-group event. If several people do "
                          f'this together set group=true and list them in "people"; otherwise pick a '
                          f"single-person sentence" + (f" from another action ({a} has none)" if only_plural else ""))
        for who, p in [("", ev)] + [(f" people[{j}]", q) for j, q in enumerate(people) if isinstance(q, dict)]:
            m = p.get("registry_match")
            if m and m not in registry.people:
                issues.append(f"{tag}{who}: registry_match '{m}' is not a known person (use known_people)")
            if m and not p.get("identifiable", True):
                issues.append(f"{tag}{who}: an unidentifiable person cannot have a registry_match")
        if not re.match(r"^\d+:\d{1,2}", str(ev.get("start", ""))):
            issues.append(f"{tag}: start must be MM:SS")
        for k in ("certainty", "visibility"):
            try:
                float(ev.get(k))
            except (TypeError, ValueError):
                issues.append(f"{tag}: {k} must be a number 0-1")
        ts = parse_ts(ev.get("overlay_timestamp_at_start"))
        if ts is not None and verified_clock is not None:
            expect = verified_clock + timedelta(seconds=mmss(ev.get("start")))
            if abs((ts - expect).total_seconds()) > 2:
                issues.append(f"{tag}: overlay_timestamp_at_start {ts} disagrees with the verified clock "
                              f"({expect} at {ev.get('start')})")
        if ev.get("bbox") is not None:
            b = ev["bbox"]
            if not (isinstance(b, list) and len(b) == 4):
                issues.append(f"{tag}: bbox must be [x, y, w, h] fractions or null")
    return issues


def tool_catalogue():
    return json.dumps(TOOL_SPECS, indent=1)
