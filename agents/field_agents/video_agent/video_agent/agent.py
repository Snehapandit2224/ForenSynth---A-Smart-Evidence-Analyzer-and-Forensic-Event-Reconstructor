"""
VideoAgent: one CCTV clip -> observation records with the case-file fields

    obs_id, entity, role, modality, source, location, content, timestamp, time_offset, confidence

Division of labour
  vision model : what is visible (overlay text, camera view, people, actions, sentence choice,
                 certainty/visibility). It never sees other observations or ground truth.
  this code    : everything rule-based, copied from the generator so fields line up:
                 source label mapping, location lookup, role-from-action, Person_NN labels,
                 time_offset from the case anchor, and confidence from measured quality.
"""
import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

from . import config as C
from .backends import BACKENDS
from .loop import run_loop
from .prompt import SYSTEM, build_user_prompt
from .quality import measure
from .registry import PeopleRegistry
from .tools import ClipTools, crop_frac, ffmpeg_frame

FIELDS = ["obs_id", "entity", "role", "modality", "source", "location",
          "content", "timestamp", "time_offset", "confidence"]


# ---------------------------------------------------------------- helpers
def parse_ts(text):
    if not text:
        return None
    text = str(text).strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            continue
    return None


def mmss(text):
    m = re.match(r"^\s*(\d+):(\d{1,2})(?:\.\d+)?\s*$", str(text or ""))
    return int(m.group(1)) * 60 + int(m.group(2)) if m else 0


def num(x, default=0.5):
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return default


def make_context(case_id, anchor=C.DEFAULT_ANCHOR):
    """Everything the agent knows about the case: its ID (-> domain) and where time_offset starts."""
    return {"case_id": case_id, "domain": C.domain_from_case_id(case_id), "anchor": parse_ts(anchor)}


def source_from_label(label):
    if not label:
        return None
    key = re.sub(r"[\s_\-:#]", "", str(label).upper())
    key = re.sub(r"^(CAM|CAMERA)O", r"\g<1>0", key)        # OCR / model reading "CAMO1" for "CAM01"
    if key in C.OVERLAY_LABEL_TO_SOURCE:
        return C.OVERLAY_LABEL_TO_SOURCE[key]
    for k in sorted(C.OVERLAY_LABEL_TO_SOURCE, key=len, reverse=True):
        if key.startswith(k) and len(k) >= 4:
            return C.OVERLAY_LABEL_TO_SOURCE[k]
    # tolerate small misreads (AIM-INT for ATM-INT, CAMO03 for CAM03)
    import difflib
    close = difflib.get_close_matches(key, list(C.OVERLAY_LABEL_TO_SOURCE), n=1, cutoff=0.8)
    return C.OVERLAY_LABEL_TO_SOURCE[close[0]] if close else None


def source_from_filename(name):
    low = name.lower()
    for s in sorted(C.VIDEO_SOURCES, key=len, reverse=True):
        if s in low:
            return s
    return None


# ---------------------------------------------------------------- agent
class VideoAgent:
    def __init__(self, ctx, out_dir, backend="gemini", model=C.DEFAULT_MODEL, fps=C.DEFAULT_FPS,
                 registry_seed=None, overrides=None, use_cache=True, keep_registry=False, max_steps=8):
        self.ctx = ctx
        self.out_dir = Path(out_dir)
        self.vocab = C.load_vocabulary(ctx["domain"])
        self.locations = C.VIDEO_LOCATIONS.get(ctx["domain"], {})
        self.backend = BACKENDS[backend](model=model, fps=fps)
        self.fps, self.max_steps = fps, max_steps
        self.model_id = f"{backend}:{model}@{fps}fps:agent-v2"
        self.registry = PeopleRegistry(self.out_dir / "people_registry.json", ctx["case_id"], registry_seed,
                                       keep_existing=keep_registry)
        self.overrides = overrides or {}
        self.use_cache = use_cache
        self.cache_dir = self.out_dir / "cache"
        cal = self.out_dir.parent / "calibration.json"
        self.calibration = json.loads(cal.read_text()) if cal.exists() else None

    # ---- agent loop with on-disk cache (same clip + prompt + model -> same answer, no API calls)
    def _analyze(self, clip, duration):
        user = build_user_prompt(self.vocab, self.registry.people)
        h = hashlib.sha256(Path(clip).read_bytes())
        h.update((self.model_id + SYSTEM + user + str(self.max_steps)).encode())
        cache = self.cache_dir / f"{h.hexdigest()[:24]}.json"
        if self.use_cache and cache.exists():
            data = json.loads(cache.read_text(encoding="utf-8"))
            data["cached"] = True
            return data
        tools = ClipTools(clip, duration, self.vocab, self.registry, self.ctx, self.locations)
        try:
            final, trace, issues = run_loop(self.backend, clip, self.fps, SYSTEM, user, tools, self.ctx,
                                            max_steps=self.max_steps)
        finally:
            tools.close()
        data = {"final": final, "trace": trace, "unresolved_issues": issues,
                "verified_clock": tools.verified_clock.isoformat() if tools.verified_clock else None,
                "tool_calls": len(tools.calls)}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        data["cached"] = False
        return data

    def _save_still(self, clip, label, ev):
        """Reference image of a newly seen person, shown to the agent in later clips."""
        p = self.registry.people[label]
        if p.get("still") or not p.get("identifiable", True):
            return
        try:
            t = float(ev.get("ref_time") or mmss(ev.get("start")))
            people_dir = self.out_dir / "people"
            people_dir.mkdir(parents=True, exist_ok=True)
            img = ffmpeg_frame(clip, t, people_dir / f"_{label}.png")
            (people_dir / f"_{label}.png").unlink()
            bbox = ev.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                x, y, w, h = [float(v) for v in bbox]
                img = crop_frac(img, [x - w * 0.15, y - h * 0.1, w * 1.3, h * 1.2])
            img.thumbnail((640, 640))
            out = people_dir / f"{label}.jpg"
            img.save(out, "JPEG", quality=88)
            p["still"] = str(out)
        except Exception as e:   # a missing still never blocks extraction
            p["still_error"] = str(e)

    @staticmethod
    def _members(ev):
        """The people in an event, lead first. A group event lists them in "people"; the lead's
        details on the event itself win over the lead's entry in that list."""
        keys = ("person", "identifiable", "registry_match", "ref_time", "bbox")
        lead = {k: ev[k] for k in keys if ev.get(k) is not None}
        people = ev.get("people") if ev.get("group") and isinstance(ev.get("people"), list) else []
        people = [p if isinstance(p, dict) else {"person": str(p)} for p in people if p]
        if not people:
            return [lead]
        same = lambda p: str(p.get("person", "")).strip().lower() == str(lead.get("person", "")).strip().lower()
        return [{**people[0], **lead}] + [p for p in people[1:] if not same(p)]

    def _confidence(self, certainty, visibility, quality, issue, ts_verified, identifiable, ts_found):
        w, p = C.CONF_WEIGHTS, C.CONF_PENALTY
        x = w["certainty"] * certainty + w["visibility"] * visibility + w["quality"] * quality
        x -= p["footage_issue"] if issue else 0
        x -= 0 if ts_verified else (p["timestamp_unread"] / 2 if ts_found else p["timestamp_unread"])
        x -= 0 if identifiable else p["unidentifiable"]
        if self.calibration:
            x = self.calibration["a"] * x + self.calibration["b"]
        return round(max(C.CONF_RANGE[0], min(C.CONF_RANGE[1], x)), 3)

    def _content(self, ev, issue):
        action = ev.get("action")
        if action not in self.vocab:
            return None, ev.get("free_content") or None
        templates = self.vocab[action]["templates"]
        if issue and num(ev.get("visibility")) < 0.35:      # footage, not the act, is the finding
            return action, C.VIDEO_QUALITY_BANK[issue]
        idx = ev.get("template_index")
        if isinstance(idx, int) and 0 <= idx < len(templates):
            return action, templates[idx]
        return action, ev.get("free_content") or templates[0]

    def _timestamp(self, ev, raw, verified_clock, override_start, flags):
        """Override > clock verified by the check_clock tool > the model's own overlay readings."""
        start_s = mmss(ev.get("start"))
        if override_start:
            ts, how = parse_ts(override_start) + timedelta(seconds=start_s), "override+offset"
        elif verified_clock:
            ts, how = parse_ts(verified_clock) + timedelta(seconds=start_s), "verified_clock+offset"
        elif parse_ts(ev.get("overlay_timestamp_at_start")) and raw.get("clock_verified") is not False:
            ts, how = parse_ts(ev.get("overlay_timestamp_at_start")), "overlay_at_event(unverified)"
        elif parse_ts(raw.get("overlay_timestamp_first")) and raw.get("clock_verified") is not False:
            ts, how = parse_ts(raw.get("overlay_timestamp_first")) + timedelta(seconds=start_s), "overlay_first+offset(unverified)"
        else:
            flags.append("timestamp_unreadable")
            return None, None, "none"
        if "unverified" in how:
            flags.append("clock_not_verified")
        lo, hi = (self.ctx["anchor"] + timedelta(hours=h) for h in C.SANITY_WINDOW_HOURS)
        if not lo <= ts <= hi:
            flags.append("timestamp_outside_expected_range")
        offset = max(0, int(round((ts - self.ctx["anchor"]).total_seconds())))
        return ts.isoformat(), offset, how

    def process_clip(self, clip, obs_id):
        clip = Path(clip)
        ov = self.overrides.get(clip.name, {})
        quality = measure(clip)
        run = self._analyze(clip, quality["duration"])
        raw, cached = run["final"] or {}, run["cached"]

        flags = ["unresolved_checks"] if run["unresolved_issues"] else []
        source = ov.get("source") or source_from_label(raw.get("overlay_camera_label"))
        src_how = "override" if ov.get("source") else "overlay_label"
        if not source:
            source, src_how = source_from_filename(clip.name), "filename"
        if not source:
            source, src_how = C.CAMERA_VIEW_TO_SOURCE.get(raw.get("camera_view")), "camera_view"
        if not source:
            source, src_how = "unknown", "none"
            flags.append("source_unknown")

        issue = raw.get("footage_issue") if raw.get("footage_issue") in C.VIDEO_QUALITY_BANK else None
        if not quality["usable"] and not issue:
            issue = "low_quality"
        events = raw.get("events") or []
        if not events and issue:   # nothing visible but the footage problem itself
            events = [{"person": "unidentified person", "identifiable": False, "action": None,
                       "visibility": 0.0, "certainty": 0.3, "start": "00:00"}]

        observations, details = [], []
        for i, ev in enumerate(events):
            oid = obs_id if i == 0 else f"{obs_id}{chr(ord('a') + i)}"
            ev_flags = list(flags)
            action, content = self._content(ev, issue)
            if content is None:
                content = C.VIDEO_QUALITY_BANK[issue or "angle_unclear"]
                ev_flags.append("no_vocabulary_match")
            elif action is None or self.vocab.get(action, {}).get("templates", [None]).count(content) == 0:
                if content not in C.VIDEO_QUALITY_BANK.values():
                    ev_flags.append("free_text_content")
            role = self.vocab[action]["role"] if action in self.vocab else "witness" \
                if any(k in content.lower() for k in ("bystander", "passerby", "witness")) else "suspect"
            # a group event is one observation: every member gets a label, the lead's is the entity
            labels = []
            for m in self._members(ev):
                match = m.get("registry_match")
                label = self.registry.resolve(None if match in labels else match, m.get("person", ""),
                                              bool(m.get("identifiable", True)), oid)
                self._save_still(clip, label, {"ref_time": ev.get("ref_time"), "start": ev.get("start"), **m})
                labels.append(label)
            entity, identifiable = labels[0], bool(ev.get("identifiable", True))
            ts, offset, ts_how = self._timestamp(ev, raw, run["verified_clock"], ov.get("start"), ev_flags)
            if "timestamp_outside_expected_range" in ev_flags and not issue:
                issue_for_conf = "timestamp_inconsistent"
            else:
                issue_for_conf = issue
            conf = self._confidence(num(ev.get("certainty")), num(ev.get("visibility")),
                                    min(quality["score"], num(raw.get("image_quality"), quality["score"])),
                                    issue_for_conf, ts_how.startswith(("override", "verified")),
                                    identifiable, ts is not None)
            observations.append({
                "obs_id": oid, "entity": entity, "role": role, "modality": "video",
                "source": source, "location": self.locations.get(source, C.LOCATION_FALLBACK),
                "content": content, "timestamp": ts, "time_offset": offset, "confidence": conf,
            })
            details.append({"obs_id": oid, "action": action, "source_from": src_how,
                            "timestamp_from": ts_how, "flags": ev_flags, "person": ev.get("person"),
                            "identifiable": identifiable})
            if ev.get("group"):
                details[-1]["co_entities"] = labels[1:]

        log = {"clip": clip.name, "obs_ids": [o["obs_id"] for o in observations], "model": self.model_id,
               "cached": cached, "quality": quality, "footage_issue": issue,
               "verified_clock_at_first_frame": run["verified_clock"], "tool_calls": run["tool_calls"],
               "unresolved_checks": run["unresolved_issues"], "agent_trace": run["trace"],
               "final_answer": raw, "decisions": details}
        return observations, log

    @staticmethod
    def obs_id_for(clip, n, given=None):
        if given:
            return given
        m = re.match(r"^(O\d+)", Path(clip).stem, re.I)
        return m.group(1).upper() if m else f"V{n}"

    def process_clips(self, clips, obs_ids):
        all_obs, logs = [], []
        for clip, obs_id in zip(clips, obs_ids):
            print(f"[video] {Path(clip).name} -> {obs_id}", flush=True)
            self.registry.forget(obs_id)          # re-running a clip replaces its earlier result
            obs, log = self.process_clip(clip, obs_id)
            for o, d in zip(obs, log["decisions"]):
                tag = f"  ! {', '.join(d['flags'])}" if d["flags"] else ""
                print(f"   {o['obs_id']:<5} {o['entity']} {o['role']:<7} {o['source']:<13} "
                      f"{o['timestamp']}  conf {o['confidence']}  {o['content']}{tag}", flush=True)
            all_obs += obs
            logs.append(log)
        self.registry.save()
        return all_obs, logs
