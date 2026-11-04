"""
Model backends for the agent loop. A conversation is a list of turns:

    {"role": "user" | "model", "parts": [{"text": str} | {"image": bytes} | {"video": path, "fps": float}]}

  gemini  Gemini API (free tier key from AI Studio); video, images and text in every turn
  mock    replays a scripted conversation from <clip>.mock.json; used by the offline tests
"""
import json
import os
import re
import time
from pathlib import Path


def parse_json(text):
    text = re.sub(r"```(?:json)?", "", text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError(f"model did not return JSON: {text[:300]}")
        return json.loads(m.group(0))


class GeminiBackend:
    name = "gemini"
    INLINE_LIMIT = 18 * 1024 * 1024
    TIMEOUT_S = 180                  # per request
    OVERLOAD_RETRIES = 10            # 503 "high demand": waits 8, 16 ... 180 s, about 16 min in all

    def __init__(self, model, fps, retries=6, **_):
        # GEMINI_API_KEY, then GEMINI_API_KEY_2, _3 ...: when one key's daily quota is used up the next
        # one takes over (keys from different accounts / projects have separate quotas)
        names = ["GEMINI_API_KEY"] + [f"GEMINI_API_KEY_{i}" for i in range(2, 10)] + ["GOOGLE_API_KEY"]
        self.keys = list(dict.fromkeys(os.environ[n].strip() for n in names if os.environ.get(n, "").strip()))
        if not self.keys:
            raise SystemExit("Set GEMINI_API_KEY (free key: https://aistudio.google.com/apikey).")
        self.key_index = 0
        self.client = self._client()
        self.model, self.fps, self.retries = model, fps, retries
        self._uploaded = {}

    def _client(self):
        from google import genai
        from google.genai import types
        # without a timeout a dropped connection waits forever; a timed-out request is retried below
        return genai.Client(api_key=self.keys[self.key_index],
                            http_options=types.HttpOptions(timeout=self.TIMEOUT_S * 1000))

    def _next_key(self):
        """Switch to the next API key; False if there is none left."""
        if self.key_index + 1 >= len(self.keys):
            return False
        self.key_index += 1
        self.client = self._client()
        self._uploaded = {}          # uploaded files belong to the old key's project
        print(f"    daily quota used up, switching to API key {self.key_index + 1} of {len(self.keys)}", flush=True)
        return True

    def _video_part(self, path, fps):
        from google.genai import types
        meta = types.VideoMetadata(fps=fps)
        if Path(path).stat().st_size <= self.INLINE_LIMIT:
            return types.Part(inline_data=types.Blob(data=Path(path).read_bytes(), mime_type="video/mp4"),
                              video_metadata=meta)
        if path not in self._uploaded:
            f = self.client.files.upload(file=str(path))
            while f.state.name == "PROCESSING":
                time.sleep(2)
                f = self.client.files.get(name=f.name)
            if f.state.name == "FAILED":
                raise RuntimeError(f"Gemini could not process {path}")
            self._uploaded[path] = f
        f = self._uploaded[path]
        return types.Part(file_data=types.FileData(file_uri=f.uri, mime_type=f.mime_type), video_metadata=meta)

    def _contents(self, turns):
        from google.genai import types
        out = []
        for turn in turns:
            parts = []
            for p in turn["parts"]:
                if "text" in p:
                    parts.append(types.Part(text=p["text"]))
                elif "image" in p:
                    parts.append(types.Part(inline_data=types.Blob(data=p["image"], mime_type="image/jpeg")))
                elif "video" in p:
                    parts.append(self._video_part(p["video"], p.get("fps", self.fps)))
            out.append(types.Content(role=turn["role"], parts=parts))
        return out

    @staticmethod
    def _function_call(resp):
        """The model sometimes replies with a native function call instead of JSON text: same tool call."""
        for cand in resp.candidates or []:
            for part in (cand.content.parts if cand.content and cand.content.parts else []):
                fc = getattr(part, "function_call", None)
                if fc and fc.name:
                    return json.dumps({"thought": "", "tool": fc.name, "args": dict(fc.args or {})})
        return None

    def chat(self, system, turns):
        from google.genai import types
        config = types.GenerateContentConfig(system_instruction=system, temperature=0.0,
                                             response_mime_type="application/json")
        contents = self._contents(turns)
        delay, attempt = 8, 0
        while attempt < self.retries:
            attempt += 1
            try:
                resp = self.client.models.generate_content(model=self.model, contents=contents, config=config)
                return resp.text or self._function_call(resp) or ""
            except Exception as e:  # free-tier rate limits are the usual cause
                msg = str(e)
                if "PerDay" in msg:      # daily quota used up: waiting will not help, another key might
                    if not self._next_key():
                        raise
                    contents = self._contents(turns)      # re-upload large videos under the new key
                    delay, attempt = 8, 0
                    continue
                timed_out = "timed out" in msg.lower() or "timeout" in type(e).__name__.lower()
                # model overloaded ("high demand"): spikes last minutes, so wait longer and more often
                overloaded = "503" in msg or "UNAVAILABLE" in msg
                limit = self.OVERLOAD_RETRIES if overloaded else self.retries
                if attempt >= limit or not (timed_out or overloaded or any(k in msg for k in ("429", "RESOURCE_EXHAUSTED"))):
                    raise
                why = "request timed out" if timed_out else "model overloaded" if overloaded else "rate limited"
                print(f"    {why}, retrying in {delay}s", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 180 if overloaded else 64)


class MockBackend:
    """<clip>.mock.json = {"script": [reply, reply, ...]} (one reply per model turn), or a bare final answer."""
    name = "mock"

    def __init__(self, *_, **__):
        pass

    def chat(self, system, turns):
        clip = next(p["video"] for p in turns[0]["parts"] if "video" in p)   # the clip under analysis
        mock = Path(clip).with_suffix(".mock.json")
        data = json.loads(mock.read_text(encoding="utf-8"))
        script = data["script"] if isinstance(data, dict) and "script" in data else [{"final": data}]
        k = sum(1 for t in turns if t["role"] == "model")
        return json.dumps(script[min(k, len(script) - 1)])


BACKENDS = {"gemini": GeminiBackend, "mock": MockBackend}
