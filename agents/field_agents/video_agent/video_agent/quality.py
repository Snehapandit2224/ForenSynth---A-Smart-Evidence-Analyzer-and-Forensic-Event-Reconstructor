"""
Measured (not model-judged) footage quality, from frames sampled with ffmpeg.

Returns a 0-1 score plus the raw metrics. Degraded CCTV (low resolution,
grain, compression, dropped frames, flicker) scores lower, which feeds the
confidence field the same way the generator ties confidence to evidence quality.
"""
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    info = json.loads(out.stdout)
    s = info["streams"][0]
    num, den = (s.get("avg_frame_rate") or "0/1").split("/")
    return {
        "width": int(s["width"]), "height": int(s["height"]),
        "fps": float(num) / float(den) if float(den) else 0.0,
        "duration": float(info["format"]["duration"]),
    }


def _frames(path, fps, width=640):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", str(path),
             "-vf", f"fps={fps},scale={width}:-2,format=gray", f"{tmp}/f%04d.png"],
            check=True)
        return [np.asarray(Image.open(p), dtype=np.float32) for p in sorted(Path(tmp).glob("*.png"))]


def _hf_ratio(img):
    """High-frequency share of the image spectrum: drops when footage was upscaled or blurred."""
    a = img - img.mean()
    h, w = a.shape
    spec = np.abs(np.fft.fftshift(np.fft.fft2(a * np.outer(np.hanning(h), np.hanning(w))))) ** 2
    yy, xx = np.mgrid[-h // 2:h - h // 2, -w // 2:w - w // 2]
    r = np.sqrt((yy / (h / 2)) ** 2 + (xx / (w / 2)) ** 2)
    low = spec[(r > 0.05) & (r <= 0.5)].sum()
    return float(spec[(r > 0.5) & (r < 1.0)].sum() / low) if low else 0.0


def measure(path, fps=4):
    """
    Score in [0, 1]:  sqrt(sharpness) x noise factor x flicker factor x brightness factor.
    Grain is removed with a temporal median before measuring sharpness (CCTV cameras
    are static, so the median keeps the scene and drops sensor noise).
    """
    meta = probe(path)
    frames = _frames(path, fps)
    if len(frames) < 2:
        return {"score": 0.0, "usable": False, **meta}

    stack = np.stack(frames)
    windows = [stack[i:i + 7] for i in range(0, len(stack) - 2, 7)]
    sharp, noise = [], []
    for w in windows:
        med = np.median(w, axis=0)
        sharp.append(_hf_ratio(med))
        noise.append(float(np.median(np.abs(w - med)) * 1.4826))
    bright = stack.mean(axis=(1, 2))
    flicker = float(np.abs(np.diff(bright)).mean() / max(bright.mean(), 1.0))

    sharp_s = float(np.clip(np.median(sharp) / 0.30, 0, 1))
    noise_s = float(np.clip(1 - np.median(noise) / 15.0, 0, 1))
    flicker_s = float(np.clip(1 - flicker * 12, 0, 1))
    bright_s = float(np.clip((bright.mean() - 25) / 50.0, 0, 1))
    score = np.sqrt(sharp_s) * (0.5 + 0.5 * noise_s) * (0.5 + 0.5 * flicker_s) * bright_s

    return {
        "score": round(float(score), 3),
        "usable": bool(score > 0.05 and bright.max() > 10),
        "sharpness": round(sharp_s, 3), "noise": round(noise_s, 3),
        "flicker": round(flicker_s, 3), "brightness": round(bright_s, 3),
        **meta,
    }
