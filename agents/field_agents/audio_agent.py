#!/usr/bin/env python3
"""
ForenSynth – audio_agent.py
Audio Field Agent: raw audio evidence -> obs_only JSON for Entity Resolution.

Pipeline:
    1. Preprocess audio (mono, 16 kHz, normalized)
    2. Transcribe with baseline Whisper (segment-level timestamps)
    3. Diarize speakers with pyannote (optional - needs a HuggingFace token)
    4. Merge transcript segments with speaker labels
    5. Emit one observation per utterance in ForenSynth's obs_only format

Principles: evidence extraction only. No crime inference, no guilt
attribution, no action labelling - ER and the Timeline Agent handle that
downstream.

Usage:
    python agents\\field_agents\\audio_agent.py --input evidence\\call.wav ^
        --case-id CASE_AUD_001 --case-base-time 2024-01-15T10:00:00 ^
        --location "near ATM location, mobile network"

    # then feed the result straight into ER
    python pipeline\\run_er.py --input output\\audio\\CASE_AUD_001_obs_only.json

    # several recordings for one case -> one obs_only file
    python agents\\field_agents\\audio_agent.py --manifest evidence\\clips.json --case-id CASE_AUD_001
    # clips.json: [{"path": "call1.wav", "start_time": "2024-01-15T10:00:00",
    #               "location": "...", "source": "phone_record"}, ...]

Environment (.env):
    HF_TOKEN    enables real pyannote speaker diarization (or pass --hf-token)

Dependencies (not in the main requirements.txt - see requirements-audio.txt):
    openai-whisper, librosa, soundfile, torch; pyannote.audio is optional.

Fine-tuning: Whisper fine-tuning on AMI/Switchboard is supported but optional.
The original notebook cells are kept, commented out, at the bottom of this
file. This agent always runs baseline Whisper.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

AGENT_VERSION = "AudioAgent_v1"
MODEL_VERSION = "whisper_baseline"
SAMPLE_RATE = 16000  # Whisper requires 16 kHz mono
UNKNOWN_SPEAKER = "Speaker_Unknown"


# ==============================================================================
# Audio preprocessing
# ==============================================================================

def preprocess_audio(audio_path: str):
    """Load audio as a mono, 16 kHz, [-1, 1] float32 array."""
    import librosa
    import numpy as np

    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    if audio.ndim > 1:
        audio = librosa.to_mono(audio)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio.astype(np.float32)


# ==============================================================================
# ASR (baseline Whisper)
# ==============================================================================

def load_whisper(model_name: str, device: str):
    import whisper

    print(f"Loading Whisper '{model_name}' on {device} ...")
    return whisper.load_model(model_name, device=device)


def transcribe(audio, model, language: Optional[str]) -> List[Dict[str, Any]]:
    """Segment-level timestamped transcription with baseline Whisper."""
    result = model.transcribe(
        audio,
        verbose=False,
        language=language,
        word_timestamps=False,  # segment-level is more stable
    )
    return result.get("segments", [])


# ==============================================================================
# Speaker diarization (optional)
# ==============================================================================

def setup_diarization_pipeline(hf_token: Optional[str], device: str):
    """Load the pyannote diarization pipeline, or None if unavailable."""
    try:
        import torch
        from pyannote.audio import Pipeline

        # pyannote.audio 4.x renamed use_auth_token -> token
        import inspect
        token_arg = "token" if "token" in inspect.signature(Pipeline.from_pretrained).parameters else "use_auth_token"
        kwargs = {token_arg: hf_token} if hf_token else {}
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", **kwargs)
        if pipeline is None:
            raise RuntimeError("pipeline unavailable (missing/invalid HuggingFace token?)")
        if device == "cuda":
            pipeline = pipeline.to(torch.device("cuda"))
        return pipeline
    except Exception as exc:
        print(f"[WARNING] Diarization unavailable ({exc})")
        print(f"          All utterances will be attributed to '{UNKNOWN_SPEAKER}'.")
        return None


def perform_diarization(audio, pipeline):
    """
    Returns (segments, voice_prints):
        segments     [{speaker, start, end}, ...]
        voice_prints {speaker: unit-length embedding} - what lets ER recognise
                     the same person in another recording.

    Both are empty when no pipeline is available: speakers are never
    simulated, since invented speaker turns would reach ER as fabricated
    entities.
    """
    if pipeline is None:
        return [], {}
    import numpy as np
    import torch

    output = pipeline({
        "waveform": torch.from_numpy(audio).unsqueeze(0),
        "sample_rate": SAMPLE_RATE,
    })
    # pyannote.audio 4.x wraps the Annotation in a DiarizeOutput
    diarization = getattr(output, "speaker_diarization", output)
    segments = [
        {"speaker": speaker, "start": turn.start, "end": turn.end}
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]

    voice_prints: Dict[str, List[float]] = {}
    embeddings = getattr(output, "speaker_embeddings", None)  # one row per label, in labels() order
    if embeddings is not None:
        for label, vec in zip(diarization.labels(), embeddings):
            norm = float(np.linalg.norm(vec))
            if np.isfinite(norm) and norm > 0:  # NaN row = too little speech to embed
                voice_prints[label] = [round(float(x), 4) for x in vec / norm]
    return segments, voice_prints


def merge_transcripts_with_speakers(
    whisper_segments: List[Dict[str, Any]],
    diarization_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Assign each Whisper segment the speaker with the largest temporal overlap."""
    merged = []
    for seg in whisper_segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        max_overlap = 0.0
        speaker = UNKNOWN_SPEAKER
        for diar in diarization_segments:
            overlap = min(seg["end"], diar["end"]) - max(seg["start"], diar["start"])
            if overlap > max_overlap:
                max_overlap = overlap
                speaker = diar["speaker"]
        merged.append({
            "speaker": speaker,
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": text,
            "avg_logprob": seg.get("avg_logprob", -0.5),
        })
    return merged


# ==============================================================================
# Confidence estimation
# ==============================================================================

def estimate_confidence(segment: Dict[str, Any]) -> float:
    """
    Combine ASR confidence (Whisper avg_logprob), segment duration (saturates
    at 5 s) and speaker-assignment certainty into a [0, 1] score.
    """
    asr_confidence = math.exp(min(float(segment.get("avg_logprob", -0.5)), 0.0))
    duration_factor = min((segment["end"] - segment["start"]) / 5.0, 1.0)
    speaker_factor = 0.7 if segment["speaker"] == UNKNOWN_SPEAKER else 1.0
    return max(0.0, min(1.0, asr_confidence * duration_factor * speaker_factor))


# ==============================================================================
# ForenSynth observation schema
# ==============================================================================

def parse_base_time(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not ISO 8601 (expected e.g. 2024-01-15T10:00:00)")


def build_observations(
    segments: List[Dict[str, Any]],
    base_time: datetime,
    location: str,
    source: str,
    role: str,
    audio_file: str,
    id_prefix: str = "AUD1",
    case_start: Optional[datetime] = None,
    voice_prints: Optional[Dict[str, List[float]]] = None,
) -> List[Dict[str, Any]]:
    """
    One observation per utterance, in the same shape as the generator's
    obs_only observations. `timestamp` is base_time (when this recording
    starts) + the segment's start offset; `time_offset` is seconds since
    case_start (the earliest recording in the case). `source` must stay a
    plain string (it is a DB column).

    Speaker labels are prefixed with `source`: diarization numbers speakers
    from scratch per recording, so a bare SPEAKER_00 from two different clips
    would look like the same person to ER. Deciding that two labels ARE the
    same person is ER's job: each observation carries its speaker's voice
    print as `identity_embedding` for ER to compare across recordings.
    """
    voice_prints = voice_prints or {}
    clip_offset = (base_time - case_start).total_seconds() if case_start else 0.0
    observations = []
    for idx, seg in enumerate(segments, 1):
        obs = {
            "obs_id": f"{id_prefix}_O{idx}",
            "entity": f"{source}_{seg['speaker']}",
            "role": role,
            "modality": "audio",
            "source": source,
            "location": location,
            "content": seg["text"],
            "timestamp": (base_time + timedelta(seconds=seg["start"])).isoformat(timespec="seconds"),
            "time_offset": int(clip_offset + seg["start"]),
            "confidence": round(estimate_confidence(seg), 3),
            # provenance - ignored by ER/Timeline, preserved in the DB's raw_json
            "audio_file": audio_file,
            "segment_end_offset": round(seg["end"], 2),
            "asr_model": MODEL_VERSION,
            "extracted_by": AGENT_VERSION,
        }
        if seg["speaker"] in voice_prints:
            obs["identity_embedding"] = voice_prints[seg["speaker"]]
        observations.append(obs)
    return observations


def validate_observations(observations: List[Dict[str, Any]]) -> List[str]:
    """Sanity checks against what ER / the Timeline Agent require."""
    issues = []
    required = ["obs_id", "entity", "modality", "source", "location",
                "content", "timestamp", "time_offset", "confidence"]
    seen_ids = set()
    for idx, obs in enumerate(observations):
        missing = [f for f in required if f not in obs or obs[f] in ("", None)]
        if missing:
            issues.append(f"observations[{idx}]: missing/empty {missing}")
        if obs.get("obs_id") in seen_ids:
            issues.append(f"observations[{idx}]: duplicate obs_id {obs.get('obs_id')}")
        seen_ids.add(obs.get("obs_id"))
        if not 0.0 <= float(obs.get("confidence", -1)) <= 1.0:
            issues.append(f"observations[{idx}]: confidence out of [0, 1]")
        if not isinstance(obs.get("source"), str):
            issues.append(f"observations[{idx}]: 'source' must be a string")
    return issues


# ==============================================================================
# Full pipeline
# ==============================================================================

def process_clips(
    clips: List[Dict[str, Any]],
    case_id: str,
    role: str = "unknown",
    domain: str = "",
    whisper_model: str = "base",
    language: Optional[str] = "en",
    hf_token: Optional[str] = None,
    diarize: bool = True,
) -> Dict[str, Any]:
    """
    One or more recordings -> a single obs_only case dict (ready for
    resolve_entities()).

    Each clip is {"path", "start_time" (datetime), "location", "source"?}.
    Speakers are diarized and labelled per clip; the same person in two
    recordings gets two different labels, which ER links by voice print.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_whisper(whisper_model, device)
    pipeline = setup_diarization_pipeline(hf_token, device) if diarize else None
    case_start = min(c["start_time"] for c in clips)

    observations: List[Dict[str, Any]] = []
    used_sources: set = set()
    for n, clip in enumerate(clips, 1):
        audio_file = os.path.basename(clip["path"])
        # entity labels are prefixed with source, so it must be unique per clip
        source = base = clip.get("source") or Path(audio_file).stem
        k = 2
        while source in used_sources:
            source = f"{base}_{k}"
            k += 1
        used_sources.add(source)

        print(f"[{n}/{len(clips)}] {audio_file}")
        audio = preprocess_audio(clip["path"])
        whisper_segments = transcribe(audio, model, language)
        print(f"      transcribed {len(whisper_segments)} segment(s)")
        diar_segments, voice_prints = perform_diarization(audio, pipeline)
        merged = merge_transcripts_with_speakers(whisper_segments, diar_segments)
        observations += build_observations(
            merged,
            base_time=clip["start_time"],
            location=clip["location"],
            source=source,
            role=role,
            audio_file=audio_file,
            id_prefix=f"AUD{n}",
            case_start=case_start,
            voice_prints=voice_prints,
        )

    observations.sort(key=lambda o: o["time_offset"])
    return {
        "case_id": case_id,
        "domain": domain,
        "template": "",
        "fir": {},
        "observations": observations,
    }


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    """
    Manifest: a JSON list of {"path", "start_time", "location", "source"?}.
    Relative paths are resolved against the manifest's own directory.
    """
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    clips = []
    for i, e in enumerate(entries):
        for field in ("path", "start_time", "location"):
            if not str(e.get(field, "")).strip():
                raise ValueError(f"manifest[{i}]: missing '{field}'")
        clip_path = Path(e["path"])
        if not clip_path.is_absolute():
            clip_path = path.parent / clip_path
        clips.append({
            "path": str(clip_path),
            "start_time": parse_base_time(e["start_time"]),
            "location": e["location"],
            "source": e.get("source") or None,
        })
    return clips


def parse_args():
    p = argparse.ArgumentParser(description="ForenSynth Audio Field Agent")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Path to one audio file (.wav, .mp3, .m4a, .flac, .ogg)")
    group.add_argument("--manifest", help="JSON list of clips for one case: "
                       '[{"path", "start_time", "location", "source"}, ...]')
    p.add_argument("--case-base-time", type=parse_base_time,
                   help="(with --input) ISO 8601 wall-clock time at which the recording starts, e.g. 2024-01-15T10:00:00")
    p.add_argument("--location",
                   help="(with --input) Where the recording was made (audio alone cannot determine this)")
    p.add_argument("--source", default=None,
                   help="(with --input) Source label, e.g. phone_record (default: input file name)")
    p.add_argument("--case-id", default=None, help="Case ID (default: input/manifest file name)")
    p.add_argument("--output", default="./output/audio", help="Output directory")
    p.add_argument("--role", default="unknown", help="Role applied to every speaker (default: unknown)")
    p.add_argument("--domain", default="", help="Case domain, e.g. ATM_Robbery")
    p.add_argument("--whisper-model", default="base", help="tiny | base | small | medium | large")
    p.add_argument("--language", default="en", help="Spoken language code, or 'auto' to detect")
    p.add_argument("--hf-token", default=None, help="HuggingFace token for pyannote (default: $HF_TOKEN)")
    p.add_argument("--no-diarization", action="store_true", help="Skip speaker diarization")
    args = p.parse_args()
    if args.input and not (args.case_base_time and args.location):
        p.error("--input requires --case-base-time and --location")
    return args


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")
    except ImportError:
        pass
    args = parse_args()

    if args.manifest:
        try:
            clips = load_manifest(Path(args.manifest))
        except (OSError, ValueError, argparse.ArgumentTypeError) as exc:
            print(f"[ERROR] Bad manifest: {exc}"); sys.exit(1)
        if not clips:
            print("[ERROR] Manifest lists no clips."); sys.exit(1)
    else:
        clips = [{
            "path": args.input,
            "start_time": args.case_base_time,
            "location": args.location,
            "source": args.source,
        }]
    for clip in clips:
        if not Path(clip["path"]).is_file():
            print(f"[ERROR] File not found: {clip['path']}"); sys.exit(1)

    case_id = args.case_id or Path(args.manifest or args.input).stem
    case = process_clips(
        clips,
        case_id=case_id,
        role=args.role,
        domain=args.domain,
        whisper_model=args.whisper_model,
        language=None if args.language.lower() == "auto" else args.language,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        diarize=not args.no_diarization,
    )
    observations = case["observations"]

    issues = validate_observations(observations)
    for issue in issues:
        print(f"[WARNING] {issue}")
    if not observations:
        print("[WARNING] No speech was transcribed - the observations list is empty.")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_id}_obs_only.json"
    text = json.dumps(case, indent=2, ensure_ascii=False)
    # keep each 256-number voice print on one line so the file stays readable
    text = re.sub(r"\[\s*(-?\d[\d.eE+-]*(?:,\s*-?\d[\d.eE+-]*)+)\s*\]",
                  lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]", text)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    speakers = sorted({o["entity"] for o in observations})
    print(f"\nCase         : {case_id}")
    print(f"Clips        : {len(clips)}")
    print(f"Observations : {len(observations)}")
    print(f"Speakers     : {', '.join(speakers) if speakers else '-'}")
    print(f"Saved        : {out_path}")


if __name__ == "__main__":
    main()


# ==============================================================================
# OPTIONAL: Whisper fine-tuning (AMI + Switchboard) - DISABLED
# ==============================================================================
# Fine-tuning is supported but optional. The notebook cells below (9-17) are
# kept for reference and are never executed: the agent above always uses
# baseline Whisper. To use them, run the cells in a notebook with
# `datasets`, `transformers`, `evaluate` and `jiwer` installed, then load the
# resulting checkpoint in transcribe() in place of whisper.load_model().
#
# """---
# ## 📦 CELL 9: Load AMI Meeting Corpus
# **What this does**: Loads a subset of the AMI corpus for domain adaptation
#
# **Why AMI?**
# - Multi-speaker meeting scenarios (forensically relevant)
# - Natural conversational speech with overlaps
# - Time-aligned transcripts
# - Diverse acoustic conditions
# """
#
# def load_ami_corpus(subset_fraction: float = 0.01) -> Dict:
#     """
#     Load AMI Meeting Corpus from HuggingFace
#
#     AMI contains:
#     - ~100 hours of meeting recordings
#     - 4 speakers per meeting
#     - Scenario and non-scenario meetings
#     - Multiple microphones (headset, lapel, array)
#
#     Args:
#         subset_fraction: Fraction of data to load (for Colab efficiency)
#
#     Returns:
#         Dictionary with train/validation/test splits
#     """
#     print(f"📚 Loading AMI Meeting Corpus ({subset_fraction*100}% subset)...")
#
#     try:
#         # Load AMI corpus
#         # Note: Using 'ihm' (individual headset microphone) for best quality
#         ami_dataset = load_dataset(
#             "edinburghcstr/ami",
#             "ihm",
#             split="train",
#             trust_remote_code=True
#         )
#
#         # Take subset for efficiency
#         num_samples = int(len(ami_dataset) * subset_fraction)
#         ami_subset = ami_dataset.select(range(num_samples))
#
#         # Cast audio to 16kHz
#         ami_subset = ami_subset.cast_column("audio", Audio(sampling_rate=16000))
#
#         print(f"✅ AMI corpus loaded")
#         print(f"   Total samples: {len(ami_subset)}")
#         print(f"   Features: {ami_subset.features}")
#
#         # Show sample
#         sample = ami_subset[0]
#         print(f"\n   Sample:")
#         print(f"   Text: {sample['text'][:100]}...")
#         print(f"   Audio duration: {len(sample['audio']['array'])/16000:.2f}s")
#
#         return {'train': ami_subset}
#
#     except Exception as e:
#         print(f"⚠️ Could not load AMI: {e}")
#         print("   Continuing without AMI dataset")
#         return None
#
#
# # Load AMI corpus
# ami_data = load_ami_corpus(subset_fraction=CONFIG['ami_subset_size'])
#
# if ami_data:
#     print("💎 AMI Dataset Features:")
#     print(ami_data['train'].features)
#
# """---
# ## 📦 CELL 10: Load Switchboard Corpus
# **What this does**: Loads Switchboard telephone conversation corpus
#
# **Why Switchboard?**
# - Spontaneous telephone conversations (forensically relevant)
# - Informal speech patterns, disfluencies
# - Lower audio quality (band-limited telephone)
# - Complements AMI's meeting domain
#
# **Why NOT LibriSpeech?**
# - LibriSpeech is read audiobooks (formal, clean)
# - Forensic audio is conversational, spontaneous
# - Different acoustic characteristics
# """
#
# def load_switchboard_corpus(subset_fraction: float = 0.01) -> Dict:
#     """
#     Load Switchboard telephone conversation corpus
#
#     Switchboard contains:
#     - ~300 hours of telephone conversations
#     - 2-speaker dialogues
#     - Spontaneous, informal speech
#     - 8kHz telephone bandwidth
#
#     Args:
#         subset_fraction: Fraction of data to load
#
#     Returns:
#         Dictionary with dataset split
#     """
#     print(f"📚 Loading Switchboard Corpus ({subset_fraction*100}% subset)...")
#
#     try:
#         # Note: Using a publicly available subset or alternative
#         # If full Switchboard not available, we'll create synthetic conversational data
#
#         # Attempt to load from HuggingFace
#         switchboard_dataset = load_dataset(
#             "switchboard_corpus",
#             split="train",
#             trust_remote_code=True
#         )
#
#         num_samples = int(len(switchboard_dataset) * subset_fraction)
#         switchboard_subset = switchboard_dataset.select(range(num_samples))
#
#         # Cast audio to 16kHz
#         switchboard_subset = switchboard_subset.cast_column("audio", Audio(sampling_rate=16000))
#
#         print(f"✅ Switchboard corpus loaded")
#         print(f"   Total samples: {len(switchboard_subset)}")
#
#         return {'train': switchboard_subset}
#
#     except Exception as e:
#         print(f"⚠️ Could not load Switchboard: {e}")
#         print("   Note: Switchboard may require special access or license")
#         print("   Continuing with AMI only")
#         return None
#
#
# # Load Switchboard corpus
# switchboard_data = load_switchboard_corpus(subset_fraction=CONFIG['switchboard_subset_size'])
#
# """---
# ## 📦 CELL 11: Prepare Dataset for Whisper Fine-tuning
# **What this does**: Converts loaded datasets into Whisper-compatible format
# """
#
# # Load Whisper processor for feature extraction
# print("🔧 Loading Whisper processor...")
# processor = WhisperProcessor.from_pretrained(f"openai/whisper-{CONFIG['whisper_model']}")
# print("✅ Processor loaded")
#
#
# def prepare_dataset(batch):
#     """
#     Prepare a batch for Whisper fine-tuning:
#     1. Extract audio features (mel spectrogram)
#     2. Tokenize transcripts
#     3. Prepare labels
#
#     Args:
#         batch: Batch from HuggingFace dataset
#
#     Returns:
#         Processed batch with input_features and labels
#     """
#     # Load and preprocess audio
#     audio = batch["audio"]
#
#     # Compute log-Mel spectrogram features
#     batch["input_features"] = processor.feature_extractor(
#         audio["array"],
#         sampling_rate=audio["sampling_rate"]
#     ).input_features[0]
#
#     # Tokenize transcripts
#     batch["labels"] = processor.tokenizer(batch["text"]).input_ids
#
#     return batch
#
#
# # Prepare AMI dataset if available
# if ami_data is not None:
#     print("🔄 Preparing AMI dataset for training...")
#     ami_prepared = ami_data['train'].map(
#         prepare_dataset,
#         remove_columns=ami_data['train'].column_names,
#         num_proc=1  # Single process for Colab stability
#     )
#     print(f"✅ AMI dataset prepared: {len(ami_prepared)} samples")
# else:
#     ami_prepared = None
#     print("⚠️ Skipping AMI preparation (not loaded)")
#
#
# # Prepare Switchboard if available
# if switchboard_data is not None:
#     print("🔄 Preparing Switchboard dataset for training...")
#     switchboard_prepared = switchboard_data['train'].map(
#         prepare_dataset,
#         remove_columns=switchboard_data['train'].column_names,
#         num_proc=1
#     )
#     print(f"✅ Switchboard dataset prepared: {len(switchboard_prepared)} samples")
# else:
#     switchboard_prepared = None
#     print("⚠️ Skipping Switchboard preparation (not loaded)")
#
# """---
# ## 📦 CELL 12: Fine-tune Whisper for Conversational Robustness
# **What this does**: Selectively fine-tunes Whisper on conversational data
#
# **Important Notes**:
# - We do NOT train from scratch
# - We fine-tune encoder + decoder for domain adaptation
# - Goal: Improve performance on conversational, multi-speaker audio
# - NOT training a new ASR system
# """
#
# # Data collator for Whisper
# from dataclasses import dataclass
# from typing import Any, Dict, List, Union
#
# @dataclass
# class DataCollatorSpeechSeq2SeqWithPadding:
#     """
#     Custom data collator for Whisper:
#     - Pads input features to same length
#     - Pads labels and replaces padding with -100 (ignored in loss)
#     """
#     processor: Any
#
#     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
#         # Split inputs and labels
#         input_features = [{"input_features": feature["input_features"]} for feature in features]
#         label_features = [{"input_ids": feature["labels"]} for feature in features]
#
#         # Pad inputs
#         batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
#
#         # Pad labels
#         labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
#
#         # Replace padding with -100 to ignore in loss
#         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
#
#         # Remove BOS token if present
#         if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
#             labels = labels[:, 1:]
#
#         batch["labels"] = labels
#
#         return batch
#
#
# # Create data collator
# data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
# print("✅ Data collator created")
#
# """---
# ## 📦 CELL 13: Define Training Metrics
# **What this does**: Sets up Word Error Rate (WER) metric for evaluation
# """
#
# # # Load WER metric
# # metric = evaluate.load("wer")
#
#
# # def compute_metrics(pred):
# #     """
# #     Compute Word Error Rate (WER) during training
#
# #     WER = (Substitutions + Deletions + Insertions) / Total Words
# #     Lower is better (0 = perfect)
#
# #     Args:
# #         pred: Predictions from model
#
# #     Returns:
# #         Dictionary with WER score
# #     """
# #     pred_ids = pred.predictions
# #     label_ids = pred.label_ids
#
# #     # Replace -100 with pad token id
# #     label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
#
# #     # Decode predictions and labels
# #     pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
# #     label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
#
# #     # Compute WER
# #     wer = 100 * metric.compute(predictions=pred_str, references=label_str)
#
# #     return {"wer": wer}
#
#
# # print("✅ Evaluation metrics defined (WER)")
# import jiwer
#
# def compute_metrics(pred):
#     pred_ids = pred.predictions
#     label_ids = pred.label_ids
#
#     # Replace -100 with pad token id
#     label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
#
#     # Decode predictions and labels
#     pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
#
#     # Compute WER directly using jiwer
#     wer = 100 * jiwer.wer(label_str, pred_str)
#
#     return {"wer": wer}
#
# print("✅ Evaluation metrics defined using direct jiwer call")
#
# """---
# ## 📦 CELL 14: Initialize Whisper Model for Fine-tuning
# **What this does**: Loads pretrained Whisper model and prepares it for training
# """
#
# # --- MODIFIED CELL 14 ---
# # Load pretrained Whisper model for fine-tuning
# print(f"📥 Loading Whisper model for fine-tuning...")
# model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{CONFIG['whisper_model']}")
#
# # Configure model for training
# # FIX: Force the decoder to use English and the transcription task
# # This prevents the 'Multiple languages detected' error during evaluation batches
# model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
# model.config.suppress_tokens = []
#
# # Enable gradient checkpointing to save memory
# model.config.use_cache = False
#
# print("✅ Model loaded and configured")
# print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
# print(f"   Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")
#
# """---
# ## 📦 CELL 15: Configure Training Arguments
# **What this does**: Sets up training hyperparameters and optimization strategy
# """
#
# # Updated Cell 15: Configure Training Arguments with Overfitting Prevention
# training_args = Seq2SeqTrainingArguments(
#     output_dir=CONFIG['model_save_dir'],
#
#     # Training schedule
#     num_train_epochs=CONFIG['num_epochs'],
#     per_device_train_batch_size=CONFIG['batch_size'],
#     gradient_accumulation_steps=2,  # Effective batch size = 8 * 2 = 16
#
#     # Optimization & Overfitting Prevention
#     learning_rate=CONFIG['learning_rate'],
#     warmup_steps=100,               # Increased warmup for stability
#     weight_decay=0.05,              # Penalizes complex models to prevent memorization
#
#     # Evaluation Strategy
#     evaluation_strategy="epoch",    # Changed from "no" to track validation progress
#     save_strategy="epoch",          # Must match evaluation_strategy
#     load_best_model_at_end=True,    # Automatically reverts to the best version of the model
#     metric_for_best_model="wer",    # Use Word Error Rate to determine "best"
#     greater_is_better=False,        # Lower WER is better
#
#     # Logging
#     logging_steps=25,
#     report_to=["none"],
#
#     # Saving
#     save_total_limit=2,             # Keeps only the 2 best/most recent checkpoints
#
#     # Generation (for evaluation)
#     predict_with_generate=True,
#     generation_max_length=225,
#
#     generation_config=model.generation_config,
#     # Performance
#     fp16=torch.cuda.is_available(),
#     dataloader_num_workers=2,
#
#     # Reproducibility
#     seed=42,
# )
#
# print("✅ Training arguments configured")
# print(f"   Epochs: {training_args.num_train_epochs}")
# print(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
# print(f"   Learning rate: {training_args.learning_rate}")
# print(f"   FP16: {training_args.fp16}")
#
# # Manually creating the split without rerunning Cell 9 or 11
# if ami_prepared is not None:
#     # Split the already 'prepared' (vectorized) data
#     ami_split = ami_prepared.train_test_split(test_size=0.2, seed=42)
#
#     # Define the variables the Trainer needs
#     ami_prepared = ami_split["train"]
#     eval_dataset = ami_split["test"]
#
#     print(f"✅ Injection Successful:")
#     print(f"   Training samples: {len(ami_prepared)}")
#     print(f"   Evaluation samples: {len(eval_dataset)}")
# else:
#     print("❌ Error: 'ami_prepared' not found. You must run Cell 11 first.")
#
# """---
# ## 📦 CELL 16: Run Fine-tuning (OPTIONAL - Comment out if datasets unavailable)
# **What this does**: Executes the fine-tuning process on conversational data
#
# **Note**: This cell can be skipped if datasets are not available. The baseline Whisper will still work.
# """
#
# # Only run if we have prepared data
# if ami_prepared is not None:
#     print("🚀 Starting fine-tuning...")
#     print("   This may take 15-30 minutes on GPU, longer on CPU")
#     print("   Feel free to skip this cell and use baseline Whisper")
#
#     # Initialize trainer
#     trainer = Seq2SeqTrainer(
#         args=training_args,
#         model=model,
#         train_dataset=ami_prepared,
#         eval_dataset=eval_dataset,
#         data_collator=data_collator,
#         compute_metrics=compute_metrics,
#         tokenizer=processor.feature_extractor,
#     )
#
#     # Train
#     trainer.train()
#
#     # Save final model
#     trainer.save_model(CONFIG['model_save_dir'])
#     processor.save_pretrained(CONFIG['model_save_dir'])
#
#     print("✅ Fine-tuning complete!")
#     print(f"   Model saved to: {CONFIG['model_save_dir']}")
#
#     # Mark that we have a fine-tuned model
#     FINETUNED_MODEL_AVAILABLE = True
#
# else:
#     print("⚠️ No training data available - skipping fine-tuning")
#     print("   Will use baseline Whisper for all downstream tasks")
#     FINETUNED_MODEL_AVAILABLE = False
#
# """---
# ## 📦 CELL 17: Load Fine-tuned Model (if available)
# **What this does**: Loads the fine-tuned Whisper model for inference
# """
#
# # Load fine-tuned model if available, otherwise use baseline
# if FINETUNED_MODEL_AVAILABLE and os.path.exists(CONFIG['model_save_dir']):
#     print("📥 Loading fine-tuned Whisper model...")
#     finetuned_whisper = whisper.load_model(CONFIG['model_save_dir'])
#     print("✅ Fine-tuned model loaded")
#     ACTIVE_WHISPER = finetuned_whisper
#     MODEL_VERSION = "whisper_finetuned_v1"
# else:
#     print("📥 Using baseline Whisper model")
#     ACTIVE_WHISPER = baseline_whisper
#     MODEL_VERSION = "whisper_baseline"
#
# print(f"   Active model: {MODEL_VERSION}")
