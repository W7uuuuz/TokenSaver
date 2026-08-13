"""
Stage 1: Audio extraction + Whisper ASR with word-level timestamps.
"""
import subprocess
import os
import json
import warnings
from pathlib import Path
import numpy as np

from . import config


def extract_audio(video_path: str, output_dir: str = None) -> str:
    """Extract 16kHz mono WAV from video via ffmpeg."""
    if output_dir is None:
        output_dir = config.AUDIO_TEMP_DIR
    os.makedirs(output_dir, exist_ok=True)

    video_stem = Path(video_path).stem
    wav_path = os.path.join(output_dir, f"{video_stem}.wav")

    if os.path.exists(wav_path):
        return wav_path

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",                    # no video
        "-acodec", "pcm_s16le",   # 16-bit PCM
        "-ar", str(config.AUDIO_SAMPLE_RATE),
        "-ac", "1",               # mono
        "-loglevel", "error",
        wav_path,
    ]
    subprocess.run(cmd, check=True)
    return wav_path


def transcribe(wav_path: str, model_name: str = None) -> dict:
    """Run Whisper ASR and return segments with word-level timestamps."""
    import whisper

    if model_name is None:
        model_name = config.WHISPER_MODEL

    model = whisper.load_model(model_name, device=config.WHISPER_DEVICE)

    result = model.transcribe(
        wav_path,
        language=config.WHISPER_LANGUAGE,
        word_timestamps=True,
        verbose=False,
    )
    return result


def result_to_timeseries(result: dict) -> list[dict]:
    """
    Convert Whisper result to per-second text density input.
    Returns list of {second: int, text: str, words: list} for each second.
    """
    segments = result.get("segments", [])
    if not segments:
        return []

    # Determine video duration from last segment
    last_end = segments[-1].get("end", 0)
    total_seconds = int(np.ceil(last_end))

    # Initialize per-second bins
    timeseries = [
        {"second": s, "text": "", "words": []}
        for s in range(total_seconds + 1)
    ]

    for seg in segments:
        seg_text = seg.get("text", "").strip()
        seg_start = seg.get("start", 0)
        words = seg.get("words", [])

        for w in words if words else []:
            w_start = w.get("start", seg_start)
            sec = int(w_start)
            if 0 <= sec < len(timeseries):
                timeseries[sec]["words"].append({
                    "word": w.get("word", "").strip(),
                    "start": w.get("start", 0),
                    "end": w.get("end", 0),
                    "probability": w.get("probability", 1.0),
                })

        # Distribute segment text across covered seconds
        seg_end = seg.get("end", seg_start + 1)
        covered = set()
        for t in range(int(seg_start), min(int(seg_end) + 1, len(timeseries))):
            covered.add(t)
        for s in sorted(covered):
            if 0 <= s < len(timeseries):
                timeseries[s]["text"] += " " + seg_text

    return timeseries


def run_asr(video_path: str, cache_dir: str = None) -> dict:
    """Full ASR pipeline: extract → transcribe → timeseries. Returns ASR result dict."""
    if cache_dir is None:
        cache_dir = os.path.join(config.OUTPUT_DIR, "asr_cache")

    wav_path = extract_audio(video_path, os.path.join(cache_dir, "audio"))

    video_stem = Path(video_path).stem
    result_path = os.path.join(cache_dir, f"{video_stem}_whisper_result.json")

    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)
    else:
        os.makedirs(cache_dir, exist_ok=True)
        result = transcribe(wav_path)
        # Serialize without numpy arrays
        serializable = {
            "text": result.get("text", ""),
            "language": result.get("language", ""),
            "segments": result.get("segments", []),
        }
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    ts = result_to_timeseries(result)
    return {
        "video_path": video_path,
        "full_text": result.get("text", ""),
        "segments": result.get("segments", []),
        "timeseries": ts,
        "total_seconds": len(ts),
    }
