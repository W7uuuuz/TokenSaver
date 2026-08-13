"""
Qwen3 ASR — audio-to-text via DashScope file transcription API.
Uses qwen3-asr-flash-filetrans model which returns character-level timestamps.
"""
import os
import subprocess
import json
import time
from pathlib import Path
from typing import Optional

import requests

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")


def extract_audio(video_path: str, output_dir: str = None) -> str:
    """Extract 16kHz mono WAV from video via ffmpeg. Returns WAV path."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(video_path), "audio")
    os.makedirs(output_dir, exist_ok=True)

    video_stem = Path(video_path).stem
    wav_path = os.path.join(output_dir, f"{video_stem}.wav")

    if os.path.exists(wav_path):
        return wav_path

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-loglevel", "error",
        wav_path,
    ]
    subprocess.run(cmd, check=True)
    return wav_path


def transcribe_qwen(
    wav_path: str,
    api_key: str = None,
) -> dict:
    """
    Transcribe audio via Qwen3-ASR-Flash-Filetrans (DashScope async API)
    WITH character-level word timestamps.

    Returns:
        {
            "text": str,
            "words": [{"word": str, "start": float, "end": float}, ...],
            "sentences": [{"text": str, "start": float, "end": float}, ...],
        }
    """
    key = api_key or DASHSCOPE_API_KEY
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY not set.")

    try:
        from dashscope import Files
        from dashscope.audio.qwen_asr import QwenTranscription
    except ImportError:
        raise ImportError("dashscope package not installed. Run: pip install dashscope")

    # Step 1: Upload WAV to DashScope
    upload_result = Files.upload(
        file_path=wav_path,
        purpose="transcription",
        api_key=key,
    )
    if upload_result.status_code != 200:
        raise RuntimeError(f"File upload failed: status={upload_result.status_code}, "
                           f"message={upload_result.message}")

    uploaded = upload_result.output["uploaded_files"][0]
    file_url = uploaded.get("url")
    if not file_url:
        fid = uploaded["file_id"]
        info = Files.get(fid, api_key=key)
        file_url = info.output.get("url")
    if not file_url:
        raise RuntimeError("Could not get file URL after upload")

    # Step 2: Submit async transcription
    response = QwenTranscription.async_call(
        model="qwen3-asr-flash-filetrans",
        file_url=file_url,
        api_key=key,
        enable_words=True,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Transcription submission failed: status={response.status_code}, "
                           f"message={response.message}")

    task_id = response.output["task_id"]

    # Step 3: Poll until complete
    for _ in range(60):  # max 2 minutes
        time.sleep(2)
        result = QwenTranscription.fetch(task_id, api_key=key)
        status = result.output.get("task_status")
        if status == "SUCCEEDED":
            transcription_url = result.output["result"]["transcription_url"]
            break
        elif status == "FAILED":
            raise RuntimeError(f"Transcription task failed: {result.output}")
    else:
        raise RuntimeError(f"Transcription timed out for task {task_id}")

    # Step 4: Download result JSON
    resp = requests.get(transcription_url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Step 5: Parse transcripts into our format
    transcripts = data.get("transcripts", [])
    words = []
    sentences = []
    full_text_parts = []

    for t in transcripts:
        for s in t.get("sentences", []):
            s_text = s.get("text", "")
            if s_text:
                sentences.append({
                    "text": s_text,
                    "start": s["begin_time"] / 1000.0,
                    "end": s["end_time"] / 1000.0,
                })
                full_text_parts.append(s_text)

            for w in s.get("words", []):
                words.append({
                    "word": w["text"],
                    "start": w["begin_time"] / 1000.0,
                    "end": w["end_time"] / 1000.0,
                })

    return {
        "text": "".join(full_text_parts),
        "words": words,
        "sentences": sentences,
    }


def transcribe_video(
    video_path: str,
    api_key: str = None,
    cache_dir: str = None,
) -> dict:
    """
    Full pipeline: extract audio -> transcribe with timestamps -> return result.

    Returns: {
        "video_id": str,
        "text": str,
        "words": [{word, start, end}, ...],
        "sentences": [{text, start, end}, ...],
    }
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(video_path), "asr_output")
    os.makedirs(cache_dir, exist_ok=True)

    video_id = Path(video_path).stem
    cache_path = os.path.join(cache_dir, f"{video_id}_qwen_asr.json")

    # Check cache
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        n_words = len(data.get("words", []))
        n_sentences = len(data.get("sentences", []))
        print(f"  [{video_id}] ASR loaded from cache ({n_words} words, {n_sentences} sentences)")
        return {"video_id": video_id, **data}

    # Extract audio
    print(f"  [{video_id}] Extracting audio...")
    wav_path = extract_audio(video_path, os.path.join(cache_dir, "wav"))

    # Transcribe via Qwen3 file transcription API
    print(f"  [{video_id}] Transcribing via Qwen3-ASR-Flash-Filetrans (with word timestamps)...")
    result = transcribe_qwen(wav_path, api_key)

    # Cache
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    n_words = len(result.get("words", []))
    n_sentences = len(result.get("sentences", []))
    print(f"  [{video_id}] ASR complete: {len(result['text'])} chars, "
          f"{n_words} words, {n_sentences} sentences")

    return {"video_id": video_id, **result}
