#!/usr/bin/env python3
"""
Token-Saver Round 1 — Full Orchestrator (v2: Qwen3 ASR + timestamps for ALL videos).
1. Qwen3 ASR Flash on all audio videos → text + word timestamps
2. DeepSeek V4 Flash text cleaning
3. Pipeline with accurate timestamps + cleaned text
"""
import os
import sys
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env before any API imports
def _load_dotenv():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    if key not in os.environ:
                        os.environ[key] = val
_load_dotenv()

from pilot_round1 import config

ROUND1_DIR = PROJECT_ROOT / "round1"
VIDEOS_DIR = ROUND1_DIR / "videos"
SCRIPTS_DIR = ROUND1_DIR / "scripts"
MANIFEST_PATH = ROUND1_DIR / "manifest.json"
OUTPUT_DIR = PROJECT_ROOT / "pilot_output"
CACHE_DIR = OUTPUT_DIR / "qwen_cache"


def load_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_video_path(local_id: str) -> str:
    for root, dirs, files in os.walk(str(VIDEOS_DIR)):
        for f in files:
            if f.startswith(local_id) and f.endswith(".mp4"):
                return os.path.join(root, f)
    return None


def main():
    print("=" * 60)
    print("  Token-Saver Round 1 — Qwen3 ASR + Timestamps")
    print("=" * 60)

    manifest = load_manifest()
    materials = manifest["materials"]
    print(f"Loaded {len(materials)} videos from manifest\n")

    # ---- Step 1: Collect videos, check audio ----
    video_paths = {}
    categories = {}
    no_audio = set()

    for m in materials:
        vid = m["local_id"]
        cat = m["category"]
        categories[vid] = cat
        vpath = get_video_path(vid)
        if vpath is None:
            print(f"  [SKIP] {vid}: video file not found")
            continue
        video_paths[vid] = vpath

        has_audio = check_audio_track(vpath)
        if not has_audio:
            no_audio.add(vid)
            print(f"  [{vid}] ({cat}): NO audio track — pure visual mode")
        else:
            print(f"  [{vid}] ({cat}): audio OK — will run Qwen3 ASR")

    # ---- Step 2: Qwen3 ASR Flash on ALL audio videos ----
    print(f"\n[Step 2] Qwen3 ASR Flash on {len(video_paths) - len(no_audio)} audio video(s)...")
    from pilot_round1.asr_qwen import transcribe_video

    qwen_results = {}  # vid -> {text, words, sentences}
    for vid, vpath in sorted(video_paths.items()):
        if vid in no_audio:
            continue
        try:
            result = transcribe_video(
                vpath,
                cache_dir=str(CACHE_DIR / "asr"),
            )
            qwen_results[vid] = result
        except Exception as e:
            print(f"  [ERROR] Qwen3 ASR failed for {vid}: {e}")
            import traceback
            traceback.print_exc()

    # ---- Step 3: DeepSeek text cleaning ----
    print(f"\n[Step 3] DeepSeek V4 Flash text cleaning ({len(qwen_results)} texts)...")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
    cleaned_texts = {}

    if deepseek_key:
        from pilot_round1.text_cleaner import clean_texts_batch
        raw_texts = {vid: r["text"] for vid, r in qwen_results.items()}
        cleaned_texts = clean_texts_batch(
            raw_texts,
            api_key=deepseek_key,
            cache_dir=str(CACHE_DIR / "cleaned"),
        )
        for vid in cleaned_texts:
            if cleaned_texts[vid] != qwen_results[vid]["text"]:
                print(f"  [{vid}] Cleaned: {len(qwen_results[vid]['text'])} → {len(cleaned_texts[vid])} chars")
    else:
        print("  DEEPSEEK_API_KEY not set — using Qwen3 raw text (no cleaning)")
        cleaned_texts = {vid: r["text"] for vid, r in qwen_results.items()}

    # ---- Step 4: Run Token-Saver pipeline ----
    print(f"\n[Step 4] Running Token-Saver pipeline on {len(video_paths)} videos...")
    from pilot_round1.pipeline import run_pipeline

    budgets = [3, 10, 20, 30, 50]
    all_results = {}

    for i, (vid, vpath) in enumerate(sorted(video_paths.items())):
        print(f"\n{'#'*60}")
        print(f"  Video {i+1}/{len(video_paths)}: {vid} ({categories.get(vid, '?')})")
        print(f"{'#'*60}")

        is_no_audio = vid in no_audio
        qwen_words = qwen_results[vid].get("words", []) if vid in qwen_results else []
        cleaned_text = cleaned_texts.get(vid, "")

        try:
            result = run_pipeline(
                video_path=vpath,
                output_dir=str(OUTPUT_DIR),
                budgets=budgets,
                skip_visual=False,
                no_audio=is_no_audio,
                qwen_words=qwen_words if qwen_words else None,
                cleaned_text=cleaned_text if cleaned_text else None,
            )
            all_results[vid] = {
                "category": categories.get(vid, "?"),
                "no_audio": is_no_audio,
                "composite_mean": sum(result["composite_scores"]) / max(len(result["composite_scores"]), 1),
                "routing_distribution": result.get("routing_distribution", {}),
            }
        except Exception as e:
            print(f"\n  ERROR on {vid}: {e}")
            import traceback
            traceback.print_exc()

    # ---- Step 5: Summary ----
    print(f"\n{'='*60}")
    print(f"  Round 1 Complete — Summary")
    print(f"{'='*60}")
    for vid, info in sorted(all_results.items()):
        mode = "NO-AUDIO" if info["no_audio"] else "Qwen3+TS"
        print(f"  {vid:<20s} [{info['category']:<6s}] {mode:<14s} "
              f"mean_score={info['composite_mean']:.1f}")
    print(f"\nOutput: {OUTPUT_DIR.absolute()}")


def check_audio_track(video_path: str) -> bool:
    try:
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=10, encoding="utf-8", errors="replace",
            creationflags=creationflags,
        )
        return "audio" in result.stdout
    except Exception:
        return True


if __name__ == "__main__":
    main()
