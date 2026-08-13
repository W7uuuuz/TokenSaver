#!/usr/bin/env python3
"""
Token-Saver Round 1 Pilot — CLI Entry Point.

Usage:
    # Single video with automatic ASR (Whisper)
    python -m pilot_round1.run --video path/to/video.mp4

    # Single video with pre-extracted ASR text
    python -m pilot_round1.run --video video.mp4 --script "口播文本内容..."

    # Video with no audio track (pure visual mode)
    python -m pilot_round1.run --video video.mp4 --no-audio

    # Directory (all videos matching config.VIDEO_EXTENSIONS)
    python -m pilot_round1.run --dir path/to/videos/

    # With custom budget levels
    python -m pilot_round1.run --video video.mp4 --budgets 5 15 30 50

    # Skip visual features (text-density-only mode)
    python -m pilot_round1.run --video video.mp4 --skip-visual

Output:
    pilot_output/<video_name>/
        pilot_<video_name>_dashboard.png    # Main density dashboard
        pilot_<video_name>_budget_summary.png  # Budget comparison
        results.json                           # Machine-readable results
"""
import argparse
import os
import sys
from pathlib import Path

from . import config
from .pipeline import run_pipeline


def find_videos(directory: str) -> list:
    """Find all video files in directory matching configured extensions."""
    videos = []
    for root, _, files in os.walk(directory):
        for f in files:
            if any(f.lower().endswith(ext) for ext in config.VIDEO_EXTENSIONS):
                videos.append(os.path.join(root, f))
    return sorted(videos)


def check_dependencies():
    """Check that required packages are installed. Print guidance if missing."""
    missing = []

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")

    try:
        import jieba
    except ImportError:
        missing.append("jieba")

    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")

    # Optional: whisper (only needed if not using --script or --no-audio)
    try:
        import whisper
    except ImportError:
        pass  # optional — user may provide script_text or use --no-audio

    # Optional: dashscope (only needed for Qwen3 ASR)
    try:
        import dashscope
    except ImportError:
        pass  # optional

    # Optional: openai (only needed for DeepSeek text cleaning)
    try:
        import openai
    except ImportError:
        pass  # optional

    if missing:
        print("Missing required packages:")
        for m in missing:
            print(f"  - {m}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        print("\nAlso need ffmpeg on PATH: https://ffmpeg.org/download.html")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Token-Saver Round 1 Pilot — Density Curve Visualization & Frame Selection"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="Path to a single video file")
    group.add_argument("--dir", type=str, help="Path to a directory of videos")

    parser.add_argument("--output", type=str, default=config.OUTPUT_DIR,
                        help=f"Output directory (default: {config.OUTPUT_DIR})")
    parser.add_argument("--budgets", type=int, nargs="+", default=[3, 10, 20, 30, 50],
                        help="Token budget levels to test")
    parser.add_argument("--skip-visual", action="store_true",
                        help="Skip visual feature extraction (text-density-only mode)")
    parser.add_argument("--script", type=str, default=None,
                        help="Pre-extracted ASR text — bypasses ASR stage")
    parser.add_argument("--no-audio", action="store_true",
                        help="Video has no audio track — pure visual driving mode")

    args = parser.parse_args()

    if not check_dependencies():
        sys.exit(1)

    mode_parts = []
    if args.no_audio:
        mode_parts.append("NO-AUDIO (pure visual)")
    elif args.script:
        mode_parts.append("SCRIPT (pre-extracted text)")
    else:
        mode_parts.append("ASR (auto)")
    if args.skip_visual:
        mode_parts.append("VISUAL=OFF")

    print("=" * 60)
    print(f"  Token-Saver Round 1 Pilot")
    print(f"  Mode: {' | '.join(mode_parts)}")
    print(f"  Budgets: {args.budgets}")
    print("=" * 60)

    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: video not found: {args.video}")
            sys.exit(1)
        videos = [args.video]
        scripts = {Path(args.video).stem: args.script} if args.script else {}
    else:
        videos = find_videos(args.dir)
        if not videos:
            print(f"Error: no video files found in {args.dir}")
            print(f"  Supported extensions: {config.VIDEO_EXTENSIONS}")
            sys.exit(1)
        print(f"Found {len(videos)} video(s) in {args.dir}")
        scripts = {}  # --script not supported with --dir

    for i, video_path in enumerate(videos):
        video_id = Path(video_path).stem
        print(f"\n{'#'*60}")
        print(f"  Video {i+1}/{len(videos)}: {Path(video_path).name}")
        print(f"{'#'*60}")
        try:
            run_pipeline(
                video_path=video_path,
                output_dir=args.output,
                budgets=args.budgets,
                skip_visual=args.skip_visual,
                script_text=scripts.get(video_id),
                no_audio=args.no_audio,
            )
        except Exception as e:
            print(f"\n  ERROR processing {video_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nAll done. Output saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
