"""
Token-Saver Round 1 Pilot — Full Pipeline Orchestrator.
Wires Stage 1 (ASR + Visual Features) → Stage 2 (Scoring + Routing + Coupling) → Stage 3 (MMR).
"""
import math
import os
import json
import time as time_module
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from . import config
from . import asr_engine
from . import density_scorer
from . import frame_selector
from . import visualize


# ============================================================================
# Visual feature extraction
# ============================================================================
def _detect_fast_cut(video_path: str) -> bool:
    """Quick pre-scan: detect fast-cut editing using dual-condition metric.
    Condition 1: median frame-difference > ABS_THRESHOLD (high absolute motion)
    Condition 2: p90/median ratio > RATIO_THRESHOLD (spiky motion pattern)
    Both must be true. This separates fast-cut montages (high median + spikes)
    from talking-head (low median, consistent motion)."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return False

    prev_gray = None
    motions = []
    frame_idx = 0
    max_frames = int(fps * 10)  # first 10 seconds

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motions.append(float(np.mean(diff)))
        prev_gray = gray
        frame_idx += 1

    cap.release()
    if len(motions) < 2:
        return False

    arr = np.array(motions, dtype=float)
    median_val = float(np.median(arr))
    p90 = float(np.percentile(arr, 90))

    # Dual condition: high absolute motion AND spiky pattern
    is_fast = (median_val > config.FAST_CUT_ABS_THRESHOLD and
               p90 / max(median_val, 1e-10) > config.FAST_CUT_RATIO_THRESHOLD)
    if is_fast:
        print(f"  Fast-cut detected (median={median_val:.1f}, p90/median={p90/median_val:.1f})")
    return is_fast


def _minmax_to_0_10(values: list) -> list:
    """Min-max normalize a list to [0, 10] range."""
    a = np.array(values, dtype=float)
    a_min, a_max = a.min(), a.max()
    if a_max - a_min < 1e-10:
        return [5.0] * len(values)
    return ((a - a_min) / (a_max - a_min) * 10).tolist()


def extract_visual_features(video_path: str, total_seconds: int,
                            scene_threshold: float = None) -> Dict:
    """
    Extract visual features at 1fps:
    - Color histogram entropy (HSV)
    - Optical flow magnitude (sparse Lucas-Kanade)
    - Scene boundaries (via ffmpeg scene detection)
    - (Object detection skipped in pilot for speed)
    Returns dict with keys: entropy, motion, scene_boundaries, object_counts.
    """
    if scene_threshold is None:
        scene_threshold = config.SCENE_THRESHOLD
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_duration = total_frames / fps if fps > 0 else total_seconds

    # Limit to actual video duration
    n_seconds = min(int(np.ceil(actual_duration)), total_seconds + 5)

    hsv_entropy = [0.0] * n_seconds
    motion_magnitude = [0.0] * n_seconds
    object_counts = [0.0] * n_seconds
    hsv_histograms: List[Optional[np.ndarray]] = [None] * n_seconds

    prev_gray = None
    prev_frame_idx = -1

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        second = int(frame_idx / fps) if fps > 0 else frame_idx
        if second >= n_seconds:
            break

        # Only process 1 frame per second
        current_second_frame = int(frame_idx / fps) if fps > 0 else frame_idx
        if current_second_frame != prev_frame_idx and prev_frame_idx >= 0:
            # We already processed a frame for this second, skip
            pass

        # --- Color entropy (HSV) ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None,
                            list(config.HSV_BINS),
                            [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hsv_histograms[second] = hist
        # Shannon entropy
        nonzero = hist[hist > 0]
        hsv_entropy[second] = float(-np.sum(nonzero * np.log2(nonzero))) if len(nonzero) > 0 else 0.0
        # Normalize by max possible entropy
        max_entropy = math.log2(np.prod(config.HSV_BINS))
        if max_entropy > 0:
            hsv_entropy[second] = hsv_entropy[second] / max_entropy

        # --- Optical flow (sparse LK) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            # Grid sample points
            h, w = gray.shape
            grid_size = config.OPTICAL_FLOW_GRID
            y_step = max(1, h // grid_size)
            x_step = max(1, w // grid_size)
            p0 = []
            for y in range(y_step // 2, h, y_step):
                for x in range(x_step // 2, w, x_step):
                    p0.append([x, y])
            p0 = np.array(p0, dtype=np.float32).reshape(-1, 1, 2)

            p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
            if p1 is not None and status is not None:
                valid = status.flatten() == 1
                if valid.sum() > 0:
                    motion = np.linalg.norm(p1[valid] - p0[valid], axis=1)
                    motion_magnitude[second] = float(np.mean(motion))
                else:
                    motion_magnitude[second] = 0.0
            else:
                motion_magnitude[second] = 0.0
        else:
            motion_magnitude[second] = 0.0

        prev_gray = gray
        prev_frame_idx = current_second_frame
        frame_idx += 1

    cap.release()

    # Normalize features within video
    def _minmax(arr):
        a = np.array(arr, dtype=float)
        a_min, a_max = a.min(), a.max()
        if a_max - a_min < 1e-10:
            return np.ones_like(a) * 0.5
        return ((a - a_min) / (a_max - a_min)).tolist()

    hsv_entropy_norm = _minmax(hsv_entropy)
    motion_norm = _minmax(motion_magnitude)

    # --- Scene boundaries via ffmpeg ---
    scene_boundaries = _detect_scenes_ffmpeg(video_path, n_seconds, threshold=scene_threshold)

    return {
        "entropy_raw": hsv_entropy,
        "entropy_norm": hsv_entropy_norm,
        "motion_raw": motion_magnitude,
        "motion_norm": motion_norm,
        "object_counts": object_counts,  # all zeros in pilot
        "scene_boundaries": scene_boundaries,
        "hsv_histograms": hsv_histograms,
        "n_seconds": n_seconds,
    }


def _detect_scenes_ffmpeg(video_path: str, total_seconds: int, threshold: float = None) -> List[int]:
    """Detect scene changes using ffmpeg's scene detection filter.
    Falls back to OpenCV histogram-difference method if ffmpeg detects nothing."""
    import subprocess
    if threshold is None:
        threshold = config.SCENE_THRESHOLD
    boundaries = []
    try:
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"select='gt(scene\\,{threshold})',showinfo",
            "-f", "null", "-loglevel", "info", "-"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                                encoding="utf-8", errors="replace")
        for line in result.stderr.split("\n"):
            if "pts_time:" in line:
                parts = line.split("pts_time:")
                if len(parts) > 1:
                    time_str = parts[1].split()[0]
                    try:
                        t = int(float(time_str))
                        if t < total_seconds:
                            boundaries.append(t)
                    except ValueError:
                        pass

        boundaries = sorted(set(boundaries))
    except Exception:
        pass

    # Fallback: if ffmpeg found nothing, use OpenCV histogram-difference method
    if not boundaries:
        boundaries = _detect_scenes_opencv(video_path, total_seconds,
                                           sigma_multiplier=config.SCENE_OPENCV_SIGMA)

    # Merge close boundaries
    merged = []
    for b in boundaries:
        if not merged or b - merged[-1] >= config.MIN_SCENE_SEGMENT_SEC:
            merged.append(b)
    return merged


def _detect_scenes_opencv(video_path: str, total_seconds: int,
                          sigma_multiplier: float = None) -> List[int]:
    if sigma_multiplier is None:
        sigma_multiplier = config.SCENE_OPENCV_SIGMA
    """Detect scene changes using HSV histogram differences between consecutive seconds.
    A scene boundary is declared when the histogram difference exceeds
    mean + sigma_multiplier * std. Default sigma_multiplier=2.0."""
    import cv2
    boundaries = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return boundaries

    prev_hist = None
    diffs = []
    frame_pairs = []  # (second, diff)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        second = int(frame_idx / fps)
        if second >= total_seconds:
            break

        # Sample 1 frame per second (first frame of each second)
        if frame_idx % int(fps) == 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [8, 12],
                                [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            if prev_hist is not None:
                # Correlation comparison: 1=identical, -1=inverse
                corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                diff = 1.0 - corr  # 0=identical, 2=maximally different
                diffs.append(diff)
                frame_pairs.append((second, diff))

            prev_hist = hist

        frame_idx += 1

    cap.release()

    if len(diffs) < 2:
        return boundaries

    diffs_arr = np.array(diffs, dtype=float)
    mean_diff = float(np.mean(diffs_arr))
    std_diff = float(np.std(diffs_arr))
    threshold = mean_diff + sigma_multiplier * std_diff

    for second, diff in frame_pairs:
        if diff > threshold and diff > 0.05:  # minimum 5% diff floor
            boundaries.append(second)

    return boundaries


# ============================================================================
# Text window construction from ASR timeseries
# ============================================================================
def build_text_windows(
    asr_result: dict,
    total_seconds: int,
) -> tuple:
    """
    Build 5-second sliding windows from ASR timeseries.
    Returns (window_texts, window_tokens) aligned to each second.
    """
    ts = asr_result.get("timeseries", [])
    window_size = int(config.WINDOW_SIZE_SEC)
    half_window = window_size // 2

    jieba = density_scorer._ensure_jieba()
    window_texts = []
    window_tokens = []

    for sec in range(total_seconds):
        start = max(0, sec - half_window)
        end = min(total_seconds, sec + half_window + 1)

        # Collect text from timeseries
        text_parts = []
        for t in range(start, end):
            if t < len(ts):
                text_parts.append(ts[t].get("text", ""))

        window_text = " ".join(text_parts).strip()
        tokens = list(jieba.cut(window_text)) if window_text else []
        # Filter whitespace-only tokens
        tokens = [t.strip() for t in tokens if t.strip()]

        window_texts.append(window_text)
        window_tokens.append(tokens)

    return window_texts, window_tokens


def build_text_per_second(asr_result: dict, total_seconds: int) -> List[str]:
    """Build per-second text for coupling detection."""
    ts = asr_result.get("timeseries", [])
    return [ts[i].get("text", "") if i < len(ts) else "" for i in range(total_seconds)]


# ============================================================================
# Visual variance for coupling detection
# ============================================================================
def compute_visual_variance(visual_entropy: List[float]) -> List[float]:
    """σ_vis(t) = rolling std of visual entropy."""
    window = config.SCD_SIGMA_WINDOW_SEC
    n = len(visual_entropy)
    result = [0.0] * n
    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)
        vals = visual_entropy[start:end]
        result[i] = float(np.std(vals)) if vals else 0.0
    # Normalize
    a = np.array(result, dtype=float)
    a_min, a_max = a.min(), a.max()
    if a_max - a_min > 1e-10:
        result = ((a - a_min) / (a_max - a_min)).tolist()
    return result


# ============================================================================
# Budget summary metrics
# ============================================================================
def compute_budget_metrics(
    selected_frames: List[int],
    candidates: List[frame_selector.CandidateFrame],
    coupling_flags: List[int],
    scene_boundaries: List[int],
    total_seconds: int,
) -> dict:
    """Compute summary metrics for a given budget selection."""
    n_sel = len(selected_frames)
    selected_set = set(selected_frames)

    # Average importance of selected frames
    avg_imp = np.mean([c.importance_total for c in candidates if c.second in selected_set]) if selected_set else 0.0

    # What fraction of coupling seconds are covered?
    coupling_seconds = [i for i, f in enumerate(coupling_flags) if f == 1]
    coupling_covered = len([s for s in coupling_seconds if s in selected_set]) / max(len(coupling_seconds), 1)

    # What fraction of scene boundaries are covered?
    scene_covered = len([s for s in scene_boundaries if s in selected_set]) / max(len(scene_boundaries), 1)

    return {
        "n_frames": n_sel,
        "avg_importance": round(avg_imp, 4),
        "coupling_coverage": round(coupling_covered * 100, 1),
        "scene_coverage": round(scene_covered * 100, 1),
    }


# ============================================================================
# Main pipeline
# ============================================================================
def _make_empty_asr(total_seconds: int) -> dict:
    """Create empty ASR result for videos with no audio."""
    return {
        "video_path": "",
        "full_text": "",
        "segments": [],
        "timeseries": [{"second": s, "text": "", "words": []} for s in range(total_seconds)],
        "total_seconds": total_seconds,
    }


def _build_asr_from_text(full_text: str, total_seconds: int) -> dict:
    """
    Build ASR-compatible result dict from a pre-extracted text string
    WITHOUT word timestamps. Distributes text evenly across seconds.

    Prefer _build_asr_with_timestamps() when audio is available.
    """
    timeseries = []
    chars_per_sec = max(1, len(full_text) / max(total_seconds, 1))

    for sec in range(total_seconds):
        sec_chars_start = int(sec * chars_per_sec)
        sec_chars_end = int((sec + 1) * chars_per_sec)
        sec_text = full_text[sec_chars_start:sec_chars_end]
        timeseries.append({
            "second": sec,
            "text": sec_text,
            "words": [],
        })

    return {
        "video_path": "",
        "full_text": full_text,
        "segments": [],
        "timeseries": timeseries,
        "total_seconds": total_seconds,
    }


def _get_qwen_timestamps(video_path: str, cache_dir: str) -> dict:
    """
    Run Qwen3 ASR Flash on video audio to get word-level timestamps.
    Returns dict with {text, words: [{word, start, end}], sentences}.
    Cached to disk.
    """
    from . import asr_qwen

    result = asr_qwen.transcribe_video(
        video_path,
        cache_dir=os.path.join(cache_dir, "qwen_asr"),
    )
    return {
        "text": result.get("text", ""),
        "words": result.get("words", []),
        "sentences": result.get("sentences", []),
    }


def _build_asr_from_words(
    words: list,
    total_seconds: int,
    display_text: str = None,
) -> dict:
    """
    Build ASR result from pre-computed word timestamps (e.g. Qwen3 output).
    Uses provided word→time mapping for accurate per-second text distribution.
    Optionally uses display_text (cleaned) for `full_text` while keeping
    timestamp-aligned text in the timeseries.
    """
    if not words:
        return _make_empty_asr(total_seconds)

    timeseries = []
    for sec in range(total_seconds):
        sec_words = [w for w in words if int(w["start"]) == sec]
        sec_text = "".join(w["word"] for w in sec_words)
        timeseries.append({
            "second": sec,
            "text": sec_text,
            "words": sec_words,
        })

    full_text = display_text if display_text else "".join(w["word"] for w in words)

    return {
        "video_path": "",
        "full_text": full_text,
        "segments": [],
        "timeseries": timeseries,
        "total_seconds": total_seconds,
    }


def _build_asr_with_timestamps(
    video_path: str,
    total_seconds: int,
    cache_dir: str,
) -> dict:
    """
    Build ASR result using Qwen3 ASR Flash word-level timestamps.
    Gives accurate per-second text distribution for density scoring.
    """
    qwen = _get_qwen_timestamps(video_path, cache_dir)
    words = qwen["words"]

    if not words:
        # Fallback: even distribution if Qwen3 returned no timestamps
        print("  [WARN] Qwen3 returned no word timestamps — falling back to even distribution")
        return _build_asr_from_text(qwen["text"], total_seconds)

    # Build per-second bins from word timestamps
    timeseries = []
    for sec in range(total_seconds):
        sec_words = [w for w in words if int(w["start"]) == sec]
        sec_text = "".join(w["word"] for w in sec_words)
        timeseries.append({
            "second": sec,
            "text": sec_text,
            "words": sec_words,
        })

    return {
        "video_path": video_path,
        "full_text": qwen["text"],
        "segments": [],
        "timeseries": timeseries,
        "total_seconds": total_seconds,
    }


def run_pipeline(
    video_path: str,
    output_dir: str = None,
    budgets: List[int] = None,
    skip_visual: bool = False,
    script_text: str = None,
    no_audio: bool = False,
    qwen_words: list = None,
    cleaned_text: str = None,
) -> dict:
    """
    Run full Token-Saver pipeline on a single video.

    Args:
        video_path: Path to video file
        output_dir: Output directory for generated files
        budgets: Token budget levels to test
        skip_visual: Skip visual feature extraction
        script_text: (deprecated) Pre-extracted text without timestamps
        no_audio: Video has no audio track — skips ASR, uses empty text
        qwen_words: Pre-computed Qwen3 word timestamps [{word, start, end}]
        cleaned_text: DeepSeek-cleaned text for density scoring (used with qwen_words)
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    if budgets is None:
        budgets = [3, 10, 20, 30, 50]

    os.makedirs(output_dir, exist_ok=True)
    video_name = Path(video_path).stem
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Token-Saver Pipeline: {video_name}")
    print(f"{'='*60}")

    # ---- Stage 1a: ASR / Text Input ----
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_seconds = int(n_frames / fps) if fps > 0 else 60
    cap.release()

    if no_audio or (script_text is not None and script_text == ""):
        print("[Stage 1a] No audio — pure visual mode")
        asr_result = _make_empty_asr(total_seconds)
        is_visual_only = True
        print(f"  Duration (from video): {total_seconds}s")
    elif qwen_words and len(qwen_words) > 0:
        # Pre-computed Qwen3 timestamps provided — no API call needed
        print("[Stage 1a] Using pre-computed Qwen3 word timestamps...")
        display_text = cleaned_text if cleaned_text else "".join(w["word"] for w in qwen_words)
        asr_result = _build_asr_from_words(qwen_words, total_seconds, display_text)
        if cleaned_text:
            asr_result["script_text"] = cleaned_text
        is_visual_only = False
        n_ts_words = sum(len(ts.get("words", [])) for ts in asr_result.get("timeseries", []))
        print(f"  Words: {len(qwen_words)} timestamps, {n_ts_words} assigned to seconds")
        print(f"  Text: {len(display_text)} chars")
    else:
        # Fallback: run Qwen3 ASR Flash internally
        print("[Stage 1a] Running Qwen3 ASR Flash (with word timestamps)...")
        t0 = time_module.time()
        asr_cache = os.path.join(output_dir, "asr_cache")
        asr_result = _build_asr_with_timestamps(video_path, total_seconds, asr_cache)
        if script_text:
            asr_result["script_text"] = script_text
        is_visual_only = False
        n_words = sum(len(ts.get("words", [])) for ts in asr_result.get("timeseries", []))
        print(f"  Done in {time_module.time() - t0:.1f}s. Duration: {total_seconds}s")
        print(f"  Qwen3 transcript: {len(asr_result['full_text'])} chars, {n_words} word timestamps")
        if script_text:
            print(f"  Script text preserved: {len(script_text)} chars")

    # ---- Stage 1b: Visual Features ----
    if not skip_visual:
        print("[Stage 1b] Extracting visual features...")
        t0 = time_module.time()
        # Detect fast-cut style for adaptive scene threshold
        is_fast_cut = _detect_fast_cut(video_path)
        if is_fast_cut:
            print(f"  Fast-cut video detected — scene threshold lowered to {config.SCENE_THRESHOLD_FAST_CUT}")
        visual = extract_visual_features(
            video_path, total_seconds,
            scene_threshold=config.SCENE_THRESHOLD_FAST_CUT if is_fast_cut else config.SCENE_THRESHOLD,
        )
        total_seconds = visual["n_seconds"]  # Use actual video duration
        print(f"  Done in {time_module.time() - t0:.1f}s.")
        print(f"  Scene boundaries detected: {len(visual['scene_boundaries'])}")
    else:
        print("[Stage 1b] Skipped (--skip-visual)")
        is_fast_cut = False
        visual = {
            "entropy_raw": [0.0] * total_seconds,
            "entropy_norm": [0.0] * total_seconds,
            "motion_raw": [0.0] * total_seconds,
            "motion_norm": [0.0] * total_seconds,
            "object_counts": [0.0] * total_seconds,
            "scene_boundaries": [],
            "hsv_histograms": [None] * total_seconds,
            "n_seconds": total_seconds,
        }

    # ---- Stage 2: Scoring —— visual-only vs text-density path ----
    if is_visual_only:
        print("[Stage 2] Visual-only scoring (no audio)...")
        # Use visual entropy + motion as composite score (0-10 scale)
        raw_visual = [
            config.VISUAL_ONLY_WEIGHT_ENTROPY * visual["entropy_norm"][t]
            + config.VISUAL_ONLY_WEIGHT_MOTION * visual["motion_norm"][t]
            for t in range(total_seconds)
        ]
        composite_scores = _minmax_to_0_10(raw_visual)
        dim_scores = {"composite": composite_scores}
        routing_decisions = density_scorer.apply_global_local_routing(
            composite_scores, total_seconds, visual["scene_boundaries"]
        )
        coupling_flags = [0] * total_seconds
        lcd_flags = [0] * total_seconds
        scd_flags = [0] * total_seconds
        text_per_second = [""] * total_seconds
        print(f"  Mean visual composite score: {np.mean(composite_scores):.1f}/10")
    else:
        print("[Stage 2] Building text windows and scoring...")
        window_texts, window_tokens = build_text_windows(asr_result, total_seconds)
        text_per_second = build_text_per_second(asr_result, total_seconds)

        # ---- Stage 2a: 5-dim scoring ----
        dim_scores = density_scorer.compute_five_dim_scores(window_texts, window_tokens)
        composite_scores = dim_scores["composite"]
        print(f"  Mean composite score: {np.mean(composite_scores):.1f}/10")

        # ---- Stage 2b: Routing ----
        routing_decisions = density_scorer.apply_global_local_routing(
            composite_scores, total_seconds, visual["scene_boundaries"]
        )

        # ---- Stage 2c: Coupling Detection ----
        visual_variance = compute_visual_variance(visual["entropy_norm"])
        text_density_norm = [s / 10.0 for s in composite_scores]

        lcd_flags = density_scorer.detect_lexical_coupling(text_per_second, visual_variance)
        scd_flags = density_scorer.detect_structural_coupling(text_density_norm, visual_variance)
        coupling_flags = density_scorer.combine_coupling_flags(lcd_flags, scd_flags)
        print(f"  Coupling events: LCD={sum(lcd_flags)}s, SCD={sum(scd_flags)}s, Combined={sum(coupling_flags)}s")

    # ---- Stage 3: MMR Frame Selection ----
    print("[Stage 3] MMR frame selection...")
    candidates = frame_selector.build_candidates(
        total_seconds,
        coupling_flags,
        composite_scores,
        visual["entropy_norm"],
        visual["motion_norm"],
        visual["object_counts"],
        visual["scene_boundaries"],
    )

    # Adaptive λ: when coupling coverage is very low (<5% of video),
    # the pipeline has no reliable text-visual alignment signal.
    # Force diversity (lower λ) to avoid over-concentrating on noise.
    coupling_ratio = sum(coupling_flags) / max(total_seconds, 1)
    if coupling_ratio < 0.05:
        adaptive_lambda = 0.4
        print(f"  Low coupling ({coupling_ratio:.1%}) — adaptive λ={adaptive_lambda}")
    else:
        adaptive_lambda = None  # use config default
        print(f"  Coupling ratio: {coupling_ratio:.1%}")

    selected_frames = frame_selector.select_frames_multi_budget(
        candidates, budgets, visual.get("hsv_histograms"),
        lambda_rel=adaptive_lambda,
    )

    # Budget summary metrics
    budget_results = {}
    for k, frames in selected_frames.items():
        budget_results[k] = compute_budget_metrics(
            frames, candidates, coupling_flags, visual["scene_boundaries"], total_seconds
        )
        print(f"  B={k}: {len(frames)} frames selected")

    # ---- Time axis ----
    time_axis = list(range(total_seconds))

    # ---- Visualize ----
    print("[Visualize] Generating dashboard...")
    dashboard_path = visualize.plot_density_dashboard(
        video_name=video_name,
        total_seconds=total_seconds,
        time_axis=time_axis,
        composite_scores=composite_scores,
        routing_decisions=routing_decisions,
        coupling_flags=coupling_flags,
        dim_scores=dim_scores,
        visual_entropy=visual["entropy_norm"],
        visual_motion=visual["motion_norm"],
        coupling_lcd=lcd_flags,
        coupling_scd=scd_flags,
        selected_frames=selected_frames,
        scene_boundaries=visual["scene_boundaries"],
        output_dir=video_output_dir,
        output_prefix="pilot",
    )
    print(f"  Dashboard saved: {dashboard_path}")

    summary_path = visualize.plot_budget_summary(
        video_name=video_name,
        budget_results=budget_results,
        output_dir=video_output_dir,
        output_prefix="pilot",
    )
    print(f"  Budget summary saved: {summary_path}")

    # ---- Text report ----
    visualize.print_text_report(
        video_name=video_name,
        total_seconds=total_seconds,
        composite_scores=composite_scores,
        routing_decisions=routing_decisions,
        coupling_flags=coupling_flags,
        coupling_lcd=lcd_flags,
        coupling_scd=scd_flags,
        selected_frames=selected_frames,
        scene_boundaries=visual["scene_boundaries"],
    )

    # ---- Save results JSON ----
    results = {
        "video_name": video_name,
        "total_seconds": total_seconds,
        "composite_scores": composite_scores,
        "coupling_flags": coupling_flags,
        "selected_frames": {str(k): v for k, v in selected_frames.items()},
        "budget_metrics": {str(k): v for k, v in budget_results.items()},
        "scene_boundaries": visual["scene_boundaries"],
        "routing_distribution": _get_routing_distribution(routing_decisions[:total_seconds]),
    }
    result_path = os.path.join(video_output_dir, "results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Results JSON: {result_path}")

    return results


def _get_routing_distribution(routing_decisions: List[Dict]) -> Dict[str, int]:
    counts = {}
    for rd in routing_decisions:
        cat = rd.get("local_cat", "?")
        counts[cat] = counts.get(cat, 0) + 1
    return counts
