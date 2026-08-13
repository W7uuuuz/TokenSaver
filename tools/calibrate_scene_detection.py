#!/usr/bin/env python3
"""
Calibrate the OpenCV scene detection sigma multiplier against human-annotated key frames.

Evaluates k in [1.0, 1.5, 2.0, 2.5, 3.0] for scene boundary recall, precision, F1,
and "useful" rate (within 1s of any key frame).  Saves results and prints a
recommendation table.
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pilot_round1.pipeline import _detect_scenes_opencv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ANNOTATIONS_DIR = PROJECT_ROOT / "annotations"
OUTPUT_DIR = PROJECT_ROOT / "pilot_output"
MANIFEST_PATH = PROJECT_ROOT / "round1" / "manifest.json"
VIDEOS_DIR = PROJECT_ROOT / "round1" / "videos"

# Sigma multipliers to sweep
K_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0]

# Evaluation tolerances (seconds)
SCENE_CHANGE_TOLERANCE_SEC = 1.0   # ± this many seconds to match a scene change
KEYFRAME_USEFUL_TOLERANCE_SEC = 1.0  # boundary within this of any keyframe → "useful"
MIN_SCENE_GAP_SEC = 3.0            # minimum gap between consecutive keyframes that
                                   # counts as a scene change


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> Optional[dict]:
    """Load a JSON file, returning None if missing / unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"  [WARN] Could not load {path}: {exc}")
        return None


def _find_video_path(local_id: str) -> Optional[str]:
    """Walk VIDEOS_DIR to find the .mp4 file for *local_id*."""
    for root, _dirs, files in os.walk(str(VIDEOS_DIR)):
        for fname in files:
            if fname.startswith(local_id) and fname.lower().endswith(".mp4"):
                return os.path.join(root, fname)
    return None


def _derive_scene_changes(key_frames: List[int]) -> List[float]:
    """Derive scene-change ground truth from human-annotated key frames.

    Two consecutive key frames >= MIN_SCENE_GAP_SEC apart imply a scene change
    between them.  The scene-change instant is estimated as their midpoint.
    Returns a sorted list of fractional seconds.
    """
    kf = sorted(set(key_frames))
    changes: List[float] = []
    for i in range(len(kf) - 1):
        if kf[i + 1] - kf[i] >= MIN_SCENE_GAP_SEC:
            changes.append((kf[i] + kf[i + 1]) / 2.0)
    return changes


def _match_detections(
    predicted: List[int],
    ground_truth: List[float],
    tolerance: float,
) -> Tuple[int, int, int]:
    """Greedy bipartite match. Returns (tp, fp, fn).

    *ground_truth* may be fractional (scene-change midpoints).
    *predicted* are integer seconds from OpenCV.
    Each ground-truth event can be matched at most once.
    """
    pred = sorted(predicted)
    gt = sorted(ground_truth)
    matched_gt = set()
    tp = 0

    for p in pred:
        best_idx = -1
        best_dist = float("inf")
        for j, g in enumerate(gt):
            if j in matched_gt:
                continue
            d = abs(p - g)
            if d <= tolerance and d < best_dist:
                best_dist = d
                best_idx = j
        if best_idx >= 0:
            tp += 1
            matched_gt.add(best_idx)

    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn


def _prec_rec_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Compute precision, recall, F1. Return 0.0 when undefined."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def _useful_fraction(predicted: List[int], key_frames: List[int], tolerance: float) -> float:
    """Fraction of *predicted* boundaries within *tolerance* seconds of ANY key frame."""
    if not predicted:
        return 1.0  # vacuously useful
    kf = sorted(set(key_frames))
    useful = 0
    for p in predicted:
        if any(abs(p - k) <= tolerance for k in kf):
            useful += 1
    return useful / len(predicted)


# ---------------------------------------------------------------------------
# Per-video evaluation
# ---------------------------------------------------------------------------
def evaluate_video(
    video_id: str,
    video_path: str,
    key_frames: List[int],
    total_seconds: int,
) -> Dict:
    """Run OpenCV scene detection at every k and compute metrics.

    Returns a dict keyed by k with {precision, recall, f1, n_detected, useful_frac,
    n_scene_gt, ...}.
    """
    results: Dict[float, dict] = {}

    scene_gt = _derive_scene_changes(key_frames)

    for k in K_VALUES:
        try:
            boundaries = _detect_scenes_opencv(
                video_path, total_seconds, sigma_multiplier=k
            )
        except Exception as exc:
            print(f"    [ERROR] OpenCV failed for k={k}: {exc}")
            boundaries = []

        # Scene-change evaluation
        tp, fp, fn = _match_detections(
            boundaries, scene_gt, SCENE_CHANGE_TOLERANCE_SEC
        )
        p, r, f1 = _prec_rec_f1(tp, fp, fn)

        useful = _useful_fraction(boundaries, key_frames, KEYFRAME_USEFUL_TOLERANCE_SEC)

        results[k] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "n_detected": len(boundaries),
            "n_scene_gt": len(scene_gt),
            "useful_fraction": round(useful, 4),
            "detected_seconds": boundaries,
        }

    return results


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------
def _fmt_p_r_f1(metrics: dict) -> str:
    p = metrics["precision"]
    r = metrics["recall"]
    f1 = metrics["f1"]
    return f"{p:.2f}/{r:.2f}/{f1:.2f}"


def print_summary_table(
    per_video: Dict[str, dict],
    k_values: List[float],
) -> None:
    """Print the calibration summary table and recommendation."""
    header_cols = ["Video", "Ann Sc"]
    for k in k_values:
        header_cols.append(f"k={k}")
    header_cols.append("Best k")

    print("\nScene Detection Calibration")
    print("=" * 100)
    # Build formatted rows
    header = f"{'Video':<25s} | {'Ann Sc':>6s}"
    for k in k_values:
        header += f" | {'k=' + str(k):<16s}"
    header += " | Best k"
    print(header)

    # Sub-header for P/R/F1 columns
    sub_header = f"{'':25s} | {'':>6s}"
    for _ in k_values:
        sub_header += " | P / R / F1    "
    sub_header += " |"
    print(sub_header)
    print("-" * 100)

    # Aggregate metrics across videos for each k
    agg_f1: Dict[float, list] = {k: [] for k in k_values}
    agg_p: Dict[float, list] = {k: [] for k in k_values}
    agg_r: Dict[float, list] = {k: [] for k in k_values}

    for vid, info in sorted(per_video.items()):
        n_gt = info["n_scene_gt"]
        row = f"{vid:<25s} | {n_gt:>6d}"

        best_k = None
        best_f1 = -1.0

        for k in k_values:
            m = info["results"][k]
            agg_f1[k].append(m["f1"])
            agg_p[k].append(m["precision"])
            agg_r[k].append(m["recall"])
            row += f" | {_fmt_p_r_f1(m):<16s}"
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_k = k

        row += f" | k={best_k} (F1={best_f1:.2f})"
        print(row)

    # Mean row
    print("-" * 100)
    mean_row = f"{'MEAN':<25s} | {'':>6s}"
    best_mean_k = None
    best_mean_f1 = -1.0
    for k in k_values:
        mean_f1 = np.mean(agg_f1[k])
        mean_p = np.mean(agg_p[k])
        mean_r = np.mean(agg_r[k])
        mean_row += f" | {mean_p:.2f}/{mean_r:.2f}/{mean_f1:.2f}"
        mean_row += " " * (16 - len(f"{mean_p:.2f}/{mean_r:.2f}/{mean_f1:.2f}"))
        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            best_mean_k = k
    print(mean_row)
    print("=" * 100)

    # Recommendation
    print(f"\nBest k by mean F1:  k={best_mean_k}  (mean F1={best_mean_f1:.2f})")
    current_f1 = np.mean(agg_f1.get(2.0, [0.0]))
    print(f"Current k=2.0:       mean F1={current_f1:.2f}")

    if abs(best_mean_k - 2.0) < 0.01:
        print("Recommendation:      keep k=2.0 (current default is optimal)")
    else:
        improvement = best_mean_f1 - current_f1
        print(f"Recommendation:      k={best_mean_k} "
              f"(ΔF1={'+' if improvement >= 0 else ''}{improvement:.2f} vs current k=2.0)")

    # Per-k detailed stats
    print(f"\n{'k':>6s} | {'mean P':>7s} | {'mean R':>7s} | {'mean F1':>7s} | {'mean N':>7s} | "
          f"{'mean Useful':>11s}")
    print("-" * 65)
    for k in k_values:
        n_detected = [
            per_video[vid]["results"][k]["n_detected"]
            for vid in per_video
        ]
        useful = [
            per_video[vid]["results"][k]["useful_fraction"]
            for vid in per_video
        ]
        print(f"{k:>6.1f} | {np.mean(agg_p[k]):>7.3f} | {np.mean(agg_r[k]):>7.3f} | "
              f"{np.mean(agg_f1[k]):>7.3f} | {np.mean(n_detected):>7.1f} | "
              f"{np.mean(useful):>11.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # 1. Collect annotation files
    annotation_files = sorted(ANNOTATIONS_DIR.glob("*_keyframes.json"))
    if not annotation_files:
        print("No annotation files found in:", ANNOTATIONS_DIR)
        print()
        print("To create annotations:")
        print("  1. Run:  python tools/prep_annotation_data.py")
        print("     This generates annotation_data.json in pilot_output/<video_id>/")
        print("  2. Open tools/annotate.html in a browser to annotate key frames")
        print("  3. Download keyframes JSON into annotations/<video_id>_keyframes.json")
        print()
        print("Each annotation file should have the format:")
        print('  {"video_id": "M-xxx", "key_frames": [3, 7, 15, ...]}')
        return

    print(f"Found {len(annotation_files)} annotation file(s):")
    for f in annotation_files:
        print(f"  {f.name}")

    # 2. Load manifest for video durations
    manifest = _load_json(MANIFEST_PATH)
    duration_map: Dict[str, int] = {}
    if manifest and "materials" in manifest:
        for m in manifest["materials"]:
            vid = m.get("local_id", "")
            if vid:
                duration_map[vid] = m.get("duration_seconds", 60)

    # 3. Evaluate each annotated video
    per_video: Dict[str, dict] = {}

    for anno_path in annotation_files:
        data = _load_json(anno_path)
        if data is None:
            continue

        video_id = data.get("video_id", anno_path.stem.replace("_keyframes", ""))
        key_frames = data.get("key_frames", [])

        if not key_frames:
            print(f"\n  [SKIP] {video_id}: empty key_frames list")
            continue

        print(f"\n  [{video_id}] {len(key_frames)} key frames")

        # Find video file
        vpath = _find_video_path(video_id)
        if vpath is None:
            print(f"    [SKIP] Video file not found for {video_id}")
            continue

        # Duration: manifest or estimate from keyframes
        total_seconds = duration_map.get(video_id, max(key_frames) + 10)

        # Evaluate
        results = evaluate_video(video_id, vpath, key_frames, total_seconds)

        n_scene_gt = len(_derive_scene_changes(key_frames))
        per_video[video_id] = {
            "video_id": video_id,
            "n_key_frames": len(key_frames),
            "n_scene_gt": n_scene_gt,
            "results": results,
        }

    if not per_video:
        print("\nNo videos could be evaluated.")
        return

    # 4. Print summary table
    print_summary_table(per_video, K_VALUES)

    # 5. Save calibration results
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    calibration_output = {}
    for vid, info in sorted(per_video.items()):
        cal_entry = {
            "video_id": vid,
            "n_key_frames": info["n_key_frames"],
            "n_scene_gt": info["n_scene_gt"],
            "k_results": {},
        }
        for k in K_VALUES:
            m = info["results"][k]
            cal_entry["k_results"][str(k)] = {
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "n_detected": m["n_detected"],
                "useful_fraction": m["useful_fraction"],
                "detected_seconds": m["detected_seconds"],
            }
        calibration_output[vid] = cal_entry

    # Summary stats
    summary = {"k_summary": {}}
    for k in K_VALUES:
        f1_vals = [info["results"][k]["f1"] for info in per_video.values()]
        p_vals = [info["results"][k]["precision"] for info in per_video.values()]
        r_vals = [info["results"][k]["recall"] for info in per_video.values()]
        n_vals = [info["results"][k]["n_detected"] for info in per_video.values()]
        u_vals = [info["results"][k]["useful_fraction"] for info in per_video.values()]

        summary["k_summary"][str(k)] = {
            "mean_precision": round(np.mean(p_vals), 4),
            "mean_recall": round(np.mean(r_vals), 4),
            "mean_f1": round(np.mean(f1_vals), 4),
            "mean_n_detected": round(np.mean(n_vals), 1),
            "mean_useful_fraction": round(np.mean(u_vals), 4),
        }

    # Best k recommendation
    best_k = max(summary["k_summary"].items(), key=lambda kv: kv[1]["mean_f1"])
    summary["best_k"] = float(best_k[0])
    summary["best_mean_f1"] = best_k[1]["mean_f1"]
    summary["current_k"] = 2.0
    current_f1 = summary["k_summary"].get("2.0", {}).get("mean_f1", 0.0)
    summary["current_mean_f1"] = current_f1

    output = {
        "calibration_date": None,  # fill in manually if desired
        "sigma_values_tested": K_VALUES,
        "tolerance_scene_change_sec": SCENE_CHANGE_TOLERANCE_SEC,
        "tolerance_keyframe_useful_sec": KEYFRAME_USEFUL_TOLERANCE_SEC,
        "min_scene_gap_sec": MIN_SCENE_GAP_SEC,
        "summary": summary,
        "per_video": calibration_output,
    }

    out_path = OUTPUT_DIR / "scene_calibration.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nCalibration results saved to: {out_path}")


if __name__ == "__main__":
    main()
