"""
MMR Lambda (relevance vs diversity) sensitivity analysis
using human-annotated key frames.

Re-runs MMR frame selection across a range of lambda values [0.3, 0.9]
and evaluates against human annotations to find the optimal trade-off.
"""
import os
import sys
import json
import copy
import math
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pilot_round1 import config
from pilot_round1 import frame_selector
from pilot_round1.pipeline import extract_visual_features

OUTPUT_DIR = PROJECT_ROOT / "pilot_output"
ANNOTATIONS_DIR = PROJECT_ROOT / "annotations"
MANIFEST_PATH = PROJECT_ROOT / "round1" / "manifest.json"

LAMBDA_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BUDGETS = [10, 20]

# Matplotlib setup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# Helpers
# ============================================================================

def load_manifest() -> list:
    """Load video manifest entries that have results."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    # Filter to entries that have results.json
    available = []
    for m in manifest["materials"]:
        vid = m["local_id"]
        if (OUTPUT_DIR / vid / "results.json").exists():
            available.append(m)
    return available


def find_video_path(local_id: str) -> Optional[str]:
    """Locate the actual video file by scanning the videos directory."""
    videos_dir = PROJECT_ROOT / "round1" / "videos"
    if not videos_dir.exists():
        return None
    for root, dirs, files in os.walk(str(videos_dir)):
        for f in files:
            if f.startswith(local_id) and f.endswith(".mp4"):
                return os.path.join(root, f)
    return None


def load_results(video_id: str) -> Optional[dict]:
    """Load results.json for a video."""
    path = OUTPUT_DIR / video_id / "results.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_key_frames(video_id: str) -> Optional[Set[int]]:
    """Load human-annotated key frames.

    Expected format: {"video_id": "...", "key_frames": [3, 5, 12, ...]}
    Returns None if annotation file is missing.
    """
    path = ANNOTATIONS_DIR / f"{video_id}_keyframes.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Support both formats
    if isinstance(data, list):
        return set(data)
    if "key_frames" in data:
        return set(data["key_frames"])
    return None


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(
    selected: List[int],
    key_frames: Optional[Set[int]],
    coupling_flags: List[int],
    scene_boundaries: List[int],
    total_seconds: int,
) -> Dict[str, float]:
    """Compute all evaluation metrics for a given selection.

    Returns dict with: precision, recall, f1, coupling_coverage,
    scene_coverage, temporal_spread.
    Precision/recall/f1 may be None if key_frames is None.
    """
    selected_set = set(selected)
    n_sel = len(selected_set)

    # --- Key frame precision / recall / F1 ---
    prec, rec, f1 = None, None, None
    if key_frames is not None and len(key_frames) > 0:
        overlap = selected_set & key_frames
        prec = len(overlap) / n_sel if n_sel > 0 else 0.0
        rec = len(overlap) / len(key_frames)
        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0

    # --- Coupling coverage ---
    coupling_seconds = [i for i, f in enumerate(coupling_flags) if f == 1]
    if coupling_seconds:
        coupling_covered = len([s for s in coupling_seconds if s in selected_set])
        coupling_cov = coupling_covered / len(coupling_seconds) * 100
    else:
        coupling_cov = 100.0  # no coupling → trivially covered

    # --- Scene coverage ---
    if scene_boundaries:
        scene_covered = len([s for s in scene_boundaries if s in selected_set])
        scene_cov = scene_covered / len(scene_boundaries) * 100
    else:
        scene_cov = 100.0

    # --- Temporal spread: std of inter-frame intervals ---
    selected_sorted = sorted(selected)
    if len(selected_sorted) >= 2:
        intervals = [selected_sorted[i + 1] - selected_sorted[i]
                     for i in range(len(selected_sorted) - 1)]
        temporal_spread = float(np.std(intervals))
    else:
        temporal_spread = 0.0

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "coupling_coverage": coupling_cov,
        "scene_coverage": scene_cov,
        "temporal_spread": temporal_spread,
    }


# ============================================================================
# Per-video sensitivity analysis
# ============================================================================

def build_candidates_fresh(
    results: dict,
    entropy_norm: List[float],
    motion_norm: List[float],
    object_counts: List[float],
) -> List[frame_selector.CandidateFrame]:
    """Build a fresh CandidateFrame list from results + visual features.

    Returns a NEW list that can be safely mutated by MMR selection
    (scene boundary importance boost modifies in-place).
    """
    total_seconds = results["total_seconds"]
    composite_scores = results["composite_scores"]
    coupling_flags = results["coupling_flags"]
    scene_boundaries = results["scene_boundaries"]

    return frame_selector.build_candidates(
        n_seconds=total_seconds,
        coupling_flags=coupling_flags,
        composite_text_scores=composite_scores,
        visual_entropy=entropy_norm,
        visual_motion=motion_norm,
        object_counts=object_counts,
        scene_boundaries=scene_boundaries,
    )


def analyze_video(video_id: str, video_path: str) -> Optional[Dict]:
    """Run full lambda sensitivity analysis for one video.

    Returns: dict keyed by (lambda_val, budget) -> metrics dict,
             plus metadata for aggregate averaging.
    Returns None if visual features cannot be extracted.
    """
    results = load_results(video_id)
    if results is None:
        print(f"  [SKIP] {video_id}: no results.json")
        return None

    key_frames = load_key_frames(video_id)
    has_annotations = key_frames is not None
    total_seconds = results["total_seconds"]

    # --- Extract visual features ---
    try:
        visual = extract_visual_features(video_path, total_seconds)
    except Exception as e:
        print(f"  [SKIP] {video_id}: visual extraction failed ({e})")
        return None

    entropy_norm = visual["entropy_norm"]
    motion_norm = visual["motion_norm"]
    object_counts = visual["object_counts"]
    hsv_histograms = visual.get("hsv_histograms", None)

    coupling_flags = results["coupling_flags"]
    scene_boundaries = results["scene_boundaries"]

    # Pad arrays to match total_seconds
    def _pad(arr, n, default=0.0):
        if len(arr) < n:
            return arr + [default] * (n - len(arr))
        return arr[:n]

    entropy_norm = _pad(entropy_norm, total_seconds)
    motion_norm = _pad(motion_norm, total_seconds)
    object_counts = _pad(object_counts, total_seconds)
    coupling_flags = _pad(coupling_flags, total_seconds, 0)

    # Build base candidates (without mutation)
    base_candidates = build_candidates_fresh(
        results, entropy_norm, motion_norm, object_counts
    )

    # --- For each lambda, run MMR at B=10 and B=20 ---
    per_lambda = {}
    for lam in LAMBDA_VALUES:
        for B in BUDGETS:
            # Deep-copy candidates because mmr_select mutates importance_total
            cands = copy.deepcopy(base_candidates)
            selected = frame_selector.mmr_select(
                cands, B, hsv_histograms, lambda_rel=lam
            )
            metrics = compute_metrics(
                selected, key_frames, coupling_flags, scene_boundaries, total_seconds
            )
            per_lambda[(lam, B)] = metrics

    return {
        "video_id": video_id,
        "total_seconds": total_seconds,
        "has_annotations": has_annotations,
        "per_lambda": per_lambda,
    }


# ============================================================================
# Aggregation
# ============================================================================

def aggregate(results: List[Dict]) -> Tuple[Dict, Dict]:
    """Aggregate per-video results into mean metrics across videos.

    Returns: (agg_b10, agg_b20) each mapping lambda -> mean metrics dict.
    """
    agg = {B: {} for B in BUDGETS}

    for B in BUDGETS:
        for lam in LAMBDA_VALUES:
            # Collect per-video metrics for this (lam, B)
            precs, recs, f1s = [], [], []
            coups, scenes, spreads = [], [], []

            for r in results:
                metric = r["per_lambda"].get((lam, B))
                if metric is None:
                    continue
                if metric["precision"] is not None:
                    precs.append(metric["precision"])
                    recs.append(metric["recall"])
                    f1s.append(metric["f1"])
                coups.append(metric["coupling_coverage"])
                scenes.append(metric["scene_coverage"])
                spreads.append(metric["temporal_spread"])

            agg[B][lam] = {
                "precision": np.mean(precs) if precs else None,
                "recall": np.mean(recs) if recs else None,
                "f1": np.mean(f1s) if f1s else None,
                "coupling_coverage": np.mean(coups) if coups else None,
                "scene_coverage": np.mean(scenes) if scenes else None,
                "temporal_spread": np.mean(spreads) if spreads else None,
            }

    return agg[10], agg[20]


# ============================================================================
# Output
# ============================================================================

def print_table(agg_b10: Dict, agg_b20: Dict):
    """Print formatted results table."""
    headers = ["λ", "Prec", "Rec", "F1", "Coup%", "Scene%", "Spread"]
    sep = "------|-------|-------|-------|-------|--------|------"

    def _row(lam, m):
        prec = f"{m['precision']:.2f}" if m['precision'] is not None else "N/A"
        rec = f"{m['recall']:.2f}" if m['recall'] is not None else "N/A"
        f1 = f"{m['f1']:.2f}" if m['f1'] is not None else "N/A"
        coup = f"{m['coupling_coverage']:.0f}" if m['coupling_coverage'] is not None else "N/A"
        scene = f"{m['scene_coverage']:.0f}" if m['scene_coverage'] is not None else "N/A"
        spread = f"{m['temporal_spread']:.1f}" if m['temporal_spread'] is not None else "N/A"
        return f"{lam:<5} | {prec:<5} | {rec:<5} | {f1:<5} | {coup:<5} | {scene:<6} | {spread}"

    for B_label, agg in [("B=10:", agg_b10), ("B=20:", agg_b20)]:
        print(f"\n{B_label}")
        print(" ".join(headers))
        print(sep)
        for lam in LAMBDA_VALUES:
            print(_row(lam, agg[lam]))


def make_figure(agg_b10: Dict, agg_b20: Dict,
                n_annotated: int, n_total: int, output_path: str):
    """Generate 2-panel matplotlib figure and save to output_path."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        f"MMR λ Sensitivity Analysis (N={n_total} videos, {n_annotated} annotated)",
        fontsize=13, fontweight="bold",
    )

    lam_arr = np.array(LAMBDA_VALUES)

    def _safe(arr, default=np.nan):
        return np.array([v if v is not None else default for v in arr])

    # ---- Panel 1: λ vs Key Frame Precision / Recall / F1 ----
    for B, agg, color_map in [
        (10, agg_b10, {"prec": "#2196F3", "rec": "#FF9800", "f1": "#4CAF50"}),
        (20, agg_b20, {"prec": "#90CAF9", "rec": "#FFCC80", "f1": "#A5D6A7"}),
    ]:
        precs = _safe([agg[lam]["precision"] for lam in LAMBDA_VALUES])
        recs = _safe([agg[lam]["recall"] for lam in LAMBDA_VALUES])
        f1s = _safe([agg[lam]["f1"] for lam in LAMBDA_VALUES])
        ls = "-" if B == 10 else "--"
        lw = 2.0 if B == 10 else 1.5
        ax1.plot(lam_arr, precs, marker="o", linestyle=ls, linewidth=lw,
                 color=color_map["prec"], label=f"Prec B={B}")
        ax1.plot(lam_arr, recs, marker="s", linestyle=ls, linewidth=lw,
                 color=color_map["rec"], label=f"Rec B={B}")
        ax1.plot(lam_arr, f1s, marker="D", linestyle=ls, linewidth=lw,
                 color=color_map["f1"], label=f"F1 B={B}")

    ax1.set_xlabel("λ (relevance weight)")
    ax1.set_ylabel("Score")
    ax1.set_title("Key Frame Precision / Recall / F1 vs λ")
    ax1.set_xlim(0.25, 0.95)
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, ncol=2, loc="lower right")

    # ---- Panel 2: λ vs Coupling / Scene Coverage + Temporal Spread ----
    ax2_twin = ax2.twinx()

    for B, agg, color_map in [
        (10, agg_b10, {"coupling": "#E91E63", "scene": "#9C27B0", "spread": "#009688"}),
        (20, agg_b20, {"coupling": "#F48FB1", "scene": "#CE93D8", "spread": "#80CBC4"}),
    ]:
        couplings = _safe([agg[lam]["coupling_coverage"] for lam in LAMBDA_VALUES])
        scenes = _safe([agg[lam]["scene_coverage"] for lam in LAMBDA_VALUES])
        spreads = _safe([agg[lam]["temporal_spread"] for lam in LAMBDA_VALUES])
        ls = "-" if B == 10 else "--"
        lw = 2.0 if B == 10 else 1.5

        ax2.plot(lam_arr, couplings, marker="o", linestyle=ls, linewidth=lw,
                 color=color_map["coupling"], label=f"Coup% B={B}")
        ax2.plot(lam_arr, scenes, marker="s", linestyle=ls, linewidth=lw,
                 color=color_map["scene"], label=f"Scene% B={B}")
        ax2_twin.plot(lam_arr, spreads, marker="^", linestyle=ls, linewidth=lw,
                      color=color_map["spread"], label=f"Spread B={B}")

    ax2.set_xlabel("λ (relevance weight)")
    ax2.set_ylabel("Coverage (%)")
    ax2_twin.set_ylabel("Temporal Spread (std)")
    ax2.set_title("Coverage Metrics vs λ")
    ax2.set_xlim(0.25, 0.95)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)

    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7, ncol=2,
               loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {output_path}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def discover_videos():
    """Find all videos that have annotation files or results.json."""
    videos = []
    anno_dir = PROJECT_ROOT / "annotations"
    for af in sorted(anno_dir.glob("*_keyframes.json")):
        vid = af.stem.replace("_keyframes", "")
        # Skip visual-only
        if vid == "M-00fa7a854582":
            continue
        vpath = find_video_path(vid)
        if vpath:
            videos.append({"local_id": vid, "vpath": vpath})
    return videos


def main():
    print("MMR Lambda Sensitivity Analysis")
    print("=" * 36)

    videos = discover_videos()
    print(f"Loading annotations...")
    annotation_counts = {}
    for v in videos:
        vid = v["local_id"]
        kf = load_key_frames(vid)
        if kf is not None:
            annotation_counts[vid] = len(kf)
            print(f"  {vid}: {len(kf)} key frames")
        else:
            print(f"  {vid}: (no annotations)")

    # --- Run per-video analysis ---
    print(f"\nRunning MMR at λ ∈ {LAMBDA_VALUES}, B ∈ {BUDGETS}...")
    video_results = []

    for v in videos:
        vid = v["local_id"]
        vpath = v["vpath"]
        if vpath is None:
            print(f"  [SKIP] {vid}: video file not found")
            continue

        print(f"  [{vid}] Extracting visual features + running MMR...")
        r = analyze_video(vid, vpath)
        if r is not None:
            video_results.append(r)
            print(f"    Done ({r['total_seconds']}s, annotations={'yes' if r['has_annotations'] else 'no'})")

    if not video_results:
        print("\nNo videos could be analyzed. Exiting.")
        return

    # --- Aggregate ---
    agg_b10, agg_b20 = aggregate(video_results)
    n_annotated = sum(1 for r in video_results if r["has_annotations"])

    # --- Print tables ---
    print_table(agg_b10, agg_b20)

    # --- Recommendation ---
    # Find lambda that maximizes F1 for B=10 and B=20 (when annotations exist)
    best_lam = None
    best_f1 = -1.0
    for lam in LAMBDA_VALUES:
        f1_10 = agg_b10[lam]["f1"]
        f1_20 = agg_b20[lam]["f1"]
        if f1_10 is not None and f1_20 is not None:
            avg_f1 = (f1_10 + f1_20) / 2
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_lam = lam

    current_lam = config.MMR_LAMBDA_REL
    if best_lam is not None:
        current_f1 = None
        if agg_b10[current_lam]["f1"] is not None and agg_b20[current_lam]["f1"] is not None:
            current_f1 = (agg_b10[current_lam]["f1"] + agg_b20[current_lam]["f1"]) / 2

        spread_b10 = agg_b10[best_lam]["temporal_spread"]
        spread_b20 = agg_b20[best_lam]["temporal_spread"]

        print(f"\nRecommendation: λ={best_lam} (avg F1={best_f1:.3f}, spread B10={spread_b10:.1f}, B20={spread_b20:.1f})")
        if current_f1 is not None:
            diff_pct = ((best_f1 - current_f1) / current_f1 * 100) if current_f1 > 0 else 0.0
            sign = "+" if diff_pct >= 0 else ""
            print(f"Current λ={current_lam}: avg F1={current_f1:.3f} (difference: {sign}{diff_pct:.1f}%)")
    else:
        print(f"\nRecommendation: Cannot determine — no annotation data available.")
        print(f"Current λ={current_lam} (default)")

    # --- Generate figure ---
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)
    output_path = str(OUTPUT_DIR / "lambda_sensitivity.png")
    make_figure(agg_b10, agg_b20, n_annotated, len(video_results), output_path)


if __name__ == "__main__":
    main()
