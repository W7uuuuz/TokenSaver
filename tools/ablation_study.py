"""
Ablation study — measure each module's contribution by turning it off.
Compares frame selection quality (vs human annotations) across variants.
"""
import os
import sys
import json
import copy
import importlib
from pathlib import Path
from typing import List, Dict

import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k not in os.environ:
                    os.environ[k] = v

from pilot_round1 import config
from pilot_round1 import density_scorer
from pilot_round1 import frame_selector
from pilot_round1.pipeline import (
    extract_visual_features,
    _detect_scenes_ffmpeg,
    _detect_scenes_opencv,
    _detect_fast_cut,
)

ANNOTATIONS_DIR = PROJECT_ROOT / "annotations"
PILOT_OUTPUT = PROJECT_ROOT / "pilot_output"
CACHE_DIR = PILOT_OUTPUT / "qwen_cache"


def load_annotation(video_id: str) -> dict:
    path = ANNOTATIONS_DIR / f"{video_id}_keyframes.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_results(video_id: str) -> dict:
    path = PILOT_OUTPUT / video_id / "results.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_qwen_words(video_id: str) -> list:
    path = CACHE_DIR / "asr" / f"{video_id}_qwen_asr.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("words", [])
    return []


def get_video_path(video_id: str) -> str:
    videos_dir = PROJECT_ROOT / "round1" / "videos"
    for root, dirs, files in os.walk(str(videos_dir)):
        for f in files:
            if f.startswith(video_id) and f.endswith(".mp4"):
                return os.path.join(root, f)
    return None


def compute_metrics(selected: List[int], human_key_frames: List[int],
                    total_seconds: int) -> Dict:
    """Precision, Recall, F1 against human annotations, with 1s tolerance."""
    tolerance = 1
    matched = 0
    for sf in selected:
        for hf in human_key_frames:
            if abs(sf - hf) <= tolerance:
                matched += 1
                break

    prec = matched / len(selected) if selected else 0
    rec = matched / len(human_key_frames) if human_key_frames else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    # Temporal spread
    if len(selected) > 1:
        gaps = np.diff(sorted(selected))
        spread = float(np.std(gaps))
    else:
        spread = 0.0

    return {"precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(f1, 3), "temporal_spread": round(spread, 1)}


def run_variant(video_id: str, video_path: str, variant_name: str,
                variant_config: dict, total_seconds: int,
                scene_boundaries: List[int], coupling_flags: List[int],
                composite_scores: List[float], visual: dict) -> Dict:
    """Run frame selection with a specific configuration variant."""
    budgets = [3, 10, 20, 30, 50]
    results = {}

    for budget in budgets:
        # Build candidates
        candidates = []
        for t in range(total_seconds):
            text_imp = composite_scores[t] / 10.0 if t < len(composite_scores) else 0
            h_vis = visual.get("entropy_norm", [0] * total_seconds)[t] if t < total_seconds else 0
            motion = visual.get("motion_norm", [0] * total_seconds)[t] if t < total_seconds else 0

            i_visual = config.VISUAL_SIM_WEIGHTS[0] * h_vis + config.VISUAL_SIM_WEIGHTS[1] * motion

            coupling = coupling_flags[t] if t < len(coupling_flags) else 0
            boost = variant_config.get("coupling_boost", config.COUPLING_BOOST)
            importance = text_imp * (1 + boost * coupling) + i_visual

            if variant_config.get("text_only", False):
                importance = text_imp * (1 + boost * coupling)

            candidates.append(frame_selector.CandidateFrame(
                second=t,
                importance_text=text_imp,
                importance_visual=i_visual if not variant_config.get("text_only") else 0,
                coupling_flag=int(coupling),
                is_scene_boundary=t in scene_boundaries,
                importance_total=importance,
            ))

        # Deep copy candidates for mutation-safe MMR
        candidates_mut = copy.deepcopy(candidates)

        # Scene boundary boost
        scene_boost = variant_config.get("scene_boost", 1.2)
        if scene_boost > 0:
            for c in candidates_mut:
                if c.is_scene_boundary:
                    c.importance_total *= scene_boost

        # Lambda
        coupling_ratio = sum(coupling_flags) / max(total_seconds, 1)
        if variant_config.get("adaptive_lambda", True):
            if coupling_ratio < 0.05:
                lam = variant_config.get("lambda_override", 0.4)
            else:
                lam = variant_config.get("lambda_override", config.MMR_LAMBDA_REL)
        else:
            lam = variant_config.get("lambda_override", config.MMR_LAMBDA_REL)

        # Select
        hsv_hists = visual.get("hsv_histograms", [])

        if variant_config.get("topk_only", False):
            # Top-K by importance
            sorted_candidates = sorted(candidates_mut, key=lambda c: c.importance_total, reverse=True)
            selected = [c.second for c in sorted_candidates[:budget]]
            # Always include first+last
            if 0 not in selected:
                selected.insert(0, 0)
            if total_seconds - 1 not in selected:
                selected.append(total_seconds - 1)
            selected = sorted(set(selected))[:budget]
        else:
            selected = frame_selector.mmr_select(
                candidates_mut, budget, hsv_hists,
                lambda_rel=lam,
            )

        results[budget] = selected

    return results


def discover_videos():
    """Find all videos with annotations and results."""
    videos = []
    for anno_file in sorted(ANNOTATIONS_DIR.glob("*_keyframes.json")):
        video_id = anno_file.stem.replace("_keyframes", "")
        if video_id == "M-00fa7a854582":
            continue
        vpath = get_video_path(video_id)
        results = load_results(video_id)
        anno = load_annotation(video_id)
        if vpath and results and anno:
            videos.append((video_id, vpath, results, anno))
    return videos


def main():
    videos = discover_videos()
    print(f"Ablation Study — {len(videos)} videos\n")

    # Define variants
    variants = {
        "Full Pipeline": {
            "coupling_boost": config.COUPLING_BOOST,
            "adaptive_lambda": True,
            "lambda_override": None,
            "scene_boost": 1.2,
            "topk_only": False,
            "text_only": False,
        },
        "-Coupling Boost": {
            "coupling_boost": 0.0,
            "adaptive_lambda": True,
            "lambda_override": None,
            "scene_boost": 1.2,
            "topk_only": False,
            "text_only": False,
        },
        "-Adaptive lambda (l=0.7)": {
            "coupling_boost": config.COUPLING_BOOST,
            "adaptive_lambda": False,
            "lambda_override": config.MMR_LAMBDA_REL,
            "scene_boost": 1.2,
            "topk_only": False,
            "text_only": False,
        },
        "-Scene Boost": {
            "coupling_boost": config.COUPLING_BOOST,
            "adaptive_lambda": True,
            "lambda_override": None,
            "scene_boost": 0.0,
            "topk_only": False,
            "text_only": False,
        },
        "Top-K (no MMR)": {
            "coupling_boost": config.COUPLING_BOOST,
            "adaptive_lambda": True,
            "lambda_override": None,
            "scene_boost": 1.2,
            "topk_only": True,
            "text_only": False,
        },
        "Text-Only (no visual)": {
            "coupling_boost": config.COUPLING_BOOST,
            "adaptive_lambda": True,
            "lambda_override": None,
            "scene_boost": 1.2,
            "topk_only": False,
            "text_only": True,
        },
    }

    # Budget focus
    budgets = [10, 20]

    # Accumulate per-variant metrics
    all_metrics = {name: {b: [] for b in budgets} for name in variants}

    for vid, vpath, results, anno in videos:
        total_s = results["total_seconds"]
        key_frames = anno.get("key_frames", [])
        cat = anno.get("category", "?")

        composite_scores = results.get("composite_scores", [0] * total_s)
        coupling_flags = results.get("coupling_flags", [0] * total_s)
        scene_boundaries = results.get("scene_boundaries", [])

        # Load visual features
        visual = extract_visual_features(vpath, total_s, scene_boundaries)

        for var_name, var_cfg in variants.items():
            selections = run_variant(
                vid, vpath, var_name, var_cfg,
                total_s, scene_boundaries, coupling_flags,
                composite_scores, visual,
            )

            for b in budgets:
                metrics = compute_metrics(selections[b], key_frames, total_s)
                all_metrics[var_name][b].append(metrics)

    # Summary
    print(f"{'Variant':<28} {'B=10 F1':>9} {'B=20 F1':>9} {'B=10 Prec':>10} {'B=20 Prec':>10}")
    print("-" * 70)

    base_f1_10 = np.mean([m["f1"] for m in all_metrics["Full Pipeline"][10]])
    base_f1_20 = np.mean([m["f1"] for m in all_metrics["Full Pipeline"][20]])

    for var_name in variants:
        m10 = all_metrics[var_name][10]
        m20 = all_metrics[var_name][20]
        avg_f1_10 = np.mean([x["f1"] for x in m10])
        avg_f1_20 = np.mean([x["f1"] for x in m20])
        avg_prec_10 = np.mean([x["precision"] for x in m10])
        avg_prec_20 = np.mean([x["precision"] for x in m20])

        delta_10 = (avg_f1_10 - base_f1_10) / base_f1_10 * 100 if base_f1_10 > 0 else 0
        marker_10 = f"{avg_f1_10:.3f} ({delta_10:+.0f}%)"
        delta_20 = (avg_f1_20 - base_f1_20) / base_f1_20 * 100 if base_f1_20 > 0 else 0
        marker_20 = f"{avg_f1_20:.3f} ({delta_20:+.0f}%)"

        print(f"{var_name:<28} {marker_10:>9} {marker_20:>9} {avg_prec_10:>10.3f} {avg_prec_20:>10.3f}")

    # Save detailed results
    output = {
        "variants": list(variants.keys()),
        "base_f1": {"10": base_f1_10, "20": base_f1_20},
        "per_variant": {},
    }
    for var_name in variants:
        output["per_variant"][var_name] = {
            str(b): {
                "avg_f1": np.mean([x["f1"] for x in all_metrics[var_name][b]]),
                "avg_precision": np.mean([x["precision"] for x in all_metrics[var_name][b]]),
                "avg_recall": np.mean([x["recall"] for x in all_metrics[var_name][b]]),
                "avg_spread": np.mean([x["temporal_spread"] for x in all_metrics[var_name][b]]),
            }
            for b in budgets
        }

    out_path = PILOT_OUTPUT / "ablation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
