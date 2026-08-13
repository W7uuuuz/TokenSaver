#!/usr/bin/env python3
"""
Grid search over 5-dim text density weights to find the optimal combination
against human-annotated key frames.

Evaluates weight combinations using NDCG@K and AUC metrics.

Usage:
    python tools/grid_search_weights.py
"""

import json
import math
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pilot_round1 import config
from pilot_round1.density_scorer import compute_five_dim_scores
from pilot_round1.pipeline import build_text_windows


# ============================================================================
# Constants
# ============================================================================
ANNOTATIONS_DIR = PROJECT_ROOT / "annotations"
PILOT_OUTPUT_DIR = PROJECT_ROOT / "pilot_output"
ASR_CACHE_DIR = PILOT_OUTPUT_DIR / "qwen_cache" / "asr"

NUM_RANDOM_COMBOS = 500
WEIGHT_MIN = 0.05
WEIGHT_MAX = 0.40
DIM_NAMES = ["ttr", "entropy", "surprisal", "info_point", "redundancy"]
DIM_LABELS = ["TTR", "Ent", "Surp", "InfoP", "Redun"]

# Video to skip: M-00fa7a854582 has no audio (pure visual mode), does not use
# text density scoring, so it is excluded from the weight grid search.
SKIP_VIDEO_IDS = {"M-00fa7a854582"}


# ============================================================================
# Data loading helpers
# ============================================================================
def load_annotation(video_id: str) -> Optional[Dict]:
    """Load human annotation keyframes for a video. Returns None if missing."""
    path = ANNOTATIONS_DIR / f"{video_id}_keyframes.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_asr_data(video_id: str) -> Optional[Dict]:
    """Load Qwen3 ASR data from cache. Returns None if missing."""
    path = ASR_CACHE_DIR / f"{video_id}_qwen_asr.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_results(video_id: str) -> Optional[Dict]:
    """Load results.json for a video. Returns None if missing."""
    path = PILOT_OUTPUT_DIR / video_id / "results.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_asr_from_words(words: List[Dict], total_seconds: int) -> Dict:
    """
    Build an ASR-compatible result dict from word-level timestamps,
    matching the structure expected by build_text_windows().
    """
    timeseries = []
    for sec in range(total_seconds):
        sec_words = [w for w in words if int(w["start"]) == sec]
        sec_text = "".join(w["word"] for w in sec_words)
        timeseries.append({
            "second": sec,
            "text": sec_text,
            "words": sec_words,
        })

    full_text = "".join(w["word"] for w in words)
    return {
        "video_path": "",
        "full_text": full_text,
        "segments": [],
        "timeseries": timeseries,
        "total_seconds": total_seconds,
    }


# ============================================================================
# Weight generation
# ============================================================================
def generate_weight_combos(
    n_random: int = NUM_RANDOM_COMBOS,
    rng: np.random.Generator = None,
) -> List[Dict[str, float]]:
    """
    Generate random weight combinations via Dirichlet(1,1,1,1,1),
    filtering to ensure each weight is in [WEIGHT_MIN, WEIGHT_MAX].
    Also includes the current config weights and equal weights as baselines.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    combos = []

    # Sample from Dirichlet until we have enough valid combos
    attempts = 0
    max_attempts = n_random * 50  # safety valve
    while len(combos) < n_random and attempts < max_attempts:
        raw = rng.dirichlet(np.ones(5))
        if all(WEIGHT_MIN <= w <= WEIGHT_MAX for w in raw):
            combo = {name: round(float(raw[i]), 4) for i, name in enumerate(DIM_NAMES)}
            # Renormalize to exactly 1.0 after rounding
            total = sum(combo.values())
            combo = {k: round(v / total, 4) for k, v in combo.items()}
            # Fix the last one to make sum exactly 1.0000
            combo[DIM_NAMES[-1]] = round(1.0 - sum(combo[k] for k in DIM_NAMES[:-1]), 4)
            combos.append(combo)
        attempts += 1

    if len(combos) < n_random:
        print(f"  [WARN] Only generated {len(combos)} valid combos (requested {n_random})")

    # Add baselines at the end
    current = {
        "ttr": config.WEIGHT_TTR,
        "entropy": config.WEIGHT_ENTROPY,
        "surprisal": config.WEIGHT_SURPRISAL,
        "info_point": config.WEIGHT_INFO_POINT,
        "redundancy": config.WEIGHT_REDUNDANCY,
    }
    equal = {name: 0.20 for name in DIM_NAMES}

    return combos, current, equal


# ============================================================================
# Composite score computation
# ============================================================================
def compute_composite_scores(dim_norms: Dict[str, List[float]], weights: Dict[str, float]) -> List[float]:
    """
    Compute weighted composite scores (0-10 scale) from normalized dimensions.

    Args:
        dim_norms: Dict with keys 'ttr_norm', 'entropy_norm', 'surprisal_norm',
                   'info_point_norm', 'redundancy_norm'.
        weights: Dict with keys 'ttr', 'entropy', 'surprisal', 'info_point', 'redundancy'.

    Returns:
        List of composite scores in [0, 10].
    """
    n = len(dim_norms["ttr_norm"])
    composite = []
    for i in range(n):
        score = (
            weights["ttr"] * dim_norms["ttr_norm"][i]
            + weights["entropy"] * dim_norms["entropy_norm"][i]
            + weights["surprisal"] * dim_norms["surprisal_norm"][i]
            + weights["info_point"] * dim_norms["info_point_norm"][i]
            + weights["redundancy"] * dim_norms["redundancy_norm"][i]
        )
        composite.append(round(score * 10, 2))
    return composite


# ============================================================================
# Evaluation metrics
# ============================================================================
def ndcg_at_k(scores: List[float], labels: List[int], k: int) -> float:
    """
    Compute NDCG@K: how well high-scoring seconds recover the key frame labels.

    Args:
        scores: Composite scores for each second.
        labels: 1 if the second is a key frame, 0 otherwise.
        k: Number of top results to consider (typically len(key_frames)).

    Returns:
        NDCG value in [0, 1].
    """
    n = len(scores)
    k = min(k, n)
    if k == 0:
        return 0.0

    # Sort indices by descending score
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)

    # DCG@K
    dcg = 0.0
    for i in range(k):
        idx = ranked[i]
        dcg += labels[idx] / math.log2(i + 2)

    # IDCG@K (ideal: all key frames at the top)
    ideal_labels = sorted(labels, reverse=True)
    idcg = 0.0
    for i in range(k):
        idcg += ideal_labels[i] / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def compute_auc(scores: List[float], labels: List[int]) -> float:
    """
    Compute ROC AUC treating key_frame labels as binary class and composite
    scores as the predictor. Uses pair-counting (Mann-Whitney U formulation).

    Returns:
        AUC in [0, 1]. Returns 0.5 when there are no positive or negative
        samples (undefined case).
    """
    pos_indices = [i for i, l in enumerate(labels) if l == 1]
    neg_indices = [i for i, l in enumerate(labels) if l == 0]

    n_pos = len(pos_indices)
    n_neg = len(neg_indices)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # For efficiency, sort by score and track cumulative positives
    # This is O(n log n) instead of O(n^2)
    indexed = [(scores[i], labels[i]) for i in range(len(scores))]
    indexed.sort(key=lambda x: x[0])  # ascending score

    auc_sum = 0.0
    cum_pos = 0.0
    for _, label in indexed:
        if label == 1:
            cum_pos += 1.0
        else:
            auc_sum += cum_pos

    # Handle ties: same score, different labels
    # Pair-count: for each neg, count how many pos have strictly higher score (+0.5 for ties)
    # We need to correct for ties
    i = 0
    n = len(indexed)
    while i < n:
        j = i
        while j < n and indexed[j][0] == indexed[i][0]:
            j += 1
        if j > i + 1:
            # Ties exist in [i, j)
            n_pos_tied = sum(1 for k in range(i, j) if indexed[k][1] == 1)
            # Correction: for each pos-pos tie and neg-neg tie, subtract 0.5 * count
            # from the over-count in the cumulative sum
            n_neg_tied = (j - i) - n_pos_tied
            # The cumulative method over-counts: subtract 0.5 * n_pos_tied * n_neg_tied
            auc_sum -= 0.5 * n_pos_tied * n_neg_tied
            # The cumulative also mis-orders pos-pos: we need to subtract cumulated
            # pos within the tied block from being counted for subsequent negs
        i = j

    # Total pairs = n_pos * n_neg
    return auc_sum / (n_pos * n_neg)


# Better AUC implementation using rank-based approach (handles ties correctly)
def compute_auc_rank(scores: List[float], labels: List[int]) -> float:
    """
    Compute ROC AUC using the rank-sum formulation, which handles tied scores
    correctly by assigning the average rank to tied values.

    Returns:
        AUC in [0, 1]. Returns 0.5 when no positive or negative samples exist.
    """
    n = len(scores)
    pos_indices = [i for i, l in enumerate(labels) if l == 1]
    neg_indices = [i for i, l in enumerate(labels) if l == 0]

    n_pos = len(pos_indices)
    n_neg = len(neg_indices)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Create (score, label) pairs and sort by score ascending
    pairs = [(scores[i], labels[i]) for i in range(n)]
    pairs.sort(key=lambda x: x[0])

    # Assign average ranks for tied scores
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-indexed average rank
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    # Get ranks for positive samples
    pos_ranks = []
    for i in range(n):
        if pairs[i][1] == 1:
            pos_ranks.append(ranks[i])

    rank_sum = sum(pos_ranks)
    # Mann-Whitney U → AUC
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc


# ============================================================================
# Build labels from key frames
# ============================================================================
def key_frames_to_labels(key_frames: List[int], total_seconds: int) -> List[int]:
    """Convert a list of key frame second indices into a binary label array."""
    labels = [0] * total_seconds
    for kf in key_frames:
        if 0 <= kf < total_seconds:
            labels[kf] = 1
    return labels


# ============================================================================
# Precompute per-video normalized dimensions (weight-independent)
# ============================================================================
def precompute_video_dims(video_id: str) -> Optional[Dict]:
    """
    For a single video: load data, build windows, compute normalized dims once.
    Returns None if the video should be skipped or data is missing.
    """
    asr_data = load_asr_data(video_id)
    if asr_data is None:
        print(f"  {video_id}: [SKIP] No ASR data found")
        return None

    results = load_results(video_id)
    if results is None:
        print(f"  {video_id}: [SKIP] No results.json found")
        return None

    words = asr_data.get("words", [])
    total_seconds = results.get("total_seconds", 0)
    word_count = len(words)

    print(f"  {video_id}: {total_seconds}s, {word_count} words")

    # Build ASR result compatible with build_text_windows
    asr_result = build_asr_from_words(words, total_seconds)

    # Build windowed text and tokens
    window_texts, window_tokens = build_text_windows(asr_result, total_seconds)

    # Compute 5-dim scores once (normalized dims are weight-independent)
    dim_scores = compute_five_dim_scores(window_texts, window_tokens)

    # Also build per-second text for potential use
    ts = asr_result.get("timeseries", [])
    per_second_text = [ts[i].get("text", "") if i < len(ts) else "" for i in range(total_seconds)]

    return {
        "video_id": video_id,
        "total_seconds": total_seconds,
        "word_count": word_count,
        "dim_norms": {
            "ttr_norm": dim_scores["ttr_norm"],
            "entropy_norm": dim_scores["entropy_norm"],
            "surprisal_norm": dim_scores["surprisal_norm"],
            "info_point_norm": dim_scores["info_point_norm"],
            "redundancy_norm": dim_scores["redundancy_norm"],
        },
        "window_texts": window_texts,
        "per_second_text": per_second_text,
    }


def discover_video_ids() -> List[str]:
    """Discover video IDs from the pilot output directory (subdirectories)."""
    ids = []
    if not PILOT_OUTPUT_DIR.exists():
        return ids
    for entry in sorted(PILOT_OUTPUT_DIR.iterdir()):
        if entry.is_dir() and entry.name.startswith("M-"):
            ids.append(entry.name)
    return ids


# ============================================================================
# Main
# ============================================================================
def main():
    print("Weight Grid Search (500 random + 2 baselines)")
    print("=" * 62)

    # ---- Discover videos and load annotations ----
    print("\nLoading annotation data...")
    all_video_ids = discover_video_ids()
    if not all_video_ids:
        print("  [ERROR] No video output directories found under pilot_output/")
        return

    annotation_videos = []
    missing_annotations = []
    for vid in all_video_ids:
        anno = load_annotation(vid)
        if anno is None:
            missing_annotations.append(vid)
        else:
            key_frames = anno.get("key_frames", [])
            annotation_videos.append((vid, key_frames))
            print(f"  {vid}: {len(key_frames)} key frames")

    if missing_annotations:
        print(f"\n  [INFO] Missing annotations for: {', '.join(missing_annotations)}")

    if not annotation_videos:
        print("\n  [ERROR] No annotation files found in annotations/. "
              "Please create annotations/{video_id}_keyframes.json files first.")
        print("  Expected format: {\"video_id\": \"...\", \"key_frames\": [3, 7, 15, ...]}")
        return

    # Filter to videos that have annotations AND are not visual-only
    valid_videos = [(vid, kf) for vid, kf in annotation_videos if vid not in SKIP_VIDEO_IDS]
    skipped_visual = [vid for vid, _ in annotation_videos if vid in SKIP_VIDEO_IDS]
    if skipped_visual:
        print(f"  [SKIP] Visual-only videos (no text density scoring): {', '.join(skipped_visual)}")

    if not valid_videos:
        print("\n  [ERROR] No valid videos after filtering. All annotated videos are visual-only.")
        return

    # ---- Precompute normalized dimensions for each video ----
    print("\nLoading ASR data and computing normalized dimensions...")
    video_data = []
    for vid, key_frames in valid_videos:
        dims = precompute_video_dims(vid)
        if dims is None:
            continue
        dims["key_frames"] = key_frames
        dims["labels"] = key_frames_to_labels(key_frames, dims["total_seconds"])
        n_key = len(key_frames)
        if n_key == 0:
            print(f"  {vid}: [WARN] 0 key frames annotated — skipping")
            continue
        video_data.append(dims)

    if not video_data:
        print("\n  [ERROR] No video data could be loaded. Check ASR cache and results.json files.")
        return

    # ---- Generate weight combinations ----
    print(f"\nGenerating weight combinations...")
    rng = np.random.default_rng(42)
    random_combos, current_weights, equal_weights = generate_weight_combos(NUM_RANDOM_COMBOS, rng)
    print(f"  Generated {len(random_combos)} random valid combos (weight ∈ [{WEIGHT_MIN}, {WEIGHT_MAX}])")
    print(f"  Current weights: TTR={current_weights['ttr']}, Ent={current_weights['entropy']}, "
          f"Surp={current_weights['surprisal']}, InfoP={current_weights['info_point']}, "
          f"Redun={current_weights['redundancy']}")
    print(f"  Equal weights: 0.20 each")

    # ---- Evaluate all weight combos ----
    print(f"\nEvaluating {len(random_combos)} random combos + 2 baselines on {len(video_data)} videos...")

    def evaluate_combo(weights: Dict[str, float]) -> Dict:
        """Evaluate a single weight combo across all videos."""
        ndcgs = []
        aucs = []
        for vd in video_data:
            comp = compute_composite_scores(vd["dim_norms"], weights)
            k = len(vd["key_frames"])
            ndcgs.append(ndcg_at_k(comp, vd["labels"], k))
            aucs.append(compute_auc_rank(comp, vd["labels"]))
        return {
            "weights": weights,
            "ndcg_per_video": ndcgs,
            "auc_per_video": aucs,
            "ndcg_mean": float(np.mean(ndcgs)),
            "auc_mean": float(np.mean(aucs)),
        }

    # Evaluate random combos
    results = []
    for i, combo in enumerate(random_combos):
        res = evaluate_combo(combo)
        results.append(res)
        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(random_combos)} evaluated")

    # Evaluate baselines
    current_result = evaluate_combo(current_weights)
    equal_result = evaluate_combo(equal_weights)

    # ---- Sort and report ----
    results.sort(key=lambda r: r["ndcg_mean"], reverse=True)

    print(f"\nTop-10 Weight Combinations (by mean NDCG):")
    print(f"{'Rank':<5} {'TTR':<7} {'Ent':<7} {'Surp':<7} {'InfoP':<7} {'Redun':<7} {'NDCG':<8} {'AUC':<8}")
    print(f"{'-----':<5} {'-------':<7} {'-------':<7} {'-------':<7} {'-------':<7} {'-------':<7} {'--------':<8} {'--------':<8}")

    for rank, res in enumerate(results[:10], 1):
        w = res["weights"]
        print(f"{rank:<5} "
              f"{w['ttr']:<7.4f} "
              f"{w['entropy']:<7.4f} "
              f"{w['surprisal']:<7.4f} "
              f"{w['info_point']:<7.4f} "
              f"{w['redundancy']:<7.4f} "
              f"{res['ndcg_mean']:<8.4f} "
              f"{res['auc_mean']:<8.4f}")

    best = results[0]

    # ---- Comparison with baselines ----
    print(f"\nComparison with Baselines:")
    print(f"  {'Metric':<12} {'Current':<12} {'Equal':<12} {'Best Random':<14} {'Improvement':<14}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*14} {'-'*14}")
    print(f"  {'NDCG':<12} {current_result['ndcg_mean']:<12.4f} {equal_result['ndcg_mean']:<12.4f} {best['ndcg_mean']:<14.4f} "
          f"{'+{:.1f}%'.format((best['ndcg_mean'] / max(current_result['ndcg_mean'], 1e-10) - 1) * 100):<14}")
    print(f"  {'AUC':<12} {current_result['auc_mean']:<12.4f} {equal_result['auc_mean']:<12.4f} {best['auc_mean']:<14.4f} "
          f"{'+{:.1f}%'.format((best['auc_mean'] / max(current_result['auc_mean'], 1e-10) - 1) * 100):<14}")

    # ---- Per-video breakdown for best ----
    print(f"\nPer-Video Breakdown (Best Combo):")
    print(f"  {'Video':<20} {'KeyFrames':<11} {'NDCG':<8} {'AUC':<8}")
    print(f"  {'-'*20} {'-'*11} {'-'*8} {'-'*8}")
    for i, vd in enumerate(video_data):
        print(f"  {vd['video_id']:<20} {len(vd['key_frames']):<11} "
              f"{best['ndcg_per_video'][i]:<8.4f} {best['auc_per_video'][i]:<8.4f}")

    # ---- Recommendation ----
    bw = best["weights"]
    print(f"\nRecommendation:")
    print(f"  WEIGHT_TTR          = {bw['ttr']:.4f}")
    print(f"  WEIGHT_ENTROPY      = {bw['entropy']:.4f}")
    print(f"  WEIGHT_SURPRISAL    = {bw['surprisal']:.4f}")
    print(f"  WEIGHT_INFO_POINT   = {bw['info_point']:.4f}")
    print(f"  WEIGHT_REDUNDANCY   = {bw['redundancy']:.4f}")
    print(f"  Mean NDCG: {best['ndcg_mean']:.4f}  |  Mean AUC: {best['auc_mean']:.4f}")

    # ---- Stats on weight distributions in top results ----
    print(f"\nWeight Distribution in Top-10 Combos (mean +/- std):")
    for dim_name, dim_label in zip(DIM_NAMES, DIM_LABELS):
        vals = [r["weights"][dim_name] for r in results[:10]]
        print(f"  {dim_label:<7}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
