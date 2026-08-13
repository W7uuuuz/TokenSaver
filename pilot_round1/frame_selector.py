"""
Stage 3: Budget-aware MMR frame selection with temporal diversity penalty
and coupling override.
"""
import math
from typing import List, Tuple, Set, Optional
import numpy as np
from dataclasses import dataclass

from . import config


@dataclass
class CandidateFrame:
    """A candidate frame with its metadata for selection."""
    second: int
    importance_text: float       # I_text(t)
    importance_visual: float     # I_visual(t)
    coupling_flag: int           # 0 or 1
    importance_total: float      # I_text * (1 + boost*flag) + I_visual
    is_scene_boundary: bool
    is_first: bool = False
    is_last: bool = False


def compute_importance(
    composite_text_scores: List[float],
    visual_motion: List[float],
    visual_entropy: List[float],
    object_counts: List[float],
    coupling_flags: List[int],
) -> List[float]:
    """
    Compute total importance score for each second:
    I(f) = I_text(t) * (1 + coupling_boost * CouplingFlag(t)) + I_visual(t)
    """
    n = len(composite_text_scores)
    w_h, w_m, w_o = config.VISUAL_SIM_WEIGHTS

    # Normalize visual features
    def _norm(arr):
        a = np.array(arr, dtype=float)
        a_min, a_max = a.min(), a.max()
        if a_max - a_min < 1e-10:
            return np.ones_like(a) * 0.5
        return (a - a_min) / (a_max - a_min)

    motion_norm = _norm(visual_motion)
    entropy_norm = _norm(visual_entropy)
    object_norm = _norm(object_counts)

    importance = []
    for t in range(n):
        text_score = composite_text_scores[t] / 10.0  # scale back to [0, 1]
        coupling = coupling_flags[t] if t < len(coupling_flags) else 0

        i_text = text_score * (1 + config.COUPLING_BOOST * coupling)
        i_visual = (
            w_h * entropy_norm[t]
            + w_m * motion_norm[min(t, len(motion_norm) - 1)]
            + w_o * object_norm[min(t, len(object_norm) - 1)]
        )
        importance.append(i_text + i_visual)

    return importance


def compute_similarity(t1: int, t2: int, hsv_histograms: Optional[List[np.ndarray]] = None) -> float:
    """
    Compute frame similarity:
    sim(f, s) = exp(-|t_f - t_s| / tau) * (1 - D_vis(f, s))
    """
    temporal_dist = abs(t1 - t2)
    temporal_sim = math.exp(-temporal_dist / config.MMR_TEMPORAL_TAU)

    if hsv_histograms is not None and t1 < len(hsv_histograms) and t2 < len(hsv_histograms):
        h1, h2 = hsv_histograms[t1], hsv_histograms[t2]
        if h1 is not None and h2 is not None:
            # Cosine similarity between histograms
            h1_flat = h1.flatten().astype(float)
            h2_flat = h2.flatten().astype(float)
            dot = np.dot(h1_flat, h2_flat)
            norm = np.linalg.norm(h1_flat) * np.linalg.norm(h2_flat)
            if norm > 1e-10:
                hsv_sim = dot / norm
            else:
                hsv_sim = 0.0
            return temporal_sim * (1 - hsv_sim)

    return temporal_sim  # fallback: pure temporal decay


def mmr_select(
    candidates: List[CandidateFrame],
    budget_k: int,
    hsv_histograms: Optional[List[np.ndarray]] = None,
    lambda_rel: Optional[float] = None,
) -> List[int]:
    """
    MMR with temporal diversity penalty and coupling override.
    Returns sorted list of selected frame seconds.

    Args:
        candidates: List of candidate frames with pre-computed importance scores.
        budget_k: Number of frames to select.
        hsv_histograms: Per-second HSV histograms for visual similarity.
        lambda_rel: Relevance-vs-diversity weight [0, 1].
                    Higher = more relevance, lower = more diversity.
                    Defaults to config.MMR_LAMBDA_REL if not provided.
    """
    if budget_k <= 0 or not candidates:
        return []

    if lambda_rel is None:
        lambda_rel = config.MMR_LAMBDA_REL

    # Always-included frames: only first and last are truly mandatory.
    # Scene boundaries are important but must compete in MMR — otherwise
    # they consume the entire budget at low B, starving coupling frames.
    mandatory: Set[int] = set()
    for c in candidates:
        if c.is_first or c.is_last:
            mandatory.add(c.second)
    # Scene boundaries get a 20% importance boost to compete fairly in MMR
    for c in candidates:
        if c.is_scene_boundary and c.second not in mandatory:
            c.importance_total *= 1.2

    # If mandatory frames already fill budget, return them
    if len(mandatory) >= budget_k:
        return sorted(mandatory)[:budget_k]

    # Seed: highest importance frame
    remaining = [c for c in candidates if c.second not in mandatory]
    best_first = max(remaining, key=lambda c: c.importance_total)

    selected: Set[int] = mandatory | {best_first.second}
    remaining = [c for c in remaining if c.second != best_first.second]

    # Greedy MMR
    while len(selected) < budget_k and remaining:
        best_score = float("-inf")
        best_candidate = None

        for c in remaining:
            # Compute max similarity to already-selected frames
            max_sim = 0.0
            for s_sec in selected:
                sim = compute_similarity(c.second, s_sec, hsv_histograms)
                # Coupling override: disable temporal penalty for coupling-critical frames
                if c.coupling_flag == 1:
                    sim = 0.0
                max_sim = max(max_sim, sim)

            mmr_score = (
                lambda_rel * c.importance_total
                - (1 - lambda_rel) * max_sim
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_candidate = c

        if best_candidate is None:
            break

        selected.add(best_candidate.second)
        remaining = [c for c in remaining if c.second != best_candidate.second]

    return sorted(selected)


def build_candidates(
    n_seconds: int,
    coupling_flags: List[int],
    composite_text_scores: List[float],
    visual_entropy: List[float],
    visual_motion: List[float],
    object_counts: List[float],
    scene_boundaries: List[int],
) -> List[CandidateFrame]:
    """Build CandidateFrame list for all seconds."""
    w_h, w_m, w_o = config.VISUAL_SIM_WEIGHTS

    candidates = []
    for t in range(n_seconds):
        i_text = composite_text_scores[t] / 10.0 if t < len(composite_text_scores) else 0.0
        coupling = coupling_flags[t] if t < len(coupling_flags) else 0

        # Visual importance
        h_norm = visual_entropy[t] if t < len(visual_entropy) else 0.0
        m_norm = visual_motion[t] if t < len(visual_motion) else 0.0
        o_norm = object_counts[t] if t < len(object_counts) else 0.0

        i_visual = w_h * h_norm + w_m * m_norm + w_o * o_norm
        i_total = i_text * (1 + config.COUPLING_BOOST * coupling) + i_visual

        candidates.append(CandidateFrame(
            second=t,
            importance_text=i_text,
            importance_visual=i_visual,
            coupling_flag=coupling,
            importance_total=i_total,
            is_scene_boundary=t in scene_boundaries,
            is_first=(t == 0),
            is_last=(t == n_seconds - 1),
        ))

    return candidates


def select_frames_multi_budget(
    candidates: List[CandidateFrame],
    budgets: List[int],
    hsv_histograms: Optional[List[np.ndarray]] = None,
    lambda_rel: Optional[float] = None,
) -> dict:
    """Run MMR selection at multiple budget levels and return combined results.

    Args:
        candidates: List of candidate frames.
        budgets: Budget levels to select at.
        hsv_histograms: Per-second HSV histograms for visual similarity.
        lambda_rel: Optional lambda override for relevance-vs-diversity.
    """
    results = {}
    for k in budgets:
        selected = mmr_select(candidates, k, hsv_histograms, lambda_rel=lambda_rel)
        results[k] = selected
    return results
