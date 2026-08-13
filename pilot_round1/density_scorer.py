"""
Stage 2: Five-dimensional text density scoring + routing + coupling detection.
Pure CPU, deterministic, no neural inference required.
"""
import re
import math
from collections import Counter
from typing import List, Dict, Tuple, Optional
import numpy as np

from . import config


# ============================================================================
# Jieba initialization (lazy)
# ============================================================================
_jieba_loaded = False


def _ensure_jieba():
    global _jieba_loaded
    if not _jieba_loaded:
        import jieba
        jieba.setLogLevel(20)  # suppress log spam
        _jieba_loaded = True
    import jieba
    return jieba


# ============================================================================
# Dimension 1: Lexical Density (Type-Token Ratio)
# ============================================================================
def compute_ttr(tokens: List[str], window_size: int = 3) -> float:
    """Type-Token Ratio with moving-average smoothing over window_size windows."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / max(len(tokens), 1)


def compute_ttr_smoothed(all_tokens: List[List[str]], window_size: int = 3) -> List[float]:
    """Compute TTR per window, smoothed with moving average."""
    raw = [compute_ttr(t) for t in all_tokens]
    smoothed = []
    for i in range(len(raw)):
        start = max(0, i - window_size // 2)
        end = min(len(raw), i + window_size // 2 + 1)
        smoothed.append(np.mean(raw[start:end]) if raw[start:end] else 0.0)
    return smoothed


# ============================================================================
# Dimension 2: Information Entropy H(W)
# ============================================================================
def compute_entropy(tokens: List[str]) -> float:
    """Shannon entropy H(W) = -sum P(w) log2 P(w), normalized by log2(|W|)."""
    if not tokens:
        return 0.0
    n = len(tokens)
    counts = Counter(tokens)
    probs = [c / n for c in counts.values()]
    H = -sum(p * math.log2(p) for p in probs if p > 0)
    # Normalize
    max_H = math.log2(n) if n > 1 else 1.0
    return H / max_H if max_H > 0 else 0.0


# ============================================================================
# Dimension 3: Surprisal (trigram backoff model)
# ============================================================================
class TrigramSurprisal:
    """Trigram language model with Laplace backoff smoothing."""

    def __init__(self, alpha: float = None):
        self.alpha = alpha or config.TRIGRAM_FALLBACK_ALPHA
        self.trigram_counts: Dict[Tuple[str, str, str], int] = {}
        self.bigram_counts: Dict[Tuple[str, str], int] = {}
        self.unigram_counts: Dict[str, int] = {}
        self.total_bigrams = 0
        self.V = 0

    def fit(self, token_sequences: List[List[str]]):
        """Build n-gram counts from token sequences."""
        for tokens in token_sequences:
            padded = ["<s>", "<s>"] + tokens + ["</s>"]
            for w in padded:
                self.unigram_counts[w] = self.unigram_counts.get(w, 0) + 1
            for i in range(len(padded) - 1):
                bigram = (padded[i], padded[i + 1])
                self.bigram_counts[bigram] = self.bigram_counts.get(bigram, 0) + 1
                self.total_bigrams += 1
            for i in range(len(padded) - 2):
                trigram = (padded[i], padded[i + 1], padded[i + 2])
                self.trigram_counts[trigram] = self.trigram_counts.get(trigram, 0) + 1
        self.V = len(self.unigram_counts)

    def prob(self, w3: str, w1: str, w2: str) -> float:
        """P(w3 | w1, w2) with trigram → bigram → unigram backoff + Laplace smoothing."""
        tri_count = self.trigram_counts.get((w1, w2, w3), 0)
        bi_count = self.bigram_counts.get((w1, w2), 0)

        # Trigram with Laplace smoothing
        tri_prob = (tri_count + self.alpha) / (bi_count + self.alpha * self.V) if bi_count > 0 else None

        if tri_prob is not None and bi_count > 0:
            return tri_prob

        # Bigram backoff
        bi_count2 = self.bigram_counts.get((w2, w3), 0)
        uni_count = self.unigram_counts.get(w2, 0)
        bi_prob = (bi_count2 + self.alpha) / (uni_count + self.alpha * self.V) if uni_count > 0 else None

        if bi_prob is not None:
            return bi_prob

        # Unigram backoff
        return (self.unigram_counts.get(w3, 0) + self.alpha) / (self.total_bigrams + self.alpha * self.V)

    def surprisal(self, tokens: List[str]) -> float:
        """Average surprisal: (1/|W|) sum -log2 P(w_i | w_{i-2}, w_{i-1})."""
        if len(tokens) < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(2, len(tokens)):
            p = self.prob(tokens[i], tokens[i - 2], tokens[i - 1])
            total += -math.log2(max(p, 1e-10))
            count += 1
        return total / count if count > 0 else 0.0


# ============================================================================
# Dimension 4: Information-Point Density
# ============================================================================
def compile_entity_patterns() -> List[re.Pattern]:
    return [re.compile(p) for p in config.ENTITY_PATTERNS]

_entity_regexes: Optional[List[re.Pattern]] = None


def compute_info_point_density(text: str) -> float:
    """Entities per 100 characters via e-commerce lexicon + regex."""
    global _entity_regexes
    if _entity_regexes is None:
        _entity_regexes = compile_entity_patterns()

    if not text or len(text) == 0:
        return 0.0

    match_count = 0
    for pat in _entity_regexes:
        match_count += len(pat.findall(text))

    return (match_count / len(text)) * 100  # entities per 100 chars


# ============================================================================
# Dimension 5: Redundancy (inverted)
# ============================================================================
def compute_redundancy_inverted(tokens: List[str]) -> float:
    """1 - (filler_ratio + bigram_repetition_rate)."""
    if not tokens:
        return 0.0

    n = len(tokens)

    # Filler ratio
    filler_count = sum(1 for t in tokens if t in config.FILLER_WORDS)
    filler_ratio = filler_count / n

    # Bigram repetition rate
    if n >= 2:
        bigrams = [tuple(tokens[i:i + 2]) for i in range(n - 1)]
        bigram_counter = Counter(bigrams)
        repeated = sum(c - 1 for c in bigram_counter.values() if c > 1)
        bigram_rep_rate = repeated / len(bigrams) if bigrams else 0.0
    else:
        bigram_rep_rate = 0.0

    redundancy = filler_ratio + bigram_rep_rate
    return max(0.0, 1.0 - redundancy)


# ============================================================================
# Score Aggregation
# ============================================================================
def normalize_column(values: List[float]) -> List[float]:
    """Min-max normalize to [0, 1], handling constant columns."""
    arr = np.array(values, dtype=float)
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-10:
        return [0.5] * len(values)  # constant → neutral
    return ((arr - vmin) / (vmax - vmin)).tolist()


def compute_five_dim_scores(
    window_texts: List[str],
    window_tokens: List[List[str]],
    trigram_model: Optional[TrigramSurprisal] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, List[float]]:
    """
    Compute all 5 dimensions for each window.
    Returns dict with keys: ttr, entropy, surprisal, info_point, redundancy, composite.

    Args:
        window_texts: Per-second text windows.
        window_tokens: Per-second token lists.
        trigram_model: Pre-fitted TrigramSurprisal model (optional).
        weights: Optional dict with keys 'ttr', 'entropy', 'surprisal',
                 'info_point', 'redundancy'. If None, uses config defaults.
    """
    n = len(window_texts)

    # Resolve weights: use provided or fall back to config defaults
    if weights is not None:
        w_ttr = weights.get("ttr", config.WEIGHT_TTR)
        w_entropy = weights.get("entropy", config.WEIGHT_ENTROPY)
        w_surprisal = weights.get("surprisal", config.WEIGHT_SURPRISAL)
        w_info_point = weights.get("info_point", config.WEIGHT_INFO_POINT)
        w_redundancy = weights.get("redundancy", config.WEIGHT_REDUNDANCY)
    else:
        w_ttr = config.WEIGHT_TTR
        w_entropy = config.WEIGHT_ENTROPY
        w_surprisal = config.WEIGHT_SURPRISAL
        w_info_point = config.WEIGHT_INFO_POINT
        w_redundancy = config.WEIGHT_REDUNDANCY

    # Fit trigram model on all tokens if not provided
    if trigram_model is None:
        trigram_model = TrigramSurprisal()
        trigram_model.fit(window_tokens)

    # Raw scores
    ttr_raw = compute_ttr_smoothed(window_tokens)
    entropy_raw = [compute_entropy(t) for t in window_tokens]
    surprisal_raw = [trigram_model.surprisal(t) for t in window_tokens]
    ipd_raw = [compute_info_point_density(txt) for txt in window_texts]
    redundancy_raw = [compute_redundancy_inverted(t) for t in window_tokens]

    # Normalize each dimension
    ttr_norm = normalize_column(ttr_raw)
    entropy_norm = normalize_column(entropy_raw)
    surprisal_norm = normalize_column(surprisal_raw)
    ipd_norm = normalize_column(ipd_raw)
    redundancy_norm = normalize_column(redundancy_raw)

    # Weighted composite score (0-10 scale)
    composite = []
    for i in range(n):
        score = (
            w_ttr * ttr_norm[i]
            + w_entropy * entropy_norm[i]
            + w_surprisal * surprisal_norm[i]
            + w_info_point * ipd_norm[i]
            + w_redundancy * redundancy_norm[i]
        )
        composite.append(round(score * 10, 2))

    return {
        "ttr_raw": ttr_raw,
        "ttr_norm": ttr_norm,
        "entropy_raw": entropy_raw,
        "entropy_norm": entropy_norm,
        "surprisal_raw": surprisal_raw,
        "surprisal_norm": surprisal_norm,
        "info_point_raw": ipd_raw,
        "info_point_norm": ipd_norm,
        "redundancy_raw": redundancy_raw,
        "redundancy_norm": redundancy_norm,
        "composite": composite,
    }


# ============================================================================
# Routing
# ============================================================================
def get_routing_category(score: float) -> Tuple[str, int, str]:
    """Map composite score to routing category."""
    for threshold, category, fpm, strategy in config.ROUTING_TABLE:
        if score >= threshold:
            return category, fpm, strategy
    # Fallback (shouldn't reach)
    return config.ROUTING_TABLE[-1][1], config.ROUTING_TABLE[-1][2], config.ROUTING_TABLE[-1][3]


def apply_global_local_routing(
    composite_scores: List[float],
    video_duration_sec: float,
    scene_boundaries: List[int] = None,
) -> List[Dict]:
    """
    Apply global-local hierarchical routing.
    For videos shorter than MIN_DURATION_LOCAL_ROUTING, uses video-level routing only
    (per-second local routing is meaningless on short content).
    Returns per-second routing decisions: [{second, global_cat, local_cat, effective_fpm, trigger}].
    """
    n = len(composite_scores)
    if scene_boundaries is None:
        scene_boundaries = []

    # Step A: Global coarse
    global_avg = np.mean(composite_scores)
    global_cat, global_fpm, _ = get_routing_category(global_avg)

    # Short video: skip per-second local routing
    use_local_routing = video_duration_sec >= config.MIN_DURATION_LOCAL_ROUTING

    routing_decisions = []
    for sec in range(n):
        if use_local_routing:
            local_start = max(0, sec - int(config.LOCAL_WINDOW_SEC) // 2)
            local_end = min(n, sec + int(config.LOCAL_WINDOW_SEC) // 2 + 1)
            local_scores = composite_scores[local_start:local_end]
            local_avg = np.mean(local_scores) if local_scores else global_avg

            if abs(local_avg - global_avg) > config.LOCAL_GLOBAL_DIFF_THRESHOLD:
                effective_score = local_avg
            else:
                effective_score = global_avg
            overridden = abs(local_avg - global_avg) > config.LOCAL_GLOBAL_DIFF_THRESHOLD
        else:
            # Short video: use global routing for every second
            effective_score = global_avg
            local_avg = global_avg
            overridden = False

        local_cat, effective_fpm, strategy = get_routing_category(effective_score)

        routing_decisions.append({
            "second": sec,
            "global_avg": round(global_avg, 2),
            "local_avg": round(local_avg, 2),
            "effective_score": round(effective_score, 2),
            "global_cat": global_cat,
            "local_cat": local_cat,
            "effective_fpm": effective_fpm,
            "strategy": strategy,
            "is_scene_boundary": sec in scene_boundaries,
            "overridden": overridden,
            "short_video_mode": not use_local_routing,
        })

    return routing_decisions


# ============================================================================
# Coupling Detection
# ============================================================================
def detect_lexical_coupling(
    text_per_second: List[str],
    visual_variance: List[float],
) -> List[int]:
    """
    Lexical Coupling Detection (LCD):
    Find instruction words → check for visual discontinuity nearby → flag as coupling.
    Returns list of coupling flags per second.
    """
    n = len(text_per_second)
    coupling_flags = [0] * n

    # Build flat text index for keyword search
    for sec in range(n):
        text = text_per_second[sec]
        if not text:
            continue

        for category, words in config.INSTRUCTION_WORD_LEXICON.items():
            for kw in words:
                if kw not in text:
                    continue

                # Found keyword at second `sec`
                # Scan for visual change in [sec - 2, sec + 3]
                scan_start = max(0, int(sec - config.COUPLING_SCAN_WINDOW_BEFORE_SEC))
                scan_end = min(n, int(sec + config.COUPLING_SCAN_WINDOW_AFTER_SEC + 1))

                # Check if any 2s window in scan range has high visual variance
                found_visual_change = False
                for t in range(scan_start, scan_end - 1):
                    if t < n and visual_variance[t] > 0.3:
                        found_visual_change = True
                        break

                if found_visual_change:
                    # Flag coupling in [sec - 1, sec + 2]
                    flag_start = max(0, sec - 1)
                    flag_end = min(n, sec + 3)
                    for t in range(flag_start, flag_end):
                        coupling_flags[t] = 1
                    break  # one keyword match is enough
            else:
                continue
            break

    return coupling_flags


def detect_structural_coupling(
    text_density_norm: List[float],
    visual_variance: List[float],
) -> List[int]:
    """
    Structural Coupling Detection (SCD):
    SCD_score(t) = E_text(t)_norm × σ_vis(t)
    High text density + high visual variance → implicit coupling.
    """
    n = min(len(text_density_norm), len(visual_variance))
    coupling_flags = [0] * n

    for t in range(n):
        scd = text_density_norm[t] * visual_variance[t]
        if scd > config.SCD_TEXT_VISUAL_THRESHOLD:
            coupling_flags[t] = 1

    return coupling_flags


def combine_coupling_flags(lcd_flags: List[int], scd_flags: List[int]) -> List[int]:
    """OR-combine LCD and SCD flags."""
    return [1 if (l or s) else 0 for l, s in zip(lcd_flags, scd_flags)]
