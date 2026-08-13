import math
import re
from collections import Counter

import jieba
import numpy as np

# ── Constants ──

ENTITY_PATTERNS: list[str] = [
    r'¥\d+(?:\.\d+)?',
    r'\d+(?:\.\d+)?多?元',
    r'\d+(?:\.\d+)?多?折',
    r'\d+(?:\.\d+)?多?[份件个瓶盒袋箱套种只支]',
    r'\d+[gG克千][克gG]?',
    r'\d+[mM毫][升L升lL]',
    r'\d+(?:\.\d+)?多?%',
    r'\d+(?:\.\d+)?多?(?:秒|分钟|小时|天|日|周|月|年)',
    # 近似数（"一千多"→"1000多"），但后跟单位时不重复计数（如"1000多元"只由"多?元"命中）
    r'\d+(?:\.\d+)?多(?![元折%秒分钟小时天日周月年份件个瓶盒袋箱套种只支])',
    r'第[一二三四五六七八九十\d]+[名步位]',
    # 半单位（半小时/半分钟/半年）
    r'半(?:秒|分钟|小时|天|日|周|月|年)',
]

# ── 汉字数字归一化 ──
# ASR 转录（中文 Whisper）常把数字写成汉字（如"三十秒""百分之九十九点九九九"），
# 实体正则只认阿拉伯数字，需先归一化再匹配。

_CN_DIGITS = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
              '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
_CN_UNITS = {'十': 10, '百': 100, '千': 1000}
_CN_NUM_CHARS = '零一二三四五六七八九十百千万亿两'


def _cn_int(s: str) -> int:
    """'九十九' -> 99，'一百二十三' -> 123."""
    total = 0
    section = 0
    num = 0
    for ch in s:
        if ch in _CN_DIGITS:
            num = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            if ch == '十' and num == 0:
                num = 1  # '十三' 中的 '十' 表示 10
            section += num * _CN_UNITS[ch]
            num = 0
        elif ch == '万':
            total += (section + num) * 10000
            section = 0
            num = 0
        elif ch == '亿':
            total = (total + section + num) * 100000000
            section = 0
            num = 0
    return total + section + num


def _cn_body_to_arabic(s: str) -> str:
    """'九十九' -> '99'，'九十九点九九九' -> '99.999'."""
    if '点' in s:
        int_part, _, frac_part = s.partition('点')
        int_str = str(_cn_int(int_part)) if int_part else '0'
        frac_str = ''.join(str(_CN_DIGITS[c]) for c in frac_part if c in _CN_DIGITS)
        return f"{int_str}.{frac_str}" if frac_str else int_str
    return str(_cn_int(s))


def _normalize_numerals(text: str) -> str:
    """把文本中的汉字数字归一化为阿拉伯数字，供实体正则匹配.

    - '百分之九十九点九九九' -> '99.999%'
    - '三十秒' -> '30秒'，'一百多种' -> '100多种'
    """
    text = re.sub(
        rf'百分之([{_CN_NUM_CHARS}]+(?:点[{_CN_NUM_CHARS}]+)?)',
        lambda m: _cn_body_to_arabic(m.group(1)) + '%',
        text,
    )
    text = re.sub(
        rf'[{_CN_NUM_CHARS}]+点[{_CN_NUM_CHARS}]+|[{_CN_NUM_CHARS}]+',
        lambda m: _cn_body_to_arabic(m.group(0)),
        text,
    )
    return text

FILLER_WORDS: set[str] = {
    '就是', '然后', '那个', '这个', '嗯', '啊', '呃', '吧',
    '嘛', '呢', '的话', '就是说', '其实', '所以', '因此',
    '那么', '而且', '还有', '或者', '但是', '不过', '呃就是',
}


# ── SurprisalModel ──

class SurprisalModel:
    """Trigram backoff language model with Laplace smoothing α=0.1."""

    def __init__(self, segments: list[dict], alpha: float = 0.1) -> None:
        self.alpha = alpha

        all_tokens: list[str] = []
        for seg in segments:
            all_tokens.extend(jieba.lcut(seg["text"]))

        self.all_tokens = all_tokens
        self.vocab_size = len(set(all_tokens))
        self.total_tokens = len(all_tokens)

        self.unigram_counts: Counter = Counter(all_tokens)
        self.bigram_counts: Counter = Counter()
        self.trigram_counts: Counter = Counter()

        for i in range(len(all_tokens) - 1):
            self.bigram_counts[(all_tokens[i], all_tokens[i + 1])] += 1
        for i in range(len(all_tokens) - 2):
            self.trigram_counts[(all_tokens[i], all_tokens[i + 1], all_tokens[i + 2])] += 1

    def _unigram_prob(self, w: str) -> float:
        V = self.vocab_size
        a = self.alpha
        return (self.unigram_counts.get(w, 0) + a) / (self.total_tokens + a * V)

    def _bigram_prob(self, w1: str, w2: str) -> float:
        V = self.vocab_size
        a = self.alpha
        bigram_count = self.bigram_counts.get((w1, w2), 0)
        context_count = self.unigram_counts.get(w1, 0)
        if context_count > 0:
            return (bigram_count + a) / (context_count + a * V)
        return self._unigram_prob(w2)

    def _trigram_prob(self, w1: str, w2: str, w3: str) -> float:
        V = self.vocab_size
        a = self.alpha
        trigram_count = self.trigram_counts.get((w1, w2, w3), 0)
        context_count = self.bigram_counts.get((w1, w2), 0)
        if context_count > 0:
            return (trigram_count + a) / (context_count + a * V)
        return self._bigram_prob(w2, w3)

    def score_window(self, tokens: list[str]) -> float:
        """Average surprisal (bits) for all tokens in the window."""
        if not tokens:
            return 0.0

        total = 0.0
        for i, token in enumerate(tokens):
            if i >= 2:
                prob = self._trigram_prob(tokens[i - 2], tokens[i - 1], token)
            elif i == 1:
                prob = self._bigram_prob(tokens[0], token)
            else:
                prob = self._unigram_prob(token)

            prob = max(prob, 1e-10)
            total += -math.log2(prob)

        return total / len(tokens)


# ── Public Functions ──

def prejudge_video_type(segments: list[dict], duration: float) -> dict:
    """Classify video as 'oral' or 'review' based on pause ratio.

    Pause ratio = total seconds of wordless gaps (>1.5s between segments) / duration.

    Returns dict with keys: video_type, pause_ratio, total_gap, segment_count, gap_count.
    """
    if not segments or duration <= 0:
        return {"video_type": "oral", "pause_ratio": 0.0,
                "total_gap": 0.0, "segment_count": 0, "gap_count": 0}

    sorted_segs = sorted(segments, key=lambda s: s["start"])
    total_gap = 0.0
    gap_count = 0
    for i in range(len(sorted_segs) - 1):
        gap = sorted_segs[i + 1]["start"] - sorted_segs[i]["end"]
        if gap > 1.5:
            total_gap += gap
            gap_count += 1

    pause_ratio = total_gap / duration
    return {
        "video_type": "oral" if pause_ratio < 0.15 else "review",
        "pause_ratio": round(pause_ratio, 4),
        "total_gap": round(total_gap, 1),
        "segment_count": len(segments),
        "gap_count": gap_count,
    }


def compute_density_scores(
    segments: list[dict],
    duration: float,
    window: float = 5.0,
    stride: float = 1.0,
    return_dimensions: bool = False,
) -> list[float] | tuple[list[float], dict]:
    """Compute per-second 5D composite information density scores.

    If return_dimensions=True, returns (scores, dim_stats) where dim_stats
    contains per-dimension raw statistics for visualization.

    Returns list[float] with length = ceil(duration), one score per second.
    """
    n_seconds = math.ceil(duration)

    # Build (token, timestamp) pairs — token 在 ASR 段内均匀分布。
    # 粗粒度 ASR 分段（段长可达数十秒）若用段中点，token 全聚在中心 5s 窗口，
    # 其余时间全是空窗口，minmax 归一化后形成 0/1 双峰导致 norm_std 虚高。
    token_times: list[tuple[str, float]] = []
    for seg in segments:
        tokens = jieba.lcut(seg["text"])
        seg_span = seg["end"] - seg["start"]
        n = len(tokens)
        for j, t in enumerate(tokens):
            tm = seg["start"] + seg_span * (j + 0.5) / n
            token_times.append((t, tm))

    if not token_times:
        if return_dimensions:
            empty_dims = {
                "dimensions": {dim: {"weight": 0.0, "name": "", "raw_mean": 0.0,
                 "raw_std": 0.0, "norm_mean": 0.0, "norm_std": 0.0, "contribution": 0.0}
                 for dim in ["surprisal", "info_density", "entropy", "anti_redundancy", "ttr"]},
                "composite_mean": 0.0, "composite_std": 0.0,
            }
            return [0.0] * n_seconds, empty_dims
        return [0.0] * n_seconds

    model = SurprisalModel(segments)
    entity_regexes = [re.compile(p) for p in ENTITY_PATTERNS]

    num_windows = max(1, math.ceil((duration - window) / stride) + 1)

    # Raw dimension scores per window
    raw: list[dict] = []

    for wi in range(num_windows):
        t_start = wi * stride
        t_end = t_start + window
        center = int(t_start + window / 2)

        w_tokens = [tok for tok, tm in token_times if t_start <= tm < t_end]

        if not w_tokens:
            raw.append({"center": center, "surprisal": 0.0, "info_density": 0.0,
                         "entropy": 0.0, "anti_redundancy": 0.0, "ttr": 0.0})
            continue

        n_tok = len(w_tokens)
        tok_counts = Counter(w_tokens)
        n_unique = len(tok_counts)

        # 1. Surprisal (weight 0.40)
        surprisal = model.score_window(w_tokens)

        # 2. Information point density (weight 0.31)
        text = _normalize_numerals("".join(w_tokens))
        entity_hits = sum(len(rx.findall(text)) for rx in entity_regexes)
        info_density = entity_hits / n_tok

        # 3. Shannon entropy (weight 0.12)
        entropy = 0.0
        for c in tok_counts.values():
            p = c / n_tok
            entropy -= p * math.log2(p)
        max_entropy = math.log2(n_unique) if n_unique > 1 else 1.0
        norm_entropy = entropy / max_entropy

        # 4. Anti-redundancy (weight 0.10)
        filler_hits = sum(1 for t in w_tokens if t in FILLER_WORDS)
        filler_ratio = filler_hits / n_tok
        repeated = sum(c for c in tok_counts.values() if c > 1)
        repeat_ratio = repeated / n_tok
        anti_redundancy = 1.0 - (filler_ratio + repeat_ratio) / 2

        # 5. TTR (weight 0.07)
        ttr = n_unique / n_tok

        raw.append({"center": center, "surprisal": surprisal, "info_density": info_density,
                     "entropy": norm_entropy, "anti_redundancy": anti_redundancy, "ttr": ttr})

    # Min-max normalize each dimension across all windows
    def minmax(values: list[float]) -> list[float]:
        mn = min(values)
        mx = max(values)
        if mx == mn:
            return [0.5] * len(values)
        return [(v - mn) / (mx - mn) for v in values]

    dims = ["surprisal", "info_density", "entropy", "anti_redundancy", "ttr"]
    weights = {"surprisal": 0.40, "info_density": 0.31, "entropy": 0.12,
               "anti_redundancy": 0.10, "ttr": 0.07}

    normalized = {dim: minmax([r[dim] for r in raw]) for dim in dims}

    # Composite score → center mapping
    center_scores: dict[int, float] = {}
    for i, r in enumerate(raw):
        composite = sum(weights[dim] * normalized[dim][i] for dim in dims)
        center_scores[r["center"]] = composite

    # Nearest-neighbor interpolation to per-second array
    centers = sorted(center_scores.keys())
    result = [0.0] * n_seconds
    if centers:
        for sec in range(n_seconds):
            nearest = min(centers, key=lambda c: abs(c - sec))
            result[sec] = center_scores[nearest]

    if not return_dimensions:
        return result

    # Per-dimension raw stats for visualization
    dim_weights = {"surprisal": 0.40, "info_density": 0.31, "entropy": 0.12,
                   "anti_redundancy": 0.10, "ttr": 0.07}
    dim_names = {"surprisal": "惊奇度", "info_density": "信息点密度", "entropy": "香农熵",
                 "anti_redundancy": "反冗余", "ttr": "TTR"}
    dim_stats: dict[str, dict] = {}
    for dim in dims:
        vals = [r[dim] for r in raw]
        n = len(vals)
        mean_v = sum(vals) / n if n > 0 else 0.0
        std_v = (sum((x - mean_v) ** 2 for x in vals) / n) ** 0.5 if n > 0 else 0.0
        norm_vals = normalized[dim]
        norm_mean = sum(norm_vals) / n if n > 0 else 0.0
        norm_std = (sum((x - norm_mean) ** 2 for x in norm_vals) / n) ** 0.5 if n > 0 else 0.0
        dim_stats[dim] = {
            "weight": dim_weights[dim],
            "name": dim_names[dim],
            "raw_mean": round(mean_v, 4),
            "raw_std": round(std_v, 4),
            "norm_mean": round(norm_mean, 4),
            "norm_std": round(norm_std, 4),
            "contribution": round(dim_weights[dim] * norm_std, 4),
        }

    return result, {
        "dimensions": dim_stats,
        "composite_mean": round(sum(result) / len(result), 4) if result else 0.0,
        "composite_std": round((sum((x - sum(result) / len(result)) ** 2 for x in result) / len(result)) ** 0.5, 4) if result else 0.0,
    }


def auto_budget(duration: float) -> int:
    """Auto-compute frame budget B = duration × 0.7, floor 10."""
    return max(10, int(duration * 0.7))


# ── Adaptive Frame Selection (CV-based self-check) ──


def _uniform_sample(budget: int, total_seconds: int) -> list[int]:
    """等间距均匀采样，始终包含首尾帧."""
    if budget <= 0:
        return []
    if budget >= total_seconds:
        return list(range(total_seconds))
    if budget == 1:
        return [0]
    step = (total_seconds - 1) / (budget - 1)
    # round 而非 int：int 的浮点截断会让末帧丢失（如 59/29*29=58.999… → 58）
    return sorted(set(
        min(round(step * i), total_seconds - 1) for i in range(budget)
    ))


def _top_k_with_gap(scores: list[float], budget: int, total_seconds: int,
                    gap_mult: int = 2) -> list[int]:
    """按评分降序选帧，最小间距约束防止聚类.

    gap_mult 控制分散度：min_gap = total // (budget × gap_mult)。
    gap_mult=2 在固定预算模式下聚类浪费预算，调大 gap_mult（关键帧更分散）
    能显著改善 L1 帧匹配，对生产 increment 模式（默认 gap_mult=2）无影响。
    """
    if budget >= total_seconds:
        return list(range(total_seconds))

    arr = np.array(scores, dtype=float)
    arr[0] = max(arr[0], arr.max() * 1.1)
    arr[-1] = max(arr[-1], arr.max() * 1.1)

    min_gap = max(1, total_seconds // (budget * gap_mult))
    ranked = sorted(range(total_seconds), key=lambda t: arr[t], reverse=True)

    selected: set[int] = set()
    for t in ranked:
        if len(selected) >= budget:
            break
        if all(abs(t - s) >= min_gap for s in selected):
            selected.add(t)

    return sorted(selected)


def _fixed_budget_selection(
    scores: list[float],
    budget: int,
    total_seconds: int,
    low_cv: float,
    high_cv: float,
    blend_ratio: float,
    video_type: str,
    score_ratio: float = 0.5,
    gap_mult: int = 2,
) -> list[int]:
    """固定总预算模式：返回恰好 budget 帧.

    机制仍是「均匀基线 + 算法增量」：先铺 B 个均匀基线保证覆盖骨架，再叠加 Top-K
    关键帧，超过 B 时按评分升序裁掉冗余帧——净效果是把低密度区间的均匀帧向高密度
    区间重分配。评分平坦（低 CV）时退化为纯均匀采样。

    score_ratio 控制重分配力度，存在 L1（画面关键帧匹配）与 L2（内容理解）的
    本质张力——重分配强则 L2 好（内容理解靠文本高密度帧）但 L1 差（人工标注关键帧
    大量落在文本平淡的低评分区）；重分配弱则反之。折中取 0.5。
    """
    if budget <= 0:
        return []
    if budget >= total_seconds:
        return list(range(total_seconds))
    if budget == 1:
        return [0]

    arr = np.array(scores, dtype=float)
    cv = float(np.std(arr) / (np.mean(arr) + 1e-10))
    if cv < low_cv:
        return _uniform_sample(budget, total_seconds)

    baseline = set(_uniform_sample(budget, total_seconds))

    if cv > high_cv:
        n_key = max(1, int(budget * score_ratio))
    else:
        n_key = max(1, int(budget * blend_ratio))

    key_frames = set(_top_k_with_gap(scores, n_key, total_seconds, gap_mult))
    merged = sorted(baseline | key_frames)

    if len(merged) > budget:
        protected = {0, total_seconds - 1}
        droppable = sorted(
            (t for t in merged if t not in protected),
            key=lambda t: arr[t],
        )
        excess = len(merged) - budget
        drop_set = set(droppable[:excess])
        merged = [t for t in merged if t not in drop_set]

    return merged


def select_frames_adaptive(
    composite_scores: list[float],
    budget: int,
    total_seconds: int,
    low_cv: float = 0.15,
    high_cv: float = 0.22,
    blend_ratio: float = 0.6,
    video_type: str = "oral",
    budget_mode: str = "increment",
) -> list[int]:
    """根据评分分布的变异系数(CV)自动选择采样策略.

    - `budget_mode="increment"`（生产默认）: 均匀基线由覆盖率驱动（口播每4s一帧，
      产品每2s一帧），不受预算裁剪。budget 仅约束关键帧密度。
    - `budget_mode="total"`（固定预算实验）: 返回恰好 budget 帧，机制为「均匀基线 +
      算法增量」再裁剪到 B，用于与研究口径（固定 B 对比）对齐。

    - CV < low_cv  → 纯均匀采样
    - CV > high_cv → 均匀基线 + 评分选择关键帧
    - 灰色地带    → 均匀基线 + 混合关键帧

    Returns 选中帧索引（秒，0-indexed，升序）.
    """
    if total_seconds <= 0:
        return []

    if budget_mode == "total":
        return _fixed_budget_selection(
            composite_scores, budget, total_seconds,
            low_cv, high_cv, blend_ratio, video_type,
        )

    # 均匀基线 — 覆盖率驱动，不受预算限制
    uniform_interval = 4 if video_type == "oral" else 2
    uniform_count = max(1, int(total_seconds / uniform_interval) + 1)
    baseline = set(_uniform_sample(uniform_count, total_seconds))

    if budget <= 0:
        return sorted(baseline)

    arr = np.array(composite_scores, dtype=float)
    cv = float(np.std(arr) / (np.mean(arr) + 1e-10))

    if cv < low_cv:
        # 纯均匀：基线已够用，不追加关键帧
        result = sorted(baseline)
        return _fill_coverage_gaps(result, total_seconds)

    # 关键帧预算 — 仅用于高密度区域
    if cv > high_cv:
        score_ratio = 0.7 if video_type == "oral" else 0.5
        n_key = max(1, int(budget * score_ratio))
    else:
        n_key = max(1, int(budget * blend_ratio))

    key_frames = set(_top_k_with_gap(composite_scores, n_key, total_seconds))

    # 关键帧叠加到均匀基线上，min_gap 防止聚类
    min_gap = max(1, total_seconds // (uniform_count + n_key))
    selected = set(baseline)
    for t in sorted(key_frames):
        if all(abs(t - s) >= min_gap for s in selected):
            selected.add(t)

    result = sorted(selected)
    return _fill_coverage_gaps(result, total_seconds)


def transitions_first(
    diffs: list[float],
    budget: int,
    total_seconds: int,
    threshold: float = 0.28,
) -> list[int]:
    """转镜优先 + 均匀兜底：预算内（≤N 帧）重分配取帧.

    转镜帧（画面帧差 > threshold，通常承载动作/证据/列举，复刻必需）按帧差降序
    优先占额度；不足 budget 用均匀采样兜底保证叙事覆盖。因与转镜帧去重损耗，
    最终帧数略小于 budget（实测比纯 TS 更省帧），分布均匀。

    TS 取帧会结构性丢失一闪而过的转镜帧（人工目检确认其为重要帧），union 方案
    使帧数膨胀 1.5×；改为预算内重分配后 L1 关键帧召回 0.927→0.982，L2 理解评估
    确认视觉理解不退化。threshold=0.28 为逐秒 HSV Bhattacharyya 帧差档位的最优值。
    """
    if budget <= 0:
        return []
    if budget >= total_seconds:
        return list(range(total_seconds))

    scene = [(i, d) for i, d in enumerate(diffs) if d > threshold]
    scene.sort(key=lambda x: -x[1])  # 帧差降序，转镜最明显优先
    frames = [i for i, _ in scene[:budget]]
    if len(frames) < budget:
        need = budget - len(frames)
        for f in _uniform_sample(need, total_seconds):
            if len(frames) >= budget:
                break
            if f not in frames:
                frames.append(f)
    return sorted(frames)


def _fill_coverage_gaps(
    selected: list[int],
    total_seconds: int,
    max_gap: float | None = None,
) -> list[int]:
    """后处理：确保任意两个选中帧之间的间距不超过 max_gap 秒.

    在超过阈值的间隙中插入均匀帧，保证全片视觉覆盖率。
    不影响已有选中帧，纯粹做增量补帧。
    """
    if not selected or total_seconds <= 0:
        return list(selected)

    if max_gap is None:
        max_gap = 4.0

    filled: set[int] = set(selected)
    # 确保首尾帧始终存在
    filled.add(0)
    filled.add(total_seconds - 1)

    # 迭代填充，直到所有间隙 ≤ max_gap
    changed = True
    while changed:
        changed = False
        sorted_sel = sorted(filled)
        for i in range(len(sorted_sel) - 1):
            gap_start = sorted_sel[i]
            gap_end = sorted_sel[i + 1]
            gap_size = gap_end - gap_start

            if gap_size > max_gap:
                n_insert = int(gap_size / max_gap)
                step = gap_size / (n_insert + 1)
                for j in range(1, n_insert + 1):
                    t = int(gap_start + step * j)
                    if t > gap_start and t < gap_end and t not in filled:
                        filled.add(t)
                        changed = True

    return sorted(filled)


def select_frames_priority(
    composite_scores: list[float] | None,
    diffs: list[float] | None,
    budget: int,
    total_seconds: int,
    video_type: str = "oral",
    threshold: float = 0.28,
    low_cv: float = 0.15,
    high_cv: float = 0.22,
) -> list[int]:
    """三源融合选帧：均匀骨架（不占预算）→ 转镜帧（预算内优先）→ 算法帧（剩余预算）.

    优先级：
      1. 均匀骨架 — 覆盖率驱动（口播 4s/帧、产品 2s/帧），不消耗预算且必须全部保留，
         保证全片视觉覆盖与首末帧；
      2. 转镜帧 — 画面帧差 > threshold 的秒按帧差降序优先占预算；
      3. 算法帧 — 剩余预算给 TokenSaver 密度评分 Top-K（评分平坦低 CV 时跳过）。

    budget 仅约束「增强帧」（转镜 + 算法）总量，均匀骨架不计入。

    transitions_first 在转镜帧 ≥ 预算时不执行均匀兜底，导致首末帧缺失；
    本函数以均匀骨架先行规避该问题。
    Returns 选中帧索引（秒，升序）.
    """
    if total_seconds <= 0:
        return []

    uniform_interval = 4 if video_type == "oral" else 2
    uniform_count = max(1, int(total_seconds / uniform_interval) + 1)
    selected = set(_uniform_sample(uniform_count, total_seconds))

    if budget <= 0:
        return sorted(selected)

    enhance = 0  # 已消耗的增强帧额度（转镜 + 算法）

    if diffs:
        scene = sorted(
            ((i, d) for i, d in enumerate(diffs) if d > threshold),
            key=lambda x: -x[1],
        )
        for i, _ in scene:
            if enhance >= budget:
                break
            if i not in selected:
                selected.add(i)
                enhance += 1

    if composite_scores and enhance < budget:
        arr = np.array(composite_scores, dtype=float)
        cv = float(np.std(arr) / (np.mean(arr) + 1e-10))
        if cv >= low_cv:
            n_key = budget - enhance
            for t in _top_k_with_gap(composite_scores, n_key, total_seconds):
                if enhance >= budget:
                    break
                if t not in selected:
                    selected.add(t)
                    enhance += 1

    return sorted(selected)
