"""L1 调优网格扫描：固定预算模式的重分配力度与关键帧分散度.

诊断结论：TS 在 B=20 时把预算过度重分配到文本高评分区，而人工标注的关键帧
大量落在低评分区（画面关键、文本平淡），导致 TS 系统性丢命中。此外
_top_k_with_gap 的 min_gap 太松（total//(budget*2)，B=20 时=1），高分区聚类
浪费预算（如 M-14507 B=10 独有命中 t=40,41,42 相邻簇）。

本脚本参数化复刻固定预算选择逻辑，扫描两个旋钮：
  - gap_mult: min_gap = total // (budget*gap_mult)，越大关键帧越分散
  - score_ratio: 高 CV 时关键帧占预算比例，越小越接近均匀（保住覆盖率）

输出每组合 B=10/B=20 的 8 条平均 L1 F1（容差1s），与 Uniform 对比。

用法: cd 视频降维技术研究 && python tools/sweep_l1.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

BACKEND = Path(r"D:\DevWorkSpace\HotMirrorStudio\backend")
sys.path.insert(0, str(BACKEND))
from app.video.token_saver import (  # noqa: E402
    compute_density_scores,
    prejudge_video_type,
    _uniform_sample,
)

ROOT = Path(__file__).resolve().parent.parent
ASR_DIR = ROOT / "pilot_output" / "qwen_cache" / "asr"
ANN_DIR = ROOT / "annotations"


def load_asr(path: Path) -> tuple[list[dict], float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    sentences = d.get("sentences") or []
    segs = [{"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in sentences if s.get("text")]
    duration = max(s["end"] for s in segs) if segs else 0.0
    return segs, duration


def load_ann(path: Path) -> list[int]:
    return list(json.loads(path.read_text(encoding="utf-8"))["key_frames"])


def f1(selected: list[int], key: list[int], tol: int = 1) -> float:
    matched = sum(1 for sf in selected if any(abs(sf - hf) <= tol for hf in key))
    p = matched / len(selected) if selected else 0.0
    r = matched / len(key) if key else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def top_k_with_gap(scores: list[float], budget: int, total: int, gap_mult: int) -> list[int]:
    if budget >= total:
        return list(range(total))
    arr = scores[:]
    mx = max(arr) * 1.1
    arr[0] = max(arr[0], mx)
    arr[-1] = max(arr[-1], mx)
    min_gap = max(1, total // (budget * gap_mult))
    ranked = sorted(range(total), key=lambda t: arr[t], reverse=True)
    sel: set[int] = set()
    for t in ranked:
        if len(sel) >= budget:
            break
        if all(abs(t - s) >= min_gap for s in sel):
            sel.add(t)
    return sorted(sel)


def fixed_budget(scores: list[float], budget: int, total: int, video_type: str,
                 low_cv: float = 0.15, high_cv: float = 0.22,
                 blend_ratio: float = 0.6, score_ratio: float = 0.7,
                 gap_mult: int = 2) -> list[int]:
    if budget <= 0:
        return []
    if budget >= total:
        return list(range(total))
    if budget == 1:
        return [0]
    mean = statistics.fmean(scores)
    cv = statistics.pstdev(scores) / (mean + 1e-10)
    if cv < low_cv:
        return _uniform_sample(budget, total)
    baseline = set(_uniform_sample(budget, total))
    n_key = max(1, int(budget * score_ratio)) if cv > high_cv \
        else max(1, int(budget * blend_ratio))
    key = set(top_k_with_gap(scores, n_key, total, gap_mult))
    merged = sorted(baseline | key)
    if len(merged) > budget:
        protected = {0, total - 1}
        droppable = sorted((t for t in merged if t not in protected),
                           key=lambda t: scores[t])
        excess = len(merged) - budget
        drop = set(droppable[:excess])
        merged = [t for t in merged if t not in drop]
    return merged


def main() -> None:
    vids = []
    for asr_path in sorted(ASR_DIR.glob("*_asr.json")):
        vid = asr_path.name.replace("_qwen_asr.json", "").replace("_asr.json", "")
        ann_path = ANN_DIR / f"{vid}_keyframes.json"
        if not ann_path.exists():
            continue
        segs, duration = load_asr(asr_path)
        key = load_ann(ann_path)
        scores, _ = compute_density_scores(segs, duration, return_dimensions=True)
        vtype = prejudge_video_type(segs, duration)["video_type"]
        total = len(scores)
        vids.append({"vid": vid, "vtype": vtype, "scores": scores,
                     "key": [k for k in key if k < total], "total": total})

    grid = []
    for gap_mult in [2, 3, 4, 6]:
        for score_ratio in [0.3, 0.5, 0.7]:
            grid.append({"gap_mult": gap_mult, "score_ratio": score_ratio})

    print(f"{'gap':>3} {'ratio':>5} | " + "  ".join(
        f"B={b} TS/Unif Δ" for b in [10, 20]))
    print("-" * 58)

    best = {}
    for g in grid:
        row = []
        for B in [10, 20]:
            ts_f1s, uni_f1s = [], []
            for v in vids:
                ts = fixed_budget(v["scores"], B, v["total"], v["vtype"],
                                  score_ratio=g["score_ratio"], gap_mult=g["gap_mult"])
                uni = _uniform_sample(B, v["total"])
                ts_f1s.append(f1(ts, v["key"]))
                uni_f1s.append(f1(uni, v["key"]))
            ts_m = statistics.fmean(ts_f1s)
            uni_m = statistics.fmean(uni_f1s)
            row.append((ts_m, uni_m, ts_m - uni_m))
            for B_key, (t, u, d) in [(B, row[-1])]:
                best.setdefault(B_key, []).append((g, t, u, d))
        gap, ratio = g["gap_mult"], g["score_ratio"]
        cells = "  ".join(f"{t:.3f}/{u:.3f} {d:+.3f}" for t, u, d in row)
        print(f"{gap:>3} {ratio:>5.1f} | {cells}")

    print("\nB=10 最优 5 组合（按 Δ 降序）:")
    for g, t, u, d in sorted(best[10], key=lambda x: -x[3])[:5]:
        print(f"  gap={g['gap_mult']} ratio={g['score_ratio']}: TS {t:.3f} vs Uni {u:.3f} Δ{d:+.3f}")
    print("B=20 最优 5 组合:")
    for g, t, u, d in sorted(best[20], key=lambda x: -x[3])[:5]:
        print(f"  gap={g['gap_mult']} ratio={g['score_ratio']}: TS {t:.3f} vs Uni {u:.3f} Δ{d:+.3f}")


if __name__ == "__main__":
    main()
