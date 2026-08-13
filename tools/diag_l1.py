"""L1 落后原因诊断：逐视频对比 TS vs Uniform 的命中差异.

核心问题：固定预算下 TS 的 L1 F1 为何系统性 ≤ Uniform？
诊断输出三类证据：
  1. 每视频 B=10/20 命中统计（公共命中 / TS 独有 / Uniform 独有）
  2. Uniform 独有命中的关键帧：其评分与文本 → 若普遍低分，说明评分与人工标注错位
  3. TS 选帧的平均评分 vs Uniform → 确认 TS 确实在"重分配向高评分"，但命中为何下降

用法: cd 视频降维技术研究 && python tools/diag_l1.py [--video M-xxx] [--B 10]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

BACKEND = Path(r"D:\DevWorkSpace\HotMirrorStudio\backend")
sys.path.insert(0, str(BACKEND))
from app.video.token_saver import (  # noqa: E402
    compute_density_scores,
    prejudge_video_type,
    select_frames_adaptive,
    _uniform_sample,
)

ROOT = Path(__file__).resolve().parent.parent
ASR_DIR = ROOT / "pilot_output" / "qwen_cache" / "asr"
ANN_DIR = ROOT / "annotations"
BUDGETS = [10, 20]
TOL = 1


def load_asr(path: Path) -> tuple[list[dict], float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    sentences = d.get("sentences") or []
    segs = [{"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in sentences if s.get("text")]
    duration = max(s["end"] for s in segs) if segs else 0.0
    return segs, duration


def load_ann(path: Path) -> list[int]:
    return list(json.loads(path.read_text(encoding="utf-8"))["key_frames"])


def text_at(segs: list[dict], t: float) -> str:
    for s in segs:
        if s["start"] - 1 <= t <= s["end"] + 1:
            return s["text"]
    return ""


def hit_mask(selected: list[int], keys: list[int], tol: int = TOL) -> set[int]:
    """返回被 selected 命中的 key 集合."""
    hit = set()
    for k in keys:
        if any(abs(s - k) <= tol for s in selected):
            hit.add(k)
    return hit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--B", type=int, default=None)
    args = ap.parse_args()

    budget_list = [args.B] if args.B else BUDGETS
    video_filter = args.video or ""

    agg_ts_only = {B: [] for B in BUDGETS}
    agg_uni_only = {B: [] for B in BUDGETS}
    agg_ts_hit = {B: [] for B in BUDGETS}
    agg_uni_hit = {B: [] for B in BUDGETS}

    for asr_path in sorted(ASR_DIR.glob("*_asr.json")):
        vid = asr_path.name.replace("_qwen_asr.json", "").replace("_asr.json", "")
        if video_filter and video_filter not in vid:
            continue
        ann_path = ANN_DIR / f"{vid}_keyframes.json"
        if not ann_path.exists():
            continue
        segs, duration = load_asr(asr_path)
        key = load_ann(ann_path)
        scores, stats = compute_density_scores(segs, duration, return_dimensions=True)
        vtype = prejudge_video_type(segs, duration)["video_type"]
        cv = statistics.pstdev(scores) / (statistics.fmean(scores) + 1e-10)
        total = len(scores)
        key = [k for k in key if k < total]

        print(f"\n{'='*86}\n{vid}  ({vtype})  {duration:.0f}s  cv={cv:.3f}  关键帧={len(key)}")
        print(f"{'='*86}")

        for B in budget_list:
            ts = select_frames_adaptive(scores, B, total, video_type=vtype,
                                        budget_mode="total")
            uni = _uniform_sample(B, total)
            ts_hit = hit_mask(ts, key)
            uni_hit = hit_mask(uni, key)
            both = ts_hit & uni_hit
            ts_only = ts_hit - uni_hit
            uni_only = uni_hit - ts_hit

            ts_avg_s = statistics.fmean([scores[t] for t in ts])
            uni_avg_s = statistics.fmean([scores[t] for t in uni])
            ts_avg_key_s = statistics.fmean([scores[k] for k in ts_hit])
            uni_avg_key_s = statistics.fmean([scores[k] for k in uni_hit])

            agg_ts_hit[B].append(len(ts_hit))
            agg_uni_hit[B].append(len(uni_hit))
            agg_ts_only[B].append(len(ts_only))
            agg_uni_only[B].append(len(uni_only))

            print(f"\n  B={B}: TS 命中 {len(ts_hit)}/{len(key)} vs Uniform 命中 {len(uni_hit)}/{len(key)}"
                  f"  (公共 {len(both)}, TS独有 {len(ts_only)}, Uni独有 {len(uni_only)})")
            print(f"    选帧平均评分  TS={ts_avg_s:.3f} Uni={uni_avg_s:.3f}   "
                  f"命中帧平均评分 TS={ts_avg_key_s:.3f} Uni={uni_avg_key_s:.3f}")

            if uni_only:
                print(f"    ── Uniform 独有命中 {len(uni_only)} 个关键帧（TS 漏掉）──")
                for k in sorted(uni_only):
                    txt = text_at(segs, k)[:42]
                    print(f"       t={k:<3d} 评分={scores[k]:.3f}  文本: {txt}")
            if ts_only:
                print(f"    ── TS 独有命中 {len(ts_only)} 个关键帧（Uniform 漏掉）──")
                for k in sorted(ts_only):
                    txt = text_at(segs, k)[:42]
                    print(f"       t={k:<3d} 评分={scores[k]:.3f}  文本: {txt}")

    print(f"\n{'='*86}\n汇总（命中关键帧数，越大越好）")
    for B in BUDGETS:
        print(f"  B={B}: TS 命中 {statistics.fmean(agg_ts_hit[B]):.1f} vs Uni {statistics.fmean(agg_uni_hit[B]):.1f}  "
              f"(TS独有 {statistics.fmean(agg_ts_only[B]):.1f}, Uni独有 {statistics.fmean(agg_uni_only[B]):.1f})")


if __name__ == "__main__":
    main()
