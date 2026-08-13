"""改进后 TokenSaver 算法在 8 条研究视频上的 L1 重评估（固定预算 B=10/20）.

背景：研究阶段（算法改进前）的 L1 F1 是用旧算法跑的（纯 Top-K 选帧、汉字数字归一化
缺失、token 时间戳段中点、固定 B）。生产版算法改为「均匀采样基线 + 算法增量」两层结构。

本脚本用生产版算法的固定预算模式（budget_mode="total"，恰好返回 B 帧）在同一批
ASR + 人工标注上重算，并在相同预算 B 下对比纯均匀采样，隔离「算法增量」本身的贡献；
同时保留旧简化版 F1 作参照。

口径与旧 compare_simplified.py 一致：±1s 容差匹配（容差内算命中）。

用法: cd 视频降维技术研究 && python tools/reeval_improved.py
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
    select_frames_adaptive,
    _uniform_sample,
)

ROOT = Path(__file__).resolve().parent.parent
ASR_DIR = ROOT / "pilot_output" / "qwen_cache" / "asr"
ANN_DIR = ROOT / "annotations"
OLD_F1 = ROOT / "pilot_output" / "simplified_vs_full.json"

BUDGETS = [10, 20]


def load_asr_segments(path: Path) -> tuple[list[dict], float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    sentences = d.get("sentences") or []
    segs = [{"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in sentences if s.get("text")]
    duration = max(s["end"] for s in segs) if segs else 0.0
    return segs, duration


def load_annotation(path: Path) -> tuple[list[int], float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return list(d["key_frames"]), float(d["total_seconds"])


def f1_metrics(selected: list[int], key: list[int],
               tolerance: int = 1) -> dict:
    """容差匹配，与旧 compare_simplified.py 口径一致（±tolerance 秒内算命中）."""
    matched = 0
    for sf in selected:
        for hf in key:
            if abs(sf - hf) <= tolerance:
                matched += 1
                break
    p = matched / len(selected) if selected else 0.0
    r = matched / len(key) if key else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"f1": f1, "precision": p, "recall": r,
            "n_selected": len(selected), "n_matched": matched}


def main():
    old = json.loads(OLD_F1.read_text(encoding="utf-8"))
    old_f1 = {int(B): {v["video_id"]: v["ts_simp_f1"] for v in old[B]["per_video"]}
              for B in old}

    rows: list[dict] = []
    for asr_path in sorted(ASR_DIR.glob("*_asr.json")):
        vid = asr_path.name.replace("_qwen_asr.json", "").replace("_asr.json", "")
        ann_path = ANN_DIR / f"{vid}_keyframes.json"
        if not ann_path.exists():
            continue
        segs, duration = load_asr_segments(asr_path)
        key, _ = load_annotation(ann_path)

        scores, stats = compute_density_scores(segs, duration, return_dimensions=True)
        pj = prejudge_video_type(segs, duration)
        vtype = pj["video_type"]
        cv = statistics.pstdev(scores) / (statistics.fmean(scores) + 1e-10)
        # 帧空间由 ASR 时长决定（评分数组长度），标注仅提供关键帧
        total = len(scores)
        key = [k for k in key if k < total]

        for B in BUDGETS:
            ts = select_frames_adaptive(scores, B, total, video_type=vtype,
                                        budget_mode="total")
            uni = _uniform_sample(B, total)
            ts_m = f1_metrics(ts, key)
            uni_m = f1_metrics(uni, key)
            rows.append({
                "video": vid, "vtype": vtype, "cv": round(cv, 3),
                "duration": round(duration, 1), "key_n": len(key),
                "B": B, "ts": ts_m, "uniform": uni_m,
                "delta": round(ts_m["f1"] - uni_m["f1"], 4),
                "old_simp_f1": old_f1.get(B, {}).get(vid),
                "dim_range": None,
            })

    # 维度区分度（5D norm_std 极差）每视频取一次
    seen_vids = set()
    for r in rows:
        if r["video"] in seen_vids:
            continue
        seen_vids.add(r["video"])
        for asr_path in sorted(ASR_DIR.glob("*_asr.json")):
            if r["video"] in asr_path.name:
                segs, dur = load_asr_segments(asr_path)
                _, st = compute_density_scores(segs, dur, return_dimensions=True)
                dvals = [st["dimensions"][k]["norm_std"] for k in
                         ["surprisal", "info_density", "entropy", "anti_redundancy", "ttr"]]
                r["dim_range"] = round(max(dvals) - min(dvals), 3)
                break

    for B in BUDGETS:
        b_rows = [r for r in rows if r["B"] == B]
        print(f"\n{'='*76}\nB = {B}\n{'='*76}")
        hdr = (f"{'video':<16} {'type':<7} {'dur':>5} {'CV':>5} {'关键帧':>4} "
               f"{'TS F1':>6} {'Unif F1':>7} {'Δ':>6} {'旧简化F1':>7}")
        print(hdr); print("-" * len(hdr))
        for r in b_rows:
            oldv = f"{r['old_simp_f1']:.3f}" if r["old_simp_f1"] is not None else "-"
            print(f"{r['video']:<16} {r['vtype']:<7} {r['duration']:>5.1f} {r['cv']:>5.2f} "
                  f"{r['key_n']:>4} {r['ts']['f1']:>6.3f} {r['uniform']['f1']:>7.3f} "
                  f"{r['delta']:>+6.3f} {oldv:>7}")
        print("-" * len(hdr))
        ts_f1s = [r["ts"]["f1"] for r in b_rows]
        uni_f1s = [r["uniform"]["f1"] for r in b_rows]
        deltas = [r["delta"] for r in b_rows]
        old_f1s = [r["old_simp_f1"] for r in b_rows if r["old_simp_f1"] is not None]
        wins = sum(1 for d in deltas if d > 0)
        ties = sum(1 for d in deltas if d == 0)
        print(f"  TS 平均 F1:      {statistics.fmean(ts_f1s):.3f}")
        print(f"  Uniform 平均 F1: {statistics.fmean(uni_f1s):.3f}")
        print(f"  算法增量 Δ:      {statistics.fmean(deltas):+.3f}  "
              f"(胜 {wins}/{len(deltas)} 平 {ties})")
        print(f"  旧简化版 F1(参照): {statistics.fmean(old_f1s):.3f}  "
              f"| TS vs 旧: {statistics.fmean(ts_f1s) - statistics.fmean(old_f1s):+.3f}")

    dims = [r["dim_range"] for r in rows if r["dim_range"] is not None]
    print(f"\n维度区分度（5D norm_std 极差中位）: {statistics.median(dims):.3f}")

    out = ROOT / "pilot_output" / "reeval_improved.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()
