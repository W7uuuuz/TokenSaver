"""生产协议 L1 评估：研究口径向生产靠拢.

新协议（2026-08-06，用户方向）：
  - 对比预算 = auto_budget(duration) = max(10, int(duration×0.7))，随视频时长自适应
  - TS 帧 = select_frames_adaptive(budget_mode="increment") —— 生产算法原样，
    即「稀疏均匀骨架（oral 4s/review 2s，保叙事连贯）+ 重点采集（文本高密度关键帧）」
  - Uniform 帧 = 同帧数均匀采样（token 对齐）
  - L1 容差 ±1s 对比

动机：固定预算 B=10/20 低于覆盖阈值（oral 4s/帧需 duration/4 帧），重分配被迫
裁剪骨架、破坏叙事连贯，L1 输是物理限制而非算法缺陷。生产预算（auto_budget）
下骨架 + 增量自然构成，覆盖零牺牲，算法价值真实显现。

用法: cd 视频降维技术研究 && python tools/reeval_prod_protocol.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

BACKEND = Path(r"D:\DevWorkSpace\HotMirrorStudio\backend")
sys.path.insert(0, str(BACKEND))
from app.video.token_saver import (  # noqa: E402
    auto_budget,
    compute_density_scores,
    prejudge_video_type,
    select_frames_adaptive,
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


def f1_metrics(selected: list[int], key: list[int], tol: int = 1) -> dict:
    # 按 key 计数：一个 key 只算命中一次，避免多 selected 帧命中同 key 虚高 recall
    matched_keys = [hf for hf in key
                    if any(abs(sf - hf) <= tol for sf in selected)]
    matched = len(matched_keys)
    p = matched / len(selected) if selected else 0.0
    r = matched / len(key) if key else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"f1": f1, "precision": p, "recall": r, "n": len(selected),
            "matched": matched}


def main() -> None:
    rows = []
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
        key = [k for k in key if k < total]

        B = auto_budget(duration)
        ts = select_frames_adaptive(scores, B, total, video_type=vtype,
                                    budget_mode="increment")
        N = len(ts)
        uni = _uniform_sample(N, total)

        ts_m = f1_metrics(ts, key)
        uni_m = f1_metrics(uni, key)
        rows.append({
            "video": vid, "vtype": vtype, "duration": round(duration, 1),
            "budget": B, "n_ts": N, "key_n": len(key),
            "ts": ts_m, "uniform": uni_m,
            "delta": round(ts_m["f1"] - uni_m["f1"], 4),
        })

    print(f"{'video':<16} {'type':<7} {'dur':>5} {'B':>3} {'N(TS)':>5} "
          f"{'关键帧':>4} {'TS F1':>6} {'Unif F1':>7} {'Δ':>6}")
    print("-" * 66)
    ts_f1s, uni_f1s, deltas = [], [], []
    for r in rows:
        ts_f1s.append(r["ts"]["f1"]); uni_f1s.append(r["uniform"]["f1"])
        deltas.append(r["delta"])
        print(f"{r['video']:<16} {r['vtype']:<7} {r['duration']:>5.1f} {r['budget']:>3} "
              f"{r['n_ts']:>5} {r['key_n']:>4} {r['ts']['f1']:>6.3f} "
              f"{r['uniform']['f1']:>7.3f} {r['delta']:>+6.3f}")
    print("-" * 66)
    wins = sum(1 for d in deltas if d > 0)
    ties = sum(1 for d in deltas if d == 0)
    print(f"TS 平均 F1:      {statistics.fmean(ts_f1s):.3f}")
    print(f"Uniform 平均 F1: {statistics.fmean(uni_f1s):.3f}")
    print(f"Δ: {statistics.fmean(deltas):+.3f}  (胜 {wins}/{len(deltas)} 平 {ties})")

    out = ROOT / "pilot_output" / "reeval_prod_protocol.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()
