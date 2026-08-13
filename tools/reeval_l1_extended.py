"""21 条扩展样本 L1 评估（生产协议，TS vs Uniform）.

背景：B-研究扩展了人工标注样本至 21 条（annotations_curated.json），但此前所有
L1/L2 评估仅覆盖 8 条 round1 视频。按 A1/C1 结论，L2 存在天花板效应、不具区分度，
L1（人工标注画面关键帧匹配）才是真正判别手段。本脚本用 21 条全量样本评估。

数据源：
  - 关键帧真值: pilot_output/annotations_curated.json（人工标注画面关键帧）
  - ASR 文本:   pilot_output/{vid}/annotation_data.json 的逐秒 text（重建 ASR 段）

评估协议（与 reeval_prod_protocol.py 对齐）：
  - 预算 = auto_budget(duration) = max(10, int(duration×0.7))
  - TS 帧 = select_frames_adaptive(budget_mode="increment") —— 生产算法原样
  - Uniform 帧 = 同帧数均匀采样（token 对齐）
  - L1 容差 ±1s，按 key 计数（一个 key 只命中一次）

纯本地计算，无需 API/并发。

用法: cd 视频降维技术研究 && PYTHONIOENCODING=utf-8 python tools/reeval_l1_extended.py
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
PILOT = ROOT / "pilot_output"
CURATED = PILOT / "annotations_curated.json"


def rebuild_segs(anno_data: dict) -> tuple[list[dict], int]:
    """从 annotation_data 的逐秒 text 重建 ASR 段（相邻 1s 内合并为一段）。"""
    secs = anno_data["seconds"]
    total = len(secs)
    segs: list[dict] = []
    cur = None
    for s in secs:
        t = (s.get("text") or "").strip()
        if not t:
            continue
        st = float(s["second"])
        if cur is None or st - cur["end"] > 1.0:
            cur = {"start": st, "end": st + 0.5, "text": t}
            segs.append(cur)
        else:
            cur["end"] = st + 0.5
            cur["text"] += t
    return segs, total


def f1_metrics(selected: list[int], key: list[int], tol: int = 1) -> dict:
    matched_keys = [hf for hf in key
                    if any(abs(sf - hf) <= tol for sf in selected)]
    matched = len(matched_keys)
    p = matched / len(selected) if selected else 0.0
    r = matched / len(key) if key else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"f1": f1, "precision": p, "recall": r, "n": len(selected),
            "matched": matched}


def main() -> None:
    curated = json.loads(CURATED.read_text(encoding="utf-8"))
    print(f"扩展样本: {len(curated)} 条\n")

    rows = []
    for v in sorted(curated, key=lambda x: x["video_id"]):
        vid = v["video_id"]
        anno_path = PILOT / vid / "annotation_data.json"
        if not anno_path.exists():
            print(f"!! {vid}: 缺 annotation_data，跳过")
            continue
        key = [k for k in v.get("key_frames", []) if k < v["total_seconds"]]

        segs, total = rebuild_segs(json.loads(anno_path.read_text(encoding="utf-8")))
        scores, _ = compute_density_scores(segs, total, return_dimensions=True)
        vtype = prejudge_video_type(segs, total)["video_type"]

        B = auto_budget(v["total_seconds"])
        ts = select_frames_adaptive(scores, B, total, video_type=vtype,
                                    budget_mode="increment")
        N = len(ts)
        uni = _uniform_sample(N, total)

        ts_m = f1_metrics(ts, key)
        uni_m = f1_metrics(uni, key)
        rows.append({
            "video": vid, "vtype": vtype, "priority": v.get("visual_priority"),
            "duration": v["total_seconds"], "budget": B, "n_ts": N,
            "key_n": len(key), "ts": ts_m, "uniform": uni_m,
            "delta": round(ts_m["f1"] - uni_m["f1"], 4),
        })

    print(f"{'video':<18} {'type':<5} {'prio':<8} {'dur':>4} {'B':>3} "
          f"{'N':>3} {'KF':>3} {'TS F1':>6} {'Uni F1':>7} {'Δ':>6}")
    print("-" * 74)
    ts_f1s, uni_f1s, deltas = [], [], []
    for r in rows:
        ts_f1s.append(r["ts"]["f1"]); uni_f1s.append(r["uniform"]["f1"])
        deltas.append(r["delta"])
        print(f"{r['video']:<18} {r['vtype']:<5} {str(r['priority']):<8} "
              f"{r['duration']:>4} {r['budget']:>3} {r['n_ts']:>3} "
              f"{r['key_n']:>3} {r['ts']['f1']:>6.3f} "
              f"{r['uniform']['f1']:>7.3f} {r['delta']:>+6.3f}")
    print("-" * 74)
    wins = sum(1 for d in deltas if d > 0)
    ties = sum(1 for d in deltas if d == 0)
    print(f"TS 平均 F1:      {statistics.fmean(ts_f1s):.3f}")
    print(f"Uniform 平均 F1: {statistics.fmean(uni_f1s):.3f}")
    print(f"Δ: {statistics.fmean(deltas):+.3f}  (胜 {wins}/{len(deltas)} 平 {ties})")

    # 按主导性分组
    for prio in ("speech", "visual", "balanced"):
        sub = [r for r in rows if r["priority"] == prio]
        if not sub:
            continue
        t = statistics.fmean(r["ts"]["f1"] for r in sub)
        u = statistics.fmean(r["uniform"]["f1"] for r in sub)
        d = statistics.fmean(r["delta"] for r in sub)
        print(f"[{prio:<8}] n={len(sub):>2}  TS F1={t:.3f}  Uni F1={u:.3f}  Δ={d:+.3f}")

    out = PILOT / "reeval_l1_extended.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存: {out}  ({len(rows)} 条)")


if __name__ == "__main__":
    main()
