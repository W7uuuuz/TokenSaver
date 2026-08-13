"""主实验最终分析：成本-效果表（input/image token 成本轴 + 保持率）. 只读不跑 API.

用法: PYTHONIOENCODING=utf-8 python tools/analyze_cost_effect.py [--file reeval_l2_cost_effect21.json]
默认读 21 条扩展主实验；传 --file reeval_l2_cost_effect.json 可看 8 条参考。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pilot_output" / "reeval_l2"

ap = argparse.ArgumentParser()
ap.add_argument("--file", default="reeval_l2_cost_effect21.json")
args = ap.parse_args()
OUT_FILE = OUT_DIR / args.file

data = json.loads(OUT_FILE.read_text(encoding="utf-8"))
videos = data["videos"]
print(f"样本: {len(videos)} 条")
for vid in sorted(videos):
    print("  ", vid, f"total={videos[vid]['total']}")

print("\n" + "=" * 92)
print("最终成本-效果表（成本轴 = input token，确定性驱动；image token = 纯帧成本）")
print("=" * 92)
for qset, qname in (("content", "内容理解(+ASR) — 主证降本"), ("visual", "纯视觉(无ASR) — 副证增效")):
    print(f"\n【{qname}】")
    print(f"  {'方法':<8}{'档位':<6}{'帧数':>5}{'input':>7}{'image':>7}"
          f"{'input/全帧':>10}{'保持率':>8}")
    for m in ("ts", "tf", "uniform"):
        for r in (0.5, 0.7, 0.9):
            pts = [p for v in videos.values() for p in v["points"]
                   if p["qset"] == qset and p["method"] == m and p["ratio"] == r]
            if not pts:
                continue
            n = sum(p["n_frames"] for p in pts) / len(pts)
            it = sum(p["usage"].get("input_tokens", 0) for p in pts) / len(pts)
            im = sum((p["usage"].get("input_tokens_details") or {})
                     .get("image_tokens", 0) for p in pts) / len(pts)
            keep = sum(p["judge"]["retention"] for p in pts) / len(pts)
            b_it = sum(v["baselines"][qset]["usage"].get("input_tokens", 0)
                       for v in videos.values()) / len(videos)
            print(f"  {m:<8}{r:<6}{n:>5.0f}{it:>7.0f}{im:>7.0f}{it/b_it*100:>9.0f}%{keep:>8.3f}")
    itb = sum(v["baselines"][qset]["usage"].get("input_tokens", 0)
              for v in videos.values()) / len(videos)
    imb = sum((v["baselines"][qset]["usage"].get("input_tokens_details") or {})
              .get("image_tokens", 0) for v in videos.values()) / len(videos)
    nb = sum(v["total"] for v in videos.values()) / len(videos)
    print(f"  {'fullframe':<8}{'—':<6}{nb:>5.0f}{itb:>7.0f}{imb:>7.0f}"
          f"{100:>9.0f}%{1.0:>8.3f}")

# TF vs TS 纯视觉逐视频胜平负（稳健性）
print("\n" + "=" * 60)
print("纯视觉保持率 TF vs TS 逐视频胜平负（稳健性检查）")
print("=" * 60)
tf_w = ts_w = tie = 0
detail = []
for vid, v in sorted(videos.items()):
    for r in (0.5, 0.7, 0.9):
        def keep(m, rr):
            return next(p["judge"]["retention"] for p in v["points"]
                        if p["qset"] == "visual" and p["method"] == m
                        and p["ratio"] == rr)
        tf, ts = keep("tf", r), keep("ts", r)
        if tf > ts:
            tf_w += 1
        elif tf < ts:
            ts_w += 1
        else:
            tie += 1
        detail.append((vid[:12], r, ts, tf))
print(f"TF>TS {tf_w} | TS>TF {ts_w} | 平 {tie}   (共 {tf_w+ts_w+tie} 组比较)")
print("分档:")
for r in (0.5, 0.7, 0.9):
    r_pts = [x for x in detail if x[1] == r]
    wt = sum(1 for x in r_pts if x[3] > x[2])
    wts = sum(1 for x in r_pts if x[3] < x[2])
    tt = sum(1 for x in r_pts if x[3] == x[2])
    print(f"  r={r}: TF>TS {wt} | TS>TF {wts} | 平 {tt}")
