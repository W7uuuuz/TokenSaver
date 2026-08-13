"""R1 主实验样本扩展：8 → 21 条（annotations_curated 全集）.

背景（2026-08-07）：R1 主实验只有 8 条（round1/videos，mp4 抽帧 + 独立 qwen ASR），
作为论文主实验样本偏小 → 扩展到 B 系列 21 条人工标注全集。21 条无本地 mp4，但素材齐备：
  - pilot_output/{vid}/thumbnails/      逐秒缩略图（405×720，tsu.load_thumb 可读）
  - pilot_output/{vid}/annotation_data.json  内嵌口播文本（tsu.rebuild_segs 重建 ASR 段）
帧源、ASR、scene_diffs 全部从上述本地素材构建，与 R1 完全同口径：
  选帧拼图 160px / 全帧基线 4000px 网格 / BUDGET_RATIOS{0.5,0.7,0.9} /
  METHODS{ts,uniform,tf} / MAX_FRAMES=40 / TF_THRESHOLD=0.28。

为口径统一（ASR 来源一致：annotation_data 而非独立 qwen_asr.json），21 条全部重跑，
含与 R1 重叠的 4 条（M-720f4960f5ed / M-782f8ea8c442 / M-84e1d947b553 / M-9a12ab3f34c9）。
输出独立 JSON：pilot_output/reeval_l2/reeval_l2_cost_effect21.json（R1 的 8 条保留参考）。

用法: cd HotMirrorStudio/视频降维技术研究 && PYTHONIOENCODING=utf-8 python tools/reeval_l2_extend21.py [--video M-xxx] [--resume]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import cv2  # noqa: F401  (R1 模块 exec 依赖)
from PIL import Image  # noqa: F401

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / "pilot_output"
CURATED = PILOT / "annotations_curated.json"
OUT_FILE = PILOT / "reeval_l2" / "reeval_l2_cost_effect21.json"


def _load_mod(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cost = _load_mod("cost_effect", "reeval_l2_cost_effect.py")
tsu = _load_mod("tsu", "ts_scene_union.py")

WORKERS = int(os.environ.get("WORKERS", "32"))


def process_video(vid: str) -> dict:
    anno_path = PILOT / vid / "annotation_data.json"
    if not anno_path.exists():
        raise RuntimeError(f"{vid}: 缺 annotation_data")
    anno = json.loads(anno_path.read_text(encoding="utf-8"))
    segs = tsu.rebuild_segs(anno)
    total = int(anno["total_seconds"])

    # 帧源：逐秒缩略图 → resize 到 160px 高（与 R1 抽帧口径一致）
    thumbs = {}
    for s in range(total):
        im = tsu.load_thumb(vid, s)
        if im is None:
            continue
        w = int(im.width * cost.THUMB_HEIGHT / im.height)
        thumbs[s] = im.resize((w, cost.THUMB_HEIGHT), Image.LANCZOS)
    if len(thumbs) < total:
        print(f"  !! {vid}: 缩略图 {len(thumbs)}/{total}，以实有帧为准")
    cost.THUMB_CACHE[vid] = thumbs

    scores, _ = cost.compute_density_scores(segs, total, return_dimensions=True)
    vtype = cost.prejudge_video_type(segs, total)["video_type"]
    diffs = list(tsu.scene_diffs(vid, total))
    if len(diffs) < total:
        diffs += [0.0] * (total - len(diffs))

    print(f"\n=== {vid}  dur={total}s type={vtype} total={total} ===")
    transcript = cost.ASR_CONTEXT.format(transcript=cost.build_transcript(segs))
    prompts = {
        "content": cost.EVAL_QUESTIONS + "\n\n" + transcript,
        "visual": cost.EVAL_VISUAL_QUESTIONS,
    }

    # 各方法 × 各档位的帧集（与 R1 完全一致）
    frame_sets: dict[tuple[str, float], list[int]] = {}
    for ratio in cost.BUDGET_RATIOS:
        B = max(1, int(cost.auto_budget(total) * ratio))
        ts = cost.select_frames_adaptive(scores, B, total, video_type=vtype,
                                         budget_mode="increment")
        uni = cost._uniform_sample(len(ts), total)
        tf = cost.transitions_first(diffs, len(ts), total,
                                    threshold=cost.TF_THRESHOLD)
        frame_sets[("ts", ratio)] = cost.cap_frames(ts)
        frame_sets[("uniform", ratio)] = cost.cap_frames(uni)
        frame_sets[("tf", ratio)] = cost.cap_frames(tf)

    # 1) 基线（全帧）——重新生成以捕获 usage
    baseline_jobs = [(vid, q, list(range(total))) for q in cost.QSETS]
    print("  [基线] 全帧 qwen 作答 ...")

    def run_baseline(job):
        _vid, qset, seconds = job
        grid = cost.stitch_grid(_vid, seconds, max_side=4000)
        txt, usage = cost.with_retry(cost._call_qwen_vl, grid, prompts[qset],
                                     label=f"{_vid}/baseline/{qset}")
        return (qset,), {"n_frames": len(seconds), "answer": txt,
                         "usage": usage,
                         "cost_yuan": round(cost.cost_of(usage), 5)}

    baselines = cost._run_pool(baseline_jobs, run_baseline,
                               max_workers=WORKERS)

    # 2) 方法点答案
    ans_jobs = [(vid, q, m, r) for q in cost.QSETS
                for m in cost.METHODS for r in cost.BUDGET_RATIOS]
    print(f"  [选帧] {len(ans_jobs)} 次 qwen 作答 ...")

    def run_answer(job):
        _vid, qset, method, ratio = job
        seconds = frame_sets[(method, ratio)]
        grid = cost.stitch_grid(_vid, seconds)
        txt, usage = cost.with_retry(cost._call_qwen_vl, grid, prompts[qset],
                                     label=f"{_vid}/{qset}/{method}/r{ratio}")
        return (qset, method, ratio), {"n_frames": len(seconds), "answer": txt,
                                       "usage": usage,
                                       "cost_yuan": round(cost.cost_of(usage), 5)}

    answers = cost._run_pool(ans_jobs, run_answer, max_workers=WORKERS)

    # 3) judge：选帧答案 vs 对应全帧基线
    judge_jobs = []
    for qset in cost.QSETS:
        for method in cost.METHODS:
            for ratio in cost.BUDGET_RATIOS:
                a = answers[(qset, method, ratio)]
                b = baselines[(qset,)]
                judge_jobs.append((vid, qset, method, ratio, a["answer"],
                                   b["answer"], a["n_frames"], b["n_frames"]))
    print(f"  [judge] {len(judge_jobs)} 次 ...")

    def run_judge(job):
        _vid, qset, method, ratio, ans_a, ans_b, n_a, n_b = job
        j = cost.with_retry(cost._call_judge, ans_a, ans_b, n_a, n_b,
                            label=f"judge/{_vid}/{qset}/{method}/r{ratio}")
        keep = j["overall_a"] / j["overall_b"] if j.get("overall_b") else 0.0
        return (qset, method, ratio), {**j, "retention": round(keep, 3)}

    judges = cost._run_pool(judge_jobs, run_judge, max_workers=WORKERS)

    # 4) 组装
    rec = {"video_id": vid, "vtype": vtype, "duration": round(total, 1),
           "total": total, "baselines": {q: baselines[(q,)] for q in cost.QSETS},
           "points": []}
    for qset in cost.QSETS:
        for method in cost.METHODS:
            for ratio in cost.BUDGET_RATIOS:
                a = answers[(qset, method, ratio)]
                rec["points"].append({
                    "qset": qset, "method": method, "ratio": ratio,
                    "n_frames": a["n_frames"], "answer": a["answer"],
                    "usage": a["usage"], "cost_yuan": a["cost_yuan"],
                    "compression": round(1 - a["n_frames"] / total, 3),
                    "judge": judges[(qset, method, ratio)],
                })
    rec["points"].sort(key=lambda p: (p["qset"], p["method"], p["ratio"]))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    if args.resume and OUT_FILE.exists():
        results = json.loads(OUT_FILE.read_text(encoding="utf-8"))["videos"]
        print(f"resume: 已有 {len(results)} 条")

    curated = json.loads(CURATED.read_text(encoding="utf-8"))
    vids = [v["video_id"] for v in curated]
    if args.video:
        vids = [v for v in vids if args.video in v]
    print(f"样本: {len(vids)} 条  | 并发: {WORKERS}  | 模型: {cost.VL_MODEL}  "
          f"| 档位: {cost.BUDGET_RATIOS}  | MAX_FRAMES: {cost.MAX_FRAMES}")

    for vid in vids:
        if args.resume and vid in results:
            print(f"skip {vid} (已有)")
            continue
        try:
            rec = process_video(vid)
        except Exception as e:  # noqa: BLE001
            print(f"\n!! {vid} 失败: {e}")
            continue
        results[vid] = rec
        OUT_FILE.write_text(json.dumps({"meta": {
            "vl_model": cost.VL_MODEL, "pricing": cost.PRICING,
            "budget_ratios": cost.BUDGET_RATIOS, "methods": cost.METHODS,
            "qsets": cost.QSETS, "tf_threshold": cost.TF_THRESHOLD,
            "max_frames": cost.MAX_FRAMES, "asr_source": "annotation_data",
            "frame_source": "thumbnails_160px",
        }, "videos": results}, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n已保存: {OUT_FILE}  ({len(results)} 条)")
    if results:
        cost.summarize(results)


if __name__ == "__main__":
    main()
