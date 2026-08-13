"""增量重跑：全帧基线升级为可读网格（max_side=4000）+ 全部 judge 重判.

背景（2026-08-07 探针）：2000px 全帧网格把 50-111 帧压到每帧缩略图过小，
qwen3.7-plus 反而读不出细节（甚至有幻觉，如把标题幻觉成"成分解析"）。保持率
普遍 >1.0 是真实效应（选帧网格缩略图更大更可读），但基线答案作为论文 ground
truth 必须可信 → 基线改 4000px（每帧缩略图保持 ~190px 可读）。

复用已生成的选帧答案（不变），只重新生成 16 条全帧基线 + 重跑 144 次 judge。

用法: PYTHONIOENCODING=utf-8 python tools/rebaseline_4000px.py [--video M-xxx]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import reeval_l2_cost_effect as M  # noqa: E402

OUT_FILE = ROOT / "pilot_output" / "reeval_l2" / "reeval_l2_cost_effect.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    data = json.loads(OUT_FILE.read_text(encoding="utf-8"))
    videos = data["videos"]

    for vid in sorted(videos):
        if args.video and args.video not in vid:
            continue
        rec = videos[vid]
        vpath = M.get_video_path(vid)
        if vpath is None:
            print(f"!! {vid} 缺视频")
            continue
        asr = M.ASR_DIR / f"{vid}_qwen_asr.json"
        segs, _dur = M.load_asr_segments(asr)
        transcript = M.ASR_CONTEXT.format(transcript=M.build_transcript(segs))
        prompts = {"content": M.EVAL_QUESTIONS + "\n\n" + transcript,
                   "visual": M.EVAL_VISUAL_QUESTIONS}
        total = rec["total"]
        print(f"\n=== {vid} total={total} ===")
        print("  提取逐秒帧缓存 ...")
        M.THUMB_CACHE[vid] = M.extract_thumbs(vpath, total)

        # 1) 重新生成 4000px 全帧基线
        for qset in ("content", "visual"):
            grid = M.stitch_grid(vid, list(range(total)), max_side=4000)
            txt, usage = M.with_retry(
                M._call_qwen_vl, grid, prompts[qset],
                label=f"{vid}/baseline/{qset}")
            rec["baselines"][qset] = {
                "n_frames": total, "answer": txt, "usage": usage,
                "cost_yuan": round(M.cost_of(usage), 5)}
            print(f"  baseline/{qset}: img={usage.get('input_tokens_details', {}).get('image_tokens')} "
                  f"in={usage.get('input_tokens')} cost={rec['baselines'][qset]['cost_yuan']}")

        # 2) 重判全部选帧点 vs 新基线
        for p in rec["points"]:
            b = rec["baselines"][p["qset"]]["answer"]
            j = M.with_retry(M._call_judge, p["answer"], b, p["n_frames"], total,
                             label=f"judge/{vid}/{p['qset']}/{p['method']}/r{p['ratio']}")
            keep = j["overall_a"] / j["overall_b"] if j.get("overall_b") else 0.0
            p["judge"] = {**j, "retention": round(keep, 3)}

        OUT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print(f"  -> 已更新 {vid}")

    print(f"\n完成。重跑汇总:")
    M.summarize(videos)


if __name__ == "__main__":
    main()
