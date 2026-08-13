"""转镜优先帧集 L2 理解评估：TS / Uniform / TransitionsFirst 三组帧集 × 两套问题集.

背景（2026-08-07）：预算内「转镜优先 + 均匀兜底」（T=0.28）在 L1 关键帧召回上
21-0-0 全面胜出（召回 0.927→0.982）。本脚本验证其在 MLLM 理解上是否不退化——帧级
召回提升必须以「理解不损失」为前提才可进入生产。

设计（与 reeval_human_anno.py 同框架）：
  - 21 条样本，帧来自逐秒缩略图（不解码 mp4）
  - 三组帧集（帧数全部 = len(TS)，token 完全对齐）：
      ts     生产 TokenSaver select_frames_adaptive(increment)
      uniform 同帧数均匀采样
      tf     transitions_first(diffs, N, threshold=0.28)
  - 两套问题集：content（内容 5 问 + ASR 转录）、visual（纯视觉 5 问，无 ASR）
  - qwen3.7-plus 作答 → DeepSeek 无 Oracle 配对盲审（tf vs ts、tf vs uniform）
  - 高并发 + 限流长退避重试

用法: cd HotMirrorStudio/视频降维技术研究 && PYTHONIOENCODING=utf-8 python tools/reeval_transitions_l2.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("rha", ROOT / "tools" / "reeval_human_anno.py")
rha = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rha)

spec2 = importlib.util.spec_from_file_location("tfe", ROOT / "tools" / "transitions_first_eval.py")
tfe = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(tfe)

WORKERS = int(os.environ.get("WORKERS", "32"))
THRESHOLD = 0.28
OUT_DIR = rha.PILOT / "reeval_transitions_l2"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curated = json.loads(rha.CURATED.read_text(encoding="utf-8"))
    print(f"样本: {len(curated)} 条 | 并发 {WORKERS} | 转镜阈值 {THRESHOLD}\n")

    # 1. 本地准备：帧集 + 拼图 + 提示词
    jobs = []  # (qset, method, grid, prompt, vid)
    meta = {}
    for v in sorted(curated, key=lambda x: x["video_id"]):
        vid = v["video_id"]
        anno_path = rha.PILOT / vid / "annotation_data.json"
        if not anno_path.exists():
            print(f"!! {vid}: 缺 annotation_data，跳过")
            continue
        segs = rha.rebuild_segs(json.loads(anno_path.read_text(encoding="utf-8")))
        total = v["total_seconds"]
        ts = tfe.tsu.ts_frames(segs, total)
        N = len(ts)
        diffs = tfe.tsu.scene_diffs(vid, total)
        uni = rha._uniform_sample(N, total)
        tf = tfe.transitions_first(diffs, N, total, threshold=THRESHOLD)

        frame_sets = {"ts": ts, "uniform": uni, "tf": tf}
        meta[vid] = {"category": v.get("category"), "duration": total,
                     "n_frames": {m: len(f) for m, f in frame_sets.items()},
                     "frames": frame_sets}
        print(f"{vid:<18} {str(v.get('category')):<4} {total:>3}s "
              f"TS={N} Uni={N} TF={len(tf)}")

        transcript = rha.ASR_CONTEXT.format(transcript=rha.build_transcript(segs))
        grids = {m: rha.stitch_grid(f, vid) for m, f in frame_sets.items()}
        for m in ("ts", "uniform", "tf"):
            jobs.append(("content", m, grids[m],
                         rha.EVAL_QUESTIONS + "\n\n" + transcript, vid))
            jobs.append(("visual", m, grids[m], rha.EVAL_VISUAL_QUESTIONS, vid))

    # 2. 高并发 MLLM 作答
    print(f"\nMLLM 调用: {len(jobs)} 次  (并发 {WORKERS})")
    answers: dict[tuple, str] = {}

    def run_mllm(job):
        qset, m, grid, prompt, vid = job
        txt = rha.with_retry(rha._call_qwen_vl, grid, prompt,
                             label=f"{vid}/{qset}/{m}")
        return (vid, qset, m), txt

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(run_mllm, j) for j in jobs]
        done = 0
        for fut in as_completed(futs):
            key, txt = fut.result()
            answers[key] = txt
            done += 1
            print(f"  [{done}/{len(jobs)}] {key[0]}/{key[1]}/{key[2]}: {txt[:46]}...")

    # 3. 配对盲审：tf vs ts / tf vs uniform
    comparisons = []
    for vid in meta:
        for qset in ("content", "visual"):
            for other in ("ts", "uniform"):
                a = answers.get((vid, qset, "tf"))
                b = answers.get((vid, qset, other))
                if a and b and "ERROR" not in a and "ERROR" not in b:
                    comparisons.append((vid, qset, other, a, b))
    print(f"\nJudge 调用: {len(comparisons)} 次  (并发 {WORKERS})")
    judge_res: dict[tuple, dict] = {}

    def run_judge(cmp):
        vid, qset, other, a, b = cmp
        j = rha.with_retry(rha._call_judge, a, b,
                           label=f"judge/{vid}/{qset}/{other}")
        return (vid, qset, other), j

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(run_judge, c) for c in comparisons]
        done = 0
        for fut in as_completed(futs):
            key, j = fut.result()
            judge_res[key] = j
            done += 1
            print(f"  [{done}/{len(comparisons)}] {key[0]}/{key[1]}/tf_vs_{key[2]}: "
                  f"TF={j.get('overall_a')} {key[2]}={j.get('overall_b')} "
                  f"-> {j.get('winner')}")

    # 4. 组装 + 汇总
    results = []
    for vid, m in meta.items():
        rec = {"video_id": vid, **m, "answers": {}, "judges": {}}
        for qset in ("content", "visual"):
            rec["answers"][qset] = {meth: answers.get((vid, qset, meth), "")
                                    for meth in ("ts", "uniform", "tf")}
            rec["judges"][qset] = {
                "tf_vs_ts": judge_res.get((vid, qset, "ts")),
                "tf_vs_uniform": judge_res.get((vid, qset, "uniform")),
            }
        results.append(rec)

    out = OUT_DIR / "reeval_transitions_l2.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n已保存: {out}")

    print("\n" + "=" * 68)
    print("转镜优先帧集 L2 理解评估（TF vs TS / TF vs Uniform）")
    print("=" * 68)
    for qset, qname in (("content", "内容理解(+ASR)"), ("visual", "纯视觉(无ASR)")):
        print(f"\n【{qname}】")
        for other in ("ts", "uniform"):
            rows = [r for r in results if r["judges"][qset][f"tf_vs_{other}"]]
            t_w = sum(1 for r in rows
                      if r["judges"][qset][f"tf_vs_{other}"]["winner"] == "a")
            o_w = sum(1 for r in rows
                      if r["judges"][qset][f"tf_vs_{other}"]["winner"] == "b")
            tie = sum(1 for r in rows
                      if r["judges"][qset][f"tf_vs_{other}"]["winner"] == "tie")
            avg_t = sum(r["judges"][qset][f"tf_vs_{other}"]["overall_a"]
                        for r in rows) / len(rows) if rows else 0
            avg_o = sum(r["judges"][qset][f"tf_vs_{other}"]["overall_b"]
                        for r in rows) / len(rows) if rows else 0
            print(f"  TF vs {other:<8}  TF胜 {t_w:>2} / {other}胜 {o_w:>2} "
                  f"/ 平 {tie:>2}  (n={len(rows)})   平均 TF {avg_t:.1f} vs "
                  f"{other} {avg_o:.1f}")


if __name__ == "__main__":
    main()
