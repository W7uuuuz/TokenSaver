"""理解保持率 judge：TS(0.7×dur) vs 全帧(dur)、Uniform(0.7×dur) vs 全帧(dur).

成本-收益叙事的核心证据：TokenSaver 用 70% 帧（省 30% token）达到全帧理解的多大比例？
以 uniform 降帧作参照，说明"确定性降帧 + 密度集中"相对"均匀降帧"的保持率差异。

数据源（均带 ASR 口播文本，对齐生产口径）：
  - reeval_l2_prod_summary_withasr.json  生产协议 TS/Uniform 答案（同 0.7×dur 帧数）
  - reeval_l2_fullframe_withasr.json     补跑的全帧(dur=1fps) 答案

输出：每视频 ts_vs_full / uni_vs_full 两个 judge 结果 + 汇总保持率。

用法: cd HotMirrorStudio/视频降维技术研究 && PYTHONIOENCODING=utf-8 python tools/reeval_l2_keeprate.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

BACKEND = Path(r"D:\DevWorkSpace\HotMirrorStudio\backend")

for env_path in [BACKEND / ".env", Path(__file__).resolve().parent.parent / ".env"]:
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

deepseek_client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "pilot_output" / "reeval_l2"

JUDGE_PROMPT = """你是一个视频内容理解评估专家。你需要比较同一个视频的两组关键帧回答。

【答案A】（压缩选帧：用 {n_a} 帧回答）：
{answer_a}

【答案B】（全帧基线：用 {n_b} 帧（1fps 全部帧）回答）：
{answer_b}

请从以下维度评估答案A相对答案B的质量：
1. **完整性**：A 覆盖了 B 中的关键信息（产品、品牌、卖点、价格、场景风格）多少？
2. **准确性**：A 与真实内容是否一致，有无明显错误或幻觉？
3. **信息密度**：A 用更少帧是否仍传达了核心信息？

请输出严格JSON格式（不要包含markdown代码块标记）：
{{
  "completeness": {{"a": 1-5, "b": 1-5, "note": "10字以内解释"}},
  "accuracy": {{"a": 1-5, "b": 1-5, "note": "10字以内解释"}},
  "density": {{"a": 1-5, "b": 1-5, "note": "10字以内解释"}},
  "overall_a": 1-10,
  "overall_b": 1-10,
  "winner": "a" 或 "b" 或 "tie",
  "summary": "一句话总结判断理由"
}}"""


def judge(answer_a, answer_b, n_a, n_b):
    prompt = JUDGE_PROMPT.format(
        n_a=n_a, n_b=n_b, answer_a=answer_a[:2000], answer_b=answer_b[:2000])
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat", messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3]
    return json.loads(raw)


def main():
    summary = json.loads(
        (OUTPUT_DIR / "reeval_l2_prod_summary_withasr.json").read_text(encoding="utf-8"))
    fullframe = json.loads(
        (OUTPUT_DIR / "reeval_l2_fullframe_withasr.json").read_text(encoding="utf-8"))

    rows = []
    for r in summary:
        vid = r["video_id"]
        if vid not in fullframe:
            print(f"skip {vid} — 缺全帧答案")
            continue
        n_ts = r["n_frames"]["ts"]
        n_full = len(fullframe[vid]["full_frames"])
        ts_ans = r["answers"]["ts"]
        uni_ans = r["answers"]["uniform"]
        full_ans = fullframe[vid]["answer"]

        ts_j = judge(ts_ans, full_ans, n_ts, n_full)
        time.sleep(0.5)
        uni_j = judge(uni_ans, full_ans, n_ts, n_full)
        time.sleep(0.5)

        ts_keep = ts_j["overall_a"] / ts_j["overall_b"] if ts_j["overall_b"] else 0
        uni_keep = uni_j["overall_a"] / uni_j["overall_b"] if uni_j["overall_b"] else 0
        rows.append({
            "video_id": vid, "category": r["category"], "n_ts": n_ts, "n_full": n_full,
            "ts_vs_full": ts_j, "uni_vs_full": uni_j,
            "ts_keeprate": round(ts_keep, 3), "uni_keeprate": round(uni_keep, 3),
        })
        print(f"{vid:<16} TS({n_ts}帧)vs全帧({n_full}帧): "
              f"TS={ts_j['overall_a']}/全帧={ts_j['overall_b']} "
              f"保持率{ts_keep:.2f} -> {ts_j['winner']}"
              f"  |  Uni({n_ts}帧): {uni_j['overall_a']}/{uni_j['overall_b']} "
              f"保持率{uni_keep:.2f}")

    out = OUTPUT_DIR / "reeval_l2_keeprate_withasr.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存: {out}")

    if rows:
        ts_k = [r["ts_keeprate"] for r in rows]
        uni_k = [r["uni_keeprate"] for r in rows]
        ts_w = sum(1 for r in rows if r["ts_vs_full"]["winner"] == "a")
        ts_tie = sum(1 for r in rows if r["ts_vs_full"]["winner"] == "tie")
        uni_w = sum(1 for r in rows if r["uni_vs_full"]["winner"] == "a")
        uni_tie = sum(1 for r in rows if r["uni_vs_full"]["winner"] == "tie")
        print("\n" + "=" * 60)
        print("理解保持率汇总（相对全帧 1fps）")
        print("=" * 60)
        print(f"TS({len(rows)}条): 平均保持率 {sum(ts_k)/len(ts_k):.3f}  "
              f"(胜全帧 {ts_w} 平 {ts_tie})")
        print(f"Uniform:       平均保持率 {sum(uni_k)/len(uni_k):.3f}  "
              f"(胜全帧 {uni_w} 平 {uni_tie})")


if __name__ == "__main__":
    main()
