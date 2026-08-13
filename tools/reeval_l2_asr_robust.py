"""E-研究：ASR 鲁棒性.

TokenSaver 依赖 ASR（分词文本 + 段起止时间戳）。真实场景 ASR 会有误差：
时间戳抖动、数字/单位误写。本脚本注入两类噪声，测量选帧稳定性与视频类型翻转。

回答"ASR 误差下 TokenSaver 选帧是否稳健"：
  - 选帧稳定度 overlap = |S_noisy ∩ S_clean| / min(|S_noisy|, |S_clean|)，多 seed 平均
    （Uniform 不受 ASR 影响，稳定度恒 1.0，TS 稳定度体现对 ASR 的敏感度）
  - 视频类型（oral/review）翻转率：时间戳抖动是否跨过 pause_ratio=0.15 阈值
  - 对比指标：噪声前后 TS 选帧的 Jaccard 相似度

两级噪声：
  - time_jitter ∈ {0, 0.5, 1.0, 2.0} 秒：对每段 start/end 加均匀随机抖动
  - entity_typo ∈ {0, 0.2, 0.4}：以 rate 概率把数字/单位字符替换为形近字
    （模拟 ASR 听错关键实体 → 实体正则匹配失败 → info_density 失真）

用法: cd HotMirrorStudio/视频降维技术研究 && PYTHONIOENCODING=utf-8 python tools/reeval_l2_asr_robust.py
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import jieba

BACKEND = Path(r"D:\DevWorkSpace\HotMirrorStudio\backend")
sys.path.insert(0, str(BACKEND))

from app.video.token_saver import (  # noqa: E402
    auto_budget,
    compute_density_scores,
    prejudge_video_type,
    select_frames_adaptive,
)

ROOT = Path(__file__).resolve().parent.parent
ASR_DIR = ROOT / "pilot_output" / "qwen_cache" / "asr"
OUTPUT_DIR = ROOT / "pilot_output" / "reeval_l2"

SEEDS = 5
TIME_JITTERS = [0.0, 0.5, 1.0, 2.0]
ENTITY_TYPOS = [0.0, 0.2, 0.4]

# ASR 对数字/单位的常见误写（形近/同音），破坏 ENTITY_PATTERNS 匹配
ENTITY_TYPO = {
    "十": "是", "百": "白", "千": "前", "万": "晚",
    "九": "久", "五": "午", "三": "山", "四": "寺", "七": "漆", "八": "扒",
    "二": "贰", "一": "以", "零": "陵",
    "元": "圆", "折": "则", "瓶": "平", "盒": "合", "克": "刻",
    "秒": "苗", "分": "纷", "小时": "消时", "天": "添", "年": "念",
    "%": "％", "升": "生", "毫升": "号升", "g": "G", "m": "M",
}


def load_asr_segments(path: Path) -> tuple[list[dict], float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    sentences = d.get("sentences") or []
    segs = [{"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in sentences if s.get("text")]
    duration = max(s["end"] for s in segs) if segs else 0.0
    return segs, duration


def inject_time_jitter(segments: list[dict], jitter: float, rng: random.Random):
    segs = []
    for s in segments:
        j = rng.uniform(-jitter, jitter)
        start = max(0.0, s["start"] + j)
        end = max(start, s["end"] + j)
        segs.append({"start": start, "end": end, "text": s["text"]})
    segs.sort(key=lambda s: s["start"])
    return segs


def inject_entity_typo(segments: list[dict], rate: float, rng: random.Random):
    segs = []
    for s in segments:
        text = s["text"]
        if rate > 0:
            chars = []
            for ch in text:
                if ch in ENTITY_TYPO and rng.random() < rate:
                    chars.append(ENTITY_TYPO[ch])
                else:
                    chars.append(ch)
            text = "".join(chars)
        segs.append({"start": s["start"], "end": s["end"], "text": text})
    return segs


def select_ts(segments, duration):
    total = int(duration)
    B = max(1, int(auto_budget(duration)))
    scores, _ = compute_density_scores(segments, duration, return_dimensions=True)
    vtype = prejudge_video_type(segments, duration)["video_type"]
    frames = select_frames_adaptive(scores, B, total, video_type=vtype,
                                    budget_mode="increment")
    return frames, vtype


def overlap(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "reeval_l2_asr_robust.json"
    results = {}
    if out_path.exists():
        results = json.loads(out_path.read_text(encoding="utf-8"))

    for asr_path in sorted(ASR_DIR.glob("*_qwen_asr.json")):
        vid = asr_path.name.replace("_qwen_asr.json", "")
        if vid in results:
            continue
        segs, duration = load_asr_segments(asr_path)
        if not segs:
            continue
        clean_frames, clean_type = select_ts(segs, duration)
        print(f"\n=== {vid}  dur={duration:.0f}s clean_type={clean_type} n={len(clean_frames)} ===")

        entry = {
            "video_id": vid, "duration": round(duration, 1),
            "clean_type": clean_type, "clean_frames": clean_frames,
            "rows": [],
        }
        for tj in TIME_JITTERS:
            for typo in ENTITY_TYPOS:
                if tj == 0 and typo == 0:
                    continue
                ov = []; jac = []
                type_flips = 0
                for seed in range(SEEDS):
                    rng = random.Random(seed)
                    noisy = segs
                    if typo > 0:
                        noisy = inject_entity_typo(noisy, typo, rng)
                    if tj > 0:
                        noisy = inject_time_jitter(noisy, tj, rng)
                    frames, vtype = select_ts(noisy, duration)
                    ov.append(overlap(clean_frames, frames))
                    jac.append(jaccard(clean_frames, frames))
                    if vtype != clean_type:
                        type_flips += 1
                entry["rows"].append({
                    "time_jitter": tj, "entity_typo": typo,
                    "overlap": round(sum(ov) / len(ov), 3),
                    "jaccard": round(sum(jac) / len(jac), 3),
                    "type_flip_rate": round(type_flips / SEEDS, 2),
                })
                print(f"  tj={tj:.1f} typo={typo:.1f}  "
                      f"overlap={entry['rows'][-1]['overlap']:.3f} "
                      f"jaccard={entry['rows'][-1]['jaccard']:.3f} "
                      f"type_flip={entry['rows'][-1]['type_flip_rate']}")

        results[vid] = entry
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                            encoding="utf-8")

    print(f"\n已保存: {out_path}  ({len(results)} 条)")
    if results:
        print("\n" + "=" * 64)
        print("平均选帧稳定度（overlap，越低越敏感）")
        print("=" * 64)
        for tj in TIME_JITTERS:
            for typo in ENTITY_TYPOS:
                if tj == 0 and typo == 0:
                    continue
                vals = [v["overlap"] for r in results.values()
                        for v in r["rows"] if v["time_jitter"] == tj and v["entity_typo"] == typo]
                flips = [v["type_flip_rate"] for r in results.values()
                         for v in r["rows"] if v["time_jitter"] == tj and v["entity_typo"] == typo]
                if vals:
                    print(f"  jitter={tj:.1f}s typo={typo:.1f}: "
                          f"overlap={sum(vals)/len(vals):.3f}  type_flip={sum(flips)/len(flips):.2f}")


if __name__ == "__main__":
    main()
