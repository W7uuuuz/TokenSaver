"""标注数据治理：去重 + 删除 + 规范数据集.

依据人工标注时发现的样本质量问题（2026-08-07）：
  - M-21dd90cdfb17 与 M-e62a1221d636 转录逐字一致（同品牌同产品素材完全重复）
    → 去重，保留 M-21dd90cdfb17（主导性已标）
  - M-7fc32534a8ef（90s 牙膏直播间）曾建议拆分为口播/视觉两段，但拆分段标注在标注页
    无法正常显示，用户决定彻底删除该样本 → 从数据集移除（含原目录）

输出：
  1. pilot_output/annotations_curated.json  规范标注数据集（L1 评估输入）
  2. pilot_output/annotation_summary.json 更新（移除重复与已删样本）

用法: cd HotMirrorStudio/视频降维技术研究 && PYTHONIOENCODING=utf-8 python tools/annotate_postprocess.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / "pilot_output"
# 新导出目录：已去重、已删拆分、含用户补填主导性，重跑幂等
ANNO_DIR = Path(r"D:\DevWorkSpace\HotMirrorStudio\dev\keyframes_all_1786072967155")

DROP_VIDS = {"M-e62a1221d636", "M-7fc32534a8ef"}


def load_annotations() -> dict[str, dict]:
    out = {}
    for f in sorted(ANNO_DIR.glob("*_keyframes.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        out[d["video_id"]] = d
    return out


def main():
    annos = load_annotations()
    print(f"导入标注: {len(annos)} 条")

    # ── 1. 去重 + 删除（新导出目录已不包含这两个视频，此处为防御） ──
    for vid in DROP_VIDS:
        if vid in annos:
            annos.pop(vid)
            print(f"移除 {vid}")

    # ── 2. 更新 summary ──
    summary_path = PILOT / "annotation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    new_summary = [v for v in summary if v["video_id"] not in DROP_VIDS]
    new_summary.sort(key=lambda v: v["video_id"])
    summary_path.write_text(
        json.dumps(new_summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"summary 更新: {len(summary)} → {len(new_summary)} 条")

    # ── 3. 规范数据集 ──
    curated = []
    for vid, a in sorted(annos.items()):
        curated.append({
            "video_id": vid,
            "category": a.get("category", ""),
            "total_seconds": a.get("total_seconds"),
            "visual_priority": a.get("visual_priority"),
            "key_frames": a.get("key_frames", []),
            "n_key_frames": len(a.get("key_frames", [])),
        })
    curated_path = PILOT / "annotations_curated.json"
    curated_path.write_text(
        json.dumps(curated, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"规范数据集: {curated_path}  ({len(curated)} 条)")

    # ── 4. 待补选主导性 ──
    missing = [v["video_id"] for v in curated if not v["visual_priority"]]
    print(f"待补主导性: {missing}")


if __name__ == "__main__":
    main()
