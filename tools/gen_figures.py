"""论文结果图表：主实验（21 条）成本-效果可视化. 只读不跑 API.

用法: cd HotMirrorStudio/视频降维技术研究 && PYTHONIOENCODING=utf-8 python tools/gen_figures.py [--file reeval_l2_cost_effect21.json]

输出到 pilot_output/figures/：
  fig1_cost_retention.png  成本-保持率曲线（x = image token 占比 vs 全帧，y = 保持率）
  fig2_retention_budget.png  保持率 × 档位分组柱状图（3 方法 × 3 档位）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pilot_output" / "figures"

METHODS = ["ts", "uniform", "tf"]
METHOD_LABEL = {"ts": "TokenSaver", "uniform": "Uniform", "tf": "TransitionsFirst"}
RATIOS = [0.5, 0.7, 0.9]
QSETS = [("content", "内容理解 (+ASR)"), ("visual", "纯视觉 (无 ASR)")]
COLORS = {"ts": "#d9534f", "uniform": "#5bc0de", "tf": "#f0ad4e"}
MARKERS = {"ts": "o", "uniform": "s", "tf": "^"}


def image_tokens(usage: dict) -> float:
    itd = usage.get("input_tokens_details") or {}
    return itd.get("image_tokens", 0) or 0


def load_videos(args):
    data = json.loads((ROOT / "pilot_output" / "reeval_l2" / args.file)
                      .read_text(encoding="utf-8"))
    return data["videos"]


def fig_cost_retention(videos, out: Path) -> None:
    """x = image token 占比（选帧/全帧），y = 保持率，每方法一条线（3 档点）."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (qset, qname) in zip(axes, QSETS):
        # 全帧基线 image token（该 qset 平均）
        base_img = np.mean([image_tokens(v["baselines"][qset]["usage"])
                            for v in videos.values()])
        for m in METHODS:
            xs, ys, errs = [], [], []
            for r in RATIOS:
                pts = [p for v in videos.values() for p in v["points"]
                       if p["qset"] == qset and p["method"] == m and p["ratio"] == r]
                if not pts:
                    continue
                xs.append(np.mean([image_tokens(p["usage"]) for p in pts]) / base_img * 100)
                ys.append(np.mean([p["judge"]["retention"] for p in pts]))
                errs.append(np.std([p["judge"]["retention"] for p in pts]))
            ax.errorbar(xs, ys, yerr=errs, fmt=f"-{MARKERS[m]}", color=COLORS[m],
                        label=METHOD_LABEL[m], capsize=3, linewidth=1.6,
                        markersize=5)
        ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7)
        ax.axvline(100, color="gray", ls=":", lw=1, alpha=0.6)
        ax.set_title(qname, fontsize=12)
        ax.set_xlabel("image token 占比 vs 全帧 (%)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("保持率 (overall_selected / overall_fullframe)", fontsize=10)
    fig.suptitle("成本-效果曲线：更少 image token 是否保持理解质量", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.name}  saved")


def fig_retention_budget(videos, out: Path) -> None:
    """x = 档位 (0.5/0.7/0.9)，每组 3 方法柱，y = 平均保持率."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    x = np.arange(len(RATIOS))
    w = 0.26
    for ax, (qset, qname) in zip(axes, QSETS):
        for i, m in enumerate(METHODS):
            ys, errs = [], []
            for r in RATIOS:
                pts = [p for v in videos.values() for p in v["points"]
                       if p["qset"] == qset and p["method"] == m and p["ratio"] == r]
                ys.append(np.mean([p["judge"]["retention"] for p in pts]))
                errs.append(np.std([p["judge"]["retention"] for p in pts]))
            ax.bar(x + (i - 1) * w, ys, w, yerr=errs, capsize=3,
                   color=COLORS[m], label=METHOD_LABEL[m], alpha=0.85)
        ax.axhline(1.0, color="gray", ls="--", lw=1)
        ax.set_title(qname, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"budget ×{r}" for r in RATIOS])
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("保持率 (≥1 = 不低于全帧)", fontsize=10)
    fig.suptitle("各档位保持率（虚线 = 全帧基线水平）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.name}  saved")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="reeval_l2_cost_effect21.json")
    args = ap.parse_args()
    videos = load_videos(args)
    print(f"样本: {len(videos)} 条")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_cost_retention(videos, OUT_DIR / "fig1_cost_retention.png")
    fig_retention_budget(videos, OUT_DIR / "fig2_retention_budget.png")


if __name__ == "__main__":
    main()
