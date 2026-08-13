"""论文结果图表：L1 帧级召回对比（TS vs 转场优先 TF，T=0.28）. 只读本地计算不跑 API.

数据源：pilot_output/ts_scene_union/transitions_first_eval.json（含逐条 diffs/ts/key_frames）。
本地重算 T=0.28 的 transitions_first 帧集 + L1 指标，与 EXPERIMENTS_LOG B3 数字应一致
（TS召回 0.927 → TF召回 0.982，ΔF1 +0.091，21-0-0）。

用法: cd HotMirrorStudio/视频降维技术研究 && PYTHONIOENCODING=utf-8 python tools/gen_fig_l1.py
"""
from __future__ import annotations

import importlib.util
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
THRESHOLD = 0.28


def _load_mod(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tfe = _load_mod("tfe", "transitions_first_eval.py")
tsu = _load_mod("tsu", "ts_scene_union.py")


def main() -> None:
    rows = json.loads((ROOT / "pilot_output" / "ts_scene_union"
                       / "transitions_first_eval.json").read_text(encoding="utf-8"))
    ts_f1, ts_rec, ts_prec = [], [], []
    tf_f1, tf_rec, tf_prec = [], [], []
    win = tie = lose = 0
    for r in rows:
        diffs, total = r["diffs"], r["duration"]
        ts, key = r["ts"], r["key_frames"]
        tf = tfe.transitions_first(diffs, len(ts), total, threshold=THRESHOLD)
        m_ts, m_tf = tsu.l1_metrics(ts, key), tsu.l1_metrics(tf, key)
        ts_f1.append(m_ts["f1"]); ts_rec.append(m_ts["recall"]); ts_prec.append(m_ts["precision"])
        tf_f1.append(m_tf["f1"]); tf_rec.append(m_tf["recall"]); tf_prec.append(m_tf["precision"])
        if m_tf["f1"] > m_ts["f1"]:
            win += 1
        elif m_tf["f1"] < m_ts["f1"]:
            lose += 1
        else:
            tie += 1

    avg = lambda xs: sum(xs) / len(xs)  # noqa: E731
    print(f"T={THRESHOLD}  n={len(rows)}  胜/平/负 = {win}/{tie}/{lose}")
    print(f"  recall: TS {avg(ts_rec):.3f} -> TF {avg(tf_rec):.3f}  (+{avg(tf_rec)-avg(ts_rec):.3f})")
    print(f"  F1:     TS {avg(ts_f1):.3f} -> TF {avg(tf_f1):.3f}  (Δ{avg(tf_f1)-avg(ts_f1):+.3f})")

    # 柱状图：recall / F1 两组，TS vs TF
    fig, ax = plt.subplots(figsize=(7, 4.6))
    labels = ["recall（覆盖关键帧比例）", "F1（recall/precision 综合）"]
    x = np.arange(2)
    w = 0.32
    ts_vals = [avg(ts_rec), avg(ts_f1)]
    tf_vals = [avg(tf_rec), avg(tf_f1)]
    b1 = ax.bar(x - w / 2, ts_vals, w, label="TokenSaver", color="#5bc0de")
    b2 = ax.bar(x + w / 2, tf_vals, w, label=f"TransitionsFirst (T={THRESHOLD})",
                color="#f0ad4e")
    for bars, vals in ((b1, ts_vals), (b2, tf_vals)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008,
                    f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("L1 指标（±1s 容差，key 计数）")
    ax.set_title(f"L1 帧级召回：转场优先找回文本驱动漏掉的关键帧\n"
                 f"胜/平/负 {win}/{tie}/{lose}（n={len(rows)}）", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "fig3_l1_tf.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.name}  saved")


if __name__ == "__main__":
    main()
