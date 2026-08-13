"""
Visualization for Round 1 Pilot.
Generates density curve plots with routing overlay, coupling markers,
and selected frame indicators — the primary output for parameter tuning.
"""
import os
import math
from typing import List, Dict, Optional
import numpy as np

from . import config


# ============================================================================
# Matplotlib setup (lazy, with CJK font detection)
# ============================================================================
_plt = None
_cjk_font = None


def _ensure_matplotlib():
    global _plt, _cjk_font
    if _plt is not None:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager
    import platform

    # Try to find a CJK-capable font — check Windows system fonts directly first
    cjk_candidates = []
    if platform.system() == "Windows":
        cjk_candidates = [
            "C:/Windows/Fonts/msyh.ttc",       # Microsoft YaHei
            "C:/Windows/Fonts/simhei.ttf",      # SimHei
            "C:/Windows/Fonts/simsun.ttc",      # SimSun
            "C:/Windows/Fonts/msyhbd.ttc",      # Microsoft YaHei Bold
        ]

    cjk_font_path = None
    for path in cjk_candidates:
        if os.path.exists(path):
            cjk_font_path = path
            break

    if cjk_font_path:
        fontManager.addfont(cjk_font_path)
        try:
            from matplotlib.font_manager import FontProperties
            fp = FontProperties(fname=cjk_font_path)
            family = fp.get_name()
            plt.rcParams["font.family"] = family
            _cjk_font = fp
            print(f"  [Font] Loaded CJK font: {family} ({cjk_font_path})")
        except Exception:
            _cjk_font = FontProperties()
            print(f"  [Font] WARNING: Failed to load CJK font from {cjk_font_path}")
    else:
        print("  [Font] WARNING: No CJK font file found — Chinese characters may render as tofu")
        # Fallback: try font names
        name_candidates = [
            "SimHei", "Microsoft YaHei", "Noto Sans CJK SC",
            "WenQuanYi Micro Hei", "Source Han Sans SC",
            "PingFang SC", "Hiragino Sans GB", "STHeiti",
        ]
        _cjk_font = FontProperties()
        for name in name_candidates:
            try:
                fp = FontProperties(family=name)
                _cjk_font = fp
                plt.rcParams["font.family"] = name
                break
            except Exception:
                continue

    plt.rcParams.update({
        "axes.unicode_minus": False,
        "font.size": 10,
    })
    _plt = plt


# ============================================================================
# Color mapping for routing categories
# ============================================================================
ROUTING_COLORS = {
    "极密": "#d62728",    # red
    "密集": "#ff7f0e",    # orange
    "混合": "#2ca02c",    # green
    "稀疏": "#1f77b4",    # blue
    "极少": "#7f7f7f",    # gray
}


def _get_routing_color(category_name: str) -> str:
    for key, color in ROUTING_COLORS.items():
        if key in category_name:
            return color
    return "#7f7f7f"


# ============================================================================
# Main visualization: Density Dashboard
# ============================================================================
def plot_density_dashboard(
    video_name: str,
    total_seconds: int,
    # Time series data
    time_axis: List[float],
    composite_scores: List[float],
    routing_decisions: List[Dict],
    coupling_flags: List[int],
    # 5-dim breakdown
    dim_scores: Dict[str, List[float]],
    # Visual features
    visual_entropy: List[float],
    visual_motion: List[float],
    coupling_lcd: List[int],
    coupling_scd: List[int],
    # Selected frames at different budgets
    selected_frames: Dict[int, List[int]],
    # Scene boundaries
    scene_boundaries: List[int],
    # Output
    output_dir: str,
    output_prefix: str = "pilot",
):
    """Generate the full density dashboard figure."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(config.FIGURE_WIDTH_INCHES, config.FIGURE_HEIGHT_INCHES))
    gs = GridSpec(5, 1, figure=fig, height_ratios=[2, 1.5, 1, 1, 1.2], hspace=0.35)

    # ---- Panel 1: Composite score + routing background ----
    ax1 = fig.add_subplot(gs[0])
    _plot_routing_background(ax1, routing_decisions, total_seconds)
    ax1.plot(time_axis[:len(composite_scores)], composite_scores,
             color="black", linewidth=1.5, label="复合评分 (Composite Score)", zorder=5)
    ax1.axhline(y=8.0, color="#d62728", linestyle="--", alpha=0.5, linewidth=0.8)
    ax1.axhline(y=6.0, color="#ff7f0e", linestyle="--", alpha=0.5, linewidth=0.8)
    ax1.axhline(y=4.0, color="#2ca02c", linestyle="--", alpha=0.5, linewidth=0.8)
    ax1.axhline(y=2.0, color="#1f77b4", linestyle="--", alpha=0.5, linewidth=0.8)

    # Coupling markers
    _plot_coupling_events(ax1, coupling_flags, time_axis, composite_scores)

    # Scene boundary markers
    for sb in scene_boundaries:
        if sb < total_seconds:
            ax1.axvline(x=sb, color="#9467bd", linestyle=":", alpha=0.6, linewidth=0.8)

    ax1.set_ylabel("复合评分\n(0-10)")
    ax1.set_title(f"Token-Saver 密度仪表盘 — {video_name}", fontsize=14, fontweight="bold")
    ax1.set_xlim(0, total_seconds)
    ax1.set_ylim(0, 10.5)
    ax1.legend(loc="upper right", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # ---- Panel 2: 5-dim breakdown ----
    ax2 = fig.add_subplot(gs[1])
    colors_5dim = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    labels_5dim = [
        f"词汇密度TTR (w={config.WEIGHT_TTR})",
        f"信息熵 (w={config.WEIGHT_ENTROPY})",
        f"意外度Surprisal (w={config.WEIGHT_SURPRISAL})",
        f"信息点密度 (w={config.WEIGHT_INFO_POINT})",
        f"冗余度反向 (w={config.WEIGHT_REDUNDANCY})",
    ]
    dim_keys = ["ttr_norm", "entropy_norm", "surprisal_norm", "info_point_norm", "redundancy_norm"]

    for i, (key, label, color) in enumerate(zip(dim_keys, labels_5dim, colors_5dim)):
        values = dim_scores.get(key, [])
        ax2.plot(time_axis[:len(values)], values, color=color, linewidth=1.0,
                 alpha=0.8, label=label)

    ax2.set_ylabel("归一化值\n[0, 1]")
    ax2.set_xlim(0, total_seconds)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper right", fontsize=7, ncol=3)
    ax2.grid(True, alpha=0.3)

    # ---- Panel 3: Visual features ----
    ax3 = fig.add_subplot(gs[2])
    ax3_twin = ax3.twinx()

    ax3.plot(time_axis[:len(visual_entropy)], visual_entropy,
             color="#1f77b4", linewidth=1.0, label="视觉熵 H_vis(t)", alpha=0.8)
    ax3.plot(time_axis[:len(visual_motion)], visual_motion,
             color="#d62728", linewidth=1.0, label="光流幅值 M(t)", alpha=0.8)

    ax3.set_ylabel("视觉特征")
    ax3.set_xlim(0, total_seconds)
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ---- Panel 4: Coupling detection detail ----
    ax4 = fig.add_subplot(gs[3])
    n = min(len(coupling_lcd), len(coupling_scd), len(time_axis))
    ax4.fill_between(time_axis[:n], 0, coupling_lcd[:n],
                     color="#ff7f0e", alpha=0.4, label="LCD (词汇耦合)", step="mid")
    ax4.fill_between(time_axis[:n], 0, coupling_scd[:n],
                     color="#2ca02c", alpha=0.4, label="SCD (结构耦合)", step="mid")

    # Overlay: text density × visual variance (SCD signal)
    text_norm = [s / 10.0 for s in composite_scores[:n]]
    vis_var = _compute_rolling_std(visual_entropy[:n], config.SCD_SIGMA_WINDOW_SEC)
    scd_signal = [t * v for t, v in zip(text_norm, vis_var)]
    ax4.plot(time_axis[:len(scd_signal)], scd_signal,
             color="#d62728", linewidth=1.0, alpha=0.6, label="SCD信号 E_text×σ_vis")
    ax4.axhline(y=config.SCD_TEXT_VISUAL_THRESHOLD, color="#d62728",
                linestyle="--", alpha=0.5, linewidth=0.8,
                label=f"SCD阈值 ({config.SCD_TEXT_VISUAL_THRESHOLD})")

    ax4.set_ylabel("耦合检测")
    ax4.set_xlabel("时间 (秒)")
    ax4.set_xlim(0, total_seconds)
    ax4.set_ylim(-0.05, 1.2)
    ax4.legend(loc="upper right", fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)

    # ---- Panel 5: Selected frames at multiple budgets ----
    ax5 = fig.add_subplot(gs[4])
    budget_colors = {3: "#d62728", 10: "#ff7f0e", 20: "#2ca02c", 30: "#1f77b4", 50: "#9467bd"}
    y_offset = 0
    y_labels = []

    for budget_k in sorted(selected_frames.keys()):
        frames = selected_frames[budget_k]
        color = budget_colors.get(budget_k, "#333333")
        y = y_offset
        ax5.scatter(frames, [y] * len(frames), color=color, s=30,
                    alpha=0.7, edgecolors="black", linewidth=0.3, zorder=3)
        y_labels.append((y, f"B={budget_k} ({len(frames)}帧)"))
        y_offset += 1

    # Mark scene boundaries
    for sb in scene_boundaries:
        if sb < total_seconds:
            ax5.axvline(x=sb, color="#9467bd", linestyle=":", alpha=0.4, linewidth=0.6)

    ax5.set_yticks([y for y, _ in y_labels])
    ax5.set_yticklabels([l for _, l in y_labels], fontsize=9)
    ax5.set_xlabel("时间 (秒)")
    ax5.set_xlim(0, total_seconds)
    ax5.set_ylim(-0.8, y_offset - 0.2)
    ax5.grid(True, alpha=0.3, axis="x")

    # ---- Finalize ----
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{output_prefix}_{video_name}_dashboard.png")
    fig.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    return output_path


# ============================================================================
# Summary bar chart: per-budget metrics
# ============================================================================
def plot_budget_summary(
    video_name: str,
    budget_results: Dict[int, dict],
    output_dir: str,
    output_prefix: str = "pilot",
):
    """Bar chart comparing metrics across budget levels."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    budgets = sorted(budget_results.keys())
    n = len(budgets)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Token-Saver 预算扫描摘要 — {video_name}", fontsize=14, fontweight="bold")

    metrics = ["n_frames", "avg_importance", "coupling_coverage", "scene_coverage"]
    titles = ["选中帧数", "平均重要性", "耦合覆盖 (%)", "场景覆盖 (%)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        values = [budget_results[k].get(metric, 0) for k in budgets]
        bars = ax.bar(range(n), values, color=color, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(n))
        ax.set_xticklabels([f"B={k}" for k in budgets], fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * max(values, default=1),
                    f"{val:.1f}" if isinstance(val, float) else str(val),
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{output_prefix}_{video_name}_budget_summary.png")
    fig.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    return output_path


# ============================================================================
# Helper plotting functions
# ============================================================================
def _plot_routing_background(ax, routing_decisions: List[Dict], total_seconds: int):
    """Paint routing category background colors."""
    n = min(len(routing_decisions), total_seconds)
    for i in range(n):
        rd = routing_decisions[i]
        cat = rd.get("local_cat", "")
        color = _get_routing_color(cat)
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color=color, linewidth=0)


def _plot_coupling_events(
    ax, coupling_flags: List[int], time_axis: List[float], composite_scores: List[float]
):
    """Add vertical span markers for coupling events."""
    n = min(len(coupling_flags), len(time_axis), len(composite_scores))
    in_event = False
    event_start = 0

    for i in range(n):
        if coupling_flags[i] == 1 and not in_event:
            in_event = True
            event_start = max(0, i - 0.5)
        elif coupling_flags[i] == 0 and in_event:
            in_event = False
            ax.axvspan(event_start, i + 0.5, alpha=0.12, color="#d62728", linewidth=0)

    if in_event:
        ax.axvspan(event_start, n + 0.5, alpha=0.12, color="#d62728", linewidth=0)


def _compute_rolling_std(values: List[float], window_sec: int) -> List[float]:
    """Rolling standard deviation with left-aligned window."""
    n = len(values)
    result = [0.0] * n
    for i in range(n):
        start = max(0, i - window_sec)
        end = min(n, i + window_sec + 1)
        window = values[start:end]
        if window:
            result[i] = float(np.std(window))
    return result


# ============================================================================
# Text report
# ============================================================================
def print_text_report(
    video_name: str,
    total_seconds: int,
    composite_scores: List[float],
    routing_decisions: List[Dict],
    coupling_flags: List[int],
    coupling_lcd: List[int],
    coupling_scd: List[int],
    selected_frames: Dict[int, List[int]],
    scene_boundaries: List[int],
):
    """Print a text summary to stdout for quick inspection."""
    n = len(composite_scores)
    global_avg = np.mean(composite_scores)
    n_coupling = sum(coupling_flags)
    n_lcd = sum(coupling_lcd)
    n_scd = sum(coupling_scd)

    print(f"\n{'='*60}")
    print(f"  Token-Saver 试点报告: {video_name}")
    print(f"{'='*60}")
    print(f"  时长: {total_seconds}s")
    print(f"  全局平均密度评分: {global_avg:.1f}/10")
    print(f"  场景边界: {len(scene_boundaries)} 处")
    print(f"  耦合事件: {n_coupling}s (LCD={n_lcd}, SCD={n_scd})")
    print()

    # Routing distribution
    cat_counts = {}
    for rd in routing_decisions[:n]:
        cat = rd.get("local_cat", "?")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    print("  路由类别分布:")
    for cat, count in sorted(cat_counts.items()):
        pct = count / n * 100 if n > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"    {cat:<25s} {count:>4d}s ({pct:5.1f}%) {bar}")

    print()
    print("  各预算选中帧:")
    for k in sorted(selected_frames.keys()):
        frames = selected_frames[k]
        print(f"    B={k:>3d}: {len(frames):>3d} 帧 → {frames[:5]}..." if len(frames) > 5 else f"    B={k:>3d}: {len(frames):>3d} 帧 → {frames}")

    print(f"{'='*60}\n")
