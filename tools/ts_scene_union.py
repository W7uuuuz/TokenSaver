"""TS ∪ 转镜头取帧：补齐算法结构性丢失的「一闪而过」帧.

背景（2026-08-07）：用户视觉目检发现 TS 取帧会丢失人工标注的部分帧——这些帧一闪
而过，通常是转镜头，承载证据/列举/操作内容（成分展示、多 SKU 切换、使用动作），
拆解/复刻时不能少了这一帧的动作。方案：TS 取帧 ∪ 转镜头取帧，先不限预算看方向。

转镜头检测（秒域，与 TS/uniform 对齐，避开 C1 的帧/秒单位 bug）：
  - 逐秒缩略图 → 相邻秒 HSV 直方图 Bhattacharyya 距离
  - diff > threshold 的秒视为画面突变点（转镜/操作/列举），补进取帧
  - 多档阈值对比，看对人工关键帧的召回提升与帧数代价

输出：
  - pilot_output/ts_scene_union.json  逐条 diff / ts / union / L1 对比 / 找回·仍漏关键帧
  - pilot_output/scene_union_inspect/ 渲染对比图（TS | TS∪转镜[新增标黄] | 人工关键帧）

用法: cd HotMirrorStudio/视频降维技术研究 && PYTHONIOENCODING=utf-8 python tools/ts_scene_union.py [vid1 vid2 ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BACKEND = Path(r"D:\DevWorkSpace\HotMirrorStudio\backend")
sys.path.insert(0, str(BACKEND))
from app.video.token_saver import (  # noqa: E402
    auto_budget,
    compute_density_scores,
    prejudge_video_type,
    select_frames_adaptive,
)

ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / "pilot_output"
CURATED = PILOT / "annotations_curated.json"
OUT_DIR = PILOT / "ts_scene_union"
INSPECT_DIR = PILOT / "scene_union_inspect"

THRESHOLDS = [0.06, 0.10, 0.15, 0.25]
RENDER_T = 0.10  # 渲染对比图用的主阈值
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"


# ── 转镜头检测（逐秒缩略图帧差） ──


def load_thumb(vid: str, second: int):
    p = PILOT / vid / "thumbnails" / f"t_{second + 1:03d}.jpg"
    if not p.exists():
        return None
    return Image.open(p).convert("RGB")


def _hist(im: Image.Image) -> np.ndarray:
    small = im.resize((32, 32))
    hsv = cv2.cvtColor(np.array(small), cv2.COLOR_RGB2HSV)
    return cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256]).flatten()


def scene_diffs(vid: str, total: int) -> list[float]:
    """逐秒缩略图帧差（vs 前一秒），diff[0]=0。返回 [0..total)。"""
    diffs: list[float] = []
    prev = None
    for s in range(total):
        im = load_thumb(vid, s)
        if im is None:
            diffs.append(0.0)
            continue
        h = _hist(im)
        if prev is None:
            diffs.append(0.0)
        else:
            diffs.append(float(cv2.compareHist(prev, h, cv2.HISTCMP_BHATTACHARYYA)))
        prev = h
    return diffs


def rebuild_segs(anno_data: dict) -> list[dict]:
    secs = anno_data["seconds"]
    segs: list[dict] = []
    cur = None
    for s in secs:
        t = (s.get("text") or "").strip()
        if not t:
            continue
        st = float(s["second"])
        if cur is None or st - cur["end"] > 1.0:
            cur = {"start": st, "end": st + 0.5, "text": t}
            segs.append(cur)
        else:
            cur["end"] = st + 0.5
            cur["text"] += t
    return segs


def ts_frames(segs: list[dict], total: int) -> list[int]:
    scores, _ = compute_density_scores(segs, total, return_dimensions=True)
    vtype = prejudge_video_type(segs, total)["video_type"]
    B = auto_budget(total)
    return select_frames_adaptive(scores, B, total, video_type=vtype,
                                  budget_mode="increment")


# ── L1 评估（±1s，key 计数，与 reeval_l1_extended.py 一致） ──


def l1_metrics(selected: list[int], key: list[int], tol: int = 1) -> dict:
    matched_keys = [hf for hf in key
                    if any(abs(sf - hf) <= tol for sf in selected)]
    matched = len(matched_keys)
    p = matched / len(selected) if selected else 0.0
    r = matched / len(key) if key else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"f1": f1, "precision": p, "recall": r, "n": len(selected),
            "matched": matched}


def recovered_keys(union: list[int], ts: list[int], key: list[int],
                   tol: int = 1) -> tuple[list[int], list[int]]:
    """union 相比 ts 找回的关键帧；以及 union 仍漏的关键帧。"""
    matched_ts = {hf for hf in key if any(abs(sf - hf) <= tol for sf in ts)}
    matched_uni = {hf for hf in key if any(abs(sf - hf) <= tol for sf in union)}
    return sorted(matched_uni - matched_ts), sorted(set(key) - matched_uni)


# ── 渲染（复用 render_selected_frames 风格，新增帧标黄） ──


def render_grid(vid: str, title: str, seconds: list[int], out_path: Path,
                highlight: set[int] | None = None) -> None:
    items = [(s, load_thumb(vid, s)) for s in seconds]
    items = [(s, im) for s, im in items if im is not None]
    if not items:
        print(f"  !! {vid} 无可用缩略图，跳过")
        return
    n = len(items)
    cols, th = min(5, n), 240
    rows = (n + cols - 1) // cols
    resized = []
    for _s, im in items:
        w = int(im.width * th / im.height)
        resized.append(im.resize((w, th), Image.LANCZOS))
    col_ws = [max((resized[r * cols + c].width for r in range(rows)
                   if r * cols + c < n), default=0) for c in range(cols)]
    pad, label_h, header_h = 6, 18, 44
    grid_w = sum(col_ws) + (cols + 1) * pad
    grid_h = header_h + rows * (th + label_h + pad) + pad
    grid = Image.new("RGB", (grid_w, grid_h), (22, 22, 26))
    draw = ImageDraw.Draw(grid)
    try:
        font_t = ImageFont.truetype(FONT_PATH, 20)
        font_l = ImageFont.truetype(FONT_PATH, 14)
    except Exception:
        font_t = font_l = ImageFont.load_default()
    draw.text((pad + 2, 10), title, fill=(240, 240, 240), font=font_t)
    for c in range(cols):
        x0 = pad + sum(col_ws[:c]) + c * pad
        for r in range(rows):
            idx = r * cols + c
            if idx >= n:
                break
            y = header_h + pad + r * (th + label_h + pad)
            grid.paste(resized[idx], (x0, y))
            label = f"t={items[idx][0]}s"
            is_new = highlight is not None and items[idx][0] in highlight
            fill = (255, 220, 60) if is_new else (200, 200, 200)
            tw = draw.textlength(label, font=font_l)
            draw.text((x0 + (col_ws[c] - tw) / 2, y + th + 2), label, fill=fill,
                      font=font_l)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path, quality=90)
    print(f"  {out_path.name}  ({n} 帧)  {grid.size}")


def build_compare(paths: list[Path], out_path: Path) -> None:
    ims = [Image.open(p).convert("RGB") for p in paths if p.exists()]
    if not ims:
        return
    gap, pad = 12, 10
    w = sum(im.width for im in ims) + gap * (len(ims) - 1) + pad * 2
    h = max(im.height for im in ims) + pad * 2
    canvas = Image.new("RGB", (w, h), (12, 12, 15))
    x = pad
    for im in ims:
        canvas.paste(im, (x, pad))
        x += im.width + gap
    canvas.save(out_path, quality=92)
    print(f"  {out_path.name}  {canvas.size}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INSPECT_DIR.mkdir(parents=True, exist_ok=True)
    curated = json.loads(CURATED.read_text(encoding="utf-8"))
    by_vid = {v["video_id"]: v for v in curated}
    vids = sys.argv[1:] or sorted(by_vid.keys())

    rows = []
    for vid in vids:
        v = by_vid.get(vid)
        if v is None:
            print(f"!! {vid}: 不在 21 条标注集，跳过")
            continue
        anno_path = PILOT / vid / "annotation_data.json"
        if not anno_path.exists():
            print(f"!! {vid}: 缺 annotation_data，跳过")
            continue
        segs = rebuild_segs(json.loads(anno_path.read_text(encoding="utf-8")))
        total = v["total_seconds"]
        key = [k for k in v.get("key_frames", []) if k < total]
        ts = ts_frames(segs, total)
        diffs = scene_diffs(vid, total)

        rec = {"video_id": vid, "duration": total, "priority": v.get("visual_priority"),
               "n_ts": len(ts), "ts": ts, "diffs": diffs,
               "key_frames": key, "per_threshold": {}}
        for T in THRESHOLDS:
            scene = sorted(i for i in range(total) if diffs[i] > T)
            union = sorted(set(ts) | set(scene))
            m_ts, m_uni = l1_metrics(ts, key), l1_metrics(union, key)
            recv, miss = recovered_keys(union, ts, key)
            rec["per_threshold"][str(T)] = {
                "n_scene": len(scene), "scene": scene, "n_union": len(union),
                "union": union, "ts_l1": m_ts, "union_l1": m_uni,
                "delta_f1": round(m_uni["f1"] - m_ts["f1"], 4),
                "recovered": recv, "still_miss": miss,
            }
        rows.append(rec)

        # 渲染（主阈值）
        sc = rec["per_threshold"][str(RENDER_T)]
        extra = sorted(set(sc["union"]) - set(ts))
        print(f"\n=== {vid}  {str(v.get('visual_priority')):<8} dur={total}s "
              f"TS={len(ts)}  Union(T={RENDER_T})={sc['n_union']} "
              f"(新增转镜 {len(extra)}) ===")
        render_grid(vid, f"{vid}  TokenSaver ({len(ts)}帧)", ts,
                    INSPECT_DIR / f"{vid}_ts.jpg")
        render_grid(vid, f"{vid}  TS∪转镜 ({sc['n_union']}帧, 黄色=新增)", sc["union"],
                    INSPECT_DIR / f"{vid}_union.jpg", highlight=set(extra))
        render_grid(vid, f"{vid}  人工关键帧 ({len(key)}帧)", key,
                    INSPECT_DIR / f"{vid}_keyframes.jpg")
        build_compare([INSPECT_DIR / f"{vid}_ts.jpg",
                       INSPECT_DIR / f"{vid}_union.jpg"],
                      INSPECT_DIR / f"{vid}_compare.jpg")

    out = OUT_DIR / "ts_scene_union.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存: {out}")

    # 汇总：每档阈值
    print("\n" + "=" * 78)
    print("TS ∪ 转镜取帧 对人工关键帧 L1 影响（Δ = union - ts）")
    print("=" * 78)
    print(f"{'T':<6} {'ΔF1均':>6} {'胜/平/负':>10} {'找回KF':>7} "
          f"{'仍漏KF':>7} {'∪帧均':>6} {'∪/TS':>6}")
    for T in THRESHOLDS:
        t = str(T)
        deltas = [r["per_threshold"][t]["delta_f1"] for r in rows]
        recv = [r["per_threshold"][t]["recovered"] for r in rows]
        miss = [r["per_threshold"][t]["still_miss"] for r in rows]
        n_uni = [r["per_threshold"][t]["n_union"] for r in rows]
        n_ts = [r["n_ts"] for r in rows]
        win = sum(1 for d in deltas if d > 0)
        tie = sum(1 for d in deltas if d == 0)
        avg = sum(deltas) / len(deltas)
        print(f"{T:<6} {avg:>+6.3f} {win:>3}/{tie:>1}/"
              f"{sum(1 for d in deltas if d < 0):>3} "
              f"{sum(len(x) for x in recv):>7} {sum(len(x) for x in miss):>7} "
              f"{sum(n_uni)/len(n_uni):>6.1f} {sum(n_uni)/sum(n_ts):>6.2f}")


if __name__ == "__main__":
    main()
