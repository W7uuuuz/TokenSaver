"""渲染算法取帧效果：TS / Uniform / 人工关键帧 三组拼图对比.

用 21 条样本的逐秒缩略图（pilot_output/{vid}/thumbnails/t_XXX.jpg）直接渲染，
无需重新解码视频。每条视频输出三张带标题与时间戳的拼图：
  {out}/{vid}_ts.jpg         算法（production increment）取帧
  {out}/{vid}_uniform.jpg    均匀采样取帧（同帧数）
  {out}/{vid}_keyframes.jpg  人工标注关键帧（L1 真值）

用于人工目检：验证算法取的帧到底长什么样、是否和画面关键帧错位。

用法: cd HotMirrorStudio/视频降维技术研究 && PYTHONIOENCODING=utf-8 python tools/render_selected_frames.py [vid1 vid2 ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BACKEND = Path(r"D:\DevWorkSpace\HotMirrorStudio\backend")
sys.path.insert(0, str(BACKEND))
from app.video.token_saver import (  # noqa: E402
    _uniform_sample,
    auto_budget,
    compute_density_scores,
    prejudge_video_type,
    select_frames_adaptive,
)

ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / "pilot_output"
CURATED = PILOT / "annotations_curated.json"
OUT_DIR = PILOT / "frame_inspect"

THUMB_H = 240
COLS = 5
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"


def rebuild_segs(anno_data: dict) -> list[dict]:
    """从 annotation_data 的逐秒 text 重建 ASR 段（相邻 1s 内合并为一段）。"""
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


def load_thumb(vid: str, second: int):
    """第 second 秒的缩略图（t_XXX.jpg，1-based）。"""
    p = PILOT / vid / "thumbnails" / f"t_{second + 1:03d}.jpg"
    if not p.exists():
        return None
    return Image.open(p).convert("RGB")


def render_grid(vid: str, title: str, seconds: list[int], out_path: Path) -> None:
    """把给定秒列表渲染成带标题 + 时间戳的拼图。"""
    items = [(s, load_thumb(vid, s)) for s in seconds]
    items = [(s, im) for s, im in items if im is not None]
    if not items:
        print(f"  !! {vid} 无可用缩略图，跳过")
        return
    n = len(items)
    cols = min(COLS, n)
    rows = (n + cols - 1) // cols

    resized = []
    for _s, im in items:
        w = int(im.width * THUMB_H / im.height)
        resized.append(im.resize((w, THUMB_H), Image.LANCZOS))

    col_ws = []
    for c in range(cols):
        col_ws.append(max((resized[r * cols + c].width for r in range(rows)
                           if r * cols + c < n), default=0))

    pad, label_h, header_h = 6, 18, 44
    grid_w = sum(col_ws) + (cols + 1) * pad
    grid_h = header_h + rows * (THUMB_H + label_h + pad) + pad
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
            y = header_h + pad + r * (THUMB_H + label_h + pad)
            grid.paste(resized[idx], (x0, y))
            tw = draw.textlength(f"t={items[idx][0]}s", font=font_l)
            draw.text((x0 + (col_ws[c] - tw) / 2, y + THUMB_H + 2),
                      f"t={items[idx][0]}s", fill=(200, 200, 200), font=font_l)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path, quality=90)
    print(f"  {out_path.name}  ({n} 帧)  {grid.size}")


def build_compare(vid: str, paths: list[Path], out_path: Path) -> None:
    """并排拼图：算法取帧 | 人工关键帧，一张图同时对比。"""
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
    curated = json.loads(CURATED.read_text(encoding="utf-8"))
    by_vid = {v["video_id"]: v for v in curated}
    vids = sys.argv[1:] or sorted(by_vid.keys())

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
        scores, _ = compute_density_scores(segs, total, return_dimensions=True)
        vtype = prejudge_video_type(segs, total)["video_type"]
        B = auto_budget(total)
        ts = select_frames_adaptive(scores, B, total, video_type=vtype,
                                    budget_mode="increment")
        N = len(ts)
        uni = _uniform_sample(N, total)
        key = [k for k in v.get("key_frames", []) if k < total]

        print(f"\n=== {vid}  {v.get('visual_priority','?'):<8} dur={total}s "
              f"B={B} TS={N} Uni={N} KF={len(key)} vtype={vtype} ===")
        render_grid(vid, f"{vid}  TokenSaver 取帧 ({N}帧)", ts,
                    OUT_DIR / f"{vid}_ts.jpg")
        render_grid(vid, f"{vid}  均匀采样取帧 ({N}帧)", uni,
                    OUT_DIR / f"{vid}_uniform.jpg")
        render_grid(vid, f"{vid}  人工标注关键帧 ({len(key)}帧)", key,
                    OUT_DIR / f"{vid}_keyframes.jpg")
        build_compare(vid, [OUT_DIR / f"{vid}_ts.jpg",
                            OUT_DIR / f"{vid}_keyframes.jpg"],
                      OUT_DIR / f"{vid}_compare.jpg")

    print(f"\n已渲染: {OUT_DIR}")


if __name__ == "__main__":
    main()
