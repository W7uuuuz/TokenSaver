"""补跑全帧基线答案：TS(0.7×dur) vs 全帧(dur) 理解保持率实验.

生产协议 L2 同帧数对比（TS vs Uniform 同 0.7×dur）显示 TS 不占优（1-7-0），
因此论文转向成本-收益叙事：TS 用 70% 帧能否达到"全帧(1fps)"的理解水平。

本脚本只补跑**全帧基线**答案（每条视频 1 次 qwen 看图，1fps 全部帧），
TS/Uniform 答案已存于 reeval_l2_prod_summary.json，由配套 judge 脚本离线比较。

全帧拼图不做帧数裁剪（靠 stitch_grid 的 max_side=2000 等比压缩），
否则"全帧"名不副实。

用法: cd HotMirrorStudio/视频降维技术研究 && PYTHONIOENCODING=utf-8 python tools/reeval_l2_fullframe.py [--video M-xxx]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import cv2
import dashscope
from PIL import Image, ImageDraw, ImageFont

BACKEND = Path(r"D:\DevWorkSpace\HotMirrorStudio\backend")
sys.path.insert(0, str(BACKEND))

for env_path in [BACKEND / ".env", Path(__file__).resolve().parent.parent / ".env"]:
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")

from app.video.token_saver import compute_density_scores  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "round1" / "videos"
ASR_DIR = ROOT / "pilot_output" / "qwen_cache" / "asr"
ANN_DIR = ROOT / "annotations"
OUTPUT_DIR = ROOT / "pilot_output" / "reeval_l2"

VL_MODEL = "qwen3.7-plus"

EVAL_QUESTIONS = """请根据这些从视频中提取的关键帧，回答以下问题：
1. 这个视频在卖什么产品？品牌/产品名称是什么？
2. 产品的核心卖点或功能特点有哪些？（列出2-4个）
3. 视频中是否提到了价格、优惠、折扣信息？具体是什么？
4. 视频展示了哪些产品使用场景或产品细节？
5. 这个视频的整体风格属于哪种？（口播讲解 / 产品评测 / 使用演示 / 混剪）

请用中文简洁回答，每个问题2-3句话，直接给出答案不要解释过程。"""

# 生产对齐：视觉 MLLM 同时携带口播转录文本。口播文本辅助内容理解，
# 但保留「严禁通过口播反推画面内容」约束，冲突时以画面为准。
ASR_CONTEXT = """以下是与帧同源的 ASR 口播转录文本（带时间戳），供内容理解参考。
严禁通过口播文本反推画面内容；当口播与画面信息冲突时，以画面为准。
{transcript}"""


def build_transcript(segments: list[dict]) -> str:
    lines = [f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments]
    return "\n".join(lines)


def get_video_path(video_id: str):
    for root, _dirs, files in os.walk(str(VIDEO_DIR)):
        for f in files:
            if f.startswith(video_id) and f.endswith(".mp4"):
                return Path(root) / f
    return None


def load_asr_segments(path: Path) -> tuple[list[dict], float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    sentences = d.get("sentences") or []
    segs = [{"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in sentences if s.get("text")]
    duration = max(s["end"] for s in segs) if segs else 0.0
    return segs, duration


def extract_frames(video_path, frame_indices):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return []
    frames = []
    for idx in sorted(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx * fps))
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        else:
            frames.append(None)
    cap.release()
    return frames


def stitch_grid(frames, timestamps, cols=6, thumb_height=160):
    valid = [(i, f) for i, f in zip(timestamps, frames) if f is not None]
    if not valid:
        return None
    timestamps, frames = zip(*valid)
    n = len(frames)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    thumbs = []
    for f in frames:
        ratio = thumb_height / f.height
        thumbs.append(f.resize((int(f.width * ratio), thumb_height), Image.LANCZOS))
    col_widths = []
    for c in range(cols):
        w = max((thumbs[r * cols + c].width for r in range(rows) if r * cols + c < n),
                default=0)
        col_widths.append(w)
    label_h = 18
    grid_w = sum(col_widths) + (cols + 1) * 4
    grid_h = rows * (thumb_height + 4 + label_h) + 4
    grid = Image.new("RGB", (grid_w, grid_h), (25, 25, 25))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 12)
    except Exception:
        font = ImageFont.load_default()
    for c in range(cols):
        x_offset = 4 + sum(col_widths[:c]) + c * 4
        for r in range(rows):
            idx = r * cols + c
            if idx >= n:
                break
            y = 4 + r * (thumb_height + 4 + label_h)
            grid.paste(thumbs[idx], (x_offset, y))
            label = f"t={timestamps[idx]}s"
            tw = draw.textlength(label, font=font) if hasattr(draw, "textlength") else len(label) * 8
            tx = x_offset + (col_widths[c] - tw) / 2
            draw.text((tx, y + thumb_height + 2), label, fill=(200, 200, 200), font=font)
    max_side = 2000
    if max(grid.size) > max_side:
        scale = max_side / max(grid.size)
        grid = grid.resize((int(grid.width * scale), int(grid.height * scale)),
                           Image.LANCZOS)
    return grid


def call_qwen_vl(image, prompt):
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    messages = [{"role": "user", "content": [
        {"image": f"data:image/jpeg;base64,{img_b64}"}, {"text": prompt},
    ]}]
    response = dashscope.MultiModalConversation.call(model=VL_MODEL, messages=messages)
    if response.status_code == 200:
        return response.output.choices[0].message.content[0]["text"]
    return f"ERROR: code={response.code} msg={response.message}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="已有结果时跳过已完成的视频")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "reeval_l2_fullframe_withasr.json"
    results = {}
    if args.resume and out_path.exists():
        results = json.loads(out_path.read_text(encoding="utf-8"))

    for asr_path in sorted(ASR_DIR.glob("*_asr.json")):
        vid = asr_path.name.replace("_qwen_asr.json", "").replace("_asr.json", "")
        if args.video and args.video not in vid:
            continue
        if args.resume and vid in results:
            print(f"skip {vid} (已有)")
            continue
        ann_path = ANN_DIR / f"{vid}_keyframes.json"
        vpath = get_video_path(vid)
        if not ann_path.exists() or vpath is None:
            continue

        segs, duration = load_asr_segments(asr_path)
        transcript = ASR_CONTEXT.format(transcript=build_transcript(segs))
        scores, _ = compute_density_scores(segs, duration, return_dimensions=True)
        total = len(scores)
        full_frames = list(range(total))  # 1fps 全帧

        print(f"\n=== {vid}  full={len(full_frames)}f (dur={duration:.0f}s) ===")
        imgs = extract_frames(vpath, full_frames)
        valid = [f for f in imgs if f is not None]
        if len(valid) < 2:
            print("  insufficient frames")
            continue
        grid = stitch_grid(imgs, full_frames)
        grid.save(OUTPUT_DIR / f"{vid}_fullframe.jpg", quality=85)
        ans = call_qwen_vl(grid, EVAL_QUESTIONS + "\n\n" + transcript)
        print(f"  fullframe ({len(full_frames)}f): {ans[:60]}...")
        results[vid] = {"full_frames": full_frames, "answer": ans,
                        "duration": duration, "total_seconds": total}
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        time.sleep(1)

    print(f"\n已保存: {out_path}  ({len(results)} 条)")


if __name__ == "__main__":
    main()
