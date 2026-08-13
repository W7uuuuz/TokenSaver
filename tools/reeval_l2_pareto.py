"""A-研究：压缩率-保持率帕累托曲线.

对 round1 的 8 条视频（已有全帧基线答案），在多个压缩档位下用 TS 选帧回答
L2 内容问题，与全帧基线比较理解保持率。回答"压缩率越高，密度集中选帧的
理解保持率是否比均匀采样下降更平缓"。

压缩档位：budget_ratio ∈ {0.50, 0.60, 0.70, 0.80, 0.90} × auto_budget(duration)。
TS 用生产 increment 模式；对照 Uniform 同帧数均匀采样。
保持率 = judge(选帧答案, 全帧答案).overall_a / overall_b。

数据源：
  - reeval_l2_prod_summary_withasr.json  (TS/Uniform 0.7×dur 已有答案)
  - reeval_l2_fullframe_withasr.json     (全帧基线答案)
对 0.7 档直接复用已有答案（省一半 qwen 调用），其余档位现跑。

用法: cd HotMirrorStudio/视频降维技术研究 && PYTHONIOENCODING=utf-8 python tools/reeval_l2_pareto.py [--resume]
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
from openai import OpenAI
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
deepseek_client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

from app.video.token_saver import (  # noqa: E402
    _uniform_sample,
    auto_budget,
    compute_density_scores,
    prejudge_video_type,
    select_frames_adaptive,
)

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "round1" / "videos"
ASR_DIR = ROOT / "pilot_output" / "qwen_cache" / "asr"
OUTPUT_DIR = ROOT / "pilot_output" / "reeval_l2"

VL_MODEL = "qwen3.7-plus"
MAX_L2_FRAMES = 40
BUDGET_RATIOS = [0.50, 0.60, 0.70, 0.80, 0.90]

EVAL_QUESTIONS = """请根据这些从视频中提取的关键帧，回答以下问题：
1. 这个视频在卖什么产品？品牌/产品名称是什么？
2. 产品的核心卖点或功能特点有哪些？（列出2-4个）
3. 视频中是否提到了价格、优惠、折扣信息？具体是什么？
4. 视频展示了哪些产品使用场景或产品细节？
5. 这个视频的整体风格属于哪种？（口播讲解 / 产品评测 / 使用演示 / 混剪）

请用中文简洁回答，每个问题2-3句话，直接给出答案不要解释过程。"""

ASR_CONTEXT = """以下是与帧同源的 ASR 口播转录文本（带时间戳），供内容理解参考。
严禁通过口播文本反推画面内容；当口播与画面信息冲突时，以画面为准。
{transcript}"""

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


def build_transcript(segments: list[dict]) -> str:
    return "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments)


def load_asr_segments(path: Path) -> tuple[list[dict], float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    sentences = d.get("sentences") or []
    segs = [{"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in sentences if s.get("text")]
    duration = max(s["end"] for s in segs) if segs else 0.0
    return segs, duration


def get_video_path(video_id: str):
    for root, _dirs, files in os.walk(str(VIDEO_DIR)):
        for f in files:
            if f.startswith(video_id) and f.endswith(".mp4"):
                return Path(root) / f
    return None


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


def stitch_grid(frames, timestamps, cols=5, thumb_height=200):
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
    label_h = 22
    grid_w = sum(col_widths) + (cols + 1) * 4
    grid_h = rows * (thumb_height + 4 + label_h) + 4
    grid = Image.new("RGB", (grid_w, grid_h), (25, 25, 25))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 14)
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


def cap_frames(frame_indices, max_n=MAX_L2_FRAMES):
    if len(frame_indices) <= max_n:
        return list(frame_indices)
    idx = _uniform_sample(max_n, len(frame_indices))
    return [frame_indices[i] for i in idx]


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


def call_judge(answer_a, answer_b, n_a, n_b):
    prompt = JUDGE_PROMPT.format(
        n_a=n_a, n_b=n_b, answer_a=answer_a[:2000], answer_b=answer_b[:2000])
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat", messages=[{"role": "user", "content": prompt}],
        temperature=0.0)
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3]
    return json.loads(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "reeval_l2_pareto.json"
    results = {}
    if args.resume and out_path.exists():
        results = json.loads(out_path.read_text(encoding="utf-8"))

    summary = json.loads((OUTPUT_DIR / "reeval_l2_prod_summary_withasr.json")
                         .read_text(encoding="utf-8"))
    fullframe = json.loads((OUTPUT_DIR / "reeval_l2_fullframe_withasr.json")
                           .read_text(encoding="utf-8"))
    existing_070 = {r["video_id"]: r for r in summary}

    for vid, ff in fullframe.items():
        if vid in results:
            continue
        asr_path = ASR_DIR / f"{vid}_qwen_asr.json"
        vpath = get_video_path(vid)
        if not asr_path.exists() or vpath is None:
            continue
        segs, duration = load_asr_segments(asr_path)
        transcript = ASR_CONTEXT.format(transcript=build_transcript(segs))
        scores, _ = compute_density_scores(segs, duration, return_dimensions=True)
        vtype = prejudge_video_type(segs, duration)["video_type"]
        total = len(scores)
        full_ans = ff["answer"]
        n_full = len(ff["full_frames"])

        print(f"\n=== {vid}  dur={duration:.0f}s type={vtype} ===")
        per_video = {"video_id": vid, "category": "?", "vtype": vtype,
                     "duration": round(duration, 1), "total": total,
                     "full_answer": full_ans, "n_full": n_full,
                     "points": []}

        for ratio in BUDGET_RATIOS:
            B = max(1, int(auto_budget(duration) * ratio))
            ts_frames = select_frames_adaptive(
                scores, B, total, video_type=vtype, budget_mode="increment")
            uni_frames = _uniform_sample(len(ts_frames), total)
            ts_frames = cap_frames(ts_frames)
            uni_frames = cap_frames(uni_frames)

            # 0.7 档复用已有生产答案（TS/Uniform 同口径）
            if ratio == 0.70 and vid in existing_070:
                ts_ans = existing_070[vid]["answers"]["ts"]
                uni_ans = existing_070[vid]["answers"]["uniform"]
            else:
                ts_ans = call_qwen_vl(
                    stitch_grid(extract_frames(vpath, ts_frames), ts_frames),
                    EVAL_QUESTIONS + "\n\n" + transcript)
                time.sleep(0.8)
                uni_ans = call_qwen_vl(
                    stitch_grid(extract_frames(vpath, uni_frames), uni_frames),
                    EVAL_QUESTIONS + "\n\n" + transcript)
                time.sleep(0.8)

            ts_j = call_judge(ts_ans, full_ans, len(ts_frames), n_full)
            time.sleep(0.5)
            uni_j = call_judge(uni_ans, full_ans, len(uni_frames), n_full)
            time.sleep(0.5)

            ts_keep = ts_j["overall_a"] / ts_j["overall_b"] if ts_j["overall_b"] else 0
            uni_keep = uni_j["overall_a"] / uni_j["overall_b"] if uni_j["overall_b"] else 0
            per_video["points"].append({
                "ratio": ratio, "budget": B,
                "n_ts": len(ts_frames), "n_uni": len(uni_frames),
                "compression": 1 - len(ts_frames) / total,
                "ts_keeprate": round(ts_keep, 3),
                "uni_keeprate": round(uni_keep, 3),
                "ts_score": ts_j["overall_a"], "uni_score": uni_j["overall_a"],
                "full_score": ts_j["overall_b"], "winner_ts": ts_j["winner"],
                "winner_uni": uni_j["winner"],
            })
            print(f"  r={ratio:.2f} B={B}  TS({len(ts_frames)}f)保持率{ts_keep:.2f} "
                  f"({ts_j['overall_a']}/{ts_j['overall_b']})  "
                  f"Uni({len(uni_frames)}f)保持率{uni_keep:.2f}")

        results[vid] = per_video
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                            encoding="utf-8")

    print(f"\n已保存: {out_path}  ({len(results)} 条)")

    # 汇总：按档位聚合平均保持率
    if results:
        print("\n" + "=" * 70)
        print("压缩档位汇总（平均理解保持率 vs 全帧）")
        print("=" * 70)
        for ratio in BUDGET_RATIOS:
            ts_k = [v["ts_keeprate"] for v in results.values()
                    for p in v["points"] if p["ratio"] == ratio]
            uni_k = [v["uni_keeprate"] for v in results.values()
                     for p in v["points"] if p["ratio"] == ratio]
            comp = [v["compression"] for v in results.values()
                    for p in v["points"] if p["ratio"] == ratio]
            if ts_k:
                print(f"  档位{ratio:.2f} (压缩率{(sum(comp)/len(comp))*100:.0f}%): "
                      f"TS 保持率 {sum(ts_k)/len(ts_k):.3f}  |  "
                      f"Uniform {sum(uni_k)/len(uni_k):.3f}")


if __name__ == "__main__":
    main()
