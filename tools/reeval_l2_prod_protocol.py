"""生产协议 L2 MLLM 内容理解评估：auto_budget + increment 选帧，token 对齐.

与 reeval_prod_protocol.py（L1）同一口径，用 qwen3.7-plus 看图回答 5 个内容
问题、DeepSeek 盲审比较 TS vs Uniform（以人工关键帧答案为 Oracle 参考）。

关键设置：
  - 预算 = auto_budget(duration) = max(10, int(duration×0.7))
  - TS 帧 = select_frames_adaptive(budget_mode="increment")（生产算法原样）
  - Uniform 帧 = 同帧数均匀采样（token 完全对齐）
  - MAX_L2_FRAMES=40：长视频（110s→68 帧）拼图超 MLLM 输入上限，TS/Uniform
    对称均匀降采样到 40 帧，不影响两者可比性

用法: cd HotMirrorStudio/视频降维技术研究 && PYTHONIOENCODING=utf-8 python tools/reeval_l2_prod_protocol.py
"""
from __future__ import annotations

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
from openai import OpenAI

BACKEND = Path(r"D:\DevWorkSpace\HotMirrorStudio\backend")
sys.path.insert(0, str(BACKEND))

# 加载 API key（后端 .env）
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
    auto_budget,
    _uniform_sample,
    compute_density_scores,
    prejudge_video_type,
    select_frames_adaptive,
)

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "round1" / "videos"
ASR_DIR = ROOT / "pilot_output" / "qwen_cache" / "asr"
ANN_DIR = ROOT / "annotations"
OUTPUT_DIR = ROOT / "pilot_output" / "reeval_l2"

VL_MODEL = "qwen3.7-plus"
MAX_L2_FRAMES = 40

EVAL_QUESTIONS = """请根据这些从视频中提取的关键帧，回答以下问题：
1. 这个视频在卖什么产品？品牌/产品名称是什么？
2. 产品的核心卖点或功能特点有哪些？（列出2-4个）
3. 视频中是否提到了价格、优惠、折扣信息？具体是什么？
4. 视频展示了哪些产品使用场景或产品细节？
5. 这个视频的整体风格属于哪种？（口播讲解 / 产品评测 / 使用演示 / 混剪）

请用中文简洁回答，每个问题2-3句话，直接给出答案不要解释过程。"""

# 生产对齐：视觉 MLLM 同时携带口播转录文本（visual_deconstruct_agent 把 transcript 作为
# system prompt 传入）。口播文本辅助内容理解，但保留「严禁通过口播反推画面内容」约束，
# 保证画面与口播冲突时以画面为准。
ASR_CONTEXT = """以下是与帧同源的 ASR 口播转录文本（带时间戳），供内容理解参考。
严禁通过口播文本反推画面内容；当口播与画面信息冲突时，以画面为准。
{transcript}"""


def build_transcript(segments: list[dict]) -> str:
    lines = [f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments]
    return "\n".join(lines)

JUDGE_PROMPT = """你是一个视频内容理解评估专家。你需要比较两个MLLM（多模态大模型）对同一视频的回答质量。

关于这个视频的【Oracle答案】（基于人工挑选的关键帧生成，视为参考答案）：
{oracle_answer}

现在有两个答案需要评估：
【答案A】（Token-Saver 自动选帧）：
{ts_answer}

【答案B】（均匀采样选帧）：
{uniform_answer}

请从以下维度比较答案A和答案B：
1. **完整性**：哪个答案覆盖了Oracle中的关键信息（产品、品牌、卖点、价格、场景风格）更多？
2. **准确性**：哪个答案与Oracle答案的信息一致，没有明显错误或幻觉？
3. **信息密度**：哪个答案在相同信息量下更简洁，冗余更少？

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


def call_deepseek_judge(oracle_answer, ts_answer, uniform_answer):
    prompt = JUDGE_PROMPT.format(
        oracle_answer=oracle_answer[:2000],
        ts_answer=ts_answer[:2000],
        uniform_answer=uniform_answer[:2000],
    )
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for asr_path in sorted(ASR_DIR.glob("*_asr.json")):
        vid = asr_path.name.replace("_qwen_asr.json", "").replace("_asr.json", "")
        ann_path = ANN_DIR / f"{vid}_keyframes.json"
        vpath = get_video_path(vid)
        if not ann_path.exists() or vpath is None:
            continue

        anno = json.loads(ann_path.read_text(encoding="utf-8"))
        key = anno["key_frames"]
        segs, duration = load_asr_segments(asr_path)
        transcript = ASR_CONTEXT.format(transcript=build_transcript(segs))
        scores, _ = compute_density_scores(segs, duration, return_dimensions=True)
        vtype = prejudge_video_type(segs, duration)["video_type"]
        total = len(scores)
        oracle_frames = [k for k in key if k < total]

        B = auto_budget(duration)
        ts_frames = select_frames_adaptive(scores, B, total, video_type=vtype,
                                           budget_mode="increment")
        N = len(ts_frames)
        uni_frames = _uniform_sample(N, total)
        # MLLM 输入帧数上限：TS/Uniform 对称均匀降采样
        ts_frames = cap_frames(ts_frames)
        uni_frames = cap_frames(uni_frames)

        print(f"\n=== {vid}  ({anno.get('category','?')})  {duration:.0f}s  "
              f"type={vtype}  budget={B}  TS={len(ts_frames)}f  Uni={len(uni_frames)}f  "
              f"oracle={len(oracle_frames)}f ===")

        if len(oracle_frames) < 2:
            print("  oracle 关键帧不足，跳过本视频")
            continue
        oracle_ans = call_qwen_vl(
            stitch_grid(extract_frames(vpath, oracle_frames), oracle_frames),
            EVAL_QUESTIONS + "\n\n" + transcript)
        print(f"  oracle  ({len(oracle_frames)}f): {oracle_ans[:60]}...")
        time.sleep(1)

        answers = {}
        for method, frame_indices in [("ts", ts_frames), ("uniform", uni_frames)]:
            imgs = extract_frames(vpath, frame_indices)
            valid = [f for f in imgs if f is not None]
            if len(valid) < 2:
                answers[method] = "ERROR: insufficient frames"
                continue
            grid = stitch_grid(imgs, frame_indices)
            grid.save(OUTPUT_DIR / f"{vid}_{method}_prod.jpg", quality=85)
            answers[method] = call_qwen_vl(grid, EVAL_QUESTIONS + "\n\n" + transcript)
            print(f"  {method:<8}({len(frame_indices)}f): {answers[method][:50]}...")
            time.sleep(1)

        if any(a.startswith("ERROR") for a in answers.values()):
            print("  SKIP judge — 有答案缺失")
            continue

        judgment = call_deepseek_judge(oracle_ans, answers["ts"], answers["uniform"])
        v_result = {
            "video_id": vid, "category": anno.get("category", "?"),
            "vtype": vtype, "budget": B, "total_seconds": total,
            "n_frames": {"ts": len(ts_frames), "uniform": len(uni_frames),
                         "oracle": len(oracle_frames)},
            "frames": {"ts": ts_frames, "uniform": uni_frames,
                       "oracle": oracle_frames},
            "answers": {"ts": answers["ts"], "uniform": answers["uniform"],
                        "oracle": oracle_ans},
            "judgment": judgment,
        }
        all_results.append(v_result)
        print(f"  judge: TS={judgment.get('overall_a')} Uni={judgment.get('overall_b')} "
              f"-> {judgment.get('winner')}: {judgment.get('summary','')}")
        time.sleep(1)

    print("\n" + "=" * 65)
    print("L2 生产协议汇总（auto_budget + increment，token 对齐）")
    print("=" * 65)
    if all_results:
        ts_w = sum(1 for r in all_results if r["judgment"]["winner"] == "a")
        uni_w = sum(1 for r in all_results if r["judgment"]["winner"] == "b")
        tie = sum(1 for r in all_results if r["judgment"]["winner"] == "tie")
        avg_ts = sum(r["judgment"].get("overall_a", 0) for r in all_results) / len(all_results)
        avg_uni = sum(r["judgment"].get("overall_b", 0) for r in all_results) / len(all_results)
        print(f"TS {ts_w}-{uni_w}-{tie} (胜-负-平), 平均 TS {avg_ts:.1f} vs Uni {avg_uni:.1f}")
        for r in all_results:
            j = r["judgment"]
            print(f"  {r['video_id']:<16} {r['category']:<8} "
                  f"TS={j.get('overall_a'):>2} Uni={j.get('overall_b'):>2} "
                  f"-> {j.get('winner'):<4} {j.get('summary','')[:40]}")

    out = OUTPUT_DIR / "reeval_l2_prod_summary_withasr.json"
    out.write_text(json.dumps(all_results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()
