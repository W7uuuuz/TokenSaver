"""
MLLM evaluation for Simplified Token-Saver pipeline vs Uniform vs Oracle.
Uses token_saver.compress_video() for frame selection.
"""
import os, sys, json, base64, time
from pathlib import Path
from io import BytesIO
from typing import List, Optional

import cv2
from PIL import Image, ImageDraw, ImageFont
import dashscope
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                if key not in os.environ:
                    os.environ[key] = val

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

from token_saver import compress_video

OUTPUT_DIR = PROJECT_ROOT / "pilot_output" / "round2_eval_simplified"
ANNOTATIONS_DIR = PROJECT_ROOT / "annotations"

EVAL_QUESTIONS = """请根据这些从视频中提取的关键帧，回答以下问题：
1. 这个视频在卖什么产品？品牌/产品名称是什么？
2. 产品的核心卖点或功能特点有哪些？（列出2-4个）
3. 视频中是否提到了价格、优惠、折扣信息？具体是什么？
4. 视频展示了哪些产品使用场景或产品细节？
5. 这个视频的整体风格属于哪种？（口播讲解 / 产品评测 / 使用演示 / 混剪）

请用中文简洁回答，每个问题2-3句话，直接给出答案不要解释过程。"""

JUDGE_PROMPT = """你是一个视频内容理解评估专家。你需要比较两个MLLM（多模态大模型）对同一视频的回答质量。

关于这个视频的【Oracle答案】（基于人工挑选的关键帧生成，视为参考答案）：
{oracle_answer}

现在有两个答案需要评估：
【答案A】（Token-Saver简化版自动选帧）：
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


def get_video_path(video_id: str) -> Optional[str]:
    videos_dir = PROJECT_ROOT / "round1" / "videos"
    for root, _dirs, files in os.walk(str(videos_dir)):
        for f in files:
            if f.startswith(video_id) and f.endswith(".mp4"):
                return os.path.join(root, f)
    return None


def extract_frames(video_path: str, frame_indices: List[int]) -> List[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return []
    frames = []
    for idx in sorted(frame_indices):
        frame_num = int(idx * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        else:
            frames.append(None)
    cap.release()
    return frames


def stitch_grid(frames: List[Image.Image], timestamps: List[int],
                cols: int = 5, thumb_height: int = 200) -> Image.Image:
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
        w = max((thumbs[r * cols + c].width for r in range(rows) if r * cols + c < n), default=0)
        col_widths.append(w)
    label_h = 22
    grid_w = sum(col_widths) + (cols + 1) * 4
    grid_h = rows * (thumb_height + 4 + label_h) + 4
    grid = Image.new('RGB', (grid_w, grid_h), (25, 25, 25))
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
            tw = draw.textlength(label, font=font) if hasattr(draw, 'textlength') else len(label) * 8
            tx = x_offset + (col_widths[c] - tw) / 2
            draw.text((tx, y + thumb_height + 2), label, fill=(200, 200, 200), font=font)
    return grid


def call_qwen_vl(image: Image.Image, prompt: str, model: str = "qwen3.6-flash") -> str:
    buf = BytesIO()
    image.save(buf, format='JPEG', quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    messages = [{
        "role": "user",
        "content": [
            {"image": f"data:image/jpeg;base64,{img_b64}"},
            {"text": prompt},
        ]
    }]
    response = dashscope.MultiModalConversation.call(model=model, messages=messages)
    if response.status_code == 200:
        return response.output.choices[0].message.content[0]["text"]
    else:
        return f"ERROR: code={response.code} msg={response.message}"


def call_deepseek_judge(oracle_answer: str, ts_answer: str, uniform_answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        oracle_answer=oracle_answer[:2000],
        ts_answer=ts_answer[:2000],
        uniform_answer=uniform_answer[:2000],
    )
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3]
    return json.loads(raw)


def compute_uniform_indices(total_seconds: int, budget: int) -> List[int]:
    if budget >= total_seconds:
        return list(range(total_seconds))
    step = (total_seconds - 1) / (budget - 1) if budget > 1 else 0
    indices = [int(step * i) for i in range(budget)]
    return sorted(set(min(i, total_seconds - 1) for i in indices))


def main():
    budget = 10

    videos = []
    for anno_file in sorted(ANNOTATIONS_DIR.glob("*_keyframes.json")):
        video_id = anno_file.stem.replace("_keyframes", "")
        if video_id == "M-00fa7a854582":
            continue
        with open(anno_file, encoding="utf-8") as f:
            anno = json.load(f)
        vpath = get_video_path(video_id)
        if not vpath:
            continue
        videos.append((video_id, vpath, anno))

    print(f"Simplified TS MLLM Evaluation — {len(videos)} videos @ B={budget}\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = []

    for vid, vpath, anno in videos:
        total_s = anno["total_seconds"]
        oracle_frames = anno["key_frames"]
        cat = anno.get("category", "?")

        # Run simplified pipeline
        print(f"[{vid}] cat={cat} {total_s}s — running simplified TS...")
        try:
            result = compress_video(vpath, budget=budget)
            ts_frames = result["frame_indices"]
        except Exception as e:
            print(f"  ERROR: {e}")
            ts_frames = compute_uniform_indices(total_s, budget)

        uni_frames = compute_uniform_indices(total_s, budget)
        print(f"  TS-Simp={len(ts_frames)}f, Uni={len(uni_frames)}f, Oracle={len(oracle_frames)}f")

        answers = {}
        for method, frame_indices in [
            ("oracle", oracle_frames),
            ("ts_simp", ts_frames),
            ("uniform", uni_frames),
        ]:
            imgs = extract_frames(vpath, frame_indices)
            valid = [f for f in imgs if f is not None]
            if len(valid) < 2:
                print(f"  {method}: only {len(valid)} frames — skip")
                answers[method] = "ERROR: insufficient frames"
                continue

            grid = stitch_grid(imgs, frame_indices)
            grid_path = OUTPUT_DIR / f"{vid}_{method}_b{budget}.jpg"
            grid.save(grid_path, quality=85)

            print(f"  Calling Qwen-VL for {method}...")
            answer = call_qwen_vl(grid, EVAL_QUESTIONS)
            answers[method] = answer

            answer_path = OUTPUT_DIR / f"{vid}_{method}_b{budget}_answer.txt"
            with open(answer_path, "w", encoding="utf-8") as f:
                f.write(f"Video: {vid}  Method: {method}  Budget: {budget}\n")
                f.write(f"Frames ({len(frame_indices)}): {frame_indices}\n\n{answer}")

            preview = answer[:120].replace("\n", " ")
            print(f"    -> {preview}...")
            time.sleep(1)

        # Judge: TS-Simp vs Uniform
        if all(a and not a.startswith("ERROR") for a in answers.values()):
            print(f"  DeepSeek judging...")
            try:
                judgment = call_deepseek_judge(
                    answers["oracle"], answers["ts_simp"], answers["uniform"]
                )
                v_result = {
                    "video_id": vid,
                    "category": cat,
                    "budget": budget,
                    "total_seconds": total_s,
                    "n_frames": {
                        "oracle": len(oracle_frames),
                        "ts_simp": len(ts_frames),
                        "uniform": len(uni_frames),
                    },
                    "frames": {
                        "oracle": oracle_frames,
                        "ts_simp": ts_frames,
                        "uniform": uni_frames,
                    },
                    "judgment": judgment,
                }
                all_results.append(v_result)
                ts_score = judgment.get("overall_a", "?")
                uni_score = judgment.get("overall_b", "?")
                winner = judgment.get("winner", "?")
                print(f"    TS-Simp={ts_score} vs Uni={uni_score} -> {winner}: {judgment.get('summary', '')}")
            except Exception as e:
                print(f"    Judge error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  SKIP judging — missing answers")
        print()

    # Summary
    print("=" * 65)
    print("Simplified TS MLLM Evaluation — Summary")
    print("=" * 65)

    ts_wins = sum(1 for r in all_results if r.get("judgment", {}).get("winner") == "a")
    uni_wins = sum(1 for r in all_results if r.get("judgment", {}).get("winner") == "b")
    ties = sum(1 for r in all_results if r.get("judgment", {}).get("winner") == "tie")

    print(f"{'Video':<22} {'Cat':<6} {'TS-Simp':>8} {'Uni':>8} {'Winner':>8}")
    print("-" * 56)
    for r in all_results:
        j = r.get("judgment", {})
        ts = j.get("overall_a", "?")
        uni = j.get("overall_b", "?")
        w = j.get("winner", "?")
        print(f"{r['video_id']:<22} {r['category']:<6} {ts:>8} {uni:>8} {w:>8}")

    print("-" * 56)
    print(f"TS-Simp wins: {ts_wins}  |  Uniform wins: {uni_wins}  |  Ties: {ties}")

    for cat in ["口播", "评测"]:
        cat_results = [r for r in all_results if cat in str(r.get("category", ""))]
        if cat_results:
            ts_w = sum(1 for r in cat_results if r.get("judgment", {}).get("winner") == "a")
            uni_w = sum(1 for r in cat_results if r.get("judgment", {}).get("winner") == "b")
            tie = sum(1 for r in cat_results if r.get("judgment", {}).get("winner") == "tie")
            avg_ts = sum(r.get("judgment", {}).get("overall_a", 0) for r in cat_results) / len(cat_results)
            avg_uni = sum(r.get("judgment", {}).get("overall_b", 0) for r in cat_results) / len(cat_results)
            print(f"  {cat} ({len(cat_results)} videos): TS-Simp={avg_ts:.1f} vs Uni={avg_uni:.1f} | W:{ts_w} L:{uni_w} T:{tie}")

    summary_path = OUTPUT_DIR / "round2_simplified_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nFull results: {OUTPUT_DIR}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
