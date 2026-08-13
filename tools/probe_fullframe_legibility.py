"""探针：全帧基线可读性测试.

问题：保持率普遍 >1.0，怀疑是 max_side=2000 把多帧网格压小、每帧缩略图太糊，
qwen3.7-plus 反而读不出全帧基线的细节。本探针在 max_side=4000 下重新生成
全帧基线答案（visual 无 ASR 口径），对比旧答案的细节丰富度与 judge 保持率，
判断是否应该把主实验基线改成可读网格。

用法: PYTHONIOENCODING=utf-8 python tools/probe_fullframe_legibility.py M-9a12ab3f34c9
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import dashscope
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

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
from openai import OpenAI  # noqa: E402
deepseek_client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "round1" / "videos"
ASR_DIR = ROOT / "pilot_output" / "qwen_cache" / "asr"
OUT_DIR = ROOT / "pilot_output" / "reeval_l2"
VL_MODEL = "qwen3.7-plus"

EVAL_VISUAL_QUESTIONS = """请仅根据这些从视频中提取的关键帧（不要猜测口播内容），回答以下视觉问题：
1. 产品的颜色、材质、外观特征是什么？请具体描述你看到的。
2. 产品在什么环境/场景中展示？（如直播间背景、室内实景、棚拍、户外等）
3. 视频中如何演示或使用产品？（实物展示 / 试用演示 / 画面对应讲解 / 文字说明等）
4. 画面中有哪些其他视觉元素？（人物、道具、背景摆设、装饰元素等）
5. 画面中有哪些文字、logo、贴片、角标信息？（字幕、价格贴、品牌logo、优惠标签等）

请用中文简洁回答，每个问题2-3句话，直接给出答案不要解释过程。"""

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


def get_video_path(video_id: str):
    for root, _dirs, files in os.walk(str(VIDEO_DIR)):
        for f in files:
            if f.startswith(video_id) and f.endswith(".mp4"):
                return Path(root) / f
    return None


def extract_thumbs(vpath: Path, total: int, thumb_h: int = 160):
    cap = cv2.VideoCapture(str(vpath))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return {}
    every = max(1, round(fps))
    thumbs = {}
    idx = sec = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every == 0 and sec < total:
            im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            w = int(im.width * thumb_h / im.height)
            thumbs[sec] = im.resize((w, thumb_h), Image.LANCZOS)
            sec += 1
        idx += 1
    cap.release()
    return thumbs


def stitch_grid(items, cols=5, max_side=2000, label=True):
    """items: list[(sec, PIL)]"""
    if not items:
        return None
    n = len(items)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    resized = [im for _s, im in items]
    col_ws = [max((resized[r * cols + c].width for r in range(rows)
                   if r * cols + c < n), default=0) for c in range(cols)]
    pad, label_h = 4, (20 if label else 0)
    grid_w = sum(col_ws) + (cols + 1) * pad
    grid_h = rows * (THUMB_H + label_h + pad) + pad
    grid = Image.new("RGB", (grid_w, grid_h), (24, 24, 28))
    if label:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(grid)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 13)
        except Exception:
            font = ImageFont.load_default()
    else:
        draw = None
    for c in range(cols):
        x0 = pad + sum(col_ws[:c]) + c * pad
        for r in range(rows):
            idx = r * cols + c
            if idx >= n:
                break
            y = pad + r * (THUMB_H + label_h + pad)
            grid.paste(resized[idx], (x0, y))
            if draw:
                tw = draw.textlength(f"t={items[idx][0]}s", font=font)
                draw.text((x0 + (col_ws[c] - tw) / 2, y + THUMB_H + 2),
                          f"t={items[idx][0]}s", fill=(200, 200, 200), font=font)
    if max(grid.size) > max_side:
        scale = max_side / max(grid.size)
        grid = grid.resize((int(grid.width * scale), int(grid.height * scale)),
                           Image.LANCZOS)
    return grid


THUMB_H = 160


def call_qwen(image, prompt):
    import base64
    from io import BytesIO
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    resp = dashscope.MultiModalConversation.call(
        model=VL_MODEL,
        messages=[{"role": "user", "content": [
            {"image": f"data:image/jpeg;base64,{b64}"}, {"text": prompt}]}])
    if resp.status_code == 200:
        u = getattr(resp, "usage", None)
        usage = {}
        if u is not None:
            usage["input"] = getattr(u, "input_tokens", 0)
            itd = getattr(u, "input_tokens_details", None)
            if itd is not None and isinstance(itd, dict):
                usage["image"] = itd.get("image_tokens", 0)
        return resp.output.choices[0].message.content[0]["text"], usage
    raise RuntimeError(f"qwen code={resp.code} msg={resp.message}")


def judge(a, b, n_a, n_b):
    prompt = JUDGE_PROMPT.format(n_a=n_a, n_b=n_b, answer_a=a[:2000], answer_b=b[:2000])
    resp = deepseek_client.chat.completions.create(
        model="deepseek-chat", messages=[{"role": "user", "content": prompt}],
        temperature=0.0)
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3]
    return json.loads(raw)


def main():
    vid = sys.argv[1]
    data = json.loads((OUT_DIR / "reeval_l2_cost_effect.json").read_text(encoding="utf-8"))
    rec = data["videos"][vid]
    total = rec["total"]
    vpath = get_video_path(vid)
    thumbs = extract_thumbs(vpath, total)
    items = sorted(thumbs.items())

    print(f"=== {vid} total={total} ===")
    old_ans = rec["baselines"]["visual"]["answer"]
    old_usage = rec["baselines"]["visual"]["usage"]
    print(f"旧全帧(2000px): image_tokens={old_usage.get('input_tokens_details', {}).get('image_tokens')} input={old_usage.get('input_tokens')}")

    new_ans, usage = call_qwen(stitch_grid(items, max_side=4000),
                               EVAL_VISUAL_QUESTIONS)
    print(f"新全帧(4000px): image_tokens={usage.get('image')} input={usage.get('input')}")
    print(f"新答案字数: {len(new_ans)}  旧答案字数: {len(old_ans)}")
    print("\n--- 新(4000px)全帧答案 ---")
    print(new_ans[:500])
    print("\n--- 旧(2000px)全帧答案 ---")
    print(old_ans[:500])

    # 对比 ts@0.5 答案在新旧基线下的保持率
    ts05 = next(p for p in rec["points"] if p["qset"] == "visual"
                and p["method"] == "ts" and p["ratio"] == 0.5)
    j_new = judge(ts05["answer"], new_ans, ts05["n_frames"], total)
    j_old = judge(ts05["answer"], old_ans, ts05["n_frames"], total)
    print(f"\nts@0.5 保持率: 旧基线 {ts05['judge']['retention']} "
          f"(overall {ts05['judge']['overall_a']}/{ts05['judge']['overall_b']})")
    print(f"           新基线 {j_new['overall_a']}/{j_new['overall_b']} "
          f"= {j_new['overall_a']/j_new['overall_b']:.3f}  winner={j_new['winner']}")
    print(f"           新基线重审 {j_old['overall_a']}/{j_old['overall_b']} "
          f"(同旧答案重judge，检验judge噪声)")


if __name__ == "__main__":
    main()
