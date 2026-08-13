"""主实验：以 MLLM 识别完整视频为基准，对比取帧方法的成本-效果（降本主线 + 增效副线）.

设计（2026-08-07 方法论决议，用户指定）：
  - 论文主线 = 降本：TS / Uniform / 转场优先(TF) 用更少帧回答 L2 问题，成本与效果
    均以「全帧基线（1fps 全部帧喂 MLLM）」为锚点。成本用**真实 API usage** ×
    本地 LLM 网关定价（llm-models.json 中 qwen3.7-plus 1.6/6.4/0.32 元/百万 token），
    效果用 DeepSeek judge 的「保持率」= overall_a(选帧)/overall_b(全帧)。
  - 增效副线：两套问题集，content(+ASR) 主证降本（ASR 兜底 → 天花板效应）、
    visual(纯视觉无 ASR) 副证帧集本身对画面理解的贡献。
  - 方法 × 档位：TS / Uniform / TF(T=0.28) × budget_ratio {0.5, 0.7, 0.9}
    （每档位帧数对齐：TF 与 TS 同预算，转镜优先重分配）。
  - 基线：content 全帧答案 + visual 全帧答案，均**重新生成以捕获 usage**。

数据源：
  - round1/videos/ 8 条视频 mp4（时长从 ASR 段推导，与 fullframe 基线一致）
  - pilot_output/qwen_cache/asr/*_qwen_asr.json  ASR 段
  - 帧统一从 mp4 解码（逐秒取帧，与 pilot 逐秒缩略图同路径），缓存小图供拼图。

用法: cd HotMirrorStudio/视频降维技术研究 && PYTHONIOENCODING=utf-8 python tools/reeval_l2_cost_effect.py [--resume] [--video M-xxx]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import cv2
import dashscope
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(line_buffering=True)  # 实时进度，不缓冲

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

from app.video.frame_extractor import compute_scene_diffs  # noqa: E402
from app.video.token_saver import (  # noqa: E402
    _uniform_sample,
    auto_budget,
    compute_density_scores,
    prejudge_video_type,
    select_frames_adaptive,
    transitions_first,
)

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "round1" / "videos"
ASR_DIR = ROOT / "pilot_output" / "qwen_cache" / "asr"
OUT_DIR = ROOT / "pilot_output" / "reeval_l2"
OUT_FILE = OUT_DIR / "reeval_l2_cost_effect.json"

VL_MODEL = "qwen3.7-plus"
# 本地 LLM 网关定价（用户填写，元/百万 token）——论文成本口径的唯一来源
PRICING = {
    "qwen3.7-plus": {"input": 1.6, "output": 6.4, "cache_hit": 0.32},
}

BUDGET_RATIOS = [0.50, 0.70, 0.90]
METHODS = ["ts", "uniform", "tf"]
QSETS = ["content", "visual"]
MAX_FRAMES = 40          # 选帧点帧数上限（与 L2 生产协议一致）
TF_THRESHOLD = 0.28      # 转镜判定阈值（B3 校准）
THUMB_HEIGHT = 160
WORKERS = int(os.environ.get("WORKERS", "20"))
RETRIES = 8

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


# ── 数据准备 ──


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


THUMB_CACHE: dict[str, dict[int, Image.Image]] = {}


def extract_thumbs(vpath: Path, total: int) -> dict[int, Image.Image]:
    """顺序读帧，逐秒取一帧降采样（与 pilot 逐秒缩略图同路径）."""
    cap = cv2.VideoCapture(str(vpath))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            return {}
        sample_every = max(1, round(fps))
        thumbs: dict[int, Image.Image] = {}
        idx = 0
        sec = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_every == 0 and sec < total:
                im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                w = int(im.width * THUMB_HEIGHT / im.height)
                thumbs[sec] = im.resize((w, THUMB_HEIGHT), Image.LANCZOS)
                sec += 1
            idx += 1
        return thumbs
    finally:
        cap.release()


def stitch_grid(vid: str, seconds: list[int], cols: int = 5,
                max_side: int = 2000) -> Image.Image | None:
    """帧拼图。max_side=4000 用于全帧基线（多帧网格需保缩略图可读，
    否则每帧缩略图过小 → MLLM 反读不出细节，见 2026-08-07 探针）。"""
    cache = THUMB_CACHE[vid]
    items = [(s, cache[s]) for s in seconds if s in cache]
    if not items:
        return None
    n = len(items)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    col_ws = [max((items[r * cols + c][1].width for r in range(rows)
                   if r * cols + c < n), default=0) for c in range(cols)]
    pad, label_h = 4, 20
    grid_w = sum(col_ws) + (cols + 1) * pad
    grid_h = rows * (THUMB_HEIGHT + label_h + pad) + pad
    grid = Image.new("RGB", (grid_w, grid_h), (24, 24, 28))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 13)
    except Exception:
        font = ImageFont.load_default()
    for c in range(cols):
        x0 = pad + sum(col_ws[:c]) + c * pad
        for r in range(rows):
            idx = r * cols + c
            if idx >= n:
                break
            y = pad + r * (THUMB_HEIGHT + label_h + pad)
            grid.paste(items[idx][1], (x0, y))
            tw = draw.textlength(f"t={items[idx][0]}s", font=font)
            draw.text((x0 + (col_ws[c] - tw) / 2, y + THUMB_HEIGHT + 2),
                      f"t={items[idx][0]}s", fill=(200, 200, 200), font=font)
    if max(grid.size) > max_side:
        scale = max_side / max(grid.size)
        grid = grid.resize((int(grid.width * scale), int(grid.height * scale)),
                           Image.LANCZOS)
    return grid


def cap_frames(frame_indices: list[int], max_n: int = MAX_FRAMES) -> list[int]:
    if len(frame_indices) <= max_n:
        return list(frame_indices)
    idx = _uniform_sample(max_n, len(frame_indices))
    return [frame_indices[i] for i in idx]


# ── API 调用（带 usage 捕获 + 重试） ──


def _as_dict(obj) -> dict:
    if isinstance(obj, dict):
        return obj
    return obj.__dict__ if hasattr(obj, "__dict__") else {}


def _usage(resp) -> dict:
    u = getattr(resp, "usage", None)
    if u is None:
        return {}
    ud = _as_dict(u)
    out = {k: v for k, v in ud.items()
           if k in ("input_tokens", "output_tokens", "image_tokens") and v is not None}
    itd = _as_dict(ud.get("input_tokens_details"))
    if itd:
        out["input_tokens_details"] = {k: v for k, v in itd.items()
                                       if k in ("image_tokens", "text_tokens")
                                       and v is not None}
    ptd = _as_dict(ud.get("prompt_tokens_details"))
    if ptd:
        out["prompt_tokens_details"] = {k: v for k, v in ptd.items()
                                        if k in ("cached_tokens",) and v is not None}
    return out


def _call_qwen_vl(image: Image.Image, prompt: str) -> tuple[str, dict]:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    messages = [{"role": "user", "content": [
        {"image": f"data:image/jpeg;base64,{img_b64}"}, {"text": prompt},
    ]}]
    resp = dashscope.MultiModalConversation.call(model=VL_MODEL, messages=messages)
    if resp.status_code == 200:
        return resp.output.choices[0].message.content[0]["text"], _usage(resp)
    raise RuntimeError(f"qwen code={resp.code} msg={resp.message}")


def cost_of(usage: dict) -> float:
    """真实 usage × 本地网关定价（元/百万 token）."""
    it = usage.get("input_tokens", 0) or 0
    ot = usage.get("output_tokens", 0) or 0
    ptd = usage.get("prompt_tokens_details") or {}
    cached = ptd.get("cached_tokens", 0) or 0
    p = PRICING[VL_MODEL]
    return ((it - cached) * p["input"] + cached * p["cache_hit"] + ot * p["output"]) / 1e6


def _call_judge(ans_a: str, ans_b: str, n_a: int, n_b: int) -> dict:
    prompt = JUDGE_PROMPT.format(
        n_a=n_a, n_b=n_b, answer_a=ans_a[:2000], answer_b=ans_b[:2000])
    resp = deepseek_client.chat.completions.create(
        model="deepseek-chat", messages=[{"role": "user", "content": prompt}],
        temperature=0.0)
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3]
    return json.loads(raw)


def with_retry(fn, *args, label: str = "", max_tries: int = RETRIES):
    last = None
    for i in range(max_tries):
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001
            last = e
            text = str(e)
            is_throttle = ("Throttling" in text or "429" in text
                           or "Too many" in text or "rate limit" in text.lower())
            delay = min(10 * (2 ** i), 60) if is_throttle else 2 * (i + 1)
            print(f"  !! {label} 重试 {i + 1}/{max_tries} (等{delay}s): {text[:80]}")
            time.sleep(delay)
    raise RuntimeError(f"{label} 失败: {last}")


def _run_pool(jobs, worker, max_workers):
    """并发执行 jobs（worker(job)->key,value），返回 dict，遇错即中止并抛错."""
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(worker, j) for j in jobs]
        done = 0
        for fut in as_completed(futs):
            key, value = fut.result()
            results[key] = value
            done += 1
            if done % 5 == 0 or done == len(jobs):
                print(f"    [{done}/{len(jobs)}] {key}")
    return results


# ── 主流程 ──


def process_video(vid: str) -> dict:
    vpath = get_video_path(vid)
    asr_path = ASR_DIR / f"{vid}_qwen_asr.json"
    if vpath is None or not asr_path.exists():
        raise RuntimeError(f"{vid}: 缺 mp4 或 ASR")

    segs, duration = load_asr_segments(asr_path)
    transcript = ASR_CONTEXT.format(transcript=build_transcript(segs))
    scores, _ = compute_density_scores(segs, duration, return_dimensions=True)
    vtype = prejudge_video_type(segs, duration)["video_type"]
    total = len(scores)
    diffs = list(compute_scene_diffs(vpath, total_seconds=total))
    if len(diffs) < total:
        diffs += [0.0] * (total - len(diffs))

    print(f"\n=== {vid}  dur={duration:.0f}s type={vtype} total={total} ===")
    print("  提取逐秒帧缓存 ...")
    THUMB_CACHE[vid] = extract_thumbs(vpath, total)

    # 各方法 × 各档位的帧集（帧数对齐：同预算，转镜优先重分配）
    frame_sets: dict[tuple[str, float], list[int]] = {}
    for ratio in BUDGET_RATIOS:
        B = max(1, int(auto_budget(duration) * ratio))
        ts = select_frames_adaptive(scores, B, total, video_type=vtype,
                                    budget_mode="increment")
        uni = _uniform_sample(len(ts), total)
        tf = transitions_first(diffs, len(ts), total, threshold=TF_THRESHOLD)
        frame_sets[("ts", ratio)] = cap_frames(ts)
        frame_sets[("uniform", ratio)] = cap_frames(uni)
        frame_sets[("tf", ratio)] = cap_frames(tf)

    prompts = {
        "content": EVAL_QUESTIONS + "\n\n" + transcript,
        "visual": EVAL_VISUAL_QUESTIONS,
    }

    # 1) 基线（全帧）——重新生成以捕获 usage
    baseline_jobs = []
    for qset in QSETS:
        baseline_jobs.append((vid, qset, list(range(total))))
    print("  [基线] 全帧 qwen 作答 ...")

    def run_baseline(job):
        _vid, qset, seconds = job
        grid = stitch_grid(_vid, seconds, max_side=4000)
        txt, usage = with_retry(_call_qwen_vl, grid, prompts[qset],
                                label=f"{_vid}/baseline/{qset}")
        return (qset,), {"n_frames": len(seconds), "answer": txt,
                         "usage": usage, "cost_yuan": round(cost_of(usage), 5)}

    baselines = _run_pool(baseline_jobs, run_baseline, max_workers=WORKERS)

    # 2) 方法点答案
    ans_jobs = []
    for qset in QSETS:
        for method in METHODS:
            for ratio in BUDGET_RATIOS:
                ans_jobs.append((vid, qset, method, ratio))
    print(f"  [选帧] {len(ans_jobs)} 次 qwen 作答 ...")

    def run_answer(job):
        _vid, qset, method, ratio = job
        seconds = frame_sets[(method, ratio)]
        grid = stitch_grid(_vid, seconds)
        txt, usage = with_retry(_call_qwen_vl, grid, prompts[qset],
                                label=f"{_vid}/{qset}/{method}/r{ratio}")
        return (qset, method, ratio), {"n_frames": len(seconds), "answer": txt,
                                       "usage": usage,
                                       "cost_yuan": round(cost_of(usage), 5)}

    answers = _run_pool(ans_jobs, run_answer, max_workers=WORKERS)

    # 3) judge：选帧答案 vs 对应全帧基线
    judge_jobs = []
    for qset in QSETS:
        for method in METHODS:
            for ratio in BUDGET_RATIOS:
                a = answers[(qset, method, ratio)]
                b = baselines[(qset,)]
                judge_jobs.append((vid, qset, method, ratio,
                                   a["answer"], b["answer"],
                                   a["n_frames"], b["n_frames"]))
    print(f"  [judge] {len(judge_jobs)} 次 ...")

    def run_judge(job):
        _vid, qset, method, ratio, ans_a, ans_b, n_a, n_b = job
        j = with_retry(_call_judge, ans_a, ans_b, n_a, n_b,
                       label=f"judge/{_vid}/{qset}/{method}/r{ratio}")
        keep = j["overall_a"] / j["overall_b"] if j.get("overall_b") else 0.0
        return (qset, method, ratio), {**j, "retention": round(keep, 3)}

    judges = _run_pool(judge_jobs, run_judge, max_workers=WORKERS)

    # 4) 组装
    rec = {"video_id": vid, "vtype": vtype, "duration": round(duration, 1),
           "total": total, "baselines": {q: baselines[(q,)] for q in QSETS},
           "points": []}
    for qset in QSETS:
        for method in METHODS:
            for ratio in BUDGET_RATIOS:
                a = answers[(qset, method, ratio)]
                rec["points"].append({
                    "qset": qset, "method": method, "ratio": ratio,
                    "n_frames": a["n_frames"], "answer": a["answer"],
                    "usage": a["usage"], "cost_yuan": a["cost_yuan"],
                    "compression": round(1 - a["n_frames"] / total, 3),
                    "judge": judges[(qset, method, ratio)],
                })
    rec["points"].sort(key=lambda p: (p["qset"], p["method"], p["ratio"]))
    return rec


def summarize(results: dict) -> None:
    print("\n" + "=" * 78)
    print("成本-效果汇总（保持率 vs 全帧基线；成本 = 真实usage×本地定价 元/次调用）")
    print("=" * 78)
    for qset, qname in (("content", "内容理解(+ASR) 主证降本"), ("visual", "纯视觉(无ASR) 副证增效")):
        print(f"\n【{qname}】")
        header = (f"  {'方法':<8}{'档位':<6}{'帧数':>5}{'压缩率':>8}{'成本¥':>10}"
                  f"{'token':>9}{'保持率':>8}")
        print(header)
        rows = []
        for method in METHODS:
            for ratio in BUDGET_RATIOS:
                pts = [p for v in results.values() for p in v["points"]
                       if p["qset"] == qset and p["method"] == method
                       and p["ratio"] == ratio]
                if not pts:
                    continue
                nf = sum(p["n_frames"] for p in pts) / len(pts)
                comp = sum(p["compression"] for p in pts) / len(pts)
                cost = sum(p["cost_yuan"] for p in pts) / len(pts)
                keep = sum(p["judge"]["retention"] for p in pts) / len(pts)
                tok = sum((p["usage"].get("input_tokens", 0) or 0)
                          for p in pts) / len(pts)
                rows.append((method, ratio, nf, comp, cost, tok, keep))
        for method, ratio, nf, comp, cost, tok, keep in rows:
            print(f"  {method:<8}{ratio:<6}{nf:>5.0f}{comp:>8.1%}{cost:>10.4f}"
                  f"{tok:>9.0f}{keep:>8.3f}")
        # 全帧基线成本（content/visual 各一条，取该 qset 的成本均值）
        bcosts = [v["baselines"][qset]["cost_yuan"] for v in results.values()]
        btok = [(v["baselines"][qset]["usage"].get("input_tokens", 0) or 0)
                for v in results.values()]
        print(f"  {'fullframe':<8}{'—':<6}{sum(v['total'] for v in results.values())/len(results):>5.0f}"
              f"{'—':>8}{sum(bcosts)/len(bcosts):>10.4f}{sum(btok)/len(btok):>9.0f}{1.0:>8.3f}")

        # 降本主线：同档位成本对齐下，方法间保持率差 + 相对全帧的成本比例
        print("\n  降本主线（TS 为主）：")
        for ratio in BUDGET_RATIOS:
            row_ts = next(r for r in rows if r[0] == "ts" and r[1] == ratio)
            row_tf = next(r for r in rows if r[0] == "tf" and r[1] == ratio)
            row_uni = next(r for r in rows if r[0] == "uniform" and r[1] == ratio)
            bcost = sum(bcosts) / len(bcosts)
            print(f"    r={ratio:.2f}  成本 ¥{row_ts[4]:.4f} (全帧 {bcost:.4f} 的 "
                  f"{row_ts[4]/bcost:.0%})  "
                  f"TS保持率 {row_ts[6]:.3f} | TF {row_tf[6]:.3f} | Uni {row_uni[6]:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--video", default=None, help="只跑指定视频前缀")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    if args.resume and OUT_FILE.exists():
        results = json.loads(OUT_FILE.read_text(encoding="utf-8"))["videos"]
        print(f"resume: 已有 {len(results)} 条")

    # 用全帧基线文件确定 8 条视频 ID
    ff = json.loads((OUT_DIR / "reeval_l2_fullframe_withasr.json")
                    .read_text(encoding="utf-8"))
    vids = sorted(ff.keys())
    if args.video:
        vids = [v for v in vids if args.video in v]
    print(f"样本: {len(vids)} 条  | 并发: {WORKERS}  | 模型: {VL_MODEL}  "
          f"| 档位: {BUDGET_RATIOS}  | MAX_FRAMES: {MAX_FRAMES}")

    for vid in vids:
        if args.resume and vid in results:
            print(f"skip {vid} (已有)")
            continue
        try:
            rec = process_video(vid)
        except Exception as e:  # noqa: BLE001
            print(f"\n!! {vid} 失败: {e}")
            continue
        results[vid] = rec
        OUT_FILE.write_text(
            json.dumps({"meta": {
                "vl_model": VL_MODEL, "pricing": PRICING,
                "budget_ratios": BUDGET_RATIOS, "methods": METHODS,
                "qsets": QSETS, "tf_threshold": TF_THRESHOLD,
                "max_frames": MAX_FRAMES,
            }, "videos": results}, ensure_ascii=False, indent=1),
            encoding="utf-8")

    print(f"\n已保存: {OUT_FILE}  ({len(results)} 条)")
    if results:
        summarize(results)


if __name__ == "__main__":
    main()
