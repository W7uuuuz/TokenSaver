"""人工标注帧 MLLM 评估：TS / Uniform / 人工关键帧 三组帧集 × 两套问题集.

背景（2026-08-07）：用户视觉目检后重新定性——人工关键帧是「预算极度紧张时的叙事
逻辑安全清单」，转镜头帧占主导；均匀采样已能保证叙事逻辑。此前 L2 把人工标注帧当作
Oracle 参考，本实验改为把人工标注帧当成一种「取帧方法」本身，检验它对 MLLM 理解的
贡献是否真的优于 TS / Uniform。若人工标注帧并不明显胜出，则佐证「帧集价值≈叙事覆盖，
人工标注只是成本昂贵的均匀采样」的结论。

设计：
  - 21 条样本（pilot_output/annotations_curated.json），帧一律用逐秒缩略图
    （pilot_output/{vid}/thumbnails/t_XXX.jpg），无需解码 mp4。
  - 三组帧集：
      TS      生产算法 select_frames_adaptive(budget_mode="increment")
      Uniform 同帧数均匀采样
      Human   人工标注关键帧（annotations_curated.json）
    长帧集统一 cap 到 MAX_FRAMES=40（MLLM 输入上限），三组对称处理。
  - 两套问题集（用户选定「两套都跑」）：
      content 内容理解 5 问 + ASR 转录（生产对齐）
      visual  纯视觉 5 问，不带 ASR（检验帧本身对画面理解贡献）
  - qwen3.7-plus 作答 → DeepSeek 无 Oracle 配对盲审（TS vs Human、Uniform vs Human）。

并发：ThreadPoolExecutor 高并发跑 MLLM 与 judge（qwen 支持高并发，成本换时间），
每次调用带指数退避重试。WORKERS 环境变量可调（默认 10）。

用法: cd HotMirrorStudio/视频降维技术研究 && PYTHONIOENCODING=utf-8 python tools/reeval_human_anno.py
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import dashscope
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

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
OUT_DIR = PILOT / "reeval_human_anno"

VL_MODEL = "qwen3.7-plus"
MAX_FRAMES = 40
# qwen 对并发在途请求有硬上限：126 全开曾触发 Throttling.Concurrency 崩溃；
# 探针显示 40 并发小图安全。默认 32（大拼图更稳），显式 WORKERS 可覆盖。
WORKERS = int(os.environ.get("WORKERS", "32"))
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

# 纯视觉问题集：不带 ASR，强制 MLLM 只从帧本身作答，检验帧集对画面理解的贡献。
EVAL_VISUAL_QUESTIONS = """请仅根据这些从视频中提取的关键帧（不要猜测口播内容），回答以下视觉问题：
1. 产品的颜色、材质、外观特征是什么？请具体描述你看到的。
2. 产品在什么环境/场景中展示？（如直播间背景、室内实景、棚拍、户外等）
3. 视频中如何演示或使用产品？（实物展示 / 试用演示 / 画面对应讲解 / 文字说明等）
4. 画面中有哪些其他视觉元素？（人物、道具、背景摆设、装饰元素等）
5. 画面中有哪些文字、logo、贴片、角标信息？（字幕、价格贴、品牌logo、优惠标签等）

请用中文简洁回答，每个问题2-3句话，直接给出答案不要解释过程。"""

JUDGE_PAIR_PROMPT = """你是一个视频内容理解评估专家。两个MLLM分别用不同的关键帧集合看了同一段视频，对同一组问题给出答案。你需要在没有参考答案的情况下，判断哪个答案更可靠。

【答案A】：
{ans_a}

【答案B】：
{ans_b}

请从以下维度比较答案A和答案B：
1. **完整性**：哪个答案覆盖的视频信息更多、更全面？
2. **准确性**：哪个答案更可信，没有明显的猜测、空泛或幻觉？
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


# ── 数据准备 ──


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


def build_transcript(segments: list[dict]) -> str:
    lines = [f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments]
    return "\n".join(lines)


def load_thumb(vid: str, second: int):
    """第 second 秒的缩略图（t_XXX.jpg，1-based）。"""
    p = PILOT / vid / "thumbnails" / f"t_{second + 1:03d}.jpg"
    if not p.exists():
        return None
    return Image.open(p).convert("RGB")


def stitch_grid(seconds: list[int], vid: str, thumb_height: int = 160,
                cols: int = 5) -> Image.Image:
    """把秒列表渲染成带时间戳的拼图（全部来自逐秒缩略图）。"""
    items = [(s, im) for s in seconds if (im := load_thumb(vid, s)) is not None]
    if not items:
        raise ValueError(f"{vid} 无可用缩略图")
    n = len(items)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    resized = []
    for _s, im in items:
        w = int(im.width * thumb_height / im.height)
        resized.append(im.resize((w, thumb_height), Image.LANCZOS))
    col_ws = []
    for c in range(cols):
        col_ws.append(max((resized[r * cols + c].width for r in range(rows)
                           if r * cols + c < n), default=0))
    pad, label_h = 4, 20
    grid_w = sum(col_ws) + (cols + 1) * pad
    grid_h = rows * (thumb_height + label_h + pad) + pad
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
            y = pad + r * (thumb_height + label_h + pad)
            grid.paste(resized[idx], (x0, y))
            tw = draw.textlength(f"t={items[idx][0]}s", font=font)
            draw.text((x0 + (col_ws[c] - tw) / 2, y + thumb_height + 2),
                      f"t={items[idx][0]}s", fill=(200, 200, 200), font=font)
    return grid


def _workers(n: int) -> int:
    """WORKERS=0（默认）时全任务并发；显式设置则限流。"""
    return n if WORKERS == 0 else min(WORKERS, n)


def cap_frames(frame_indices: list[int], max_n: int = MAX_FRAMES) -> list[int]:
    if len(frame_indices) <= max_n:
        return list(frame_indices)
    idx = _uniform_sample(max_n, len(frame_indices))
    return [frame_indices[i] for i in idx]


# ── API 调用（带重试） ──


def _call_qwen_vl(image: Image.Image, prompt: str) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    messages = [{"role": "user", "content": [
        {"image": f"data:image/jpeg;base64,{img_b64}"}, {"text": prompt},
    ]}]
    resp = dashscope.MultiModalConversation.call(model=VL_MODEL, messages=messages)
    if resp.status_code == 200:
        return resp.output.choices[0].message.content[0]["text"]
    raise RuntimeError(f"qwen code={resp.code} msg={resp.message}")


def _call_judge(ans_a: str, ans_b: str) -> dict:
    prompt = JUDGE_PAIR_PROMPT.format(ans_a=ans_a[:2000], ans_b=ans_b[:2000])
    resp = deepseek_client.chat.completions.create(
        model="deepseek-chat", messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
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
            # 限流长退避（10→20→40→60s）让在途请求排空，避免重试风暴打崩
            delay = min(10 * (2 ** i), 60) if is_throttle else 2 * (i + 1)
            print(f"  !! {label} 重试 {i + 1}/{max_tries} (等{delay}s): {text[:80]}")
            time.sleep(delay)
    raise RuntimeError(f"{label} 失败: {last}")


# ── 主流程 ──


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curated = json.loads(CURATED.read_text(encoding="utf-8"))
    print(f"样本: {len(curated)} 条  | 并发: {WORKERS}  | 模型: {VL_MODEL}  "
          f"| MAX_FRAMES: {MAX_FRAMES}\n")

    # 1. 本地准备：帧集 + 拼图 + 提示词
    jobs = []  # (vid, qset, method, grid, prompt)
    meta = {}
    for v in sorted(curated, key=lambda x: x["video_id"]):
        vid = v["video_id"]
        anno_path = PILOT / vid / "annotation_data.json"
        if not anno_path.exists():
            print(f"!! {vid}: 缺 annotation_data，跳过")
            continue
        segs = rebuild_segs(json.loads(anno_path.read_text(encoding="utf-8")))
        total = v["total_seconds"]
        key = [k for k in v.get("key_frames", []) if k < total]
        scores, _ = compute_density_scores(segs, total, return_dimensions=True)
        vtype = prejudge_video_type(segs, total)["video_type"]
        B = auto_budget(total)
        ts = select_frames_adaptive(scores, B, total, video_type=vtype,
                                    budget_mode="increment")
        N = len(ts)
        uni = _uniform_sample(N, total)

        ts, uni, key = (cap_frames(ts), cap_frames(uni), cap_frames(key))
        frame_sets = {"ts": ts, "uniform": uni, "human": key}
        meta[vid] = {
            "category": v.get("category"), "vtype": vtype,
            "duration": total, "budget": B,
            "n_frames": {m: len(f) for m, f in frame_sets.items()},
            "frames": frame_sets,
        }
        print(f"{vid:<18} {str(v.get('category')):<4} {vtype:<6} "
              f"{total:>3}s B={B:>3} TS={len(ts):>2} Uni={len(uni):>2} "
              f"Human={len(key):>2}")

        transcript = ASR_CONTEXT.format(transcript=build_transcript(segs))
        grids = {m: stitch_grid(f, vid) for m, f in frame_sets.items()}
        for m in ("ts", "uniform", "human"):
            jobs.append(("content", m, grids[m],
                         EVAL_QUESTIONS + "\n\n" + transcript, vid))
            jobs.append(("visual", m, grids[m], EVAL_VISUAL_QUESTIONS, vid))

    # 2. 高并发 MLLM 作答
    print(f"\nMLLM 调用: {len(jobs)} 次  (并发 {_workers(len(jobs))})")
    answers: dict[tuple, str] = {}

    def run_mllm(job):
        qset, m, grid, prompt, vid = job
        txt = with_retry(_call_qwen_vl, grid, prompt, label=f"{vid}/{qset}/{m}")
        return (vid, qset, m), txt

    with ThreadPoolExecutor(max_workers=_workers(len(jobs))) as ex:
        futs = [ex.submit(run_mllm, j) for j in jobs]
        done = 0
        for fut in as_completed(futs):
            key, txt = fut.result()
            answers[key] = txt
            done += 1
            print(f"  [{done}/{len(jobs)}] {key[0]}/{key[1]}/{key[2]}: "
                  f"{txt[:50]}...")

    # 3. 高并发 DeepSeek 配对盲审
    comparisons = []
    for vid in meta:
        for qset in ("content", "visual"):
            for label, other in (("ts", "ts"), ("uniform", "uniform")):
                a = answers.get((vid, qset, "human"))
                b = answers.get((vid, qset, other))
                if a and b and not a.startswith("ERROR") and not b.startswith("ERROR"):
                    comparisons.append((vid, qset, other, a, b))
    print(f"\nJudge 调用: {len(comparisons)} 次  (并发 {_workers(len(comparisons))})")
    judge_res: dict[tuple, dict] = {}

    def run_judge(cmp):
        vid, qset, other, a, b = cmp
        j = with_retry(_call_judge, a, b, label=f"judge/{vid}/{qset}/{other}")
        return (vid, qset, other), j

    with ThreadPoolExecutor(max_workers=_workers(len(comparisons))) as ex:
        futs = [ex.submit(run_judge, c) for c in comparisons]
        done = 0
        for fut in as_completed(futs):
            key, j = fut.result()
            judge_res[key] = j
            done += 1
            print(f"  [{done}/{len(comparisons)}] {key[0]}/{key[1]}/"
                  f"human_vs_{key[2]}: Human={j.get('overall_a')} "
                  f"{key[2]}={j.get('overall_b')} -> {j.get('winner')}")

    # 4. 组装结果 + 汇总
    results = []
    for vid, m in meta.items():
        rec = {"video_id": vid, **m,
               "answers": {}, "judges": {}}
        for qset in ("content", "visual"):
            rec["answers"][qset] = {meth: answers.get((vid, qset, meth), "")
                                    for meth in ("ts", "uniform", "human")}
            rec["judges"][qset] = {
                "human_vs_ts": judge_res.get((vid, qset, "ts")),
                "human_vs_uniform": judge_res.get((vid, qset, "uniform")),
            }
        results.append(rec)

    out = OUT_DIR / "reeval_human_anno.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n已保存: {out}")

    print("\n" + "=" * 68)
    print("人工标注帧评估汇总（Human vs TS / Human vs Uniform）")
    print("=" * 68)
    for qset, qname in (("content", "内容理解(+ASR)"), ("visual", "纯视觉(无ASR)")):
        print(f"\n【{qname}】")
        for other in ("ts", "uniform"):
            rows = [r for r in results if r["judges"][qset][f"human_vs_{other}"]]
            h_w = sum(1 for r in rows
                      if r["judges"][qset][f"human_vs_{other}"]["winner"] == "a")
            o_w = sum(1 for r in rows
                      if r["judges"][qset][f"human_vs_{other}"]["winner"] == "b")
            tie = sum(1 for r in rows
                      if r["judges"][qset][f"human_vs_{other}"]["winner"] == "tie")
            avg_h = sum(r["judges"][qset][f"human_vs_{other}"]["overall_a"]
                        for r in rows) / len(rows) if rows else 0
            avg_o = sum(r["judges"][qset][f"human_vs_{other}"]["overall_b"]
                        for r in rows) / len(rows) if rows else 0
            print(f"  Human vs {other:<8}  Human胜 {h_w:>2} / {other}胜 {o_w:>2} "
                  f"/ 平 {tie:>2}  (n={len(rows)})   平均 Human {avg_h:.1f} vs "
                  f"{other} {avg_o:.1f}")


if __name__ == "__main__":
    main()
