"""为新候选视频构建人工标注数据管道（方向 B 扩样本）。

round1 的 prep_annotation_data.py 依赖 DeconstructSG 输出的 results.json
（composite_scores / coupling_flags / scene_boundaries）。新候选视频来自
data/materials/（仓库根目录），没有 results.json，因此本脚本现场重算：
  1. 现跑 ASR（复用 backend transcriber，缺 ASR 才跑，缓存到 backend/data/asr_cache）
  2. compute_density_scores 算 5D 复合密度（×10 对齐 round1 的 0-10 标尺）
  3. ffmpeg 抽 1fps 缩略图（720 高）
  4. 生成 pilot_output/{vid}/annotation_data.json + annotation_summary.json

标注口径与 round1 一致：人工看图选关键帧（口播 8-15 帧 / 评测 12-20 帧）。

用法: cd HotMirrorStudio/视频降维技术研究 && PYTHONIOENCODING=utf-8 python tools/prep_annotation_v2.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

BACKEND = Path(r"D:\DevWorkSpace\HotMirrorStudio\backend")
sys.path.insert(0, str(BACKEND))

for env_path in [BACKEND / ".env", Path(__file__).resolve().parent.parent / ".env"]:
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from app.audio.transcriber import transcribe  # noqa: E402
from app.video.token_saver import compute_density_scores  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MATERIALS = Path(r"D:\DevWorkSpace\HotMirrorStudio\data\materials")
BACKEND_ASR = Path(r"D:\DevWorkSpace\HotMirrorStudio\data\asr_cache")
OUTPUT_DIR = ROOT / "pilot_output"

# 方向 B 新候选视频（data/materials 中 20-150s、round1 未标注的 19 条）
CANDIDATES = [
    "M-e74cdc74c49c", "M-b22d611e2038", "M-bbf4b6261479",
    "M-21dd90cdfb17", "M-2c696f4a9877", "M-e62a1221d636",
    "M-29b5fbaa2b08", "M-13ead59e0125", "M-93435751b184",
    "M-9e5b204d4fa5", "M-fc907c66c53a", "M-f7eb95196c5f",
    "M-443188070671", "M-9e447c56b729", "M-528bfe313870",
    "M-d012fac68e61", "M-7fc32534a8ef", "M-16ef9e2d688a",
    "M-9f811b808555",
]


def get_video_path(vid: str) -> Path | None:
    vp = MATERIALS / vid / "raw_video.mp4"
    return vp if vp.exists() else None


def load_asr(vid: str) -> dict:
    """加载 ASR（{text, segments, words}）。缺则现场跑 transcriber。"""
    cache = BACKEND_ASR / f"{vid}_asr.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    vpath = get_video_path(vid)
    if vpath is None:
        return {}
    wav_dir = OUTPUT_DIR / "asr_cache" / "qwen_asr" / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    wav = wav_dir / f"{vid}.wav"
    if not wav.exists():
        subprocess.run([
            "ffmpeg", "-y", "-i", str(vpath),
            "-ac", "1", "-ar", "16000", str(wav),
        ], capture_output=True, timeout=180,
           creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    segments = []
    for attempt in range(4):
        try:
            segments = transcribe(wav)  # 写缓存到 BACKEND_ASR
            break
        except Exception as e:
            msg = str(e)
            if attempt < 3 and ("429" in msg or "SSL" in msg or "Max retries" in msg):
                time.sleep(20 * (attempt + 1))
                continue
            raise
    if not segments:
        return {}
    # transcriber 缓存已落盘，重新读取拿 words
    return json.loads(cache.read_text(encoding="utf-8"))


def get_duration(vpath: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(vpath)],
        capture_output=True, text=True, timeout=30)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def extract_thumbnails(vpath: Path, out_dir: Path, total_seconds: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(vpath),
        "-vf", "fps=1,scale=-1:720",
        "-q:v", "2",
        str(out_dir / "t_%03d.jpg"),
    ], capture_output=True, timeout=180,
       creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)


def build_per_second_text(words: list, total_seconds: int) -> list:
    texts = [""] * total_seconds
    for w in words:
        sec = int(w.get("start", 0))
        if 0 <= sec < total_seconds:
            texts[sec] += w.get("word", "")
    return texts


def build_annotation_data(vid: str, vpath: Path) -> dict | None:
    asr = load_asr(vid)
    segs = asr.get("segments", [])
    if not segs:
        print(f"  SKIP {vid}: ASR 为空")
        return None

    duration = get_duration(vpath)
    if duration <= 0:
        duration = max(s["end"] for s in segs)
    total_seconds = max(1, int(round(duration)))

    scores = compute_density_scores(segs, duration)
    comp10 = [round(s * 10, 1) for s in scores]
    comp10 = (comp10 + [0.0] * total_seconds)[:total_seconds]

    words = asr.get("words", [])
    per_second_text = build_per_second_text(words, total_seconds)
    per_second_text = (per_second_text + [""] * total_seconds)[:total_seconds]

    # round1 的 coupling_flags / scene_boundaries 来自 LLM 拆解；新视频无 LLM 输出，
    # 以密度局部低谷近似场景边界（仅作标注提示，不参与评分）
    scene_boundaries = [
        i for i in range(1, total_seconds - 1)
        if comp10[i] <= 3.0 and comp10[i] < comp10[i - 1] and comp10[i] < comp10[i + 1]
    ]

    thumb_dir = OUTPUT_DIR / vid / "thumbnails"
    extract_thumbnails(vpath, thumb_dir, total_seconds)

    seconds_data = []
    for s in range(total_seconds):
        seconds_data.append({
            "second": s,
            "text": per_second_text[s],
            "composite_score": comp10[s],
            "coupling_flag": False,
            "is_scene_boundary": s in scene_boundaries,
            "thumbnail": f"../pilot_output/{vid}/thumbnails/t_{s + 1:03d}.jpg",
        })

    anno = {
        "video_id": vid,
        "title": vid,
        "category": "新样本",
        "total_seconds": total_seconds,
        "seconds": seconds_data,
    }
    data_path = OUTPUT_DIR / vid / "annotation_data.json"
    data_path.write_text(json.dumps(anno, ensure_ascii=False, indent=2), encoding="utf-8")
    return anno


def main():
    existing = {}
    summary_path = OUTPUT_DIR / "annotation_summary.json"
    if summary_path.exists():
        for v in json.loads(summary_path.read_text(encoding="utf-8")):
            existing[v["video_id"]] = v

    # 保留已有条目（round1），新候选视频追加
    rows = [existing[v] for v in existing]
    for vid in CANDIDATES:
        if (OUTPUT_DIR / vid / "annotation_data.json").exists():
            print(f"skip {vid} (已有标注数据)")
            if vid in existing:
                rows.append(existing[vid])
            else:
                anno = json.loads((OUTPUT_DIR / vid / "annotation_data.json")
                                  .read_text(encoding="utf-8"))
                rows.append({
                    "video_id": vid,
                    "label": f"新样本-{anno['total_seconds']}s",
                    "total_seconds": anno["total_seconds"],
                    "data_path": f"../pilot_output/{vid}/annotation_data.json",
                })
            continue
        vpath = get_video_path(vid)
        if vpath is None:
            print(f"skip {vid} (无视频)")
            continue
        print(f"\n=== {vid} ===")
        anno = build_annotation_data(vid, vpath)
        if anno is None:
            continue
        rows.append({
            "video_id": vid,
            "label": f"新样本-{anno['total_seconds']}s",
            "total_seconds": anno["total_seconds"],
            "data_path": f"../pilot_output/{vid}/annotation_data.json",
        })
        print(f"  {vid} {anno['total_seconds']}s → annotation_data.json")

    summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存: {summary_path}  ({len(rows)} 条)")


if __name__ == "__main__":
    main()
