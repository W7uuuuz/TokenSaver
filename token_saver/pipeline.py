"""
TokenSaver 端到端管道：ASR → 五维文本密度评分 → 自适应帧选择。

核心算法见 `token_saver.core`（与生产实现一致），本模块只负责接线：
读视频时长 → ASR 转写 → 评分 → 预判类型 → 自适应选帧。
"""
import os
import sys
from pathlib import Path
from typing import Optional, Dict

from . import core

# 允许从同级的 pilot_round1 导入 ASR 模块
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pilot_round1.asr_qwen import transcribe_video  # noqa: E402


def compress_video(
    video_path: str,
    budget: Optional[int] = None,
    asr_cache_dir: Optional[str] = None,
) -> Dict:
    """
    把视频压缩到 `budget` 个关键帧索引（秒级，0-indexed）。

    Args:
        video_path: 视频文件路径（.mp4 等）。
        budget: 帧预算。None 时按视频时长自动计算（duration × 0.7，下限 10）。
        asr_cache_dir: ASR 缓存目录（默认 ./pilot_output/qwen_cache/asr）。

    Returns:
        Dict:
          - frame_indices: 选中帧索引（秒，升序）
          - composite_scores: 逐秒信息密度评分（0-10）
          - total_seconds: 视频时长
          - full_text: 完整 ASR 转写
          - video_type: 口播 / 评测（停顿比预判）
          - budget: 实际帧预算
    """
    import cv2

    # 视频时长
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0:
        fps = 30.0
    total_seconds = int(total_frames / fps)

    # ASR 转写（句级时间戳）
    if asr_cache_dir is None:
        asr_cache_dir = str(_PROJECT_ROOT / "pilot_output" / "qwen_cache" / "asr")
    os.makedirs(asr_cache_dir, exist_ok=True)

    asr_data = transcribe_video(video_path, cache_dir=asr_cache_dir)
    full_text = asr_data.get("text", "")
    sentences = asr_data.get("sentences", [])

    # 五维文本密度评分（[0,1] 区间）
    scores = core.compute_density_scores(sentences, total_seconds)

    # 视频类型预判（口播 / 评测）
    prej = core.prejudge_video_type(sentences, total_seconds)
    video_type = prej["video_type"]

    # 帧预算：None → 按时长自适应
    if budget is None:
        budget = core.auto_budget(total_seconds)

    # 自适应帧选择（CV 自检 + 均匀基线 + 算法增量 + 覆盖兜底）
    frame_indices = core.select_frames_adaptive(
        scores, budget, total_seconds, video_type=video_type
    )

    return {
        "frame_indices": frame_indices,
        "composite_scores": [round(s * 10, 2) for s in scores],
        "total_seconds": total_seconds,
        "full_text": full_text,
        "video_type": video_type,
        "budget": budget,
    }
