"""
TokenSaver: 面向电商短视频的确定性免训练帧选择。

用法:
    from token_saver import compress_video
    result = compress_video("video.mp4", budget=10)
    print(result["frame_indices"])  # [0, 3, 7, 15, 22, ...]

核心算法见 `token_saver.core`。
"""
from .pipeline import compress_video
from .core import (
    compute_density_scores,
    prejudge_video_type,
    auto_budget,
    select_frames_adaptive,
    select_frames_priority,
    transitions_first,
)

__all__ = [
    "compress_video",
    "compute_density_scores",
    "prejudge_video_type",
    "auto_budget",
    "select_frames_adaptive",
    "select_frames_priority",
    "transitions_first",
]
