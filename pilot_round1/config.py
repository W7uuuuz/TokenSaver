"""
Token-Saver Round 1 Pilot — Configuration
All tunable parameters in one place for rapid iteration.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# ============================================================================
# Video I/O
# ============================================================================
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".flv")
AUDIO_SAMPLE_RATE = 16000
AUDIO_TEMP_DIR = "./pilot_output/audio"

# ============================================================================
# Stage 1: ASR — Qwen3 ASR Flash (DashScope) or Whisper fallback
# ============================================================================
ASR_BACKEND = "qwen3"            # "qwen3" | "whisper"
QWEN_ASR_MODEL = "qwen3-asr-flash"
QWEN_ASR_LANGUAGE = "zh"
WHISPER_MODEL = "tiny"           # fallback
WHISPER_LANGUAGE = "zh"
WHISPER_DEVICE = "cpu"

# DeepSeek text cleaning
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_CLEANING_ENABLED = True

# ============================================================================
# Stage 2: Text Density Scoring
# ============================================================================
WINDOW_SIZE_SEC = 5.0        # sliding window for density scoring
WINDOW_STRIDE_SEC = 1.0      # stride (1s resolution)

# 5-dim weights (must sum to 1.0)
# Calibrated via grid search against human key-frame annotations (2026-07-04)
# Surprisal dominates — domain-specific terms (brand names, product claims)
# are the strongest signal for e-commerce key moments.
WEIGHT_TTR = 0.07            # Lexical Density (Type-Token Ratio)
WEIGHT_ENTROPY = 0.12        # Information Entropy H(W)
WEIGHT_SURPRISAL = 0.40      # Surprisal (trigram backoff)
WEIGHT_INFO_POINT = 0.31     # Information-Point Density (entities)
WEIGHT_REDUNDANCY = 0.10     # Redundancy-inverted

# Surprisal model
TRIGRAM_FALLBACK_ALPHA = 0.1  # Laplace smoothing for unseen trigrams

# E-commerce entity patterns (Chinese)
ENTITY_PATTERNS = [
    r'¥\d+(?:\.\d+)?',           # ¥ price
    r'\d+(?:\.\d+)?元',          # X元
    r'\d+(?:\.\d+)?折',          # X折
    r'\d+(?:\.\d+)?[份件个瓶盒袋箱套]',  # quantity + classifier
    r'\d+[gG克千][克gG]?',       # weight
    r'\d+[mM毫][升L升lL]',       # volume
    r'\d+%',                      # percentage
    r'第[一二三四五六七八九十\d]+[名步位]',  # ranking
]

# Filler words (Chinese e-commerce)
FILLER_WORDS = {
    '就是', '然后', '那个', '这个', '嗯', '啊', '呃', '吧',
    '嘛', '呢', '的话', '就是说', '其实', '所以', '因此',
    '那么', '而且', '还有', '或者', '但是', '不过', '呃就是',
}

# ============================================================================
# Routing: Short video handling
# ============================================================================
MIN_DURATION_LOCAL_ROUTING = 60   # videos shorter than this skip per-second local routing

# ============================================================================
# Stage 2: Routing Table
# ============================================================================
# (score_threshold, category_name, frames_per_minute, strategy_label)
ROUTING_TABLE: List[Tuple[float, str, int, str]] = [
    (8.0, "极密 (Ultra-Dense)",  30, "每2s + 运动峰值"),
    (6.0, "密集 (Dense)",        22, "每3s + 场景边界 + 关键词触发"),
    (4.0, "混合 (Mixed)",        12, "场景边界 + 关键词触发 + 均匀插值"),
    (2.0, "稀疏 (Sparse)",        6, "场景边界 + 首尾帧"),
    (0.0, "极少 (Minimal)",       4, "仅场景边界 + 首尾帧"),
]

# Global-local routing
LOCAL_WINDOW_SEC = 10.0         # local sliding window
LOCAL_STRIDE_SEC = 5.0
LOCAL_GLOBAL_DIFF_THRESHOLD = 2.0  # override if local differs from global by >2

# Multi-signal joint trigger weights
TRIGGER_W_TEXT = 0.5
TRIGGER_W_VISUAL = 0.3
TRIGGER_W_AUDIO = 0.2
TRIGGER_THRESHOLD = 0.5

# ============================================================================
# Stage 2: Modality Coupling Detection
# ============================================================================
INSTRUCTION_WORD_LEXICON: Dict[str, List[str]] = {
    "deictic":       ["看这里", "大家看", "注意看", "看一下", "瞧瞧", "瞅瞅", "看这"],
    "comparative":   ["对比一下", "区别在于", "左边", "右边", "之前", "之后", "差别"],
    "demonstrative": ["展示一下", "给大家看", "效果", "实测", "上手", "试试", "演示"],
    "transformative": ["变成", "变身", "变了", "瞬间", "一下子", "马上"],
    "emphatic":       ["重点", "关键是", "核心在于", "必须看", "一定要", "千万"],
}

COUPLING_SCAN_WINDOW_BEFORE_SEC = 2.0
COUPLING_SCAN_WINDOW_AFTER_SEC = 3.0
COUPLING_VISUAL_CHANGE_WINDOW_SEC = 2.0

# SCD parameters
SCD_SIGMA_WINDOW_SEC = 5          # σ_vis lookback/lookahead
SCD_TEXT_VISUAL_THRESHOLD = 0.6   # E_text_norm × σ_vis > threshold → coupling

COUPLING_BOOST = 0.5              # multiply I_text by (1+boost) when coupling

# ============================================================================
# Stage 3: MMR Frame Selection
# ============================================================================
MMR_LAMBDA_REL = 0.7             # relevance vs diversity trade-off (re-calibrated 2026-07-04 on 8 videos)
MMR_TEMPORAL_TAU = 5.0           # temporal decay in sim(f,s)
VISUAL_SIM_WEIGHTS = (0.3, 0.4, 0.3)  # H_vis, motion, objects

# Visual feature extraction
OPTICAL_FLOW_GRID = 16           # Lucas-Kanade grid
OBJECT_DETECTION_EVERY_N = 5     # YOLO every Nth frame
HSV_BINS = (8, 12, 4)            # H, S, V bins for color entropy

# ============================================================================
# Scene Detection
# ============================================================================
SCENE_THRESHOLD = 0.4            # ffmpeg scene change threshold (standard)
SCENE_THRESHOLD_FAST_CUT = 0.2   # lowered threshold for fast-cut videos
MIN_SCENE_SEGMENT_SEC = 2.0      # minimum scene segment duration
SCENE_OPENCV_SIGMA = 1.0        # sigma multiplier for OpenCV scene detection (calibrated 2026-07-04)
FAST_CUT_ABS_THRESHOLD = 5.0     # median frame-diff in first 10s → high absolute motion
FAST_CUT_RATIO_THRESHOLD = 2.0   # p90/median ratio → spiky motion pattern

# ============================================================================
# Visual-Only Mode (no audio track)
# ============================================================================
VISUAL_ONLY_WEIGHT_ENTROPY = 0.5  # weight for H_vis in visual-only composite score
VISUAL_ONLY_WEIGHT_MOTION = 0.5  # weight for motion in visual-only composite score

# ============================================================================
# Output & Visualization
# ============================================================================
OUTPUT_DIR = "./pilot_output"
FIGURE_DPI = 150
FIGURE_WIDTH_INCHES = 16
FIGURE_HEIGHT_INCHES = 10
