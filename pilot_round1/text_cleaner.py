"""
DeepSeek V4 Flash text cleaner — ASR typo correction + punctuation restoration.
Uses DeepSeek API (OpenAI-compatible endpoint) for text cleaning.
"""
import os
import json
from typing import Optional


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"  # DeepSeek V4 Flash equivalent

CLEAN_PROMPT = """你是一个中文ASR后处理助手。请对以下电商短视频的口播ASR转录文本进行修正：

1. **修正错别字**：修正ASR识别错误（同音字、近音字错误），恢复正确的产品名词、品牌名、专业术语
2. **补全标点符号**：根据语义添加适当的标点（逗号、句号、问号、感叹号），使文本可读
3. **保持原意**：不要改写、不要润色、不要总结、不要添加原文没有的信息
4. **保持原有风格**：保留口语化表达、感叹词、重复强调等口播特征

只输出修正后的文本，不要添加任何解释或标记。"""


def clean_text(raw_text: str, api_key: str = None, model: str = None) -> str:
    """
    Clean ASR text via DeepSeek V4 Flash.

    Args:
        raw_text: Raw ASR transcript with potential typos
        api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var)
        model: Model name (defaults to deepseek-chat)

    Returns:
        Cleaned text with typos fixed and punctuation added
    """
    if not raw_text or not raw_text.strip():
        return raw_text

    key = api_key or DEEPSEEK_API_KEY
    if not key:
        print("  [WARN] DEEPSEEK_API_KEY not set — skipping text cleaning, using raw text")
        return raw_text

    model_name = model or DEEPSEEK_MODEL

    try:
        from openai import OpenAI

        client = OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": CLEAN_PROMPT},
                {"role": "user", "content": raw_text},
            ],
            temperature=0.1,   # low temp for consistent correction
            max_tokens=2048,
        )

        cleaned = response.choices[0].message.content.strip()
        return cleaned

    except ImportError:
        print("  [WARN] openai package not installed — skipping text cleaning")
        return raw_text
    except Exception as e:
        print(f"  [WARN] DeepSeek API call failed: {e} — falling back to raw text")
        return raw_text


def clean_texts_batch(
    texts: dict[str, str],
    api_key: str = None,
    cache_dir: str = None,
) -> dict[str, str]:
    """
    Clean multiple texts, caching results to avoid re-API calls.

    Args:
        texts: {video_id: raw_text} mapping
        api_key: DeepSeek API key
        cache_dir: Directory to store cleaned text cache

    Returns:
        {video_id: cleaned_text} mapping
    """
    results = {}

    for vid, raw_text in texts.items():
        # Check cache
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{vid}_cleaned.txt")
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    results[vid] = f.read()
                print(f"  [{vid}] Loaded from cache")
                continue

        print(f"  [{vid}] Cleaning via DeepSeek...")
        cleaned = clean_text(raw_text, api_key)

        # Save cache
        if cache_dir and cleaned != raw_text:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{vid}_cleaned.txt")
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(cleaned)

        # Save raw for comparison
        if cache_dir:
            raw_path = os.path.join(cache_dir, f"{vid}_raw.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(raw_text)

        results[vid] = cleaned

    return results
