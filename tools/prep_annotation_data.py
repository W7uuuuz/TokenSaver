"""
Extract 1fps thumbnails and generate annotation data JSON for each video.
Run this before opening tools/annotate.html.
"""
import os
import sys
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "pilot_output"
CACHE_DIR = OUTPUT_DIR / "qwen_cache"
ANNOTATIONS_DIR = PROJECT_ROOT / "annotations"


def load_manifest():
    with open(PROJECT_ROOT / "round1" / "manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)


def get_video_path(local_id: str) -> str:
    videos_dir = PROJECT_ROOT / "round1" / "videos"
    for root, dirs, files in os.walk(str(videos_dir)):
        for f in files:
            if f.startswith(local_id) and f.endswith(".mp4"):
                return os.path.join(root, f)
    return None


def extract_thumbnails(video_path: str, output_dir: str, total_seconds: int):
    """Extract 1fps thumbnails at 64px height."""
    os.makedirs(output_dir, exist_ok=True)
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", "fps=1,scale=-1:720",
        "-q:v", "2",
        os.path.join(output_dir, "t_%03d.jpg")
    ]
    subprocess.run(cmd, capture_output=True, timeout=120,
                   creationflags=creationflags)
    print(f"  Thumbnails: {output_dir}")


def load_results(video_id: str) -> dict:
    path = OUTPUT_DIR / video_id / "results.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_qwen_text(video_id: str) -> str:
    """Load Qwen3 ASR text from cache."""
    path = CACHE_DIR / "asr" / f"{video_id}_qwen_asr.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("text", "")
    return ""


def load_qwen_words(video_id: str) -> list:
    """Load Qwen3 word timestamps for per-second text."""
    path = CACHE_DIR / "asr" / f"{video_id}_qwen_asr.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("words", [])
    return []


def build_per_second_text(words: list, total_seconds: int) -> list:
    """Build per-second text from word timestamps."""
    texts = [""] * total_seconds
    for w in words:
        sec = int(w.get("start", 0))
        if 0 <= sec < total_seconds:
            texts[sec] += w.get("word", "")
    return texts


def find_videos_with_results():
    """Find all videos that have results.json in pilot_output."""
    videos = []
    for d in OUTPUT_DIR.iterdir():
        if d.is_dir() and (d / "results.json").exists():
            vid = d.name
            vpath = get_video_path(vid)
            if vpath:
                videos.append((vid, vpath))
    return sorted(videos)


def main():
    videos = find_videos_with_results()
    print(f"Preparing annotation data for {len(videos)} videos...\n")

    for vid, vpath in videos:
        # Results JSON has the accurate total_seconds
        results = load_results(vid)
        total_seconds = results.get("total_seconds", 60)
        category = results.get("routing_distribution", {})
        cat_str = list(category.keys())[0] if category else "?"

        print(f"  [{vid}] ({cat_str}) {total_seconds}s")

        # Extract thumbnails
        thumb_dir = str(OUTPUT_DIR / vid / "thumbnails")
        extract_thumbnails(vpath, thumb_dir, total_seconds)

        # Build per-second data
        words = load_qwen_words(vid)
        per_second_text = build_per_second_text(words, total_seconds)

        composite_scores = results.get("composite_scores", [0] * total_seconds)
        coupling_flags = results.get("coupling_flags", [0] * total_seconds)
        scene_boundaries = results.get("scene_boundaries", [])

        # Pad arrays to total_seconds
        composite_scores = (composite_scores + [0] * total_seconds)[:total_seconds]
        coupling_flags = (coupling_flags + [0] * total_seconds)[:total_seconds]
        per_second_text = (per_second_text + [""] * total_seconds)[:total_seconds]

        # Annotation data
        seconds_data = []
        for s in range(total_seconds):
            seconds_data.append({
                "second": s,
                "text": per_second_text[s],
                "composite_score": round(composite_scores[s], 1),
                "coupling_flag": bool(coupling_flags[s]),
                "is_scene_boundary": s in scene_boundaries,
                "thumbnail": f"../pilot_output/{vid}/thumbnails/t_{s+1:03d}.jpg",
            })

        anno_data = {
            "video_id": vid,
            "title": vid,
            "category": cat_str,
            "total_seconds": total_seconds,
            "seconds": seconds_data,
        }

        # Save
        data_path = OUTPUT_DIR / vid / "annotation_data.json"
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(anno_data, f, ensure_ascii=False, indent=2)
        print(f"  Data: {data_path}")

    # Also create a summary for the HTML page
    summary = []
    for vid, vpath in videos:
        data_path = OUTPUT_DIR / vid / "annotation_data.json"
        if data_path.exists():
            with open(data_path, encoding="utf-8") as f:
                ad = json.load(f)
            summary.append({
                "video_id": vid,
                "title": vid,
                "category": ad.get("category", "?"),
                "total_seconds": ad.get("total_seconds", 60),
                "data_path": f"../pilot_output/{vid}/annotation_data.json",
            })

    with open(OUTPUT_DIR / "annotation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Summary: {OUTPUT_DIR / 'annotation_summary.json'}")
    print(f"Next: Open tools/annotate.html in a browser (via HTTP server)")
    print(f"  cd {PROJECT_ROOT}")
    print(f"  python -m http.server 8080")
    print(f"  Then open: http://localhost:8080/tools/annotate.html")


if __name__ == "__main__":
    main()
