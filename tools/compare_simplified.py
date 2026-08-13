"""
Compare simplified Token-Saver pipeline vs Full pipeline vs Uniform vs Human annotations.
"""
import os, sys, json
from pathlib import Path
from typing import List
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k not in os.environ:
                    os.environ[k] = v

from token_saver import compress_video

ANNOTATIONS_DIR = PROJECT_ROOT / "annotations"
PILOT_OUTPUT = PROJECT_ROOT / "pilot_output"


def get_video_path(video_id: str) -> str:
    videos_dir = PROJECT_ROOT / "round1" / "videos"
    for root, dirs, files in os.walk(str(videos_dir)):
        for f in files:
            if f.startswith(video_id) and f.endswith(".mp4"):
                return os.path.join(root, f)
    return None


def compute_metrics(selected: List[int], key_frames: List[int]) -> dict:
    tolerance = 1
    matched = 0
    for sf in selected:
        for hf in key_frames:
            if abs(sf - hf) <= tolerance:
                matched += 1
                break
    prec = matched / len(selected) if selected else 0
    rec = matched / len(key_frames) if key_frames else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    gaps = np.diff(sorted(selected))
    spread = float(np.std(gaps)) if len(gaps) > 1 else 0
    return {"precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(f1, 3), "temporal_spread": round(spread, 1),
            "matched": matched, "n_selected": len(selected)}


def compute_uniform(total_s: int, budget: int) -> List[int]:
    if budget >= total_s:
        return list(range(total_s))
    step = (total_s - 1) / (budget - 1) if budget > 1 else 0
    return sorted(set(int(step * i) for i in range(budget)))


def main():
    budgets = [10, 20]
    videos = []

    for anno_file in sorted(ANNOTATIONS_DIR.glob("*_keyframes.json")):
        vid = anno_file.stem.replace("_keyframes", "")
        if vid == "M-00fa7a854582":
            continue
        vpath = get_video_path(vid)
        if not vpath:
            continue
        with open(anno_file, encoding="utf-8") as f:
            anno = json.load(f)
        videos.append((vid, vpath, anno))

    print(f"Token-Saver Simplified vs Full Pipeline vs Uniform ({len(videos)} videos)\n")

    all_results = {}

    for budget in budgets:
        print(f"{'='*70}")
        print(f"Budget = {budget}")
        print(f"{'='*70}")
        print(f"{'Video':<22} {'Cat':<6} {'TS-Simp':>8} {'TS-Full':>8} {'Uniform':>8} {'Winner':>8}")
        print("-" * 58)

        ts_wins_simp = ts_wins_full = uni_wins = ties = 0
        simp_f1s = []
        full_f1s = []
        uni_f1s = []

        for vid, vpath, anno in videos:
            total_s = anno["total_seconds"]
            key_frames = anno["key_frames"]
            cat = anno.get("category", "?")

            # Simplified TS
            try:
                result = compress_video(vpath, budget=budget)
                simp_frames = result["frame_indices"]
            except Exception as e:
                print(f"  {vid}: TS-Simp ERROR: {e}")
                simp_frames = compute_uniform(total_s, budget)

            # Full TS from cached results
            results_path = PILOT_OUTPUT / vid / "results.json"
            if results_path.exists():
                with open(results_path, encoding="utf-8") as f:
                    full_data = json.load(f)
                full_frames = full_data["selected_frames"].get(str(budget), [])
            else:
                full_frames = compute_uniform(total_s, budget)

            # Uniform
            uni_frames = compute_uniform(total_s, budget)

            # Metrics
            m_simp = compute_metrics(simp_frames, key_frames)
            m_full = compute_metrics(full_frames, key_frames)
            m_uni = compute_metrics(uni_frames, key_frames)

            simp_f1s.append(m_simp["f1"])
            full_f1s.append(m_full["f1"])
            uni_f1s.append(m_uni["f1"])

            # Determine winner among TS variants
            scores = {"TS-Simp": m_simp["f1"], "TS-Full": m_full["f1"], "Uniform": m_uni["f1"]}
            best = max(scores, key=scores.get)
            if scores[best] == 0:
                winner = "tie"
            else:
                winner = best

            if m_simp["f1"] > m_uni["f1"]:
                ts_wins_simp += 1
            if m_full["f1"] > m_uni["f1"]:
                ts_wins_full += 1
            if m_uni["f1"] > max(m_simp["f1"], m_full["f1"]):
                uni_wins += 1

            print(f"{vid:<22} {cat:<6} {m_simp['f1']:>8.3f} {m_full['f1']:>8.3f} {m_uni['f1']:>8.3f} {winner:>8}")

        print("-" * 58)
        avg_simp = np.mean(simp_f1s) if simp_f1s else 0
        avg_full = np.mean(full_f1s) if full_f1s else 0
        avg_uni = np.mean(uni_f1s) if uni_f1s else 0
        print(f"{'AVERAGE':<22} {'':<6} {avg_simp:>8.3f} {avg_full:>8.3f} {avg_uni:>8.3f}")
        print(f"TS-Simp beats Uniform: {ts_wins_simp}/{len(videos)}, "
              f"TS-Full beats Uniform: {ts_wins_full}/{len(videos)}, "
              f"Uniform beats both: {uni_wins}/{len(videos)}")

        results_per_video = []
        for i, (vid, vpath, anno) in enumerate(videos):
            results_per_video.append({
                "video_id": vid,
                "ts_simp_f1": float(simp_f1s[i]),
                "ts_full_f1": float(full_f1s[i]),
                "uniform_f1": float(uni_f1s[i]),
            })
        all_results[str(budget)] = {
            "per_video": results_per_video,
            "avg_ts_simp": float(avg_simp),
            "avg_ts_full": float(avg_full),
            "avg_uniform": float(avg_uni),
        }
        print()

    # Save
    out_path = PILOT_OUTPUT / "simplified_vs_full.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
