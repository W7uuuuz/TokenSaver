"""
Compare Token-Saver frame selection vs Uniform sampling vs human annotations.
Computes precision/recall/F1 of selected frames against human key frame ground truth.
No API calls needed — this is the "simplified" evaluation using annotation overlap.
"""
import os
import sys
import json
from pathlib import Path
from typing import List, Dict
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "pilot_output"
ANNOTATIONS_DIR = PROJECT_ROOT / "annotations"


def load_annotations(video_id: str) -> dict:
    path = ANNOTATIONS_DIR / f"{video_id}_keyframes.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_results(video_id: str) -> dict:
    path = OUTPUT_DIR / video_id / "results.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def uniform_sampling(total_seconds: int, budget: int) -> List[int]:
    """Uniform sampling: pick frames evenly spaced across the video."""
    if budget <= 0 or total_seconds <= 0:
        return []
    if budget >= total_seconds:
        return list(range(total_seconds))
    step = (total_seconds - 1) / (budget - 1) if budget > 1 else 0
    frames = [int(round(i * step)) for i in range(budget)]
    return sorted(set(frames))


def compute_metrics(selected: List[int], key_frames: List[int], total: int) -> dict:
    sel_set = set(selected)
    key_set = set(key_frames)
    inter = sel_set & key_set

    precision = len(inter) / len(sel_set) if sel_set else 0.0
    recall = len(inter) / len(key_set) if key_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Temporal spread: std of inter-frame gaps
    sorted_frames = sorted(selected)
    if len(sorted_frames) >= 2:
        gaps = [sorted_frames[i+1] - sorted_frames[i] for i in range(len(sorted_frames)-1)]
        spread = float(np.std(gaps))
    else:
        spread = 0.0

    # Coupling coverage
    coupling_covered = 0
    scene_covered = 0
    n_coupling = 0
    n_scene = 0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_selected": len(selected),
        "n_key_frames": len(key_frames),
        "n_matched": len(inter),
        "temporal_spread": round(spread, 2),
    }


def main():
    print("Token-Saver vs Uniform Sampling vs Human Annotations")
    print("=" * 60)

    # Find videos with annotations
    anno_files = list(ANNOTATIONS_DIR.glob("*_keyframes.json"))
    if not anno_files:
        print("\nNo annotation files found in annotations/")
        print("Please complete the annotation step first:")
        print("  1. Open http://localhost:8080/tools/annotate.html")
        print("  2. Annotate key frames for each video")
        print("  3. Save exported JSON files to annotations/ directory")
        return

    budgets = [3, 10, 20, 30, 50]
    all_results = {}

    for af in sorted(anno_files):
        anno = load_annotations(af.stem.replace("_keyframes", ""))
        if not anno:
            continue
        vid = anno["video_id"]
        key_frames = anno["key_frames"]
        total_seconds = anno.get("total_seconds", max(key_frames) + 1 if key_frames else 60)

        results = load_results(vid)
        ts_selected = {}
        if results:
            for k, v in results.get("selected_frames", {}).items():
                ts_selected[int(k)] = v

        print(f"\n{'─'*60}")
        print(f"  {vid} ({anno.get('category', '?')})")
        print(f"  Key frames: {len(key_frames)}/{total_seconds} ({len(key_frames)/max(total_seconds,1)*100:.1f}%)")
        print(f"  Annotated key frames: {key_frames[:10]}{'...' if len(key_frames) > 10 else ''}")
        print()

        video_results = {}
        for budget in budgets:
            print(f"  ── B={budget} ──")

            # Token-Saver
            ts_frames = ts_selected.get(budget, [])
            ts_m = compute_metrics(ts_frames, key_frames, total_seconds)
            print(f"  Token-Saver:  Prec={ts_m['precision']:.3f}  Rec={ts_m['recall']:.3f}  F1={ts_m['f1']:.3f}  spread={ts_m['temporal_spread']:.1f}s  ({ts_m['n_matched']}/{ts_m['n_selected']} matched)")

            # Uniform
            uni_frames = uniform_sampling(total_seconds, budget)
            uni_m = compute_metrics(uni_frames, key_frames, total_seconds)
            print(f"  Uniform:      Prec={uni_m['precision']:.3f}  Rec={uni_m['recall']:.3f}  F1={uni_m['f1']:.3f}  spread={uni_m['temporal_spread']:.1f}s  ({uni_m['n_matched']}/{len(uni_frames)} matched)")

            # Improvement
            f1_delta = (ts_m['f1'] - uni_m['f1']) / max(uni_m['f1'], 0.001) * 100
            print(f"  Δ F1: {f1_delta:+.1f}% vs Uniform")

            video_results[budget] = {
                "token_saver": ts_m,
                "uniform": uni_m,
                "f1_delta_pct": round(f1_delta, 1),
            }

        all_results[vid] = {
            "category": anno.get("category", "?"),
            "n_key_frames": len(key_frames),
            "total_seconds": total_seconds,
            "budgets": video_results,
        }

    # Summary
    print(f"\n{'='*60}")
    print("  Overall Summary: Token-Saver vs Uniform F1")
    print(f"{'='*60}")
    print(f"{'Video':<22s} {'Cat':<8s}", end="")
    for b in budgets:
        print(f" {'B='+str(b):>10s}", end="")
    print()
    for vid, info in sorted(all_results.items()):
        print(f"{vid:<22s} {info['category']:<8s}", end="")
        for b in budgets:
            ts_f1 = info["budgets"][b]["token_saver"]["f1"]
            uni_f1 = info["budgets"][b]["uniform"]["f1"]
            delta = info["budgets"][b]["f1_delta_pct"]
            sign = "+" if delta >= 0 else ""
            print(f" {ts_f1:.3f} ({sign}{delta:.0f}%)", end="")
        print()

    # Save
    save_path = OUTPUT_DIR / "mllm_eval_results.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved: {save_path}")


if __name__ == "__main__":
    main()
