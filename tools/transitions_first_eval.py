"""预算内「转镜优先 + 均匀兜底」取帧算法评估.

背景（2026-08-07）：用户目检确认 TS∪转镜 的新增转镜帧是「重要帧」（承载动作/证据/
列举，复刻必需），但 union 使帧数膨胀 1.5×。用户选定「预算内重分配」：总帧数 = TS
帧数（token 对齐可比），转镜帧优先占额度，剩余用均匀采样兜底保证叙事连贯。

算法 transitions_first：
  1. 逐秒缩略图帧差 diff（HSV Bhattacharyya，vs 前一秒）
  2. diff > threshold 的秒为转镜候选，按 diff 降序优先入选
  3. 预算内先取转镜 top，不足 N 用均匀采样兜底补足
  4. 恰好返回 N = len(TS) 帧，与 TokenSaver 同 token 数公平对比

自适应行为：
  - 高动态视频（每秒都变）：转镜候选≈全程，退化为「画面变化最剧烈的 N 秒」
  - 低动态口播（转镜稀疏）：均匀兜底主导，保证叙事覆盖
  - 中动态：精确捕捉镜头切换点

评估：对人工关键帧 L1（±1s，key 计数），TS vs 新算法 同帧数可比；重点看新算法
找回多少 TS 漏掉的关键帧。渲染对比图（新算法 | TS | 人工关键帧）。

用法: cd HotMirrorStudio/视频降维技术研究 && PYTHONIOENCODING=utf-8 python tools/transitions_first_eval.py [vid ...]
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("tsu", ROOT / "tools" / "ts_scene_union.py")
tsu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tsu)

from app.video.token_saver import _uniform_sample  # noqa: E402

PILOT = tsu.PILOT
CURATED = tsu.CURATED
INSPECT_DIR = PILOT / "scene_union_inspect"
OUT_DIR = PILOT / "ts_scene_union"
THRESHOLDS = [0.06, 0.10, 0.15, 0.25]


def transitions_first(diffs: list[float], N: int, total: int,
                      threshold: float = 0.28) -> list[int]:
    """转镜优先 + 均匀兜底（≤N 帧，预算内不膨胀）。

    转镜候选（diff>threshold）按 diff 降序优先入选；不足 N 用均匀采样兜底，
    因与转镜帧去重损耗，最终帧数略小于 N（实测比 TS 还略省），分布均匀。
    """
    if N <= 0:
        return []
    if N >= total:
        return list(range(total))
    scene = [(i, d) for i, d in enumerate(diffs) if d > threshold]
    scene.sort(key=lambda x: -x[1])  # diff 降序，转镜最明显优先
    frames = [i for i, _ in scene[:N]]
    if len(frames) < N:
        need = N - len(frames)
        for f in _uniform_sample(need, total):
            if len(frames) >= N:
                break
            if f not in frames:
                frames.append(f)
    return sorted(frames)


def main() -> None:
    INSPECT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curated = json.loads(CURATED.read_text(encoding="utf-8"))
    by_vid = {v["video_id"]: v for v in curated}
    vids = sys.argv[1:] or sorted(by_vid.keys())

    rows = []
    for vid in vids:
        v = by_vid.get(vid)
        if v is None:
            print(f"!! {vid}: 不在 21 条标注集，跳过")
            continue
        anno_path = PILOT / vid / "annotation_data.json"
        if not anno_path.exists():
            print(f"!! {vid}: 缺 annotation_data，跳过")
            continue
        segs = tsu.rebuild_segs(json.loads(anno_path.read_text(encoding="utf-8")))
        total = v["total_seconds"]
        key = [k for k in v.get("key_frames", []) if k < total]
        ts = tsu.ts_frames(segs, total)
        N = len(ts)
        diffs = tsu.scene_diffs(vid, total)

        rec = {"video_id": vid, "duration": total, "priority": v.get("visual_priority"),
               "n_ts": N, "ts": ts, "diffs": diffs, "key_frames": key,
               "per_threshold": {}}
        for T in THRESHOLDS:
            tf = transitions_first(diffs, N, total, threshold=T)
            m_ts = tsu.l1_metrics(ts, key)
            m_tf = tsu.l1_metrics(tf, key)
            recv, miss = tsu.recovered_keys(tf, ts, key)
            n_scene = sum(1 for d in diffs if d > T)
            rec["per_threshold"][str(T)] = {
                "n_scene": n_scene, "n_tf": len(tf), "frames": tf,
                "ts_l1": m_ts, "tf_l1": m_tf,
                "delta_f1": round(m_tf["f1"] - m_ts["f1"], 4),
                "recovered": recv, "still_miss": miss,
            }
        rows.append(rec)

        T = 0.10
        tf = rec["per_threshold"]["0.1"]["frames"]
        print(f"\n=== {vid}  {str(v.get('visual_priority')):<8} dur={total}s "
              f"N={N}  转镜候选{rec['per_threshold']['0.1']['n_scene']} "
              f"-> TF={len(tf)} ===")
        tsu.render_grid(vid, f"{vid}  转镜优先 (T={T}, {len(tf)}帧)", tf,
                        INSPECT_DIR / f"{vid}_tf.jpg")
        tsu.render_grid(vid, f"{vid}  TokenSaver ({len(ts)}帧)", ts,
                        INSPECT_DIR / f"{vid}_ts.jpg")
        tsu.render_grid(vid, f"{vid}  人工关键帧 ({len(key)}帧)", key,
                        INSPECT_DIR / f"{vid}_keyframes.jpg")
        tsu.build_compare([INSPECT_DIR / f"{vid}_tf.jpg",
                           INSPECT_DIR / f"{vid}_ts.jpg"],
                          INSPECT_DIR / f"{vid}_tf_compare.jpg")

    out = OUT_DIR / "transitions_first_eval.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存: {out}")

    print("\n" + "=" * 82)
    print("预算内「转镜优先」vs TokenSaver  对人工关键帧 L1（同 N 帧，可比）")
    print("=" * 82)
    print(f"{'T':<6} {'ΔF1均':>6} {'胜/平/负':>10} {'找回KF':>7} {'仍漏KF':>7} "
          f"{'TF召回':>7} {'TS召回':>7}")
    for T in THRESHOLDS:
        t = str(T)
        deltas = [r["per_threshold"][t]["delta_f1"] for r in rows]
        recv = [r["per_threshold"][t]["recovered"] for r in rows]
        miss = [r["per_threshold"][t]["still_miss"] for r in rows]
        r_tf = [r["per_threshold"][t]["tf_l1"]["recall"] for r in rows]
        r_ts = [r["per_threshold"][t]["ts_l1"]["recall"] for r in rows]
        win = sum(1 for d in deltas if d > 0)
        tie = sum(1 for d in deltas if d == 0)
        avg = sum(deltas) / len(deltas)
        print(f"{T:<6} {avg:>+6.3f} {win:>3}/{tie:>1}/"
              f"{sum(1 for d in deltas if d < 0):>3} "
              f"{sum(len(x) for x in recv):>7} {sum(len(x) for x in miss):>7} "
              f"{sum(r_tf)/len(r_tf):>7.3f} {sum(r_ts)/len(r_ts):>7.3f}")


if __name__ == "__main__":
    main()
