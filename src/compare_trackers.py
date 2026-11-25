"""比較多個追蹤器輸出的軌跡品質（無標註資料時的代理指標）。

用法：
    python src/compare_trackers.py outputs/cmp_bytetrack outputs/cmp_botsort outputs/cmp_botsort_reid

指標說明：
- unique_ids：總 ID 數。偵測器相同時，越少代表 ID 切換越少
- short_tracks：存活 < 1 秒的軌跡（多半是誤檢或斷掉的碎片）
- mid_births：第一次出現位置離畫面邊緣 > margin 的 ID，代表人不是從畫面外走進來，
  多半是原本的人換了新 ID（ID 切換的主要訊號）
- handoffs：新 ID 出生前 3 秒內、80px 內剛有別的 ID 消失 → 最直接的 ID 切換估計
- avg_track_sec：平均軌跡長度（越長越穩）
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def analyze(run_dir: Path, margin: int = 40, handoff_sec: float = 3.0, handoff_px: int = 80) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    w, h = summary["width"], summary["height"]
    fps = summary["video_fps"]
    first: dict[int, tuple] = {}
    frames: dict[int, list[int]] = defaultdict(list)
    dup = 0  # 同一幀出現兩個相同 ID（必須是 0，否則 ID 對應有 bug）
    with open(run_dir / "tracks.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tid, fr = int(r["track_id"]), int(r["frame"])
            box = tuple(int(r[k]) for k in ("x1", "y1", "x2", "y2"))
            dup += bool(frames[tid]) and frames[tid][-1] == fr
            frames[tid].append(fr)
            first.setdefault(tid, box)

    def near_edge(b):
        x1, y1, x2, y2 = b
        return x1 < margin or y1 < margin or x2 > w - margin or y2 > h - margin

    lengths = [len(v) / fps for v in frames.values()]

    # 交接（handoff）：新 ID 出生前 handoff_sec 內，附近剛好有別的 ID 消失 → 高度疑似 ID 切換
    last: dict[int, tuple] = {}
    with open(run_dir / "tracks.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            last[int(r["track_id"])] = (int(r["frame"]), (int(r["x1"]) + int(r["x2"])) / 2, int(r["y2"]))
    births = {tid: (min(v), (first[tid][0] + first[tid][2]) / 2, first[tid][3]) for tid, v in frames.items()}
    handoffs = 0
    for tid, (bf, bx, by) in births.items():
        for oid, (ef, ex, ey) in last.items():
            if oid != tid and 0 <= bf - ef <= handoff_sec * fps and abs(bx - ex) + abs(by - ey) <= handoff_px:
                handoffs += 1
                break

    return {
        "run": run_dir.name,
        "tracker": summary["tracker"],
        "unique_ids": len(frames),
        "short_tracks": sum(l < 1 for l in lengths),
        "mid_births": sum(not near_edge(b) for b in first.values()),
        "handoffs": handoffs,
        "dup_id_frames": dup,
        "avg_track_sec": round(sum(lengths) / len(lengths), 1),
        "avg_fps": summary["avg_fps"],
        "intrusion": summary["event_counts"]["intrusion"],
        "loitering": summary["event_counts"]["loitering"],
        "line_in/out": "/".join(str(v) for v in next(iter(summary["lines"].values())).values()),
    }


def main():
    rows = [analyze(Path(p)) for p in sys.argv[1:]]
    keys = list(rows[0])
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for r in rows:
        print("| " + " | ".join(str(r[k]) for k in keys) + " |")


if __name__ == "__main__":
    main()
