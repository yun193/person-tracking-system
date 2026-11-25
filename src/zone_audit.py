"""危險區域（電子圍籬）入侵稽核工具：用 tracks.csv 重播並輸出每次入侵的前／當下／後截圖。

用法：
    python src/zone_audit.py --run outputs/vtest --video data/vtest.avi --config configs/vtest.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from analytics import point_in_polygon
from main import load_config


def replay_zones(run: Path, config: Path, fps: float):
    an, _ = load_config(config, fps)
    per_frame: dict[int, dict[int, tuple[int, int, int, int]]] = defaultdict(dict)
    with open(run / "tracks.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            per_frame[int(r["frame"])][int(r["track_id"])] = tuple(int(r[k]) for k in ("x1", "y1", "x2", "y2"))
    n = max(per_frame) + 1
    for fr in range(n):
        an.update(fr, {t: ((b[0] + b[2]) / 2, b[3]) for t, b in per_frame.get(fr, {}).items()})
    intrusions = [e for e in an.events if e.kind in ("intrusion", "loitering")]
    return intrusions, per_frame, an, n


def analyze_track_zone_presence(per_frame, an, n_frames: int):
    """列出所有曾經進過禁區的 track_id 及其在禁區內的影格區間，方便排查重複告警或漏報。"""
    zone = an.zones[0]
    in_frames: dict[int, list[int]] = defaultdict(list)
    seen_frames: dict[int, list[int]] = defaultdict(list)
    for fr in range(n_frames):
        for tid, b in per_frame.get(fr, {}).items():
            seen_frames[tid].append(fr)
            foot = ((b[0] + b[2]) / 2, b[3])
            if point_in_polygon(foot, zone.polygon):
                in_frames[tid].append(fr)
    print(f"=== 禁區 {zone.name} 內出現過的軌跡統計 ===")
    for tid in sorted(in_frames):
        frs = in_frames[tid]
        # 切成連續區間（gap > 1 視為中斷）
        spans = []
        s = e = frs[0]
        for f in frs[1:]:
            if f <= e + 1:
                e = f
            else:
                spans.append((s, e))
                s = e = f
        spans.append((s, e))
        span_str = ", ".join(f"{a}-{b}({b-a+1}f)" for a, b in spans)
        print(f"  ID {tid:>3} | 總出現 {seen_frames[tid][0]}..{seen_frames[tid][-1]} ({len(seen_frames[tid])}f) | "
              f"禁區內 {len(frs)}f | 區間: {span_str}")


def build_gallery(events, per_frame, an, video: str, out: Path, offset: int = 5, size: int = 180):
    cap = cv2.VideoCapture(video)
    zone = an.zones[0]
    pts = np.array(zone.polygon, np.int32)
    rows = []
    for i, ev in enumerate(events):
        if ev.kind != "intrusion":
            continue
        tiles = []
        ref_box = per_frame[ev.frame][ev.track_id]
        cx, cy = int((ref_box[0] + ref_box[2]) / 2), int(ref_box[3])
        for tag, fr in (("before", max(0, ev.frame - offset)), ("trigger", ev.frame), ("after", ev.frame + offset)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, img = cap.read()
            if not ok:
                img = np.zeros((576, 768, 3), np.uint8)
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 255))
            img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
            cv2.polylines(img, [pts], True, (0, 0, 255), 2)
            box = per_frame.get(fr, {}).get(ev.track_id)
            if box:
                cv2.rectangle(img, box[:2], box[2:], (0, 255, 255) if tag != "trigger" else (0, 0, 255), 2)
                fx, fy = int((box[0] + box[2]) / 2), int(box[3])
                cv2.circle(img, (fx, fy), 4, (0, 255, 0), -1)
            h, w = img.shape[:2]
            x0 = max(0, min(w - 2 * size, cx - size))
            y0 = max(0, min(h - 2 * size, cy - size))
            crop = cv2.resize(img[y0:y0 + 2 * size, x0:x0 + 2 * size], (size, size))
            cv2.putText(crop, f"{tag} f{fr}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            tiles.append(crop)
        label = np.zeros((size, 150, 3), np.uint8)
        for j, t in enumerate([f"#{i + 1} intrusion", f"f{ev.frame} ({ev.time_sec}s)", f"ID {ev.track_id}", ev.target]):
            cv2.putText(label, t, (8, 30 + j * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        rows.append(np.hstack([*tiles, label]))
    cap.release()
    if rows:
        cv2.imwrite(str(out), np.vstack(rows))
        print("gallery:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--offset", type=int, default=5)
    args = ap.parse_args()

    fps = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))["video_fps"]
    events, per_frame, an, n = replay_zones(args.run, args.config, fps)
    print(f"=== 禁區事件 ({len(events)} 筆) ===")
    for e in events:
        print(f"  f{e.frame:>3} ({e.time_sec:>5.1f}s)  {e.kind:<10}  ID {e.track_id:<3}  @ {e.target}")
    analyze_track_zone_presence(per_frame, an, n)
    build_gallery(events, per_frame, an, args.video, args.run / "intrusion_audit.jpg", offset=args.offset)


if __name__ == "__main__":
    main()
