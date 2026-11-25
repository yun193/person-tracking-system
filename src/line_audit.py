"""劃線計數稽核：不重跑 YOLO，直接用 tracks.csv 重播行為分析，並輸出每次跨線的前後截圖供人工核對。

用法：
    python src/line_audit.py --run outputs/cmp_bytetrack_recovery --video data/vtest.avi \
        --config configs/vtest.json --margins 0 5 10 15 --gallery-margin 0
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from main import load_config


def replay(run: Path, config: Path, fps: float, margin: float, confirm: int = 1):
    an, _ = load_config(config, fps)
    an.line_margin_px = margin
    an.line_confirm_frames = confirm
    per_frame: dict[int, dict[int, tuple]] = defaultdict(dict)
    with open(run / "tracks.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            per_frame[int(r["frame"])][int(r["track_id"])] = tuple(int(r[k]) for k in ("x1", "y1", "x2", "y2"))
    n = max(per_frame) + 1
    for fr in range(n):
        an.update(fr, {t: ((b[0] + b[2]) / 2, b[3]) for t, b in per_frame.get(fr, {}).items()})
    lines = [e for e in an.events if e.kind.startswith("line")]
    return lines, per_frame, an


def gallery(events, per_frame, an, video: str, out: Path, offset=4, size=160):
    cap = cv2.VideoCapture(video)
    ln = an.lines[0]
    rows = []
    for i, ev in enumerate(events):
        tiles = []
        for fr in (ev.frame - offset, ev.frame + offset):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fr))
            ok, img = cap.read()
            if not ok:
                img = np.zeros((576, 768, 3), np.uint8)
            cv2.line(img, tuple(map(int, ln.a)), tuple(map(int, ln.b)), (255, 200, 0), 2)
            box = per_frame.get(fr, {}).get(ev.track_id)
            if box:
                cv2.rectangle(img, box[:2], box[2:], (0, 0, 255), 2)
            cx, cy = per_frame[ev.frame][ev.track_id][0::2]
            cx, cy = int(sum(per_frame[ev.frame][ev.track_id][0::2]) / 2), per_frame[ev.frame][ev.track_id][3]
            x0, y0 = max(0, cx - size), max(0, cy - size)
            tiles.append(cv2.resize(img[y0:y0 + 2 * size, x0:x0 + 2 * size], (size, size)))
        label = np.zeros((size, 120, 3), np.uint8)
        for j, t in enumerate([f"#{i}", f"f{ev.frame}", f"ID {ev.track_id}", ev.kind[5:]]):
            cv2.putText(label, t, (5, 25 + j * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        rows.append(np.hstack([*tiles, label]))
    cols = 3
    while len(rows) % cols:
        rows.append(np.zeros_like(rows[0]))
    cv2.imwrite(str(out), np.vstack([np.hstack(rows[i:i + cols]) for i in range(0, len(rows), cols)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--margins", type=float, nargs="+", default=[0, 5, 10, 15])
    ap.add_argument("--confirms", type=int, nargs="+", default=[1])
    ap.add_argument("--gallery-margin", type=float)
    args = ap.parse_args()
    fps = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))["video_fps"]
    for k in args.confirms:
        for m in args.margins:
            ev, _, an = replay(args.run, args.config, fps, m, k)
            ln = an.lines[0]
            print(f"margin {m:>4}px confirm {k}: in {ln.in_count:>2} out {ln.out_count:>2}  "
                  + " ".join(f"{e.frame}:{e.track_id}{'+' if e.kind == 'line_in' else '-'}" for e in ev))
    if args.gallery_margin is not None:
        ev, per_frame, an = replay(args.run, args.config, fps, args.gallery_margin, args.confirms[0])
        out = args.run / f"line_audit_m{args.gallery_margin:g}.jpg"
        gallery(ev, per_frame, an, args.video, out)
        print("gallery:", out)


if __name__ == "__main__":
    main()
