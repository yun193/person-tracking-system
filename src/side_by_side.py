"""把兩支標註影片左右並排，方便肉眼對照追蹤器差異。

用法：
    python src/side_by_side.py outputs/cmp_bytetrack outputs/cmp_botsort_reid outputs/compare_bytetrack_vs_reid.mp4
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


def open_writer(path: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    for code in ("avc1", "mp4v"):  # avc1(H.264) 播放相容性較好，系統沒有編碼器時退回 mp4v
        wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*code), fps, size)
        if wr.isOpened():
            print("codec:", code)
            return wr
    raise RuntimeError("無可用的影片編碼器")


def main():
    left, right, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    caps = [cv2.VideoCapture(str(d / "annotated.mp4")) for d in (left, right)]
    titles = []
    for d in (left, right):
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        titles.append(f"{Path(s['tracker']).stem}  IDs={s['unique_track_ids']}  {s['avg_fps']} FPS")
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    w, h = 640, 480
    wr = open_writer(out, fps, (w * 2, h + 30))
    while True:
        frames = [c.read() for c in caps]
        if not all(ok for ok, _ in frames):
            break
        row = np.hstack([cv2.resize(f, (w, h)) for _, f in frames])
        bar = np.zeros((30, w * 2, 3), np.uint8)
        for i, t in enumerate(titles):
            cv2.putText(bar, t, (10 + i * w, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        wr.write(np.vstack([bar, row]))
    wr.release()


if __name__ == "__main__":
    main()
