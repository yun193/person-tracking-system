"""YOLO 人物偵測 + ByteTrack 跟蹤 + 電子圍籬／逗留／劃線計數。

用法：
    python src/main.py --source data/vtest.avi --config configs/vtest.json --out outputs/vtest
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from analytics import BehaviorAnalyzer, CountLine, Zone
from recovery import TrackRecovery

ZONE_COLOR = (0, 0, 255)
LINE_COLOR = (255, 200, 0)


def load_config(path: Path, fps: float) -> tuple[BehaviorAnalyzer, dict]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    zones = [Zone(z["name"], [tuple(p) for p in z["polygon"]], z["loiter_sec"]) for z in cfg.get("zones", [])]
    lines = [CountLine(l["name"], tuple(l["a"]), tuple(l["b"])) for l in cfg.get("lines", [])]
    return BehaviorAnalyzer(zones, lines, fps, cfg.get("lost_frames", 30), cfg.get("exit_frames", 1),
                            cfg.get("line_margin_px", 0.0), cfg.get("line_confirm_frames", 1)), cfg


def crop(frame, box) -> np.ndarray:
    x1, y1, x2, y2 = (max(0, int(v)) for v in box)
    return frame[y1:y2, x1:x2].copy()


def appearance_feature(frame, box) -> np.ndarray | None:
    """上半身與下半身各一個 HSV 色彩直方圖（略過頭部），串成一條外觀向量。"""
    x1, y1, x2, y2 = (max(0, int(v)) for v in box)
    h = y2 - y1
    if h < 20 or x2 - x1 < 6:
        return None
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    parts = []
    for a, b in ((0.15, 0.55), (0.55, 0.95)):
        roi = hsv[int(h * a):int(h * b)]
        hist = cv2.calcHist([roi], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
        parts.append(hist / (hist.sum() + 1e-9))
    return np.concatenate(parts).astype(np.float32)


# 上衣比褲子有鑑別力（很多人都穿牛仔褲）。vtest 實測：唯一一次接錯人（深色外套→淺藍外套）上衣相似度 0.42，
# 6 次正確找回的上衣相似度都 ≥ 0.51，所以上衣低於此值直接判定不是同一人。樣本只有 7 筆，換場景要重新校正。
UPPER_MIN_SIM = 0.47


def hist_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """上衣、褲子各算 Bhattacharyya 相似度後取平均（1 = 一模一樣）；上衣不像直接回 0。"""
    n = len(a) // 2
    upper, lower = (1 - cv2.compareHist(a[s].astype(np.float32), b[s].astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
                    for s in (slice(0, n), slice(n, None)))
    if upper < UPPER_MIN_SIM:
        return 0.0
    return float((upper + lower) / 2)


def save_merge_gallery(pairs, path: Path, cell=(60, 120)):
    """每列一次找回：左 = 消失前最後樣子，右 = 新出現的樣子，方便人工檢查有沒有接錯人。"""
    if not pairs:
        return
    rows = []
    for old, new, m in pairs:
        tiles = [cv2.resize(c, cell) if c is not None and c.size else np.zeros((cell[1], cell[0], 3), np.uint8)
                 for c in (old, new)]
        label = np.zeros((cell[1], 150, 3), np.uint8)
        for i, t in enumerate([f"f{m.frame} ID{m.global_id}", f"raw {m.raw_id}", f"gap {m.gap_frames}f",
                               f"d {m.dist_px}px", f"sim {m.app_sim}"]):
            cv2.putText(label, t, (5, 20 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rows.append(np.hstack([*tiles, label]))
    cols = 4
    while len(rows) % cols:
        rows.append(np.zeros_like(rows[0]))
    grid = np.vstack([np.hstack(rows[i:i + cols]) for i in range(0, len(rows), cols)])
    cv2.imwrite(str(path), grid)


def track_color(tid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(tid)
    return tuple(int(c) for c in rng.integers(64, 256, 3))


def draw(frame, an: BehaviorAnalyzer, boxes, trails, frame_idx, alert_until, fps_now):
    overlay = frame.copy()
    for z in an.zones:
        pts = np.array(z.polygon, np.int32)
        cv2.fillPoly(overlay, [pts], ZONE_COLOR)
        cv2.polylines(frame, [pts], True, ZONE_COLOR, 2)
        cv2.putText(frame, z.name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ZONE_COLOR, 2)
    frame[:] = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

    for ln in an.lines:
        a, b = tuple(map(int, ln.a)), tuple(map(int, ln.b))
        cv2.line(frame, a, b, LINE_COLOR, 2)
        cv2.putText(frame, f"{ln.name} in:{ln.in_count} out:{ln.out_count}", (a[0] + 5, a[1] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, LINE_COLOR, 2)

    active_alerts: list[str] = []
    for tid, (x1, y1, x2, y2) in boxes.items():
        in_zones = [(z.name, an.stay_seconds(tid, z.name, frame_idx)) for z in an.zones]
        in_zones = [(name, s) for name, s in in_zones if s is not None]
        alerting = bool(in_zones) or alert_until.get(tid, -1) >= frame_idx
        color = (0, 0, 255) if alerting else track_color(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if alerting else 2)
        label = f"ID {tid}"
        for name, s in in_zones:
            label += f" {name}:{s:.1f}s"
            active_alerts.append(f"ID {tid}@{name}({s:.1f}s)")
        cv2.putText(frame, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        pts = np.array(trails[tid], np.int32)
        if len(pts) > 1:
            cv2.polylines(frame, [pts], False, track_color(tid), 2)

    cv2.putText(frame, f"frame {frame_idx}  people {len(boxes)}  {fps_now:.1f} FPS", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if active_alerts:
        banner = "ALERT: " + " | ".join(active_alerts[:4])
        cv2.rectangle(frame, (0, 28), (frame.shape[1], 56), (0, 0, 200), -1)
        cv2.putText(frame, banner, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    # vtest 上實測 0.1 與 0.3 的 ID 切換數幾乎相同（15→14），維持 0.3 以減少誤檢
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--recovery", action="store_true", help="啟用應用層遺失軌跡找回")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    an, _ = load_config(args.config, fps)
    args.out.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    model = YOLO(args.model)
    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=40))
    alert_until: dict[int, int] = {}
    seen_ids: set[int] = set()
    track_rows: list[tuple] = []
    recovery = TrackRecovery(fps, similarity=hist_similarity) if args.recovery else None
    last_crop: dict[int, np.ndarray] = {}
    merge_pairs: list[tuple] = []
    infer_ms: list[float] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        res = model.track(frame, persist=True, classes=[0], conf=args.conf, imgsz=args.imgsz,
                          tracker=args.tracker, verbose=False)[0]

        raw_boxes = {}
        if res.boxes.id is not None:
            for box, tid in zip(res.boxes.xyxy.int().tolist(), res.boxes.id.int().tolist()):
                raw_boxes[tid] = tuple(box)
        if recovery is not None:
            feats = {r: appearance_feature(frame, b) for r, b in raw_boxes.items()}
            n_before, n_revoked = len(recovery.merges), len(recovery.revoked)
            id_map = recovery.update(frame_idx, raw_boxes, feats)
            for m in recovery.merges[n_before:]:
                print(f"[{frame_idx / fps:6.1f}s] recover    raw {m.raw_id} -> ID {m.global_id} "
                      f"(gap {m.gap_frames}f, dist {m.dist_px}px, sim {m.app_sim})")
                merge_pairs.append((last_crop.get(m.global_id), crop(frame, raw_boxes[m.raw_id]), m))
            for f, raw, gid in recovery.revoked[n_revoked:]:
                print(f"[{f / fps:6.1f}s] revoke     raw {raw} 還回 ID {gid}（原軌跡回來了）")
        else:
            id_map = {r: r for r in raw_boxes}
        infer_ms.append((time.perf_counter() - t0) * 1000)

        boxes, feet = {}, {}
        for raw, (x1, y1, x2, y2) in raw_boxes.items():
            tid = id_map[raw]
            boxes[tid] = (x1, y1, x2, y2)
            feet[tid] = ((x1 + x2) / 2, y2)
            trails[tid].append(((x1 + x2) // 2, y2))
            seen_ids.add(tid)
            track_rows.append((frame_idx, tid, x1, y1, x2, y2))
            last_crop[tid] = crop(frame, (x1, y1, x2, y2))

        alert_events = []
        for ev in an.update(frame_idx, feet):
            if ev.kind in ("intrusion", "loitering"):
                alert_until[ev.track_id] = frame_idx + int(fps)  # 告警框至少亮 1 秒
                alert_events.append(ev)
            print(f"[{ev.time_sec:6.1f}s] {ev.kind:<10} ID {ev.track_id:<3} @ {ev.target}")

        draw(frame, an, boxes, trails, frame_idx, alert_until, 1000 / np.mean(infer_ms[-30:]))
        if alert_events:
            alert_dir = args.out / "alerts"
            alert_dir.mkdir(exist_ok=True)
            for ev in alert_events:
                cv2.imwrite(str(alert_dir / f"{ev.kind}_f{ev.frame:04d}_id{ev.track_id}.jpg"), frame)
        writer.write(frame)
        if args.show:
            cv2.imshow("tracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    with open(args.out / "events.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["frame", "time_sec", "kind", "track_id", "target"])
        wr.writeheader()
        wr.writerows(asdict(e) for e in an.events)

    with open(args.out / "tracks.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "track_id", "x1", "y1", "x2", "y2"])
        wr.writerows(track_rows)

    if recovery is not None:
        with open(args.out / "merges.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=["frame", "raw_id", "global_id", "gap_frames", "dist_px", "app_sim"])
            wr.writeheader()
            wr.writerows(asdict(m) for m in recovery.merges)
        save_merge_gallery(merge_pairs, args.out / "merges_gallery.jpg")

    warm = infer_ms[10:] or infer_ms
    summary = {
        "source": args.source,
        "tracker": args.tracker,
        "conf": args.conf,
        "recovery": args.recovery,
        "recovered": len(recovery.merges) if recovery else 0,
        "imgsz": args.imgsz,
        "width": w,
        "height": h,
        "video_fps": fps,
        "frames": frame_idx,
        "unique_track_ids": len(seen_ids),
        "avg_infer_ms": round(float(np.mean(warm)), 2),
        "avg_fps": round(1000 / float(np.mean(warm)), 1),
        "lines": {l.name: {"in": l.in_count, "out": l.out_count} for l in an.lines},
        "event_counts": {k: sum(e.kind == k for e in an.events)
                         for k in ("intrusion", "loitering", "line_in", "line_out")},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
