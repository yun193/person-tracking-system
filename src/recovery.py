"""應用層遺失軌跡找回：追蹤器給出新 ID 時，比對剛消失的人，像的話接回原本的 ID。

ultralytics 的 BoT-SORT 只在 IoU 夠高時才看外觀；這裡改成用「預測位置 + 外觀 + 框大小」
在 IoU 失效的情況下（人被遮擋後已經走開）把 ID 接回來。

只吃座標與外觀向量，不依賴 YOLO，方便單元測試。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

Box = tuple[float, float, float, float]


@dataclass
class RecoveryConfig:
    max_gap_sec: float = 3.0  # 消失多久內還能被找回（需 <= 分析器的 lost_frames）
    base_radius_px: float = 40.0  # 位置閘門基本半徑
    radius_per_frame_px: float = 8.0  # 每消失一幀，半徑放寬多少（人走路的速度上限）
    min_app_sim: float = 0.55  # 外觀相似度下限
    max_height_ratio: float = 1.6  # 新舊框高度比上限（遠近不會一下差太多）
    w_app: float = 0.6  # 成本中外觀的權重，其餘給位置
    ema: float = 0.3  # 外觀向量更新速度
    occlusion_iou: float = 0.3  # 與別人重疊超過此值時不更新外觀，避免被別人的衣服污染


@dataclass
class _Gallery:
    last_frame: int
    foot: np.ndarray
    height: float
    feat: np.ndarray | None
    history: deque = field(default_factory=lambda: deque(maxlen=6))  # (frame, foot)

    def velocity(self) -> np.ndarray:
        if len(self.history) < 2:
            return np.zeros(2)
        (f0, p0), (f1, p1) = self.history[0], self.history[-1]
        return (p1 - p0) / max(1, f1 - f0)


@dataclass
class Merge:
    frame: int
    raw_id: int
    global_id: int
    gap_frames: int
    dist_px: float
    app_sim: float


def _iou(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _foot(b: Box) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2, b[3]], float)


class TrackRecovery:
    def __init__(self, fps: float, cfg: RecoveryConfig | None = None,
                 similarity: Callable[[np.ndarray, np.ndarray], float] | None = None):
        self.cfg = cfg or RecoveryConfig()
        self.max_gap = int(round(self.cfg.max_gap_sec * fps))
        self.sim = similarity or (lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
        self.raw_to_global: dict[int, int] = {}
        self.gallery: dict[int, _Gallery] = {}
        self.merges: list[Merge] = []
        self.bound_frame: dict[int, int] = {}  # raw 第一次被綁定 global ID 的幀
        self.revoked: list[tuple[int, int, int]] = []  # (frame, raw_id, 被收回的 global_id)

    def update(self, frame: int, boxes: dict[int, Box], feats: dict[int, np.ndarray]) -> dict[int, int]:
        """輸入本幀 {raw_id: box} 與外觀 {raw_id: 向量}，回傳 {raw_id: global_id}。"""
        known = [r for r in boxes if r in self.raw_to_global]
        births = [r for r in boxes if r not in self.raw_to_global]

        # 衝突：追蹤器把舊的 raw 自己接回來了，但它的 global ID 已經交給新的 raw。
        # 追蹤器自己的接續（IoU + 卡爾曼）比我們的猜測可靠 → 較早綁定的 raw 留下，較晚的拆回自己的 ID。
        by_gid: dict[int, list[int]] = {}
        for r in known:
            by_gid.setdefault(self.raw_to_global[r], []).append(r)
        for gid, raws in by_gid.items():
            if len(raws) > 1:
                raws.sort(key=lambda r: self.bound_frame.get(r, -1))
                for loser in raws[1:]:
                    self.raw_to_global[loser] = loser
                    self.revoked.append((frame, loser, gid))
        active = {self.raw_to_global[r] for r in known}

        for raw in births:
            gid = self._match(frame, raw, boxes[raw], feats.get(raw), active)
            if gid is None:
                gid = raw
            self.raw_to_global[raw] = gid
            self.bound_frame[raw] = frame
            active.add(gid)

        for raw, box in boxes.items():
            gid = self.raw_to_global[raw]
            occluded = any(_iou(box, b) > self.cfg.occlusion_iou for r, b in boxes.items() if r != raw)
            self._update_gallery(frame, gid, box, None if occluded else feats.get(raw))

        for gid in [g for g, s in self.gallery.items() if frame - s.last_frame > self.max_gap]:
            del self.gallery[gid]
        return {r: self.raw_to_global[r] for r in boxes}

    def _match(self, frame: int, raw: int, box: Box, feat: np.ndarray | None, active: set[int]) -> int | None:
        if feat is None:
            return None
        c = self.cfg
        foot, h = _foot(box), box[3] - box[1]
        best, best_cost, best_info = None, float("inf"), (0, 0.0, 0.0)
        for gid, g in self.gallery.items():
            gap = frame - g.last_frame
            if gid in active or gap <= 0 or gap > self.max_gap or g.feat is None:
                continue
            ratio = max(h, g.height) / max(1e-6, min(h, g.height))
            if ratio > c.max_height_ratio:
                continue
            pred = g.foot + g.velocity() * gap
            radius = c.base_radius_px + c.radius_per_frame_px * gap
            dist = float(np.linalg.norm(foot - pred))
            if dist > radius:
                continue
            sim = self.sim(feat, g.feat)
            if sim < c.min_app_sim:
                continue
            cost = c.w_app * (1 - sim) + (1 - c.w_app) * dist / radius
            if cost < best_cost:
                best, best_cost, best_info = gid, cost, (gap, dist, sim)
        if best is not None:
            gap, dist, sim = best_info
            self.merges.append(Merge(frame, raw, best, gap, round(dist, 1), round(sim, 3)))
        return best

    def _update_gallery(self, frame: int, gid: int, box: Box, feat: np.ndarray | None):
        foot = _foot(box)
        g = self.gallery.get(gid)
        if g is None:
            g = self.gallery[gid] = _Gallery(frame, foot, box[3] - box[1], feat)
        else:
            g.last_frame, g.foot, g.height = frame, foot, box[3] - box[1]
            if feat is not None:
                g.feat = feat if g.feat is None else (1 - self.cfg.ema) * g.feat + self.cfg.ema * feat
        g.history.append((frame, foot))
