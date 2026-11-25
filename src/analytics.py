"""行為分析邏輯：電子圍籬入侵、禁區逗留、劃線進出計數。

只吃「track_id + 腳底座標 + 影格編號」，不依賴 YOLO，方便單元測試。
"""
from __future__ import annotations

from dataclasses import dataclass, field

Point = tuple[float, float]


def point_in_polygon(p: Point, poly: list[Point]) -> bool:
    """射線法判斷點是否在多邊形內。"""
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


def _side(p: Point, a: Point, b: Point) -> float:
    """點在有向線段 a→b 的哪一側（外積正負）。"""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


@dataclass
class Zone:
    name: str
    polygon: list[Point]
    loiter_sec: float


@dataclass
class CountLine:
    name: str
    a: Point
    b: Point
    in_count: int = 0
    out_count: int = 0


@dataclass
class Event:
    frame: int
    time_sec: float
    kind: str  # intrusion / loitering / line_in / line_out
    track_id: int
    target: str


@dataclass
class _TrackState:
    last_point: Point
    last_frame: int
    zone_enter_frame: dict[str, int] = field(default_factory=dict)
    zone_outside_frames: dict[str, int] = field(default_factory=dict)
    loiter_alerted: set[str] = field(default_factory=set)
    line_side: dict[str, tuple[int, Point, float]] = field(default_factory=dict)  # 已確認側、確認時位置、距離
    line_pending: dict[str, int] = field(default_factory=dict)  # 已連續幾次出現在另一側


class BehaviorAnalyzer:
    def __init__(self, zones: list[Zone], lines: list[CountLine], fps: float,
                 lost_frames: int = 30, exit_frames: int = 1, line_margin_px: float = 0.0,
                 line_confirm_frames: int = 1):
        self.zones = zones
        self.lines = lines
        self.fps = fps
        self.lost_frames = lost_frames
        # 腳底點在邊界抖動時，需連續 exit_frames 幀在外才算離開，避免重複入侵告警
        self.exit_frames = exit_frames
        # 劃線遲滯：離線超過這個距離才算確定站在某一側
        self.line_margin_px = line_margin_px
        self.line_confirm_frames = line_confirm_frames
        self.tracks: dict[int, _TrackState] = {}
        self.events: list[Event] = []

    def _emit(self, frame: int, kind: str, tid: int, target: str) -> Event:
        ev = Event(frame, round(frame / self.fps, 2), kind, tid, target)
        self.events.append(ev)
        return ev

    def update(self, frame: int, detections: dict[int, Point]) -> list[Event]:
        """輸入本影格所有 {track_id: 腳底座標}，回傳本影格新事件。"""
        new_events: list[Event] = []
        for tid, pt in detections.items():
            st = self.tracks.get(tid)
            if st is None:
                st = self.tracks[tid] = _TrackState(last_point=pt, last_frame=frame)
            for ln in self.lines:
                kind = self._line_crossing(st, ln, pt)
                if kind == "line_in":
                    ln.in_count += 1
                elif kind == "line_out":
                    ln.out_count += 1
                if kind:
                    new_events.append(self._emit(frame, kind, tid, ln.name))

            for z in self.zones:
                inside = point_in_polygon(pt, z.polygon)
                in_zone = z.name in st.zone_enter_frame
                if inside:
                    st.zone_outside_frames.pop(z.name, None)
                    if not in_zone:
                        st.zone_enter_frame[z.name] = frame
                        new_events.append(self._emit(frame, "intrusion", tid, z.name))
                elif in_zone:
                    out = st.zone_outside_frames[z.name] = st.zone_outside_frames.get(z.name, 0) + 1
                    if out >= self.exit_frames:
                        del st.zone_enter_frame[z.name]
                        del st.zone_outside_frames[z.name]
                        st.loiter_alerted.discard(z.name)
                if z.name in st.zone_enter_frame and z.name not in st.loiter_alerted:
                    stay = (frame - st.zone_enter_frame[z.name]) / self.fps
                    if stay >= z.loiter_sec:
                        st.loiter_alerted.add(z.name)
                        new_events.append(self._emit(frame, "loitering", tid, z.name))

            st.last_point = pt
            st.last_frame = frame

        for tid in [t for t, s in self.tracks.items() if frame - s.last_frame > self.lost_frames]:
            del self.tracks[tid]
        return new_events

    def _line_crossing(self, st: _TrackState, ln: CountLine, pt: Point) -> str | None:
        """帶空間＋時間遲滯的跨線判斷。

        - 空間：離線 >= line_margin_px 才算確定站在某一側，線附近的抖動不改變狀態
        - 時間：換到另一側要連續 line_confirm_frames 次才成立（擋掉 ID 接錯又收回造成的一進一出）
        - 跨越點要落在線段範圍內（不是線的延長線）
        - 軌跡第一次出現就在線附近時，先用正負號暫定一側（人在線上才被偵測到時也算得到）
        由線的左側(負)走到右側(正)視為「進」。
        """
        ax, ay = ln.a
        bx, by = ln.b
        length = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
        d = _side(pt, ln.a, ln.b) / length  # 帶正負號的像素距離
        if d == 0:
            return None
        side = 1 if d > 0 else -1
        prev = st.line_side.get(ln.name)
        if prev is None:
            st.line_side[ln.name] = (side, pt, d)
            return None
        if abs(d) < self.line_margin_px:
            return None
        if prev[0] == side:
            st.line_side[ln.name] = (side, pt, d)
            st.line_pending.pop(ln.name, None)
            return None
        n = st.line_pending[ln.name] = st.line_pending.get(ln.name, 0) + 1
        if n < self.line_confirm_frames:
            return None
        st.line_pending.pop(ln.name, None)
        st.line_side[ln.name] = (side, pt, d)
        _, anchor, d_anchor = prev
        t = d_anchor / (d_anchor - d)  # anchor→pt 與線的交點位置
        cx, cy = anchor[0] + (pt[0] - anchor[0]) * t, anchor[1] + (pt[1] - anchor[1]) * t
        u = ((cx - ax) * (bx - ax) + (cy - ay) * (by - ay)) / length ** 2
        if not -0.05 <= u <= 1.05:
            return None
        return "line_in" if side > 0 else "line_out"

    def stay_seconds(self, tid: int, zone_name: str, frame: int) -> float | None:
        st = self.tracks.get(tid)
        if st is None or zone_name not in st.zone_enter_frame:
            return None
        return (frame - st.zone_enter_frame[zone_name]) / self.fps
