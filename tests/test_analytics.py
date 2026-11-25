import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analytics import BehaviorAnalyzer, CountLine, Zone, point_in_polygon  # noqa: E402

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_point_in_polygon():
    assert point_in_polygon((5, 5), SQUARE)
    assert not point_in_polygon((15, 5), SQUARE)


def test_intrusion_once_and_loitering_after_threshold():
    an = BehaviorAnalyzer([Zone("z", SQUARE, loiter_sec=2)], [], fps=10)
    kinds = []
    for f in range(30):  # 一直待在禁區 3 秒
        kinds += [e.kind for e in an.update(f, {1: (5, 5)})]
    assert kinds == ["intrusion", "loitering"]
    assert an.events[1].frame == 20


def test_leave_and_reenter_triggers_new_intrusion():
    an = BehaviorAnalyzer([Zone("z", SQUARE, loiter_sec=99)], [], fps=10)
    for f, p in enumerate([(5, 5), (20, 5), (5, 5)]):
        an.update(f, {1: p})
    assert [e.kind for e in an.events] == ["intrusion", "intrusion"]


def test_boundary_jitter_does_not_retrigger_intrusion():
    an = BehaviorAnalyzer([Zone("z", SQUARE, loiter_sec=99)], [], fps=10, exit_frames=5)
    path = [(5, 5), (11, 5), (9, 5), (11, 5), (11, 5), (9, 5)]  # 邊界來回抖動
    for f, p in enumerate(path):
        an.update(f, {1: p})
    assert [e.kind for e in an.events] == ["intrusion"]
    for f in range(6, 12):  # 真正離開 5 幀以上
        an.update(f, {1: (20, 5)})
    an.update(12, {1: (5, 5)})
    assert [e.kind for e in an.events] == ["intrusion", "intrusion"]


def test_line_direction_counts():
    ln = CountLine("L", (0, 0), (0, 10))  # 垂直線，由上往下
    an = BehaviorAnalyzer([], [ln], fps=10)
    an.update(0, {1: (5, 5)})
    an.update(1, {1: (-5, 5)})  # 跨線一次
    an.update(2, {1: (5, 5)})  # 反向跨回
    assert ln.in_count + ln.out_count == 2
    assert ln.in_count == 1 and ln.out_count == 1


def test_no_count_when_not_crossing_segment():
    ln = CountLine("L", (0, 0), (0, 10))
    an = BehaviorAnalyzer([], [ln], fps=10)
    an.update(0, {1: (5, 20)})
    an.update(1, {1: (-5, 20)})  # 從線段延長線外側經過
    assert ln.in_count == ln.out_count == 0


def test_line_jitter_inside_margin_counts_nothing():
    ln = CountLine("L", (0, 0), (0, 100))
    an = BehaviorAnalyzer([], [ln], fps=10, line_margin_px=10)
    for f, x in enumerate([20, 5, -5, 5, -5, 5, 20]):  # 在線附近來回，最後回到原側
        an.update(f, {1: (x, 50)})
    assert ln.in_count == ln.out_count == 0


def test_slow_crossing_through_margin_counts_once():
    ln = CountLine("L", (0, 0), (0, 100))
    an = BehaviorAnalyzer([], [ln], fps=10, line_margin_px=10)
    for f, x in enumerate([20, 8, 3, -2, 4, -6, -15, -30]):  # 慢慢走過去，中途晃一下
        an.update(f, {1: (x, 50)})
    assert ln.in_count + ln.out_count == 1


def test_one_frame_flip_and_back_needs_confirmation():
    ln = CountLine("L", (0, 0), (0, 100))
    an = BehaviorAnalyzer([], [ln], fps=10, line_margin_px=10, line_confirm_frames=3)
    for f, x in enumerate([-30, -30, 30, -30, -30]):  # ID 接錯一幀跳到另一側又跳回
        an.update(f, {1: (x, 50)})
    assert ln.in_count == ln.out_count == 0
    for f, x in enumerate([30, 35, 40], start=5):  # 真的過去並待著
        an.update(f, {1: (x, 50)})
    assert ln.in_count + ln.out_count == 1


def test_track_born_on_line_still_counts():
    ln = CountLine("L", (0, 0), (0, 100))
    an = BehaviorAnalyzer([], [ln], fps=10, line_margin_px=10)
    for f, x in enumerate([2, -5, -15, -30]):  # 第一次被偵測到就在線上
        an.update(f, {1: (x, 50)})
    assert ln.in_count + ln.out_count == 1


def test_lost_track_is_purged():
    an = BehaviorAnalyzer([], [], fps=10, lost_frames=5)
    an.update(0, {1: (0, 0)})
    an.update(10, {2: (0, 0)})
    assert 1 not in an.tracks and 2 in an.tracks
