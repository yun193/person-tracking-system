import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recovery import TrackRecovery  # noqa: E402

RED = np.array([1.0, 0.0, 0.0])
BLUE = np.array([0.0, 0.0, 1.0])


def box_at(x, y, h=60):
    return (x - 10, y - h, x + 10, y)


def walk(tr, raw, frames, x0, dx, feat, y=300):
    out = {}
    for i, f in enumerate(frames):
        out = tr.update(f, {raw: box_at(x0 + dx * i, y)}, {raw: feat})
    return out


def test_reappearing_person_gets_old_id():
    tr = TrackRecovery(fps=10)
    walk(tr, 1, range(0, 10), 100, 5, RED)  # 走到 x=145 後消失
    out = tr.update(15, {7: box_at(170, 300)}, {7: RED})  # 5 幀後在預測位置附近出現
    assert out == {7: 1}
    assert tr.merges[0].raw_id == 7 and tr.merges[0].global_id == 1


def test_different_appearance_is_not_merged():
    tr = TrackRecovery(fps=10)
    walk(tr, 1, range(0, 10), 100, 5, RED)
    assert tr.update(15, {7: box_at(170, 300)}, {7: BLUE}) == {7: 7}


def test_too_far_is_not_merged():
    tr = TrackRecovery(fps=10)
    walk(tr, 1, range(0, 10), 100, 5, RED)
    assert tr.update(12, {7: box_at(600, 300)}, {7: RED}) == {7: 7}


def test_gap_longer_than_limit_is_not_merged():
    tr = TrackRecovery(fps=10)
    walk(tr, 1, range(0, 10), 100, 5, RED)
    assert tr.update(9 + 31, {7: box_at(150, 300)}, {7: RED}) == {7: 7}


def test_active_id_is_never_reused():
    tr = TrackRecovery(fps=10)
    tr.update(0, {1: box_at(100, 300)}, {1: RED})
    # 1 號還在畫面上，同一幀出現長得一樣的 2 號，不能被併成 1 號
    out = tr.update(1, {1: box_at(105, 300), 2: box_at(110, 300)}, {1: RED, 2: RED})
    assert out == {1: 1, 2: 2}


def test_original_track_returning_takes_id_back():
    tr = TrackRecovery(fps=10)
    walk(tr, 1, range(0, 10), 100, 5, RED)
    assert tr.update(15, {7: box_at(170, 300)}, {7: RED}) == {7: 1}  # 7 被接成 1
    # 追蹤器把原本的 1 號接回來，同一幀 1 和 7 同時在畫面上 → 1 留下、7 拆回自己的 ID
    out = tr.update(16, {1: box_at(150, 300), 7: box_at(175, 300)}, {1: RED, 7: RED})
    assert out == {1: 1, 7: 7}
    assert tr.revoked == [(16, 7, 1)]


def test_merged_id_stays_merged_on_later_frames():
    tr = TrackRecovery(fps=10)
    walk(tr, 1, range(0, 10), 100, 5, RED)
    tr.update(15, {7: box_at(170, 300)}, {7: RED})
    assert tr.update(16, {7: box_at(175, 300)}, {7: RED}) == {7: 1}
