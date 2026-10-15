"""Твины: float, Vector3, Quaternion, кортежи, отмена, цепочка."""

from __future__ import annotations

from itertools import pairwise

import pytest

from engine.core.math3d import Quaternion, Vector3
from engine.core.scheduling import Scheduler, TimeCategory, tween


class Box:
    """Цель твина: обычные атрибуты."""

    def __init__(self) -> None:
        self.f = 0.0
        self.i = 0
        self.v = Vector3(0, 0, 0)
        self.q = Quaternion.identity()
        self.color = (0.0, 0.0, 0.0, 1.0)
        self.name = "x"


def test_float_linear_midpoint_and_end() -> None:
    sched, box = Scheduler(), Box()
    tween(sched, box, "f", 1.0, 1.0, ease="linear")
    sched.update_game(0.0)
    sched.update_game(0.5)
    assert box.f == pytest.approx(0.5)
    sched.update_game(0.5)
    assert box.f == 1.0 and sched.count == 0


def test_default_ease_is_in_out_quad_symmetric() -> None:
    sched, box = Scheduler(), Box()
    tween(sched, box, "f", 1.0, 1.0)
    sched.update_game(0.5)
    assert box.f == pytest.approx(0.5)


def test_ease_out_quad_midpoint_above_half() -> None:
    sched, box = Scheduler(), Box()
    tween(sched, box, "f", 1.0, 1.0, ease="ease_out_quad")
    sched.update_game(0.5)
    assert box.f > 0.5


def test_vector3_linear() -> None:
    sched, box = Scheduler(), Box()
    tween(sched, box, "v", Vector3(10, 0, 0), 1.0, ease="linear")
    sched.update_game(0.5)
    assert box.v.x == pytest.approx(5.0)
    sched.update_game(0.5)
    assert box.v.x == 10.0


def test_quaternion_slerp_is_monotonic() -> None:
    sched, box = Scheduler(), Box()
    start = box.q
    end = Quaternion.from_euler(0, 90, 0)
    tween(sched, box, "q", end, 1.0, ease="linear")
    dots = []
    for _ in range(10):
        sched.update_game(0.1)
        dots.append(box.q.dot(start))
    assert dots[0] > 0 and dots[4] > 0
    assert all(b <= a + 1e-9 for a, b in pairwise(dots))
    assert box.q.dot(end) == pytest.approx(1.0)


def test_tuple_per_component() -> None:
    sched, box = Scheduler(), Box()
    tween(sched, box, "color", (1.0, 1.0, 1.0, 0.0), 1.0, ease="linear")
    sched.update_game(0.5)
    assert box.color == pytest.approx((0.5, 0.5, 0.5, 0.5))


def test_tuple_length_mismatch() -> None:
    with pytest.raises(ValueError):
        tween(Scheduler(), Box(), "color", (1.0, 1.0), 1.0)


def test_int_is_float_in_between_and_exact_at_end() -> None:
    sched, box = Scheduler(), Box()
    tween(sched, box, "i", 10, 1.0, ease="linear")
    sched.update_game(0.25)
    assert box.i == pytest.approx(2.5)
    sched.update_game(1.0)
    assert box.i == 10 and isinstance(box.i, int)


def test_cancel_keeps_value_and_skips_on_complete() -> None:
    sched, box, calls = Scheduler(), Box(), []
    handle = tween(sched, box, "f", 1.0, 1.0, ease="linear").on_complete(
        lambda: calls.append(1)
    )
    sched.update_game(0.5)
    handle.cancel()
    sched.update_game(1.0)
    assert box.f == pytest.approx(0.5) and calls == [] and handle.is_cancelled


def test_on_update_each_frame_with_current_value() -> None:
    sched, box, seen = Scheduler(), Box(), []
    tween(sched, box, "f", 1.0, 1.0, ease="linear").on_update(seen.append)
    for _ in range(4):
        sched.update_game(0.25)
    assert seen == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_chain_and_on_complete_once() -> None:
    sched, box, log = Scheduler(), Box(), []
    handle = (
        tween(sched, box, "f", 1.0, 0.5)
        .on_complete(lambda: log.append("done"))
        .on_update(lambda v: log.append("u"))
    )
    for _ in range(3):
        sched.update_game(0.25)
    assert log.count("done") == 1 and handle.is_done and not handle.is_cancelled


def test_zero_dt_does_not_spin() -> None:
    sched, box = Scheduler(), Box()
    tween(sched, box, "f", 1.0, 1.0)
    for _ in range(3):
        assert sched.update_game(0.0) <= 1
    assert sched.count == 1


def test_two_tweens_same_attr_applied_in_insertion_order() -> None:
    sched, box = Scheduler(), Box()
    tween(sched, box, "f", 1.0, 1.0, ease="linear")
    tween(sched, box, "f", 5.0, 1.0, ease="linear")
    sched.update_game(0.5)
    assert box.f == pytest.approx(2.5)  # победил последний добавленный
    sched.update_game(0.5)
    assert box.f == 5.0


def test_duration_zero_sets_now_and_completes_on_next_update() -> None:
    sched, box, calls = Scheduler(), Box(), []
    tween(sched, box, "f", 3.0, 0.0).on_complete(lambda: calls.append(1))
    assert box.f == 3.0 and calls == []
    sched.update_game(0.0)
    assert calls == [1] and sched.count == 0


def test_ui_category_tween_runs_during_pause() -> None:
    sched, box = Scheduler(), Box()
    sched.game_paused = True
    tween(sched, box, "f", 1.0, 1.0, ease="linear", category=TimeCategory.UI)
    sched.update(0.5, 0.5)
    assert box.f == pytest.approx(0.5)


def test_unknown_ease_and_unsupported_type() -> None:
    sched, box = Scheduler(), Box()
    with pytest.raises(ValueError):
        tween(sched, box, "f", 1.0, 1.0, ease="nope")
    with pytest.raises(TypeError, match="name"):
        tween(sched, box, "name", "y", 1.0)
    with pytest.raises(TypeError):
        tween(sched, box, "f", Vector3(), 1.0)
    assert sched.count == 0


def test_exception_in_on_complete_is_logged_and_tween_ends(caplog) -> None:
    sched, box = Scheduler(), Box()
    handle = tween(sched, box, "f", 1.0, 0.1).on_complete(lambda: 1 / 0)
    with caplog.at_level("ERROR"):
        sched.update_game(0.2)
    assert handle.is_done and any(r.exc_info for r in caplog.records)


def test_scheduler_tween_method() -> None:
    sched, box = Scheduler(), Box()
    sched.tween(box, "f", 1.0, 1.0, ease="linear")
    sched.update_game(1.0)
    assert box.f == 1.0
