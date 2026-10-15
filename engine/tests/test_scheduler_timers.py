"""Таймеры after / every."""

from __future__ import annotations

import pytest

from engine.core.scheduling import Scheduler, TimeCategory, after, every


def test_after_fires_exactly_at_time_not_earlier() -> None:
    sched, calls = Scheduler(), []
    after(sched, 0.5, lambda: calls.append(1))
    for _ in range(4):
        sched.update_game(0.1)
    assert calls == []
    sched.update_game(0.1)
    assert calls == [1]
    sched.update_game(5.0)
    assert calls == [1]  # одноразовый


def test_after_cancel_before_fire() -> None:
    sched, calls = Scheduler(), []
    handle = after(sched, 0.5, lambda: calls.append(1))
    handle.cancel()
    sched.update_game(1.0)
    assert calls == [] and handle.is_cancelled


def test_after_zero_fires_on_next_update_and_negative_rejected() -> None:
    sched, calls = Scheduler(), []
    after(sched, 0.0, lambda: calls.append(1))
    sched.update_game(0.0)
    assert calls == [1]
    with pytest.raises(ValueError):
        after(sched, -1.0, lambda: None)


def test_every_fires_five_times_in_half_second() -> None:
    sched, calls = Scheduler(), []
    every(sched, 0.1, lambda: calls.append(1))
    for _ in range(5):
        sched.update_game(0.1)
    assert len(calls) == 5


def test_every_immediate_fires_six_times() -> None:
    sched, calls = Scheduler(), []
    every(sched, 0.1, lambda: calls.append(1), immediate=True)
    sched.update_game(0.0)
    assert len(calls) == 1
    for _ in range(5):
        sched.update_game(0.1)
    assert len(calls) == 6


def test_every_catches_up_after_long_frame_without_drift() -> None:
    sched, calls = Scheduler(), []
    every(sched, 0.1, lambda: calls.append(1))
    sched.update_game(0.55)
    assert len(calls) == 5
    sched.update_game(0.05)
    assert len(calls) == 6  # шестое ровно в 0.6, а не 0.65


def test_every_cancel_stops_it() -> None:
    sched, calls = Scheduler(), []
    handle = every(sched, 0.1, lambda: calls.append(1))
    sched.update_game(0.25)
    handle.cancel()
    sched.update_game(1.0)
    assert len(calls) == 2 and sched.count == 0


def test_every_can_cancel_itself() -> None:
    sched, calls = Scheduler(), []
    box = {}

    def tick() -> None:
        calls.append(1)
        if len(calls) == 3:
            box["h"].cancel()

    box["h"] = every(sched, 0.1, tick)
    sched.update_game(2.0)
    assert len(calls) == 3 and sched.count == 0


def test_every_non_positive_period_rejected() -> None:
    sched = Scheduler()
    with pytest.raises(ValueError):
        every(sched, 0.0, lambda: None)


def test_exception_in_after_is_logged_and_task_ends(caplog) -> None:
    sched = Scheduler()

    def boom() -> None:
        raise RuntimeError("крах")

    handle = after(sched, 0.1, boom)
    with caplog.at_level("ERROR"):
        sched.update_game(0.2)
    assert "крах" in caplog.text or any(r.exc_info for r in caplog.records)
    assert handle.is_done and not handle.is_cancelled and sched.count == 0


def test_exception_in_every_is_logged_and_task_continues(caplog) -> None:
    sched, calls = Scheduler(), []

    def boom() -> None:
        calls.append(1)
        raise RuntimeError("крах")

    handle = every(sched, 0.1, boom)
    with caplog.at_level("ERROR"):
        for _ in range(3):
            sched.update_game(0.1)
    assert len(calls) == 3 and not handle.is_done
    assert len([r for r in caplog.records if r.exc_info]) == 3


def test_on_error_hook_replaces_logging(caplog) -> None:
    sched, errors = Scheduler(), []
    after(
        sched, 0.1, lambda: 1 / 0, on_error=lambda exc: errors.append(type(exc))
    )
    with caplog.at_level("ERROR"):
        sched.update_game(0.2)
    assert errors == [ZeroDivisionError] and caplog.records == []


def test_owner_is_used_by_cancel_all() -> None:
    sched, calls = Scheduler(), []
    mine = object()
    after(sched, 0.1, lambda: calls.append("a"), owner=mine)
    every(sched, 0.1, lambda: calls.append("e"), owner=mine)
    after(sched, 0.1, lambda: calls.append("free"))
    assert sched.cancel_all(mine) == 2
    sched.update_game(0.5)
    assert calls == ["free"]


def test_ui_category_timer_ignores_pause() -> None:
    sched, calls = Scheduler(), []
    sched.game_paused = True
    after(sched, 0.1, lambda: calls.append("ui"), category=TimeCategory.UI)
    after(sched, 0.1, lambda: calls.append("game"))
    sched.update(0.2, 0.2)
    assert calls == ["ui"]


def test_scheduler_methods_delegate() -> None:
    sched, calls = Scheduler(), []
    sched.after(0.1, lambda: calls.append("a"))
    sched.every(0.1, lambda: calls.append("e"), immediate=True)
    sched.update_game(0.1)
    # e(0.0) сразу, затем равные 0.1: a поставлен раньше, e переставлен позже
    assert calls == ["e", "a", "e"]
