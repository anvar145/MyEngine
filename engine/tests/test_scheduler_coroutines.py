"""Корутины планировщика: Wait, WaitForFrame, WaitForFixed, WaitUntil."""

from __future__ import annotations

import pytest

from engine.core.game_loop import GameLoop
from engine.core.scene import Scene
from engine.core.scheduling import (
    Scheduler,
    TimeCategory,
    Wait,
    WaitForFixed,
    WaitForFrame,
    WaitUntil,
    start_coroutine,
)


def test_wait_respects_pause() -> None:
    sched, log = Scheduler(), []

    def seq():
        log.append("a")
        yield Wait(1.0)
        log.append("b")

    start_coroutine(sched, seq())
    sched.update_game(0.0)  # первый шаг
    assert log == ["a"]
    sched.update_game(0.9)
    assert log == ["a"]
    sched.update_game(0.1)
    assert log == ["a", "b"]


def test_sequence_of_waits_has_no_drift() -> None:
    sched, log = Scheduler(), []

    def seq():
        for i in range(3):
            yield Wait(0.5)
            log.append(i)

    start_coroutine(sched, seq())
    sched.update_game(0.0)
    sched.update_game(1.6)  # длинный кадр: каждая пауза отсчитывается от «сейчас»
    assert log == [0]
    sched.update_game(0.5)
    sched.update_game(0.5)
    assert log == [0, 1, 2]


def test_wait_until_waits_for_predicate() -> None:
    sched, log, box = Scheduler(), [], {"ready": False}

    def seq():
        yield WaitUntil(lambda: box["ready"])
        log.append("go")

    start_coroutine(sched, seq())
    for _ in range(5):
        sched.update_game(0.1)
    assert log == []
    box["ready"] = True
    sched.update_game(0.1)
    assert log == ["go"]


def test_wait_for_frame_counts_frames_and_ignores_time_scale() -> None:
    scene = Scene("S")
    loop = GameLoop(scene, time_scale=0.0001)
    log: list[int] = []

    def seq():
        for _ in range(3):
            yield WaitForFrame()
            log.append(scene.scheduler.frame)

    scene.scheduler.start_coroutine(seq())
    loop.step(0.016)  # первый шаг, ставит ожидание кадра
    assert log == []
    for _ in range(3):
        loop.step(0.016)
    assert len(log) == 3 and log == sorted(set(log))


def test_wait_for_frame_waits_exactly_one_update() -> None:
    sched, log = Scheduler(), []

    def seq():
        yield WaitForFrame()
        log.append(sched.frame)

    start_coroutine(sched, seq())
    sched.update_game(0.1)  # frame 1: старт, встали на кадр 2
    assert log == []
    sched.update_game(0.1)
    assert log == [2]


def test_wait_for_fixed_counts_fixed_steps() -> None:
    scene = Scene("S")
    log: list[int] = []

    def seq():
        yield WaitForFixed(3)
        log.append(scene.scheduler.fixed_frame)

    scene.scheduler.start_coroutine(seq())
    scene.update(0.016)  # старт
    scene.fixed_update(0.02)
    scene.fixed_update(0.02)
    assert log == []
    scene.fixed_update(0.02)
    assert log == [3]
    scene.fixed_update(0.02)
    assert log == [3]


def test_time_scale_affects_wait_but_not_unscaled() -> None:
    scene = Scene("S")
    loop = GameLoop(scene, time_scale=0.5)
    log: list[str] = []

    def scaled():
        yield Wait(0.5)
        log.append("scaled")

    def unscaled():
        yield Wait(0.5, unscaled=True)
        log.append("unscaled")

    scene.scheduler.start_coroutine(scaled())
    scene.scheduler.start_coroutine(unscaled())
    for _ in range(6):  # 0.6 сырых: unscaled прошёл 0.5, scaled только 0.3
        loop.step(0.1)
    assert log == ["unscaled"]
    for _ in range(6):
        loop.step(0.1)
    assert log == ["unscaled", "scaled"]


def test_unscaled_wait_stops_on_pause_but_ui_wait_does_not() -> None:
    scene = Scene("S")
    loop = GameLoop(scene)
    log: list[str] = []

    def raw():
        yield Wait(0.3, unscaled=True)
        log.append("raw")

    def ui():
        yield Wait(0.3)
        log.append("ui")

    scene.scheduler.start_coroutine(raw())
    scene.scheduler.start_coroutine(ui(), category=TimeCategory.UI)
    loop.step(0.0)
    loop.pause()
    for _ in range(10):
        loop.step(0.1)
    assert log == ["ui"]
    loop.resume()
    for _ in range(5):
        loop.step(0.1)
    assert log == ["ui", "raw"]


def test_exception_is_logged_and_coroutine_ends(caplog) -> None:
    sched = Scheduler()

    def seq():
        yield Wait(0.1)
        raise RuntimeError("крах")

    handle = start_coroutine(sched, seq())
    with caplog.at_level("ERROR"):
        sched.update_game(0.0)
        sched.update_game(0.2)
    assert any(r.exc_info for r in caplog.records)
    assert handle.is_done and not handle.is_cancelled and sched.count == 0


def test_bad_yield_is_error(caplog) -> None:
    sched = Scheduler()

    def seq():
        yield "oops"

    handle = start_coroutine(sched, seq())
    with caplog.at_level("ERROR"):
        sched.update_game(0.0)
    assert handle.is_done and any(r.exc_info for r in caplog.records)


def test_predicate_exception_ends_coroutine() -> None:
    sched, errors = Scheduler(), []

    def seq():
        yield WaitUntil(lambda: 1 / 0)

    handle = start_coroutine(sched, seq(), on_error=lambda e: errors.append(e))
    sched.update_game(0.0)
    sched.update_game(0.1)
    assert handle.is_done and isinstance(errors[0], ZeroDivisionError)


def test_cancel_all_by_owner_cancels_and_closes_generator() -> None:
    sched, log = Scheduler(), []
    owner = object()

    def seq():
        try:
            yield Wait(10.0)
            log.append("never")
        finally:
            log.append("closed")

    handle = start_coroutine(sched, seq(), owner=owner)
    sched.update_game(0.0)
    assert sched.cancel_all(owner) == 1
    assert handle.is_cancelled and log == ["closed"]
    sched.update_game(20.0)
    assert "never" not in log and sched.count == 0


def test_cancel_frame_waiter() -> None:
    sched, log = Scheduler(), []

    def seq():
        while True:
            yield WaitForFrame()
            log.append(1)

    handle = start_coroutine(sched, seq())
    sched.update_game(0.1)
    sched.update_game(0.1)
    handle.cancel()
    sched.update_game(0.1)
    sched.update_game(0.1)
    assert log == [1] and sched.count == 0


def test_return_value_goes_to_handle_result() -> None:
    sched = Scheduler()

    def seq():
        yield Wait(0.1)
        return 42

    handle = start_coroutine(sched, seq())
    assert handle.result is None
    sched.update_game(0.0)
    sched.update_game(0.2)
    assert handle.is_done and handle.result == 42


def test_coroutine_can_cancel_itself() -> None:
    sched, box = Scheduler(), {}

    def seq():
        yield WaitForFrame()
        box["h"].cancel()
        yield Wait(1.0)

    box["h"] = start_coroutine(sched, seq())
    sched.update_game(0.1)
    sched.update_game(0.1)
    assert box["h"].is_cancelled and sched.count == 0
    assert sched._dead[TimeCategory.GAME] == 0  # noqa: SLF001


def test_wait_zero_is_next_frame_not_busy_loop() -> None:
    sched, log = Scheduler(), []

    def seq():
        for _ in range(3):
            yield Wait(0)
            log.append(sched.frame)

    start_coroutine(sched, seq())
    for _ in range(4):
        sched.update_game(0.0)
    assert log == [2, 3, 4]


def test_numeric_yield_is_wait() -> None:
    sched, log = Scheduler(), []

    def seq():
        yield 0.5
        log.append("x")

    start_coroutine(sched, seq())
    sched.update_game(0.0)
    sched.update_game(0.5)
    assert log == ["x"]


def test_coroutine_can_start_another() -> None:
    sched, log = Scheduler(), []

    def child():
        yield Wait(0.1)
        log.append("child")

    def parent():
        start_coroutine(sched, child())
        yield Wait(0.3)
        log.append("parent")

    start_coroutine(sched, parent())
    sched.update_game(0.0)
    sched.update_game(0.0)
    sched.update_game(0.1)
    assert log == ["child"]
    sched.update_game(0.2)
    assert log == ["child", "parent"]


def test_non_generator_rejected() -> None:
    with pytest.raises(TypeError):
        start_coroutine(Scheduler(), lambda: None)  # type: ignore[arg-type]


def test_scheduler_clear_cancels_waiters() -> None:
    sched, log = Scheduler(), []

    def seq():
        try:
            yield WaitForFixed(5)
        finally:
            log.append("closed")

    start_coroutine(sched, seq())
    sched.update_game(0.0)
    sched.clear()
    assert log == ["closed"] and sched.count == 0
