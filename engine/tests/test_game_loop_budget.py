"""GameLoop: бюджет времени фиксированных шагов и троттлинг предупреждений."""

from __future__ import annotations

import logging

import pytest

from engine.core.game_loop import GameLoop

DT = 1.0 / 60.0


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _loop(clock: FakeClock, cost_ms: float, **kw) -> GameLoop:
    loop = GameLoop(fixed_delta=DT, clock=clock, **kw)

    def hook(_dt: float) -> None:
        clock.now += cost_ms / 1000.0

    loop.add_fixed_update(hook)
    return loop


def test_without_budget_behaviour_is_unchanged() -> None:
    clock = FakeClock()
    loop = _loop(clock, 6.0, max_substeps=3)
    loop.step(3.5 * DT)
    assert loop.stats.fixed_steps == 3
    assert loop.stats.budget_stops == 0


def test_budget_limits_substeps_by_predicted_cost() -> None:
    clock = FakeClock()
    loop = _loop(clock, 6.0, max_substeps=3, max_fixed_ms=10.0)
    loop.step(DT)  # прогрев оценки: 1 шаг по 6 мс
    before = loop.stats.fixed_steps
    loop.step(3 * DT)
    # первый шаг всегда; второй (6+6 > 10) не укладывается
    assert loop.stats.fixed_steps - before == 1
    assert loop.stats.budget_stops == 1


def test_cheap_steps_fit_into_budget() -> None:
    clock = FakeClock()
    loop = _loop(clock, 3.0, max_substeps=3, max_fixed_ms=10.0)
    loop.step(DT)
    before = loop.stats.fixed_steps
    loop.step(3 * DT)
    assert loop.stats.fixed_steps - before == 3


def test_minimum_one_step_and_alpha_in_range() -> None:
    clock = FakeClock()
    loop = _loop(clock, 50.0, max_substeps=3, max_fixed_ms=9.0)
    for _ in range(10):
        loop.step(0.04)
        assert loop.stats.fixed_steps >= 1
        assert 0.0 <= loop.time.alpha <= 1.0
        assert loop.accumulator < DT


def test_constant_long_frames_do_not_run_away() -> None:
    clock = FakeClock()
    loop = _loop(clock, 8.0, max_substeps=3, max_fixed_ms=9.0)
    for _ in range(100):
        loop.step(0.025)
    assert loop.accumulator < DT
    assert loop.stats.fixed_steps <= 100 * 2


def test_overrun_warning_is_throttled(caplog: pytest.LogCaptureFixture) -> None:
    clock = FakeClock()
    loop = GameLoop(fixed_delta=DT, max_substeps=2, clock=clock)
    with caplog.at_level(logging.WARNING, logger="engine.core.loop"):
        for _ in range(50):
            loop.step(0.2)
            clock.now += 0.001
    assert loop.stats.overrun_frames == 50
    assert len(caplog.records) == 1
    clock.now += 2.0
    with caplog.at_level(logging.WARNING, logger="engine.core.loop"):
        loop.step(0.2)
    assert len(caplog.records) == 2


def test_budget_warning_is_throttled(caplog: pytest.LogCaptureFixture) -> None:
    clock = FakeClock()
    loop = _loop(clock, 12.0, max_substeps=3, max_fixed_ms=9.0)
    with caplog.at_level(logging.WARNING, logger="engine.core.loop"):
        for _ in range(50):
            loop.step(0.05)
    assert loop.stats.budget_stops == 50
    assert loop.stats.last_budget_stop
    assert loop.stats.last_fixed_steps == 1
    assert len(caplog.records) == 1
    assert "бюджет" in caplog.records[0].getMessage()
    clock.now += 2.0
    with caplog.at_level(logging.WARNING, logger="engine.core.loop"):
        loop.step(0.05)
    assert len(caplog.records) == 2
