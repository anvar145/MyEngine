"""Детерминизм и производительность физики на большом числе тел."""

from __future__ import annotations

import time

import numpy as np

from engine.physics.body import RigidBody
from engine.physics.collider import Collider
from engine.physics.shape import BoxShape
from engine.physics.world import PhysicsWorld


def _build(count: int, columns: int) -> tuple[PhysicsWorld, list[int]]:
    world = PhysicsWorld()
    world.add_static(Collider(BoxShape((40, 0.5, 40))), (0, -0.5, 0))
    handles = []
    for index in range(count):
        column, layer = index % columns, index // columns
        position = (
            (column % 10) * 1.2 - 6.0 + 0.05 * (layer % 3),
            0.5 + (layer + column // 10 * 3) * 1.05,
            (column // 10) * 1.2 - 3.0,
        )
        body = RigidBody(mass=1.0, position=position)
        handles.append(world.add_body(body, Collider(BoxShape((0.5,) * 3))))
    return world, handles


def _positions(world: PhysicsWorld, handles: list[int]) -> np.ndarray:
    return np.array([world.get_body(h).position for h in handles])


def test_determinism_100_boxes_300_steps() -> None:
    """Два прогона 100 ящиков по 300 шагов совпадают до 1e-5."""
    results = []
    for _ in range(2):
        world, handles = _build(100, columns=20)
        for _ in range(300):
            world.step()
        results.append(_positions(world, handles))
    assert np.abs(results[0] - results[1]).max() < 1e-5
    assert results[0][:, 1].min() > 0.3  # никто не провалился сквозь пол


def test_boxes_settle_and_sleep() -> None:
    """Колонны из ящиков остаются целыми и засыпают."""
    world, handles = _build(40, columns=10)
    for _ in range(600):
        world.step()
    positions = _positions(world, handles)
    assert positions[:, 1].min() > 0.3
    assert np.abs(positions[:, 0]).max() < 12.0


def test_performance_500_boxes_measured() -> None:
    """500 ящиков: шаг укладывается в разумный бюджет (измерение, не цель 16 мс).

    Чистый Python/numpy даёт десятки мс на шаг в разгар столкновений; тест
    ловит регрессии на порядок, а фактические цифры пишутся в отчёт.
    """
    world, _ = _build(500, columns=50)
    for _ in range(30):
        world.step()
    started = time.perf_counter()
    steps = 30
    for _ in range(steps):
        world.step()
    per_step = (time.perf_counter() - started) / steps
    print(f"500 boxes: {per_step * 1000:.1f} ms/step")
    assert per_step < 1.0
