"""Верификация RigidBody: засыпание (can_sleep, пороги, пробуждение).

Пороги ``LINEAR_THRESHOLD`` / ``ANGULAR_THRESHOLD`` = 0.05 и ``TIME_TO_SLEEP`` =
0.5 с — константы модуля ``sleeping``, а не поля тела.
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import pytest

from engine.physics import sleeping
from engine.physics.body import RigidBody
from engine.physics.collider import Collider
from engine.physics.shape import BoxShape
from engine.physics.world import PhysicsWorld

DT = 1.0 / 60.0


def _free(**kw):
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    kw.setdefault("linear_damping", 0.0)
    kw.setdefault("angular_damping", 0.0)
    body = RigidBody(**kw)
    h = world.add_body(body, Collider(BoxShape((0.5, 0.5, 0.5))))
    return world, h, body


def _steps_until_sleep(world, body, limit=600) -> int | None:
    for i in range(1, limit + 1):
        world.step()
        if body.sleeping:
            return i
    return None


def test_constants_are_as_specified() -> None:
    assert sleeping.LINEAR_THRESHOLD == 0.05
    assert sleeping.ANGULAR_THRESHOLD == 0.05
    assert sleeping.TIME_TO_SLEEP == 0.5


def test_world_sleep_settings_default_to_constants() -> None:
    settings = PhysicsWorld().sleep_settings
    assert settings.linear_threshold == sleeping.LINEAR_THRESHOLD
    assert settings.angular_threshold == sleeping.ANGULAR_THRESHOLD
    assert settings.time_to_sleep == sleeping.TIME_TO_SLEEP


def test_world_time_to_sleep_setting_speeds_up_sleep() -> None:
    world, _, body = _free()
    world.sleep_settings.time_to_sleep = 0.1
    n = _steps_until_sleep(world, body)
    assert n is not None
    assert 0.1 <= n * DT <= 0.1 + 2 * DT


def test_world_linear_threshold_setting_lets_faster_body_sleep() -> None:
    world, _, default = _free(linear_velocity=(0.5, 0.0, 0.0))
    assert _steps_until_sleep(world, default, 120) is None  # 0.5 > 0.05
    world, _, loose = _free(linear_velocity=(0.5, 0.0, 0.0))
    world.sleep_settings.linear_threshold = 1.0
    assert _steps_until_sleep(world, loose, 120) is not None


def test_world_angular_threshold_setting_lets_faster_body_sleep() -> None:
    world, _, body = _free(angular_velocity=(0.0, 0.5, 0.0))
    world.sleep_settings.angular_threshold = 1.0
    assert _steps_until_sleep(world, body, 120) is not None


def test_sleep_settings_are_per_world() -> None:
    a, b = PhysicsWorld(), PhysicsWorld()
    a.sleep_settings.time_to_sleep = 0.1
    assert b.sleep_settings.time_to_sleep == sleeping.TIME_TO_SLEEP


def test_resting_body_falls_asleep_after_half_second() -> None:
    world, _, body = _free()
    n = _steps_until_sleep(world, body)
    assert n is not None
    assert 0.5 <= n * DT <= 0.5 + 2 * DT
    assert not body.sleeping or not body.linear_velocity.any()


def test_not_asleep_before_half_second() -> None:
    world, _, body = _free()
    for _ in range(28):
        world.step()
    assert not body.sleeping


def test_linear_speed_just_below_threshold_sleeps() -> None:
    world, _, body = _free(linear_velocity=(0.049, 0.0, 0.0))
    assert _steps_until_sleep(world, body, 60) is not None


def test_linear_speed_just_above_threshold_never_sleeps() -> None:
    world, _, body = _free(linear_velocity=(0.051, 0.0, 0.0))
    assert _steps_until_sleep(world, body, 240) is None


def test_angular_speed_threshold() -> None:
    world, _, below = _free(angular_velocity=(0.0, 0.049, 0.0))
    assert _steps_until_sleep(world, below, 60) is not None
    world, _, above = _free(angular_velocity=(0.0, 0.051, 0.0))
    assert _steps_until_sleep(world, above, 240) is None


def test_can_sleep_false_never_sleeps() -> None:
    world, _, body = _free(can_sleep=False)
    assert _steps_until_sleep(world, body, 300) is None


def test_sleep_zeroes_velocity_and_body_does_not_move() -> None:
    world, _, body = _free(linear_velocity=(0.04, 0.0, 0.0))
    assert _steps_until_sleep(world, body, 120) is not None
    pos = body.position.copy()
    assert not body.linear_velocity.any()
    body.linear_velocity = np.array([5.0, 0.0, 0.0])  # мимо world.set_velocity
    for _ in range(60):
        world.step()
    assert np.allclose(body.position, pos)  # спящее не интегрируется


def test_wake_by_impulse() -> None:
    world, h, body = _free()
    _steps_until_sleep(world, body)
    world.apply_impulse(h, (5.0, 0.0, 0.0))
    assert not body.sleeping
    for _ in range(30):
        world.step()
    assert body.position[0] > 1.0


def test_wake_by_set_velocity_and_set_pose() -> None:
    world, h, body = _free()
    _steps_until_sleep(world, body)
    world.set_velocity(h, linear=(1.0, 0.0, 0.0))
    assert not body.sleeping
    _steps_until_sleep(world, body, 120)  # v=1 м/с — выше порога, не уснёт
    assert not body.sleeping
    world.set_velocity(h, linear=(0.0, 0.0, 0.0))
    assert _steps_until_sleep(world, body) is not None
    world.set_pose(h, (3.0, 0.0, 0.0))
    assert not body.sleeping


def test_wake_by_collision_with_falling_body() -> None:
    world = PhysicsWorld()
    world.add_static(Collider(BoxShape((20.0, 0.5, 20.0))), (0.0, -0.5, 0.0))
    sleeper = RigidBody(position=(0.0, 0.5, 0.0))
    world.add_body(sleeper, Collider(BoxShape((0.5, 0.5, 0.5))))
    for _ in range(120):
        world.step()
    assert sleeper.sleeping
    faller = RigidBody(position=(0.4, 4.0, 0.0))
    world.add_body(faller, Collider(BoxShape((0.5, 0.5, 0.5))))
    woke_at = None
    for i in range(240):
        world.step()
        if not sleeper.sleeping and woke_at is None:
            woke_at = i
    assert woke_at is not None


def test_no_sinking_while_asleep_on_floor() -> None:
    world = PhysicsWorld()
    world.add_static(Collider(BoxShape((20.0, 0.5, 20.0))), (0.0, -0.5, 0.0))
    body = RigidBody(position=(0.0, 1.0, 0.0))
    world.add_body(body, Collider(BoxShape((0.5, 0.5, 0.5))))
    for _ in range(180):
        world.step()
    assert body.sleeping
    y = body.position[1]
    assert y == pytest.approx(0.5, abs=0.03)
    for _ in range(1200):  # 20 с сна
        world.step()
    assert body.position[1] == y  # ни микрометра


def test_island_sleeps_and_wakes_together() -> None:
    world = PhysicsWorld()
    world.add_static(Collider(BoxShape((20.0, 0.5, 20.0))), (0.0, -0.5, 0.0))
    bodies = []
    for i in range(3):
        b = RigidBody(position=(0.0, 0.5 + i * 1.0, 0.0))
        world.add_body(b, Collider(BoxShape((0.5, 0.5, 0.5))))
        bodies.append(b)
    for _ in range(300):
        world.step()
    assert all(b.sleeping for b in bodies)
    world.apply_impulse(world.handles()[-1], (1.0, 0.0, 0.0))
    assert not any(b.sleeping for b in bodies)


def test_can_sleep_false_body_keeps_island_awake() -> None:
    world = PhysicsWorld()
    world.add_static(Collider(BoxShape((20.0, 0.5, 20.0))), (0.0, -0.5, 0.0))
    a = RigidBody(position=(0.0, 0.5, 0.0), can_sleep=False)
    b = RigidBody(position=(0.0, 1.5, 0.0))
    world.add_body(a, Collider(BoxShape((0.5, 0.5, 0.5))))
    world.add_body(b, Collider(BoxShape((0.5, 0.5, 0.5))))
    for _ in range(300):
        world.step()
    assert not a.sleeping
    assert not b.sleeping  # остров спит только целиком


def _step_ms(world, steps=120) -> float:
    for _ in range(10):
        world.step()
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        world.step()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000.0


def test_hundred_sleeping_bodies_cost_under_one_ms() -> None:
    base = PhysicsWorld()
    base.add_static(Collider(BoxShape((60.0, 0.5, 60.0))), (0.0, -0.5, 0.0))
    world = PhysicsWorld()
    world.add_static(Collider(BoxShape((60.0, 0.5, 60.0))), (0.0, -0.5, 0.0))
    bodies = []
    for i in range(100):
        b = RigidBody(position=((i % 10) * 3.0 - 13.5, 0.5, (i // 10) * 3.0 - 13.5))
        world.add_body(b, Collider(BoxShape((0.5, 0.5, 0.5))))
        bodies.append(b)
    for _ in range(240):
        world.step()
    assert all(b.sleeping for b in bodies)
    diff = _step_ms(world) - _step_ms(base)
    print(f"\n100 спящих: +{diff:.3f} мс/шаг")
    assert diff < 1.0
