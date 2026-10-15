"""Верификация RigidBody: linear_damping / angular_damping.

Формула мира (``PhysicsWorld.step``): ``v ← v / (1 + d·dt)`` на каждом шаге,
то есть ``v(n) = v0 / (1 + d·dt)^n`` ≈ ``v0·exp(-d·t)``.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from engine.physics.body import RigidBody
from engine.physics.collider import Collider
from engine.physics.shape import BoxShape
from engine.physics.world import PhysicsWorld

DT = 1.0 / 60.0


def _setup(**kw):
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    body = RigidBody(**kw)
    world.add_body(body, Collider(BoxShape((0.5, 0.5, 0.5))))
    return world, body


def _run(world, seconds) -> None:
    for _ in range(round(seconds / world.fixed_dt)):
        world.step()


def _lin(d: float, v0: float = 10.0):
    return _setup(linear_damping=d, angular_damping=0.0,
                  linear_velocity=(v0, 0.0, 0.0))


def _ang(d: float, w0: float = 4.0):
    return _setup(linear_damping=0.0, angular_damping=d,
                  angular_velocity=(0.0, w0, 0.0))


def test_zero_linear_damping_keeps_velocity() -> None:
    world, body = _lin(0.0)
    _run(world, 1.0)
    assert abs(body.linear_velocity[0] - 10.0) < 0.01


def test_zero_angular_damping_keeps_velocity() -> None:
    world, body = _ang(0.0)
    _run(world, 1.0)
    assert abs(body.angular_velocity[1] - 4.0) < 0.01


def test_linear_damping_matches_discrete_formula() -> None:
    d = 0.5
    world, body = _lin(d)
    _run(world, 1.0)
    assert body.linear_velocity[0] == pytest.approx(
        10.0 / (1.0 + d * DT) ** 60, rel=1e-9
    )


def test_linear_damping_is_approximately_exponential() -> None:
    d = 0.5
    world, body = _lin(d)
    _run(world, 1.0)
    v1 = body.linear_velocity[0]
    _run(world, 1.0)
    v2 = body.linear_velocity[0]
    assert v1 / 10.0 == pytest.approx(math.exp(-d), rel=0.01)
    assert v2 / v1 == pytest.approx(v1 / 10.0, rel=1e-9)  # постоянная доля за 1 с


def test_angular_damping_matches_discrete_formula() -> None:
    d = 0.5
    world, body = _ang(d)
    _run(world, 1.0)
    assert body.angular_velocity[1] == pytest.approx(
        4.0 / (1.0 + d * DT) ** 60, rel=1e-9
    )


def test_larger_damping_slows_more_monotonic() -> None:
    finals = []
    for d in (0.0, 0.1, 0.5, 1.0, 3.0, 5.0):
        world, body = _lin(d)
        _run(world, 1.0)
        finals.append(body.linear_velocity[0])
    assert all(a > b for a, b in itertools.pairwise(finals))


def test_damping_one_after_half_second_is_exp_minus_half() -> None:
    """Факт: d=1.0 — это 1/с, а не «полная остановка»."""
    world, body = _lin(1.0)
    _run(world, 0.5)
    assert body.linear_velocity[0] / 10.0 == pytest.approx(math.exp(-0.5), rel=0.01)


def test_damping_one_gives_e_decay_over_half_second() -> None:
    """d=1 — e-затухание за 1/d с: за 0.5 с v/v0 ≈ 0.61 = exp(-0.5) (±2 %)."""
    world, body = _lin(1.0)
    _run(world, 0.5)
    ratio = body.linear_velocity[0] / 10.0
    assert ratio == pytest.approx(0.61, rel=0.02)
    assert ratio == pytest.approx(math.exp(-0.5), rel=0.02)


@pytest.mark.parametrize("dt", [DT, 0.1, 1.0, 10.0])
def test_huge_damping_never_flips_sign(dt: float) -> None:
    world, body = _lin(5.0)
    previous = 10.0
    for _ in range(20):
        world.step(dt)
        v = body.linear_velocity[0]
        assert 0.0 <= v <= previous  # «=» — тело уснуло и скорость обнулена
        previous = v


def test_huge_angular_damping_never_flips_sign() -> None:
    world, body = _ang(50.0)
    for _ in range(60):
        world.step()
        assert body.angular_velocity[1] >= 0.0


def test_linear_damping_does_not_touch_angular_and_vice_versa() -> None:
    world, body = _setup(
        linear_damping=5.0, angular_damping=0.0,
        linear_velocity=(10.0, 0.0, 0.0), angular_velocity=(0.0, 4.0, 0.0),
    )
    _run(world, 1.0)
    assert body.angular_velocity[1] == pytest.approx(4.0, abs=1e-9)
    assert body.linear_velocity[0] < 0.1
    world2, body2 = _setup(
        linear_damping=0.0, angular_damping=5.0,
        linear_velocity=(10.0, 0.0, 0.0), angular_velocity=(0.0, 4.0, 0.0),
    )
    _run(world2, 1.0)
    assert body2.linear_velocity[0] == pytest.approx(10.0, abs=1e-9)
    assert body2.angular_velocity[1] < 0.1


def test_damping_reduces_travelled_distance() -> None:
    free, b0 = _lin(0.0)
    damped, b1 = _lin(2.0)
    _run(free, 1.0)
    _run(damped, 1.0)
    assert b0.position[0] == pytest.approx(10.0, abs=1e-3)
    # интеграл 10·exp(-2t) за 1 с ≈ 4.32
    assert b1.position[0] == pytest.approx(5.0 * (1.0 - math.exp(-2.0)), rel=0.03)


def test_kinematic_is_not_damped() -> None:
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    body = RigidBody(body_type="kinematic", linear_damping=50.0,
                     linear_velocity=(2.0, 0.0, 0.0))
    world.add_body(body, Collider(BoxShape((0.5, 0.5, 0.5))))
    _run(world, 1.0)
    assert np.isclose(body.position[0], 2.0, atol=1e-6)
