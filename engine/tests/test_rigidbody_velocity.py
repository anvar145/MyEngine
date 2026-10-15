"""Верификация RigidBody: линейная и угловая скорость."""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.physics.body import RigidBody
from engine.physics.collider import Collider
from engine.physics.mathutil import quat_from_axis_angle
from engine.physics.shape import BoxShape
from engine.physics.world import MAX_ANGULAR_SPEED, MAX_LINEAR_SPEED, PhysicsWorld


def _world_body(**kw):
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    kw.setdefault("linear_damping", 0.0)
    kw.setdefault("angular_damping", 0.0)
    body = RigidBody(**kw)
    h = world.add_body(body, Collider(BoxShape((0.5, 0.5, 0.5))))
    return world, h, body


def _run(world, seconds) -> None:
    for _ in range(round(seconds / world.fixed_dt)):
        world.step()


def test_linear_velocity_moves_one_metre_in_one_second() -> None:
    world, _, body = _world_body(linear_velocity=(1.0, 0.0, 0.0))
    _run(world, 1.0)
    assert body.position[0] == pytest.approx(1.0, abs=1e-3)
    assert body.position[1] == pytest.approx(0.0, abs=1e-9)
    assert body.position[2] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_linear_velocity_each_axis(axis: int) -> None:
    v = np.zeros(3)
    v[axis] = -2.5
    world, _, body = _world_body(linear_velocity=v)
    _run(world, 1.0)
    expected = np.zeros(3)
    expected[axis] = -2.5
    assert np.allclose(body.position, expected, atol=1e-3)


def test_zero_velocity_stays_in_place() -> None:
    world, _, body = _world_body()
    _run(world, 1.0)
    assert np.allclose(body.position, 0.0, atol=1e-12)
    assert np.allclose(body.rotation, (0, 0, 0, 1), atol=1e-12)


def test_angular_velocity_pi_rad_per_s_rotates_180_degrees_in_one_second() -> None:
    world, _, body = _world_body(angular_velocity=(0.0, math.pi, 0.0))
    _run(world, 1.0)
    expected = quat_from_axis_angle((0.0, 1.0, 0.0), math.pi)
    assert abs(float(np.dot(body.rotation, expected))) == pytest.approx(1.0, abs=1e-3)
    assert np.linalg.norm(body.rotation) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("axis", [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)])
def test_angular_velocity_other_axes_quarter_turn(axis) -> None:
    w = np.array(axis) * (math.pi / 2)
    world, _, body = _world_body(angular_velocity=w)
    _run(world, 1.0)
    expected = quat_from_axis_angle(axis, math.pi / 2)
    assert abs(float(np.dot(body.rotation, expected))) == pytest.approx(1.0, abs=1e-3)


def test_runtime_reversal_of_linear_velocity_returns_to_start() -> None:
    world, h, body = _world_body(linear_velocity=(3.0, 0.0, 0.0))
    _run(world, 0.5)
    assert body.position[0] == pytest.approx(1.5, abs=1e-3)
    world.set_velocity(h, linear=(-3.0, 0.0, 0.0))
    _run(world, 0.5)
    assert body.position[0] == pytest.approx(0.0, abs=1e-3)


def test_runtime_reversal_of_angular_velocity_returns_to_start() -> None:
    world, h, body = _world_body(angular_velocity=(0.0, 2.0, 0.0))
    _run(world, 0.5)
    assert abs(body.rotation[3]) < 0.9  # повернулось
    world.set_velocity(h, angular=(0.0, -2.0, 0.0))
    _run(world, 0.5)
    assert abs(float(np.dot(body.rotation, (0, 0, 0, 1)))) == pytest.approx(
        1.0, abs=1e-3
    )


def test_set_velocity_none_keeps_other_component() -> None:
    world, h, body = _world_body(
        linear_velocity=(1.0, 0.0, 0.0), angular_velocity=(0.0, 1.0, 0.0)
    )
    world.set_velocity(h, linear=(2.0, 0.0, 0.0))
    assert body.angular_velocity[1] == 1.0
    world.set_velocity(h, angular=(0.0, 3.0, 0.0))
    assert body.linear_velocity[0] == 2.0


def test_velocity_is_clamped_to_world_limits() -> None:
    world, _, body = _world_body(
        linear_velocity=(1000.0, 0.0, 0.0), angular_velocity=(0.0, 1000.0, 0.0)
    )
    world.step()
    assert np.linalg.norm(body.linear_velocity) == pytest.approx(MAX_LINEAR_SPEED)
    assert np.linalg.norm(body.angular_velocity) == pytest.approx(MAX_ANGULAR_SPEED)


def test_freeze_rotation_ignores_angular_velocity() -> None:
    """Ожидание: freeze_rotation запрещает вращение, даже если задана ω."""
    world, _, body = _world_body(
        freeze_rotation=True, angular_velocity=(0.0, math.pi, 0.0)
    )
    _run(world, 1.0)
    assert abs(float(np.dot(body.rotation, (0, 0, 0, 1)))) == pytest.approx(
        1.0, abs=1e-6
    )


def test_freeze_rotation_keeps_linear_motion() -> None:
    world, _, body = _world_body(freeze_rotation=True, linear_velocity=(1.0, 0, 0))
    _run(world, 1.0)
    assert body.position[0] == pytest.approx(1.0, abs=1e-3)
