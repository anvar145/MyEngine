"""Верификация RigidBody: взаимодействие параметров."""

from __future__ import annotations

import numpy as np
import pytest

from engine.core.math3d import Vector3
from engine.core.physics_components import ColliderComponent, RigidBodyComponent
from engine.core.scene import Scene
from engine.physics.body import RigidBody
from engine.physics.collider import Collider
from engine.physics.shape import BoxShape
from engine.physics.world import PhysicsWorld

DT = 1.0 / 60.0
G = 9.81


def _body(world, **kw) -> RigidBody:
    kw.setdefault("position", (0.0, 50.0, 0.0))
    body = RigidBody(**kw)
    world.add_body(body, Collider(BoxShape((0.5, 0.5, 0.5))))
    return body


def _run(world, seconds) -> None:
    for _ in range(round(seconds / DT)):
        world.step()


def test_static_with_gravity_params_does_not_move() -> None:
    world = PhysicsWorld()
    b = _body(world, body_type="static", gravity_scale=5.0, use_gravity=True)
    _run(world, 2.0)
    assert np.allclose(b.position, (0, 50, 0))


def test_kinematic_with_gravity_scale_two_does_not_move() -> None:
    world = PhysicsWorld()
    b = _body(world, body_type="kinematic", gravity_scale=2.0)
    _run(world, 2.0)
    assert np.allclose(b.position, (0, 50, 0))
    assert not b.linear_velocity.any()


def test_dynamic_with_zero_mass_is_rejected_not_silent() -> None:
    world = PhysicsWorld()
    with pytest.raises(ValueError):
        _body(world, mass=0.0)
    assert len(world) == 0


def test_component_with_zero_mass_creates_no_body() -> None:
    scene = Scene()
    go = scene.create_game_object("Box", position=Vector3(0, 5, 0))
    go.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    rb = go.add_component(RigidBodyComponent, mass=0.0)
    scene.fixed_update(DT)
    assert rb.handle is None
    assert go.transform.world_position.y == pytest.approx(5.0)  # не падает


@pytest.mark.parametrize("damping", [0.5, 1.0, 2.0])
def test_terminal_velocity_with_linear_damping(damping: float) -> None:
    world = PhysicsWorld()
    b = _body(world, linear_damping=damping, position=(0, 1e5, 0))
    _run(world, 20.0 / damping)
    # неявное затухание: v* = g / d (установившаяся скорость)
    assert b.linear_velocity[1] == pytest.approx(-G / damping, rel=0.01)


def test_default_body_falls_lands_and_sleeps() -> None:
    scene = Scene()
    floor = scene.create_game_object(
        "Floor", position=Vector3(0, -0.5, 0), scale=Vector3(20, 1, 20)
    )
    floor.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    go = scene.create_game_object("Box", position=Vector3(0, 3, 0))
    go.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    rb = go.add_component(RigidBodyComponent)  # все значения по умолчанию
    for _ in range(300):
        scene.fixed_update(DT)
        scene.update(DT)
    body = scene.physics.world.get_body(rb.handle)
    assert go.transform.world_position.y == pytest.approx(0.5, abs=0.03)
    assert body.sleeping
    assert (body.mass, body.gravity_scale, body.use_gravity) == (1.0, 1.0, True)
    assert (body.linear_damping, body.angular_damping) == (0.05, 0.05)
    assert (body.freeze_rotation, body.can_sleep) == (False, True)


def test_awake_forever_body_still_lies_still_on_floor() -> None:
    world = PhysicsWorld()
    world.add_static(Collider(BoxShape((20.0, 0.5, 20.0))), (0.0, -0.5, 0.0))
    b = _body(world, position=(0.0, 0.5, 0.0), can_sleep=False)
    _run(world, 3.0)
    y = b.position[1]
    _run(world, 5.0)
    assert not b.sleeping
    assert abs(b.position[1] - y) < 1e-3  # нет дрожания/проседания
    assert np.linalg.norm(b.linear_velocity) < 0.05


def test_use_gravity_off_with_freeze_floats_still() -> None:
    world = PhysicsWorld()
    b = _body(world, use_gravity=False, freeze_rotation=True, can_sleep=False)
    _run(world, 2.0)
    assert np.allclose(b.position, (0, 50, 0))
    assert np.allclose(b.rotation, (0, 0, 0, 1))


def test_frozen_body_with_angular_damping_irrelevant() -> None:
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    b = _body(world, freeze_rotation=True, angular_damping=99.0, position=(0, 0, 0))
    world.apply_impulse(world.handles()[0], (0, 0, 1.0), point=(0.5, 0, 0))
    _run(world, 1.0)
    assert not b.angular_velocity.any()


def test_gravity_scale_and_damping_combine() -> None:
    world = PhysicsWorld()
    b = _body(world, gravity_scale=2.0, linear_damping=1.0, position=(0, 1e5, 0))
    _run(world, 20.0)
    assert b.linear_velocity[1] == pytest.approx(-2.0 * G, rel=0.01)


def test_mass_does_not_change_free_fall_with_damping() -> None:
    world = PhysicsWorld()
    light = _body(world, mass=0.1, position=(0, 100, 0))
    heavy = _body(world, mass=100.0, position=(5, 100, 0))
    _run(world, 1.0)
    assert light.position[1] == pytest.approx(heavy.position[1], abs=1e-9)


def test_kinematic_ignores_can_sleep_and_freeze_flags() -> None:
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    b = _body(world, body_type="kinematic", position=(0, 0, 0),
              freeze_rotation=True, can_sleep=False, linear_velocity=(1, 0, 0))
    _run(world, 1.0)
    assert b.position[0] == pytest.approx(1.0, abs=1e-6)
    assert not b.sleeping
