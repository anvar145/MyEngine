"""Верификация RigidBody: gravity_scale и use_gravity."""

from __future__ import annotations

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


def _fall(seconds: float, **kw):
    """Свободное падение с высоты 100 м без затухания; возвращает (тело, мир)."""
    world = PhysicsWorld()
    kw.setdefault("linear_damping", 0.0)
    body = RigidBody(position=(0.0, 100.0, 0.0), **kw)
    world.add_body(body, Collider(BoxShape((0.5, 0.5, 0.5))))
    for _ in range(round(seconds / DT)):
        world.step()
    return body, world


def test_gravity_acceleration_is_9_81() -> None:
    body, _ = _fall(1.0)
    assert body.linear_velocity[1] == pytest.approx(-G, rel=1e-9)


def test_half_second_fall_distance_matches_discrete_integrator() -> None:
    body, _ = _fall(0.5)
    n = 30
    expected = G * DT * DT * n * (n + 1) / 2.0  # полунеявный Эйлер
    assert 100.0 - body.position[1] == pytest.approx(expected, rel=1e-9)


def test_half_second_fall_distance_close_to_analytic() -> None:
    body, _ = _fall(0.5)
    fallen = 100.0 - body.position[1]
    assert fallen == pytest.approx(0.5 * G * 0.25, rel=0.04)  # ≈1.23 м


def test_half_second_fall_distance_within_five_percent() -> None:
    """Полунеявный Эйлер при 60 Гц: 1.267 м против 1.226 м (+3.3 %)."""
    body, _ = _fall(0.5)
    assert 100.0 - body.position[1] == pytest.approx(0.5 * G * 0.25, rel=0.05)


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0, -1.0])
def test_gravity_scale_multiplies_acceleration(scale: float) -> None:
    body, _ = _fall(1.0, gravity_scale=scale)
    assert body.linear_velocity[1] == pytest.approx(-G * scale, rel=1e-9)


def test_gravity_scale_zero_means_no_fall() -> None:
    body, _ = _fall(1.0, gravity_scale=0.0)
    assert body.position[1] == pytest.approx(100.0, abs=1e-12)
    assert body.linear_velocity[1] == 0.0


def test_negative_gravity_scale_rises() -> None:
    body, _ = _fall(1.0, gravity_scale=-1.0)
    assert body.position[1] > 100.0 + 4.0


def test_use_gravity_false_disables_gravity() -> None:
    body, _ = _fall(1.0, use_gravity=False)
    assert body.position[1] == pytest.approx(100.0, abs=1e-12)


@pytest.mark.parametrize("scale", [0.5, 2.0, -1.0, 100.0])
def test_use_gravity_false_wins_over_gravity_scale(scale: float) -> None:
    body, _ = _fall(1.0, use_gravity=False, gravity_scale=scale)
    assert body.position[1] == pytest.approx(100.0, abs=1e-12)


def test_use_gravity_false_keeps_initial_velocity() -> None:
    body, _ = _fall(1.0, use_gravity=False, linear_velocity=(0.0, 3.0, 0.0))
    assert body.position[1] == pytest.approx(103.0, abs=1e-2)


def test_world_gravity_vector_direction_respected() -> None:
    world = PhysicsWorld(gravity=(5.0, 0.0, 0.0))
    body = RigidBody(linear_damping=0.0)
    world.add_body(body, Collider(BoxShape((0.5, 0.5, 0.5))))
    for _ in range(60):
        world.step()
    assert body.linear_velocity[0] == pytest.approx(5.0)
    assert body.linear_velocity[1] == 0.0


def test_runtime_change_of_gravity_scale_and_use_gravity_via_component() -> None:
    scene = Scene()
    go = scene.create_game_object("Box", position=Vector3(0, 100, 0))
    go.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    rb = go.add_component(RigidBodyComponent, linear_damping=0.0)

    def step(n):
        for _ in range(n):
            scene.fixed_update(DT)
            scene.update(DT)

    step(60)
    assert rb.linear_velocity[1] == pytest.approx(-G, rel=1e-6)
    rb.gravity_scale = 0.0  # пересоздание тела, скорость сохраняется
    v = rb.linear_velocity[1]
    step(60)
    assert rb.linear_velocity[1] == pytest.approx(v, rel=1e-6)
    rb.gravity_scale = 2.0
    step(60)
    assert rb.linear_velocity[1] == pytest.approx(v - 2 * G, rel=1e-6)
    rb.use_gravity = False
    v = rb.linear_velocity[1]
    step(60)
    assert rb.linear_velocity[1] == pytest.approx(v, rel=1e-6)


def test_static_and_kinematic_ignore_gravity_scale() -> None:
    for kind in ("static", "kinematic"):
        world = PhysicsWorld()
        body = RigidBody(body_type=kind, position=(0, 5, 0), gravity_scale=10.0)
        world.add_body(body, Collider(BoxShape((0.5, 0.5, 0.5))))
        for _ in range(60):
            world.step()
        assert body.position[1] == pytest.approx(5.0, abs=1e-12)
