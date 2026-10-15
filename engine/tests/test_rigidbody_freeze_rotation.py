"""Верификация RigidBody: freeze_rotation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.core.math3d import Vector3
from engine.core.physics_components import ColliderComponent, RigidBodyComponent
from engine.core.scene import Scene
from engine.physics.body import RigidBody
from engine.physics.collider import Collider
from engine.physics.shape import BoxShape, SphereShape
from engine.physics.world import PhysicsWorld

DT = 1.0 / 60.0
IDENT = np.array([0.0, 0.0, 0.0, 1.0])


def _tilt(body: RigidBody) -> float:
    """Отклонение поворота от единичного (0 — нет поворота)."""
    return 1.0 - abs(float(np.dot(body.rotation, IDENT)))


def _box_world(freeze: bool, gravity=(0.0, 0.0, 0.0)):
    world = PhysicsWorld(gravity=gravity)
    body = RigidBody(freeze_rotation=freeze, linear_damping=0.0,
                     angular_damping=0.0)
    h = world.add_body(body, Collider(BoxShape((0.5, 0.5, 0.5))))
    return world, h, body


def test_6a_off_center_impulse_translates_but_does_not_rotate_when_frozen() -> None:
    world, h, body = _box_world(True)
    world.apply_impulse(h, (0.0, 0.0, 2.0), point=(0.5, 0.0, 0.0))
    assert not body.angular_velocity.any()
    for _ in range(60):
        world.step()
    assert body.position[2] == pytest.approx(2.0, abs=1e-3)
    assert _tilt(body) == pytest.approx(0.0, abs=1e-12)


def test_6a_same_impulse_rotates_when_not_frozen() -> None:
    world, h, body = _box_world(False)
    world.apply_impulse(h, (0.0, 0.0, 2.0), point=(0.5, 0.0, 0.0))
    assert abs(body.angular_velocity[1]) > 0.1
    for _ in range(60):
        world.step()
    assert _tilt(body) > 0.005  # ≈16° за 1 с


def test_frozen_inverse_inertia_is_zero() -> None:
    _, _, frozen = _box_world(True)
    _, _, free = _box_world(False)
    assert not frozen.inv_inertia_world.any()
    assert free.inv_inertia_world.any()


def _ball_hits_box(freeze: bool):
    """Шар бьёт по боковой грани ящика мимо центра масс."""
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    box = RigidBody(position=(2.0, 0.0, 0.0), freeze_rotation=freeze,
                    linear_damping=0.0, angular_damping=0.0)
    world.add_body(box, Collider(BoxShape((0.5, 0.5, 0.5))))
    ball = RigidBody(position=(0.0, 0.0, 0.4), linear_damping=0.0,
                     linear_velocity=(5.0, 0.0, 0.0))
    world.add_body(ball, Collider(SphereShape(0.3)))
    for _ in range(90):
        world.step()
    return box


def test_6b_collision_off_center_frozen_does_not_rotate() -> None:
    box = _ball_hits_box(True)
    assert box.position[0] > 2.1  # удар состоялся
    assert _tilt(box) == pytest.approx(0.0, abs=1e-9)
    assert not box.angular_velocity.any()


def test_6b_collision_off_center_free_rotates() -> None:
    box = _ball_hits_box(False)
    assert box.position[0] > 2.1
    assert _tilt(box) > 0.001 or np.linalg.norm(box.angular_velocity) > 0.01


def test_6c_manual_angular_velocity_is_ignored_when_frozen() -> None:
    world, h, body = _box_world(True)
    world.set_velocity(h, angular=(0.0, math.pi, 0.0))
    for _ in range(60):
        world.step()
    assert _tilt(body) == pytest.approx(0.0, abs=1e-6)


def test_frozen_box_stands_upright_on_ledge_edge() -> None:
    world = PhysicsWorld()
    world.add_static(Collider(BoxShape((1.0, 0.5, 1.0))), (0.0, -0.5, 0.0))
    body = RigidBody(position=(1.2, 1.0, 0.0), freeze_rotation=True)
    world.add_body(body, Collider(BoxShape((0.5, 0.5, 0.5))))
    for _ in range(240):
        world.step()
    assert _tilt(body) == pytest.approx(0.0, abs=1e-9)


def _scene_box(**kw):
    scene = Scene()
    go = scene.create_game_object("Box", position=Vector3(0, 10, 0))
    go.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    rb = go.add_component(RigidBodyComponent, linear_damping=0.0,
                          angular_damping=0.0, **kw)
    scene.physics.world.set_gravity((0, 0, 0))
    scene.fixed_update(DT)
    return scene, rb


def test_6d_runtime_toggle_through_component() -> None:
    _, rb = _scene_box()
    rb.apply_impulse((0, 0, 2.0), point=(0.5, 10.0, 0.0))
    assert abs(rb.angular_velocity[1]) > 0.1  # вращается
    rb.angular_velocity = (0, 0, 0)
    rb.freeze_rotation = True  # пересоздание тела
    rb.apply_impulse((0, 0, 2.0), point=(0.5, 10.0, 0.0))
    assert tuple(rb.angular_velocity) == (0.0, 0.0, 0.0)
    assert rb.linear_velocity[2] > 0.0
    rb.freeze_rotation = False
    rb.linear_velocity = (0, 0, 0)
    rb.apply_impulse((0, 0, 2.0), point=(0.5, 10.0, 0.0))
    assert abs(rb.angular_velocity[1]) > 0.1


def test_direct_toggle_on_live_body_has_effect() -> None:
    world, h, body = _box_world(False)
    body.freeze_rotation = True
    world.apply_impulse(h, (0, 0, 2.0), point=(0.5, 0.0, 0.0))
    assert not body.angular_velocity.any()


def test_direct_unfreeze_restores_rotation() -> None:
    world, h, body = _box_world(True)
    body.freeze_rotation = False
    world.apply_impulse(h, (0, 0, 2.0), point=(0.5, 0.0, 0.0))
    assert abs(body.angular_velocity[1]) > 0.1
