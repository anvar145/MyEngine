"""Верификация Collider: форма Capsule (radius, half_height / height)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.core.math3d import Quaternion, Vector3
from engine.core.physics_components import ColliderComponent, RigidBodyComponent
from engine.core.scene import Scene
from engine.physics.body import RigidBody
from engine.physics.collider import Collider
from engine.physics.shape import BoxShape, CapsuleShape
from engine.physics.world import PhysicsWorld

DT = 1.0 / 60.0
ROT_Z_90 = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))  # ось X


def _floor(world: PhysicsWorld) -> None:
    world.add_static(Collider(BoxShape((50.0, 0.5, 50.0))), (0.0, -0.5, 0.0))


def _drop(radius, half_height, rotation=(0.0, 0.0, 0.0, 1.0), freeze=False,
          y=4.0, seconds=4.0):
    world = PhysicsWorld()
    _floor(world)
    body = RigidBody(position=(0.0, y, 0.0), rotation=rotation,
                     freeze_rotation=freeze)
    h = world.add_body(body, Collider(CapsuleShape(radius, half_height)))
    for _ in range(round(seconds / DT)):
        world.step()
    return world, h, body


# --- физика: radius / half_height ------------------------------------------

@pytest.mark.parametrize(("r", "hh"), [(0.5, 1.0), (0.25, 0.25), (0.1, 2.0)])
def test_vertical_aabb_height_is_two_times_radius_plus_half_height(r, hh) -> None:
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    h = world.add_body(RigidBody(position=(1.0, 2.0, 3.0)),
                       Collider(CapsuleShape(r, hh)))
    lo, hi = world.get_colliders(h)[0].world_aabb
    assert hi[1] - lo[1] == pytest.approx(2.0 * (r + hh))
    assert hi[0] - lo[0] == pytest.approx(2.0 * r)
    assert hi[2] - lo[2] == pytest.approx(2.0 * r)


@pytest.mark.parametrize(("r", "hh"), [(0.5, 1.0), (0.3, 0.2)])
def test_vertical_fall_rests_at_radius_plus_half_height(r, hh) -> None:
    _, _, body = _drop(r, hh, freeze=True)
    assert body.position[1] == pytest.approx(r + hh, abs=0.02)


def test_horizontal_capsule_aabb_swaps_axes() -> None:
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    h = world.add_body(RigidBody(rotation=ROT_Z_90), Collider(CapsuleShape(0.5, 1.0)))
    lo, hi = world.get_colliders(h)[0].world_aabb
    assert hi[0] - lo[0] == pytest.approx(3.0)
    assert hi[1] - lo[1] == pytest.approx(1.0)


def test_horizontal_capsule_rests_at_radius() -> None:
    _, _, body = _drop(0.5, 1.0, rotation=ROT_Z_90, y=3.0, seconds=3.0)
    assert body.position[1] == pytest.approx(0.5, abs=0.03)


def test_thin_long_capsule_stands_at_full_half_height() -> None:
    _, _, body = _drop(0.1, 2.0, freeze=True)
    assert body.position[1] == pytest.approx(2.1, abs=0.02)


def test_thin_long_capsule_lying_rests_at_radius() -> None:
    _, _, body = _drop(0.1, 2.0, rotation=ROT_Z_90, y=2.0, seconds=3.0)
    assert body.position[1] == pytest.approx(0.1, abs=0.03)


def test_capsule_half_height_changes_stop_height() -> None:
    _, _, short = _drop(0.4, 0.2, freeze=True)
    _, _, tall = _drop(0.4, 1.2, freeze=True)
    assert tall.position[1] - short.position[1] == pytest.approx(1.0, abs=0.03)


def test_zero_half_height_rejected_by_shape() -> None:
    with pytest.raises(ValueError):
        CapsuleShape(0.5, 0.0)


def test_zero_or_negative_radius_rejected_by_shape() -> None:
    for radius in (0.0, -0.1):
        with pytest.raises(ValueError):
            CapsuleShape(radius, 1.0)


# --- компонент: radius / height ---------------------------------------------

def _scene_capsule(radius=0.5, height=2.0, scale=(1.0, 1.0, 1.0), y=4.0,
                   freeze=True):
    scene = Scene()
    floor = scene.create_game_object("Floor", position=Vector3(0, -0.5, 0),
                                     scale=Vector3(100, 1, 100))
    floor.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    go = scene.create_game_object("Capsule", position=Vector3(0, y, 0),
                                  scale=Vector3(*scale))
    col = go.add_component(ColliderComponent, shape="capsule", radius=radius,
                           height=height)
    rb = go.add_component(RigidBodyComponent, freeze_rotation=freeze)
    scene.fixed_update(DT)
    return scene, go, col, rb


def _body(scene, go) -> RigidBody:
    return scene.physics.world.get_body(scene.physics.handle_of(go))


def _aabb(scene, go):
    return scene.physics.world.get_colliders(scene.physics.handle_of(go))[0].world_aabb


def _settle(scene, seconds: float = 3.0) -> None:
    for _ in range(round(seconds / DT)):
        scene.fixed_update(DT)


def test_component_height_is_full_height_with_caps() -> None:
    scene, go, _, _ = _scene_capsule(radius=0.5, height=3.0)
    lo, hi = _aabb(scene, go)
    assert hi[1] - lo[1] == pytest.approx(3.0, abs=0.01)
    assert hi[0] - lo[0] == pytest.approx(1.0, abs=0.01)
    _settle(scene)
    assert _body(scene, go).position[1] == pytest.approx(1.5, abs=0.02)


def test_component_scale_radius_uses_xz_and_height_uses_y() -> None:
    scene, go, _, _ = _scene_capsule(radius=0.5, height=2.0, scale=(2.0, 3.0, 1.0))
    lo, hi = _aabb(scene, go)
    assert hi[0] - lo[0] == pytest.approx(2.0, abs=0.01)  # 0.5 · max(2, 1) · 2
    assert hi[1] - lo[1] == pytest.approx(6.0, abs=0.01)  # 2.0 · 3


def test_component_height_not_larger_than_diameter_is_clamped() -> None:
    """height ≤ 2·radius → цилиндрическая часть 1e-3 (в физике — ValueError)."""
    for height in (1.0, 0.2, 0.0, -2.0):
        scene, go, _, rb = _scene_capsule(radius=0.5, height=height)
        assert rb.handle is not None
        lo, hi = _aabb(scene, go)
        assert hi[1] - lo[1] == pytest.approx(1.0 + 0.002, abs=1e-3)


def test_component_zero_radius_is_clamped() -> None:
    scene, go, _, rb = _scene_capsule(radius=0.0, height=2.0)
    assert rb.handle is not None
    lo, hi = _aabb(scene, go)
    assert hi[0] - lo[0] == pytest.approx(0.002, abs=1e-4)


def test_component_runtime_height_and_radius_change() -> None:
    scene, go, col, _ = _scene_capsule(radius=0.5, height=2.0)
    _settle(scene, 2.0)
    assert _body(scene, go).position[1] == pytest.approx(1.0, abs=0.02)
    col.height = 4.0
    _settle(scene, 3.0)
    assert _body(scene, go).position[1] == pytest.approx(2.0, abs=0.03)
    col.radius = 1.0
    _settle(scene, 3.0)
    lo, hi = _aabb(scene, go)
    assert hi[0] - lo[0] == pytest.approx(2.0, abs=0.01)
    assert hi[1] - lo[1] == pytest.approx(4.0, abs=0.01)


def test_capsule_ignores_size_field() -> None:
    scene = Scene()
    go = scene.create_game_object("C", position=Vector3(0, 3, 0))
    go.add_component(ColliderComponent, shape="capsule", radius=0.5, height=2.0,
                     size=(9.0, 9.0, 9.0))
    go.add_component(RigidBodyComponent)
    scene.fixed_update(DT)
    lo, hi = _aabb(scene, go)
    assert np.allclose(hi - lo, (1.0, 2.0, 1.0), atol=0.01)


def test_rotated_object_rotates_capsule_axis() -> None:
    scene = Scene()
    rot = Quaternion(*ROT_Z_90)
    go = scene.create_game_object("C", position=Vector3(0, 3, 0), rotation=rot)
    go.add_component(ColliderComponent, shape="capsule", radius=0.5, height=3.0)
    go.add_component(RigidBodyComponent, use_gravity=False)
    scene.fixed_update(DT)
    lo, hi = _aabb(scene, go)
    assert hi[0] - lo[0] == pytest.approx(3.0, abs=0.01)
    assert hi[1] - lo[1] == pytest.approx(1.0, abs=0.01)
