"""Верификация Collider: форма Sphere (radius)."""

from __future__ import annotations

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


def _floor(world: PhysicsWorld) -> None:
    world.add_static(Collider(BoxShape((50.0, 0.5, 50.0))), (0.0, -0.5, 0.0))


# --- физика: radius -----------------------------------------------------------

@pytest.mark.parametrize("radius", [0.05, 0.5, 2.0])
def test_aabb_is_position_plus_minus_radius(radius: float) -> None:
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    h = world.add_body(RigidBody(position=(1.0, 2.0, 3.0)),
                       Collider(SphereShape(radius)))
    lo, hi = world.get_colliders(h)[0].world_aabb
    assert np.allclose(lo, np.array([1.0, 2.0, 3.0]) - radius)
    assert np.allclose(hi, np.array([1.0, 2.0, 3.0]) + radius)


@pytest.mark.parametrize("radius", [0.1, 0.5, 1.5])
def test_rest_height_equals_radius(radius: float) -> None:
    world = PhysicsWorld()
    _floor(world)
    ball = RigidBody(position=(0.0, 3.0, 0.0))
    world.add_body(ball, Collider(SphereShape(radius)))
    for _ in range(240):
        world.step()
    assert ball.position[1] == pytest.approx(radius, abs=0.02)


@pytest.mark.parametrize(("r1", "r2"), [(0.5, 0.5), (0.3, 1.0), (1.0, 0.2)])
def test_spheres_touch_at_sum_of_radii(r1: float, r2: float) -> None:
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    world.add_static(Collider(SphereShape(r1)), (0.0, 0.0, 0.0))
    body = RigidBody(position=(5.0, 0.0, 0.0), linear_velocity=(-2.0, 0.0, 0.0),
                     linear_damping=0.0)
    world.add_body(body, Collider(SphereShape(r2)))
    closest = 99.0
    for _ in range(240):
        world.step()
        closest = min(closest, float(body.position[0]))
    # после удара шар отлетает на ≈18 % (docs/CORE.md, B1): меряем сближение
    assert closest == pytest.approx(r1 + r2, abs=0.1)


def test_larger_sphere_stops_earlier() -> None:
    def closest(radius: float) -> float:
        world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
        world.add_static(Collider(BoxShape((0.5, 5.0, 5.0))), (0.0, 0.0, 0.0))
        body = RigidBody(position=(5.0, 0.0, 0.0), linear_velocity=(-2.0, 0.0, 0.0),
                         linear_damping=0.0)
        world.add_body(body, Collider(SphereShape(radius)))
        best = 99.0
        for _ in range(240):
            world.step()
            best = min(best, float(body.position[0]))
        return best

    assert closest(1.0) - closest(0.25) == pytest.approx(0.75, abs=0.1)


def test_zero_radius_rejected_by_shape() -> None:
    with pytest.raises(ValueError):
        SphereShape(0.0)


def test_negative_radius_rejected_by_shape() -> None:
    with pytest.raises(ValueError):
        SphereShape(-0.5)


def test_nan_and_inf_radius_rejected_by_shape() -> None:
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            SphereShape(value)


# --- компонент: radius --------------------------------------------------------

def _scene_ball(radius: float, scale=(1.0, 1.0, 1.0), y: float = 3.0):
    scene = Scene()
    floor = scene.create_game_object("Floor", position=Vector3(0, -0.5, 0),
                                     scale=Vector3(100, 1, 100))
    floor.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    go = scene.create_game_object("Ball", position=Vector3(0, y, 0),
                                  scale=Vector3(*scale))
    col = go.add_component(ColliderComponent, shape="sphere", radius=radius)
    rb = go.add_component(RigidBodyComponent)
    scene.fixed_update(DT)
    return scene, go, col, rb


def _body(scene, go) -> RigidBody:
    return scene.physics.world.get_body(scene.physics.handle_of(go))


def _aabb(scene, go):
    return scene.physics.world.get_colliders(scene.physics.handle_of(go))[0].world_aabb


def _settle(scene, seconds: float = 3.0) -> None:
    for _ in range(round(seconds / DT)):
        scene.fixed_update(DT)


def test_component_radius_sets_aabb_and_rest_height() -> None:
    scene, go, _, _ = _scene_ball(0.75)
    lo, hi = _aabb(scene, go)
    assert np.allclose(hi - lo, 1.5, atol=0.01)
    _settle(scene)
    assert _body(scene, go).position[1] == pytest.approx(0.75, abs=0.02)


def test_component_radius_scaled_by_largest_object_scale() -> None:
    scene, go, _, _ = _scene_ball(0.5, scale=(1.0, 4.0, 2.0))
    lo, hi = _aabb(scene, go)
    assert np.allclose(hi - lo, 4.0, atol=0.01)  # 0.5 · max(1, 4, 2) · 2


def test_component_zero_radius_is_clamped_not_rejected() -> None:
    """Адаптер зажимает 0 до 1e-3 (в физике то же значение — ValueError)."""
    scene, go, _, rb = _scene_ball(0.0)
    assert rb.handle is not None
    lo, hi = _aabb(scene, go)
    assert np.allclose(hi - lo, 0.002, atol=1e-4)


def test_component_negative_radius_is_clamped_not_rejected() -> None:
    scene, go, _, rb = _scene_ball(-1.0)
    assert rb.handle is not None
    lo, hi = _aabb(scene, go)
    assert np.allclose(hi - lo, 0.002, atol=1e-4)


def test_component_runtime_radius_change() -> None:
    scene, go, col, _ = _scene_ball(0.25)
    _settle(scene, 2.0)
    assert _body(scene, go).position[1] == pytest.approx(0.25, abs=0.02)
    col.radius = 1.0
    _settle(scene, 3.0)
    assert _body(scene, go).position[1] == pytest.approx(1.0, abs=0.03)
    lo, hi = _aabb(scene, go)
    assert np.allclose(hi - lo, 2.0, atol=0.01)


def test_unused_fields_do_not_affect_sphere() -> None:
    """size и height у сферы не читаются."""
    scene = Scene()
    go = scene.create_game_object("Ball", position=Vector3(0, 3, 0))
    go.add_component(ColliderComponent, shape="sphere", radius=0.5,
                     size=(9.0, 9.0, 9.0), height=9.0)
    go.add_component(RigidBodyComponent)
    scene.fixed_update(DT)
    lo, hi = _aabb(scene, go)
    assert np.allclose(hi - lo, 1.0, atol=0.01)
