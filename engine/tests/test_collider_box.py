"""Верификация Collider: форма Box (half_extents / size)."""

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
    """Статический пол: верхняя грань на y = 0."""
    world.add_static(Collider(BoxShape((50.0, 0.5, 50.0))), (0.0, -0.5, 0.0))


def _drop(half, y: float = 3.0, seconds: float = 3.0):
    world = PhysicsWorld()
    _floor(world)
    body = RigidBody(position=(0.0, y, 0.0))
    h = world.add_body(body, Collider(BoxShape(half)))
    for _ in range(round(seconds / DT)):
        world.step()
    return world, h, body


# --- физика: half_extents ----------------------------------------------------

@pytest.mark.parametrize("half", [(0.5, 0.5, 0.5), (0.25, 1.0, 2.0), (3.0, 0.1, 1.0)])
def test_aabb_equals_position_plus_minus_half_extents(half) -> None:
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    h = world.add_body(RigidBody(position=(1.0, 2.0, 3.0)), Collider(BoxShape(half)))
    lo, hi = world.get_colliders(h)[0].world_aabb
    assert np.allclose(lo, np.array([1.0, 2.0, 3.0]) - half)
    assert np.allclose(hi, np.array([1.0, 2.0, 3.0]) + half)


@pytest.mark.parametrize("hy", [0.1, 0.5, 1.0, 2.5])
def test_stop_height_equals_half_extent_y(hy: float) -> None:
    _, _, body = _drop((0.5, hy, 0.5))
    assert body.position[1] == pytest.approx(hy, abs=0.02)
    assert not body.linear_velocity.any() or abs(body.linear_velocity[1]) < 0.05


def test_taller_box_rests_higher_than_shorter() -> None:
    _, _, low = _drop((0.5, 0.25, 0.5))
    _, _, high = _drop((0.5, 1.5, 0.5))
    assert high.position[1] - low.position[1] == pytest.approx(1.25, abs=0.03)


@pytest.mark.parametrize("hy", [0.25, 1.0])
def test_penetration_depth_after_rest_is_small(hy: float) -> None:
    """Нижняя грань остаётся в пределах 2 см от поверхности пола."""
    _, _, body = _drop((0.5, hy, 0.5))
    bottom = body.position[1] - hy
    assert -0.02 <= bottom <= 0.02


@pytest.mark.parametrize("axis", [0, 2])
def test_half_extent_on_horizontal_axis_sets_contact_distance(axis: int) -> None:
    """Блок врезается в статику: центр останавливается на hA + hB по оси."""
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    ha = (0.5, 0.5, 0.5)
    hb = (0.25, 0.5, 0.25)
    hb = tuple(1.0 if i == axis else v for i, v in enumerate(hb))
    world.add_static(Collider(BoxShape(ha)), (0.0, 0.0, 0.0))
    start = [0.0, 0.0, 0.0]
    start[axis] = 5.0
    vel = [0.0, 0.0, 0.0]
    vel[axis] = -2.0
    body = RigidBody(position=start, linear_velocity=vel, linear_damping=0.0)
    world.add_body(body, Collider(BoxShape(hb)))
    closest = 99.0
    for _ in range(240):
        world.step()
        closest = min(closest, float(body.position[axis]))
    # после удара тело отлетает на ≈18 % (см. docs/CORE.md, B1): меряем сближение
    assert closest == pytest.approx(ha[axis] + hb[axis], abs=0.1)


def test_zero_half_extent_rejected_by_shape() -> None:
    with pytest.raises(ValueError):
        BoxShape((0.5, 0.0, 0.5))


def test_negative_half_extent_rejected_by_shape() -> None:
    with pytest.raises(ValueError):
        BoxShape((-0.5, 0.5, 0.5))


def test_nan_half_extent_rejected_by_shape() -> None:
    with pytest.raises(ValueError):
        BoxShape((float("nan"), 0.5, 0.5))


def test_huge_static_box_holds_small_ball() -> None:
    world = PhysicsWorld()
    world.add_static(Collider(BoxShape((500.0, 500.0, 500.0))), (0.0, -500.0, 0.0))
    ball = RigidBody(position=(10.0, 4.0, -20.0))
    world.add_body(ball, Collider(SphereShape(0.25)))
    for _ in range(240):
        world.step()
    assert ball.position[1] == pytest.approx(0.25, abs=0.02)
    assert np.all(np.isfinite(ball.position))


def test_huge_dynamic_box_aabb_finite_and_ball_stops_on_it() -> None:
    world = PhysicsWorld()
    big = RigidBody(position=(0.0, -50.0, 0.0), body_type="static")
    world.add_body(big, Collider(BoxShape((1000.0, 50.0, 1000.0))))
    ball = RigidBody(position=(0.0, 2.0, 0.0))
    world.add_body(ball, Collider(SphereShape(0.1)))
    for _ in range(180):
        world.step()
    assert ball.position[1] == pytest.approx(0.1, abs=0.02)


# --- компонент: size → half_extents -----------------------------------------

def _scene_box(size, scale=(1.0, 1.0, 1.0), y: float = 3.0, dynamic: bool = True):
    scene = Scene()
    floor = scene.create_game_object("Floor", position=Vector3(0, -0.5, 0),
                                     scale=Vector3(100, 1, 100))
    floor.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    go = scene.create_game_object("Box", position=Vector3(0, y, 0),
                                  scale=Vector3(*scale))
    col = go.add_component(ColliderComponent, shape="box", size=size)
    rb = go.add_component(RigidBodyComponent) if dynamic else None
    scene.fixed_update(DT)
    return scene, go, col, rb


def _body(scene, go) -> RigidBody:
    return scene.physics.world.get_body(scene.physics.handle_of(go))


def _aabb(scene, go):
    handle = scene.physics.handle_of(go)
    return scene.physics.world.get_colliders(handle)[0].world_aabb


def _settle(scene, seconds: float = 3.0) -> None:
    for _ in range(round(seconds / DT)):
        scene.fixed_update(DT)


def test_component_size_is_full_dimensions() -> None:
    scene, go, _, _ = _scene_box((2.0, 4.0, 6.0))
    lo, hi = _aabb(scene, go)
    assert np.allclose(hi - lo, (2.0, 4.0, 6.0), atol=1e-6)


def test_component_size_scaled_by_object_scale() -> None:
    scene, go, _, _ = _scene_box((1.0, 1.0, 1.0), scale=(2.0, 3.0, 0.5))
    lo, hi = _aabb(scene, go)
    assert np.allclose(hi - lo, (2.0, 3.0, 0.5), atol=1e-6)


def test_component_rest_height_is_half_of_size_y() -> None:
    scene, go, _, _ = _scene_box((1.0, 3.0, 1.0))
    _settle(scene)
    assert _body(scene, go).position[1] == pytest.approx(1.5, abs=0.02)


def test_component_zero_size_axis_is_clamped_not_rejected() -> None:
    """Адаптер зажимает 0 до 1e-3 (в физике то же значение — ValueError)."""
    scene, go, _, rb = _scene_box((1.0, 0.0, 1.0))
    assert rb.handle is not None
    lo, hi = _aabb(scene, go)
    assert (hi - lo)[1] == pytest.approx(0.002, abs=1e-9)
    _settle(scene, 2.0)
    body = _body(scene, go)
    assert np.all(np.isfinite(body.position))
    assert -0.05 < body.position[1] < 0.05  # плоская «плита» лежит на полу


def test_component_negative_size_is_taken_by_absolute_value() -> None:
    """Отрицательный размер — по модулю, как и отрицательный масштаб."""
    scene, go, _, rb = _scene_box((-1.0, 1.0, 1.0))
    assert rb.handle is not None
    lo, hi = _aabb(scene, go)
    assert np.allclose(hi - lo, (1.0, 1.0, 1.0), atol=1e-6)
    scene, go, _, _ = _scene_box((-2.0, 1.0, 1.0))
    lo, hi = _aabb(scene, go)
    assert np.allclose(hi - lo, (2.0, 1.0, 1.0), atol=1e-6)


def test_component_negative_size_collides_like_positive() -> None:
    def rest_height(size) -> float:
        scene, go, _, _ = _scene_box(size)
        _settle(scene, 3.0)
        return float(_body(scene, go).position[1])

    assert rest_height((-1.0, 1.0, 1.0)) == rest_height((1.0, 1.0, 1.0))
    assert rest_height((1.0, -2.0, 1.0)) == pytest.approx(1.0, abs=0.02)


def test_component_runtime_size_change_rebuilds_shape() -> None:
    scene, go, col, _ = _scene_box((1.0, 1.0, 1.0))
    _settle(scene, 2.0)
    assert _body(scene, go).position[1] == pytest.approx(0.5, abs=0.02)
    col.size = (1.0, 3.0, 1.0)
    scene.fixed_update(DT)
    lo, hi = _aabb(scene, go)
    assert (hi - lo)[1] == pytest.approx(3.0, abs=0.01)  # допуск на микроповорот
    _settle(scene, 3.0)
    assert _body(scene, go).position[1] == pytest.approx(1.5, abs=0.03)


def test_component_runtime_size_change_keeps_velocity() -> None:
    scene, go, col, _ = _scene_box((1.0, 1.0, 1.0), y=10.0)
    for _ in range(10):
        scene.fixed_update(DT)
    v_before = float(_body(scene, go).linear_velocity[1])
    col.size = (2.0, 1.0, 2.0)
    scene.fixed_update(DT)
    v_after = float(_body(scene, go).linear_velocity[1])
    assert v_before < -1.0
    assert v_after < v_before  # скорость сохранена и продолжает расти


def test_size_beyond_three_values_is_ignored() -> None:
    scene, go, _, rb = _scene_box((1.0, 2.0, 3.0, 99.0))
    assert rb.handle is not None
    lo, hi = _aabb(scene, go)
    assert np.allclose(hi - lo, (1.0, 2.0, 3.0), atol=1e-6)
