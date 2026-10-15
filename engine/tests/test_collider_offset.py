"""Верификация Collider: offset (смещение формы от начала тела/объекта).

Смещение задано в локальных осях тела, умножается на мировой масштаб объекта
(по осям) и поворачивается вместе с телом. Центр масс остаётся в начале тела.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.core.math3d import Quaternion, Vector3
from engine.core.physics_components import ColliderComponent, RigidBodyComponent
from engine.core.scene import Scene
from engine.physics.body import RigidBody
from engine.physics.collider import Collider
from engine.physics.events import TriggerEnter
from engine.physics.shape import BoxShape, SphereShape
from engine.physics.world import PhysicsWorld

DT = 1.0 / 60.0
ROT_Z_90 = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))  # X → Y


def _floor(world: PhysicsWorld) -> None:
    world.add_static(Collider(BoxShape((50.0, 0.5, 50.0))), (0.0, -0.5, 0.0))


def _center(world: PhysicsWorld, handle: int) -> np.ndarray:
    lo, hi = world.get_colliders(handle)[0].world_aabb
    return (lo + hi) / 2.0


# --- уровень PhysicsWorld -------------------------------------------------------

def test_offset_moves_collider_center_and_aabb() -> None:
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    h = world.add_body(
        RigidBody(position=(1.0, 2.0, 3.0)),
        Collider(SphereShape(0.5), offset=(0.5, -1.0, 2.0)))
    assert np.allclose(_center(world, h), (1.5, 1.0, 5.0))
    lo, hi = world.get_colliders(h)[0].world_aabb
    assert np.allclose(hi - lo, 1.0)  # размер не меняется


def test_offset_is_rotated_with_body() -> None:
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    h = world.add_body(
        RigidBody(position=(0.0, 0.0, 0.0), rotation=ROT_Z_90),
        Collider(SphereShape(0.25), offset=(1.0, 0.0, 0.0)))
    assert np.allclose(_center(world, h), (0.0, 1.0, 0.0), atol=1e-9)


def test_zero_offset_keeps_center_at_body_origin() -> None:
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    h = world.add_body(RigidBody(position=(1.0, 2.0, 3.0)),
                       Collider(SphereShape(0.5)))
    assert np.allclose(_center(world, h), (1.0, 2.0, 3.0))


def test_vertical_offset_shifts_rest_height_of_body_origin() -> None:
    """Куб 1×1×1 со смещением +0.5 вверх: начало тела опускается на пол."""
    world = PhysicsWorld()
    _floor(world)
    body = RigidBody(position=(0.0, 3.0, 0.0), freeze_rotation=True)
    world.add_body(body, Collider(BoxShape((0.5, 0.5, 0.5)),
                                  offset=(0.0, 0.5, 0.0)))
    for _ in range(240):
        world.step()
    assert body.position[1] == pytest.approx(0.0, abs=0.02)


def test_negative_vertical_offset_lifts_body_origin() -> None:
    world = PhysicsWorld()
    _floor(world)
    body = RigidBody(position=(0.0, 3.0, 0.0), freeze_rotation=True)
    world.add_body(body, Collider(BoxShape((0.5, 0.5, 0.5)),
                                  offset=(0.0, -1.0, 0.0)))
    for _ in range(240):
        world.step()
    assert body.position[1] == pytest.approx(1.5, abs=0.02)


def test_horizontal_offset_stops_wall_hit_at_shifted_distance() -> None:
    def closest(offset_x: float) -> float:
        world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
        world.add_static(Collider(BoxShape((0.5, 5.0, 5.0))), (0.0, 0.0, 0.0))
        body = RigidBody(position=(5.0, 0.0, 0.0), linear_velocity=(-2.0, 0.0, 0.0),
                         linear_damping=0.0, freeze_rotation=True)
        world.add_body(body, Collider(SphereShape(0.25), offset=(offset_x, 0.0, 0.0)))
        best = 99.0
        for _ in range(240):
            world.step()
            best = min(best, float(body.position[0]))
        return best

    # центр шара не может зайти за x = 0.5 + 0.25 → начало тела = 0.75 − offset
    assert closest(0.0) == pytest.approx(0.75, abs=0.1)
    assert closest(1.0) == pytest.approx(-0.25, abs=0.1)
    assert closest(-1.0) == pytest.approx(1.75, abs=0.1)


def test_offset_collider_swings_body_into_stable_pose() -> None:
    """Шар, смещённый на 1 м от центра масс, ложится как неваляшка: центр масс
    (начало тела) оказывается под шаром, а не рядом с ним."""
    world = PhysicsWorld()
    _floor(world)
    body = RigidBody(position=(0.0, 3.0, 0.0), can_sleep=False)
    h = world.add_body(body, Collider(SphereShape(0.25), offset=(1.0, 0.0, 0.0)))
    for _ in range(900):
        world.step()
    assert _center(world, h)[1] == pytest.approx(0.25, abs=0.03)
    assert body.position[1] < 0.0  # начало тела ниже центра шара (≈ 0.25 − 1)


def test_offset_increases_rotational_inertia_by_parallel_axis_theorem() -> None:
    def inertia(offset) -> np.ndarray:
        world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
        body = RigidBody(mass=2.0)
        world.add_body(body, Collider(SphereShape(0.5), offset=offset))
        return np.linalg.inv(body.inv_inertia_local)

    base = inertia((0.0, 0.0, 0.0))
    shifted = inertia((2.0, 0.0, 0.0))
    # вращение вокруг Y и Z (перпендикулярно смещению) растёт на m·d² = 8
    assert shifted[1, 1] - base[1, 1] == pytest.approx(2.0 * 4.0)
    assert shifted[2, 2] - base[2, 2] == pytest.approx(2.0 * 4.0)
    # вокруг оси смещения (X) не меняется
    assert shifted[0, 0] == pytest.approx(base[0, 0])


def test_offset_collider_gives_smaller_spin_from_same_impulse() -> None:
    def spin(offset) -> float:
        world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
        body = RigidBody(mass=1.0, linear_damping=0.0, angular_damping=0.0,
                         can_sleep=False)
        world.add_body(body, Collider(SphereShape(0.5), offset=offset))
        # угловой импульс (0, 1, 0): ω = I⁻¹ · L
        body.angular_velocity = body.inv_inertia_world @ np.array([0.0, 1.0, 0.0])
        return float(body.angular_velocity[1])

    assert spin((3.0, 0.0, 0.0)) < spin((0.0, 0.0, 0.0)) / 5.0


def test_two_offset_colliders_form_a_dumbbell() -> None:
    """Два шара на ±1 по X у одного тела: стоит на двух опорах без качения."""
    world = PhysicsWorld()
    _floor(world)
    body = RigidBody(position=(0.0, 2.0, 0.0), can_sleep=False)
    h = world.add_body(body, [
        Collider(SphereShape(0.25), offset=(-1.0, 0.0, 0.0)),
        Collider(SphereShape(0.25), offset=(1.0, 0.0, 0.0)),
    ])
    for _ in range(300):
        world.step()
    assert body.position[1] == pytest.approx(0.25, abs=0.03)
    # ось гантели осталась горизонтальной: оба шара на одной высоте
    heights = [float(c.world_position[1]) for c in world.get_colliders(h)]
    assert heights == pytest.approx([0.25, 0.25], abs=0.03)
    assert abs(body.position[0]) < 0.05


def test_offset_is_ignored_for_collision_filtering_between_colliders_of_one_body() -> None:
    """Коллайдеры одного тела не сталкиваются, даже если смещения их пересекают."""
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    body = RigidBody(position=(0.0, 0.0, 0.0), can_sleep=False)
    world.add_body(body, [
        Collider(SphereShape(0.5), offset=(-0.1, 0.0, 0.0)),
        Collider(SphereShape(0.5), offset=(0.1, 0.0, 0.0)),
    ])
    for _ in range(60):
        world.step()
    assert np.allclose(body.position, 0.0, atol=1e-9)


# --- компонент -------------------------------------------------------------------

def _scene_obj(offset, scale=(1.0, 1.0, 1.0), rotation=None, shape="sphere",
               static=True, **kwargs):
    scene = Scene()
    extra = {} if rotation is None else {"rotation": rotation}
    go = scene.create_game_object("Obj", position=Vector3(0, 5, 0),
                                  scale=Vector3(*scale), **extra)
    col = go.add_component(ColliderComponent, shape=shape, radius=0.5,
                           size=(1, 1, 1), offset=offset, **kwargs)
    rb = None
    if not static:
        rb = go.add_component(RigidBodyComponent, use_gravity=False)
    scene.fixed_update(DT)
    return scene, go, col, rb


def _aabb_center(scene, go) -> np.ndarray:
    lo, hi = scene.physics.world.get_colliders(
        scene.physics.handle_of(go))[0].world_aabb
    return (lo + hi) / 2.0


def test_component_offset_moves_center() -> None:
    scene, go, _, _ = _scene_obj((1.0, 2.0, -3.0))
    assert np.allclose(_aabb_center(scene, go), (1.0, 7.0, -3.0), atol=1e-6)


def test_component_default_offset_is_zero() -> None:
    assert tuple(ColliderComponent().offset) == (0.0, 0.0, 0.0)


def test_component_offset_is_scaled_by_object_scale_per_axis() -> None:
    scene, go, _, _ = _scene_obj((1.0, 1.0, 1.0), scale=(2.0, 3.0, 4.0))
    assert np.allclose(_aabb_center(scene, go), (2.0, 5.0 + 3.0, 4.0), atol=1e-6)


def test_component_offset_with_negative_scale_uses_absolute_scale() -> None:
    """Мировой масштаб берётся нормой столбцов матрицы → знак теряется."""
    scene, go, _, _ = _scene_obj((1.0, 0.0, 0.0), scale=(-2.0, 1.0, 1.0))
    assert _aabb_center(scene, go)[0] == pytest.approx(2.0, abs=1e-6)


def test_component_offset_is_rotated_with_object() -> None:
    rot = Quaternion(*ROT_Z_90)
    scene, go, _, _ = _scene_obj((1.0, 0.0, 0.0), rotation=rot)
    assert np.allclose(_aabb_center(scene, go), (0.0, 6.0, 0.0), atol=1e-6)


def test_component_offset_with_rigidbody_moves_collider_with_body() -> None:
    scene, go, _, _ = _scene_obj((1.0, 0.0, 0.0), static=False)
    handle = scene.physics.handle_of(go)
    body = scene.physics.world.get_body(handle)
    body.position[0] = 10.0
    scene.physics.world.step(DT)
    assert _aabb_center(scene, go)[0] == pytest.approx(11.0, abs=1e-3)


def test_component_runtime_offset_change_moves_collider() -> None:
    scene, go, col, _ = _scene_obj((0.0, 0.0, 0.0))
    assert _aabb_center(scene, go)[0] == pytest.approx(0.0, abs=1e-6)
    col.offset = (4.0, 0.0, 0.0)
    scene.fixed_update(DT)
    assert _aabb_center(scene, go)[0] == pytest.approx(4.0, abs=1e-6)


def test_component_offset_fourth_value_is_ignored() -> None:
    scene, go, _, _ = _scene_obj((1.0, 0.0, 0.0, 99.0))
    assert np.allclose(_aabb_center(scene, go), (1.0, 5.0, 0.0), atol=1e-6)


def test_component_offset_does_not_change_object_transform() -> None:
    scene, go, _, _ = _scene_obj((3.0, 0.0, 0.0), static=False)
    for _ in range(30):
        scene.fixed_update(DT)
    assert go.transform.world_position.x == pytest.approx(0.0, abs=1e-6)


def test_component_static_offset_moves_surface() -> None:
    """Пол с offset (0,-1,0): верхняя грань опускается с y=0 до y=-1."""
    scene = Scene()
    floor = scene.create_game_object("Floor", position=Vector3(0, -0.5, 0),
                                     scale=Vector3(100, 1, 100))
    floor.add_component(ColliderComponent, shape="box", size=(1, 1, 1),
                        offset=(0.0, -1.0 / 1.0, 0.0))
    ball = scene.create_game_object("Ball", position=Vector3(0, 3, 0))
    ball.add_component(ColliderComponent, shape="sphere", radius=0.25)
    ball.add_component(RigidBodyComponent, can_sleep=False)
    for _ in range(300):
        scene.fixed_update(DT)
    body = scene.physics.world.get_body(scene.physics.handle_of(ball))
    # offset (0,-1,0) · scale y=1 → пол сдвинут вниз на 1: верх на y = −1
    assert body.position[1] == pytest.approx(-1.0 + 0.25, abs=0.03)


def test_component_offset_dynamic_rests_with_origin_above_collider() -> None:
    scene = Scene()
    scene.create_game_object("Floor", position=Vector3(0, -0.5, 0),
                             scale=Vector3(100, 1, 100)).add_component(
        ColliderComponent, shape="box", size=(1, 1, 1))
    go = scene.create_game_object("Box", position=Vector3(0, 3, 0))
    go.add_component(ColliderComponent, shape="box", size=(1, 1, 1),
                     offset=(0.0, -1.0, 0.0))
    go.add_component(RigidBodyComponent, freeze_rotation=True)
    for _ in range(240):
        scene.fixed_update(DT)
    body = scene.physics.world.get_body(scene.physics.handle_of(go))
    assert body.position[1] == pytest.approx(1.5, abs=0.03)  # куб на 0.5, начало +1


def test_component_two_colliders_have_independent_offsets() -> None:
    scene = Scene()
    go = scene.create_game_object("Dumbbell", position=Vector3(0, 5, 0))
    go.add_component(ColliderComponent, shape="sphere", radius=0.25,
                     offset=(-2.0, 0.0, 0.0))
    go.add_component(ColliderComponent, shape="sphere", radius=0.25,
                     offset=(2.0, 0.0, 0.0))
    go.add_component(RigidBodyComponent, use_gravity=False)
    scene.fixed_update(DT)
    cols = scene.physics.world.get_colliders(scene.physics.handle_of(go))
    centers = sorted(float((c.world_aabb[0][0] + c.world_aabb[1][0]) / 2)
                     for c in cols)
    assert centers == pytest.approx([-2.0, 2.0], abs=1e-6)


def test_component_offset_trigger_detects_at_shifted_place() -> None:
    scene = Scene()
    zone = scene.create_game_object("Zone", position=Vector3(0, 1, 0))
    zone.add_component(ColliderComponent, shape="box", size=(2, 2, 2),
                       is_trigger=True, offset=(10.0, 0.0, 0.0))
    near = scene.create_game_object("Near", position=Vector3(0, 6, 0))
    near.add_component(ColliderComponent, shape="sphere", radius=0.25)
    near.add_component(RigidBodyComponent)
    far = scene.create_game_object("Far", position=Vector3(10, 6, 0))
    far.add_component(ColliderComponent, shape="sphere", radius=0.25)
    far.add_component(RigidBodyComponent)
    seen: list = []
    scene.events.subscribe(TriggerEnter, seen.append)
    for _ in range(120):
        scene.fixed_update(DT)
        scene.update(DT)
    assert [e.user_b for e in seen] == [far]
