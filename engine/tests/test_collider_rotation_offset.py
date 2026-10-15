"""Верификация Collider: rotation_offset (поворот формы относительно тела).

Тесты проверяют физику на уровне ``engine.physics.Collider`` и поле
``ColliderComponent.rotation_offset`` (адаптер передаёт его в коллайдер).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.core.physics_components import ColliderComponent
from engine.physics.body import RigidBody
from engine.physics.collider import Collider
from engine.physics.shape import BoxShape, CapsuleShape, SphereShape
from engine.physics.world import PhysicsWorld

DT = 1.0 / 60.0
S = math.sin(math.pi / 4)
ROT_Z_90 = (0.0, 0.0, S, S)  # X → Y
ROT_X_90 = (S, 0.0, 0.0, S)  # Y → Z
ROT_Y_90 = (0.0, S, 0.0, S)  # Z → X
IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _floor(world: PhysicsWorld) -> None:
    world.add_static(Collider(BoxShape((50.0, 0.5, 50.0))), (0.0, -0.5, 0.0))


def _extent(world: PhysicsWorld, handle: int) -> np.ndarray:
    lo, hi = world.get_colliders(handle)[0].world_aabb
    return hi - lo


def _body_with(collider: Collider, **body_kwargs) -> tuple[PhysicsWorld, RigidBody, int]:
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    body = RigidBody(**body_kwargs)
    return world, body, world.add_body(body, collider)


# --- форма и AABB ----------------------------------------------------------------

def test_identity_rotation_offset_keeps_shape_axes() -> None:
    world, _, h = _body_with(Collider(BoxShape((2.0, 0.5, 0.25)),
                                      rotation_offset=IDENTITY))
    assert np.allclose(_extent(world, h), (4.0, 1.0, 0.5))


def test_rotation_offset_z90_swaps_x_and_y_extents() -> None:
    world, _, h = _body_with(Collider(BoxShape((2.0, 0.5, 0.25)),
                                      rotation_offset=ROT_Z_90))
    assert np.allclose(_extent(world, h), (1.0, 4.0, 0.5), atol=1e-9)


def test_rotation_offset_x90_swaps_y_and_z_extents() -> None:
    world, _, h = _body_with(Collider(BoxShape((2.0, 0.5, 0.25)),
                                      rotation_offset=ROT_X_90))
    assert np.allclose(_extent(world, h), (4.0, 0.5, 1.0), atol=1e-9)


def test_rotation_offset_y90_swaps_x_and_z_extents() -> None:
    world, _, h = _body_with(Collider(BoxShape((2.0, 0.5, 0.25)),
                                      rotation_offset=ROT_Y_90))
    assert np.allclose(_extent(world, h), (0.5, 1.0, 4.0), atol=1e-9)


def test_rotation_offset_does_not_move_collider_center() -> None:
    world, _, h = _body_with(Collider(BoxShape((2.0, 0.5, 0.25)),
                                      rotation_offset=ROT_Z_90, offset=(0.0, 3.0, 0.0)),
                             position=(1.0, 0.0, 0.0))
    lo, hi = world.get_colliders(h)[0].world_aabb
    assert np.allclose((lo + hi) / 2, (1.0, 3.0, 0.0), atol=1e-9)


def test_rotation_offset_does_not_rotate_the_offset_vector() -> None:
    """offset задан в осях тела; rotation_offset поворачивает только форму."""
    world, _, h = _body_with(Collider(BoxShape((0.5, 0.5, 0.5)),
                                      rotation_offset=ROT_Z_90, offset=(2.0, 0.0, 0.0)))
    lo, hi = world.get_colliders(h)[0].world_aabb
    assert np.allclose((lo + hi) / 2, (2.0, 0.0, 0.0), atol=1e-9)


def test_rotation_offset_composes_with_body_rotation() -> None:
    """Тело повёрнуто на 90° вокруг Z, форма ещё на 90° → 180°: оси как у identity."""
    world, _, h = _body_with(Collider(BoxShape((2.0, 0.5, 0.25)),
                                      rotation_offset=ROT_Z_90), rotation=ROT_Z_90)
    assert np.allclose(_extent(world, h), (4.0, 1.0, 0.5), atol=1e-9)


def test_rotation_offset_of_sphere_changes_nothing() -> None:
    world, _, h = _body_with(Collider(SphereShape(0.5), rotation_offset=ROT_X_90))
    assert np.allclose(_extent(world, h), 1.0, atol=1e-9)


# --- динамика ---------------------------------------------------------------------

def test_rotated_box_rests_on_its_new_lower_face() -> None:
    """Плита 4×1×1 (half 2,.5,.5), повёрнутая на 90° вокруг Z, стоит на торце."""
    world = PhysicsWorld()
    _floor(world)
    body = RigidBody(position=(0.0, 4.0, 0.0), freeze_rotation=True)
    world.add_body(body, Collider(BoxShape((2.0, 0.5, 0.5)),
                                  rotation_offset=ROT_Z_90))
    for _ in range(240):
        world.step()
    assert body.position[1] == pytest.approx(2.0, abs=0.03)


def test_unrotated_box_rests_on_the_long_side_for_contrast() -> None:
    world = PhysicsWorld()
    _floor(world)
    body = RigidBody(position=(0.0, 4.0, 0.0), freeze_rotation=True)
    world.add_body(body, Collider(BoxShape((2.0, 0.5, 0.5))))
    for _ in range(240):
        world.step()
    assert body.position[1] == pytest.approx(0.5, abs=0.03)


def test_capsule_axis_follows_rotation_offset() -> None:
    """Вертикальная по умолчанию капсула с rotation_offset Z90 лежит на боку."""
    world = PhysicsWorld()
    _floor(world)
    body = RigidBody(position=(0.0, 3.0, 0.0), freeze_rotation=True)
    world.add_body(body, Collider(CapsuleShape(0.5, 1.0), rotation_offset=ROT_Z_90))
    for _ in range(240):
        world.step()
    assert body.position[1] == pytest.approx(0.5, abs=0.03)


def test_rotation_offset_rotates_inertia_tensor() -> None:
    def inertia(rotation_offset) -> np.ndarray:
        world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
        body = RigidBody(mass=3.0)
        world.add_body(body, Collider(BoxShape((2.0, 0.5, 0.25)),
                                      rotation_offset=rotation_offset))
        return np.linalg.inv(body.inv_inertia_local)

    base = np.diag(inertia(IDENTITY))
    rotated = np.diag(inertia(ROT_Z_90))
    assert rotated[0] == pytest.approx(base[1], rel=1e-6)
    assert rotated[1] == pytest.approx(base[0], rel=1e-6)
    assert rotated[2] == pytest.approx(base[2], rel=1e-6)


def test_rotated_box_blocks_wall_at_rotated_distance() -> None:
    def closest(rotation_offset) -> float:
        world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
        world.add_static(Collider(BoxShape((0.5, 5.0, 5.0))), (0.0, 0.0, 0.0))
        body = RigidBody(position=(6.0, 0.0, 0.0), linear_velocity=(-2.0, 0.0, 0.0),
                         linear_damping=0.0, freeze_rotation=True)
        world.add_body(body, Collider(BoxShape((2.0, 0.5, 0.5)),
                                      rotation_offset=rotation_offset))
        best = 99.0
        for _ in range(240):
            world.step()
            best = min(best, float(body.position[0]))
        return best

    assert closest(IDENTITY) == pytest.approx(0.5 + 2.0, abs=0.1)
    assert closest(ROT_Z_90) == pytest.approx(0.5 + 0.5, abs=0.1)


# --- поле компонента ----------------------------------------------------------------

def _component_extent(rotation_offset, size=(1.0, 2.0, 1.0)) -> np.ndarray:
    from engine.core.math3d import Vector3
    from engine.core.scene import Scene

    scene = Scene()
    go = scene.create_game_object("Obj", position=Vector3(0, 5, 0))
    go.add_component(ColliderComponent, shape="box", size=size,
                     rotation_offset=rotation_offset)
    scene.fixed_update(DT)
    handle = scene.physics.handle_of(go)
    lo, hi = scene.physics.world.get_colliders(handle)[0].world_aabb
    return hi - lo


def test_component_rotation_offset_defaults_to_identity() -> None:
    comp = ColliderComponent()
    assert tuple(comp.rotation_offset) == IDENTITY
    assert "rotation_offset" not in comp.to_dict()


def test_component_rotation_offset_z90_turns_box_aabb() -> None:
    """Куб (1, 2, 1), 90° вокруг Z: AABB (1, 2, 1) -> (2, 1, 1)."""
    assert np.allclose(_component_extent(IDENTITY), (1.0, 2.0, 1.0), atol=1e-9)
    assert np.allclose(_component_extent(ROT_Z_90), (2.0, 1.0, 1.0), atol=1e-9)


def test_component_rotation_offset_y90_swaps_x_and_z() -> None:
    """90° вокруг Y меняет местами X и Z (для куба (1, 2, 1) размеры те же)."""
    assert np.allclose(_component_extent(ROT_Y_90, size=(1.0, 2.0, 3.0)),
                       (3.0, 2.0, 1.0), atol=1e-9)
    assert np.allclose(_component_extent(ROT_Y_90), (1.0, 2.0, 1.0), atol=1e-9)


def test_component_rotation_offset_is_normalized_and_zero_is_identity() -> None:
    scaled = tuple(2.0 * v for v in ROT_Z_90)
    assert np.allclose(_component_extent(scaled), (2.0, 1.0, 1.0), atol=1e-9)
    assert np.allclose(_component_extent((0.0, 0.0, 0.0, 0.0)), (1.0, 2.0, 1.0),
                       atol=1e-9)


def test_component_rotation_offset_change_rebuilds_body() -> None:
    from engine.core.math3d import Vector3
    from engine.core.scene import Scene

    scene = Scene()
    go = scene.create_game_object("Obj", position=Vector3(0, 5, 0))
    col = go.add_component(ColliderComponent, shape="box", size=(1.0, 2.0, 1.0))
    scene.fixed_update(DT)
    col.rotation_offset = ROT_Z_90
    scene.fixed_update(DT)
    lo, hi = scene.physics.world.get_colliders(
        scene.physics.handle_of(go))[0].world_aabb
    assert np.allclose(hi - lo, (2.0, 1.0, 1.0), atol=1e-9)
