"""Геометрия отладочной визуализации физики (без GL)."""

from __future__ import annotations

import numpy as np
import pytest

from engine.core.math3d import Vector3
from engine.core.physics_components import (
    CharacterControllerComponent,
    ColliderComponent,
    RigidBodyComponent,
)
from engine.core.scene import Scene
from engine.rendering.physics_debug import COLORS, build_debug_lines


def _scene() -> Scene:
    scene = Scene()
    floor = scene.create_game_object("Floor", position=Vector3(0, -0.5, 0))
    floor.add_component(ColliderComponent, shape="box", size=(10, 1, 10))
    box = scene.create_game_object("Box", position=Vector3(0, 3, 0))
    box.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    box.add_component(RigidBodyComponent)
    ball = scene.create_game_object("Ball", position=Vector3(3, 3, 0))
    ball.add_component(ColliderComponent, shape="sphere", radius=0.5)
    ball.add_component(RigidBodyComponent, body_type="kinematic")
    cap = scene.create_game_object("Cap", position=Vector3(-3, 3, 0))
    cap.add_component(ColliderComponent, shape="capsule", radius=0.4, height=1.8)
    cap.add_component(RigidBodyComponent)
    zone = scene.create_game_object("Zone", position=Vector3(0, 1, 5))
    zone.add_component(ColliderComponent, shape="box", is_trigger=True)
    hero = scene.create_game_object("Hero", position=Vector3(5, 1, 5))
    hero.add_component(CharacterControllerComponent)
    scene.physics.sync_structure()
    return scene


def _color_set(colors: np.ndarray) -> set[tuple[float, ...]]:
    return {tuple(np.round(c, 3)) for c in colors}


def test_lines_pairs_and_all_kinds_colored():
    scene = _scene()
    positions, colors = build_debug_lines(
        scene.physics.world, scene.physics.character_handles()
    )
    assert positions.dtype == np.float32
    assert positions.shape == colors.shape
    assert positions.shape[0] % 2 == 0
    expected = {tuple(np.round(np.asarray(c, dtype=np.float32), 3)) for c in COLORS.values()}
    assert _color_set(colors) == expected


def test_box_has_twelve_edges_at_world_corners():
    scene = Scene()
    box = scene.create_game_object("Box", position=Vector3(1, 2, 3))
    box.add_component(ColliderComponent, shape="box", size=(2, 4, 6))
    scene.physics.sync_structure()
    positions, _ = build_debug_lines(scene.physics.world)
    assert positions.shape == (24, 3)
    assert positions.min(axis=0) == pytest.approx((0, 0, 0))
    assert positions.max(axis=0) == pytest.approx((2, 4, 6))


def test_sphere_lines_stay_on_radius():
    scene = Scene()
    ball = scene.create_game_object("Ball", position=Vector3(0, 1, 0))
    ball.add_component(ColliderComponent, shape="sphere", radius=0.5)
    scene.physics.sync_structure()
    positions, _ = build_debug_lines(scene.physics.world)
    radii = np.linalg.norm(positions - np.array([0, 1, 0]), axis=1)
    assert radii == pytest.approx(0.5, abs=1e-5)


def test_capsule_extent_matches_height():
    scene = Scene()
    cap = scene.create_game_object("Cap", position=Vector3(0, 0, 0))
    cap.add_component(ColliderComponent, shape="capsule", radius=0.4, height=1.8)
    scene.physics.sync_structure()
    positions, _ = build_debug_lines(scene.physics.world)
    assert positions[:, 1].max() == pytest.approx(0.9, abs=1e-5)
    assert positions[:, 1].min() == pytest.approx(-0.9, abs=1e-5)
    assert np.abs(positions[:, 0]).max() == pytest.approx(0.4, abs=1e-5)


def test_empty_world_gives_empty_arrays():
    scene = Scene()
    positions, colors = build_debug_lines(scene.physics.world)
    assert positions.shape == (0, 3) and colors.shape == (0, 3)


def test_sleeping_body_is_dimmed():
    scene = Scene()
    floor = scene.create_game_object("Floor", position=Vector3(0, -0.5, 0))
    floor.add_component(ColliderComponent, shape="box", size=(10, 1, 10))
    box = scene.create_game_object("Box", position=Vector3(0, 0.5, 0))
    box.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    box.add_component(RigidBodyComponent)
    for _ in range(200):
        scene.fixed_update(1 / 60)
    _, colors = build_debug_lines(scene.physics.world)
    dyn = np.asarray(COLORS["dynamic"], dtype=np.float32)
    assert any(np.allclose(c, dyn * 0.45, atol=1e-3) for c in colors)


# --- верификация Collider: каркас отражает параметры компонента ---------------------

def _lines(scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    scene.physics.sync_structure()
    return build_debug_lines(scene.physics.world)


def _single(**collider) -> Scene:
    scene = Scene()
    go = scene.create_game_object("Obj", position=Vector3(0, 2, 0))
    go.add_component(ColliderComponent, **collider)
    return scene


def _extent(positions: np.ndarray) -> np.ndarray:
    return positions.max(axis=0) - positions.min(axis=0)


def _rgb(name: str) -> np.ndarray:
    return np.asarray(COLORS[name], dtype=np.float32)


@pytest.mark.parametrize(("size", "scale"), [
    ((1, 1, 1), (1, 1, 1)),
    ((2, 3, 4), (1, 1, 1)),
    ((1, 1, 1), (2, 3, 4)),
    ((2, 2, 2), (0.5, 1, 3)),
])
def test_box_wireframe_extent_is_size_times_scale(size, scale) -> None:
    scene = Scene()
    go = scene.create_game_object("Obj", scale=Vector3(*scale))
    go.add_component(ColliderComponent, shape="box", size=size)
    positions, _ = _lines(scene)
    assert _extent(positions) == pytest.approx(np.array(size) * np.array(scale))


@pytest.mark.parametrize("radius", [0.1, 0.5, 2.0])
def test_sphere_wireframe_follows_radius(radius: float) -> None:
    positions, _ = _lines(_single(shape="sphere", radius=radius))
    center = np.array([0, 2, 0])
    assert np.linalg.norm(positions - center, axis=1) == pytest.approx(
        radius, abs=1e-5)


@pytest.mark.parametrize(("radius", "height"), [(0.3, 1.0), (0.5, 2.0), (0.5, 4.0)])
def test_capsule_wireframe_follows_radius_and_height(radius, height) -> None:
    positions, _ = _lines(_single(shape="capsule", radius=radius, height=height))
    ext = _extent(positions)
    assert ext[1] == pytest.approx(height, abs=1e-5)
    assert ext[0] == pytest.approx(2 * radius, abs=1e-5)
    assert ext[2] == pytest.approx(2 * radius, abs=1e-5)


def test_wireframe_matches_collider_aabb_for_every_shape() -> None:
    for kwargs in (
        {"shape": "box", "size": (1, 2, 3)},
        {"shape": "sphere", "radius": 0.7},
        {"shape": "capsule", "radius": 0.4, "height": 2.0},
    ):
        scene = _single(**kwargs)
        positions, _ = _lines(scene)
        world = scene.physics.world
        lo, hi = next(c for _, _, c in world.iter_colliders()).world_aabb
        assert positions.min(axis=0) == pytest.approx(lo, abs=1e-4)
        assert positions.max(axis=0) == pytest.approx(hi, abs=1e-4)


def test_wireframe_follows_offset() -> None:
    positions, _ = _lines(_single(shape="sphere", radius=0.5, offset=(1.5, 0.25, -2)))
    center = (positions.min(axis=0) + positions.max(axis=0)) / 2
    assert center == pytest.approx((1.5, 2.25, -2.0), abs=1e-5)


def test_wireframe_offset_is_scaled_with_object() -> None:
    scene = Scene()
    go = scene.create_game_object("Obj", scale=Vector3(2, 2, 2))
    go.add_component(ColliderComponent, shape="sphere", radius=0.5, offset=(1, 0, 0))
    positions, _ = _lines(scene)
    center = (positions.min(axis=0) + positions.max(axis=0)) / 2
    assert center == pytest.approx((2.0, 0.0, 0.0), abs=1e-5)


def test_wireframe_follows_rotation_offset_at_world_level() -> None:
    from engine.physics.collider import Collider
    from engine.physics.shape import BoxShape
    from engine.physics.world import PhysicsWorld

    s = np.sin(np.pi / 4)
    world = PhysicsWorld()
    world.add_static(Collider(BoxShape((2.0, 0.5, 0.25)), rotation_offset=(0, 0, s, s)))
    positions, _ = build_debug_lines(world)
    assert _extent(positions) == pytest.approx((1.0, 4.0, 0.5), abs=1e-5)


def test_wireframe_updates_after_live_edit() -> None:
    scene = _single(shape="sphere", radius=0.5)
    go = scene.find("Obj")
    comp = go.get_component(ColliderComponent)
    before, _ = _lines(scene)
    comp.radius = 1.5
    after, _ = _lines(scene)
    assert _extent(before)[0] == pytest.approx(1.0, abs=1e-5)
    assert _extent(after)[0] == pytest.approx(3.0, abs=1e-5)


def test_each_collider_of_one_body_gets_its_own_wireframe() -> None:
    scene = Scene()
    go = scene.create_game_object("Obj")
    go.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    go.add_component(ColliderComponent, shape="sphere", radius=0.5, offset=(3, 0, 0))
    go.add_component(RigidBodyComponent)
    positions, _ = _lines(scene)
    assert positions.shape[0] == 24 + 3 * 24 * 2  # 12 рёбер + 3 окружности по 24


# --- цвета ------------------------------------------------------------------------

@pytest.mark.parametrize("shape", ["box", "sphere", "capsule"])
@pytest.mark.parametrize(("body", "color"), [
    (None, "static"), ("dynamic", "dynamic"), ("kinematic", "kinematic"),
])
def test_color_by_body_type_for_each_shape(shape, body, color) -> None:
    scene = Scene()
    go = scene.create_game_object("Obj", position=Vector3(0, 50, 0))
    go.add_component(ColliderComponent, shape=shape)
    if body:
        go.add_component(RigidBodyComponent, body_type=body)
    _, colors = _lines(scene)
    assert np.allclose(colors, _rgb(color))


@pytest.mark.parametrize("shape", ["box", "sphere", "capsule"])
def test_trigger_is_yellow_for_each_shape(shape) -> None:
    _, colors = _lines(_single(shape=shape, is_trigger=True))
    assert np.allclose(colors, _rgb("trigger"))


@pytest.mark.parametrize("body", [None, "dynamic", "kinematic"])
def test_trigger_color_wins_over_body_type(body) -> None:
    scene = _single(shape="box", is_trigger=True)
    if body:
        scene.find("Obj").add_component(RigidBodyComponent, body_type=body)
    _, colors = _lines(scene)
    assert np.allclose(colors, _rgb("trigger"))


def test_toggling_is_trigger_changes_color_live() -> None:
    scene = _single(shape="box")
    comp = scene.find("Obj").get_component(ColliderComponent)
    _, plain = _lines(scene)
    comp.is_trigger = True
    _, zone = _lines(scene)
    assert np.allclose(plain, _rgb("static"))
    assert np.allclose(zone, _rgb("trigger"))


def test_sleeping_dims_dynamic_but_not_awake_body() -> None:
    scene = Scene()
    floor = scene.create_game_object("Floor", position=Vector3(0, -0.5, 0))
    floor.add_component(ColliderComponent, shape="box", size=(10, 1, 10))
    box = scene.create_game_object("Box", position=Vector3(0, 0.5, 0))
    box.add_component(ColliderComponent, shape="box")
    rb = box.add_component(RigidBodyComponent)
    scene.fixed_update(1 / 60)
    _, awake = build_debug_lines(scene.physics.world)
    dyn = _rgb("dynamic")
    assert any(np.allclose(c, dyn) for c in awake)
    for _ in range(200):
        scene.fixed_update(1 / 60)
    assert rb.handle is not None
    _, asleep = build_debug_lines(scene.physics.world)
    assert not any(np.allclose(c, dyn) for c in asleep)
    assert any(np.allclose(c, dyn * 0.45, atol=1e-3) for c in asleep)


def test_sleeping_trigger_body_stays_yellow() -> None:
    """Триггер сон не прерывает и не тускнеет: цвет определяет вид «trigger»."""
    scene = Scene()
    go = scene.create_game_object("Zone", position=Vector3(0, 0.0, 0))
    go.add_component(ColliderComponent, shape="box", is_trigger=True)
    go.add_component(RigidBodyComponent, use_gravity=False)
    for _ in range(200):
        scene.fixed_update(1 / 60)
    _, colors = build_debug_lines(scene.physics.world)
    assert np.allclose(colors, _rgb("trigger"))


def test_only_handles_filters_bodies():
    scene = _scene()
    physics = scene.physics
    box = scene.find("Box")
    all_lines, _ = build_debug_lines(physics.world, physics.character_handles())
    one, _ = build_debug_lines(
        physics.world, physics.character_handles(), physics.handles_for([box])
    )
    none, _ = build_debug_lines(physics.world, (), frozenset())
    assert 0 < one.shape[0] < all_lines.shape[0]
    assert one.shape[0] == 24  # 12 рёбер куба
    assert none.shape[0] == 0
