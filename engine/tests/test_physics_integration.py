"""Интеграция физики с ECS: компоненты, scene.physics, события, персонаж."""

from __future__ import annotations

import numpy as np
import pytest

from engine.core.math3d import Quaternion, Vector3
from engine.core.physics_components import (
    CharacterControllerComponent,
    ColliderComponent,
    RigidBodyComponent,
)
from engine.core.scene import Scene
from engine.physics.events import CollisionEnter, TriggerEnter, TriggerExit

DT = 1.0 / 60.0


def _floor(scene: Scene):
    go = scene.create_game_object(
        "Floor", position=Vector3(0, -0.5, 0), scale=Vector3(20, 1, 20)
    )
    go.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    return go


def _box(scene: Scene, y: float = 2.0, name: str = "Box"):
    go = scene.create_game_object(name, position=Vector3(0, y, 0))
    go.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    go.add_component(RigidBodyComponent, mass=1.0)
    return go


def _run(scene: Scene, steps: int) -> None:
    for _ in range(steps):
        scene.fixed_update(DT)
        scene.update(DT)


def test_scene_without_physics_does_not_create_world():
    scene = Scene()
    scene.create_game_object("A")
    _run(scene, 3)
    assert scene.physics_if_exists is None


def test_box_falls_and_rests_on_floor():
    scene = Scene()
    _floor(scene)
    box = _box(scene)
    _run(scene, 240)
    y = box.transform.world_position.y
    assert y == pytest.approx(0.5, abs=0.03)


def test_dynamic_body_writes_rotation_back():
    scene = Scene()
    go = scene.create_game_object("Free", position=Vector3(0, 5, 0))
    go.add_component(ColliderComponent, shape="sphere", radius=0.5)
    go.add_component(RigidBodyComponent, mass=1.0)
    scene.physics.world.set_gravity((0, 0, 0))
    scene.physics.sync_in()
    scene.physics.set_velocity(go, angular=(0, 3.0, 0))
    _run(scene, 30)
    rot = go.transform.rotation
    assert abs(rot.y) > 0.1


def test_teleport_from_script_moves_body():
    scene = Scene()
    _floor(scene)
    box = _box(scene, y=0.5)
    _run(scene, 60)
    box.transform.position = Vector3(5, 3, 0)
    _run(scene, 1)
    assert box.transform.world_position.x == pytest.approx(5.0, abs=0.05)
    _run(scene, 240)
    assert box.transform.world_position.y == pytest.approx(0.5, abs=0.03)


def test_static_collider_scaled_by_object_scale():
    scene = Scene()
    _floor(scene)
    ball = scene.create_game_object("Ball", position=Vector3(9.5, 3, 0))
    ball.add_component(ColliderComponent, shape="sphere", radius=0.5)
    ball.add_component(RigidBodyComponent)
    _run(scene, 240)
    # край пола на x=10: шар (x=9.5) лежит на нём
    assert ball.transform.world_position.y == pytest.approx(0.5, abs=0.05)


def test_kinematic_follows_transform_and_pushes():
    scene = Scene()
    _floor(scene)
    mover = scene.create_game_object("Mover", position=Vector3(-3, 0.5, 0))
    mover.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    mover.add_component(RigidBodyComponent, body_type="kinematic")
    box = _box(scene, y=0.5)
    for i in range(120):
        mover.transform.position = Vector3(-3 + i * 0.04, 0.5, 0)
        scene.fixed_update(DT)
        scene.update(DT)
    assert box.transform.world_position.x > 0.5


def test_destroy_removes_body():
    scene = Scene()
    _floor(scene)
    box = _box(scene)
    _run(scene, 5)
    count = len(scene.physics.world)
    scene.destroy_game_object(box)
    _run(scene, 3)
    assert len(scene.physics.world) == count - 1


def test_component_change_rebuilds_body():
    scene = Scene()
    _floor(scene)
    box = _box(scene, y=0.5)
    _run(scene, 60)
    body = box.get_component(RigidBodyComponent)
    handle_before = body.handle
    body.mass = 10.0
    _run(scene, 1)
    assert body.handle != handle_before
    assert box.transform.world_position.y == pytest.approx(0.5, abs=0.03)


def test_trigger_events_reach_event_bus():
    scene = Scene()
    zone = scene.create_game_object("Zone", position=Vector3(0, 1, 0))
    zone.add_component(ColliderComponent, shape="box", size=(2, 2, 2), is_trigger=True)
    ball = scene.create_game_object("Ball", position=Vector3(0, 6, 0))
    ball.add_component(ColliderComponent, shape="sphere", radius=0.25)
    ball.add_component(RigidBodyComponent)
    seen: list = []
    scene.events.subscribe(TriggerEnter, seen.append)
    scene.events.subscribe(TriggerExit, seen.append)
    _run(scene, 120)
    kinds = [type(e).__name__ for e in seen]
    assert kinds == ["TriggerEnter", "TriggerExit"]
    assert seen[0].a == scene.physics.handle_of(zone)
    assert seen[0].user_b is ball


def test_collision_event_has_game_objects():
    scene = Scene()
    _floor(scene)
    box = _box(scene, y=1.0)
    seen: list = []
    scene.events.subscribe(CollisionEnter, seen.append)
    _run(scene, 60)
    assert any(box in (e.user_a, e.user_b) for e in seen)


def test_parented_dynamic_body_world_pose():
    scene = Scene()
    _floor(scene)
    parent = scene.create_game_object("Parent", position=Vector3(2, 0, 0))
    child = scene.create_game_object(
        "Child", parent=parent, position=Vector3(0, 3, 0)
    )
    child.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    child.add_component(RigidBodyComponent)
    _run(scene, 240)
    wp = child.transform.world_position
    assert wp.x == pytest.approx(2.0, abs=0.05)
    assert wp.y == pytest.approx(0.5, abs=0.03)


def test_character_component_moves_and_stands():
    scene = Scene()
    _floor(scene)
    hero = scene.create_game_object("Hero", position=Vector3(0, 0.9 + 0.03, 0))
    ctrl = hero.add_component(CharacterControllerComponent)
    _run(scene, 5)
    for _ in range(60):
        ctrl.simple_move((2.0, 0.0, 0.0))
        scene.fixed_update(DT)
    pos = hero.transform.world_position
    assert pos.x == pytest.approx(2.0, abs=0.1)
    assert pos.y == pytest.approx(0.9, abs=0.05)
    assert ctrl.is_grounded


def test_character_jump_leaves_ground():
    scene = Scene()
    _floor(scene)
    hero = scene.create_game_object("Hero", position=Vector3(0, 0.92, 0))
    ctrl = hero.add_component(CharacterControllerComponent)
    for _ in range(10):
        ctrl.simple_move((0, 0, 0))
    assert ctrl.jump(5.0)
    heights = []
    for _ in range(40):
        ctrl.simple_move((0, 0, 0))
        heights.append(hero.transform.world_position.y)
    assert max(heights) > 1.5
    assert heights[-1] < max(heights)


def test_old_scene_without_physics_components_unchanged():
    scene = Scene()
    go = scene.create_game_object("Cube", position=Vector3(1, 2, 3))
    _run(scene, 10)
    pos = go.transform.position
    assert (pos.x, pos.y, pos.z) == (1.0, 2.0, 3.0)


def test_world_rotation_and_set_world_pose():
    scene = Scene()
    parent = scene.create_game_object(
        "P", position=Vector3(1, 0, 0),
        rotation=Quaternion.from_axis_angle(Vector3(0, 1, 0), 90),
    )
    child = scene.create_game_object("C", parent=parent)
    target = Quaternion.from_axis_angle(Vector3(0, 1, 0), 30)
    child.transform.set_world_pose(Vector3(1, 2, 3), target)
    wp = child.transform.world_position
    assert (wp.x, wp.y, wp.z) == pytest.approx((1.0, 2.0, 3.0), abs=1e-5)
    wr = child.transform.world_rotation
    assert np.allclose(wr.to_numpy(), target.to_numpy(), atol=1e-5) or np.allclose(
        wr.to_numpy(), -target.to_numpy(), atol=1e-5
    )
