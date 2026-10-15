"""Верификация Collider: смена формы (shape) в рантайме."""

from __future__ import annotations

import numpy as np
import pytest

from engine.core.math3d import Vector3
from engine.core.physics_components import ColliderComponent, RigidBodyComponent
from engine.core.scene import Scene
from engine.physics.body import RigidBody
from engine.physics.shape import BoxShape, CapsuleShape, SphereShape

DT = 1.0 / 60.0


def _scene(thick_floor: bool = True) -> Scene:
    scene = Scene()
    if thick_floor:
        pos, scale = Vector3(0, -5.0, 0), Vector3(100, 10, 100)
    else:
        pos, scale = Vector3(0, -0.5, 0), Vector3(100, 1, 100)
    floor = scene.create_game_object("Floor", position=pos, scale=scale)
    floor.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    return scene


def _spawn(scene, name="Obj", y=3.0, x=0.0, body_type="dynamic", **collider):
    go = scene.create_game_object(name, position=Vector3(x, y, 0))
    col = go.add_component(ColliderComponent, **collider)
    rb = go.add_component(RigidBodyComponent, body_type=body_type,
                          freeze_rotation=True)
    return go, col, rb


def _world(scene):
    return scene.physics.world


def _body(scene, go) -> RigidBody:
    return _world(scene).get_body(scene.physics.handle_of(go))


def _shape(scene, go):
    return _world(scene).get_colliders(scene.physics.handle_of(go))[0].shape


def _run(scene, seconds: float) -> None:
    for _ in range(round(seconds / DT)):
        scene.fixed_update(DT)


def test_box_to_sphere_changes_shape_class_and_rest_height() -> None:
    scene = _scene()
    go, col, _ = _spawn(scene, shape="box", size=(1, 1, 1))
    _run(scene, 2.0)
    assert isinstance(_shape(scene, go), BoxShape)
    assert _body(scene, go).position[1] == pytest.approx(0.5, abs=0.02)
    col.shape = "sphere"
    col.radius = 0.25
    _run(scene, 2.0)
    assert isinstance(_shape(scene, go), SphereShape)
    assert _body(scene, go).position[1] == pytest.approx(0.25, abs=0.02)


def test_sphere_to_capsule_at_runtime() -> None:
    scene = _scene()
    go, col, _ = _spawn(scene, shape="sphere", radius=0.5)
    _run(scene, 2.0)
    assert _body(scene, go).position[1] == pytest.approx(0.5, abs=0.02)
    col.shape = "capsule"
    col.height = 3.0
    _run(scene, 2.0)
    assert isinstance(_shape(scene, go), CapsuleShape)
    assert _body(scene, go).position[1] == pytest.approx(1.5, abs=0.03)


def test_switch_keeps_position_and_velocity() -> None:
    scene = _scene()
    go, col, _ = _spawn(scene, y=20.0, shape="box")
    _run(scene, 0.5)
    pos = _body(scene, go).position.copy()
    vel = _body(scene, go).linear_velocity.copy()
    col.shape = "sphere"
    scene.fixed_update(DT)
    body = _body(scene, go)
    assert vel[1] < -4.0
    assert np.allclose(body.position[[0, 2]], pos[[0, 2]])
    assert body.linear_velocity[1] < vel[1]  # скорость сохранена, падение идёт дальше
    assert body.position[1] < pos[1]


def test_switch_to_unknown_shape_skips_object_with_warning(caplog) -> None:
    scene = _scene()
    _, col, rb = _spawn(scene, shape="box")
    _run(scene, 0.1)
    assert rb.handle is not None
    col.shape = "cone"
    with caplog.at_level("WARNING"):
        scene.fixed_update(DT)
    assert rb.handle is None  # тело не создано, симуляция остальной сцены идёт
    assert any("пропущен" in r.message for r in caplog.records)
    col.shape = "box"  # исправление возвращает тело
    scene.fixed_update(DT)
    assert rb.handle is not None


def test_kinematic_switch_updates_broadphase() -> None:
    """Шар, брошенный после роста кинематики, ложится на новый верх."""
    scene = _scene()
    _, col, _ = _spawn(scene, "Kin", y=1.0, body_type="kinematic",
                       shape="box", size=(4, 1, 4))
    scene.fixed_update(DT)
    col.size = (4.0, 3.0, 4.0)  # верх был 1.5, стал 2.5
    scene.fixed_update(DT)
    ball, _, _ = _spawn(scene, "Ball", y=8.0, shape="sphere", radius=0.25)
    _run(scene, 3.0)
    assert _body(scene, ball).position[1] == pytest.approx(2.5 + 0.25, abs=0.03)


def test_kinematic_switch_to_capsule_updates_broadphase() -> None:
    scene = _scene()
    kin, col, _ = _spawn(scene, "Kin", y=1.0, body_type="kinematic",
                         shape="box", size=(4, 1, 4))
    scene.fixed_update(DT)
    col.shape = "capsule"
    col.radius = 1.0
    col.height = 6.0  # центр 1.0, верх 4.0
    scene.fixed_update(DT)
    ball, _, _ = _spawn(scene, "Ball", y=9.0, x=0.3, shape="sphere", radius=0.25)
    _run(scene, 1.5)
    assert isinstance(_shape(scene, kin), CapsuleShape)
    assert _body(scene, ball).position[1] > 3.5  # лежит на/скатывается с верха


def test_kinematic_shape_aabb_is_current_after_switch() -> None:
    scene = _scene()
    kin, col, _ = _spawn(scene, "Kin", y=2.0, body_type="kinematic",
                         shape="box", size=(1, 1, 1))
    scene.fixed_update(DT)
    col.shape = "sphere"
    col.radius = 2.0
    scene.fixed_update(DT)
    handle = scene.physics.handle_of(kin)
    lo, hi = _world(scene).get_colliders(handle)[0].world_aabb
    assert hi[1] - lo[1] == pytest.approx(4.0, abs=0.01)
    assert hi[1] == pytest.approx(4.0, abs=0.01)


def test_sleeping_body_switch_wakes_and_resizes() -> None:
    """Смена формы пересоздаёт тело: сон теряется, новая форма действует."""
    scene = _scene()
    go, col, _ = _spawn(scene, y=1.0, shape="box", size=(1, 1, 1))
    _run(scene, 3.0)
    assert _body(scene, go).sleeping  # исходно заснуло
    col.shape = "sphere"
    col.radius = 1.0
    scene.fixed_update(DT)
    assert not _body(scene, go).sleeping  # пересозданное тело бодрствует
    _run(scene, 3.0)
    assert _body(scene, go).position[1] == pytest.approx(1.0, abs=0.03)
    assert _body(scene, go).sleeping  # и снова заснуло


def test_sleeping_body_on_top_of_switched_body_is_pushed_up() -> None:
    """Спящий ящик на опоре, опора растёт: верхний не должен остаться внутри."""
    scene = _scene()
    base, base_col, _ = _spawn(scene, "Base", y=0.5, shape="box", size=(2, 1, 2))
    top, _, _ = _spawn(scene, "Top", y=1.5, shape="box", size=(1, 1, 1))
    _run(scene, 3.0)
    assert _body(scene, base).sleeping and _body(scene, top).sleeping
    base_col.size = (2.0, 3.0, 2.0)  # верх опоры был на 1.0, станет на 3.0
    _run(scene, 3.0)
    assert _body(scene, top).position[1] > 3.0 - 0.05  # вытолкнут на верх опоры
    assert _body(scene, top).position[1] == pytest.approx(
        _body(scene, base).position[1] + 1.5 + 0.5, abs=0.1
    )


def test_sleeping_ball_is_woken_by_kinematic_growth() -> None:
    scene = _scene()
    _, col, _ = _spawn(scene, "Kin", y=1.0, body_type="kinematic",
                       shape="box", size=(4, 1, 4))
    ball, _, _ = _spawn(scene, "Ball", y=6.0, x=0.3, shape="sphere", radius=0.25)
    _run(scene, 3.0)
    assert _body(scene, ball).sleeping
    col.size = (4.0, 3.0, 4.0)  # верх опоры 1.5 → 2.5, шар внутри
    _run(scene, 3.0)
    assert _body(scene, ball).position[1] > 2.5 + 0.25 - 0.05


def test_sleeping_ball_is_woken_by_static_growth() -> None:
    scene = _scene()
    stat = scene.create_game_object("Pillar", position=Vector3(0, 0.5, 0))
    col = stat.add_component(ColliderComponent, shape="box", size=(4, 1, 4))
    ball, _, _ = _spawn(scene, "Ball", y=6.0, x=0.3, shape="sphere", radius=0.25)
    _run(scene, 3.0)
    assert _body(scene, ball).sleeping
    col.size = (4.0, 3.0, 4.0)
    _run(scene, 3.0)
    assert _body(scene, ball).position[1] > 2.0 + 0.25 - 0.05


def test_sleeping_ball_is_woken_when_static_support_shrinks() -> None:
    """Опора сузилась из-под спящего шара — он просыпается и падает на пол."""
    scene = _scene()
    stat = scene.create_game_object("Pillar", position=Vector3(0, 0.5, 0))
    col = stat.add_component(ColliderComponent, shape="box", size=(4, 1, 4))
    ball, _, _ = _spawn(scene, "Ball", y=6.0, x=0.3, shape="sphere", radius=0.25)
    _run(scene, 3.0)
    assert _body(scene, ball).sleeping
    col.size = (0.2, 1.0, 0.2)
    _run(scene, 3.0)
    assert _body(scene, ball).position[1] == pytest.approx(0.25, abs=0.03)


def test_sleeping_ball_is_woken_when_static_support_is_destroyed() -> None:
    scene = _scene()
    stat = scene.create_game_object("Pillar", position=Vector3(0, 0.5, 0))
    stat.add_component(ColliderComponent, shape="box", size=(4, 1, 4))
    ball, _, _ = _spawn(scene, "Ball", y=6.0, x=0.3, shape="sphere", radius=0.25)
    _run(scene, 3.0)
    assert _body(scene, ball).sleeping
    stat.destroy()
    _run(scene, 3.0)
    assert _body(scene, ball).position[1] == pytest.approx(0.25, abs=0.03)


def test_spawned_static_wakes_sleeping_ball_inside_it() -> None:
    scene = _scene()
    ball, _, _ = _spawn(scene, "Ball", y=3.0, shape="sphere", radius=0.25)
    _run(scene, 3.0)
    assert _body(scene, ball).sleeping
    # плита 0..0.2 по высоте задевает нижнюю часть шара (он стоит на y=0.25)
    slab = scene.create_game_object("Slab", position=Vector3(0, 0.1, 0))
    slab.add_component(ColliderComponent, shape="box", size=(4, 0.2, 4))
    _run(scene, 3.0)
    assert _body(scene, ball).position[1] == pytest.approx(0.45, abs=0.03)


def test_capsule_grown_to_floor_underside_is_pushed_out() -> None:
    scene = _scene(thick_floor=False)
    go, col, _ = _spawn(scene, shape="sphere", radius=0.5)
    _run(scene, 2.0)
    col.shape = "capsule"
    col.height = 3.0  # низ капсулы ровно на нижней грани пола
    _run(scene, 2.0)
    assert _body(scene, go).position[1] == pytest.approx(1.5, abs=0.05)
