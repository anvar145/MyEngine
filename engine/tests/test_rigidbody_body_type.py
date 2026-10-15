"""Верификация RigidBody: тип тела (static / kinematic / dynamic)."""

from __future__ import annotations

import numpy as np
import pytest

from engine.core.math3d import Vector3
from engine.core.physics_components import ColliderComponent, RigidBodyComponent
from engine.core.scene import Scene
from engine.physics.body import RigidBody
from engine.physics.collider import Collider
from engine.physics.shape import BoxShape
from engine.physics.world import PhysicsWorld

DT = 1.0 / 60.0


def _world(gravity=(0.0, -9.81, 0.0)) -> PhysicsWorld:
    return PhysicsWorld(gravity=gravity)


def _add(world, body_type, pos=(0.0, 5.0, 0.0), **kw) -> int:
    body = RigidBody(body_type=body_type, position=pos, **kw)
    return world.add_body(body, Collider(BoxShape((0.5, 0.5, 0.5))))


def _run(world, seconds) -> None:
    for _ in range(round(seconds / world.fixed_dt)):
        world.step()


# --- мир: поведение каждого типа -------------------------------------------

def test_static_ignores_gravity_and_velocity() -> None:
    world = _world()
    h = _add(world, "static")
    body = world.get_body(h)
    body.linear_velocity = np.array([1.0, 0.0, 0.0])
    _run(world, 1.0)
    # static не попадает в awake/movers: ни g, ни своя скорость не двигают
    assert np.allclose(body.position, (0.0, 5.0, 0.0))


def test_static_ignores_impulse() -> None:
    world = _world()
    h = _add(world, "static")
    world.apply_impulse(h, (100.0, 0.0, 0.0))
    body = world.get_body(h)
    assert not body.linear_velocity.any()
    assert body.inv_mass == 0.0


def test_static_is_not_pushed_by_falling_dynamic() -> None:
    world = _world()
    hs = _add(world, "static", pos=(0.0, 0.0, 0.0))
    hd = _add(world, "dynamic", pos=(0.0, 3.0, 0.0))
    _run(world, 2.0)
    assert np.allclose(world.get_body(hs).position, (0.0, 0.0, 0.0))
    # динамика легла сверху: центр на 1.0 выше центра статики (две полувысоты 0.5)
    assert world.get_body(hd).position[1] == pytest.approx(1.0, abs=0.03)


def test_kinematic_ignores_gravity() -> None:
    world = _world()
    h = _add(world, "kinematic")
    _run(world, 1.0)
    assert np.allclose(world.get_body(h).position, (0.0, 5.0, 0.0))


def test_kinematic_moves_with_own_velocity() -> None:
    world = _world()
    h = _add(world, "kinematic")
    world.get_body(h).linear_velocity = np.array([2.0, 0.0, 0.0])
    _run(world, 1.0)
    pos = world.get_body(h).position
    assert pos[0] == pytest.approx(2.0, abs=1e-6)
    assert pos[1] == pytest.approx(5.0, abs=1e-9)  # g не действует


def test_kinematic_ignores_impulse() -> None:
    world = _world()
    h = _add(world, "kinematic")
    world.apply_impulse(h, (100.0, 0.0, 0.0))
    assert not world.get_body(h).linear_velocity.any()


def test_kinematic_pushes_dynamic_and_is_not_pushed() -> None:
    world = _world(gravity=(0.0, 0.0, 0.0))
    hk = _add(world, "kinematic", pos=(0.0, 0.0, 0.0))
    hd = _add(world, "dynamic", pos=(2.0, 0.0, 0.0))
    for i in range(120):
        world.move_kinematic(hk, (i * 0.02, 0.0, 0.0))
        world.step()
    assert world.get_body(hd).position[0] > 2.5  # его сдвинули
    kin_x = world.get_body(hk).position[0]
    assert kin_x == pytest.approx(119 * 0.02, abs=1e-9)  # ход не изменён контактом


def test_dynamic_falls_with_g() -> None:
    world = _world()
    h = _add(world, "dynamic")
    _run(world, 0.5)
    assert world.get_body(h).position[1] < 5.0 - 1.0


def test_dynamic_impulse_changes_velocity() -> None:
    world = _world()
    h = _add(world, "dynamic", mass=2.0)
    world.apply_impulse(h, (4.0, 0.0, 0.0))
    assert world.get_body(h).linear_velocity[0] == pytest.approx(2.0)


def test_unknown_body_type_rejected() -> None:
    with pytest.raises(ValueError):
        RigidBody(body_type="ghost")


def test_only_dynamic_has_inverse_mass() -> None:
    assert RigidBody(body_type="dynamic", mass=4.0).inv_mass == pytest.approx(0.25)
    assert RigidBody(body_type="static", mass=4.0).inv_mass == 0.0
    assert RigidBody(body_type="kinematic", mass=4.0).inv_mass == 0.0


# --- сцена: переключение типа в рантайме -------------------------------------

def _scene_box(body_type: str):
    scene = Scene()
    go = scene.create_game_object("Box", position=Vector3(0, 10, 0))
    go.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    rb = go.add_component(RigidBodyComponent, body_type=body_type)
    return scene, go, rb


def _step(scene, steps) -> None:
    for _ in range(steps):
        scene.fixed_update(DT)
        scene.update(DT)


def test_runtime_dynamic_to_static_freezes_body() -> None:
    scene, go, rb = _scene_box("dynamic")
    _step(scene, 30)
    y0 = go.transform.world_position.y
    assert y0 < 10.0 - 0.5
    old_handle = rb.handle
    rb.body_type = "static"  # mark_dirty → тело пересоздаётся
    _step(scene, 60)
    assert rb.handle is not None
    assert rb.handle != old_handle  # переключение требует пересоздания
    assert go.transform.world_position.y == pytest.approx(y0, abs=1e-4)


def test_runtime_static_to_dynamic_starts_falling() -> None:
    scene, go, rb = _scene_box("static")
    _step(scene, 30)
    assert go.transform.world_position.y == pytest.approx(10.0, abs=1e-6)
    rb.body_type = "dynamic"
    _step(scene, 30)
    assert go.transform.world_position.y < 10.0 - 0.5


def test_runtime_kinematic_to_dynamic_and_back() -> None:
    scene, go, rb = _scene_box("kinematic")
    _step(scene, 30)
    assert go.transform.world_position.y == pytest.approx(10.0, abs=1e-6)
    rb.body_type = "dynamic"
    _step(scene, 30)
    y = go.transform.world_position.y
    assert y < 10.0 - 0.5
    rb.body_type = "kinematic"
    _step(scene, 30)
    assert go.transform.world_position.y == pytest.approx(y, abs=1e-4)


def test_runtime_switch_keeps_velocity() -> None:
    scene, _, rb = _scene_box("dynamic")
    scene.physics.world.set_gravity((0, 0, 0))
    _step(scene, 1)
    rb.linear_velocity = (3.0, 0.0, 0.0)
    rb.mass = 2.0  # пересоздание тела
    v = rb.linear_velocity
    assert v[0] == pytest.approx(3.0, abs=1e-6)


def test_invalid_body_type_in_component_drops_body() -> None:
    scene, _, rb = _scene_box("dynamic")
    _step(scene, 2)
    assert rb.handle is not None
    rb.body_type = "ghost"
    _step(scene, 2)
    assert rb.handle is None  # тело не создаётся, объект пропущен (warning)
