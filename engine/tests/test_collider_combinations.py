"""Верификация Collider: сочетания параметров (не конфликтуют друг с другом)."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from engine.core.math3d import Quaternion, Vector3
from engine.core.physics_components import ColliderComponent, RigidBodyComponent
from engine.core.scene import Scene
from engine.physics.events import TriggerEnter

DT = 1.0 / 60.0
S = math.sin(math.pi / 4)


def _floor(scene: Scene, **kwargs) -> None:
    scene.create_game_object("Floor", position=Vector3(0, -5, 0),
                             scale=Vector3(100, 10, 100)).add_component(
        ColliderComponent, shape="box", size=(1, 1, 1), **kwargs)


def _body(scene, go):
    return scene.physics.world.get_body(scene.physics.handle_of(go))


def _colliders(scene, go):
    return scene.physics.world.get_colliders(scene.physics.handle_of(go))


def _run(scene, steps: int, update: bool = False) -> None:
    for _ in range(steps):
        scene.fixed_update(DT)
        if update:
            scene.update(DT)


# --- material: friction × restitution ---------------------------------------------

def _bounce_apex(restitution: float, friction: float) -> float:
    scene = Scene()
    _floor(scene, restitution=0.0, friction=friction)
    ball = scene.create_game_object("Ball", position=Vector3(0, 1.25, 0))
    ball.add_component(ColliderComponent, shape="sphere", radius=0.25,
                       restitution=restitution, friction=friction)
    ball.add_component(RigidBodyComponent, can_sleep=False, freeze_rotation=True)
    body = None
    apex, bounced, descended = 0.0, False, False
    for _ in range(300):
        scene.fixed_update(DT)
        body = body or _body(scene, ball)
        vy = float(body.linear_velocity[1])
        if vy < -0.05:
            descended = True
            if bounced:
                break
        elif descended and vy > 0.05:
            bounced = True
        if bounced:
            apex = max(apex, float(body.position[1]) - 0.25)
    return apex


def test_friction_does_not_change_vertical_bounce_height() -> None:
    low = _bounce_apex(0.8, 0.0)
    high = _bounce_apex(0.8, 5.0)
    assert low == pytest.approx(0.64, rel=0.15)
    assert high == pytest.approx(low, rel=0.05)


def test_restitution_does_not_change_sliding_friction_stop() -> None:
    def stop(restitution: float) -> float:
        scene = Scene()
        _floor(scene, friction=0.5, restitution=restitution)
        box = scene.create_game_object("Box", position=Vector3(0, 0.5, 0))
        box.add_component(ColliderComponent, shape="box", size=(1, 1, 1),
                          friction=0.5, restitution=restitution)
        box.add_component(RigidBodyComponent, freeze_rotation=True, can_sleep=False,
                          linear_damping=0.0)
        scene.fixed_update(DT)
        _body(scene, box).linear_velocity[0] = 2.0
        _run(scene, 300)
        return float(_body(scene, box).position[0])

    assert stop(0.0) == pytest.approx(stop(1.0), rel=0.05)


# --- trigger × остальные поля -------------------------------------------------------

def test_trigger_ignores_friction_and_restitution() -> None:
    scene = Scene()
    zone = scene.create_game_object("Zone", position=Vector3(0, 0, 0))
    zone.add_component(ColliderComponent, shape="box", size=(4, 1, 4),
                       is_trigger=True, friction=5.0, restitution=1.0)
    ball = scene.create_game_object("Ball", position=Vector3(0, 3, 0))
    ball.add_component(ColliderComponent, shape="sphere", radius=0.25,
                       restitution=1.0)
    ball.add_component(RigidBodyComponent, can_sleep=False)
    _run(scene, 120)
    assert _body(scene, ball).position[1] < -1.0


def test_trigger_with_layer_mask_only_reports_matching_bodies() -> None:
    scene = Scene()
    zone = scene.create_game_object("Zone", position=Vector3(0, 0, 0))
    zone.add_component(ColliderComponent, shape="box", size=(4, 1, 4),
                       is_trigger=True, layer=0b10, mask=0b100)
    hit = scene.create_game_object("Hit", position=Vector3(-1, 3, 0))
    hit.add_component(ColliderComponent, shape="sphere", radius=0.25, layer=0b100,
                      mask=0b10)
    hit.add_component(RigidBodyComponent)
    miss = scene.create_game_object("Miss", position=Vector3(1, 3, 0))
    miss.add_component(ColliderComponent, shape="sphere", radius=0.25, layer=0b1,
                       mask=0b10)
    miss.add_component(RigidBodyComponent)
    seen: list = []
    scene.events.subscribe(TriggerEnter, seen.append)
    _run(scene, 120, update=True)
    assert [e.user_b for e in seen] == [hit]


def test_trigger_with_offset_and_scale_and_radius() -> None:
    scene = Scene()
    zone = scene.create_game_object("Zone", position=Vector3(0, 1, 0),
                                    scale=Vector3(2, 2, 2))
    zone.add_component(ColliderComponent, shape="sphere", radius=0.5,
                       offset=(3.0, 0.0, 0.0), is_trigger=True)
    ball = scene.create_game_object("Ball", position=Vector3(6, 6, 0))
    ball.add_component(ColliderComponent, shape="sphere", radius=0.25)
    ball.add_component(RigidBodyComponent)
    seen: list = []
    scene.events.subscribe(TriggerEnter, seen.append)
    _run(scene, 120, update=True)  # зона: центр x=6, радиус 1 (0.5 · 2)
    assert [e.user_b for e in seen] == [ball]


# --- составное тело: несколько коллайдеров -----------------------------------------

def test_compound_solid_and_trigger_on_one_object() -> None:
    """Корпус (твёрдый) + датчик над ним (триггер) на одном объекте: зонд
    проходит сквозь датчик (событие входа) и ложится на корпус."""
    scene = Scene()
    _floor(scene)
    go = scene.create_game_object("Sensorbox", position=Vector3(0, 0.5, 0))
    go.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    go.add_component(ColliderComponent, shape="box", size=(3, 1, 3),
                     offset=(0.0, 2.0, 0.0), is_trigger=True)
    go.add_component(RigidBodyComponent, freeze_rotation=True)
    probe = scene.create_game_object("Probe", position=Vector3(0, 8, 0))
    probe.add_component(ColliderComponent, shape="sphere", radius=0.2)
    probe.add_component(RigidBodyComponent, can_sleep=False)
    seen: list = []
    scene.events.subscribe(TriggerEnter, seen.append)
    _run(scene, 240, update=True)
    assert _body(scene, go).position[1] == pytest.approx(0.5, abs=0.03)
    assert _body(scene, probe).position[1] == pytest.approx(1.2, abs=0.03)
    assert len(seen) == 1
    assert seen[0].user_a is go and seen[0].user_b is probe


def test_compound_colliders_with_different_layers() -> None:
    """Ножка слоя 2 останавливает шар слоя-2-маски, плечо слоя 4 — пропускает."""
    scene = Scene()
    go = scene.create_game_object("Wall", position=Vector3(0, 0, 0))
    go.add_component(ColliderComponent, shape="box", size=(1, 4, 4), layer=0b100,
                     offset=(0.0, 0.0, 0.0))
    ball = scene.create_game_object("Ball", position=Vector3(-4, 0, 0))
    ball.add_component(ColliderComponent, shape="sphere", radius=0.25, mask=0b10)
    ball.add_component(RigidBodyComponent, use_gravity=False, linear_damping=0.0,
                       can_sleep=False)
    scene.fixed_update(DT)
    _body(scene, ball).linear_velocity[0] = 3.0
    _run(scene, 180)
    assert _body(scene, ball).position[0] > 3.0  # слой 4 для маски 0b10 невидим


def test_compound_collider_materials_are_independent() -> None:
    scene = Scene()
    go = scene.create_game_object("Two", position=Vector3(0, 2, 0))
    go.add_component(ColliderComponent, shape="sphere", radius=0.25, friction=0.1,
                     restitution=0.9, offset=(-1.0, 0.0, 0.0))
    go.add_component(ColliderComponent, shape="sphere", radius=0.25, friction=2.0,
                     restitution=0.0, offset=(1.0, 0.0, 0.0))
    go.add_component(RigidBodyComponent, use_gravity=False)
    scene.fixed_update(DT)
    mats = sorted((c.material.friction, c.material.restitution)
                  for c in _colliders(scene, go))
    assert mats == [(0.1, 0.9), (2.0, 0.0)]


# --- shape-switch × остальные поля -------------------------------------------------

def test_runtime_shape_switch_keeps_offset_trigger_layer_and_material() -> None:
    scene = Scene()
    go = scene.create_game_object("Obj", position=Vector3(0, 5, 0))
    col = go.add_component(ColliderComponent, shape="box", size=(1, 1, 1),
                           offset=(2.0, 0.0, 0.0), is_trigger=True, layer=0b100,
                           mask=0b1010, friction=0.3, restitution=0.6)
    go.add_component(RigidBodyComponent, use_gravity=False)
    scene.fixed_update(DT)
    col.shape = "sphere"
    col.radius = 0.5
    scene.fixed_update(DT)
    c = _colliders(scene, go)[0]
    assert c.is_trigger and (c.layer, c.mask) == (0b100, 0b1010)
    assert (c.material.friction, c.material.restitution) == (0.3, 0.6)
    lo, hi = c.world_aabb
    assert (lo[0] + hi[0]) / 2 == pytest.approx(2.0, abs=1e-6)


# --- масштаб × смещение × поворот ---------------------------------------------------

def test_scale_offset_rotation_radius_all_at_once() -> None:
    scene = Scene()
    rot = Quaternion(0.0, 0.0, S, S)  # 90° вокруг Z: локальная X → мировая Y
    go = scene.create_game_object("All", position=Vector3(1, 5, 0),
                                  scale=Vector3(2, 3, 2), rotation=rot)
    go.add_component(ColliderComponent, shape="sphere", radius=0.5,
                     offset=(1.0, 0.0, 0.0))
    go.add_component(RigidBodyComponent, use_gravity=False)
    scene.fixed_update(DT)
    lo, hi = _colliders(scene, go)[0].world_aabb
    # радиус 0.5 · max(2,3,2) = 1.5; смещение (1·2, 0, 0) повёрнуто → (0, 2, 0)
    assert np.allclose((lo + hi) / 2, (1.0, 7.0, 0.0), atol=1e-6)
    assert np.allclose(hi - lo, 3.0, atol=1e-6)


# --- устойчивость к странным значениям --------------------------------------------

def test_extreme_values_do_not_break_simulation() -> None:
    scene = Scene()
    _floor(scene)
    cases = [
        {"shape": "box", "size": (1e-6, 1e-6, 1e-6)},
        {"shape": "box", "size": (1e4, 1e4, 1e4)},
        {"shape": "sphere", "radius": 1e-9},
        {"shape": "sphere", "radius": 1e3},
        {"shape": "capsule", "radius": 0.1, "height": 1e4},
        {"shape": "box", "size": (1, 1, 1), "friction": 1e6, "restitution": 1e6},
        {"shape": "box", "size": (1, 1, 1), "layer": -1, "mask": -1},
        {"shape": "box", "size": (1, 1, 1), "offset": (1e5, -1e5, 1e5)},
    ]
    objects = []
    for i, kwargs in enumerate(cases):
        go = scene.create_game_object(f"X{i}", position=Vector3(i * 3.0, 5, 0))
        go.add_component(ColliderComponent, **kwargs)
        rb = go.add_component(RigidBodyComponent)
        objects.append((go, rb))
    _run(scene, 120)
    for go, rb in objects:
        assert rb.handle is not None, go.name
        assert np.all(np.isfinite(_body(scene, go).position)), go.name


def test_random_parameter_combinations_stay_finite() -> None:
    rng = random.Random(1234)
    scene = Scene()
    _floor(scene)
    objects = []
    for i in range(40):
        go = scene.create_game_object(
            f"R{i}", position=Vector3(rng.uniform(-5, 5), rng.uniform(1, 6),
                                      rng.uniform(-5, 5)),
            scale=Vector3(*(rng.uniform(0.3, 2.0) for _ in range(3))))
        go.add_component(
            ColliderComponent,
            shape=rng.choice(["box", "sphere", "capsule"]),
            size=tuple(rng.uniform(0.3, 2.0) for _ in range(3)),
            radius=rng.uniform(0.1, 0.8),
            height=rng.uniform(0.3, 3.0),
            offset=tuple(rng.uniform(-1, 1) for _ in range(3)),
            is_trigger=rng.random() < 0.2,
            friction=rng.uniform(0, 2), restitution=rng.uniform(0, 1),
            layer=rng.choice([1, 2, 4]), mask=rng.choice([0xFFFFFFFF, 3, 6]),
        )
        go.add_component(RigidBodyComponent, body_type=rng.choice(
            ["dynamic", "dynamic", "kinematic"]))
        objects.append(go)
    _run(scene, 240)
    for go in objects:
        assert np.all(np.isfinite(_body(scene, go).position)), go.name


# --- детерминизм ------------------------------------------------------------------

def _determinism_run() -> list[np.ndarray]:
    rng = random.Random(7)
    scene = Scene()
    _floor(scene)
    objects = []
    for i in range(15):
        go = scene.create_game_object(
            f"D{i}", position=Vector3(rng.uniform(-2, 2), 1.0 + i * 0.8,
                                      rng.uniform(-2, 2)))
        go.add_component(ColliderComponent,
                         shape=["box", "sphere", "capsule"][i % 3],
                         offset=(0.1 * (i % 3), 0.0, 0.0),
                         friction=0.4, restitution=0.2)
        go.add_component(RigidBodyComponent)
        objects.append(go)
    _run(scene, 300)
    return [_body(scene, go).position.copy() for go in objects]


def test_same_scene_gives_bitwise_identical_result() -> None:
    first, second = _determinism_run(), _determinism_run()
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a, b)
