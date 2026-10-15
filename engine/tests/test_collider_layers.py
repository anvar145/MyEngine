"""Верификация Collider: layer / mask (фильтр столкновений).

Правило мира: пара взаимодействует, если ``a.layer & b.mask`` и
``b.layer & a.mask`` ненулевые (И, а не ИЛИ). Фильтр применяется до проверки
триггера, поэтому действует и на Trigger-события.
"""

from __future__ import annotations

import pytest

from engine.core.math3d import Vector3
from engine.core.physics_components import ColliderComponent, RigidBodyComponent
from engine.core.scene import Scene
from engine.physics.body import RigidBody
from engine.physics.collider import Collider
from engine.physics.events import CollisionEnter, TriggerEnter
from engine.physics.shape import BoxShape, SphereShape
from engine.physics.world import PhysicsWorld

DT = 1.0 / 60.0
ALL = 0xFFFFFFFF


def _blocked(a: dict, b: dict, *, b_trigger: bool = False) -> bool:
    """Шар A летит со скоростью 3 м/с на статическую стену B (слои заданы dict).

    Returns:
        True, если стена остановила шар (взаимодействие произошло).
    """
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    world.add_static(Collider(BoxShape((0.5, 2.0, 2.0)), is_trigger=b_trigger, **b))
    ball = RigidBody(position=(-4.0, 0.0, 0.0), linear_velocity=(3.0, 0.0, 0.0),
                     linear_damping=0.0, can_sleep=False)
    world.add_body(ball, Collider(SphereShape(0.25), **a))
    for _ in range(180):
        world.step()
    return float(ball.position[0]) < 1.0  # без стены к 3 с он был бы в x = +5


# --- дефолты ----------------------------------------------------------------------

def test_collider_defaults_are_layer_one_and_mask_all() -> None:
    col = Collider(SphereShape(0.5))
    assert col.layer == 1
    assert col.mask == ALL


def test_component_defaults_match_collider_defaults() -> None:
    comp = ColliderComponent()
    assert comp.layer == 1
    assert comp.mask == ALL


def test_default_colliders_collide() -> None:
    assert _blocked({}, {})


# --- пары слоёв -------------------------------------------------------------------

def test_matching_layer_and_mask_collide() -> None:
    assert _blocked({"layer": 1, "mask": 0b10}, {"layer": 0b10, "mask": ALL})


def test_mask_without_other_layer_ignores() -> None:
    """A (layer 1, mask 0b10) видит только слой 2: C (слой 4) проходит насквозь."""
    assert not _blocked({"layer": 1, "mask": 0b10}, {"layer": 0b100, "mask": ALL})


def test_three_layer_scenario() -> None:
    a = {"layer": 0b1, "mask": 0b10}
    b = {"layer": 0b10, "mask": ALL}
    c = {"layer": 0b100, "mask": ALL}
    assert _blocked(a, b)
    assert not _blocked(a, c)


def test_filter_is_and_not_or_one_sided_mask_is_not_enough() -> None:
    """A хочет B (mask включает слой B), но B не хочет A (mask не включает A).

    Правило И: взаимодействия нет. ЗАФИКСИРОВАНО как текущее поведение (решение
    пользователя: И или ИЛИ).
    """
    a = {"layer": 0b1, "mask": 0b10}
    b = {"layer": 0b10, "mask": 0b100}
    assert not _blocked(a, b)


def test_asymmetric_masks_are_symmetric_in_effect() -> None:
    """Пара (A,B) и (B,A) ведут себя одинаково: фильтр симметричен."""
    a = {"layer": 0b1, "mask": 0b10}
    b = {"layer": 0b10, "mask": 0b100}
    assert _blocked(a, b) == _blocked(b, a)


def test_mask_zero_collides_with_nothing() -> None:
    assert not _blocked({"mask": 0}, {})
    assert not _blocked({}, {"mask": 0})


def test_layer_zero_collides_with_nothing() -> None:
    assert not _blocked({"layer": 0}, {})
    assert not _blocked({}, {"layer": 0})


def test_layer_bit_31_works() -> None:
    assert _blocked({"layer": 1 << 31, "mask": 1 << 31},
                    {"layer": 1 << 31, "mask": 1 << 31})
    assert not _blocked({"layer": 1 << 31, "mask": 1 << 31},
                        {"layer": 1 << 30, "mask": 1 << 31})


def test_multi_bit_layer_matches_any_bit() -> None:
    a = {"layer": 0b11, "mask": 0b1000}
    b = {"layer": 0b1000, "mask": 0b10}
    assert _blocked(a, b)  # a.layer & b.mask = 0b10, b.layer & a.mask = 0b1000


def test_layer_above_31_bits_is_not_validated_and_ignored_by_32_bit_mask() -> None:
    """layer = 1<<40 — валидации нет; с mask «все 32 бита» взаимодействия нет."""
    assert not _blocked({"layer": 1 << 40}, {"mask": ALL})
    # а 64-битная маска его «видит»: Python-int без ограничений
    assert _blocked({"layer": 1 << 40, "mask": 1 << 40},
                    {"layer": 1 << 40, "mask": 1 << 40})


def test_negative_mask_is_not_validated() -> None:
    """-1 (все биты в дополнительном коде) ведёт себя как «все слои»."""
    assert _blocked({"mask": -1}, {"mask": -1})


# --- динамика против динамики ---------------------------------------------------

def test_dynamic_dynamic_pair_filtered_by_layers() -> None:
    def run(layer_b: int) -> float:
        world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
        a = RigidBody(position=(-3.0, 0.0, 0.0), linear_velocity=(3.0, 0.0, 0.0),
                      linear_damping=0.0, can_sleep=False)
        b = RigidBody(position=(0.0, 0.0, 0.0), can_sleep=False)
        world.add_body(a, Collider(SphereShape(0.5), layer=0b1, mask=0b10))
        world.add_body(b, Collider(SphereShape(0.5), layer=layer_b, mask=ALL))
        for _ in range(120):
            world.step()
        return float(b.position[0])

    assert run(0b10) > 0.5          # B сдвинут ударом
    assert run(0b100) == pytest.approx(0.0, abs=1e-9)  # B не заметил A


# --- триггеры и события ---------------------------------------------------------

def test_layer_filter_also_hides_trigger_events() -> None:
    def enters(layer_trigger: int) -> int:
        world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
        world.add_static(Collider(BoxShape((0.5, 2.0, 2.0)), is_trigger=True,
                                  layer=layer_trigger))
        ball = RigidBody(position=(-4.0, 0.0, 0.0),
                         linear_velocity=(3.0, 0.0, 0.0), can_sleep=False)
        world.add_body(ball, Collider(SphereShape(0.25), layer=0b1, mask=0b10))
        count = 0
        for _ in range(180):
            world.step()
            count += sum(1 for e in world.pop_events() if type(e) is TriggerEnter)
        return count

    assert enters(0b10) == 1
    assert enters(0b100) == 0


def test_layer_filter_also_hides_collision_events() -> None:
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    world.add_static(Collider(BoxShape((0.5, 2.0, 2.0)), layer=0b100))
    ball = RigidBody(position=(-4.0, 0.0, 0.0), linear_velocity=(3.0, 0.0, 0.0),
                     can_sleep=False)
    world.add_body(ball, Collider(SphereShape(0.25), layer=0b1, mask=0b10))
    events = []
    for _ in range(180):
        world.step()
        events += world.pop_events()
    assert not any(type(e) is CollisionEnter for e in events)


# --- запросы (raycast / overlap) ------------------------------------------------

def _query_world() -> tuple[PhysicsWorld, int, int]:
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    near = world.add_static(Collider(BoxShape((0.5, 1.0, 1.0)), layer=0b10),
                            (2.0, 0.0, 0.0))
    far = world.add_static(Collider(BoxShape((0.5, 1.0, 1.0)), layer=0b100),
                           (6.0, 0.0, 0.0))
    return world, near, far


def test_raycast_layer_mask_selects_layers() -> None:
    world, near, far = _query_world()
    origin, direction = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
    assert world.raycast(origin, direction).handle == near
    assert world.raycast(origin, direction, layer_mask=0b100).handle == far
    assert world.raycast(origin, direction, layer_mask=0b1000) is None


def test_raycast_ignores_collider_mask() -> None:
    """Запрос смотрит только на layer коллайдера; его mask роли не играет."""
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    target = world.add_static(Collider(BoxShape((0.5, 1.0, 1.0)), layer=0b10, mask=0),
                              (2.0, 0.0, 0.0))
    hit = world.raycast((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), layer_mask=0b10)
    assert hit is not None and hit.handle == target


def test_overlap_sphere_layer_mask_selects_layers() -> None:
    world, near, far = _query_world()
    assert world.overlap_sphere((4.0, 0.0, 0.0), 10.0) == sorted([near, far])
    assert world.overlap_sphere((4.0, 0.0, 0.0), 10.0, layer_mask=0b10) == [near]
    assert world.overlap_sphere((4.0, 0.0, 0.0), 10.0, layer_mask=0b1000) == []


# --- компонент --------------------------------------------------------------------

def _scene_pair(a_kwargs: dict, b_kwargs: dict):
    scene = Scene()
    wall = scene.create_game_object("Wall", position=Vector3(0, 0, 0))
    wall.add_component(ColliderComponent, shape="box", size=(1, 4, 4), **b_kwargs)
    ball = scene.create_game_object("Ball", position=Vector3(-4, 0, 0))
    ball.add_component(ColliderComponent, shape="sphere", radius=0.25, **a_kwargs)
    ball.add_component(RigidBodyComponent, use_gravity=False, linear_damping=0.0,
                       can_sleep=False)
    return scene, wall, ball


def _launch(scene, ball, steps: int = 180) -> float:
    body = scene.physics.world.get_body(scene.physics.handle_of(ball))
    body.linear_velocity[0] = 3.0
    for _ in range(steps):
        scene.fixed_update(DT)
    return float(body.position[0])


def test_component_matching_layers_collide() -> None:
    scene, _, ball = _scene_pair({"layer": 1, "mask": 0b10}, {"layer": 0b10})
    scene.fixed_update(DT)
    assert _launch(scene, ball) < 1.0


def test_component_non_matching_layers_pass_through() -> None:
    scene, _, ball = _scene_pair({"layer": 1, "mask": 0b10}, {"layer": 0b100})
    scene.fixed_update(DT)
    assert _launch(scene, ball) > 3.0


def test_component_runtime_layer_change_takes_effect() -> None:
    scene, wall, ball = _scene_pair({"layer": 1, "mask": 0b10}, {"layer": 0b100})
    scene.fixed_update(DT)
    wall.get_component(ColliderComponent).layer = 0b10  # теперь A видит стену
    scene.fixed_update(DT)
    assert _launch(scene, ball) < 1.0


def test_component_runtime_mask_change_takes_effect() -> None:
    scene, _, ball = _scene_pair({"layer": 1, "mask": 0b10}, {"layer": 0b100})
    scene.fixed_update(DT)
    ball.get_component(ColliderComponent).mask = 0b100
    scene.fixed_update(DT)
    assert _launch(scene, ball) < 1.0


def test_component_runtime_layer_change_to_zero_releases_body() -> None:
    scene, wall, ball = _scene_pair({}, {})
    scene.fixed_update(DT)
    wall.get_component(ColliderComponent).layer = 0
    scene.fixed_update(DT)
    assert _launch(scene, ball) > 3.0


def test_component_layers_reach_physics_collider() -> None:
    scene, _, ball = _scene_pair({"layer": 0b100, "mask": 0b1010}, {})
    scene.fixed_update(DT)
    col = scene.physics.world.get_colliders(scene.physics.handle_of(ball))[0]
    assert (col.layer, col.mask) == (0b100, 0b1010)


def test_scene_raycast_layer_mask() -> None:
    scene, wall, _ = _scene_pair({}, {"layer": 0b10})
    scene.fixed_update(DT)
    hit = scene.physics.raycast((-2.0, 0.0, 0.0), (1.0, 0.0, 0.0), layer_mask=0b10)
    assert hit is not None and hit["object"] is wall
    assert scene.physics.raycast((-2.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                 layer_mask=0b100) is None
