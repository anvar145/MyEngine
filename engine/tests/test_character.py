"""Тесты CharacterController: скольжение, ступени, склоны, прыжок, триггеры."""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.physics.body import RigidBody
from engine.physics.character import CharacterController
from engine.physics.collider import Collider
from engine.physics.events import TriggerEnter, TriggerExit
from engine.physics.mathutil import quat_from_axis_angle
from engine.physics.shape import BoxShape
from engine.physics.world import PhysicsWorld

HEIGHT = 1.8


def make_world() -> PhysicsWorld:
    world = PhysicsWorld()
    world.add_static(Collider(BoxShape((50, 0.5, 50))), (0, -0.5, 0))
    return world


def make_char(world, x=0.0, z=0.0, **kw) -> CharacterController:
    return CharacterController(world, (x, HEIGHT / 2 + 0.05, z), height=HEIGHT, **kw)


def settle(char, steps=30) -> None:
    for _ in range(steps):
        char.simple_move((0, 0, 0))
        char.world.step()


def walk(char, velocity, seconds) -> None:
    for _ in range(round(seconds * 60)):
        char.simple_move(velocity)
        char.world.step()


def test_character_validation() -> None:
    world = make_world()
    with pytest.raises(ValueError):
        CharacterController(world, height=0.5, radius=0.4)


def test_character_stands_on_floor_with_skin_gap() -> None:
    world = make_world()
    char = make_char(world)
    settle(char)
    assert char.is_grounded
    foot = (char.position[1] - char.height / 2)
    assert 0.0 <= foot <= 0.1
    assert np.allclose(char.ground_normal, (0, 1, 0), atol=1e-3)
    assert char.vertical_velocity == 0.0


def test_character_walks_flat_ground() -> None:
    world = make_world()
    char = make_char(world)
    settle(char)
    walk(char, (3, 0, 0), 1.0)
    assert char.position[0] == pytest.approx(3.0, abs=0.1)
    assert char.is_grounded


def test_character_blocked_by_wall_and_slides() -> None:
    world = make_world()
    world.add_static(Collider(BoxShape((0.5, 3, 10))), (5, 3, 0))  # стена x=4.5
    char = make_char(world)
    settle(char)
    walk(char, (3, 0, 0), 2.0)
    assert char.position[0] < 4.5 - char.radius + 1e-6
    assert char.position[0] > 4.5 - char.radius - 0.1
    # диагональ вдоль стены: скользит по z
    z0 = char.position[2]
    walk(char, (3, 0, 3), 1.0)
    assert char.position[2] - z0 > 2.0
    assert char.position[0] < 4.5 - char.radius + 1e-6


def test_character_climbs_low_steps_but_not_high() -> None:
    world = make_world()
    for i in range(1, 4):  # ступени по 0.2
        world.add_static(
            Collider(BoxShape((0.5, 0.1 * i, 3))), (3 + i, 0.1 * i, 0)
        )
    char = make_char(world, step_offset=0.3)
    settle(char)
    walk(char, (2.5, 0, 0), 2.8)
    assert char.position[0] > 6.0
    assert (char.position[1] - char.height / 2) == pytest.approx(0.6, abs=0.1)
    assert char.is_grounded

    world2 = make_world()
    world2.add_static(Collider(BoxShape((0.5, 0.5, 3))), (3, 0.5, 0))  # ступень 1.0
    c2 = make_char(world2, step_offset=0.3)
    settle(c2)
    walk(c2, (2.5, 0, 0), 2.0)
    assert c2.position[0] < 2.5 - c2.radius + 0.1


def test_character_stairs_down_stays_grounded() -> None:
    world = make_world()
    world.add_static(Collider(BoxShape((1.0, 0.1, 3))), (0, 0.1, 0))
    world.add_static(Collider(BoxShape((1.0, 0.05, 3))), (2, 0.05, 0))
    char = CharacterController(world, (0, 0.2 + HEIGHT / 2 + 0.05, 0), height=HEIGHT)
    settle(char)
    grounded_all = True
    for _ in range(90):
        char.simple_move((2, 0, 0))
        world.step()
        grounded_all &= char.is_grounded
    assert grounded_all
    assert char.position[0] > 2.5


def slope_world(deg: float) -> PhysicsWorld:
    world = PhysicsWorld()
    world.add_static(Collider(BoxShape((50, 0.5, 50))), (0, -0.5, 0))
    rot = quat_from_axis_angle((0, 0, 1), math.radians(deg))
    t = math.radians(deg)
    # плоскость склона начинается в x=2 и поднимается вдоль +X
    centre = np.array([2 + 5 * math.cos(t), 5 * math.sin(t), 0]) - 0.5 * np.array(
        [-math.sin(t), math.cos(t), 0]
    )
    world.add_static(Collider(BoxShape((5, 0.5, 5))), centre, rot)
    return world


def test_character_walks_up_gentle_slope() -> None:
    world = slope_world(20)
    char = make_char(world, slope_limit=45)
    settle(char)
    walk(char, (3, 0, 0), 1.5)
    assert char.position[0] > 4.0
    assert (char.position[1] - char.height / 2) > 0.5
    assert char.is_grounded


def test_character_cannot_climb_steep_slope() -> None:
    world = slope_world(55)
    char = make_char(world, slope_limit=45)
    settle(char)
    walk(char, (3, 0, 0), 2.0)
    assert (char.position[1] - char.height / 2) < 0.6


def test_character_jump_height() -> None:
    world = make_world()
    char = make_char(world)
    settle(char)
    assert char.jump(5.0)
    assert not char.jump(5.0)  # в воздухе нельзя
    y0 = char.position[1]
    peak = y0
    for _ in range(120):
        char.simple_move((0, 0, 0))
        world.step()
        peak = max(peak, char.position[1])
    assert peak - y0 == pytest.approx(5.0 ** 2 / (2 * 9.81), abs=0.15)
    assert char.is_grounded


def test_character_head_bump_stops_jump() -> None:
    world = make_world()
    world.add_static(Collider(BoxShape((5, 0.5, 5))), (0, 2.8, 0))  # потолок y=2.3
    char = make_char(world)
    settle(char)
    char.jump(8.0)
    for _ in range(10):
        char.simple_move((0, 0, 0))
        world.step()
    assert char.position[1] + HEIGHT / 2 <= 2.3 + 1e-6
    assert char.vertical_velocity <= 0.0


def test_character_triggers_world_events() -> None:
    world = make_world()
    world.add_static(Collider(BoxShape((1, 2, 1)), is_trigger=True), (3, 1, 0))
    char = make_char(world)
    settle(char)
    world.pop_events()
    kinds = []
    for _ in range(180):
        char.simple_move((2.5, 0, 0))
        world.step()
        kinds += [type(e) for e in world.pop_events() if type(e) is not object]
    assert TriggerEnter in kinds and TriggerExit in kinds


def test_character_blocks_dynamic_box_and_box_blocks_character() -> None:
    world = make_world()
    body = RigidBody(position=(2.0, 0.5, 0))
    world.add_body(body, Collider(BoxShape((0.5, 0.5, 0.5))))
    char = make_char(world)
    settle(char)
    walk(char, (2, 0, 0), 1.0)
    assert char.position[0] < 2.0 - 0.5 - char.radius + 0.5


def test_character_remove() -> None:
    world = make_world()
    char = make_char(world)
    char.remove()
    assert char.handle not in world
    char.remove()


def test_character_standing_still_limits_collide_calls(monkeypatch) -> None:
    """Стоя на месте персонаж не делает лишних свипов (посадка = один свип)."""
    import engine.physics.narrowphase as narrow
    import engine.physics.world as world_mod

    world = make_world()
    char = make_char(world)
    settle(char)
    calls = {"n": 0}

    def counting(real):
        def wrapper(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)
        return wrapper

    monkeypatch.setattr(world_mod, "collide", counting(world_mod.collide))
    monkeypatch.setattr(narrow, "collide", counting(narrow.collide))
    for _ in range(60):
        char.simple_move((0, 0, 0))
        world.step()
    assert char.grounded
    assert calls["n"] / 60 <= 10
