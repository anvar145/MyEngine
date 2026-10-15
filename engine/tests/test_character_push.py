"""Взаимодействие персонажа с динамикой и платформами: фиксация текущего дизайна.

F5 — персонаж **не толкает** динамические тела: они для него препятствия.
F6 — персонаж **не едет** на движущихся платформах. Это осознанные ограничения
(см. docs/CORE.md §7.2); тесты документируют поведение и не требуют его менять.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.physics.body import RigidBody
from engine.physics.collider import Collider
from engine.physics.shape import BoxShape
from engine.tests.character_kit import WALK, make_char, make_world, settle, walk


def _standing():
    world = make_world()
    char = make_char(world)
    settle(char)
    return world, char


def _box(world, pos, mass=1.0, half=0.5) -> RigidBody:
    body = RigidBody(mass=mass, position=pos)
    world.add_body(body, Collider(BoxShape((half,) * 3)))
    return body


@pytest.mark.parametrize("mass", [0.1, 1.0, 50.0])
def test_character_does_not_push_dynamic_box(mass):
    """F5: ящик остаётся на месте при любой массе, персонаж упирается в него."""
    world, char = _standing()
    box = _box(world, (2.5, 0.5, 0.0), mass=mass)
    walk(char, (WALK, 0.0, 0.0), 3.0)
    assert box.position[0] == pytest.approx(2.5, abs=0.01)
    assert abs(box.linear_velocity[0]) < 0.01
    # персонаж остановился у грани ящика: 2.0 - radius - skin
    assert char.position[0] == pytest.approx(2.0 - char.radius - char.skin_width,
                                              abs=0.03)


def test_character_does_not_push_awake_box_either():
    world, char = _standing()
    box = _box(world, (2.5, 0.5, 0.0))
    for _ in range(180):
        box.wake()  # ящик бодрствует каждый кадр: сон не причина неподвижности
        char.simple_move((WALK, 0.0, 0.0))
        world.step()
    assert box.position[0] == pytest.approx(2.5, abs=0.02)


def test_box_blocks_character_like_a_wall_and_does_not_move_it():
    world, char = _standing()
    _box(world, (2.5, 0.5, 0.0))
    walk(char, (WALK, 0.0, 0.0), 2.0)
    x = char.position[0]
    walk(char, (WALK, 0.0, 0.0), 2.0)
    assert char.position[0] == pytest.approx(x, abs=1e-3)


def test_falling_box_does_not_push_character_down():
    """Динамика, упавшая на персонажа, для него не сила: он остаётся на месте."""
    world, char = _standing()
    box = _box(world, (0.0, 4.0, 0.0), mass=20.0)
    for _ in range(180):
        char.simple_move((0.0, 0.0, 0.0))
        world.step()
    # тяжёлый ящик на макушке вжимает персонажа максимум на skin (0.02 м)
    assert char.position[1] == pytest.approx(0.92, abs=0.03)
    assert np.isfinite(box.position).all()
    assert box.position[1] > char.position[1]  # ящик не проник сквозь персонажа


def test_character_stands_on_dynamic_box_without_sinking():
    world, char = _standing()
    box = _box(world, (3.0, 0.5, 0.0), half=0.5)
    char.teleport((3.0, 1.0 + 0.95, 0.0))
    for _ in range(120):
        char.simple_move((0.0, 0.0, 0.0))
        world.step()
    assert char.grounded
    assert char.position[1] == pytest.approx(1.0 + 0.92, abs=0.03)
    assert box.position[1] == pytest.approx(0.5, abs=0.02)


def test_character_does_not_ride_moving_kinematic_platform():
    """F6: платформа уезжает из-под персонажа, он стоит на месте (потом падает)."""
    world, char = _standing()
    platform = RigidBody(mass=1.0, position=(5.0, 0.25, 0.0), body_type="kinematic")
    handle = world.add_body(platform, Collider(BoxShape((2.0, 0.25, 2.0))))
    char.teleport((5.0, 0.5 + 0.95, 0.0))
    for _ in range(10):
        char.simple_move((0.0, 0.0, 0.0))
        world.step()
    assert char.grounded
    start_x = char.position[0]
    pos = np.array([5.0, 0.25, 0.0])
    for _ in range(60):
        pos[0] += 1.0 / 60.0  # платформа едет +X со скоростью 1 м/с
        world.move_kinematic(handle, pos)
        char.simple_move((0.0, 0.0, 0.0))
        world.step()
    assert char.position[0] == pytest.approx(start_x, abs=0.01)  # не поехал
    assert pos[0] == pytest.approx(6.0, abs=1e-6)


def test_character_does_not_inherit_platform_velocity():
    _world, char = _standing()
    assert char.vertical_velocity == 0.0
    # скорость персонажа никогда не складывается со скоростью опоры
    assert not hasattr(char, "ground_velocity")
    assert not hasattr(char, "platform_velocity")
