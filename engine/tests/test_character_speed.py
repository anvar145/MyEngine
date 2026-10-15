"""Верификация CharacterController: скорость перемещения.

Параметра ``move_speed`` в API нет (вариант A задания): скорость задаёт вызывающий
через ``simple_move(velocity)``; смещение за шаг = ``velocity * fixed_dt``.
"""

from __future__ import annotations

import numpy as np
import pytest
from character_kit import DT, make_char, make_world, settle, walk

from engine.physics.collider import Collider
from engine.physics.shape import BoxShape
from engine.physics.world import PhysicsWorld


def _walked(velocity, seconds: float = 1.0, **kwargs) -> np.ndarray:
    world = make_world()
    char = make_char(world, **kwargs)
    settle(char)
    start = char.position.copy()
    walk(char, velocity, seconds)
    return char.position - start


# --- 6a: пропорциональность скорости --------------------------------------------------------

@pytest.mark.parametrize("speed", [0.5, 1.0, 2.5, 4.5, 8.0, 20.0])
def test_displacement_equals_speed_times_time(speed: float) -> None:
    delta = _walked((speed, 0.0, 0.0))
    assert delta[0] == pytest.approx(speed, rel=0.01)
    assert abs(delta[1]) < 1e-3
    assert abs(delta[2]) < 1e-6


def test_displacement_scales_with_time() -> None:
    assert _walked((3.0, 0.0, 0.0), 0.5)[0] == pytest.approx(1.5, rel=0.01)
    assert _walked((3.0, 0.0, 0.0), 2.0)[0] == pytest.approx(6.0, rel=0.01)


def test_z_axis_speed_is_symmetric_to_x() -> None:
    assert _walked((0.0, 0.0, 4.5))[2] == pytest.approx(4.5, rel=0.01)


def test_single_step_displacement_is_velocity_times_fixed_dt() -> None:
    world = make_world()
    char = make_char(world)
    settle(char)
    x0 = float(char.position[0])
    char.simple_move((6.0, 0.0, 0.0))
    assert float(char.position[0]) - x0 == pytest.approx(6.0 * DT, abs=1e-9)


def test_simple_move_uses_world_fixed_dt() -> None:
    world = PhysicsWorld(gravity=(0.0, -9.81, 0.0), fixed_dt=1.0 / 30.0)
    world.add_static(Collider(BoxShape((50.0, 0.5, 50.0))), (0.0, -0.5, 0.0))
    char = make_char(world)
    settle(char)
    x0 = float(char.position[0])
    char.simple_move((6.0, 0.0, 0.0))
    assert float(char.position[0]) - x0 == pytest.approx(6.0 / 30.0, abs=1e-9)


def test_explicit_dt_overrides_world_fixed_dt() -> None:
    world = make_world()
    char = make_char(world)
    settle(char)
    x0 = float(char.position[0])
    char.simple_move((6.0, 0.0, 0.0), dt=0.1)
    assert float(char.position[0]) - x0 == pytest.approx(0.6, abs=1e-9)


# --- 6b: нулевая и отрицательная скорость ------------------------------------------------------

def test_zero_velocity_keeps_character_still_on_ground() -> None:
    delta = _walked((0.0, 0.0, 0.0), 2.0)
    assert np.linalg.norm(delta) < 1e-3


def test_zero_velocity_does_not_disable_gravity() -> None:
    world = make_world()
    char = make_char(world, y=5.0)
    for _ in range(30):
        char.simple_move((0.0, 0.0, 0.0))
        world.step()
    assert float(char.position[1]) < 4.0


@pytest.mark.parametrize("speed", [1.0, 4.5])
def test_negative_velocity_moves_backwards_symmetrically(speed: float) -> None:
    forward = _walked((speed, 0.0, 0.0))
    backward = _walked((-speed, 0.0, 0.0))
    assert backward[0] == pytest.approx(-forward[0], abs=1e-6)


# --- 6c: диагональ ---------------------------------------------------------------------------------

def test_diagonal_is_not_normalized_by_controller() -> None:
    """Нормализация — задача скрипта: (v, v) даёт |d| = v·√2."""
    delta = _walked((4.5, 0.0, 4.5))
    assert delta[0] == pytest.approx(4.5, rel=0.01)
    assert delta[2] == pytest.approx(4.5, rel=0.01)
    assert np.linalg.norm(delta[[0, 2]]) == pytest.approx(4.5 * 2 ** 0.5, rel=0.01)


def test_normalized_diagonal_moves_at_walk_speed() -> None:
    v = 4.5 / 2 ** 0.5
    delta = _walked((v, 0.0, v))
    assert np.linalg.norm(delta[[0, 2]]) == pytest.approx(4.5, rel=0.01)


# --- 6d: независимость от формы ----------------------------------------------------------------------

@pytest.mark.parametrize(("radius", "height"), [(0.3, 1.8), (0.5, 2.0), (0.2, 1.2)])
def test_speed_does_not_depend_on_capsule_size(radius: float, height: float) -> None:
    delta = _walked((4.5, 0.0, 0.0), radius=radius, height=height, y=height / 2 + 0.05)
    assert delta[0] == pytest.approx(4.5, rel=0.01)


# --- 6e: вертикальная составляющая ---------------------------------------------------------------------

def test_vertical_velocity_component_is_overridden_by_gravity() -> None:
    """``simple_move`` игнорирует y-скорость: вертикаль ведёт гравитация."""
    world = make_world()
    char = make_char(world)
    settle(char)
    y0 = float(char.position[1])
    for _ in range(30):
        char.simple_move((0.0, 5.0, 0.0))
        world.step()
    assert float(char.position[1]) == pytest.approx(y0, abs=1e-3)
