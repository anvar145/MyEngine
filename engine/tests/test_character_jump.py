"""Верификация CharacterController: прыжок и гравитация.

Параметров ``jump_velocity``/``gravity`` в API нет (вариант A задания): скорость
прыжка — аргумент ``jump(speed)``, гравитация — ``world.gravity[1]``.
Высота подъёма ≈ v²/2g − v·dt/2: дискретный шаг (гравитация до сдвига) съедает
полшага скорости (при v=5, g=-40 это 13 % высоты). Допуск — v·dt.
"""

from __future__ import annotations

import pytest
from character_kit import make_char, make_world, settle

DT = 1.0 / 60.0


def _jump_height(speed: float, gravity: float = -9.81, steps: int = 600) -> float:
    world = make_world(gravity)
    char = make_char(world)
    settle(char)
    start = float(char.position[1])
    char.jump(speed)
    peak = start
    for _ in range(steps):
        char.simple_move((0.0, 0.0, 0.0))
        world.step()
        peak = max(peak, float(char.position[1]))
    return peak - start


# --- 5a: высота по формуле v²/2g ----------------------------------------------------------

@pytest.mark.parametrize("speed", [3.0, 5.0, 7.0, 10.0])
def test_jump_height_matches_formula(speed: float) -> None:
    expected = speed ** 2 / (2 * 9.81)
    assert _jump_height(speed) == pytest.approx(expected, abs=speed * DT)


@pytest.mark.parametrize("gravity", [-3.0, -9.81, -20.0, -40.0])
def test_jump_height_follows_world_gravity(gravity: float) -> None:
    expected = 25.0 / (2 * -gravity)
    assert _jump_height(5.0, gravity) == pytest.approx(expected, abs=5.0 * DT)


def test_higher_speed_gives_higher_jump() -> None:
    assert _jump_height(3.0) < _jump_height(5.0) < _jump_height(8.0)


def test_stronger_gravity_gives_lower_jump() -> None:
    assert _jump_height(5.0, -30.0) < _jump_height(5.0, -9.81) < _jump_height(5.0, -3.0)


def test_discrete_step_makes_jump_slightly_lower_than_formula() -> None:
    """Фиксируем знак погрешности: ниже формулы, не выше."""
    assert _jump_height(5.0) <= 25.0 / (2 * 9.81) + 1e-6


# --- 5b: возврат на землю ------------------------------------------------------------------

@pytest.mark.parametrize("gravity", [-9.81, -20.0])
def test_character_returns_to_original_height(gravity: float) -> None:
    world = make_world(gravity)
    char = make_char(world)
    settle(char)
    rest = float(char.position[1])
    char.jump(5.0)
    for _ in range(300):
        char.simple_move((0.0, 0.0, 0.0))
        world.step()
    assert char.grounded
    assert float(char.position[1]) == pytest.approx(rest, abs=1e-3)


# --- 5c: граничные значения скорости -----------------------------------------------------------

@pytest.mark.parametrize("speed", [0.0, -3.0, -1e-9])
def test_non_positive_jump_speed_raises(speed: float) -> None:
    """F7: ``speed <= 0`` — ValueError; персонаж остаётся на земле."""
    world = make_world()
    char = make_char(world)
    settle(char)
    with pytest.raises(ValueError, match="speed"):
        char.jump(speed)
    assert char.grounded
    assert char.vertical_velocity == 0.0


def test_jump_default_speed_is_five() -> None:
    world = make_world()
    char = make_char(world)
    settle(char)
    assert char.jump() is True
    assert char.vertical_velocity == 5.0


def test_invalid_speed_raises_even_in_air() -> None:
    world = make_world()
    char = make_char(world, y=5.0)
    with pytest.raises(ValueError):
        char.jump(-1.0)


def test_huge_speed_stays_finite_and_lands() -> None:
    world = make_world()
    char = make_char(world)
    settle(char)
    char.jump(200.0)
    for _ in range(3000):
        char.simple_move((0.0, 0.0, 0.0))
        world.step()
    assert char.grounded
    assert float(char.position[1]) == pytest.approx(0.92, abs=1e-2)


# --- 5d: повторные прыжки ---------------------------------------------------------------------------

def test_jump_returns_true_on_ground_and_false_in_air() -> None:
    world = make_world()
    char = make_char(world)
    settle(char)
    assert char.jump(5.0) is True
    char.simple_move((0.0, 0.0, 0.0))
    world.step()
    assert not char.grounded
    assert char.jump(5.0) is False


def test_double_jump_in_one_frame_is_not_additive() -> None:
    world = make_world()
    char = make_char(world)
    settle(char)
    start = float(char.position[1])
    char.jump(5.0)
    char.jump(5.0)
    peak = start
    for _ in range(600):
        char.simple_move((0.0, 0.0, 0.0))
        world.step()
        peak = max(peak, float(char.position[1]))
    assert peak - start == pytest.approx(25.0 / (2 * 9.81), abs=5.0 * DT)


def test_air_jump_does_not_change_trajectory() -> None:
    def peak_with(air_jump: bool) -> float:
        world = make_world()
        char = make_char(world)
        settle(char)
        start = float(char.position[1])
        char.jump(5.0)
        peak = start
        for i in range(600):
            if air_jump and i == 20:
                char.jump(5.0)
            char.simple_move((0.0, 0.0, 0.0))
            world.step()
            peak = max(peak, float(char.position[1]))
        return peak

    assert peak_with(True) == pytest.approx(peak_with(False), abs=1e-9)


def test_second_jump_after_landing_works() -> None:
    world = make_world()
    char = make_char(world)
    settle(char)
    char.jump(5.0)
    for _ in range(150):
        char.simple_move((0.0, 0.0, 0.0))
        world.step()
    assert char.grounded
    assert char.jump(5.0) is True


# --- 5e: гравитация без прыжка -----------------------------------------------------------------------

def test_free_fall_distance_matches_gravity() -> None:
    world = make_world()
    char = make_char(world, y=20.0)
    for _ in range(30):  # 0.5 с
        char.simple_move((0.0, 0.0, 0.0))
        world.step()
    fallen = 20.0 - float(char.position[1])
    assert fallen == pytest.approx(0.5 * 9.81 * 0.25, rel=0.06)


def test_zero_gravity_character_floats() -> None:
    world = make_world(0.0)
    char = make_char(world, y=5.0)
    for _ in range(60):
        char.simple_move((0.0, 0.0, 0.0))
        world.step()
    assert float(char.position[1]) == pytest.approx(5.0, abs=1e-6)


def test_jump_with_zero_gravity_never_returns() -> None:
    world = make_world(0.0)
    char = make_char(world)
    settle(char)
    char.jump(5.0)
    for _ in range(120):
        char.simple_move((0.0, 0.0, 0.0))
        world.step()
    assert not char.grounded
    assert float(char.position[1]) > 5.0


@pytest.mark.parametrize(("speed", "gravity"), [(3.0, -9.81), (5.0, -20.0), (5.0, -40.0)])
def test_jump_deficit_is_half_step_of_velocity(speed: float, gravity: float) -> None:
    """Погрешность формулы — ровно ``v·dt/2`` (≈ полшага явного Эйлера)."""
    expected = speed ** 2 / (2 * -gravity) - speed * DT / 2
    assert _jump_height(speed, gravity) == pytest.approx(expected, abs=2e-3)
