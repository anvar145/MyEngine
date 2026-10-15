"""Верификация CharacterController: определение земли (``grounded``).

Параметра ``ground_check_distance`` в API нет (вариант A задания): проба земли
фиксирована и равна ``skin_width + 0.03`` под нижним полюсом капсулы. Тесты
проверяют поведение ``grounded`` на зазорах вокруг этой пробы.
"""

from __future__ import annotations

import numpy as np
import pytest
from character_kit import WALK, foot_y, make_char, make_world, settle, walk

from engine.physics.collider import Collider
from engine.physics.shape import BoxShape

HALF = 0.9  # половина высоты по умолчанию; зазор под ногами = y - HALF


def _at_gap(gap: float, **kwargs):
    return make_char(make_world(), y=HALF + gap, **kwargs)


# --- 4a: зазор под ногами ------------------------------------------------------------

@pytest.mark.parametrize("gap", [0.0, 0.01, 0.02, 0.03, 0.04])
def test_grounded_within_probe_distance(gap: float) -> None:
    assert _at_gap(gap).grounded


@pytest.mark.parametrize("gap", [0.06, 0.1, 0.3, 1.0])
def test_not_grounded_beyond_probe_distance(gap: float) -> None:
    assert not _at_gap(gap).grounded


def test_probe_distance_follows_skin_width() -> None:
    """Проба = skin_width + 0.03: при skin 0.1 зазор 0.1 ещё «земля», при 0.02 — нет."""
    assert _at_gap(0.10, skin_width=0.1).grounded
    assert not _at_gap(0.10, skin_width=0.02).grounded
    assert not _at_gap(0.20, skin_width=0.1).grounded


def test_ground_normal_is_up_on_flat_floor() -> None:
    char = _at_gap(0.0)
    assert np.allclose(char.ground_normal, (0.0, 1.0, 0.0), atol=1e-6)


def test_falling_character_is_not_grounded_and_lands() -> None:
    world = make_world()
    char = make_char(world, y=3.0)
    assert not char.grounded
    states = []
    for _ in range(90):
        char.simple_move((0.0, 0.0, 0.0))
        world.step()
        states.append(char.grounded)
    assert not states[0]
    assert states[-1]
    # после приземления «земля» не пропадает
    first = states.index(True)
    assert all(states[first:])
    assert foot_y(char) == pytest.approx(0.02, abs=1e-3)


# --- 4b: прыжок и приземление --------------------------------------------------------

def test_jump_leaves_ground_and_lands_back() -> None:
    world = make_world()
    char = make_char(world)
    settle(char)
    assert char.jump(5.0)
    states = []
    for _ in range(120):
        char.simple_move((0.0, 0.0, 0.0))
        world.step()
        states.append(char.grounded)
    assert not states[1]
    assert states[-1]
    assert not all(states)


def test_jump_is_airborne_for_expected_time() -> None:
    """Время полёта ≈ 2v/g (v=5 → 1.02 с): допуск ±15 %."""
    world = make_world()
    char = make_char(world)
    settle(char)
    char.jump(5.0)
    air = 0
    for _ in range(150):
        char.simple_move((0.0, 0.0, 0.0))
        world.step()
        air += not char.grounded
    assert air / 60.0 == pytest.approx(2 * 5.0 / 9.81, rel=0.15)


def test_grounded_stays_true_while_walking_on_flat_floor() -> None:
    world = make_world()
    char = make_char(world)
    settle(char)
    for _ in range(180):
        char.simple_move((WALK, 0.0, 0.0))
        world.step()
        assert char.grounded


def test_walking_off_ledge_drops_grounded_then_restores() -> None:
    world = make_world()
    world.add_static(Collider(BoxShape((2.0, 0.5, 3.0))), (0.0, 1.0, 0.0))  # верх y=1.5
    char = make_char(world, y=1.5 + 0.92)
    settle(char)
    assert char.grounded
    states = []
    for _ in range(120):
        char.simple_move((WALK, 0.0, 0.0))
        world.step()
        states.append(char.grounded)
    assert not all(states)
    assert states[-1]
    assert foot_y(char) == pytest.approx(0.02, abs=1e-2)


# --- 4c: потолок -----------------------------------------------------------------------

def test_ceiling_above_head_does_not_affect_grounded() -> None:
    world = make_world()
    world.add_static(Collider(BoxShape((5.0, 0.5, 5.0))), (0.0, 2.35, 0.0))  # низ 1.85
    char = make_char(world)
    settle(char)
    assert char.grounded
    assert foot_y(char) == pytest.approx(0.02, abs=1e-3)


def test_jump_under_low_ceiling_returns_to_ground() -> None:
    world = make_world()
    world.add_static(Collider(BoxShape((5.0, 0.5, 5.0))), (0.0, 2.6, 0.0))  # низ 2.1
    char = make_char(world)
    settle(char)
    char.jump(6.0)
    top = 0.0
    for _ in range(150):
        char.simple_move((0.0, 0.0, 0.0))
        world.step()
        top = max(top, float(char.position[1] + 0.9))
    assert top <= 2.1 + 1e-3  # в потолок не проходит
    assert char.grounded


# --- 4d: неровный пол ---------------------------------------------------------------------

def test_small_floor_bumps_do_not_flicker_grounded() -> None:
    """Неровности до 2 см не должны отрывать персонажа от земли."""
    world = make_world()
    for i in range(10):
        h = 0.01 + 0.01 * (i % 3)  # 0.01 / 0.02 / 0.03
        world.add_static(Collider(BoxShape((0.25, h / 2, 3.0))),
                         (1.0 + i * 0.5, h / 2, 0.0))
    char = make_char(world)
    settle(char)
    airborne = 0
    for _ in range(120):
        char.simple_move((WALK * 0.5, 0.0, 0.0))
        world.step()
        airborne += not char.grounded
    assert airborne <= 5  # допускаем единичные кадры у уступов


def test_first_frame_after_jump_is_airborne_and_later_grounded() -> None:
    world = make_world()
    char = make_char(world)
    settle(char)
    char.jump(5.0)
    char.simple_move((0.0, 0.0, 0.0))
    world.step()
    assert not char.grounded
    walk(char, (0.0, 0.0, 0.0), 2.0)
    assert char.grounded


def test_walk_reaches_target_distance_on_flat_floor() -> None:
    """Контроль: движение по земле идёт со скоростью ходьбы (±3 %)."""
    world = make_world()
    char = make_char(world)
    settle(char)
    x0 = float(char.position[0])
    walk(char, (WALK, 0.0, 0.0), 1.0)
    assert float(char.position[0]) - x0 == pytest.approx(WALK, rel=0.03)
