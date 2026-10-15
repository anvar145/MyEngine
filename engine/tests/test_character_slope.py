"""Верификация CharacterController: ``slope_limit`` (проходимость склонов).

Склон начинается на полу в ``x=2`` и поднимается вдоль +X; персонаж идёт 2 с
со скоростью 4.5 м/с. «Взошёл» — нижняя точка выше 1 м.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from character_kit import (
    WALK,
    add_slope,
    add_step,
    foot_y,
    make_char,
    make_world,
    settle,
    walk,
)

CLIMB_Y = 1.0


def _walk_slope(degrees: float, seconds: float = 2.0, **kwargs):
    world = make_world()
    add_slope(world, degrees)
    char = make_char(world, **kwargs)
    settle(char)
    walk(char, (WALK, 0.0, 0.0), seconds)
    return char


# --- 3a-3e: углы относительно лимита 45° ----------------------------------------------

@pytest.mark.parametrize("degrees", [10, 20, 30, 40, 44])
def test_slope_below_limit_is_climbed(degrees: float) -> None:
    char = _walk_slope(degrees)
    assert foot_y(char) > CLIMB_Y
    assert char.grounded


@pytest.mark.parametrize("degrees", [46, 50, 60, 80, 89])
def test_slope_above_limit_is_not_climbed(degrees: float) -> None:
    char = _walk_slope(degrees)
    assert foot_y(char) < 0.5
    assert char.position[0] < 2.0 + 0.5  # остановился у подножия


@pytest.mark.parametrize("degrees", [10, 30, 44])
def test_slope_speed_along_surface_is_not_faster_than_flat(degrees: float) -> None:
    """Подъём по склону не быстрее ходьбы по плоскости."""
    char = _walk_slope(degrees, seconds=1.5)
    start = np.array([0.0, 0.92, 0.0])
    travelled = float(np.linalg.norm(char.position - start))
    assert travelled <= WALK * 1.5 * 1.05


def test_ground_normal_on_slope_matches_angle() -> None:
    char = _walk_slope(30.0, seconds=1.2)
    assert char.grounded
    angle = math.degrees(math.acos(float(char.ground_normal[1])))
    assert angle == pytest.approx(30.0, abs=1.0)


def test_steep_slope_is_not_standable_after_dropping_onto_it() -> None:
    """60°: персонаж не удерживается и скатывается на пол."""
    world = make_world()
    add_slope(world, 60.0)
    char = make_char(world, x=2.4, y=0.4 * math.tan(math.radians(60.0)) + 3.0)
    for _ in range(180):
        char.simple_move((0.0, 0.0, 0.0))
        world.step()
    assert foot_y(char) < 0.5


def test_walkable_slope_holds_still_without_sliding() -> None:
    world = make_world()
    add_slope(world, 30.0)
    char = make_char(world)
    settle(char)
    walk(char, (WALK, 0.0, 0.0), 1.0)
    before = char.position.copy()
    settle(char, 120)
    assert np.linalg.norm(char.position - before) < 0.02


# --- 3f-3g: изменение slope_limit ------------------------------------------------------

def test_raising_slope_limit_allows_steeper_slope() -> None:
    assert foot_y(_walk_slope(50.0, slope_limit=45.0)) < 0.5
    assert foot_y(_walk_slope(50.0, slope_limit=55.0)) > CLIMB_Y


def test_lowering_slope_limit_blocks_gentler_slope() -> None:
    assert foot_y(_walk_slope(30.0, slope_limit=45.0)) > CLIMB_Y
    assert foot_y(_walk_slope(30.0, slope_limit=25.0)) < 0.5


@pytest.mark.parametrize(("limit", "climbed"), [(31.0, True), (29.0, False)])
def test_slope_limit_boundary_around_30_degrees(limit: float, climbed: bool) -> None:
    char = _walk_slope(30.0, slope_limit=limit)
    assert bool(foot_y(char) > CLIMB_Y) is climbed


def test_slope_limit_zero_keeps_flat_floor_walkable() -> None:
    world = make_world()
    char = make_char(world, slope_limit=0.0)
    settle(char)
    assert char.grounded
    walk(char, (WALK, 0.0, 0.0), 1.0)
    assert char.position[0] == pytest.approx(WALK, abs=0.1)
    assert char.grounded


def test_slope_limit_zero_blocks_even_gentle_slope() -> None:
    assert foot_y(_walk_slope(10.0, slope_limit=0.0)) < 0.5


def test_slope_limit_90_climbs_steep_slope() -> None:
    assert foot_y(_walk_slope(80.0, slope_limit=90.0)) > 0.5


def test_slope_limit_90_keeps_flat_floor_walkable() -> None:
    char = make_char(make_world(), slope_limit=90.0)
    settle(char)
    assert char.grounded


# --- 3h: переход горизонталь → склон -------------------------------------------------------

def test_transition_flat_to_slope_keeps_contact() -> None:
    world = make_world()
    add_slope(world, 40.0)
    char = make_char(world)
    settle(char)
    airborne = 0
    for _ in range(150):
        char.simple_move((WALK, 0.0, 0.0))
        world.step()
        airborne += not char.grounded
    assert foot_y(char) > CLIMB_Y
    assert airborne <= 10  # измеряется отдельно: слабое подпрыгивание на стыке


def test_transition_from_slope_back_to_flat_is_smooth() -> None:
    """Спуск со склона вниз не вызывает проваливания ниже пола."""
    world = make_world()
    add_slope(world, 30.0)
    char = make_char(world)
    settle(char)
    walk(char, (WALK, 0.0, 0.0), 1.5)
    lowest = 10.0
    for _ in range(120):
        char.simple_move((-WALK, 0.0, 0.0))
        world.step()
        lowest = min(lowest, float(foot_y(char)))
    assert lowest >= -0.01


# --- 3i: step_offset + slope_limit ------------------------------------------------------------

def test_step_and_slope_limit_combination_on_30_degree_step() -> None:
    """Ступень 0.25 (меньше step_offset) на 30°-склоне не мешает подъёму."""
    world = make_world()
    add_slope(world, 30.0)
    add_step(world, 0.25, x=2.0 + 0.5, depth=0.2)
    char = make_char(world, step_offset=0.3, slope_limit=45.0)
    settle(char)
    walk(char, (WALK, 0.0, 0.0), 2.0)
    assert foot_y(char) > 0.5


def test_step_offset_does_not_override_slope_limit() -> None:
    """Большой step_offset не позволяет взойти на склон круче slope_limit."""
    char = _walk_slope(60.0, step_offset=1.0, slope_limit=45.0)
    assert foot_y(char) < 0.5
