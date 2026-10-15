"""Верификация CharacterController: ``skin_width`` (зазор до поверхностей).

Персонаж останавливается не вплотную, а на расстоянии ``skin_width`` от стены
и пола; проба земли равна ``skin_width + 0.03`` (см. ``test_character_ground``).
"""

from __future__ import annotations

import numpy as np
import pytest
from character_kit import (
    WALK,
    add_step,
    add_wall,
    foot_y,
    make_char,
    make_world,
    settle,
    walk,
)


def _gap_to_wall(skin: float, **kwargs) -> float:
    world = make_world()
    face = add_wall(world)
    char = make_char(world, skin_width=skin, **kwargs)
    settle(char)
    walk(char, (WALK, 0.0, 0.0), 2.0)
    return float(face - (char.position[0] + char.radius))


# --- 7a: зазор до стены и пола --------------------------------------------------------------

@pytest.mark.parametrize("skin", [0.005, 0.01, 0.02, 0.05, 0.1])
def test_gap_to_wall_equals_skin_width(skin: float) -> None:
    assert _gap_to_wall(skin) == pytest.approx(skin, abs=2e-3)


@pytest.mark.parametrize("skin", [0.01, 0.02, 0.05, 0.1])
def test_resting_height_above_floor_equals_skin_width(skin: float) -> None:
    char = make_char(make_world(), y=1.2, skin_width=skin)
    settle(char)
    assert foot_y(char) == pytest.approx(skin, abs=2e-3)
    assert char.grounded


def test_larger_skin_means_larger_visible_gap() -> None:
    assert _gap_to_wall(0.01) < _gap_to_wall(0.05) < _gap_to_wall(0.1)


def test_gap_is_stable_while_pressing_into_wall() -> None:
    world = make_world()
    face = add_wall(world)
    char = make_char(world)
    settle(char)
    walk(char, (WALK, 0.0, 0.0), 1.5)
    gaps = []
    for _ in range(60):
        char.simple_move((WALK, 0.0, 0.0))
        world.step()
        gaps.append(float(face - (char.position[0] + char.radius)))
    assert max(gaps) - min(gaps) < 2e-3


# --- 7b: skin_width = 0 -----------------------------------------------------------------------

def test_skin_zero_raises() -> None:
    """F3: ``skin_width == 0`` — ValueError (с ним персонаж не двигался по полу)."""
    with pytest.raises(ValueError, match="skin_width"):
        make_char(make_world(), skin_width=0.0)


def test_minimal_skin_character_walks_and_does_not_tunnel() -> None:
    """Нижняя граница из адаптера (0.001) работает: идёт и упирается в стену."""
    world = make_world()
    face = add_wall(world)
    char = make_char(world, skin_width=1e-3)
    settle(char)
    assert char.grounded
    walk(char, (WALK, 0.0, 0.0), 2.0)
    assert char.position[0] + char.radius <= face + 1e-3
    assert face - (char.position[0] + char.radius) == pytest.approx(1e-3, abs=1e-3)


def test_minimal_skin_character_walks_on_flat_floor() -> None:
    char = make_char(make_world(), skin_width=1e-3)
    settle(char)
    walk(char, (WALK, 0.0, 0.0), 1.0)
    assert char.position[0] == pytest.approx(WALK, abs=0.2)


# --- 7c: skin_width > step_offset -----------------------------------------------------------------

def test_skin_larger_than_step_offset_still_stands() -> None:
    char = make_char(make_world(), y=1.5, skin_width=0.4, step_offset=0.1)
    settle(char)
    assert char.grounded
    assert foot_y(char) == pytest.approx(0.4, abs=5e-3)


def test_skin_larger_than_step_offset_still_walks_on_flat_floor() -> None:
    world = make_world()
    char = make_char(world, y=1.5, skin_width=0.4, step_offset=0.1)
    settle(char)
    walk(char, (WALK, 0.0, 0.0), 1.0)
    assert char.position[0] == pytest.approx(WALK, abs=0.2)
    assert char.grounded


def test_large_skin_does_not_block_step_within_offset() -> None:
    """Измерено: ступень 0.25 берётся и при skin 0.02, и при skin 0.2 (запас не съедается)."""
    def climbed(skin: float) -> bool:
        world = make_world()
        add_step(world, 0.25)
        char = make_char(world, y=0.9 + skin + 0.03, skin_width=skin)
        settle(char)
        walk(char, (WALK, 0.0, 0.0), 1.2)
        return bool(char.position[0] > 4.0)

    assert climbed(0.02)
    assert climbed(0.2)


# --- 7d: «парение» -------------------------------------------------------------------------------------

@pytest.mark.parametrize(("skin", "expected_hover"), [(0.02, 0.02), (0.1, 0.1), (0.3, 0.3)])
def test_hover_height_is_exactly_skin(skin: float, expected_hover: float) -> None:
    char = make_char(make_world(), y=2.0, skin_width=skin)
    settle(char, 60)
    assert float(foot_y(char)) == pytest.approx(expected_hover, abs=5e-3)


def test_character_with_large_skin_is_visibly_floating_in_default_ui_terms() -> None:
    """Зазор 0.1 м у ног — предел «приемлемого» парения: фиксируем число."""
    char = make_char(make_world(), y=2.0, skin_width=0.1)
    settle(char, 60)
    assert np.isclose(foot_y(char), 0.1, atol=5e-3)


def test_negative_skin_is_rejected() -> None:
    with pytest.raises(ValueError):
        make_char(make_world(), skin_width=-0.01)
