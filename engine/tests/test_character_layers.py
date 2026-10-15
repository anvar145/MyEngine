"""Параметры ``layer`` и ``mask`` контроллера персонажа: реальный эффект."""

from __future__ import annotations

import pytest

from engine.physics.collider import Collider
from engine.physics.shape import BoxShape
from engine.tests.character_kit import WALK, make_char, make_world, settle, walk


def _wall(world, layer: int) -> None:
    world.add_static(Collider(BoxShape((0.5, 3.0, 10.0)), layer=layer), (4.0, 3.0, 0.0))


def _ahead(mask: int, wall_layer: int):
    world = make_world()
    char = make_char(world, mask=mask)
    settle(char)
    _wall(world, wall_layer)
    walk(char, (WALK, 0.0, 0.0), 2.5)
    return char


def test_mask_including_wall_layer_blocks():
    char = _ahead(mask=0b11, wall_layer=0b10)
    assert char.position[0] < 4.0


def test_mask_excluding_wall_layer_lets_character_pass():
    char = _ahead(mask=0b01, wall_layer=0b10)
    assert char.position[0] > 4.5  # прошёл сквозь стену


def test_mask_zero_falls_through_floor():
    world = make_world()
    char = make_char(world, mask=0)
    for _ in range(60):
        char.simple_move((0.0, 0.0, 0.0))
        world.step()
    assert char.position[1] < -1.0
    assert not char.grounded


def test_mask_bit_31_is_honoured():
    wall_layer = 1 << 31
    blocked = _ahead(mask=1 | (1 << 31), wall_layer=wall_layer)
    free = _ahead(mask=1, wall_layer=wall_layer)
    assert blocked.position[0] < 4.0
    assert free.position[0] > 4.5


def test_mask_applies_to_ground_probe_not_just_sweep():
    """Пол на слое вне маски: персонаж не grounded и проваливается."""
    world = make_world()
    char = make_char(world, mask=0b10)  # пол по умолчанию на слое 1
    for _ in range(30):
        char.simple_move((0.0, 0.0, 0.0))
        world.step()
    assert not char.grounded
    assert char.position[1] < 0.8


def test_character_layer_is_visible_to_raycast_masks():
    world = make_world()
    char = make_char(world, layer=0b100)
    settle(char)
    origin, down = (0.0, 5.0, 0.0), (0.0, -1.0, 0.0)
    hit = world.raycast(origin, down, 10.0, 0b100)
    assert hit is not None and hit.handle == char.handle
    miss = world.raycast(origin, down, 10.0, 0b001)
    assert miss is not None and miss.handle != char.handle  # видит только пол


def test_character_layer_excluded_from_ray_mask_is_not_hit():
    world = make_world()
    char = make_char(world, layer=0b100)
    settle(char)
    hit = world.raycast((0.0, 5.0, 0.0), (0.0, -1.0, 0.0), 10.0, 0b001)
    assert hit is not None
    assert hit.handle != char.handle
    assert hit.point[1] == pytest.approx(0.0, abs=1e-4)
