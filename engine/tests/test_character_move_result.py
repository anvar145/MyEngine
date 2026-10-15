"""Данные, которые возвращает ``CharacterController.move``.

Результат — только ``ndarray`` фактического смещения; структуры ``MoveResult``
(список ударов, флаги столкновений) нет — это зафиксировано тестами.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.tests.character_kit import (
    add_slope,
    add_wall,
    make_char,
    make_world,
    settle,
)

_Y_NOISE = 1e-3  # шум по Y: float32 позы + точность бисекции свипа (≈0.3 мм)


def _standing(**kwargs):
    world = make_world()
    char = make_char(world, **kwargs)
    settle(char)
    return world, char


def test_move_returns_ndarray_of_three_floats():
    _world, char = _standing()
    result = char.move((0.1, 0.0, 0.0))
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert result.dtype == np.float64


def test_free_move_returns_requested_motion():
    _world, char = _standing()
    result = char.move((0.1, 0.0, -0.05))
    assert result[0] == pytest.approx(0.1, abs=1e-6)
    assert result[1] == pytest.approx(0.0, abs=_Y_NOISE)
    assert result[2] == pytest.approx(-0.05, abs=1e-6)


def test_zero_motion_returns_zero_vector():
    _world, char = _standing()
    # на земле прилипание даёт шум по Y порядка float32 (≈7e-6), не дрейф
    assert np.allclose(char.move((0.0, 0.0, 0.0)), 0.0, atol=_Y_NOISE)


def test_result_equals_position_delta():
    _world, char = _standing()
    before = char.position.copy()
    result = char.move((0.2, 0.0, 0.1))
    assert np.allclose(char.position - before, result)


def test_result_is_independent_copy_of_position():
    _world, char = _standing()
    result = char.move((0.1, 0.0, 0.0))
    snapshot = result.copy()
    char.move((0.1, 0.0, 0.0))
    assert np.array_equal(result, snapshot)


def test_motion_into_wall_returns_less_than_requested():
    world, char = _standing()
    face = add_wall(world, x=1.0)
    char.move((0.55, 0.0, 0.0))  # почти до стены
    result = char.move((2.0, 0.0, 0.0))
    # ближняя грань стены face; центр останавливается на radius + skin от неё
    assert result[0] < 0.5
    assert char.position[0] <= face - char.radius + 1e-6
    assert result[0] >= 0.0


def test_motion_into_wall_has_no_perpendicular_leak():
    world, char = _standing()
    add_wall(world, x=1.0)
    for _ in range(10):
        result = char.move((0.5, 0.0, 0.0))
    assert abs(result[2]) < 1e-6
    assert abs(result[1]) < 1e-6


def test_diagonal_into_wall_returns_slide_component():
    world, char = _standing()
    add_wall(world, x=1.0)
    for _ in range(5):
        char.move((0.3, 0.0, 0.0))
    result = char.move((0.3, 0.0, 0.3))
    assert result[2] == pytest.approx(0.3, abs=0.02)
    assert result[0] < 0.05


def test_vertical_free_fall_returns_full_dy():
    world = make_world()
    char = make_char(world, y=3.0)
    result = char.move((0.0, -0.1, 0.0))
    assert result[1] == pytest.approx(-0.1, abs=1e-6)
    assert not char.grounded


def test_vertical_motion_into_floor_is_truncated():
    _world, char = _standing()
    result = char.move((0.0, -1.0, 0.0))
    assert abs(result[1]) < 0.05
    assert char.grounded


def test_upward_motion_into_ceiling_is_truncated():
    world, char = _standing()
    from engine.physics.collider import Collider
    from engine.physics.shape import BoxShape

    world.add_static(Collider(BoxShape((5.0, 0.5, 5.0))), (0.0, 2.7, 0.0))
    # макушка на 1.82, потолок снизу на 2.2: свободно 0.38 минус skin 0.02
    result = char.move((0.0, 2.0, 0.0))
    assert result[1] == pytest.approx(0.36, abs=0.01)


def test_ground_normal_is_up_on_flat_floor():
    _world, char = _standing()
    assert char.grounded
    assert np.allclose(char.ground_normal, (0.0, 1.0, 0.0), atol=1e-6)


def test_ground_normal_follows_walkable_slope():
    world = make_world()
    add_slope(world, 20.0)
    char = make_char(world, x=5.0, y=3.0)
    for _ in range(120):
        char.simple_move((0.0, 0.0, 0.0))
        world.step()
    assert char.grounded
    expected = (-math.sin(math.radians(20.0)), math.cos(math.radians(20.0)), 0.0)
    assert np.allclose(char.ground_normal, expected, atol=0.02)


def test_ground_normal_reset_to_up_in_air():
    world = make_world()
    char = make_char(world, y=5.0)
    char.move((0.0, -0.05, 0.0))
    assert not char.grounded
    assert np.allclose(char.ground_normal, (0.0, 1.0, 0.0))


def test_simple_move_returns_actual_displacement_including_gravity():
    world = make_world()
    char = make_char(world, y=5.0)
    result = char.simple_move((1.2, 0.0, 0.0))
    assert result[0] == pytest.approx(1.2 / 60.0, abs=1e-6)
    assert result[1] < 0.0


def test_move_has_no_hit_list_or_collision_flags():
    """Фиксируем отсутствие структуры результата: нужно — расширять отдельно."""
    _world, char = _standing()
    result = char.move((0.1, 0.0, 0.0))
    assert not hasattr(result, "hits")
    for name in ("hits", "last_hits", "collisions", "collision_flags", "last_move"):
        assert not hasattr(char, name), name
