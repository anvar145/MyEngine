"""Скольжение вдоль стен (collide-and-slide, ``MAX_SLIDES = 4``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.physics.character import MAX_SLIDES
from engine.physics.collider import Collider
from engine.physics.mathutil import quat_from_axis_angle
from engine.physics.shape import BoxShape
from engine.tests.character_kit import (
    DT,
    WALK,
    add_step,
    add_wall,
    make_char,
    make_world,
    settle,
    walk,
)


def _standing(**kwargs):
    world = make_world()
    char = make_char(world, **kwargs)
    settle(char)
    return world, char


def _wall(world, centre, half=(0.5, 3.0, 10.0), degrees: float = 0.0) -> None:
    rot = quat_from_axis_angle((0.0, 1.0, 0.0), math.radians(degrees))
    world.add_static(Collider(BoxShape(half)), centre, rot)


def test_perpendicular_wall_stops_without_sideways_drift():
    world, char = _standing()
    face = add_wall(world, x=4.0)
    walk(char, (WALK, 0.0, 0.0), 2.0)
    assert char.position[0] == pytest.approx(face - char.radius - char.skin_width,
                                              abs=0.03)
    assert abs(char.position[2]) < 1e-3
    assert char.grounded


def test_diagonal_into_perpendicular_wall_slides_along_it():
    world, char = _standing()
    add_wall(world, x=4.0)
    walk(char, (WALK, 0.0, WALK), 2.0)  # диагональ, длина 6.36
    # вдоль стены ушли на скорости только z-компоненты
    assert char.position[2] == pytest.approx(WALK * 2.0 - 0.0, abs=1.0)
    assert char.position[2] > 6.0
    assert char.position[0] < 4.0 - 0.5 + 1e-3


def test_slide_along_wall_keeps_distance_to_it():
    world, char = _standing()
    face = add_wall(world, x=4.0)
    walk(char, (WALK, 0.0, WALK), 1.5)
    gap = face - char.position[0] - char.radius
    assert 0.0 <= gap <= char.skin_width + 0.02


def test_wall_at_45_degrees_deflects_motion_along_wall():
    """Стена повёрнута на 45°: движение +X отклоняется вдоль неё, не останавливается."""
    world, char = _standing()
    _wall(world, (4.0, 3.0, 0.0), degrees=45.0)
    walk(char, (WALK, 0.0, 0.0), 2.0)
    assert abs(char.position[2]) > 1.0  # ушёл вдоль стены
    # нормаль стены после поворота на 45° вокруг Y; сквозь плоскость не прошёл
    normal = np.array([math.cos(math.radians(45.0)), 0.0,
                       -math.sin(math.radians(45.0))])
    signed = float((char.position - np.array([4.0, 0.0, 0.0])) @ normal)
    assert -(0.5 + char.radius + 0.1) < signed < -(0.5 + char.radius) + 0.01
    assert char.grounded


def test_slide_speed_along_45_wall_is_projection_of_input():
    world, char = _standing()
    _wall(world, (4.0, 3.0, 0.0), degrees=45.0)
    walk(char, (WALK, 0.0, 0.0), 1.0)  # доходим и входим в скольжение
    before = char.position.copy()
    walk(char, (WALK, 0.0, 0.0), 1.0)
    moved = float(np.linalg.norm((char.position - before)[[0, 2]]))
    assert moved == pytest.approx(WALK * math.cos(math.radians(45.0)), abs=0.5)


def test_corner_of_two_walls_does_not_pass_or_jitter():
    world, char = _standing()
    add_wall(world, x=4.0)
    _wall(world, (0.0, 3.0, 4.0), half=(10.0, 3.0, 0.5))
    walk(char, (WALK, 0.0, WALK), 3.0)
    end = char.position.copy()
    assert end[0] < 3.5 and end[2] < 3.5
    walk(char, (WALK, 0.0, WALK), 1.0)
    assert np.linalg.norm((char.position - end)[[0, 2]]) < 0.02
    assert char.grounded


def test_corner_does_not_make_character_fall_or_jump():
    world, char = _standing()
    add_wall(world, x=4.0)
    _wall(world, (0.0, 3.0, 4.0), half=(10.0, 3.0, 0.5))
    ys = []
    for _ in range(180):
        char.simple_move((WALK, 0.0, WALK))
        world.step()
        ys.append(char.position[1])
    assert max(ys) - min(ys) < 0.01


def test_wall_plus_stair_slides_and_climbs_without_leaving_ground():
    world, char = _standing()
    add_wall(world, x=8.0)
    add_step(world, 0.2, x=3.0, depth=1.0)
    airborne_on_climb = 0
    airborne_total = 0
    for _ in range(180):
        char.simple_move((WALK, 0.0, 0.5 * WALK))
        world.step()
        if not char.grounded:
            airborne_total += 1
            # подъём и ходьба по верху уступа (x < 5): в воздухе быть не должен
            airborne_on_climb += 1 if char.position[0] < 5.0 else 0
    assert airborne_on_climb == 0
    # сход с кромки уступа (0.2 м): падение 8 кадров, прилипания к земле нет —
    # нормаль на ребре непроходима (см. docs/CORE.md, game feel)
    assert airborne_total <= 10
    assert char.position[1] == pytest.approx(0.92, abs=0.02)
    assert char.position[0] < 7.5
    assert char.grounded


def test_three_walls_funnel_character_stops_inside():
    """Три стены воронкой (3 контакта за кадр): внутрь не просачивается."""
    world, char = _standing()
    add_wall(world, x=4.0)
    _wall(world, (0.0, 3.0, 3.0), half=(10.0, 3.0, 0.5))
    _wall(world, (0.0, 3.0, -3.0), half=(10.0, 3.0, 0.5))
    for direction in ((WALK, 0.0, 0.0), (WALK, 0.0, WALK), (WALK, 0.0, -WALK)):
        walk(char, direction, 1.5)
        assert char.position[0] < 3.5 + 1e-3
        assert abs(char.position[2]) < 2.5 + 1e-3


def test_acute_wedge_character_does_not_tunnel():
    """Клин из двух стен под острым углом: за MAX_SLIDES итераций не пролезть."""
    world, char = _standing(x=-1.0)
    _wall(world, (3.0, 3.0, 2.0), degrees=20.0, half=(0.5, 3.0, 6.0))
    _wall(world, (3.0, 3.0, -2.0), degrees=-20.0, half=(0.5, 3.0, 6.0))
    for _ in range(240):
        char.simple_move((WALK, 0.0, 0.0))
        world.step()
        assert char.position[0] < 6.0
    assert char.grounded


def test_max_slides_is_four_and_remainder_is_dropped():
    assert MAX_SLIDES == 4
    # воронка из пяти поверхностей не бесконечна: move() возвращается за конечное время
    world, char = _standing()
    for ang in (0.0, 30.0, -30.0, 60.0, -60.0):
        _wall(world, (3.0, 3.0, 0.0), degrees=ang, half=(0.2, 3.0, 6.0))
    for _ in range(60):
        char.simple_move((WALK, 0.0, 0.0))
        world.step()
    assert np.isfinite(char.position).all()
    assert char.position[0] < 5.0


def test_thin_wall_is_not_tunneled_at_walk_speed():
    world, char = _standing()
    add_wall(world, x=4.0, half_thickness=0.05)
    walk(char, (WALK, 0.0, 0.0), 2.0)
    assert char.position[0] < 4.0


@pytest.mark.parametrize("speed", [4.5, 20.0, 100.0, 400.0])
def test_thin_wall_is_not_tunneled_at_high_speed(speed):
    """Тонкая стена 0.1 м не пробивается при любой скорости (свип, а не дискретка)."""
    world, char = _standing()
    add_wall(world, x=4.0, half_thickness=0.05)
    for _ in range(30):
        char.simple_move((speed, 0.0, 0.0))
        world.step()
    assert char.position[0] < 4.0


def test_slide_is_deterministic():
    runs = []
    for _ in range(2):
        world, char = _standing()
        _wall(world, (4.0, 3.0, 0.0), degrees=45.0)
        walk(char, (WALK, 0.0, 0.3 * WALK), 2.0)
        runs.append(char.position.copy())
    assert np.array_equal(runs[0], runs[1])


def test_dt_constant_is_one_sixtieth():
    assert pytest.approx(1.0 / 60.0) == DT
