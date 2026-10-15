"""Узкая фаза «капсула/бокс против треугольника» (A11)."""

from __future__ import annotations

import math

import pytest

from engine.physics.mesh_collide import box_triangle_overlap, capsule_triangle

A, B, C = (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 0.0, 4.0)
# нормаль этого треугольника: (b-a) x (c-a) = (0, -16, 0) -> вниз; берём обратный порядок
TRI = (A, C, B)  # нормаль +Y


def test_sphere_resting_on_floor_pushes_up() -> None:
    p = (1.0, 0.4, 1.0)
    hit = capsule_triangle(p, p, 0.5, *TRI)
    assert hit is not None
    nx, ny, nz, depth, px, py, pz = hit
    assert (nx, ny, nz) == pytest.approx((0.0, 1.0, 0.0))
    assert depth == pytest.approx(0.1)
    assert (px, py, pz) == pytest.approx((1.0, 0.0, 1.0))


def test_sphere_above_misses() -> None:
    p = (1.0, 0.6, 1.0)
    assert capsule_triangle(p, p, 0.5, *TRI) is None


def test_capsule_vertical_standing_on_floor() -> None:
    hit = capsule_triangle((1.0, 0.45, 1.0), (1.0, 1.55, 1.0), 0.5, *TRI)
    assert hit is not None
    assert hit[:3] == pytest.approx((0.0, 1.0, 0.0))
    assert hit[3] == pytest.approx(0.05)


def test_edge_contact_normal_points_away_from_edge() -> None:
    # сфера снаружи у ребра a-b (z = 0): ближайшая точка на ребре
    p = (2.0, -0.1, -0.3)
    hit = capsule_triangle(p, p, 0.5, *TRI)
    assert hit is not None
    n = hit[:3]
    assert math.sqrt(sum(c * c for c in n)) == pytest.approx(1.0)
    assert n[2] < 0.0 and hit[3] == pytest.approx(0.5 - math.hypot(0.1, 0.3))


def test_vertex_contact() -> None:
    p = (-0.3, 0.0, -0.3)
    hit = capsule_triangle(p, p, 0.5, *TRI)
    assert hit is not None
    assert hit[3] == pytest.approx(0.5 - math.hypot(0.3, 0.3))
    assert hit[0] < 0.0 and hit[2] < 0.0


def test_axis_piercing_face_pushes_out_by_smaller_depth() -> None:
    # ось капсулы пронзает грань; нижний конец глубже -> выталкивание вверх
    hit = capsule_triangle((1.0, -0.2, 1.0), (1.0, 1.0, 1.0), 0.3, *TRI)
    assert hit is not None
    assert hit[:3] == pytest.approx((0.0, 1.0, 0.0))
    assert hit[3] == pytest.approx(0.5)
    # верхний конец глубже -> вниз
    hit = capsule_triangle((1.0, -1.0, 1.0), (1.0, 0.2, 1.0), 0.3, *TRI)
    assert hit[:3] == pytest.approx((0.0, -1.0, 0.0))


def test_far_capsule_and_degenerate_triangle() -> None:
    assert capsule_triangle((10.0, 0.0, 10.0), (10.0, 1.0, 10.0), 0.5, *TRI) is None
    assert capsule_triangle((0, 0, 0), (0, 1, 0), 0.5, A, A, B) is None


AXES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def test_box_overlap_floor() -> None:
    half = (0.5, 0.5, 0.5)
    assert box_triangle_overlap((1.0, 0.4, 1.0), AXES, half, *TRI)
    assert not box_triangle_overlap((1.0, 0.6, 1.0), AXES, half, *TRI)
    assert not box_triangle_overlap((6.0, 0.0, 6.0), AXES, half, *TRI)


def test_box_rotated_corner_touches_edge() -> None:
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    axes = ((c, 0.0, -s), (0.0, 1.0, 0.0), (s, 0.0, c))  # поворот вокруг Y
    half = (0.5, 0.5, 0.5)
    # угол бокса смотрит вдоль ребра a-b (z = 0), выступает на 0.707
    assert box_triangle_overlap((2.0, 0.0, -0.6), axes, half, *TRI)
    assert not box_triangle_overlap((2.0, 0.0, -0.8), axes, half, *TRI)
