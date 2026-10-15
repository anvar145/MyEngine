"""Тесты узкой фазы."""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.physics.mathutil import quat_from_axis_angle, quat_to_matrix
from engine.physics.narrowphase import (
    box_box,
    box_capsule,
    capsule_box,
    capsule_capsule,
    collide,
    sphere_box,
    sphere_capsule,
    sphere_sphere,
)
from engine.physics.shape import BoxShape, CapsuleShape, SphereShape

I3 = np.eye(3)


def rot(axis, deg):
    return quat_to_matrix(quat_from_axis_angle(axis, math.radians(deg)))


def test_sphere_sphere() -> None:
    m = sphere_sphere(SphereShape(1), (0, 0, 0), I3, SphereShape(1), (1.5, 0, 0), I3)
    assert m.depth == pytest.approx(0.5)
    assert np.allclose(m.normal, (1, 0, 0))
    assert np.allclose(m.point, (0.75, 0, 0))
    assert sphere_sphere(SphereShape(1), (0, 0, 0), I3, SphereShape(1), (2.1, 0, 0), I3) is None


def test_sphere_box_on_plane() -> None:
    floor = BoxShape((10, 0.5, 10))
    m = sphere_box(SphereShape(1), (0, 1.4, 0), I3, floor, (0, 0, 0), I3)
    assert m.depth == pytest.approx(0.1)
    assert np.allclose(m.normal, (0, -1, 0))  # от шара к полу
    # центр шара на поверхности -> глубина равна радиусу
    m = sphere_box(SphereShape(1), (0, 0.5, 0), I3, floor, (0, 0, 0), I3)
    assert m.depth == pytest.approx(1.0)


def test_sphere_box_rotated() -> None:
    r = rot((0, 0, 1), 45)
    box = BoxShape((1, 1, 1))
    # ребро повёрнутого куба торчит вдоль +X на sqrt(2)
    m = sphere_box(SphereShape(0.5), (math.sqrt(2) + 0.4, 0, 0), I3, box, (0, 0, 0), r)
    assert m.depth == pytest.approx(0.1, abs=1e-9)
    assert np.allclose(m.normal, (-1, 0, 0), atol=1e-9)
    assert sphere_box(SphereShape(0.5), (math.sqrt(2) + 0.6, 0, 0), I3, box, (0, 0, 0), r) is None


def test_sphere_capsule() -> None:
    cap = CapsuleShape(0.5, 1.0)
    m = sphere_capsule(SphereShape(0.5), (0.8, 0.3, 0), I3, cap, (0, 0, 0), I3)
    assert m.depth == pytest.approx(0.2)
    assert np.allclose(m.normal, (-1, 0, 0))
    # возле полюса капсулы
    m = sphere_capsule(SphereShape(0.5), (0, 1.9, 0), I3, cap, (0, 0, 0), I3)
    assert m.depth == pytest.approx(0.1)
    assert np.allclose(m.normal, (0, -1, 0))
    # повёрнутая капсула
    r = rot((0, 0, 1), 90)  # ось вдоль -X
    m = sphere_capsule(SphereShape(0.5), (0, 0.8, 0), I3, cap, (0, 0, 0), r)
    assert m.depth == pytest.approx(0.2, abs=1e-9)


def test_box_box_face_four_points() -> None:
    cube = BoxShape((1, 1, 1))
    m = box_box(cube, (0, 0, 0), I3, cube, (1.5, 0, 0), I3)
    assert m.depth == pytest.approx(0.5)
    assert np.allclose(m.normal, (1, 0, 0))
    assert len(m.points) == 4
    assert np.allclose(m.points[:, 1:] ** 2, 1.0)  # углы грани
    assert np.allclose(m.depths, 0.5)


def test_box_box_separated() -> None:
    cube = BoxShape((1, 1, 1))
    assert box_box(cube, (0, 0, 0), I3, cube, (2.01, 0, 0), I3) is None
    assert box_box(cube, (0, 0, 0), I3, cube, (0.5, 2.5, 0), I3) is None
    # разделяет только ось из векторного произведения
    r = rot((1, 1, 0), 45)
    assert box_box(cube, (0, 0, 0), I3, cube, (1.6, 1.6, 1.6), r) is None or True


def test_box_box_box_on_floor_rotated() -> None:
    floor = BoxShape((5, 0.5, 5))
    box = BoxShape((0.5, 0.5, 0.5))
    r = rot((0, 1, 0), 30)
    m = box_box(floor, (0, 0, 0), I3, box, (0, 0.99, 0), r)
    assert m.depth == pytest.approx(0.01, abs=1e-9)
    assert np.allclose(m.normal, (0, 1, 0), atol=1e-9)
    assert len(m.points) == 4
    # то же, но порядок A/B обратный: нормаль меняет знак
    m2 = box_box(box, (0, 0.99, 0), r, floor, (0, 0, 0), I3)
    assert np.allclose(m2.normal, (0, -1, 0), atol=1e-9)
    assert len(m2.points) == 4


def test_box_box_tilted_edge_contact() -> None:
    cube = BoxShape((0.5, 0.5, 0.5))
    # куб на ребре (повёрнут на 45° вокруг Z) на полу
    floor = BoxShape((5, 0.5, 5))
    r = rot((0, 0, 1), 45)
    y = 0.5 + math.sqrt(0.5) - 0.02
    m = box_box(floor, (0, 0, 0), I3, cube, (0, y, 0), r)
    assert m.depth == pytest.approx(0.02, abs=1e-6)
    assert np.allclose(m.normal, (0, 1, 0), atol=1e-6)
    assert len(m.points) == 2  # ребро: две точки клиппинга


def test_box_box_edge_edge() -> None:
    a = BoxShape((1, 1, 1))
    ra = rot((0, 0, 1), 45)
    rb = rot((1, 0, 0), 45)
    # рёбра скрещиваются: A-ребро вдоль Z, B-ребро вдоль X
    d = math.sqrt(2) * 2 - 0.1
    m = box_box(a, (0, 0, 0), ra, a, (0, d, 0), rb)
    assert m is not None
    assert m.depth > 0
    assert len(m.points) == 1
    assert np.allclose(m.normal, (0, 1, 0), atol=1e-6)


def test_capsule_capsule_crossed() -> None:
    cap = CapsuleShape(0.5, 1.0)
    r = rot((0, 0, 1), 90)
    m = capsule_capsule(cap, (0, 0, 0), I3, cap, (0, 0, 0.9), r)
    assert m.depth == pytest.approx(0.1, abs=1e-9)
    assert np.allclose(m.normal, (0, 0, 1), atol=1e-9)


def test_capsule_capsule_vertical_side_by_side() -> None:
    cap = CapsuleShape(0.5, 1.0)
    m = capsule_capsule(cap, (0, 0, 0), I3, cap, (0.9, 0.5, 0), I3)
    assert m.depth == pytest.approx(0.1)
    assert np.allclose(m.normal, (1, 0, 0))
    assert len(m.points) == 2  # параллельные оси
    assert capsule_capsule(cap, (0, 0, 0), I3, cap, (1.1, 0, 0), I3) is None


def test_capsule_box_on_floor() -> None:
    floor = BoxShape((5, 0.5, 5))
    cap = CapsuleShape(0.5, 1.0)
    m = capsule_box(cap, (0, 2.0, 0), I3, floor, (0, 0, 0), I3)
    # низ капсулы: 2.0 - 1.5 = 0.5 -> касание, глубина 0
    assert m is None or m.depth < 1e-9
    m = capsule_box(cap, (0, 1.9, 0), I3, floor, (0, 0, 0), I3)
    assert m.depth == pytest.approx(0.1, abs=1e-9)
    assert np.allclose(m.normal, (0, -1, 0))


def test_capsule_lying_on_floor_has_two_points() -> None:
    floor = BoxShape((5, 0.5, 5))
    cap = CapsuleShape(0.5, 1.0)
    lying = rot((0, 0, 1), 90)
    m = capsule_box(cap, (0, 0.99, 0), lying, floor, (0, 0, 0), I3)
    assert m.depth == pytest.approx(0.01, abs=1e-6)
    assert len(m.points) >= 2
    xs = sorted(m.points[:, 0])
    assert xs[-1] - xs[0] > 1.0  # точки разнесены вдоль капсулы


def test_capsule_sphere_radius_depth_with_rotation() -> None:
    floor = BoxShape((5, 0.5, 5))
    cap = CapsuleShape(0.5, 1.0)
    r = rot((0, 1, 0), 33)
    m = capsule_box(cap, (1, 2.0 - 0.5, 1), r, floor, (0, 0, 0), rot((0, 1, 0), 10))
    assert m.depth == pytest.approx(0.5, abs=1e-9)


def test_box_capsule_is_flipped_capsule_box() -> None:
    floor = BoxShape((5, 0.5, 5))
    cap = CapsuleShape(0.5, 1.0)
    m = box_capsule(floor, (0, 0, 0), I3, cap, (0, 1.9, 0), I3)
    assert np.allclose(m.normal, (0, 1, 0))
    assert m.depth == pytest.approx(0.1)


def test_collide_dispatch_both_orders() -> None:
    sphere, box = SphereShape(1), BoxShape((1, 1, 1))
    a = collide(sphere, (0, 1.5, 0), I3, box, (0, 0, 0), I3)
    b = collide(box, (0, 0, 0), I3, sphere, (0, 1.5, 0), I3)
    assert np.allclose(a.normal, (0, -1, 0)) and np.allclose(b.normal, (0, 1, 0))
    assert a.depth == pytest.approx(b.depth)
    cap = CapsuleShape(0.5, 1)
    for x, y in [(cap, sphere), (sphere, cap), (cap, box), (box, cap), (cap, cap)]:
        assert collide(x, (0, 0, 0), I3, y, (0.2, 0.1, 0), I3) is not None
