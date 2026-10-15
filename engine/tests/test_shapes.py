"""Тесты форм столкновений."""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.physics.mathutil import quat_from_axis_angle, quat_to_matrix
from engine.physics.shape import BoxShape, CapsuleShape, SphereShape


def test_box_aabb_axis_aligned() -> None:
    lo, hi = BoxShape((1, 2, 3)).aabb((1, 1, 1), np.eye(3))
    assert np.allclose(lo, (0, -1, -2)) and np.allclose(hi, (2, 3, 4))


def test_box_aabb_rotated_90_swaps_x_z() -> None:
    rot = quat_to_matrix(quat_from_axis_angle((0, 1, 0), math.pi / 2))
    lo, hi = BoxShape((1, 2, 3)).aabb((0, 0, 0), rot)
    assert np.allclose(hi - lo, (6, 4, 2), atol=1e-9)


def test_capsule_aabb_vertical_and_horizontal() -> None:
    cap = CapsuleShape(0.5, 1.0)
    lo, hi = cap.aabb((0, 0, 0), np.eye(3))
    assert np.allclose(hi - lo, (1, 3, 1))
    rot = quat_to_matrix(quat_from_axis_angle((0, 0, 1), math.pi / 2))
    lo, hi = cap.aabb((0, 0, 0), rot)
    assert np.allclose(hi - lo, (3, 1, 1), atol=1e-9)


def test_volumes() -> None:
    assert BoxShape((1, 2, 3)).volume() == pytest.approx(48)
    assert SphereShape(2).volume() == pytest.approx(4 / 3 * math.pi * 8)
    r, hh = 0.5, 1.0
    expected = math.pi * r * r * 2 * hh + 4 / 3 * math.pi * r**3
    assert CapsuleShape(r, hh).volume() == pytest.approx(expected)


def test_inertia_sphere_and_cube() -> None:
    assert np.allclose(SphereShape(2).inertia_tensor(3), np.eye(3) * 0.4 * 3 * 4)
    cube = BoxShape((1, 1, 1)).inertia_tensor(6)  # ребро 2: m/6 * a^2
    assert np.allclose(cube, np.eye(3) * 6 * 4 / 6)


def test_capsule_inertia_degenerates_to_sphere() -> None:
    inertia = CapsuleShape(1.0, 1e-9).inertia_tensor(1.0)
    assert np.allclose(np.diag(inertia), 0.4, atol=1e-6)


def test_support_points() -> None:
    assert np.allclose(BoxShape((1, 2, 3)).local_support((1, -1, 1)), (1, -2, 3))
    assert np.allclose(SphereShape(2).local_support((0, 3, 0)), (0, 2, 0))
    assert np.allclose(CapsuleShape(0.5, 1).local_support((0, -1, 0)), (0, -1.5, 0))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BoxShape((1, 0, 1)),
        lambda: SphereShape(-1),
        lambda: CapsuleShape(0.5, 0),
    ],
)
def test_validation(factory) -> None:
    with pytest.raises(ValueError):
        factory()
