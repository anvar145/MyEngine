"""Обход треугольников процедурных примитивов согласован с нормалями вершин."""

from __future__ import annotations

import numpy as np
import pytest

from engine.core.resources import (
    cube_mesh,
    cylinder_mesh,
    plane_mesh,
    uv_sphere_mesh,
)


@pytest.mark.parametrize(
    "factory", [uv_sphere_mesh, cube_mesh, plane_mesh, cylinder_mesh]
)
def test_winding_agrees_with_normals(factory) -> None:
    asset = factory()
    triangles = asset.indices.reshape(-1, 3)
    p0, p1, p2 = (asset.positions[triangles[:, k]] for k in range(3))
    face_normal = np.cross(p1 - p0, p2 - p0)
    vertex_normal = asset.normals[triangles].mean(axis=1)
    dot = np.einsum("ij,ij->i", face_normal, vertex_normal)
    area = np.linalg.norm(face_normal, axis=1)
    # Вырожденные треугольники (у полюсов сферы) направления не имеют.
    valid = area > 1e-6
    assert valid.any()
    assert (dot[valid] > 0).all()


def test_cylinder_caps_sit_at_half_height() -> None:
    """Крышки цилиндра лежат на y = ±h/2, нормали строго ±Y."""
    height = 2.0
    asset = cylinder_mesh(0.5, height)
    up = np.all(asset.normals == np.array([0.0, 1.0, 0.0], np.float32), axis=1)
    down = np.all(asset.normals == np.array([0.0, -1.0, 0.0], np.float32), axis=1)
    assert up.any() and down.any()
    assert np.allclose(asset.positions[up][:, 1], height / 2)
    assert np.allclose(asset.positions[down][:, 1], -height / 2)
