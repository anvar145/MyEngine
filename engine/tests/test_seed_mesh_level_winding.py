"""Геометрия демо-уровня: все треугольники смотрят наружу (back-face culling)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import seed_mesh_level  # noqa: E402


def _check_outward(vertices, normals, indices) -> None:
    centroid = vertices.mean(axis=0)
    for tri in indices.reshape(-1, 3):
        a, b, c = vertices[tri]
        face = np.cross(b - a, c - a)
        assert np.dot(face, normals[tri[0]]) > 0
        assert np.dot(face, (a + b + c) / 3 - centroid) > 0


def test_wedge_faces_point_outward() -> None:
    _check_outward(*seed_mesh_level._wedge(-8.0, -2.0, 2.0, 5.0, 1.2))
