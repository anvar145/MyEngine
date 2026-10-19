"""Точный луч против треугольников (пикинг редактора)."""

from __future__ import annotations

import numpy as np
import pytest

from engine.core.ray_triangles import ray_mesh_distance, ray_triangles_distance

QUAD = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float)
QUAD_TRIS = np.array([[0, 1, 2], [0, 2, 3]])


def test_hit_from_both_sides_and_miss() -> None:
    front = ray_triangles_distance(np.array([0.2, 0.3, -4]), np.array([0, 0, 1.0]), QUAD, QUAD_TRIS)
    back = ray_triangles_distance(np.array([0.2, 0.3, 4]), np.array([0, 0, -1.0]), QUAD, QUAD_TRIS)
    assert front == pytest.approx(4.0)
    assert back == pytest.approx(4.0)
    assert ray_triangles_distance(np.array([3, 0, -4]), np.array([0, 0, 1.0]), QUAD, QUAD_TRIS) is None


def test_behind_origin_and_parallel_are_misses() -> None:
    assert ray_triangles_distance(np.array([0, 0, 4]), np.array([0, 0, 1.0]), QUAD, QUAD_TRIS) is None
    assert ray_triangles_distance(np.array([0, 0, -1]), np.array([1, 0, 0.0]), QUAD, QUAD_TRIS) is None
    assert ray_triangles_distance(np.zeros(3), np.array([0, 0, 1.0]), QUAD, np.zeros((0, 3), int)) is None


def test_nearest_of_several_layers() -> None:
    far = QUAD + np.array([0, 0, 5])
    vertices = np.concatenate([QUAD, far])
    tris = np.concatenate([QUAD_TRIS, QUAD_TRIS + 4])
    assert ray_triangles_distance(np.array([0, 0, -2]), np.array([0, 0, 1.0]), vertices, tris) == pytest.approx(2.0)


def test_world_matrix_translation_and_rotation() -> None:
    matrix = np.eye(4)
    matrix[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]  # поворот на 90° вокруг Y
    matrix[:3, 3] = [10, 0, 0]
    distance = ray_mesh_distance(np.zeros(3), np.array([1.0, 0, 0]), matrix, QUAD, QUAD_TRIS)
    assert distance == pytest.approx(10.0)
