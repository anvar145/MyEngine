"""Точное пересечение луча с треугольниками меша (чистый numpy, без GL).

Используется пикингом редактора: AABB служит только грубым отсевом, а объект
выбирается по ближайшему треугольнику под курсором. Тест двусторонний: из
комнаты-уровня видны внутренние грани стен, и они тоже должны попадать.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ray_mesh_distance", "ray_triangles_distance"]

_EPS = 1e-12


def ray_triangles_distance(
    origin: np.ndarray,
    direction: np.ndarray,
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> float | None:
    """Расстояние до ближайшего треугольника вдоль луча (Мёллер — Трумбор).

    Args:
        origin: Начало луча ``(3,)``.
        direction: Направление ``(3,)`` (не обязано быть единичным: результат в
            долях ``direction``).
        vertices: Вершины ``(n, 3)``.
        triangles: Индексы ``(m, 3)``.

    Returns:
        Параметр ``t >= 0`` ближайшего попадания или ``None``.
    """
    if len(triangles) == 0:
        return None
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(triangles, dtype=np.int64)]
    a = tri[:, 0]
    e1 = tri[:, 1] - a
    e2 = tri[:, 2] - a
    p = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > _EPS
    inv = np.zeros_like(det)
    inv[ok] = 1.0 / det[ok]
    s = o - a
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    v = (q @ d) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t >= 0.0)
    if not hit.any():
        return None
    return float(t[hit].min())


def ray_mesh_distance(
    origin: np.ndarray,
    direction: np.ndarray,
    world_matrix: np.ndarray,
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> float | None:
    """Мировое расстояние до меша объекта с матрицей ``world_matrix``.

    Луч переводится в локальные оси объекта той же матрицей без нормализации,
    поэтому параметр ``t`` остаётся в мировых единицах (если ``direction``
    единичный) при любом масштабе и повороте.
    """
    matrix = np.asarray(world_matrix, dtype=np.float64).reshape(4, 4)
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None
    o = inverse @ np.append(np.asarray(origin, dtype=np.float64), 1.0)
    d = inverse @ np.append(np.asarray(direction, dtype=np.float64), 0.0)
    return ray_triangles_distance(o[:3], d[:3], vertices, triangles)
