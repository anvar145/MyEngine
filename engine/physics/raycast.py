"""Пересечение луча с формами (аналитически, в локальных осях формы)."""

from __future__ import annotations

import math

import numpy as np

from engine.physics.mathutil import as_matrix
from engine.physics.shape import BoxShape, CapsuleShape, Shape, SphereShape


def ray_sphere(
    origin: np.ndarray, direction: np.ndarray, center: np.ndarray, radius: float
) -> tuple[float, np.ndarray] | None:
    """Луч (единичное ``direction``) против шара: ``(t, нормаль)`` или ``None``.

    Если начало луча внутри шара, возвращается ``t = 0``.
    """
    oc = origin - center
    b = float(oc @ direction)
    c = float(oc @ oc) - radius * radius
    if c <= 0.0:
        return 0.0, -direction
    if b > 0.0:
        return None
    disc = b * b - c
    if disc < 0.0:
        return None
    t = -b - math.sqrt(disc)
    point = origin + direction * t
    return t, (point - center) / radius


def ray_box(
    origin: np.ndarray,
    direction: np.ndarray,
    half_extents: np.ndarray,
    position: np.ndarray,
    rotation: np.ndarray,
) -> tuple[float, np.ndarray] | None:
    """Луч против ориентированного бокса (slab-метод)."""
    local_o = rotation.T @ (origin - position)
    local_d = rotation.T @ direction
    t_min, t_max = 0.0, math.inf
    axis_hit = -1
    sign = 1.0
    inside = bool(np.all(np.abs(local_o) <= half_extents))
    if inside:
        return 0.0, -direction
    for i in range(3):
        d = local_d[i]
        if abs(d) < 1e-12:
            if abs(local_o[i]) > half_extents[i]:
                return None
            continue
        t1 = (-half_extents[i] - local_o[i]) / d
        t2 = (half_extents[i] - local_o[i]) / d
        s = -1.0
        if t1 > t2:
            t1, t2 = t2, t1
            s = 1.0
        if t1 > t_min:
            t_min, axis_hit, sign = t1, i, s
        t_max = min(t_max, t2)
        if t_min > t_max:
            return None
    if axis_hit < 0:
        return None
    normal = np.zeros(3)
    normal[axis_hit] = sign
    return t_min, rotation @ normal


def ray_capsule(
    origin: np.ndarray,
    direction: np.ndarray,
    shape: CapsuleShape,
    position: np.ndarray,
    rotation: np.ndarray,
) -> tuple[float, np.ndarray] | None:
    """Луч против капсулы (цилиндр + две полусферы), ось — локальный Y."""
    o = rotation.T @ (origin - position)
    d = rotation.T @ direction
    r, hh = shape.radius, shape.half_height
    best: tuple[float, np.ndarray] | None = None

    # Боковая поверхность бесконечного цилиндра вокруг Y.
    a = d[0] * d[0] + d[2] * d[2]
    if a > 1e-12:
        b = o[0] * d[0] + o[2] * d[2]
        c = o[0] * o[0] + o[2] * o[2] - r * r
        disc = b * b - a * c
        if disc >= 0.0:
            t = (-b - math.sqrt(disc)) / a
            if t >= 0.0:
                y = o[1] + d[1] * t
                if -hh <= y <= hh:
                    p = o + d * t
                    n = np.array([p[0], 0.0, p[2]]) / r
                    best = (t, n)
            elif c < 0.0 and -hh <= o[1] <= hh:
                return 0.0, -direction
    for sign in (-1.0, 1.0):
        hit = ray_sphere(o, d, np.array([0.0, sign * hh, 0.0]), r)
        if hit is not None and (best is None or hit[0] < best[0]):
            best = hit
    if best is None:
        return None
    return best[0], rotation @ best[1]


def ray_shape(
    origin: np.ndarray,
    direction: np.ndarray,
    shape: Shape,
    position: np.ndarray,
    rotation,
) -> tuple[float, np.ndarray] | None:
    """Луч против любой поддерживаемой формы: ``(t, мировая нормаль)``."""
    rot = as_matrix(rotation)
    if isinstance(shape, SphereShape):
        return ray_sphere(origin, direction, position, shape.radius)
    if isinstance(shape, BoxShape):
        return ray_box(origin, direction, np.asarray(shape.half_extents), position, rot)
    return ray_capsule(origin, direction, shape, position, rot)


def ray_aabb(
    origin: np.ndarray, direction: np.ndarray, lo: np.ndarray, hi: np.ndarray,
    max_distance: float,
) -> bool:
    """Грубая проверка луча против AABB (для отсечения до точного теста)."""
    t_min, t_max = 0.0, max_distance
    for i in range(3):
        d = direction[i]
        if abs(d) < 1e-12:
            if origin[i] < lo[i] or origin[i] > hi[i]:
                return False
            continue
        t1 = (lo[i] - origin[i]) / d
        t2 = (hi[i] - origin[i]) / d
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return False
    return True
