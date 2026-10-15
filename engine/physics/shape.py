"""Базовые формы столкновений: куб, шар, капсула (ось капсулы — локальный Y)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from engine.physics.mathutil import as_matrix, vec3


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not value > 0.0 or not math.isfinite(value):
        raise ValueError(f"{name} должен быть > 0, получено {value}")
    return value


@dataclass(frozen=True, slots=True)
class BoxShape:
    """Параллелепипед с половинными размерами по локальным осям."""

    half_extents: tuple[float, float, float]

    def __post_init__(self) -> None:
        hx, hy, hz = self.half_extents
        object.__setattr__(
            self,
            "half_extents",
            (_positive("half_extents.x", hx), _positive("half_extents.y", hy),
             _positive("half_extents.z", hz)),
        )

    def aabb(self, position, rotation) -> tuple[np.ndarray, np.ndarray]:
        """Мировой AABB (min, max)."""
        r = as_matrix(rotation)
        extent = np.abs(r) @ np.asarray(self.half_extents)
        p = vec3(position)
        return p - extent, p + extent

    def volume(self) -> float:
        hx, hy, hz = self.half_extents
        return 8.0 * hx * hy * hz

    def local_support(self, direction) -> np.ndarray:
        """Крайняя точка в локальном направлении."""
        d = vec3(direction)
        h = np.asarray(self.half_extents)
        return np.where(d >= 0.0, h, -h)

    def inertia_tensor(self, mass: float) -> np.ndarray:
        hx, hy, hz = self.half_extents
        k = mass / 3.0
        return np.diag([k * (hy * hy + hz * hz), k * (hx * hx + hz * hz),
                        k * (hx * hx + hy * hy)])


@dataclass(frozen=True, slots=True)
class SphereShape:
    """Шар."""

    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius", _positive("radius", self.radius))

    def aabb(self, position, rotation) -> tuple[np.ndarray, np.ndarray]:
        p = vec3(position)
        return p - self.radius, p + self.radius

    def volume(self) -> float:
        return 4.0 / 3.0 * math.pi * self.radius**3

    def local_support(self, direction) -> np.ndarray:
        d = vec3(direction)
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            return np.zeros(3)
        return d / n * self.radius

    def inertia_tensor(self, mass: float) -> np.ndarray:
        return np.eye(3) * (0.4 * mass * self.radius**2)


@dataclass(frozen=True, slots=True)
class CapsuleShape:
    """Капсула: цилиндр высотой ``2 * half_height`` и две полусферы радиуса ``radius``.

    Полная высота — ``2 * (radius + half_height)``, ось — локальный Y.
    """

    radius: float
    half_height: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius", _positive("radius", self.radius))
        object.__setattr__(self, "half_height",
                           _positive("half_height", self.half_height))

    def segment(self, position, rotation) -> tuple[np.ndarray, np.ndarray]:
        """Мировые концы оси капсулы (центры полусфер)."""
        axis = as_matrix(rotation)[:, 1] * self.half_height
        p = vec3(position)
        return p - axis, p + axis

    def aabb(self, position, rotation) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.segment(position, rotation)
        return np.minimum(a, b) - self.radius, np.maximum(a, b) + self.radius

    def volume(self) -> float:
        h = 2.0 * self.half_height
        return math.pi * self.radius**2 * h + 4.0 / 3.0 * math.pi * self.radius**3

    def local_support(self, direction) -> np.ndarray:
        d = vec3(direction)
        n = float(np.linalg.norm(d))
        tip = np.array([0.0, self.half_height if d[1] >= 0.0 else -self.half_height,
                        0.0])
        if n < 1e-12:
            return tip
        return tip + d / n * self.radius

    def inertia_tensor(self, mass: float) -> np.ndarray:
        r, hh = self.radius, self.half_height
        v_cyl = math.pi * r * r * 2.0 * hh
        v_sph = 4.0 / 3.0 * math.pi * r**3
        m_cyl = mass * v_cyl / (v_cyl + v_sph)
        m_sph = mass - m_cyl
        h = 2.0 * hh
        i_axis = m_cyl * r * r / 2.0 + 2.0 / 5.0 * m_sph * r * r
        i_perp = (
            m_cyl * (3.0 * r * r + h * h) / 12.0
            + m_sph * (2.0 / 5.0 * r * r + hh * hh + 3.0 / 8.0 * r * hh * 2.0)
        )
        return np.diag([i_perp, i_axis, i_perp])


Shape = BoxShape | SphereShape | CapsuleShape
