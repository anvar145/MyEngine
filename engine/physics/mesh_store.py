"""Статические mesh-коллайдеры мира: BVH + поза + слои (только запросы).

Mesh-коллайдер не входит в общий список коллайдеров, широкую фазу и солвер:
его видят только ``raycast``/``shapecast``/``overlap`` мира (поэтому у него нет
тела, событий столкновений и динамики - динамическое тело сквозь него
проваливается). Геометрия хранится в локальном пространстве меша с уже
применённым масштабом объекта; поза (смещение + поворот) задаётся отдельно и
меняется без перестройки BVH.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.physics.bvh import Bvh
from engine.physics.material import PhysicsMaterial
from engine.physics.mathutil import quat_to_matrix, vec3
from engine.physics.mesh_collide import box_triangle_overlap, capsule_triangle
from engine.physics.shape import BoxShape, CapsuleShape

__all__ = ["MeshCollider"]


def _swept_axis(shape, rot: np.ndarray) -> tuple[float, float, float]:
    """Половина оси капсулы (для шара - ноль) в мировых координатах."""
    if isinstance(shape, CapsuleShape):
        axis = rot[:, 1] * shape.half_height
        return float(axis[0]), float(axis[1]), float(axis[2])
    return 0.0, 0.0, 0.0


@dataclass(eq=False)
class MeshCollider:
    """Статический mesh-коллайдер (запись :class:`MeshColliderStore`).

    Attributes:
        handle: Handle в мире (общий счётчик с телами, не пересекается с ними).
        bvh: BVH в локальном пространстве меша (масштаб уже применён).
        position: Мировое смещение.
        rotation: Мировой поворот, кватернион ``xyzw``.
        layer: Слой коллайдера.
        mask: Слои, с которыми он взаимодействует.
        is_trigger: Триггер: виден запросам только при ``include_triggers``.
        material: Физический материал.
        user_data: Произвольные данные владельца.
        version: Растёт при каждой смене позы (для отпечатка области).
    """

    handle: int
    bvh: Bvh
    position: np.ndarray
    rotation: np.ndarray
    layer: int = 1
    mask: int = 0xFFFFFFFF
    is_trigger: bool = False
    material: PhysicsMaterial = field(default_factory=PhysicsMaterial)
    user_data: Any = None
    version: int = 0
    matrix: np.ndarray = field(init=False, repr=False)
    world_aabb: tuple[np.ndarray, np.ndarray] = field(init=False, repr=False)
    pose_is_identity: bool = field(init=False, repr=False)
    _aabb_t: tuple = field(init=False, repr=False)
    _pos_t: tuple = field(init=False, repr=False)
    _rows_t: tuple = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.set_pose(self.position, self.rotation)

    def set_pose(self, position, rotation) -> None:
        """Задаёт позу и пересчитывает мировой AABB."""
        self.position = vec3(position).copy()
        q = np.asarray(rotation, dtype=np.float64).reshape(4)
        norm = float(np.linalg.norm(q))
        self.rotation = q / norm if norm > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])
        self.matrix = quat_to_matrix(self.rotation)
        lo, hi = self.bvh.bounds
        center = (lo + hi) * 0.5
        half = (hi - lo) * 0.5
        wc = self.matrix @ center + self.position
        wh = np.abs(self.matrix) @ half
        self.world_aabb = (wc - wh, wc + wh)
        # скалярные копии позы для быстрого raycast без numpy
        self._aabb_t = (tuple((wc - wh).tolist()), tuple((wc + wh).tolist()))
        self._pos_t = tuple(self.position.tolist())
        self._rows_t = tuple(tuple(row) for row in self.matrix.tolist())
        self.pose_is_identity = bool(np.array_equal(self.matrix, np.eye(3)))
        self.version += 1

    def key_bytes(self) -> bytes:
        """Байты для отпечатка области: handle, слои, поза и версия."""
        return b"".join((
            np.int64(self.handle).tobytes(), np.int64(self.layer).tobytes(),
            np.int64(self.mask).tobytes(), np.int64(self.is_trigger).tobytes(),
            self.position.tobytes(), self.rotation.tobytes(),
            np.int64(self.version).tobytes(),
        ))

    # ----------------------------------------------------------- координаты

    def to_local(self, point) -> np.ndarray:
        """Мировая точка -> локальное пространство меша."""
        return self.matrix.T @ (np.asarray(point, dtype=np.float64) - self.position)

    def _local_aabb(self, lo: np.ndarray, hi: np.ndarray):
        center = self.matrix.T @ ((lo + hi) * 0.5 - self.position)
        half = np.abs(self.matrix.T) @ ((hi - lo) * 0.5)
        return center - half, center + half

    def _triangles(self, ids: list[int]):
        verts = self.bvh.vertices
        tris = self.bvh.triangles
        return [tuple(map(tuple, verts[tris[i]].tolist())) for i in ids]

    # ----------------------------------------------------------------- запросы

    def raycast(self, origin, direction, max_distance: float):
        """Луч (мировой) -> ``(расстояние, нормаль в мире)`` или ``None``.

        Направление нормируется. Нулевое направление - промах.
        """
        ox, oy, oz = (float(v) for v in origin)
        dx, dy, dz = (float(v) for v in direction)
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-12:
            return None
        return self.raycast_floats(
            ox, oy, oz, dx / length, dy / length, dz / length, max_distance
        )

    def raycast_floats(
        self, ox: float, oy: float, oz: float,
        dx: float, dy: float, dz: float, max_distance: float,
    ):
        """Быстрый :meth:`raycast`: числа float, нормированное направление.

        Без numpy на горячем пути: отсечение по мировому AABB и перевод луча в
        локальное пространство делаются на скалярах (при единичном повороте
        перевод - просто сдвиг). Нормаль возвращается кортежем ``(x, y, z)``.
        """
        lo, hi = self._aabb_t
        t_min, t_max = 0.0, max_distance
        for o, d, a, b in ((ox, dx, lo[0], hi[0]), (oy, dy, lo[1], hi[1]),
                           (oz, dz, lo[2], hi[2])):
            if -1e-12 < d < 1e-12:
                if o < a or o > b:
                    return None
                continue
            t1 = (a - o) / d
            t2 = (b - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            if t1 > t_min:
                t_min = t1
            if t2 < t_max:
                t_max = t2
            if t_min > t_max:
                return None
        px, py, pz = self._pos_t
        if self.pose_is_identity:
            result = self.bvh.raycast_unit(
                ox - px, oy - py, oz - pz, dx, dy, dz, max_distance
            )
            if result is None:
                return None
            return result.distance, result.normal
        (m00, m01, m02), (m10, m11, m12), (m20, m21, m22) = self._rows_t
        ox, oy, oz = ox - px, oy - py, oz - pz
        # локальные = M^T * мировые
        result = self.bvh.raycast_unit(
            m00 * ox + m10 * oy + m20 * oz,
            m01 * ox + m11 * oy + m21 * oz,
            m02 * ox + m12 * oy + m22 * oz,
            m00 * dx + m10 * dy + m20 * dz,
            m01 * dx + m11 * dy + m21 * dz,
            m02 * dx + m12 * dy + m22 * dz,
            max_distance,
        )
        if result is None:
            return None
        nx, ny, nz = result.normal
        return result.distance, (
            m00 * nx + m01 * ny + m02 * nz,
            m10 * nx + m11 * ny + m12 * nz,
            m20 * nx + m21 * ny + m22 * nz,
        )

    def raycast_batch(self, origins: np.ndarray, directions: np.ndarray,
                      max_distance: float):
        """Пакет мировых лучей с нормированными направлениями.

        Лучи, не задевающие мировой AABB, отсекаются векторно; остальные идут в
        :meth:`Bvh.raycast_batch`.

        Returns:
            ``(distance (n,), normal (n, 3))``: расстояние ``inf`` и нормаль
            ``nan`` у промахов; нормаль повёрнута к началу луча, в мире.
        """
        count = len(origins)
        dist = np.full(count, np.inf)
        normal = np.full((count, 3), np.nan)
        if not count:
            return dist, normal
        lo, hi = self.world_aabb
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - origins) / directions
            t2 = (hi - origins) / directions
        parallel = np.abs(directions) < 1e-12
        inside = (origins >= lo) & (origins <= hi)
        t_near = np.where(parallel, -np.inf, np.minimum(t1, t2)).max(axis=1)
        t_far = np.where(parallel, np.inf, np.maximum(t1, t2)).min(axis=1)
        keep = (
            (np.where(parallel, inside, True).all(axis=1))
            & (np.maximum(t_near, 0.0) <= np.minimum(t_far, max_distance))
        )
        idx = np.flatnonzero(keep)
        if not len(idx):
            return dist, normal
        m = self.matrix
        local_o = (origins[idx] - self.position) @ m
        local_d = directions[idx] @ m
        hit_t, hit_tri, _ = self.bvh.raycast_batch(local_o, local_d, max_distance)
        found = hit_tri >= 0
        if not found.any():
            return dist, normal
        sel = idx[found]
        dist[sel] = hit_t[found]
        corners = self.bvh.vertices[self.bvh.triangles[hit_tri[found]]]
        n = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        n /= np.linalg.norm(n, axis=1)[:, None]
        flip = np.einsum("ij,ij->i", n, local_d[found]) > 0.0
        n[flip] = -n[flip]
        normal[sel] = n @ m.T
        return dist, normal

    def contacts(self, shape, position, rot: np.ndarray):
        """Контакты шара/капсулы с треугольниками.

        Формат: ``[(nx, ny, nz, depth, px, py, pz)]``.

        Нормали и точки - в мировых координатах (нормаль от поверхности к форме).
        """
        if isinstance(shape, BoxShape):
            return []
        p = vec3(position)
        ax = np.array(_swept_axis(shape, rot))
        radius = shape.radius
        lo = np.minimum(p - ax, p + ax) - radius
        hi = np.maximum(p - ax, p + ax) + radius
        wlo, whi = self.world_aabb
        if (lo > whi).any() or (hi < wlo).any():
            return []
        llo, lhi = self._local_aabb(lo, hi)
        ids = self.bvh.query_aabb(llo, lhi)
        if not ids:
            return []
        a0 = tuple(self.to_local(p - ax).tolist())
        a1 = tuple(self.to_local(p + ax).tolist())
        out = []
        matrix = self.matrix
        for tri in self._triangles(ids):
            hit = capsule_triangle(a0, a1, radius, *tri)
            if hit is not None:
                n = matrix @ np.array(hit[:3])
                pt = matrix @ np.array(hit[4:7]) + self.position
                out.append((float(n[0]), float(n[1]), float(n[2]), hit[3],
                            float(pt[0]), float(pt[1]), float(pt[2])))
        return out

    def overlaps(self, shape, position, rot: np.ndarray) -> bool:
        """Пересекает ли форма меш."""
        if isinstance(shape, BoxShape):
            return bool(self._box_hits(shape, vec3(position), rot))
        return bool(self.contacts(shape, position, rot))

    def _box_prepare(self, shape: BoxShape, pos: np.ndarray, rot: np.ndarray):
        half = np.array(shape.half_extents)
        reach = np.abs(rot) @ half
        wlo, whi = self.world_aabb
        if ((pos - reach) > whi).any() or ((pos + reach) < wlo).any():
            return None
        return pos - reach, pos + reach

    def _box_hits(self, shape: BoxShape, pos: np.ndarray, rot: np.ndarray):
        box = self._box_prepare(shape, pos, rot)
        if box is None:
            return []
        llo, lhi = self._local_aabb(*box)
        ids = self.bvh.query_aabb(llo, lhi)
        local = self.matrix.T @ rot
        center = tuple(self.to_local(pos).tolist())
        axes = tuple(tuple(local[:, i].tolist()) for i in range(3))
        return [
            tri for tri in self._triangles(ids)
            if box_triangle_overlap(center, axes, shape.half_extents, *tri)
        ]

    def prepare_sweep(
        self, shape, position, rot: np.ndarray, direction: np.ndarray,
        limit: float, lo0: np.ndarray, hi0: np.ndarray,
    ) -> tuple[Callable[[float], bool], Callable[[float], tuple]] | None:
        """Готовит свип формы вдоль ``direction`` (единичный) на длину ``limit``.

        Returns:
            ``None``, если вдоль пути нет ни одного треугольника, иначе пара
            ``(overlap(t) -> bool, contact(t) -> (t, нормаль, точка))``.
        """
        wlo, whi = self.world_aabb
        end = direction * limit
        slo = np.minimum(lo0, lo0 + end)
        shi = np.maximum(hi0, hi0 + end)
        if (slo > whi).any() or (shi < wlo).any():
            return None
        llo, lhi = self._local_aabb(slo, shi)
        ids = self.bvh.query_aabb(llo, lhi)
        if not ids:
            return None
        pos = vec3(position)
        dloc = self.matrix.T @ direction
        dx, dy, dz = dloc.tolist()
        matrix = self.matrix
        origin = self.position
        if isinstance(shape, BoxShape):
            return self._box_sweep(shape, pos, rot, direction, self._triangles(ids))
        ax = _swept_axis(shape, rot)
        c0 = self.to_local(pos).tolist()
        axl = (self.matrix.T @ np.array(ax)).tolist()
        radius = shape.radius
        # Точный AABB заметаемого объёма в локальных координатах (капсула
        # движется линейно): треугольники вне него в свипе не участвуют.
        cen0 = np.array(c0)
        cen1 = cen0 + dloc * limit
        axv = np.array(axl)
        box_lo = np.minimum(cen0, cen1) - np.abs(axv) - radius
        box_hi = np.maximum(cen0, cen1) + np.abs(axv) + radius
        corners = self.bvh.vertices[self.bvh.triangles[ids]]
        near = ((corners.max(axis=1) >= box_lo) & (corners.min(axis=1) <= box_hi)
                ).all(axis=1)
        if not near.any():
            return None
        corners = corners[near]
        tris = [tuple(map(tuple, tri)) for tri in corners.tolist()]
        # AABB треугольников, раздутые на радиус: дешёвый отсев по позиции t
        t_lo = (corners.min(axis=1) - radius).tolist()
        t_hi = (corners.max(axis=1) + radius).tolist()
        near_tris = list(zip(tris, t_lo, t_hi, strict=True))

        def ends(t):
            cx, cy, cz = c0[0] + dx * t, c0[1] + dy * t, c0[2] + dz * t
            return ((cx - axl[0], cy - axl[1], cz - axl[2]),
                    (cx + axl[0], cy + axl[1], cz + axl[2]))

        def candidates(p0, p1):
            x0, y0, z0 = p0
            x1, y1, z1 = p1
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0
            if z0 > z1:
                z0, z1 = z1, z0
            return [
                tri for tri, lo, hi in near_tris
                if x1 >= lo[0] and x0 <= hi[0] and y1 >= lo[1] and y0 <= hi[1]
                and z1 >= lo[2] and z0 <= hi[2]
            ]

        def overlap(t: float) -> bool:
            p0, p1 = ends(t)
            for tri in candidates(p0, p1):
                if capsule_triangle(p0, p1, radius, *tri) is not None:
                    return True
            return False

        def contact(t: float):
            p0, p1 = ends(t)
            best = None
            for tri in candidates(p0, p1):
                hit = capsule_triangle(p0, p1, radius, *tri)
                if hit is not None and (best is None or hit[3] > best[3]):
                    best = hit
            if best is None:
                return None
            n = matrix @ np.array(best[:3])
            pt = matrix @ np.array(best[4:7]) + origin
            return t, n, pt

        return overlap, contact

    def _box_sweep(self, shape: BoxShape, pos, rot, direction, tris):
        local = self.matrix.T @ rot
        axes = tuple(tuple(local[:, i].tolist()) for i in range(3))
        c0 = self.to_local(pos)
        dloc = self.matrix.T @ direction

        def hits(t: float):
            center = tuple((c0 + dloc * t).tolist())
            return [tri for tri in tris
                    if box_triangle_overlap(center, axes, shape.half_extents, *tri)]

        def overlap(t: float) -> bool:
            return bool(hits(t))

        def contact(t: float):
            found = hits(t)
            if not found:
                return None
            a, b, c = found[0]
            n = np.cross(np.subtract(b, a), np.subtract(c, a))
            n /= max(float(np.linalg.norm(n)), 1e-18)
            n = self.matrix @ n
            if float(n @ direction) > 0.0:
                n = -n
            return t, n, pos + direction * t

        return overlap, contact
