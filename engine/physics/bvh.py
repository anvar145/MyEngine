"""BVH по треугольникам статического меша: лучи и выборки по объёму.

Дерево строится разбиением по медиане центроидов AABB треугольников вдоль
самой длинной оси (не SAH: строится за десятки миллисекунд на 10k треугольников
и достаточно для статических уровней). Узлы лежат плоскими массивами; потомки
узла всегда соседние (``left`` и ``left + 1``), поэтому узлу хватает двух чисел:

* ``count == 0`` - внутренний узел, ``first`` = индекс левого потомка;
* ``count > 0`` - лист, ``first`` = начало диапазона в порядке обхода дерева,
  ``count`` треугольников (не больше ``leaf_size``).

Запросы одиночные (один луч/объём за вызов) и написаны на чистом Python над
списками float: для нескольких десятков узлов это быстрее, чем накладные
расходы numpy. Все индексы треугольников в ответах - индексы исходного массива
``triangles``. Геометрия живёт в локальном пространстве меша; перевод в мир
(поворот, смещение) делает вызывающий код.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["Bvh", "BvhHit", "closest_point_triangle", "hit"]

# «Бесконечность» для 1/0 в слэб-тесте: конечна, чтобы 0 * inf не давал NaN.
_BIG = 1e300
_EPS = 1e-12
# С этого числа лучей пакет обходит дерево послойно на numpy (иначе - циклом).
BATCH_MIN_NUMPY = 16


@dataclass(slots=True)
class BvhHit:
    """Попадание луча в треугольник.

    Attributes:
        distance: Расстояние вдоль (нормированного) луча.
        triangle: Индекс треугольника в исходном массиве.
        normal: Единичная нормаль грани, повёрнутая к началу луча.
    """

    distance: float
    triangle: int
    normal: tuple[float, float, float]


class Bvh:
    """Статическое BVH над треугольной сеткой.

    Args:
        vertices: Вершины ``(N, 3)``.
        triangles: Индексы вершин ``(M, 3)``.
        leaf_size: Максимум треугольников в листе.

    Raises:
        ValueError: Если формы массивов неверны или индекс вне диапазона.
    """

    def __init__(self, vertices, triangles, leaf_size: int = 8) -> None:
        verts = np.ascontiguousarray(vertices, dtype=np.float64)
        tris = np.ascontiguousarray(triangles, dtype=np.int64)
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError("vertices: нужна форма (N, 3)")
        tris = tris.reshape(-1, 3)
        if len(tris) and (tris.min() < 0 or tris.max() >= len(verts)):
            raise ValueError("triangles: индекс вершины вне диапазона")
        self.vertices = verts
        self.triangles = tris
        self.leaf_size = max(int(leaf_size), 1)
        self._build()

    # ------------------------------------------------------------- построение

    def _build(self) -> None:
        count = len(self.triangles)
        corners = self.vertices[self.triangles] if count else np.zeros((0, 3, 3))
        tri_lo = corners.min(axis=1) if count else np.zeros((0, 3))
        tri_hi = corners.max(axis=1) if count else np.zeros((0, 3))
        centroid = (tri_lo + tri_hi) * 0.5
        order = np.arange(count, dtype=np.int64)

        first: list[int] = [0]
        size: list[int] = [count]
        depth: list[int] = [0]
        ranges: list[tuple[int, int]] = [(0, count)]
        index = 0
        leaf = self.leaf_size
        while index < len(ranges):
            start, end = ranges[index]
            if end - start <= leaf:
                first[index] = start
                size[index] = end - start
            else:
                chunk = order[start:end]
                keys = centroid[chunk]
                axis = int(np.argmax(keys.max(axis=0) - keys.min(axis=0)))
                mid = (end - start) // 2
                part = np.argpartition(keys[:, axis], mid)
                order[start:end] = chunk[part]
                left = len(ranges)
                first[index] = left
                size[index] = 0
                ranges.append((start, start + mid))
                ranges.append((start + mid, end))
                first.extend((0, 0))
                size.extend((0, 0))
                depth.extend((depth[index] + 1,) * 2)
            index += 1

        nodes = len(ranges)
        node_first = np.asarray(first, dtype=np.int64)
        node_size = np.asarray(size, dtype=np.int64)
        node_depth = np.asarray(depth, dtype=np.int64)
        lo = np.zeros((nodes, 3))
        hi = np.zeros((nodes, 3))
        ordered_lo = tri_lo[order]
        ordered_hi = tri_hi[order]
        leaves = np.flatnonzero(node_size > 0)
        if len(leaves):
            starts = node_first[leaves]
            lo[leaves] = np.minimum.reduceat(ordered_lo, starts, axis=0)
            hi[leaves] = np.maximum.reduceat(ordered_hi, starts, axis=0)
        inner = np.flatnonzero((node_size == 0) & (node_first > 0))
        for level in range(int(node_depth.max(initial=0)), -1, -1):
            ids = inner[node_depth[inner] == level]
            if len(ids):
                kids = node_first[ids]
                lo[ids] = np.minimum(lo[kids], lo[kids + 1])
                hi[ids] = np.maximum(hi[kids], hi[kids + 1])

        self.node_lo = lo
        self.node_hi = hi
        self.node_first = node_first
        self.node_count = node_size
        self.depth = int(node_depth.max(initial=0)) + 1
        self.order = order  # позиция в обходе -> индекс исходного треугольника
        # списки для быстрых одиночных запросов на чистом Python
        self._lo = lo.tolist()
        self._hi = hi.tolist()
        self._first = node_first.tolist()
        self._size = node_size.tolist()
        self._ids = order.tolist()
        ordered = corners[order]
        edge1 = ordered[:, 1] - ordered[:, 0]
        edge2 = ordered[:, 2] - ordered[:, 0]
        self._tri = np.concatenate(
            (ordered[:, 0], edge1, edge2), axis=1
        ).tolist()  # v0, e1, e2 по 3 числа
        self._tlo = ordered_lo.tolist()
        self._thi = ordered_hi.tolist()
        self._box_tables: dict[int, list] = {}
        self._tri_np: tuple[np.ndarray, ...] | None = None
        self._box_packed: np.ndarray | None = None

    # ----------------------------------------------------------------- свойства

    @property
    def triangle_count(self) -> int:
        """Число треугольников."""
        return len(self.triangles)

    @property
    def node_total(self) -> int:
        """Число узлов дерева."""
        return len(self._first)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Габарит меша ``(lo, hi)`` (пустое дерево - нулевой бокс)."""
        return self.node_lo[0].copy(), self.node_hi[0].copy()

    def triangle(self, index: int) -> np.ndarray:
        """Вершины треугольника исходного массива, ``(3, 3)``."""
        return self.vertices[self.triangles[index]]

    # -------------------------------------------------------------------- луч

    def _boxes_for(self, sign: int) -> list:
        """Боксы узлов для знаков направления: ``(near xyz, far xyz)``.

        ``sign`` - биты ``dx < 0``, ``dy < 0``, ``dz < 0``. Ближняя вершина
        бокса относительно луча заранее выбрана по знаку, поэтому слэб-тест
        не меняет границы местами. Таблицы строятся лениво (до 8 штук).
        """
        table = self._box_tables.get(sign)
        if table is None:
            lo, hi = self.node_lo, self.node_hi
            near = np.where([bool(sign & 1), bool(sign & 2), bool(sign & 4)], hi, lo)
            far = np.where([bool(sign & 1), bool(sign & 2), bool(sign & 4)], lo, hi)
            table = np.concatenate((near, far), axis=1).tolist()
            self._box_tables[sign] = table
        return table

    def _trace(
        self, ox: float, oy: float, oz: float,
        dx: float, dy: float, dz: float, best: float,
    ) -> tuple[float, int]:
        """Ядро луча: нормированное направление, ближайший потомок первым.

        Дети проверяются слэб-тестом при помещении в стек; в стеке вместе с
        узлом лежит ``tmin`` входа луча, поэтому узлы дальше найденного
        попадания отсекаются при извлечении без повторного теста.

        Returns:
            ``(расстояние, позиция треугольника в обходе)``; позиция ``-1``,
            если попаданий нет (расстояние тогда равно ``best``).
        """
        ix = 1.0 / dx if abs(dx) > 1e-300 else _BIG
        iy = 1.0 / dy if abs(dy) > 1e-300 else _BIG
        iz = 1.0 / dz if abs(dz) > 1e-300 else _BIG
        best_tri = -1
        boxes = self._boxes_for((dx < 0.0) | ((dy < 0.0) << 1) | ((dz < 0.0) << 2))
        first, size = self._first, self._size
        tris = self._tri
        stack = [0]
        times = [-_BIG]
        pop = stack.pop
        push = stack.append
        pop_t = times.pop
        push_t = times.append
        while stack:
            node = pop()
            if pop_t() > best:
                continue
            count = size[node]
            if count == 0:
                left = first[node]
                nx, ny, nz, fx, fy, fz = boxes[left]
                t1 = (nx - ox) * ix
                t3 = (ny - oy) * iy
                if t3 > t1:
                    t1 = t3
                t3 = (nz - oz) * iz
                if t3 > t1:
                    t1 = t3
                t2 = (fx - ox) * ix
                t3 = (fy - oy) * iy
                if t3 < t2:
                    t2 = t3
                t3 = (fz - oz) * iz
                if t3 < t2:
                    t2 = t3
                near_ok = t2 >= 0.0 and t1 <= t2 and t1 <= best
                right = left + 1
                nx, ny, nz, fx, fy, fz = boxes[right]
                t3 = (nx - ox) * ix
                t5 = (ny - oy) * iy
                if t5 > t3:
                    t3 = t5
                t5 = (nz - oz) * iz
                if t5 > t3:
                    t3 = t5
                t4 = (fx - ox) * ix
                t5 = (fy - oy) * iy
                if t5 < t4:
                    t4 = t5
                t5 = (fz - oz) * iz
                if t5 < t4:
                    t4 = t5
                if t4 >= 0.0 and t3 <= t4 and t3 <= best:
                    if near_ok:
                        # оба потомка: дальний кладём первым, ближний достанем раньше
                        if t1 <= t3:
                            push(right)
                            push_t(t3)
                            push(left)
                            push_t(t1)
                        else:
                            push(left)
                            push_t(t1)
                            push(right)
                            push_t(t3)
                    else:
                        push(right)
                        push_t(t3)
                elif near_ok:
                    push(left)
                    push_t(t1)
                continue
            start = first[node]
            for k in range(start, start + count):
                v0x, v0y, v0z, e1x, e1y, e1z, e2x, e2y, e2z = tris[k]
                # Мёллер - Трамбор
                px = dy * e2z - dz * e2y
                py = dz * e2x - dx * e2z
                pz = dx * e2y - dy * e2x
                det = e1x * px + e1y * py + e1z * pz
                if -_EPS < det < _EPS:
                    continue
                inv = 1.0 / det
                sx, sy, sz = ox - v0x, oy - v0y, oz - v0z
                u = (sx * px + sy * py + sz * pz) * inv
                if u < 0.0 or u > 1.0:
                    continue
                qx = sy * e1z - sz * e1y
                qy = sz * e1x - sx * e1z
                qz = sx * e1y - sy * e1x
                v = (dx * qx + dy * qy + dz * qz) * inv
                if v < 0.0 or u + v > 1.0:
                    continue
                t = (e2x * qx + e2y * qy + e2z * qz) * inv
                if 0.0 <= t < best:
                    best = t
                    best_tri = k
        return best, best_tri

    def raycast(
        self, origin, direction, max_distance: float = math.inf
    ) -> BvhHit | None:
        """Ближайшее пересечение луча с треугольниками (обе стороны грани).

        Args:
            origin: Начало луча.
            direction: Направление (нормируется).
            max_distance: Предельное расстояние.

        Returns:
            Ближайшее попадание или ``None``.
        """
        if not self.triangle_count:
            return None
        ox, oy, oz = (float(v) for v in origin)
        dx, dy, dz = (float(v) for v in direction)
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < _EPS:
            return None
        return self.raycast_unit(
            ox, oy, oz, dx / length, dy / length, dz / length, max_distance
        )

    def raycast_unit(
        self, ox: float, oy: float, oz: float,
        dx: float, dy: float, dz: float, max_distance: float = math.inf,
    ) -> BvhHit | None:
        """:meth:`raycast` для уже нормированного направления и чисел float."""
        if not self.triangle_count:
            return None
        best, best_tri = self._trace(ox, oy, oz, dx, dy, dz, float(max_distance))
        if best_tri < 0:
            return None
        _, _, _, e1x, e1y, e1z, e2x, e2y, e2z = self._tri[best_tri]
        nx = e1y * e2z - e1z * e2y
        ny = e1z * e2x - e1x * e2z
        nz = e1x * e2y - e1y * e2x
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        nx, ny, nz = nx / norm, ny / norm, nz / norm
        if nx * dx + ny * dy + nz * dz > 0.0:
            nx, ny, nz = -nx, -ny, -nz
        return BvhHit(best, self._ids[best_tri], (nx, ny, nz))

    def raycast_batch(
        self, origins, directions, max_distance: float = math.inf
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ближайшие попадания для пакета лучей без объектов на луч.

        Пакеты >= 16 лучей (``BATCH_MIN_NUMPY``) идут numpy-обходом; меньше -
        циклом (numpy overhead не окупается). Numpy-обход проходит дерево
        послойно: пары «луч - узел» отсекаются слэб-тестом сразу для всех лучей.

        Args:
            origins: Начала лучей ``(n, 3)``.
            directions: Направления ``(n, 3)`` или ``(1, 3)`` (нормируются;
                нулевые - промах).
            max_distance: Предельное расстояние.

        Returns:
            ``(hit_t, hit_tri, hit_point)``: расстояния ``(n,)`` (``inf`` при
            промахе), индексы исходных треугольников ``(n,)`` (``-1`` при
            промахе) и точки попадания ``(n, 3)`` (``nan`` при промахе).

        Raises:
            ValueError: Если форма ``directions`` не подходит.
        """
        orig = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
        dirs = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
        if dirs.shape != orig.shape:
            if dirs.shape[0] != 1:
                raise ValueError("directions: нужна форма (n, 3) или (1, 3)")
            dirs = np.broadcast_to(dirs, orig.shape)
        count = len(orig)
        hit_t = np.full(count, np.inf)
        hit_tri = np.full(count, -1, dtype=np.int64)
        lengths = np.sqrt(np.einsum("ij,ij->i", dirs, dirs))
        valid = lengths >= _EPS
        unit = dirs / np.where(valid, lengths, 1.0)[:, None]
        if self.triangle_count and count:
            limit = float(max_distance)
            rays = np.flatnonzero(valid)
            if len(rays) >= BATCH_MIN_NUMPY:
                best_t, best_k = self._trace_levels(orig[rays], unit[rays], limit)
                found = best_k >= 0
                hit_t[rays[found]] = best_t[found]
                hit_tri[rays[found]] = self.order[best_k[found]]
            else:
                trace, ids = self._trace, self._ids
                o_list, d_list = orig.tolist(), unit.tolist()
                for i in rays.tolist():
                    ox, oy, oz = o_list[i]
                    dx, dy, dz = d_list[i]
                    best, tri = trace(ox, oy, oz, dx, dy, dz, limit)
                    if tri >= 0:
                        hit_t[i] = best
                        hit_tri[i] = ids[tri]
        hit_point = np.full((count, 3), np.nan)
        found = hit_tri >= 0
        hit_point[found] = orig[found] + unit[found] * hit_t[found][:, None]
        return hit_t, hit_tri, hit_point

    def _trace_levels(
        self, orig: np.ndarray, unit: np.ndarray, limit: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Послойный обход для пакета: ``(best_t (n,), позиция треугольника)``.

        Позиция ``-1`` - промах. На каждом шаге живые пары «луч - узел»
        проходят слэб-тест (с отсечением по найденному попаданию), листья
        раскрываются в пары «луч - треугольник» (Мёллер - Трамбор), внутренние
        узлы - в двух детей.
        """
        n = len(orig)
        inv = np.where(np.abs(unit) > 1e-300, 1.0 / np.where(
            np.abs(unit) > 1e-300, unit, 1.0), _BIG)
        best_t = np.full(n, limit)
        best_k = np.full(n, -1, dtype=np.int64)
        boxes = self._boxes_packed()
        first, size = self.node_first, self.node_count
        v0, e1, e2 = self._tri_arrays()
        rays = np.concatenate((orig, inv), axis=1)  # (n, 6): начало и 1/d
        ray = np.arange(n)
        node = np.zeros(n, dtype=np.int64)
        while len(ray):
            r = rays[ray]
            box = boxes[node]
            t1 = (box[:, 0:3] - r[:, 0:3]) * r[:, 3:6]
            t2 = (box[:, 3:6] - r[:, 0:3]) * r[:, 3:6]
            lo_t = np.minimum(t1, t2)
            hi_t = np.maximum(t1, t2)
            t_near = np.maximum(np.maximum(lo_t[:, 0], lo_t[:, 1]), lo_t[:, 2])
            t_far = np.minimum(np.minimum(hi_t[:, 0], hi_t[:, 1]), hi_t[:, 2])
            keep = (t_far >= 0.0) & (t_near <= t_far) & (t_near <= best_t[ray])
            ray, node = ray[keep], node[keep]
            if not len(ray):
                break
            count = size[node]
            leaf = count > 0
            if leaf.any():
                lr, ln, lc = ray[leaf], node[leaf], count[leaf]
                total = int(lc.sum())
                starts = np.cumsum(lc) - lc
                within = np.arange(total) - np.repeat(starts, lc)
                tray = np.repeat(lr, lc)
                tk = np.repeat(first[ln], lc) + within
                self._hit_triangles(tray, tk, orig, unit, v0, e1, e2, best_t, best_k)
                inner = ~leaf
                ray, node = ray[inner], node[inner]
            kids = np.repeat(first[node], 2)
            kids[1::2] += 1
            ray = np.repeat(ray, 2)
            node = kids
        return best_t, best_k

    @staticmethod
    def _hit_triangles(tray, tk, orig, unit, v0, e1, e2, best_t, best_k) -> None:
        """Мёллер - Трамбор для пар «луч - треугольник»; обновляет лучшие."""
        dx, dy, dz = unit[tray].T
        a1, a2, a3 = e1[tk].T
        b1, b2, b3 = e2[tk].T
        px = dy * b3 - dz * b2
        py = dz * b1 - dx * b3
        pz = dx * b2 - dy * b1
        det = a1 * px + a2 * py + a3 * pz
        ok = np.abs(det) >= _EPS
        inv = 1.0 / np.where(ok, det, 1.0)
        s = orig[tray] - v0[tk]
        u = (s[:, 0] * px + s[:, 1] * py + s[:, 2] * pz) * inv
        qx = s[:, 1] * a3 - s[:, 2] * a2
        qy = s[:, 2] * a1 - s[:, 0] * a3
        qz = s[:, 0] * a2 - s[:, 1] * a1
        v = (dx * qx + dy * qy + dz * qz) * inv
        t = (b1 * qx + b2 * qy + b3 * qz) * inv
        ok &= (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (u + v <= 1.0)
        ok &= (t >= 0.0) & (t < best_t[tray])
        if not ok.any():
            return
        r, t, k = tray[ok], t[ok], tk[ok]
        np.minimum.at(best_t, r, t)
        mine = t == best_t[r]
        best_k[r[mine]] = k[mine]

    def _boxes_packed(self) -> np.ndarray:
        """Боксы узлов одним массивом ``(узлов, 6)``: ``lo`` и ``hi`` (лениво)."""
        packed = self._box_packed
        if packed is None:
            packed = np.concatenate((self.node_lo, self.node_hi), axis=1)
            self._box_packed = packed
        return packed

    def _tri_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Треугольники в порядке обхода как массивы ``v0, e1, e2`` (лениво)."""
        arrays = self._tri_np
        if arrays is None:
            flat = np.asarray(self._tri, dtype=np.float64).reshape(-1, 9)
            arrays = (flat[:, 0:3].copy(), flat[:, 3:6].copy(), flat[:, 6:9].copy())
            self._tri_np = arrays
        return arrays

    # --------------------------------------------------------------- выборки

    def query_aabb(self, lo, hi) -> list[int]:
        """Индексы треугольников, чьи AABB пересекают бокс ``[lo, hi]``.

        Выборка грубая (по AABB треугольников), порядок не определён.
        """
        if not self.triangle_count:
            return []
        lx, ly, lz = (float(v) for v in lo)
        hx, hy, hz = (float(v) for v in hi)
        found: list[int] = []
        nlo, nhi, first, size = self._lo, self._hi, self._first, self._size
        tlo, thi, ids = self._tlo, self._thi, self._ids
        stack = [0]
        while stack:
            node = stack.pop()
            a, b = nlo[node], nhi[node]
            if (a[0] > hx or b[0] < lx or a[1] > hy or b[1] < ly
                    or a[2] > hz or b[2] < lz):
                continue
            count = size[node]
            if count == 0:
                left = first[node]
                stack.append(left)
                stack.append(left + 1)
                continue
            start = first[node]
            for k in range(start, start + count):
                a, b = tlo[k], thi[k]
                if not (a[0] > hx or b[0] < lx or a[1] > hy or b[1] < ly
                        or a[2] > hz or b[2] < lz):
                    found.append(ids[k])
        return found

    def query_sphere(self, center, radius: float) -> list[int]:
        """Индексы треугольников на расстоянии не больше ``radius`` от точки.

        Узлы отсекаются по расстоянию до их бокса, треугольники проверяются
        точно (ближайшая точка треугольника).
        """
        if not self.triangle_count:
            return []
        cx, cy, cz = (float(v) for v in center)
        r2 = float(radius) ** 2
        found: list[int] = []
        nlo, nhi, first, size = self._lo, self._hi, self._first, self._size
        tris, ids = self._tri, self._ids
        stack = [0]
        while stack:
            node = stack.pop()
            a, b = nlo[node], nhi[node]
            gx = a[0] - cx if cx < a[0] else (cx - b[0] if cx > b[0] else 0.0)
            gy = a[1] - cy if cy < a[1] else (cy - b[1] if cy > b[1] else 0.0)
            gz = a[2] - cz if cz < a[2] else (cz - b[2] if cz > b[2] else 0.0)
            if gx * gx + gy * gy + gz * gz > r2:
                continue
            count = size[node]
            if count == 0:
                left = first[node]
                stack.append(left)
                stack.append(left + 1)
                continue
            start = first[node]
            for k in range(start, start + count):
                v0x, v0y, v0z, e1x, e1y, e1z, e2x, e2y, e2z = tris[k]
                p = closest_point_triangle(
                    (cx, cy, cz),
                    (v0x, v0y, v0z),
                    (v0x + e1x, v0y + e1y, v0z + e1z),
                    (v0x + e2x, v0y + e2y, v0z + e2z),
                )
                if (p[0] - cx) ** 2 + (p[1] - cy) ** 2 + (p[2] - cz) ** 2 <= r2:
                    found.append(ids[k])
        return found


def hit(bvh: Bvh, origin, direction, max_distance: float = math.inf) -> BvhHit | None:
    """Функциональная форма :meth:`Bvh.raycast`."""
    return bvh.raycast(origin, direction, max_distance)


def closest_point_triangle(p, a, b, c) -> tuple[float, float, float]:
    """Ближайшая к точке ``p`` точка треугольника ``abc`` (Эрикссон, чистый Python)."""
    abx, aby, abz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    acx, acy, acz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    apx, apy, apz = p[0] - a[0], p[1] - a[1], p[2] - a[2]
    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    if d1 <= 0.0 and d2 <= 0.0:
        return (a[0], a[1], a[2])
    bpx, bpy, bpz = p[0] - b[0], p[1] - b[1], p[2] - b[2]
    d3 = abx * bpx + aby * bpy + abz * bpz
    d4 = acx * bpx + acy * bpy + acz * bpz
    if d3 >= 0.0 and d4 <= d3:
        return (b[0], b[1], b[2])
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return (a[0] + abx * v, a[1] + aby * v, a[2] + abz * v)
    cpx, cpy, cpz = p[0] - c[0], p[1] - c[1], p[2] - c[2]
    d5 = abx * cpx + aby * cpy + abz * cpz
    d6 = acx * cpx + acy * cpy + acz * cpz
    if d6 >= 0.0 and d5 <= d6:
        return (c[0], c[1], c[2])
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return (a[0] + acx * w, a[1] + acy * w, a[2] + acz * w)
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return (b[0] + (c[0] - b[0]) * w, b[1] + (c[1] - b[1]) * w,
                b[2] + (c[2] - b[2]) * w)
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return (a[0] + abx * v + acx * w, a[1] + aby * v + acy * w,
            a[2] + abz * v + acz * w)
