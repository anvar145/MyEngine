"""Широкая фаза: Sweep-and-Prune по оси X на numpy."""

from __future__ import annotations

import numpy as np

from engine.physics.store import ColliderStore


class Broadphase:
    """Хранит AABB по handle и находит потенциально пересекающиеся пары."""

    def __init__(self) -> None:
        self._boxes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self._boxes)

    def __contains__(self, handle: int) -> bool:
        return handle in self._boxes

    def add(self, handle: int, aabb) -> None:
        """Добавляет объект (или заменяет существующий)."""
        lo, hi = aabb
        self._boxes[handle] = (
            np.asarray(lo, dtype=np.float64),
            np.asarray(hi, dtype=np.float64),
        )
        self._cache = None

    def update(self, handle: int, aabb) -> None:
        """Обновляет AABB существующего объекта."""
        if handle not in self._boxes:
            raise KeyError(handle)
        self.add(handle, aabb)

    def remove(self, handle: int) -> None:
        """Удаляет объект."""
        del self._boxes[handle]
        self._cache = None

    def clear(self) -> None:
        """Удаляет все объекты."""
        self._boxes.clear()
        self._cache = None

    def get_aabb(self, handle: int) -> tuple[np.ndarray, np.ndarray]:
        """AABB объекта."""
        return self._boxes[handle]

    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Массивы (handles, mins, maxs), отсортированные по min.x, затем handle."""
        if self._cache is None:
            n = len(self._boxes)
            handles = np.fromiter(self._boxes.keys(), dtype=np.int64, count=n)
            if n:
                mins = np.stack([b[0] for b in self._boxes.values()])
                maxs = np.stack([b[1] for b in self._boxes.values()])
            else:
                mins = maxs = np.zeros((0, 3))
            order = np.lexsort((handles, mins[:, 0]))
            self._cache = (handles[order], mins[order], maxs[order])
        return self._cache

    def query_pairs(self) -> list[tuple[int, int]]:
        """Пары пересекающихся AABB, отсортированные по (min_id, max_id)."""
        lo, hi = self.query_pair_arrays()
        return list(zip(lo.tolist(), hi.tolist(), strict=True))

    def query_pair_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Те же пары, что :meth:`query_pairs`, двумя массивами ``(min_id, max_id)``."""
        handles, mins, maxs = self._arrays()
        n = len(handles)
        empty = np.zeros(0, dtype=np.int64)
        if n < 2:
            return empty, empty
        # Для каждого i кандидаты j > i с mins[j].x <= maxs[i].x.
        hi = np.searchsorted(mins[:, 0], maxs[:, 0], side="right")
        counts = np.maximum(hi - np.arange(1, n + 1), 0)
        total = int(counts.sum())
        if total == 0:
            return empty, empty
        ii = np.repeat(np.arange(n), counts)
        starts = np.cumsum(counts) - counts
        jj = np.arange(total) - np.repeat(starts, counts) + ii + 1
        ok = (
            (mins[jj, 1] <= maxs[ii, 1])
            & (mins[ii, 1] <= maxs[jj, 1])
            & (mins[jj, 2] <= maxs[ii, 2])
            & (mins[ii, 2] <= maxs[jj, 2])
        )
        a = handles[ii[ok]]
        b = handles[jj[ok]]
        lo = np.minimum(a, b)
        hi_ = np.maximum(a, b)
        order = np.lexsort((hi_, lo))
        return lo[order], hi_[order]

    def query_aabb(self, aabb) -> list[int]:
        """Handles, чьи AABB пересекают заданный; отсортированы по handle."""
        handles, mins, maxs = self._arrays()
        if not len(handles):
            return []
        lo = np.asarray(aabb[0], dtype=np.float64)
        hi = np.asarray(aabb[1], dtype=np.float64)
        ok = np.all((mins <= hi) & (maxs >= lo), axis=1)
        return sorted(handles[ok].tolist())


class StoreBroadphase(Broadphase):
    """Широкая фаза поверх AABB из ``ColliderStore`` (без собственных данных).

    Алгоритм и результаты те же, что у :class:`Broadphase`; ключ — cid коллайдера.
    Состав задаёт ``ColliderStore`` (``add``/``remove``/``clear`` здесь недоступны),
    изменение AABB отслеживается по ``store.version``.
    """

    def __init__(self, store: ColliderStore) -> None:
        super().__init__()
        self._store = store
        self._version = -1

    def __len__(self) -> int:
        return self._store.count

    def __contains__(self, handle: int) -> bool:
        index = self._store.row_of_cid
        return 0 <= handle < len(index) and index[handle] >= 0

    def add(self, handle: int, aabb) -> None:
        raise TypeError("коллайдеры добавляются через PhysicsWorld.add_body")

    def remove(self, handle: int) -> None:
        raise TypeError("коллайдеры удаляются через PhysicsWorld.remove_body")

    def clear(self) -> None:
        raise TypeError("мир очищается через PhysicsWorld.clear")

    def update(self, handle: int, aabb) -> None:
        """Записывает AABB коллайдера ``handle`` в хранилище."""
        if handle not in self:
            raise KeyError(handle)
        row = int(self._store.row_of_cid[handle])
        lo, hi = aabb
        self._store.aabb_lo[row] = lo
        self._store.aabb_hi[row] = hi
        self._store.version += 1

    def get_aabb(self, handle: int) -> tuple[np.ndarray, np.ndarray]:
        if handle not in self:
            raise KeyError(handle)
        row = int(self._store.row_of_cid[handle])
        return self._store.aabb_lo[row], self._store.aabb_hi[row]

    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        store = self._store
        if self._cache is None or self._version != store.version:
            n = store.count
            ids = store.cid[:n]
            mins = store.aabb_lo[:n]
            maxs = store.aabb_hi[:n]
            order = np.lexsort((ids, mins[:, 0]))
            self._cache = (ids[order], mins[order], maxs[order])
            self._version = store.version
        return self._cache
