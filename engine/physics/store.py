"""SoA-хранилища физики: тела и коллайдеры как строки numpy-массивов.

``BodyStore`` держит состояние тел (позы, скорости, массы, флаги), ``ColliderStore`` —
конфигурацию и мировые позы/AABB коллайдеров. Горячие пути мира работают с
массивами целиком; ``RigidBody`` и ``Collider`` — тонкие фасады над строками.

Строки хранилищ плотные (``0..count``): удаление — swap-remove (последняя строка
встаёт на место удалённой), рост — удвоением ёмкости. Порядок обхода, от которого
зависит результат (тела — по handle, коллайдеры — по cid), задаётся не номером
строки, а ключом, поэтому состояние не зависит от истории добавлений/удалений.

Ограничение: ``np.ndarray``-view, полученный из свойства (``body.position``),
валиден только до ближайшего ``add_body``/``remove_body`` — после роста ёмкости он
указывает на старый буфер, после swap-remove строка может достаться другому телу.
Кэшировать view нельзя: свойство нужно читать заново.
"""

from __future__ import annotations

import numpy as np

from engine.physics.shape import BoxShape, CapsuleShape, SphereShape

# Основной тип данных физики. Для эксперимента с float32 меняется только здесь.
FLOAT = np.float64

BODY_STATIC = 0
BODY_KINEMATIC = 1
BODY_DYNAMIC = 2
BODY_CODES = {"static": BODY_STATIC, "kinematic": BODY_KINEMATIC,
              "dynamic": BODY_DYNAMIC}

KIND_BOX = 0
KIND_SPHERE = 1
KIND_CAPSULE = 2
KIND_OTHER = 3

MASK64 = (1 << 64) - 1  # слои и маски хранятся как uint64 (отрицательные — по модулю)
_INITIAL_CAPACITY = 64


class _RowStore:
    """Плотное хранилище строк с удвоением ёмкости и swap-remove."""

    # (имя массива, форма «хвоста» строки, dtype)
    _SPECS: tuple[tuple[str, tuple[int, ...], type], ...] = ()

    def __init__(self, capacity: int = _INITIAL_CAPACITY) -> None:
        self.capacity = 0
        self.count = 0
        self.items: list = []  # объект-владелец строки (фасад) по номеру строки
        self._alloc(max(int(capacity), 1))

    def _alloc(self, capacity: int) -> None:
        for name, tail, dtype in self._SPECS:
            new = np.zeros((capacity, *tail), dtype=dtype)
            old = getattr(self, name, None)
            if old is not None and self.count:
                new[: self.count] = old[: self.count]
            setattr(self, name, new)
        self.capacity = capacity

    def _push(self, item) -> int:
        """Резервирует строку (чистую, нулевую) и возвращает её номер."""
        if self.count == self.capacity:
            self._alloc(self.capacity * 2)
        row = self.count
        self.count += 1
        self.items.append(item)
        return row

    def _pop(self, row: int):
        """Swap-remove строки; возвращает владельца, переехавшего на её место."""
        last = self.count - 1
        moved = None
        if row != last:
            for name, _tail, _dtype in self._SPECS:
                arr = getattr(self, name)
                arr[row] = arr[last]
            moved = self.items[last]
            self.items[row] = moved
        for name, _tail, _dtype in self._SPECS:
            getattr(self, name)[last] = 0
        self.items.pop()
        self.count -= 1
        return moved


def _grow_index(index: np.ndarray, key: int) -> np.ndarray:
    """Расширяет массив «ключ -> строка» (заполнение −1), чтобы влез ``key``."""
    if key < len(index):
        return index
    size = len(index)
    while size <= key:
        size *= 2
    new = np.full(size, -1, dtype=np.int32)
    new[: len(index)] = index
    return new


class BodyStore(_RowStore):
    """Тела: позы, скорости, массы, демпфирование, флаги.

    ``vel`` — (n, 6): ``[:, :3]`` линейная, ``[:, 3:]`` угловая скорость.
    ``body_type`` — код (``BODY_STATIC``/``BODY_KINEMATIC``/``BODY_DYNAMIC``).
    ``handle`` — handle тела в мире (0 у «частного» хранилища одиночного тела).
    """

    _SPECS = (
        ("position", (3,), FLOAT),
        ("rotation", (4,), FLOAT),
        ("vel", (6,), FLOAT),
        ("mass", (), FLOAT),
        ("inv_mass", (), FLOAT),
        ("inv_inertia_local", (3, 3), FLOAT),
        ("inv_inertia_world", (3, 3), FLOAT),
        ("linear_damping", (), FLOAT),
        ("angular_damping", (), FLOAT),
        ("gravity_scale", (), FLOAT),
        ("sleep_timer", (), FLOAT),
        ("body_type", (), np.uint8),
        ("sleeping", (), np.bool_),
        ("can_sleep", (), np.bool_),
        ("use_gravity", (), np.bool_),
        ("freeze_rotation", (), np.bool_),
        ("handle", (), np.int64),
    )

    def __init__(self, capacity: int = _INITIAL_CAPACITY, shared: bool = False) -> None:
        self.shared = shared  # True у хранилища мира (тело там «занято»)
        self.row_of_handle = np.full(16, -1, dtype=np.int32)
        self._order: np.ndarray | None = None
        super().__init__(capacity)

    @property
    def bodies(self) -> list:
        """Фасады ``RigidBody`` по номеру строки."""
        return self.items

    @property
    def order(self) -> np.ndarray:
        """Номера строк по возрастанию handle (кэш; сбрасывается add/remove)."""
        if self._order is None:
            self._order = np.argsort(self.handle[: self.count], kind="stable")
        return self._order

    def adopt(self, body, handle: int = 0) -> int:
        """Переносит строку ``body`` в это хранилище и перенаправляет фасад.

        Тот же Python-объект ``RigidBody`` продолжает жить: меняются только
        ``_store``/``_row``. Старая (частная) строка освобождается.
        """
        src, src_row = body._store, body._row
        row = self._push(body)
        for name, _tail, _dtype in self._SPECS:
            getattr(self, name)[row] = getattr(src, name)[src_row]
        self.handle[row] = handle
        body._store, body._row = self, row
        if handle:
            self.row_of_handle = _grow_index(self.row_of_handle, handle)
            self.row_of_handle[handle] = row
        self._order = None
        if src is not self:
            src._drop(src_row)
        return row

    def release(self, body) -> None:
        """Возвращает тело в новое частное хранилище (после удаления из мира)."""
        private = BodyStore(1)
        private.adopt(body, 0)  # освобождает строку здесь через _drop

    def _drop(self, row: int) -> None:
        """Удаляет строку (swap-remove) и чинит ссылки переехавшего тела."""
        h = int(self.handle[row])
        if h:
            self.row_of_handle[h] = -1
        moved = self._pop(row)
        if moved is not None:
            moved._row = row
            mh = int(self.handle[row])
            if mh:
                self.row_of_handle[mh] = row
        self._order = None

    def clear(self) -> None:
        """Освобождает все строки, оставляя телам частные хранилища."""
        for body in list(self.items):
            self.release(body)


class ColliderStore(_RowStore):
    """Коллайдеры: конфигурация (копия при добавлении) и мировая поза/AABB.

    ``half``: у куба — полуразмеры, у шара — ``(r, r, r)``, у капсулы —
    ``(r, half_height, 0)``. ``owner_handle`` — handle тела-владельца.
    ``version`` растёт при любом изменении AABB/состава и инвалидирует кэш
    широкой фазы.
    """

    _SPECS = (
        ("cid", (), np.int64),
        ("owner_handle", (), np.int64),
        ("kind", (), np.uint8),
        ("half", (3,), FLOAT),
        ("offset", (3,), FLOAT),
        ("rot_off", (3, 3), FLOAT),
        ("layer", (), np.uint64),
        ("mask", (), np.uint64),
        ("is_trigger", (), np.bool_),
        ("world_pos", (3,), FLOAT),
        ("world_rot", (3, 3), FLOAT),
        ("aabb_lo", (3,), FLOAT),
        ("aabb_hi", (3,), FLOAT),
    )

    def __init__(self, capacity: int = _INITIAL_CAPACITY) -> None:
        self.row_of_cid = np.full(16, -1, dtype=np.int32)
        self.version = 0
        super().__init__(capacity)

    @property
    def colliders(self) -> list:
        """Фасады ``Collider`` по номеру строки."""
        return self.items

    def add(self, col, cid: int, owner_handle: int) -> int:
        """Копирует конфигурацию ``col`` в новую строку и привязывает коллайдер."""
        row = self._push(col)
        shape = col.shape
        if isinstance(shape, BoxShape):
            kind, half = KIND_BOX, shape.half_extents
        elif isinstance(shape, SphereShape):
            kind, half = KIND_SPHERE, (shape.radius,) * 3
        elif isinstance(shape, CapsuleShape):
            kind, half = KIND_CAPSULE, (shape.radius, shape.half_height, 0.0)
        else:
            kind, half = KIND_OTHER, (0.0, 0.0, 0.0)
        self.cid[row] = cid
        self.owner_handle[row] = owner_handle
        self.kind[row] = kind
        self.half[row] = half
        self.offset[row] = col.offset
        self.rot_off[row] = col._rot_offset_matrix
        self.layer[row] = int(col.layer) & MASK64
        self.mask[row] = int(col.mask) & MASK64
        self.is_trigger[row] = bool(col.is_trigger)
        self.row_of_cid = _grow_index(self.row_of_cid, cid)
        self.row_of_cid[cid] = row
        col._store, col._row = self, row
        self.version += 1
        return row

    def remove(self, cid: int) -> None:
        """Удаляет коллайдер; его мировое состояние остаётся в самом объекте."""
        row = int(self.row_of_cid[cid])
        col = self.items[row]
        col._wp = self.world_pos[row].copy()
        col._wr = self.world_rot[row].copy()
        col._wa = (self.aabb_lo[row].copy(), self.aabb_hi[row].copy())
        col._store, col._row = None, -1
        self.row_of_cid[cid] = -1
        moved = self._pop(row)
        if moved is not None:
            moved._row = row
            self.row_of_cid[int(self.cid[row])] = row
        self.version += 1

    def clear(self) -> None:
        """Отвязывает и удаляет все коллайдеры."""
        for cid in [int(c) for c in self.cid[: self.count]]:
            self.remove(cid)

    def update_world(self, rows, body_pos, body_mat) -> None:
        """Пересчитывает мировую позу и AABB строк ``rows``.

        Args:
            rows: Номера строк коллайдеров (``(k,)`` int).
            body_pos: Позиции тел-владельцев по строкам, ``(k, 3)``.
            body_mat: Матрицы вращения тел-владельцев, ``(k, 3, 3)``.
        """
        offset = self.offset[rows]
        wpos = body_pos + (body_mat * offset[:, None, :]).sum(axis=2)
        wrot = body_mat @ self.rot_off[rows]
        self.world_pos[rows] = wpos
        self.world_rot[rows] = wrot
        kind = self.kind[rows]
        half = self.half[rows]
        first = int(kind[0]) if len(kind) else KIND_BOX
        if (kind == first).all() and first != KIND_OTHER:
            lo, hi = _aabb_by_kind(first, wpos, wrot, half)
        else:
            lo = np.empty_like(wpos)
            hi = np.empty_like(wpos)
            for code in np.unique(kind).tolist():
                sel = np.flatnonzero(kind == code)
                if code == KIND_OTHER:
                    for i in sel.tolist():
                        shape = self.items[int(rows[i])].shape
                        lo[i], hi[i] = shape.aabb(wpos[i], wrot[i])
                else:
                    lo[sel], hi[sel] = _aabb_by_kind(
                        code, wpos[sel], wrot[sel], half[sel]
                    )
        self.aabb_lo[rows] = lo
        self.aabb_hi[rows] = hi
        self.version += 1


def _aabb_by_kind(code: int, pos, rot, half):
    """Пакетный мировой AABB для коллайдеров одного вида."""
    if code == KIND_BOX:
        ext = (np.abs(rot) * half[:, None, :]).sum(axis=2)
        return pos - ext, pos + ext
    if code == KIND_SPHERE:
        return pos - half, pos + half
    axis = rot[:, :, 1] * half[:, 1:2]  # ось капсулы — локальный Y
    a = pos - axis
    b = pos + axis
    radius = half[:, 0:1]
    return np.minimum(a, b) - radius, np.maximum(a, b) + radius
