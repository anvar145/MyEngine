"""Трансформы: иерархия и векторизованный расчёт мировых матриц.

Идея модуля — «фасад + плоское хранилище»:

* :class:`Transform` — удобный фасад для игрового кода (``position``,
  ``rotation``, ``scale``, родитель, дети);
* :class:`TransformStore` — numpy-массивы на всю сцену. Локальные матрицы
  собираются одной batch-операцией, мировые — по уровням иерархии через
  ``np.matmul``, то есть без Python-цикла на каждый объект.

Такой гибрид даёт привычный API (как GameObject/Transform в Unity) и
предсказуемую производительность на слабом железе: пересчитываются только
изменённые поддеревья.
"""

from __future__ import annotations

import logging
import weakref
from typing import TYPE_CHECKING

import numpy as np

from engine.core.math3d import (
    MATRIX_DTYPE,
    Quaternion,
    Vector3,
    compose_matrices,
    identity_matrix,
    invert_matrix,
    transform_point,
    trs_from_matrix,
)

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.game_object import GameObject

__all__ = ["Transform", "TransformStore"]

_log = logging.getLogger("engine.core.transform")

ROOT_INDEX = -1


class TransformStore:
    """Плоское хранилище трансформов одной сцены.

    Все данные лежат в numpy-массивах, индексируемых «слотом» трансформа.
    Слоты переиспользуются через free-list, поэтому длительная работа сцены
    не приводит к росту памяти.

    Args:
        initial_capacity: Начальное число слотов.
        growth_factor: Во сколько раз увеличивать массивы при нехватке мест.
    """

    __slots__ = (
        "_alive",
        "_capacity",
        "_children",
        "_depth",
        "_dirty_local",
        "_dirty_world",
        "_full_dirty",
        "version",
        "_free",
        "_growth_factor",
        "_local",
        "_parent",
        "_positions",
        "_rotations",
        "_scales",
        "_count",
        "_world",
    )

    def __init__(self, initial_capacity: int = 64, growth_factor: float = 1.6) -> None:
        if initial_capacity < 1:
            raise ValueError("initial_capacity должен быть положительным")
        if growth_factor <= 1.0:
            raise ValueError("growth_factor должен быть больше 1")

        self._capacity = int(initial_capacity)
        self._growth_factor = float(growth_factor)
        # Верхняя граница индексов с учётом истории занятых слотов.
        self._count = 0

        capacity = self._capacity
        self._positions = np.zeros((capacity, 3), dtype=np.float32)
        self._rotations = np.zeros((capacity, 4), dtype=np.float32)
        self._rotations[:, 3] = 1.0
        self._scales = np.ones((capacity, 3), dtype=np.float32)
        self._local = np.zeros((capacity, 4, 4), dtype=MATRIX_DTYPE)
        self._world = np.zeros((capacity, 4, 4), dtype=MATRIX_DTYPE)
        self._parent = np.full(capacity, ROOT_INDEX, dtype=np.int32)
        self._depth = np.zeros(capacity, dtype=np.int32)
        self._alive = np.zeros(capacity, dtype=bool)
        self._children: list[list[int]] = [[] for _ in range(capacity)]

        self._free: list[int] = []
        self._dirty_local: set[int] = set()
        self._dirty_world: set[int] = set()
        self._full_dirty = False
        #: Счётчик изменений: растёт при любой правке/создании/удалении трансформа.
        self.version = 0

    # ------------------------------------------------------------------ #
    # Свойства-счётчики
    # ------------------------------------------------------------------ #
    @property
    def capacity(self) -> int:
        """Сколько слотов выделено."""
        return self._capacity

    @property
    def alive_count(self) -> int:
        """Сколько трансформов реально живо."""
        return int(self._alive.sum())

    @property
    def dirty_count(self) -> int:
        """Сколько трансформов требует пересчёта мировых матриц."""
        return len(self._dirty_world)

    # ------------------------------------------------------------------ #
    # Жизненный цикл слотов
    # ------------------------------------------------------------------ #
    def create(
        self,
        parent: int = ROOT_INDEX,
        position: Vector3 | None = None,
        rotation: Quaternion | None = None,
        scale: Vector3 | None = None,
    ) -> int:
        """Занять слот под новый трансформ.

        Args:
            parent: Индекс родителя или :data:`ROOT_INDEX`.
            position: Локальная позиция.
            rotation: Локальное вращение.
            scale: Локальный масштаб.

        Returns:
            Индекс созданного трансформа.

        Raises:
            ValueError: Если родитель не существует.
        """
        if parent != ROOT_INDEX and not self.is_alive(parent):
            raise ValueError(f"Родительский трансформ {parent} не существует")

        if self._free:
            index = self._free.pop()
        else:
            index = self._count
            self._count += 1
            if index >= self._capacity:
                self._grow()

        self._alive[index] = True
        self._parent[index] = parent
        self._children[index] = []
        self._depth[index] = 0 if parent == ROOT_INDEX else self._depth[parent] + 1

        pos = position or Vector3.zero()
        rot = rotation or Quaternion.identity()
        scl = scale or Vector3.one()
        self._positions[index] = (pos.x, pos.y, pos.z)
        self._rotations[index] = (rot.x, rot.y, rot.z, rot.w)
        self._scales[index] = (scl.x, scl.y, scl.z)

        if parent != ROOT_INDEX:
            self._children[parent].append(index)

        self._dirty_local.add(index)
        self._dirty_world.add(index)
        self.version += 1
        return index

    def release(self, index: int) -> bool:
        """Освободить слот трансформа.

        Дети (если остались) отвязываются и становятся корнями: штатно сцена
        уничтожает поддерево целиком, поэтому такая ситуация — признак ошибки
        вызывающего кода и логируется.

        Returns:
            ``True``, если слот был жив и освобождён.
        """
        if not self.is_alive(index):
            return False

        children = self._children[index]
        if children:
            _log.warning(
                "Освобождение трансформа %s с %d живыми детьми: они станут корнями",
                index,
                len(children),
            )
            for child in list(children):
                self.set_parent(child, ROOT_INDEX, keep_world=True)

        parent = int(self._parent[index])
        if parent != ROOT_INDEX and self.is_alive(parent):
            siblings = self._children[parent]
            if index in siblings:
                siblings.remove(index)

        self._children[index] = []
        self._alive[index] = False
        self._parent[index] = ROOT_INDEX
        self._depth[index] = 0
        self._dirty_local.discard(index)
        self._dirty_world.discard(index)
        self.version += 1
        self._free.append(index)
        return True

    def discard(self, index: int) -> bool:
        """Освободить слот, не обходя детей.

        Используется, когда поддерево уже перенесено в другое хранилище или
        уничтожено вызывающим кодом: обход детей здесь был бы лишней работой
        и мог бы испортить уже ненужные данные.

        Returns:
            ``True``, если слот был жив и освобождён.
        """
        if not self.is_alive(index):
            return False

        parent = int(self._parent[index])
        if parent != ROOT_INDEX and self.is_alive(parent):
            siblings = self._children[parent]
            if index in siblings:
                siblings.remove(index)

        self._children[index] = []
        self._alive[index] = False
        self._parent[index] = ROOT_INDEX
        self._depth[index] = 0
        self._dirty_local.discard(index)
        self._dirty_world.discard(index)
        self.version += 1
        self._free.append(index)
        return True

    def is_alive(self, index: int) -> bool:
        """``True``, если слот существует и занят."""
        return 0 <= index < self._count and bool(self._alive[index])

    def _grow(self) -> None:
        """Увеличить все массивы, сохранив данные."""
        old = self._capacity
        new = max(old + 1, int(old * self._growth_factor))
        self._positions = _resize(self._positions, (new, 3))
        self._rotations = _resize(self._rotations, (new, 4))
        self._scales = _resize(self._scales, (new, 3), fill=1.0)
        self._local = _resize(self._local, (new, 4, 4))
        self._world = _resize(self._world, (new, 4, 4))
        self._parent = _resize(self._parent, (new,), fill=ROOT_INDEX, dtype=np.int32)
        self._depth = _resize(self._depth, (new,), dtype=np.int32)
        self._alive = _resize(self._alive, (new,), dtype=bool)
        self._children.extend([] for _ in range(new - old))
        self._capacity = new

    # ------------------------------------------------------------------ #
    # Локальные значения
    # ------------------------------------------------------------------ #
    def local_position(self, index: int) -> Vector3:
        """Прочитать локальную позицию."""
        self._require_alive(index)
        data = self._positions[index]
        return Vector3(float(data[0]), float(data[1]), float(data[2]))

    def set_local_position(self, index: int, value: Vector3) -> None:
        """Записать локальную позицию и пометить трансформ изменённым."""
        self._require_alive(index)
        self._positions[index] = (value.x, value.y, value.z)
        self.mark_local_dirty(index)

    def local_rotation(self, index: int) -> Quaternion:
        """Прочитать локальное вращение."""
        self._require_alive(index)
        data = self._rotations[index]
        return Quaternion(
            float(data[0]), float(data[1]), float(data[2]), float(data[3])
        )

    def set_local_rotation(self, index: int, value: Quaternion) -> None:
        """Записать локальное вращение и пометить трансформ изменённым."""
        self._require_alive(index)
        self._rotations[index] = (value.x, value.y, value.z, value.w)
        self.mark_local_dirty(index)

    def local_scale(self, index: int) -> Vector3:
        """Прочитать локальный масштаб."""
        self._require_alive(index)
        data = self._scales[index]
        return Vector3(float(data[0]), float(data[1]), float(data[2]))

    def set_local_scale(self, index: int, value: Vector3) -> None:
        """Записать локальный масштаб и пометить трансформ изменённым."""
        self._require_alive(index)
        self._scales[index] = (value.x, value.y, value.z)
        self.mark_local_dirty(index)

    def set_local_batch(
        self,
        indices: np.ndarray,
        positions: np.ndarray | None = None,
        rotations: np.ndarray | None = None,
    ) -> None:
        """Записать локальные позиции и/или вращения пачкой (без цикла по объектам).

        Args:
            indices: Индексы живых слотов, ``(N,)``.
            positions: Локальные позиции ``(N, 3)`` или ``None``.
            rotations: Локальные кватернионы xyzw ``(N, 4)`` или ``None``.
        """
        if len(indices) == 0:
            return
        if positions is not None:
            self._positions[indices] = positions
        if rotations is not None:
            self._rotations[indices] = rotations
        touched = [int(i) for i in indices]
        self._dirty_local.update(touched)
        self._dirty_world.update(touched)
        self.version += 1

    def get_local_batch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Прочитать локальные позиции ``(N, 3)`` и вращения ``(N, 4)`` пачкой.

        Возвращает копии (float32); индексы должны принадлежать живым слотам.
        """
        return self._positions[indices], self._rotations[indices]

    def mark_local_dirty(self, index: int) -> None:
        """Пометить локальные данные изменёнными (мировые пересчитаются)."""
        if not self.is_alive(index):
            return
        self._dirty_local.add(index)
        self._dirty_world.add(index)
        self.version += 1

    def mark_world_dirty(self, index: int) -> None:
        """Пометить поддерево требующим пересчёта мировых матриц."""
        if self.is_alive(index):
            self._dirty_world.add(index)
            self.version += 1

    def mark_all_dirty(self) -> None:
        """Полностью инвалидировать кэш мировых матриц."""
        self._full_dirty = True
        self.version += 1

    # ------------------------------------------------------------------ #
    # Иерархия
    # ------------------------------------------------------------------ #
    def parent_index(self, index: int) -> int:
        """Индекс родителя или :data:`ROOT_INDEX`."""
        self._require_alive(index)
        return int(self._parent[index])

    def child_indices(self, index: int) -> tuple[int, ...]:
        """Индексы детей в порядке добавления."""
        self._require_alive(index)
        return tuple(self._children[index])

    def depth(self, index: int) -> int:
        """Глубина трансформа в иерархии (у корня — 0)."""
        self._require_alive(index)
        return int(self._depth[index])

    def set_parent(self, index: int, parent: int, *, keep_world: bool = True) -> None:
        """Перевесить трансформ под другого родителя.

        Args:
            index: Трансформ, который перевешивается.
            parent: Новый родитель или :data:`ROOT_INDEX`.
            keep_world: Сохранить мировое положение, пересчитав локальные
                значения (иначе объект «прыгнет» к родителю).

        Raises:
            ValueError: Если родитель не существует или создаётся цикл.
        """
        self._require_alive(index)
        if parent != ROOT_INDEX and not self.is_alive(parent):
            raise ValueError(f"Родительский трансформ {parent} не существует")
        if parent == index:
            raise ValueError("Трансформ не может быть родителем сам себе")

        current = int(self._parent[index])
        if current == parent:
            return

        # Защита от циклов: поднимаемся от нового родителя к корню.
        probe = parent
        while probe != ROOT_INDEX:
            if probe == index:
                raise ValueError("Нельзя сделать потомка родителем: цикл в иерархии")
            probe = int(self._parent[probe])

        if keep_world:
            world = self.world_matrix(index)
            local = (
                world
                if parent == ROOT_INDEX
                else invert_matrix(self.world_matrix(parent)) @ world
            )
            position, rotation, scale = trs_from_matrix(local)
        else:
            position = self.local_position(index)
            rotation = self.local_rotation(index)
            scale = self.local_scale(index)

        if current != ROOT_INDEX and self.is_alive(current):
            siblings = self._children[current]
            if index in siblings:
                siblings.remove(index)

        self._parent[index] = parent
        if parent != ROOT_INDEX:
            self._children[parent].append(index)

        self._positions[index] = (position.x, position.y, position.z)
        self._rotations[index] = (rotation.x, rotation.y, rotation.z, rotation.w)
        self._scales[index] = (scale.x, scale.y, scale.z)

        self._refresh_depths(index)
        self._dirty_local.add(index)
        self._dirty_world.add(index)
        self.version += 1

    def _refresh_depths(self, index: int) -> None:
        """Пересчитать глубины поддерева (после смены родителя)."""
        stack = [index]
        while stack:
            current = stack.pop()
            parent = int(self._parent[current])
            self._depth[current] = (
                0 if parent == ROOT_INDEX else int(self._depth[parent]) + 1
            )
            stack.extend(self._children[current])

    # ------------------------------------------------------------------ #
    # Матрицы
    # ------------------------------------------------------------------ #
    def local_matrix(self, index: int) -> np.ndarray:
        """Локальная матрица трансформа (копия ``(4, 4)``)."""
        self._require_alive(index)
        if index in self._dirty_local:
            self._compose_dirty()
        return self._local[index].copy()

    def world_matrix(self, index: int) -> np.ndarray:
        """Мировая матрица; при необходимости сначала обновляет кэш."""
        self._require_alive(index)
        if self._dirty_local or self._dirty_world or self._full_dirty:
            self.update_world_matrices()
        return self._world[index].copy()

    def forward(self, index: int) -> Vector3:
        """Направление «вперёд» (третий столбец мировой матрицы, ось Z)."""
        matrix = self.world_matrix(index)
        return Vector3(float(matrix[0, 2]), float(matrix[1, 2]), float(matrix[2, 2]))

    def update_world_matrices(self) -> int:
        """Пересчитать мировые матрицы для изменившихся поддеревьев.

        Returns:
            Число обновлённых мировых матриц (``0``, если ничего не менялось).
        """
        if not self._dirty_local and not self._dirty_world and not self._full_dirty:
            return 0

        if self._dirty_local:
            self._compose_dirty()

        alive = int(self._alive.sum())
        affected: set[int]
        if self._full_dirty or len(self._dirty_world) > max(1, alive // 2):
            self._full_dirty = False
            affected = {int(i) for i in np.flatnonzero(self._alive)}
        else:
            affected = set()
            stack = [i for i in self._dirty_world if self.is_alive(i)]
            while stack:
                current = stack.pop()
                if current in affected:
                    continue
                affected.add(current)
                stack.extend(self._children[current])
        self._dirty_world.clear()
        self._dirty_local.clear()

        if not affected:
            return 0

        levels: dict[int, list[int]] = {}
        for index in affected:
            levels.setdefault(int(self._depth[index]), []).append(index)

        for level in sorted(levels):
            indices = np.fromiter(
                levels[level], dtype=np.intp, count=len(levels[level])
            )
            parents = self._parent[indices]
            root_mask = parents == ROOT_INDEX
            if root_mask.any():
                roots = indices[root_mask]
                self._world[roots] = self._local[roots]
            child_mask = ~root_mask
            if child_mask.any():
                children = indices[child_mask]
                parent_indices = parents[child_mask]
                self._world[children] = np.matmul(
                    self._world[parent_indices], self._local[children]
                )
        return len(affected)

    def _compose_dirty(self) -> None:
        """Собрать локальные матрицы для изменённых трансформов (batch)."""
        if not self._dirty_local:
            return
        indices = np.fromiter(
            sorted(self._dirty_local), dtype=np.intp, count=len(self._dirty_local)
        )
        self._local[indices] = compose_matrices(
            self._positions[indices], self._rotations[indices], self._scales[indices]
        )

    def transform_point(self, index: int, point: Vector3) -> Vector3:
        """Перевести точку из локального пространства в мировое."""
        return transform_point(self.world_matrix(index), point)

    def _require_alive(self, index: int) -> None:
        if not self.is_alive(index):
            raise ValueError(f"Трансформ {index} не существует или уже освобождён")

    def __repr__(self) -> str:
        return (
            f"<TransformStore alive={self.alive_count} capacity={self._capacity} "
            f"dirty={self.dirty_count}>"
        )


class Transform:
    """Фасад трансформа игрового объекта.

    Объект лёгкий: он лишь ссылается на :class:`TransformStore` сцены и слот
    в нём, поэтому создавать его «на лету» (например, для родителя) безопасно.

    Example:
        >>> transform.position = Vector3(0, 2, 0)
        >>> child.set_parent(transform, keep_world=False)
    """

    __slots__ = ("_game_object_ref", "_index", "_store")

    def __init__(
        self,
        store: TransformStore,
        index: int,
        game_object: GameObject | None = None,
    ) -> None:
        self._store = store
        self._index = index
        self._game_object_ref = (
            weakref.ref(game_object) if game_object is not None else None
        )

    # ------------------------------------------------------------------ #
    # Идентичность
    # ------------------------------------------------------------------ #
    @property
    def store(self) -> TransformStore:
        """Хранилище, которому принадлежит трансформ."""
        return self._store

    @property
    def index(self) -> int:
        """Индекс слота в хранилище."""
        return self._index

    @property
    def game_object(self) -> GameObject | None:
        """Игровой объект-владелец (``None``, если объект уже удалён)."""
        if self._game_object_ref is None:
            return None
        return self._game_object_ref()

    @property
    def is_valid(self) -> bool:
        """``True``, пока слот трансформа жив."""
        return self._store.is_alive(self._index)

    # ------------------------------------------------------------------ #
    # Локальные значения
    # ------------------------------------------------------------------ #
    @property
    def position(self) -> Vector3:
        """Локальная позиция относительно родителя."""
        return self._store.local_position(self._index)

    @position.setter
    def position(self, value: Vector3) -> None:
        self._store.set_local_position(self._index, value)

    @property
    def rotation(self) -> Quaternion:
        """Локальное вращение относительно родителя."""
        return self._store.local_rotation(self._index)

    @rotation.setter
    def rotation(self, value: Quaternion) -> None:
        self._store.set_local_rotation(self._index, value)

    @property
    def scale(self) -> Vector3:
        """Локальный масштаб."""
        return self._store.local_scale(self._index)

    @scale.setter
    def scale(self, value: Vector3) -> None:
        self._store.set_local_scale(self._index, value)

    @property
    def local_matrix(self) -> np.ndarray:
        """Локальная матрица ``(4, 4)`` (копия)."""
        return self._store.local_matrix(self._index)

    @property
    def world_matrix(self) -> np.ndarray:
        """Мировая матрица ``(4, 4)`` (копия, кэш синхронизируется лениво)."""
        return self._store.world_matrix(self._index)

    @property
    def world_position(self) -> Vector3:
        """Мировая позиция (последний столбец мировой матрицы)."""
        matrix = self._store.world_matrix(self._index)
        return Vector3(float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3]))

    @property
    def world_rotation(self) -> Quaternion:
        """Мировое вращение (из мировой матрицы, масштаб убирается)."""
        matrix = self._store.world_matrix(self._index)
        _, rotation, _ = trs_from_matrix(matrix)
        return rotation

    def set_world_pose(self, position: Vector3, rotation: Quaternion) -> None:
        """Выставить мировые позицию и вращение (масштаб не меняется).

        Для объекта с родителем локальная поза вычисляется через обратную
        мировую матрицу родителя.
        """
        parent = self._store.parent_index(self._index)
        if parent == ROOT_INDEX:
            self.position = position
            self.rotation = rotation
            return
        parent_matrix = self._store.world_matrix(parent)
        _, parent_rot, _ = trs_from_matrix(parent_matrix)
        local_pos = transform_point(invert_matrix(parent_matrix), position)
        self.position = local_pos
        self.rotation = parent_rot.conjugate * rotation

    @property
    def forward(self) -> Vector3:
        """Мировое направление «вперёд» (ось Z)."""
        return self._store.forward(self._index)

    # ------------------------------------------------------------------ #
    # Иерархия
    # ------------------------------------------------------------------ #
    @property
    def parent(self) -> Transform | None:
        """Родительский трансформ или ``None`` для корня."""
        parent_index = self._store.parent_index(self._index)
        if parent_index == ROOT_INDEX:
            return None
        return Transform(self._store, parent_index)

    def set_parent(self, parent: Transform | None, *, keep_world: bool = True) -> None:
        """Назначить родителя (``None`` — сделать корнем).

        Args:
            parent: Новый родитель или ``None``.
            keep_world: Сохранить мировое положение объекта.
        """
        target = ROOT_INDEX if parent is None else parent.index
        if parent is not None and parent.store is not self._store:
            raise ValueError("Родитель принадлежит другой сцене")
        self._store.set_parent(self._index, target, keep_world=keep_world)

    @property
    def children(self) -> tuple[Transform, ...]:
        """Дочерние трансформы."""
        return tuple(
            Transform(self._store, index)
            for index in self._store.child_indices(self._index)
        )

    @property
    def depth(self) -> int:
        """Глубина в иерархии (0 — корень)."""
        return self._store.depth(self._index)

    # ------------------------------------------------------------------ #
    # Утилиты
    # ------------------------------------------------------------------ #
    def translate(self, offset: Vector3) -> None:
        """Сместить объект в локальных координатах родителя."""
        self.position = self.position + offset

    def rotate(self, axis: Vector3, degrees: float) -> None:
        """Домножить текущее вращение на поворот вокруг оси."""
        self.rotation = Quaternion.from_axis_angle(axis, degrees) * self.rotation

    def local_to_world(self, point: Vector3) -> Vector3:
        """Перевести точку из локальных координат объекта в мировые."""
        return transform_point(self._store.world_matrix(self._index), point)

    def world_to_local(self, point: Vector3) -> Vector3:
        """Перевести точку из мировых координат в локальные координаты объекта."""
        inverse = invert_matrix(self._store.world_matrix(self._index))
        return transform_point(inverse, point)

    def reset_local(self) -> None:
        """Вернуть локальные позицию, вращение и масштаб к значениям по умолчанию."""
        self.position = Vector3.zero()
        self.rotation = Quaternion.identity()
        self.scale = Vector3.one()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transform):
            return NotImplemented
        return self._store is other._store and self._index == other._index

    def __hash__(self) -> int:
        return hash((id(self._store), self._index))

    def __repr__(self) -> str:
        owner = self.game_object
        name = owner.name if owner is not None else "?"
        return f"<Transform '{name}' index={self._index} valid={self.is_valid}>"


def _resize(
    array: np.ndarray,
    shape: tuple[int, ...],
    *,
    fill: float = 0.0,
    dtype: type | None = None,
) -> np.ndarray:
    """Создать массив новой формы, скопировав данные и заполнив хвост.

    Args:
        array: Исходный массив.
        shape: Новая форма.
        fill: Значение для новых элементов.
        dtype: Тип результата; по умолчанию тип исходного массива.

    Returns:
        Новый массив нужной формы.
    """
    target_dtype = dtype or array.dtype
    result = np.full(shape, fill, dtype=target_dtype)
    slices = tuple(slice(0, size) for size in array.shape)
    result[slices] = array
    return result


def identity_world() -> np.ndarray:
    """Единичная мировая матрица (удобно для тестов и заглушек)."""
    return identity_matrix()
