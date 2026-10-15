"""Коллайдер: форма, материал и фильтры столкновений."""

from __future__ import annotations

import numpy as np

from engine.physics.material import PhysicsMaterial
from engine.physics.mathutil import IDENTITY_QUAT, quat_to_matrix
from engine.physics.shape import Shape
from engine.physics.store import MASK64


def _zeros3() -> np.ndarray:
    return np.zeros(3)


class Collider:
    """Форма, привязанная к телу.

    ``layer`` — битовая маска слоёв коллайдера (по умолчанию бит 0), ``mask`` —
    слои, с которыми он взаимодействует. Пара сталкивается, если
    ``a.layer & b.mask`` и ``b.layer & a.mask`` ненулевые.

    ``layer``, ``mask`` и ``is_trigger`` можно менять и в мире: запись сразу
    попадает в ``ColliderStore``. Остальная конфигурация (``shape``, ``offset``,
    ``rotation_offset``) копируется в хранилище при ``add_body`` и после этого не
    отслеживается: изменить коллайдер значит пересоздать его.

    ``world_position`` / ``world_rotation`` / ``world_aabb`` пока коллайдер в мире —
    view строк ``ColliderStore`` (``world_aabb`` — пара view ``(lo, hi)``); вне
    мира это обычные массивы. Как и у ``RigidBody``, view нельзя хранить дольше
    ближайшего ``add_body``/``remove_body``.
    """

    __slots__ = (
        "shape", "material", "offset", "rotation_offset", "_is_trigger", "_layer",
        "_mask", "_rot_offset_matrix", "_store", "_row", "_wp", "_wr", "_wa",
    )

    def __init__(
        self,
        shape: Shape,
        material: PhysicsMaterial | None = None,
        is_trigger: bool = False,
        layer: int = 1,
        mask: int = 0xFFFFFFFF,
        offset=None,
        rotation_offset=None,
    ) -> None:
        self.shape = shape
        self.material = PhysicsMaterial() if material is None else material
        self._is_trigger = is_trigger
        self._layer = layer
        self._mask = mask
        self.offset = np.array(
            _zeros3() if offset is None else offset, dtype=np.float64
        )
        self.rotation_offset = np.array(
            IDENTITY_QUAT if rotation_offset is None else rotation_offset,
            dtype=np.float64,
        )
        self._rot_offset_matrix = quat_to_matrix(self.rotation_offset)
        # Привязка к ColliderStore мира и «автономное» мировое состояние.
        self._store = None
        self._row = -1
        self._wp = _zeros3()
        self._wr = np.eye(3)
        self._wa = (np.zeros(3), np.zeros(3))

    def __repr__(self) -> str:
        return (
            f"Collider(shape={self.shape!r}, material={self.material!r}, "
            f"is_trigger={self._is_trigger!r}, layer={self._layer!r}, "
            f"mask={self._mask!r}, offset={self.offset!r}, "
            f"rotation_offset={self.rotation_offset!r})"
        )

    @property
    def is_trigger(self) -> bool:
        return self._is_trigger

    @is_trigger.setter
    def is_trigger(self, value: bool) -> None:
        self._is_trigger = value
        if self._store is not None:
            self._store.is_trigger[self._row] = bool(value)

    @property
    def layer(self) -> int:
        return self._layer

    @layer.setter
    def layer(self, value: int) -> None:
        self._layer = value
        if self._store is not None:
            self._store.layer[self._row] = int(value) & MASK64

    @property
    def mask(self) -> int:
        return self._mask

    @mask.setter
    def mask(self, value: int) -> None:
        self._mask = value
        if self._store is not None:
            self._store.mask[self._row] = int(value) & MASK64

    @property
    def world_position(self) -> np.ndarray:
        store = self._store
        return self._wp if store is None else store.world_pos[self._row]

    @world_position.setter
    def world_position(self, value) -> None:
        store = self._store
        if store is None:
            self._wp = np.asarray(value, dtype=np.float64)
        else:
            store.world_pos[self._row] = value

    @property
    def world_rotation(self) -> np.ndarray:
        store = self._store
        return self._wr if store is None else store.world_rot[self._row]

    @world_rotation.setter
    def world_rotation(self, value) -> None:
        store = self._store
        if store is None:
            self._wr = np.asarray(value, dtype=np.float64)
        else:
            store.world_rot[self._row] = value

    @property
    def world_aabb(self) -> tuple[np.ndarray, np.ndarray]:
        store = self._store
        if store is None:
            return self._wa
        row = self._row
        return store.aabb_lo[row], store.aabb_hi[row]

    @world_aabb.setter
    def world_aabb(self, value) -> None:
        lo, hi = value
        store = self._store
        if store is None:
            self._wa = (np.asarray(lo, dtype=np.float64),
                        np.asarray(hi, dtype=np.float64))
        else:
            store.aabb_lo[self._row] = lo
            store.aabb_hi[self._row] = hi
            store.version += 1

    def update_world(self, body_position, body_rotation_matrix) -> None:
        """Пересчитывает мировую позу и AABB по позе тела."""
        self.world_position = body_position + body_rotation_matrix @ self.offset
        self.world_rotation = body_rotation_matrix @ self._rot_offset_matrix
        self.world_aabb = self.shape.aabb(self.world_position, self.world_rotation)

    def local_inertia(self, mass: float) -> np.ndarray:
        """Тензор инерции относительно начала тела (с учётом смещения/поворота)."""
        inertia = self.shape.inertia_tensor(mass)
        rot = self._rot_offset_matrix
        inertia = rot @ inertia @ rot.T
        d = self.offset
        return inertia + mass * (float(d @ d) * np.eye(3) - np.outer(d, d))
