"""Данные твёрдого тела (без привязки к ECS)."""

from __future__ import annotations

import numpy as np

from engine.physics.mathutil import IDENTITY_QUAT, quat_to_matrix
from engine.physics.store import BODY_CODES, BodyStore

BODY_TYPES = ("static", "kinematic", "dynamic")


def _array_property(name: str, doc: str = ""):
    """Свойство-view строки (n,)-хранилища: чтение отдаёт view, запись копирует."""

    def get(self: RigidBody) -> np.ndarray:
        return getattr(self._store, name)[self._row]

    def set_(self: RigidBody, value) -> None:
        getattr(self._store, name)[self._row] = value

    return property(get, set_, doc=doc)


def _float_property(name: str):
    def get(self: RigidBody) -> float:
        return float(getattr(self._store, name)[self._row])

    def set_(self: RigidBody, value) -> None:
        getattr(self._store, name)[self._row] = value

    return property(get, set_)


def _bool_property(name: str):
    def get(self: RigidBody) -> bool:
        return bool(getattr(self._store, name)[self._row])

    def set_(self: RigidBody, value) -> None:
        getattr(self._store, name)[self._row] = value

    return property(get, set_)


class RigidBody:
    """Состояние твёрдого тела.

    Тонкий фасад над строкой :class:`~engine.physics.store.BodyStore`: одиночное
    тело держит собственное хранилище на одну строку, ``PhysicsWorld.add_body``
    переносит строку в общее хранилище мира (тот же Python-объект, меняются
    только ``_store``/``_row``). Векторные свойства (``position``, ``rotation``,
    ``linear_velocity``, ``angular_velocity``, ``inv_inertia_*``) возвращают
    **view** строки: запись в него меняет тело, но view нельзя хранить — он
    устаревает после ``add_body``/``remove_body`` мира (рост ёмкости, swap-remove).

    Накопленные импульсы для warm starting хранятся не в теле, а в кэше
    контактов мира (ключ ``(cid_a, cid_b, индекс точки)``).

    Демпфирование: ``linear_damping`` / ``angular_damping`` = d — затухание
    ``v ← v / (1 + d·dt)`` за шаг, то есть e-кратное за ``1/d`` секунд:
    d=1 → v/v0 ≈ 0.37 за 1 с (и ≈ 0.61 за 0.5 с).

    ``mass`` и ``freeze_rotation`` — свойства: запись на живом теле сразу
    пересчитывает ``inv_mass`` и обратную инерцию.
    """

    __slots__ = ("_store", "_row", "_btype", "_inertia_local", "__weakref__")

    def __init__(
        self,
        body_type: str = "dynamic",
        mass: float = 1.0,
        position=(0.0, 0.0, 0.0),
        rotation=IDENTITY_QUAT,
        linear_velocity=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
        linear_damping: float = 0.05,
        angular_damping: float = 0.05,
        gravity_scale: float = 1.0,
        use_gravity: bool = True,
        freeze_rotation: bool = False,
        sleeping: bool = False,
        sleep_timer: float = 0.0,
        can_sleep: bool = True,
        inv_mass: float = 0.0,
        inv_inertia_local=None,
        inv_inertia_world=None,
    ) -> None:
        if body_type not in BODY_TYPES:
            raise ValueError(f"неизвестный тип тела: {body_type!r}")
        if body_type == "dynamic":
            if not mass > 0.0:
                raise ValueError("масса динамического тела должна быть > 0")
            inv_mass = 1.0 / mass
        store = BodyStore(1)
        row = store._push(self)
        self._store = store
        self._row = row
        self._btype = body_type
        self._inertia_local: np.ndarray | None = None
        store.body_type[row] = BODY_CODES[body_type]
        store.mass[row] = mass
        store.inv_mass[row] = inv_mass
        store.position[row] = position
        store.rotation[row] = rotation
        store.vel[row, :3] = linear_velocity
        store.vel[row, 3:] = angular_velocity
        store.linear_damping[row] = linear_damping
        store.angular_damping[row] = angular_damping
        store.gravity_scale[row] = gravity_scale
        store.use_gravity[row] = use_gravity
        store.freeze_rotation[row] = freeze_rotation
        store.sleeping[row] = sleeping
        store.sleep_timer[row] = sleep_timer
        store.can_sleep[row] = can_sleep
        if inv_inertia_local is not None:
            store.inv_inertia_local[row] = inv_inertia_local
        if inv_inertia_world is not None:
            store.inv_inertia_world[row] = inv_inertia_world

    def __repr__(self) -> str:
        return (
            f"RigidBody(body_type={self._btype!r}, mass={self.mass}, "
            f"position={self.position.tolist()}, sleeping={self.sleeping})"
        )

    # ------------------------------------------------------------- свойства

    position = _array_property("position")
    rotation = _array_property("rotation")
    inv_inertia_local = _array_property("inv_inertia_local")
    inv_inertia_world = _array_property("inv_inertia_world")
    linear_damping = _float_property("linear_damping")
    angular_damping = _float_property("angular_damping")
    gravity_scale = _float_property("gravity_scale")
    sleep_timer = _float_property("sleep_timer")
    inv_mass = _float_property("inv_mass")
    use_gravity = _bool_property("use_gravity")
    sleeping = _bool_property("sleeping")
    can_sleep = _bool_property("can_sleep")

    @property
    def linear_velocity(self) -> np.ndarray:
        return self._store.vel[self._row, :3]

    @linear_velocity.setter
    def linear_velocity(self, value) -> None:
        self._store.vel[self._row, :3] = value

    @property
    def angular_velocity(self) -> np.ndarray:
        return self._store.vel[self._row, 3:]

    @angular_velocity.setter
    def angular_velocity(self, value) -> None:
        self._store.vel[self._row, 3:] = value

    @property
    def body_type(self) -> str:
        return self._btype

    @body_type.setter
    def body_type(self, value: str) -> None:
        if value not in BODY_TYPES:
            raise ValueError(f"неизвестный тип тела: {value!r}")
        self._btype = value
        self._store.body_type[self._row] = BODY_CODES[value]

    @property
    def mass(self) -> float:
        return float(self._store.mass[self._row])

    @mass.setter
    def mass(self, value: float) -> None:
        old = self.mass
        if self.is_dynamic and not value > 0.0:
            raise ValueError("масса динамического тела должна быть > 0")
        self._store.mass[self._row] = value
        if self.is_dynamic:
            self.inv_mass = 1.0 / value
            if self._inertia_local is not None:
                # тензор инерции пропорционален массе при той же форме
                self._inertia_local = self._inertia_local * (value / old)
                self._refresh_inertia()

    @property
    def freeze_rotation(self) -> bool:
        return bool(self._store.freeze_rotation[self._row])

    @freeze_rotation.setter
    def freeze_rotation(self, value: bool) -> None:
        old = self.freeze_rotation
        self._store.freeze_rotation[self._row] = value
        if old != bool(value):
            self._refresh_inertia()

    @property
    def is_dynamic(self) -> bool:
        return self._btype == "dynamic"

    @property
    def is_static(self) -> bool:
        return self._btype == "static"

    @property
    def is_kinematic(self) -> bool:
        return self._btype == "kinematic"

    # --------------------------------------------------------------- методы

    def set_inertia(self, inertia_local: np.ndarray) -> None:
        """Задаёт тензор инерции в локальных осях (3x3)."""
        self._inertia_local = np.array(inertia_local, dtype=np.float64)
        self._refresh_inertia()

    def _refresh_inertia(self) -> None:
        """Пересчитывает обратную инерцию по сохранённому тензору и флагам."""
        if (
            self._inertia_local is not None
            and self.is_dynamic
            and not self.freeze_rotation
        ):
            self.inv_inertia_local = np.linalg.inv(self._inertia_local)
        else:
            self.inv_inertia_local = np.zeros((3, 3))
        self.update_world_inertia()

    def update_world_inertia(self) -> None:
        """Пересчитывает обратный тензор инерции в мировых осях."""
        rot = quat_to_matrix(self.rotation)
        self.inv_inertia_world = rot @ self.inv_inertia_local @ rot.T

    def wake(self) -> None:
        """Будит тело."""
        self.sleeping = False
        self.sleep_timer = 0.0

    def apply_impulse(self, impulse, point=None) -> None:
        """Мгновенный импульс (в центр масс или в мировую точку ``point``)."""
        if not self.is_dynamic:
            return
        j = np.asarray(impulse, dtype=np.float64)
        self.wake()
        self.linear_velocity = self.linear_velocity + j * self.inv_mass
        if point is not None:
            r = np.asarray(point, dtype=np.float64) - self.position
            self.angular_velocity = self.angular_velocity + (
                self.inv_inertia_world @ np.cross(r, j)
            )
