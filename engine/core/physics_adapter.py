"""Мост между ECS-сценой и миром физики (``scene.physics``).

:class:`ScenePhysicsAdapter` создаёт тела по компонентам
(:mod:`engine.core.physics_components`), перед шагом переносит в мир позы,
изменённые снаружи (скрипты, анимация, телепорт), после шага записывает в
трансформы позы движущихся динамических тел и публикует события физики в
``scene.events`` (через ``emit_deferred`` — обработчики вызываются в ``update``).

Горячий путь (корневые объекты) работает пачками numpy: чтение и запись
трансформов идут через ``TransformStore.get_local_batch/set_local_batch``.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from engine.core.math3d import Quaternion, Vector3
from engine.core.mesh_geometry import load_mesh_geometry
from engine.core.physics_components import (
    CharacterControllerComponent,
    ColliderComponent,
    MeshColliderComponent,
    RigidBodyComponent,
    clamp_character_field,
    has_character_rigidbody_conflict,
)
from engine.physics.body import BODY_TYPES, RigidBody
from engine.physics.bvh import Bvh
from engine.physics.character import CharacterController
from engine.physics.collider import Collider
from engine.physics.material import PhysicsMaterial
from engine.physics.shape import BoxShape, CapsuleShape, SphereShape
from engine.physics.world import ALL_LAYERS, PhysicsWorld

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from collections.abc import Iterable

    from engine.core.game_object import GameObject
    from engine.core.scene import Scene

__all__ = ["ScenePhysicsAdapter"]

_log = logging.getLogger("engine.core.physics_adapter")

_MIN_SIZE = 1e-3
_POSE_EPS = 1e-4
_MIN_SKIN_WIDTH = 1e-3  # нижняя граница skin_width персонажа
_PASSIVE_EPS = 1e-7  # смещение, ниже которого пассивный шаг не пишет позу
_NO_MOTION = (0.0, 0.0, 0.0)  # результат методов движения замороженного персонажа
_SIZE_EPS = 1e-6  # изменение радиуса/высоты капсулы, с которого её пересоздают
# «Живые» поля компонента персонажа, которые адаптер переносит в контроллер.
_LIVE_NUMERIC = (
    "move_speed", "sprint_multiplier", "jump_velocity", "gravity_scale",
    "ground_check_distance", "air_control",
)
_LIVE_FLAGS = ("use_gravity", "snap_to_ground")
# Оба компонента управляют движением объекта: контроллер - кинематикой, тело -
# решателем. Вместе их работа непредсказуема, такой объект в мир не попадает.
_CONFLICT_MESSAGE = "CharacterController и RigidBody несовместимы"


class _Binding:
    """Связь «игровой объект → тело/контроллер мира»."""

    __slots__ = (
        "align", "basis", "body", "character", "component", "frozen", "game_object",
        "handle", "kind", "last_pos", "last_rot", "moved", "parented", "size",
        "tilted",
    )

    def __init__(self, game_object: GameObject, kind: str, handle: int) -> None:
        self.game_object = game_object
        self.kind = kind  # static | kinematic | dynamic | character
        self.handle = handle
        self.body: RigidBody | None = None
        self.character: CharacterController | None = None
        self.component: CharacterControllerComponent | None = None
        # Персонаж «заморожен»: компонент выключен, двойник остаётся в мире,
        # но не двигается (см. ``ScenePhysicsAdapter.sync_structure``).
        self.frozen = False
        self.align = False  # alignment_to_ground компонента
        # Персонаж: размеры капсулы (радиус, высота) после масштаба и байты
        # матрицы 3x3 мировой матрицы, при которых они посчитаны.
        self.size = (0.0, 0.0)
        self.basis = b""
        self.tilted = False  # трансформ сейчас наклонён по нормали земли
        self.moved = False  # скрипт двигал персонажа в текущем шаге
        self.parented = False
        # Последняя поза, согласованная с миром (для мировых/родительских объектов).
        self.last_pos = np.zeros(3)
        self.last_rot = np.array([0.0, 0.0, 0.0, 1.0])


class _MeshBinding:
    """Связь «объект с MeshColliderComponent → mesh-коллайдер мира»."""

    __slots__ = ("game_object", "handle", "matrix", "signature")

    def __init__(self, game_object: GameObject, handle: int, signature: tuple,
                 matrix: bytes) -> None:
        self.game_object = game_object
        self.handle = handle
        self.signature = signature
        self.matrix = matrix  # байты мировой матрицы, при которых поза выставлена


def _basis_scale(basis: list[list[float]]) -> tuple[float, float, float]:
    """Длины столбцов матрицы 3x3 (``tolist()``): мировой масштаб персонажа.

    Чистый Python: для персонажа, который поворачивается каждый кадр, это дешевле
    ``np.linalg.norm``. Коллайдеры считают масштаб в :func:`_world_pose`.
    """
    (a, b, c), (d, e, f), (g, h, i) = basis
    return (
        math.sqrt(a * a + d * d + g * g),
        math.sqrt(b * b + e * e + h * h),
        math.sqrt(c * c + f * f + i * i),
    )


def _world_pose(game_object: GameObject) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Мировые позиция, кватернион xyzw и масштаб объекта."""
    tr = game_object.transform
    matrix = tr.world_matrix
    scale = np.linalg.norm(matrix[:3, :3], axis=0)
    rot = tr.world_rotation
    return (
        matrix[:3, 3].astype(np.float64),
        np.array([rot.x, rot.y, rot.z, rot.w], dtype=np.float64),
        scale.astype(np.float64),
    )


def scaled_capsule_size(
    radius: float, height: float, scale: np.ndarray
) -> tuple[float, float, bool]:
    """Размеры капсулы с учётом мирового масштаба (общая для коллайдера и персонажа).

    ``radius`` умножается на ``max(|sx|, |sz|)``, ``height`` - на ``|sy|``; каждая
    ось не меньше ``_MIN_SIZE`` (нулевой и отрицательный масштаб - зеркало). Если
    после этого ``height`` не больше ``2 * radius`` (капсула без цилиндра), высота
    поднимается до ``2 * radius + 2 * _MIN_SIZE``.

    Args:
        radius: Радиус капсулы без масштаба.
        height: Полная высота капсулы без масштаба.
        scale: Мировой масштаб объекта ``(sx, sy, sz)``.

    Returns:
        ``(radius, height, raised)``: итоговые размеры и признак подъёма высоты.
    """
    # чистые float: функция вызывается каждый кадр для вращающегося персонажа
    sx, sy, sz = (max(abs(float(v)), _MIN_SIZE) for v in scale)
    scaled_radius = max(radius * max(sx, sz), _MIN_SIZE)
    scaled_height = height * sy
    floor = 2.0 * scaled_radius + 2.0 * _MIN_SIZE
    if scaled_height < floor:
        return scaled_radius, floor, True
    return scaled_radius, scaled_height, False


def _build_shape(comp: ColliderComponent, scale: np.ndarray):
    """Форма физики по компоненту с учётом мирового масштаба."""
    s = np.maximum(np.abs(scale), _MIN_SIZE)
    if comp.shape == "box":
        half = np.maximum(np.abs(np.asarray(comp.size[:3])) * s * 0.5, _MIN_SIZE)
        return BoxShape((float(half[0]), float(half[1]), float(half[2])))
    if comp.shape == "sphere":
        return SphereShape(max(comp.radius * float(s.max()), _MIN_SIZE))
    if comp.shape == "capsule":
        radius, height, _ = scaled_capsule_size(comp.radius, comp.height, scale)
        return CapsuleShape(radius, height * 0.5 - radius)
    raise ValueError(f"неизвестная форма коллайдера: {comp.shape!r}")


def _yaw_of(rot: Quaternion) -> Quaternion:
    """Рыскание (поворот вокруг Y) кватерниона: swing-twist без наклона."""
    length = math.hypot(rot.y, rot.w)
    if length < 1e-9:
        return Quaternion.identity()
    return Quaternion(0.0, rot.y / length, 0.0, rot.w / length)


def _aligned_rotation(rot: Quaternion, normal: np.ndarray) -> Quaternion:
    """Поворот «вверх -> нормаль земли» с сохранением рыскания ``rot``.

    Без состояния: по уже наклонённому повороту рыскание восстанавливается
    однозначно (ось наклона горизонтальна), поэтому результат стабилен.
    """
    swing = Quaternion.from_two_vectors(Vector3.up(), Vector3(*map(float, normal)))
    return swing * _yaw_of(rot)


def _unit_quat(value) -> np.ndarray:
    """Нормированный кватернион xyzw; нулевой считается единичным."""
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"rotation_offset: нужно 4 числа, получено {len(value)}")
    norm = float(np.linalg.norm(q))
    if norm < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / norm


class ScenePhysicsAdapter:
    """Физика сцены: создаётся лениво через ``Scene.physics``.

    Attributes:
        world: Мир физики.
        scene: Сцена-владелец.
    """

    def __init__(
        self,
        scene: Scene,
        gravity=(0.0, -9.81, 0.0),
        fixed_dt: float = 1.0 / 60.0,
    ) -> None:
        self.scene = scene
        self.world = PhysicsWorld(gravity=gravity, fixed_dt=fixed_dt)
        self.enabled = True
        self._bindings: dict[int, _Binding] = {}  # id объекта -> связь
        self._by_handle: dict[int, GameObject] = {}
        self._dirty: set[int] = set()
        self._all_dirty = True
        self._seen_versions: tuple[int, int] = (-1, -1)
        # Пачки для корневых (не вложенных) объектов.
        self._fast: list[_Binding] = []
        self._fast_idx = np.zeros(0, dtype=np.intp)
        self._fast_pos = np.zeros((0, 3), dtype=np.float32)
        self._fast_rot = np.zeros((0, 4), dtype=np.float32)
        self._fast_dynamic: np.ndarray = np.zeros(0, dtype=np.intp)
        # Заснувшие тела, поза которых уже записана в трансформ (ключ: индекс
        # в быстром наборе или -id привязки). Без этого тело, заснувшее на шаге,
        # остаётся в трансформе на позе предыдущего бодрствующего шага.
        self._sleep_synced: set[int] = set()
        self._slow: list[_Binding] = []
        self._characters: list[_Binding] = []
        self._live_dirty: set[int] = set()  # объекты с правкой «живых» полей
        self._layout_dirty = True
        self._warned: set[int] = set()
        # объект -> причина, по которой CharacterController не создан
        self._failures: dict[int, str] = {}
        # объекты с активными CharacterController и RigidBody одновременно
        self._conflicts: set[int] = set()
        self._clamp_warned: set[tuple[int, str]] = set()
        # Статические mesh-коллайдеры: id объекта -> связь; BVH кэшируются по
        # (ассет, масштаб), чтобы одинаковые экземпляры делили построение.
        self._meshes: dict[int, _MeshBinding] = {}
        self._bvh_cache: dict[tuple, Bvh] = {}

    # ------------------------------------------------------------ состав мира

    def mark_dirty(self, game_object: GameObject | None) -> None:
        """Помечает объект на пересоздание тела (или полный пересмотр)."""
        if game_object is None:
            self._all_dirty = True
        else:
            self._dirty.add(game_object.id)
            self._seen_versions = (-1, -1)

    def mark_live(self, game_object: GameObject | None) -> None:
        """Помечает объект: «живые» поля персонажа правились, тело не пересоздаётся."""
        if game_object is not None:
            self._live_dirty.add(game_object.id)

    def mark_toggled(self) -> None:
        """Включение/выключение ``CharacterControllerComponent``: пересмотр состава.

        Двойник при этом не пересоздаётся: ``sync_structure`` лишь переключает его
        между «активен» и «заморожен» (если не изменился сам состав мира).
        """
        self._seen_versions = (-1, -1)

    def handle_of(self, game_object: GameObject | None) -> int | None:
        """Handle тела объекта (``None``, если тела нет)."""
        if game_object is None:
            return None
        self.sync_structure()
        binding = self._bindings.get(game_object.id)
        return None if binding is None else binding.handle

    def handles_for(self, game_objects: Iterable[GameObject]) -> frozenset[int]:
        """Handle'ы тел указанных объектов (объекты без тела пропускаются)."""
        self.sync_structure()
        result = set()
        for game_object in game_objects:
            binding = self._bindings.get(game_object.id)
            if binding is not None:
                result.add(binding.handle)
            mesh = self._meshes.get(game_object.id)
            if mesh is not None:
                result.add(mesh.handle)
        return frozenset(result)

    def object_of(self, handle: int) -> GameObject | None:
        """Игровой объект по handle мира."""
        return self._by_handle.get(handle)

    def binding_count(self) -> int:
        """Число связанных объектов."""
        return len(self._bindings)

    def sync_structure(self) -> None:
        """Приводит состав мира в соответствие со сценой (если она менялась)."""
        scene = self.scene
        versions = (scene.structure_version, scene.activity_version)
        if versions == self._seen_versions and not self._dirty and not self._all_dirty:
            return
        self._seen_versions = versions
        wanted: dict[int, tuple[GameObject, Any, Any, Any]] = {}
        conflicts: dict[int, GameObject] = {}
        for go in scene.query(CharacterControllerComponent):
            if go.is_destroyed:
                continue
            comp = next(
                (c for c in go.get_components(CharacterControllerComponent)
                 if c.is_active),
                None,
            )
            if comp is None:
                comp = self._frozen_component(go)  # выключенный: двойник заморожен
                if comp is None:
                    continue
                wanted[go.id] = (go, None, [], comp)
            elif has_character_rigidbody_conflict(go):
                conflicts[go.id] = go
            else:
                wanted[go.id] = (go, None, [], comp)
        for oid, go in conflicts.items():
            if oid not in self._conflicts:
                _log.error(
                    "Физика: объект '%s' не зарегистрирован: %s "
                    "(оба компонента управляют движением; уберите один из них)",
                    go.name, _CONFLICT_MESSAGE,
                )
        self._conflicts = set(conflicts)
        for go in scene.query(ColliderComponent):
            if go.id in wanted or go.id in conflicts or go.is_destroyed:
                continue
            cols = [c for c in go.get_components(ColliderComponent) if c.is_active]
            if not cols:
                continue
            rb = next(
                (c for c in go.get_components(RigidBodyComponent) if c.is_active),
                None,
            )
            wanted[go.id] = (go, rb, cols, None)
        for oid in [o for o in self._failures if o not in wanted]:
            del self._failures[oid]
        # Лишние связи и изменённые — убрать (с сохранением состояния).
        carry: dict[int, tuple] = {}
        # Объёмы, которые тела занимали/займут: вокруг них будим спящие тела
        # (при первой загрузке сцены мир пуст и будить некого).
        touched: list[tuple[np.ndarray, np.ndarray]] | None = (
            [] if self._bindings else None
        )
        for oid in list(self._bindings):
            binding = self._bindings[oid]
            if oid not in wanted:
                self._collect_aabbs(oid, touched)
                self._unbind(oid)
            elif (self._all_dirty or oid in self._dirty
                    or self._is_stale(binding, wanted[oid][3])):
                carry[oid] = self._capture(binding)
                self._collect_aabbs(oid, touched)
                self._unbind(oid)
            elif binding.character is not None:
                self._set_frozen(binding, not wanted[oid][3].enabled)
        fresh_characters: list[int] = []
        for oid, (go, rb, cols, ch) in wanted.items():
            if oid in self._bindings:
                continue
            try:
                self._bind(go, rb, cols, ch, carry.get(oid))
            except (ValueError, TypeError) as exc:
                if ch is not None:
                    self._failures[oid] = str(exc)
                if oid not in self._warned:
                    self._warned.add(oid)
                    if ch is not None:
                        _log.error(
                            "Физика: CharacterController объекта '%s' не создан: %s",
                            go.name, exc,
                        )
                    else:
                        _log.warning("Физика: объект '%s' пропущен: %s", go.name, exc)
            else:
                self._failures.pop(oid, None)
                self._warned.discard(oid)
                self._collect_aabbs(oid, touched)
                if ch is not None:
                    fresh_characters.append(oid)
        self._sync_meshes(touched)
        # Персонажи в ``wanted`` идут раньше коллайдеров, и проба земли в
        # конструкторе не видит ещё не добавленную статику: перепроверяем,
        # когда мир собран целиком.
        for oid in fresh_characters:
            self._bindings[oid].character.refresh_ground()
        if touched:
            self.world.wake_overlapping(touched)
        self._dirty.clear()
        self._all_dirty = False
        self._layout_dirty = True

    # ------------------------------------------------------------ mesh-коллайдеры

    def _mesh_signature(
        self, comp: MeshColliderComponent, scale: np.ndarray
    ) -> tuple:
        ref = comp.mesh
        return (
            None if ref is None else (ref.guid, ref.path),
            tuple(np.round(scale, 6).tolist()),
            comp.layer, comp.mask, comp.is_trigger,
            comp.friction, comp.restitution,
        )

    def _mesh_bvh(self, go: GameObject, comp: MeshColliderComponent,
                  scale: np.ndarray) -> Bvh | None:
        """BVH меша с запечённым масштабом (из кэша или строится заново)."""
        ref = comp.mesh
        source = self.scene.asset_source
        if ref is None:
            self._error_once(go, "mesh_none",
                             "Физика: у MeshCollider объекта '%s' не задан меш, "
                             "коллизии нет", go.name)
            return None
        key = (ref.guid, ref.path, tuple(np.round(scale, 6).tolist()))
        bvh = self._bvh_cache.get(key)
        if bvh is not None:
            return bvh
        try:
            if source is None:
                raise FileNotFoundError("у сцены нет базы ассетов")
            vertices, triangles = load_mesh_geometry(source, ref)
            bvh = Bvh(vertices * scale, triangles)
        except (OSError, ValueError, KeyError) as exc:
            self._warn_once(
                go, "mesh_load",
                "Физика: MeshCollider объекта '%s' не создан: %s", go.name, exc,
            )
            return None
        self._bvh_cache[key] = bvh
        return bvh

    def _sync_meshes(
        self, touched: list[tuple[np.ndarray, np.ndarray]] | None
    ) -> None:
        """Приводит mesh-коллайдеры мира в соответствие с компонентами сцены."""
        wanted: dict[int, tuple[GameObject, MeshColliderComponent]] = {}
        for go in self.scene.query(MeshColliderComponent):
            if go.is_destroyed:
                continue
            comp = next(
                (c for c in go.get_components(MeshColliderComponent) if c.is_active),
                None,
            )
            if comp is not None:
                wanted[go.id] = (go, comp)
        world = self.world
        for oid in list(self._meshes):
            if oid not in wanted or self._all_dirty or oid in self._dirty:
                self._drop_mesh(oid, touched)
        for oid, (go, comp) in wanted.items():
            if oid in self._meshes:
                continue
            pos, rot, scale = _world_pose(go)
            bvh = self._mesh_bvh(go, comp, scale)
            if bvh is None:
                continue
            handle = world.add_mesh(
                bvh, pos, rot, layer=comp.layer, mask=comp.mask,
                is_trigger=comp.is_trigger,
                material=PhysicsMaterial(
                    max(0.0, comp.friction), min(1.0, max(0.0, comp.restitution))
                ),
                user_data=go,
            )
            binding = _MeshBinding(
                go, handle, self._mesh_signature(comp, scale),
                go.transform.world_matrix.tobytes(),
            )
            self._meshes[oid] = binding
            self._by_handle[handle] = go
            if touched is not None:
                touched.append(world.get_mesh(handle).world_aabb)

    def _drop_mesh(
        self, oid: int, touched: list[tuple[np.ndarray, np.ndarray]] | None
    ) -> None:
        binding = self._meshes.pop(oid)
        self._by_handle.pop(binding.handle, None)
        if self.world.has_mesh(binding.handle):
            if touched is not None:
                touched.append(self.world.get_mesh(binding.handle).world_aabb)
            self.world.remove_mesh(binding.handle)

    def _sync_mesh_poses(self) -> None:
        """Переносит в мир позы сдвинутых mesh-коллайдеров (без перестройки BVH)."""
        for binding in list(self._meshes.values()):
            matrix = binding.game_object.transform.world_matrix.tobytes()
            if matrix == binding.matrix:
                continue
            binding.matrix = matrix
            pos, rot, scale = _world_pose(binding.game_object)
            comp = next(
                (c for c in binding.game_object.get_components(MeshColliderComponent)
                 if c.is_active),
                None,
            )
            if (
                comp is None
                or self._mesh_signature(comp, scale)[1] != binding.signature[1]
            ):
                self.mark_dirty(binding.game_object)  # масштаб изменился: пересборка
                continue
            self.world.set_mesh_pose(binding.handle, pos, rot)

    def _collect_aabbs(
        self, oid: int, out: list[tuple[np.ndarray, np.ndarray]] | None
    ) -> None:
        """Добавляет в ``out`` AABB коллайдеров связанного объекта (если он есть)."""
        binding = self._bindings.get(oid)
        if out is None or binding is None or binding.handle not in self.world:
            return
        out.extend(c.world_aabb for c in self.world.get_colliders(binding.handle))

    @staticmethod
    def _frozen_component(
        game_object: GameObject,
    ) -> CharacterControllerComponent | None:
        """Компонент «замороженного» персонажа объекта или ``None``.

        Заморожен выключенный ``CharacterControllerComponent`` активного объекта,
        у которого нет ``RigidBodyComponent`` вовсе: если тело есть и включено,
        объектом управляет оно; если оно выключено тоже, объект вне физики.
        """
        if not game_object.active_in_hierarchy or game_object.get_components(
            RigidBodyComponent
        ):
            return None
        components = game_object.get_components(CharacterControllerComponent)
        return next(iter(components), None)

    @staticmethod
    def _is_stale(binding: _Binding, ch: CharacterControllerComponent | None) -> bool:
        """Привязка другого вида или с другим компонентом, чем нужно сейчас миру."""
        if binding.character is None:
            return ch is not None
        return ch is None or binding.component is not ch

    @staticmethod
    def _set_frozen(binding: _Binding, frozen: bool) -> None:
        """Переключает двойника между «активен» и «заморожен» без пересоздания.

        Заморозка обнуляет скорости (разморозка не должна «доиграть» падение);
        разморозка пересчитывает землю: пока двойник стоял, мир мог измениться.
        """
        if binding.frozen == frozen:
            return
        binding.frozen = frozen
        ctrl = binding.character
        if frozen:
            ctrl.vertical_velocity = 0.0
            ctrl.horizontal_velocity = np.zeros(3)
            binding.moved = False
        else:
            ctrl.refresh_ground()

    @staticmethod
    def _capture(binding: _Binding) -> tuple:
        """Состояние, которое переживает пересоздание тела/двойника.

        Первый элемент - тег: скорости тела не переносятся в контроллер и обратно.
        """
        ctrl = binding.character
        if ctrl is not None:
            return (
                "character", ctrl.vertical_velocity,
                ctrl.horizontal_velocity.copy(), binding.size[1],
            )
        body = binding.body
        if body is None:
            return ()
        return ("body", body.linear_velocity.copy(), body.angular_velocity.copy())

    def _warn_once(self, go: GameObject, key: str, message: str, *args: Any) -> None:
        """Предупреждение, которое выдаётся один раз на объект и ключ."""
        token = (go.id, key)
        if token not in self._clamp_warned:
            self._clamp_warned.add(token)
            _log.warning(message, *args)

    def _error_once(self, go: GameObject, key: str, message: str, *args: Any) -> None:
        """Ошибка, которая выдаётся один раз на объект и ключ."""
        token = (go.id, key)
        if token not in self._clamp_warned:
            self._clamp_warned.add(token)
            _log.error(message, *args)

    def _warn_clamp(self, go: GameObject, field: str, value: float,
                    clamped: float) -> None:
        """Предупреждение (раз на объект и поле) о подставленном значении."""
        self._warn_once(
            go, field, "Физика: у объекта '%s' %s=%s недопустим, используется %s",
            go.name, field, value, clamped,
        )

    def _character_size(
        self, go: GameObject, ch: CharacterControllerComponent,
        scale: tuple[float, float, float],
    ) -> tuple[float, float]:
        """Радиус и высота капсулы персонажа после мирового масштаба.

        Поля компонента проверяются как есть (``ValueError``): масштаб ошибку в них
        не лечит. Затем применяется :func:`scaled_capsule_size`; близкая к нулю ось
        масштаба и подъём ``height`` до ``2 * radius`` дают по одному предупреждению
        на объект.
        """
        if ch.radius <= 0.0:
            raise ValueError(f"radius must be > 0 (got radius={ch.radius})")
        if ch.height <= 2.0 * ch.radius:
            raise ValueError(
                f"height must be > 2*radius (got height={ch.height}, "
                f"radius={ch.radius})"
            )
        if min(scale) < _MIN_SIZE:
            self._warn_once(
                go, "scale",
                "Физика: у объекта '%s' масштаб %s близок к нулю, "
                "для капсулы персонажа ось поднята до %s",
                go.name, tuple(round(v, 6) for v in scale), _MIN_SIZE,
            )
        radius, height, raised = scaled_capsule_size(ch.radius, ch.height, scale)
        if raised:
            self._warn_once(
                go, "height",
                "Физика: у объекта '%s' после масштаба height <= 2*radius "
                "(radius=%.4g), height поднят до %.4g",
                go.name, radius, height,
            )
        return radius, height

    @staticmethod
    def _size_differs(old: tuple[float, float], radius: float, height: float) -> bool:
        """Отличаются ли размеры капсулы больше допуска (относительного для крупных)."""
        return (
            abs(radius - old[0]) > _SIZE_EPS * max(1.0, old[0])
            or abs(height - old[1]) > _SIZE_EPS * max(1.0, old[1])
        )

    @staticmethod
    def _write_world_position(transform, position: np.ndarray) -> None:
        """Выставляет мировую позицию (вращение и масштаб не трогает).

        Вращение намеренно не пересчитывается: из матрицы с отрицательным масштабом
        оно восстанавливается неверно.
        """
        parent = transform.parent
        if parent is not None:
            try:
                inverse = np.linalg.inv(parent.world_matrix)
            except np.linalg.LinAlgError:  # родитель с нулевым масштабом
                return
            position = (inverse @ np.append(position, 1.0))[:3]
        transform.position = Vector3(*map(float, position))

    def _live_values(
        self, go: GameObject, comp: CharacterControllerComponent
    ) -> dict[str, Any]:
        """«Живые» поля компонента, приведённые к допустимому диапазону.

        Недопустимое значение (отрицательная скорость, не число) заменяется
        ближайшим допустимым с одним предупреждением на объект и поле.
        """
        values: dict[str, Any] = {}
        for name in _LIVE_NUMERIC:
            raw = getattr(comp, name)
            try:
                number = float(raw)
            except (TypeError, ValueError):
                number = float("nan")
            fixed = clamp_character_field(name, number)
            if fixed != number:  # nan тоже не равен
                self._warn_clamp(go, name, raw, fixed)
            values[name] = fixed
        for name in _LIVE_FLAGS:
            values[name] = bool(getattr(comp, name))
        return values

    def _flush_live(self) -> None:
        """Переносит правки «живых» полей в контроллеры (без пересоздания)."""
        dirty, self._live_dirty = self._live_dirty, set()
        for oid in dirty:
            binding = self._bindings.get(oid)
            if binding is None or binding.character is None:
                continue  # двойник ещё не создан: _bind прочтёт поля сам
            ctrl = binding.character
            for name, value in self._live_values(
                binding.game_object, binding.component
            ).items():
                setattr(ctrl, name, value)
            align = bool(binding.component.alignment_to_ground)
            if binding.align and not align and binding.tilted:
                self._upright(binding)
            binding.align = align

    def _unbind(self, oid: int) -> None:
        binding = self._bindings.pop(oid)
        self._by_handle.pop(binding.handle, None)
        if binding.character is not None:
            binding.character.remove()
        elif binding.handle in self.world:
            self.world.remove_body(binding.handle)

    def _bind(self, go, rb, cols, ch, carry) -> None:
        pos, rot, scale = _world_pose(go)
        if ch is not None:
            skin_width = ch.skin_width
            if skin_width <= 0.0:
                self._warn_clamp(go, "skin_width", skin_width, _MIN_SKIN_WIDTH)
                skin_width = _MIN_SKIN_WIDTH
            slope_limit = ch.slope_limit
            if not 0.0 <= slope_limit <= 90.0:
                clamped = min(max(slope_limit, 0.0), 90.0)
                self._warn_clamp(go, "slope_limit", slope_limit, clamped)
                slope_limit = clamped
            matrix = go.transform.world_matrix
            radius, height = self._character_size(
                go, ch, _basis_scale(matrix[:3, :3].tolist())
            )
            # Пересоздание с другой высотой: ноги остаются на месте, центр
            # сдвигается на половину изменения высоты (скорости переносятся).
            kept = carry is not None and len(carry) > 0 and carry[0] == "character"
            shift = (height - carry[3]) / 2.0 if kept else 0.0
            if abs(shift) <= _SIZE_EPS:
                shift = 0.0
            else:
                pos = pos + np.array([0.0, shift, 0.0])
            controller = CharacterController(
                self.world, pos, radius=radius, height=height,
                step_offset=ch.step_offset, slope_limit=slope_limit,
                skin_width=skin_width, layer=ch.layer, mask=ch.mask,
                user_data=go, **self._live_values(go, ch),
            )
            if kept and ch.enabled:  # замороженный стоит: старые скорости не нужны
                controller.vertical_velocity = carry[1]
                controller.horizontal_velocity = carry[2]
            binding = _Binding(go, "character", controller.handle)
            binding.character = controller
            binding.component = ch
            binding.frozen = not ch.enabled
            binding.align = bool(ch.alignment_to_ground)
            binding.size = (radius, height)
            binding.basis = matrix[:3, :3].tobytes()
            if shift:
                self._write_world_position(go.transform, pos)
        else:
            colliders = []
            for comp in cols:
                offset = np.asarray(comp.offset[:3]) * scale
                colliders.append(Collider(
                    _build_shape(comp, scale),
                    PhysicsMaterial(
                        friction=max(comp.friction, 0.0),
                        restitution=min(max(comp.restitution, 0.0), 1.0),
                        friction_combine=comp.friction_combine,
                        restitution_combine=comp.restitution_combine,
                    ),
                    is_trigger=comp.is_trigger,
                    layer=comp.layer,
                    mask=comp.mask,
                    offset=offset,
                    rotation_offset=_unit_quat(comp.rotation_offset),
                ))
            kind = "static" if rb is None else rb.body_type
            if kind not in BODY_TYPES:
                raise ValueError(f"неизвестный тип тела: {kind!r}")
            body = RigidBody(
                body_type=kind,
                mass=1.0 if rb is None else rb.mass,
                position=pos,
                rotation=rot,
                linear_damping=0.0 if rb is None else rb.linear_damping,
                angular_damping=0.0 if rb is None else rb.angular_damping,
                gravity_scale=1.0 if rb is None else rb.gravity_scale,
                use_gravity=True if rb is None else rb.use_gravity,
                freeze_rotation=False if rb is None else rb.freeze_rotation,
                can_sleep=True if rb is None else rb.can_sleep,
            )
            if carry and carry[0] == "body":
                body.linear_velocity, body.angular_velocity = carry[1:]
            handle = self.world.add_body(body, colliders, go)
            binding = _Binding(go, kind, handle)
            binding.body = self.world.get_body(handle)
        binding.parented = go.transform.parent is not None
        binding.last_pos = pos
        binding.last_rot = rot
        self._bindings[go.id] = binding
        self._by_handle[binding.handle] = go

    def _rebuild_layout(self) -> None:
        fast = [
            b for b in self._bindings.values()
            if b.kind != "character" and not b.parented
        ]
        self._fast = fast
        self._slow = [
            b for b in self._bindings.values()
            if b.kind != "character" and b.parented
        ]
        self._characters = [
            b for b in self._bindings.values() if b.kind == "character"
        ]
        store = self.scene.transforms
        if fast:
            self._fast_idx = np.fromiter(
                (b.game_object.transform.index for b in fast),
                dtype=np.intp, count=len(fast),
            )
            self._fast_pos, self._fast_rot = store.get_local_batch(self._fast_idx)
        else:
            self._fast_idx = np.zeros(0, dtype=np.intp)
            self._fast_pos = np.zeros((0, 3), dtype=np.float32)
            self._fast_rot = np.zeros((0, 4), dtype=np.float32)
        self._fast_dynamic = np.array(
            [k for k, b in enumerate(fast) if b.kind == "dynamic"], dtype=np.intp
        )
        self._sleep_synced = set()
        self._layout_dirty = False

    # ------------------------------------------------------------ синхронизация

    def sync_in(self, dt: float | None = None) -> None:
        """Переносит в мир позы, изменённые снаружи (телепорт, кинематика)."""
        self.sync_structure()
        if self._layout_dirty:
            self._rebuild_layout()
        world = self.world
        dt = world.fixed_dt if dt is None else dt
        if self._fast:
            pos, rot = self.scene.transforms.get_local_batch(self._fast_idx)
            moved = np.flatnonzero(
                np.any(pos != self._fast_pos, axis=1)
                | np.any(rot != self._fast_rot, axis=1)
            )
            if moved.size:
                pos64 = pos[moved].astype(np.float64)
                rot64 = rot[moved].astype(np.float64)
                for i, k in enumerate(moved.tolist()):
                    self._push(self._fast[k], pos64[i], rot64[i], dt)
                self._fast_pos[moved] = pos[moved]
                self._fast_rot[moved] = rot[moved]
        for binding in self._slow:
            p, r, _ = _world_pose(binding.game_object)
            if (np.abs(p - binding.last_pos).max() > _POSE_EPS
                    or np.abs(r - binding.last_rot).max() > _POSE_EPS):
                self._push(binding, p, r, dt)
                binding.last_pos, binding.last_rot = p, r
        resized: list[GameObject] = []
        for binding in self._characters:
            ctrl = binding.character
            matrix = binding.game_object.transform.world_matrix
            target = matrix[:3, 3].astype(np.float64)
            if np.abs(target - ctrl.position).max() > _POSE_EPS:
                ctrl.teleport(target)
                binding.last_pos = ctrl.position.copy()
            # Масштаб читаем, только когда изменилась 3x3 мировой матрицы
            # (масштаб, поворот, масштаб родителя); пересоздаём - если капсула
            # отличается от текущей.
            basis = matrix[:3, :3].tobytes()
            if basis != binding.basis:
                binding.basis = basis
                radius, height, _ = scaled_capsule_size(
                    binding.component.radius, binding.component.height,
                    _basis_scale(matrix[:3, :3].tolist()),
                )
                if self._size_differs(binding.size, radius, height):
                    resized.append(binding.game_object)
        if self._meshes:
            self._sync_mesh_poses()
        if self._dirty and not resized:
            self.sync_structure()
        if resized:
            for game_object in resized:
                self.mark_dirty(game_object)
            self.sync_structure()
            if self._layout_dirty:
                self._rebuild_layout()

    def _push(self, binding: _Binding, pos, rot, dt: float) -> None:
        world = self.world
        if binding.kind == "kinematic":
            world.move_kinematic(binding.handle, pos, rot, dt)
        else:
            world.set_pose(binding.handle, pos, rot)

    def sync_out(self) -> None:
        """Записывает позы подвижных динамических тел в трансформы."""
        if self._fast_dynamic.size:
            synced = self._sleep_synced
            awake = []
            for k in self._fast_dynamic.tolist():
                if not self._fast[k].body.sleeping:
                    synced.discard(k)
                    awake.append(k)
                elif k not in synced:
                    synced.add(k)  # первая запись позы после засыпания
                    awake.append(k)
            if awake:
                bodies = [self._fast[k].body for k in awake]
                pos = np.array([b.position for b in bodies], dtype=np.float32)
                rot = np.array([b.rotation for b in bodies], dtype=np.float32)
                ks = np.asarray(awake, dtype=np.intp)
                self.scene.transforms.set_local_batch(self._fast_idx[ks], pos, rot)
                self._fast_pos[ks] = pos
                self._fast_rot[ks] = rot
        from engine.core.math3d import Quaternion, Vector3

        for binding in self._slow:
            body = binding.body
            if binding.kind != "dynamic":
                continue
            key = -id(binding)
            if body.sleeping:
                if key in self._sleep_synced:
                    continue
                self._sleep_synced.add(key)
            else:
                self._sleep_synced.discard(key)
            binding.game_object.transform.set_world_pose(
                Vector3(*map(float, body.position)),
                Quaternion(*map(float, body.rotation)),
            )
            binding.last_pos = body.position.copy()
            binding.last_rot = body.rotation.copy()

    def settle(self, max_steps: int = 240, dt: float | None = None) -> int:
        """Дать динамическим телам осесть до первого кадра.

        Шагает только мир (без скриптов, персонажей и событий) до тех пор, пока
        все динамические тела не заснут или не кончится ``max_steps``. Стартовое
        оседание иначе растягивается на первые секунды игры: бодрствующие тела
        дороги для решателя (≈20 мс на шаг на HD 520).

        Returns:
            Число сделанных шагов.
        """
        if not self.enabled:
            return 0
        dt = self.world.fixed_dt if dt is None else float(dt)
        self.sync_in(dt)
        steps = 0
        while steps < max_steps and self.world.awake_dynamic_count() > 0:
            self.world.step(dt)
            steps += 1
        if steps:
            self.sync_out()
            self.world.pop_events()
        return steps

    def step(self, dt: float | None = None) -> None:
        """Полный физический шаг: sync_in → мир → sync_out → события."""
        if not self.enabled:
            return
        dt = self.world.fixed_dt if dt is None else float(dt)
        self.sync_in(dt)
        if self._characters:
            if self._live_dirty:
                self._flush_live()
            self._step_characters(dt)
        self.world.step(dt)
        self.sync_out()
        self.publish_events()

    def _step_characters(self, dt: float) -> None:
        """Пассивный шаг персонажей, которых скрипт в этом шаге не двигал.

        Им достаётся гравитация (``use_gravity``/``gravity_scale``): персонаж,
        созданный в воздухе, падает на землю без единой строчки скрипта. Если
        скрипт вызвал ``move``/``simple_move``/``move_input`` до шага, гравитацию
        он уже получил - второй раз её не применяем. Замороженные стоят.
        """
        for binding in self._characters:
            if binding.frozen:
                continue
            if binding.moved:
                binding.moved = False
                continue
            binding.character.step(dt)
            self._commit_character(binding, force=False)

    def publish_events(self) -> None:
        """Отправляет накопленные события физики в шину событий сцены."""
        events = self.scene.events
        for event in self.world.pop_events():
            events.emit_deferred(event)

    # ------------------------------------------------------------ тела: API

    def _body_binding(self, game_object: GameObject) -> _Binding:
        self.sync_structure()
        binding = self._bindings.get(game_object.id)
        if binding is None or binding.body is None:
            if game_object.id in self._conflicts:
                raise RuntimeError(_CONFLICT_MESSAGE)
            raise RuntimeError(f"У объекта '{game_object.name}' нет физического тела")
        return binding

    def get_velocity(
        self, game_object: GameObject, *, angular: bool = False
    ) -> tuple[float, float, float]:
        """Линейная (или угловая) скорость тела."""
        body = self._body_binding(game_object).body
        v = body.angular_velocity if angular else body.linear_velocity
        return (float(v[0]), float(v[1]), float(v[2]))

    def set_velocity(self, game_object: GameObject, linear=None, angular=None) -> None:
        """Задаёт скорость тела (будит его)."""
        binding = self._body_binding(game_object)
        self.world.set_velocity(binding.handle, linear, angular)

    def stop_body(self, game_object: GameObject) -> None:
        """Обнулить скорости уже созданного тела без синхронизации структуры.

        Если тела в мире ещё нет, ничего не делает (новое тело стартует с нуля).
        """
        binding = self._bindings.get(game_object.id)
        if binding is not None and binding.body is not None:
            self.world.set_velocity(binding.handle, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    def apply_impulse(self, game_object: GameObject, impulse, point=None) -> None:
        """Импульс телу объекта."""
        binding = self._body_binding(game_object)
        self.world.apply_impulse(binding.handle, impulse, point)

    # ------------------------------------------------------------ персонаж

    def _character_binding(self, game_object: GameObject) -> _Binding | None:
        """Привязка персонажа объекта.

        ``None`` - у объекта есть компонент(ы), и все они выключены (``enabled``),
        а двойника нет (объектом управляет RigidBody или он вне физики): методы
        управления тогда ничего не делают. Объект без компонента, с включённым
        компонентом на неактивном объекте (``active_self=False``), конфликт и
        ошибка регистрации - ``RuntimeError``. У замороженного персонажа привязка
        есть (``binding.frozen``), методы движения проверяют её сами.
        """
        self.sync_structure()
        binding = self._bindings.get(game_object.id)
        if binding is None or binding.character is None:
            if game_object.id in self._conflicts:
                raise RuntimeError(_CONFLICT_MESSAGE)
            reason = self._failures.get(game_object.id)
            if reason is not None:
                raise RuntimeError(f"компонент не зарегистрирован: {reason}")
            components = game_object.get_components(CharacterControllerComponent)
            if components and not any(c.enabled for c in components):
                return None
            raise RuntimeError(
                f"У объекта '{game_object.name}' нет CharacterControllerComponent"
            )
        if self._live_dirty:
            self._flush_live()
        return binding

    def _commit_character(self, binding: _Binding, force: bool = True) -> None:
        """Записывает позу контроллера в трансформ (и наклон по земле).

        ``force=False`` (пассивный шаг): покоящийся персонаж трансформ не
        трогает - позиция не ушла дальше ``_PASSIVE_EPS``, наклон не изменился.
        """
        ctrl = binding.character
        pos = ctrl.position
        moved = force or np.abs(pos - binding.last_pos).max() > _PASSIVE_EPS
        aligned = binding.align and ctrl.grounded
        if not moved and not aligned:
            return
        transform = binding.game_object.transform
        rot = transform.world_rotation
        if aligned:
            target = _aligned_rotation(rot, ctrl.ground_normal)
            if not moved and abs(target.dot(rot)) > 1.0 - 1e-9:
                return
            rot = target
            binding.tilted = True
        transform.set_world_pose(Vector3(*map(float, pos)), rot)
        binding.last_pos = pos.copy()

    def _upright(self, binding: _Binding) -> None:
        """Возвращает наклонённому персонажу вертикальную позу (рыскание цело)."""
        transform = binding.game_object.transform
        transform.set_world_pose(
            transform.world_position, _yaw_of(transform.world_rotation)
        )
        binding.tilted = False

    def move_character(
        self, game_object: GameObject, motion
    ) -> tuple[float, float, float]:
        """Двигает персонажа на ``motion`` и обновляет трансформ.

        Замороженный (``enabled=False``) не двигается: ``(0, 0, 0)``.
        """
        binding = self._character_binding(game_object)
        if binding is None or binding.frozen:
            return _NO_MOTION
        actual = binding.character.move(motion, self.world.fixed_dt)
        binding.moved = True
        self._commit_character(binding)
        return (float(actual[0]), float(actual[1]), float(actual[2]))

    def update_character(
        self, game_object: GameObject, velocity, dt: float | None = None
    ) -> tuple[float, float, float]:
        """Шаг персонажа: горизонтальная скорость + гравитация/прыжок.

        Замороженный (``enabled=False``) не двигается: ``(0, 0, 0)``.
        """
        binding = self._character_binding(game_object)
        if binding is None or binding.frozen:
            return _NO_MOTION
        actual = binding.character.simple_move(velocity, dt)
        binding.moved = True
        self._commit_character(binding)
        return (float(actual[0]), float(actual[1]), float(actual[2]))

    def move_input_character(
        self, game_object: GameObject, direction, *, sprint: bool = False,
        air_control: float | None = None,
    ) -> tuple[float, float, float]:
        """Шаг персонажа по направлению ввода: ``move_speed``, бег, воздух.

        Замороженный (``enabled=False``) не двигается: ``(0, 0, 0)``.
        """
        binding = self._character_binding(game_object)
        if binding is None or binding.frozen:
            return _NO_MOTION
        actual = binding.character.move_input(
            direction, sprint=sprint, air_control=air_control
        )
        binding.moved = True
        self._commit_character(binding)
        return (float(actual[0]), float(actual[1]), float(actual[2]))

    def jump_character(
        self, game_object: GameObject, speed: float | None = None
    ) -> bool:
        """Прыжок персонажа (только с земли); ``speed <= 0`` -> ``ValueError``.

        ``speed=None`` - ``jump_velocity`` компонента. Замороженный не прыгает:
        ``False`` (в том числе при недопустимой ``speed``).
        """
        binding = self._character_binding(game_object)
        if binding is None or binding.frozen:
            return False
        return binding.character.jump(speed)

    def is_grounded(self, game_object: GameObject) -> bool:
        """Стоит ли персонаж на земле.

        У замороженного - значение на момент заморозки (или последней проверки
        земли: разморозки, переноса, пересоздания); без двойника - ``False``.
        """
        binding = self._character_binding(game_object)
        return binding is not None and bool(binding.character.grounded)

    # ------------------------------------------------------------ запросы

    def raycast(
        self, origin, direction, max_distance: float = float("inf"),
        layer_mask: int = ALL_LAYERS, include_triggers: bool = False,
    ) -> dict | None:
        """Луч: ``dict(object, point, normal, distance)`` или ``None``."""
        self.sync_structure()
        hit = self.world.raycast(
            origin, direction, max_distance, layer_mask, include_triggers
        )
        return None if hit is None else self._hit_dict(hit)

    def overlap_sphere(
        self, center, radius: float, layer_mask: int = ALL_LAYERS,
        include_triggers: bool = False,
    ) -> list[GameObject]:
        """Объекты, чьи коллайдеры пересекают шар."""
        self.sync_structure()
        handles = self.world.overlap_sphere(
            center, radius, layer_mask, include_triggers
        )
        return [self._by_handle[h] for h in handles if h in self._by_handle]

    def _hit_dict(self, hit) -> dict:
        return {
            "object": self._by_handle.get(hit.handle),
            "point": tuple(float(v) for v in hit.point),
            "normal": tuple(float(v) for v in hit.normal),
            "distance": float(hit.distance),
        }

    def character_handles(self) -> frozenset[int]:
        """Handle'ы тел-двойников персонажей (для отладочной раскраски)."""
        return frozenset(
            b.handle for b in self._bindings.values() if b.kind == "character"
        )

    def set_gravity(self, gravity) -> None:
        """Задаёт гравитацию мира."""
        self.world.set_gravity(gravity)

    def clear(self) -> None:
        """Удаляет все тела (сцена не меняется)."""
        self._bindings.clear()
        self._by_handle.clear()
        self._meshes.clear()
        self._characters = []
        self._live_dirty.clear()
        self.world.clear()
        self._all_dirty = True
        self._layout_dirty = True
