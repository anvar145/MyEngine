"""Физический мир: шаг симуляции, запросы, события.

Конвейер шага: гравитация/демпфирование -> широкая фаза -> пробуждение спящих
-> узкая фаза -> события -> sequential-impulse солвер -> интегрирование поз ->
засыпание островов. Мир знает только о телах и коллайдерах (без ECS/GL/Qt).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.physics.body import RigidBody
from engine.physics.broadphase import StoreBroadphase
from engine.physics.bvh import Bvh
from engine.physics.collider import Collider
from engine.physics.events import (
    CollisionEnter,
    CollisionExit,
    PhysicsEvent,
    TriggerEnter,
    TriggerExit,
    TriggerStay,
)
from engine.physics.island import build_islands
from engine.physics.material import (
    PhysicsMaterial,
    combine_friction,
    combine_restitution,
)
from engine.physics.mathutil import (
    IDENTITY_QUAT,
    as_matrix,
    quat_normalize,
    quat_to_matrix,
    quats_integrate,
    quats_to_matrices,
    vec3,
)
from engine.physics.mesh_store import MeshCollider
from engine.physics.narrowphase import (
    _capsule_box_hit,
    _segment_pair_dist,
    collide,
    overlaps,
)
from engine.physics.narrowphase_batch import box_box_batch
from engine.physics.narrowphase_round import collide_batch
from engine.physics.raycast import ray_aabb, ray_shape
from engine.physics.shape import BoxShape, CapsuleShape, Shape
from engine.physics.sleeping import SleepSettings, update_island_sleep
from engine.physics.solver import DEFAULT_ITERATIONS
from engine.physics.solver_batch import (
    MAX_SLOTS,
    ColoringCache,
    ContactBatch,
    WarmStart,
    contact_keys,
    solve_batch,
)
from engine.physics.store import (
    BODY_DYNAMIC,
    BODY_KINEMATIC,
    BODY_STATIC,
    KIND_BOX,
    KIND_OTHER,
    BodyStore,
    ColliderStore,
)

ALL_LAYERS = 0xFFFFFFFF
_KIND_SPAN = KIND_OTHER + 1  # основание кода пары видов: ``kind_a * span + kind_b``
MAX_LINEAR_SPEED = 200.0
MAX_ANGULAR_SPEED = 60.0
# Итераций бисекции в shapecast: интервал после них = шаг / 256 (≈0.3 мм для
# покадрового смещения ходьбы 7.5 см).
_BISECT_ITERATIONS = 8


def _first_overlap(overlap, limit: float, step: float) -> float | None:
    """Параметр первого пересечения вдоль свипа (марш + бисекция по предикату).

    Повторяет схему :meth:`PhysicsWorld._sweep_one` (``t = 0``, шаги ``step``,
    ``_BISECT_ITERATIONS`` делений), но вызывает лишь булев предикат.

    Returns:
        ``t`` проникающего конца отрезка бисекции или ``None``, если пересечения нет.
    """
    if overlap(0.0):
        return 0.0
    t_prev = 0.0
    t = 0.0
    while t < limit:
        t = min(t + step, limit)
        if overlap(t):
            lo, hi = t_prev, t
            for _ in range(_BISECT_ITERATIONS):
                mid = 0.5 * (lo + hi)
                if overlap(mid):
                    hi = mid
                else:
                    lo = mid
            return hi
        t_prev = t
    return None


@dataclass(slots=True)
class RaycastHit:
    """Результат луча/шейпкаста. ``normal`` направлена от поверхности наружу."""

    handle: int
    collider: Collider
    point: np.ndarray
    normal: np.ndarray
    distance: float
    user_data: Any = None


@dataclass(slots=True)
class RaycastBatch:
    """Результат пакета лучей (массивы длины ``n``, без объектов на луч).

    Attributes:
        distance: Расстояния; ``inf`` у промахов.
        handle: Handle попавшего тела/mesh; ``-1`` у промахов.
        point: Точки попадания ``(n, 3)``; ``nan`` у промахов.
        normal: Нормали ``(n, 3)``, наружу от поверхности; ``nan`` у промахов.
    """

    distance: np.ndarray
    handle: np.ndarray
    point: np.ndarray
    normal: np.ndarray


@dataclass(eq=False, slots=True)
class _Entry:
    handle: int
    body: RigidBody
    colliders: list[Collider]
    cids: list[int]
    user_data: Any = None
    group: list[int] | None = field(default=None)  # остров, уснувший вместе
    external: bool = False  # поза задана снаружи (move_kinematic)
    rot_key: bytes | None = None  # входной поворот прошлого move_kinematic


def _pair_overlap(shape, pos, rot, d, col, lo_t, hi_t):
    """Предикат ``overlap(t)`` для свипа капсулы по боксу или капсуле.

    Концы оси капсулы и данные коллайдера готовятся один раз; в точке ``t``
    остаются чистые float-операции. Для остальных пар возвращает ``None``.
    """
    if type(shape) is not CapsuleShape:
        return None
    kind = type(col.shape)
    if kind is not BoxShape and kind is not CapsuleShape:
        return None
    px, py, pz = pos.tolist()
    ax, ay, az = (rot[:, 1] * shape.half_height).tolist()
    dx, dy, dz = d.tolist()
    radius = shape.radius
    if kind is BoxShape:
        h = list(col.shape.half_extents)
        origin = col.world_position.tolist()
        m = as_matrix(col.world_rotation).tolist()
    else:
        q_a, q_b = col.shape.segment(col.world_position, col.world_rotation)
        q_a, q_b = q_a.tolist(), q_b.tolist()
        rsum = radius + col.shape.radius

    def overlap(t):
        if (
            lo_t[0] + dx * t > 0.0 or hi_t[0] + dx * t < 0.0
            or lo_t[1] + dy * t > 0.0 or hi_t[1] + dy * t < 0.0
            or lo_t[2] + dz * t > 0.0 or hi_t[2] + dz * t < 0.0
        ):
            return False
        cx, cy, cz = px + dx * t, py + dy * t, pz + dz * t
        p0 = [cx - ax, cy - ay, cz - az]
        p1 = [cx + ax, cy + ay, cz + az]
        if kind is BoxShape:
            la = [0.0, 0.0, 0.0]
            lb = [0.0, 0.0, 0.0]
            for k in range(3):
                la[k] = (
                    (p0[0] - origin[0]) * m[0][k]
                    + (p0[1] - origin[1]) * m[1][k]
                    + (p0[2] - origin[2]) * m[2][k]
                )
                lb[k] = (
                    (p1[0] - origin[0]) * m[0][k]
                    + (p1[1] - origin[1]) * m[1][k]
                    + (p1[2] - origin[2]) * m[2][k]
                )
            return _capsule_box_hit(la, lb, h, radius)
        return _segment_pair_dist(p0, p1, q_a, q_b) < rsum

    return overlap


class PhysicsWorld:
    """Мир твёрдых тел."""

    def __init__(
        self,
        gravity=(0.0, -9.81, 0.0),
        fixed_dt: float = 1.0 / 60.0,
        iterations: int = DEFAULT_ITERATIONS,
        emit_trigger_stay: bool = True,
    ) -> None:
        self.gravity = vec3(gravity)
        self.fixed_dt = float(fixed_dt)
        self.iterations = int(iterations)
        self.emit_trigger_stay = emit_trigger_stay
        self.sleep_settings = SleepSettings()
        self._bodies = BodyStore(shared=True)
        self._colliders = ColliderStore()
        self.broadphase = StoreBroadphase(self._colliders)
        self.events: list[PhysicsEvent] = []
        self._entries: dict[int, _Entry] = {}
        self._collider_owner: dict[int, tuple[_Entry, Collider]] = {}
        self._next_handle = 1
        self._next_cid = 1
        self._warm = WarmStart()  # импульсы прошлого шага по ключу контакта
        self._coloring = ColoringCache()  # раскраска контактов прошлого шага
        self._batch_round = True  # пары с шаром/капсулой пакетом (False - скалярно)
        self._touching: set[tuple[int, int]] = set()
        self._triggers: set[tuple[int, int]] = set()
        self._last_contact_count = 0
        self._external: set[int] = set()
        # статические mesh-коллайдеры: только запросы, вне SAP и солвера
        self._meshes: dict[int, MeshCollider] = {}

    # ----------------------------------------------------- mesh-коллайдеры

    def add_mesh(
        self,
        bvh: Bvh,
        position=(0.0, 0.0, 0.0),
        rotation=IDENTITY_QUAT,
        layer: int = 1,
        mask: int = ALL_LAYERS,
        is_trigger: bool = False,
        material: PhysicsMaterial | None = None,
        user_data: Any = None,
    ) -> int:
        """Добавляет статический mesh-коллайдер; возвращает его handle.

        Mesh виден только запросам (``raycast``/``shapecast``/``overlap``/
        ``mesh_contacts``): в широкую фазу и солвер он не входит, динамическое
        тело сквозь него проваливается, событий столкновений нет.
        """
        handle = self._next_handle
        self._next_handle += 1
        self._meshes[handle] = MeshCollider(
            handle, bvh, vec3(position), np.asarray(rotation, dtype=np.float64),
            int(layer), int(mask), bool(is_trigger),
            material if material is not None else PhysicsMaterial(), user_data,
        )
        return handle

    def remove_mesh(self, handle: int) -> None:
        """Удаляет mesh-коллайдер."""
        del self._meshes[handle]

    def has_mesh(self, handle: int) -> bool:
        return handle in self._meshes

    def iter_meshes(self):
        """Итератор ``(handle, MeshCollider)`` по возрастанию handle."""
        for handle in sorted(self._meshes):
            yield handle, self._meshes[handle]

    def get_mesh(self, handle: int) -> MeshCollider:
        return self._meshes[handle]

    def set_mesh_pose(self, handle: int, position, rotation=IDENTITY_QUAT) -> None:
        """Переставляет mesh-коллайдер без перестройки BVH."""
        self._meshes[handle].set_pose(position, rotation)

    @property
    def mesh_count(self) -> int:
        return len(self._meshes)

    def _mesh_accept(self, mesh: MeshCollider, layer_mask: int,
                     include_triggers: bool) -> bool:
        if mesh.is_trigger and not include_triggers:
            return False
        return bool(mesh.layer & layer_mask)

    def mesh_contacts(
        self,
        shape: Shape,
        position,
        rotation=IDENTITY_QUAT,
        layer_mask: int = ALL_LAYERS,
        include_triggers: bool = False,
    ) -> list[tuple[np.ndarray, float]]:
        """Контакты шара/капсулы с mesh: ``[(нормаль к форме, глубина)]``."""
        if not self._meshes:
            return []
        pos = vec3(position)
        rot = as_matrix(rotation)
        out = []
        for mesh in self._meshes.values():
            if not self._mesh_accept(mesh, layer_mask, include_triggers):
                continue
            for nx, ny, nz, depth, *_ in mesh.contacts(shape, pos, rot):
                out.append((np.array((nx, ny, nz)), depth))
        return out

    # ------------------------------------------------------------------ тела

    def set_gravity(self, gravity) -> None:
        """Задаёт вектор гравитации и будит тела."""
        self.gravity = vec3(gravity)
        for entry in self._entries.values():
            if entry.body.is_dynamic:
                self._wake(entry)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, handle: int) -> bool:
        return handle in self._entries

    def handles(self) -> list[int]:
        """Все handle по возрастанию."""
        return sorted(self._entries)

    def add_body(
        self,
        body: RigidBody,
        colliders: Collider | list[Collider],
        user_data: Any = None,
    ) -> int:
        """Добавляет тело с коллайдерами и возвращает handle.

        Для динамического тела инерция считается по коллайдерам: масса делится
        пропорционально объёму, центр масс — начало тела.
        """
        cols = [colliders] if isinstance(colliders, Collider) else list(colliders)
        if not cols:
            raise ValueError("у тела должен быть хотя бы один коллайдер")
        if body._store.shared:
            raise ValueError("тело уже добавлено в мир")
        handle = self._next_handle
        self._next_handle += 1
        if body.is_dynamic:
            volumes = [max(c.shape.volume(), 1e-12) for c in cols]
            total = sum(volumes)
            inertia = sum(
                (
                    c.local_inertia(body.mass * v / total)
                    for c, v in zip(cols, volumes, strict=True)
                ),
                np.zeros((3, 3)),
            )
            body.set_inertia(inertia)
        else:
            body.set_inertia(np.eye(3))
        cids = list(range(self._next_cid, self._next_cid + len(cols)))
        self._next_cid += len(cols)
        self._bodies.adopt(body, handle)
        entry = _Entry(handle, body, cols, cids, user_data)
        self._entries[handle] = entry
        for col, cid in zip(cols, cids, strict=True):
            self._colliders.add(col, cid, handle)
            self._collider_owner[cid] = (entry, col)
        self._refresh_colliders(entry)
        return handle

    def add_static(
        self,
        collider: Collider,
        position=(0.0, 0.0, 0.0),
        rotation=IDENTITY_QUAT,
        user_data: Any = None,
    ) -> int:
        """Добавляет статический коллайдер (пол, стена)."""
        body = RigidBody(body_type="static", position=position, rotation=rotation)
        return self.add_body(body, collider, user_data)

    def remove_body(self, handle: int) -> None:
        """Удаляет тело; для активных пар публикуются события Exit."""
        entry = self._entries[handle]
        owned = set(entry.cids)
        for key in sorted(k for k in self._touching if k[0] in owned or k[1] in owned):
            self._touching.discard(key)
            self._emit_pair(CollisionExit, key)
        for key in sorted(k for k in self._triggers if k[0] in owned or k[1] in owned):
            self._triggers.discard(key)
            self._emit_pair(TriggerExit, key)
        # импульсы warm start чистить не нужно: cid не переиспользуются
        del self._entries[handle]
        self._external.discard(handle)
        for cid in entry.cids:
            self._colliders.remove(cid)
            del self._collider_owner[cid]
        self._bodies.release(entry.body)
        if entry.group:
            for other in entry.group:
                if other != handle and other in self._entries:
                    self._wake(self._entries[other])

    def clear(self) -> None:
        """Удаляет всё содержимое мира."""
        self._colliders.clear()
        self._bodies.clear()
        self._entries.clear()
        self._collider_owner.clear()
        self._meshes.clear()
        self._warm.clear()
        self._touching.clear()
        self._triggers.clear()
        self.events.clear()

    def get_body(self, handle: int) -> RigidBody:
        return self._entries[handle].body

    def get_colliders(self, handle: int) -> list[Collider]:
        return self._entries[handle].colliders

    def get_user_data(self, handle: int) -> Any:
        return self._entries[handle].user_data

    def iter_colliders(self):
        """Итератор ``(handle, body, collider)`` по возрастанию handle."""
        for handle in sorted(self._entries):
            entry = self._entries[handle]
            for col in entry.colliders:
                yield handle, entry.body, col

    @property
    def contact_count(self) -> int:
        """Число контактных точек на последнем шаге (для статистики)."""
        return self._last_contact_count

    def set_pose(self, handle: int, position, rotation=None) -> None:
        """Телепорт тела (будит остров)."""
        entry = self._entries[handle]
        entry.body.position = vec3(position)
        if rotation is not None:
            entry.body.rotation = quat_normalize(rotation)
            entry.rot_key = None
            entry.body.update_world_inertia()
        self._wake(entry)
        self._refresh_colliders(entry)

    def move_kinematic(self, handle: int, position, rotation=None,
                       dt: float | None = None) -> None:
        """Переносит кинематическое тело сразу и задаёт ему скорость переноса.

        Скорость (``смещение / dt``) нужна, чтобы тела рядом реагировали на
        движение, будились и толкались; сам мир позу повторно не интегрирует, а в
        конце шага обнуляет скорость.
        """
        entry = self._entries[handle]
        body = entry.body
        dt = self.fixed_dt if dt is None else float(dt)
        new = vec3(position)
        delta = new - body.position
        body.linear_velocity = delta / max(dt, 1e-9)
        body.position = new
        entry.external = True
        self._external.add(handle)
        # Поворот не менялся: матрицы коллайдеров прежние, AABB лишь сдвигается.
        key = rotation.tobytes() if isinstance(rotation, np.ndarray) else None
        if rotation is None or (key is not None and key == entry.rot_key):
            store = self._colliders
            for col in entry.colliders:
                row = col._row
                store.world_pos[row] += delta
                store.aabb_lo[row] += delta
                store.aabb_hi[row] += delta
            store.version += 1
            return
        body.rotation = quat_normalize(rotation)
        entry.rot_key = key
        self._refresh_colliders(entry)

    def apply_impulse(self, handle: int, impulse, point=None) -> None:
        """Импульс телу (будит остров)."""
        entry = self._entries[handle]
        self._wake(entry)
        entry.body.apply_impulse(impulse, point)

    def set_velocity(self, handle: int, linear=None, angular=None) -> None:
        entry = self._entries[handle]
        if linear is not None:
            entry.body.linear_velocity = vec3(linear).copy()
        if angular is not None:
            entry.body.angular_velocity = vec3(angular).copy()
        self._wake(entry)

    def wake(self, handle: int) -> None:
        self._wake(self._entries[handle])

    def wake_overlapping(self, aabbs, margin: float = 0.02) -> None:
        """Будит спящие динамические тела, чьи коллайдеры задевают заданные AABB.

        Нужна, когда геометрия вокруг спящих тел изменилась «снаружи» мира
        (форму статики или кинематики пересоздали: старый объём освободился или
        новый занял место спящего тела). Сам мир этого не видит — спящие тела
        в паре не участвуют. Запрос использует кэш широкой фазы, поэтому
        вызывать его нужно один раз на пачку изменений, а не после каждого
        ``add_body``.

        Args:
            aabbs: Итерируемое пар ``(lo, hi)`` — объёмы, вокруг которых будить.
            margin: Запас по каждой оси, чтобы задеть тела, лежащие на опоре.
        """
        for lo, hi in aabbs:
            lo = np.asarray(lo, dtype=np.float64) - margin
            hi = np.asarray(hi, dtype=np.float64) + margin
            for cid in self.broadphase.query_aabb((lo, hi)):
                entry = self._collider_owner[cid][0]
                if entry.body.is_dynamic and entry.body.sleeping:
                    self._wake(entry)

    def _wake(self, entry: _Entry) -> None:
        group = entry.group
        entry.group = None
        entry.body.wake()
        if group:
            for other in group:
                e = self._entries.get(other)
                if e is not None and e is not entry:
                    e.group = None
                    e.body.wake()

    def _refresh_colliders(self, entry: _Entry) -> None:
        """Пересчитывает мировые позы и AABB коллайдеров тела по его позе."""
        body = entry.body
        count = len(entry.colliders)
        rows = np.fromiter((c._row for c in entry.colliders), dtype=np.intp,
                           count=count)
        pos = np.broadcast_to(body.position, (count, 3))
        rot = np.broadcast_to(quat_to_matrix(body.rotation), (count, 3, 3))
        self._colliders.update_world(rows, pos, rot)

    # ------------------------------------------------------------------ шаг

    def step(self, dt: float | None = None) -> None:
        """Один шаг симуляции длительностью ``dt`` (по умолчанию ``fixed_dt``).

        Скорости: ``v ← (v + g·gravity_scale·dt) / (1 + d·dt)`` (неявное
        демпфирование, знак скорости не меняется), затем интегрирование
        полунеявным Эйлером: позиция считается по уже обновлённой скорости.
        Артефакт схемы: свободное падение на 0.5 с при 60 Гц даёт путь больше
        аналитического ``g·t²/2`` примерно на 3 % (скорость при этом точная).

        У динамических тел с ``freeze_rotation`` угловая скорость обнуляется в
        начале шага: единственное место, через которое проходят и ``set_velocity``,
        и прямая запись ``body.angular_velocity`` скриптом. Импульсы вращать
        такое тело не могут (обратная инерция равна нулю), поэтому дальше в шаге
        ω остаётся нулевой.
        """
        dt = self.fixed_dt if dt is None else float(dt)
        entries = self._entries
        bs, cs = self._bodies, self._colliders
        rows = self._awake_rows()
        awake = self._entries_at(rows)
        # 1. Скорости: гравитация и демпфирование (массивами по бодрствующим).
        if rows.size:
            vel = bs.vel[rows]
            lin, ang = vel[:, :3], vel[:, 3:]
            pull = self.gravity * (bs.gravity_scale[rows] * dt)[:, None]
            lin = np.where(bs.use_gravity[rows][:, None], lin + pull, lin)
            vel[:, :3] = lin * (1.0 / (1.0 + bs.linear_damping[rows] * dt))[:, None]
            damp = (1.0 / (1.0 + bs.angular_damping[rows] * dt))[:, None]
            vel[:, 3:] = np.where(bs.freeze_rotation[rows][:, None], 0.0, ang * damp)
            bs.vel[rows] = vel

        # 2. Широкая фаза и фильтр пар (массивами; порядок пар — по (cid_a, cid_b)).
        ca, cb = self.broadphase.query_pair_arrays()
        row_a = cs.row_of_cid[ca].astype(np.intp)
        row_b = cs.row_of_cid[cb].astype(np.intp)
        own_a, own_b = cs.owner_handle[row_a], cs.owner_handle[row_b]
        brow_a = bs.row_of_handle[own_a].astype(np.intp)
        brow_b = bs.row_of_handle[own_b].astype(np.intp)
        trig_a, trig_b = cs.is_trigger[row_a], cs.is_trigger[row_b]
        trig = trig_a | trig_b
        type_a, type_b = bs.body_type[brow_a], bs.body_type[brow_b]
        mover = self._mover_flags()
        sleep_a, sleep_b = bs.sleeping[brow_a], bs.sleeping[brow_b]
        keep = (
            (own_a != own_b)
            & ((cs.layer[row_a] & cs.mask[row_b]) != 0)
            & ((cs.layer[row_b] & cs.mask[row_a]) != 0)
            & ~(trig_a & trig_b)
            & ~((type_a == BODY_STATIC) & (type_b == BODY_STATIC))
            & (trig | (type_a == BODY_DYNAMIC) | (type_b == BODY_DYNAMIC))
        )
        # «спящее — неподвижное»: пересчёта не требуется, помним прошлое
        inert_mask = keep & ((sleep_a & ~mover[brow_b]) | (sleep_b & ~mover[brow_a]))
        act = np.flatnonzero(keep & ~inert_mask)
        inert = np.flatnonzero(inert_mask)

        # 3. Пробуждение спящих телами, которые на них наезжают.
        woke = False
        for p in act[(sleep_a[act] != sleep_b[act]) & ~trig[act]].tolist():
            ea, col_a = self._collider_owner[int(ca[p])]
            eb, col_b = self._collider_owner[int(cb[p])]
            sa, sb = self._is_sleeping(ea), self._is_sleeping(eb)
            if sa == sb:
                continue
            sleeper, other = (ea, eb) if sa else (eb, ea)
            if not self._is_mover(other):
                continue
            if self._narrow(col_a, col_b) is not None:
                self._wake(sleeper)
                woke = True
        if woke:
            awake = self._entries_at(self._awake_rows())
        if inert.size:
            asleep = bs.sleeping[brow_a[inert]] | bs.sleeping[brow_b[inert]]
            revived = inert[~asleep]
            inert = inert[asleep]
            if revived.size:  # заснувший сосед разбужен в этом шаге
                act = np.union1d(act, revived)  # индексы пар отсортированы

        # 4. Узкая фаза.
        still = set(zip(ca[inert].tolist(), cb[inert].tolist(), strict=True))
        touching = still & self._touching
        triggers = still & self._triggers
        for p in act[trig[act]].tolist():
            if self._overlap(cs.items[row_a[p]], cs.items[row_b[p]]):
                triggers.add((int(ca[p]), int(cb[p])))
        contacts = self._contacts(
            act[~trig[act]], ca, cb, row_a, row_b, brow_a, brow_b, touching
        )
        self._last_contact_count = 0 if contacts is None else len(contacts)

        # 5. События пар.
        self._diff_events(touching, self._touching, CollisionEnter, None, CollisionExit)
        self._touching = touching
        self._diff_events(triggers, self._triggers, TriggerEnter,
                          TriggerStay if self.emit_trigger_stay else None, TriggerExit)
        self._triggers = triggers

        # 6. Солвер.
        if contacts is None:
            self._warm.clear()
        else:
            warm = self._warm.lookup(contacts.key)
            self._warm.store(
                contacts.key,
                solve_batch(
                    bs, contacts, warm, dt, self.iterations, coloring=self._coloring
                ),
            )

        # 7. Интегрирование поз динамических и кинематических тел.
        movers = awake + [
            e for h, e in sorted(entries.items())
            if e.body.is_kinematic and not e.external
            and (np.any(e.body.linear_velocity) or np.any(e.body.angular_velocity))
        ]
        if movers:
            self._integrate(movers, dt)
        for handle in sorted(self._external):
            ext = entries.get(handle)
            if ext is not None:
                ext.body.linear_velocity = np.zeros(3)
                ext.external = False
        self._external.clear()

        # 8. Засыпание островов.
        if awake:
            order = {e.handle: i for i, e in enumerate(awake)}
            dyn_flags = [True] * len(awake)
            island_pairs = [
                (order[ea.handle], order[eb.handle])
                for (ca, cb) in sorted(touching)
                for ea in (self._collider_owner[ca][0],)
                for eb in (self._collider_owner[cb][0],)
                if ea.handle in order and eb.handle in order
            ]
            driven = self._driven_handles(touching)
            for bodies_idx, _ in build_islands(len(awake), island_pairs, dyn_flags):
                members = [awake[i] for i in bodies_idx]
                if any(m.handle in driven for m in members):
                    # движущаяся кинематика не даёт острову уснуть
                    for m in members:
                        m.body.sleep_timer = 0.0
                    continue
                if update_island_sleep(
                    [m.body for m in members], dt, self.sleep_settings
                ):
                    handles = [m.handle for m in members]
                    for m in members:
                        m.group = handles

    def _awake_rows(self) -> np.ndarray:
        """Строки бодрствующих динамических тел по возрастанию handle."""
        bs = self._bodies
        order = bs.order
        return order[(bs.body_type[order] == BODY_DYNAMIC) & ~bs.sleeping[order]]

    def awake_dynamic_count(self) -> int:
        """Число бодрствующих динамических тел."""
        return int(self._awake_rows().size)

    def _entries_at(self, rows: np.ndarray) -> list[_Entry]:
        """Записи тел по строкам ``BodyStore`` (в порядке ``rows``)."""
        entries = self._entries
        return [entries[h] for h in self._bodies.handle[rows].tolist()]

    def _mover_flags(self) -> np.ndarray:
        """По строкам тел: движется ли тело (бодрствующее динамическое или
        кинематическое с ненулевой скоростью). Массив ``(count,)`` bool."""
        bs = self._bodies
        n = bs.count
        kind = bs.body_type[:n]
        moving = (kind == BODY_KINEMATIC) & (bs.vel[:n] != 0.0).any(axis=1)
        return ((kind == BODY_DYNAMIC) & ~bs.sleeping[:n]) | moving

    def _contacts(
        self,
        solid: np.ndarray,
        ca: np.ndarray,
        cb: np.ndarray,
        row_a: np.ndarray,
        row_b: np.ndarray,
        brow_a: np.ndarray,
        brow_b: np.ndarray,
        touching: set[tuple[int, int]],
    ) -> ContactBatch | None:
        """Узкая фаза твёрдых пар: контакты массивами в порядке пар ``solid``.

        Пары «куб — куб» считаются пакетом (``box_box_batch``), остальные —
        скалярным ``collide``; результаты сливаются в порядке пар (внутри пары —
        по номеру точки), поэтому порядок и ключи контактов те же, что при
        поштучной узкой фазе.

        Args:
            solid: Индексы нетриггерных пар в массивах ``ca``/``cb``.
            ca: cid первого коллайдера каждой пары.
            cb: cid второго коллайдера каждой пары.
            row_a: Строки ``ColliderStore`` первых коллайдеров.
            row_b: Строки ``ColliderStore`` вторых коллайдеров.
            brow_a: Строки ``BodyStore`` владельцев первых коллайдеров.
            brow_b: Строки ``BodyStore`` владельцев вторых коллайдеров.
            touching: Множество касаний; пополняется парами с контактом.

        Returns:
            Контакты для солвера или ``None``, если касаний нет.
        """
        if not solid.size:
            return None
        cs = self._colliders
        ra, rb = row_a[solid], row_b[solid]
        # Пары группируются по видам форм: каждая группа считается пакетом.
        combo = cs.kind[ra].astype(np.intp) * _KIND_SPAN + cs.kind[rb]
        parts = []  # (позиция пары в solid, слот, нормаль, точка, глубина)
        scalar: list[np.ndarray] = []  # группы без пакетной версии
        for code in np.unique(combo).tolist():
            pos = np.flatnonzero(combo == code)
            result = self._narrow_batch(*divmod(code, _KIND_SPAN), ra[pos], rb[pos])
            if result is None:
                scalar.append(pos)
                continue
            hit, normals, count, points, depths = result
            hit_idx = np.flatnonzero(hit)
            if hit_idx.size:
                valid = np.arange(points.shape[1]) < count[hit_idx][:, None]
                pi, slot = np.nonzero(valid)  # по парам, внутри пары по слотам
                src = hit_idx[pi]
                parts.append(
                    (pos[src], slot, normals[src], points[src, slot],
                     depths[src, slot])
                )
        if scalar:
            parts.append(self._scalar_contacts(np.sort(np.concatenate(scalar)), ra, rb))
            if not parts[-1][0].size:
                parts.pop()
        if not parts:
            return None
        if len(parts) == 1:
            pos, slot, normal, point, depth = parts[0]
        else:  # слияние в порядке пар: каждая часть уже упорядочена по (пара, слот)
            merged = [np.concatenate(column) for column in zip(*parts, strict=True)]
            order = np.argsort(merged[0], kind="stable")
            pos, slot, normal, point, depth = (column[order] for column in merged)
        if slot.max() >= MAX_SLOTS:
            raise ValueError(f"в манифолде больше {MAX_SLOTS} точек")

        # Пара — единица материала и события касания; контакт — единица солвера.
        starts = np.empty(len(pos), dtype=bool)
        starts[0] = True
        np.not_equal(pos[1:], pos[:-1], out=starts[1:])
        pair_of = np.cumsum(starts) - 1
        first = solid[pos[starts]]
        cid_a, cid_b = ca[first], cb[first]
        touching.update(zip(cid_a.tolist(), cid_b.tolist(), strict=True))
        items = cs.items
        mats = [
            (items[x].material, items[y].material)
            for x, y in zip(row_a[first].tolist(), row_b[first].tolist(), strict=True)
        ]
        friction = np.array([combine_friction(a, b) for a, b in mats])
        restitution = np.array([combine_restitution(a, b) for a, b in mats])
        whole = solid[pos]
        return ContactBatch(
            brow_a[whole], brow_b[whole], normal, point, depth,
            friction[pair_of], restitution[pair_of],
            contact_keys(cid_a[pair_of], cid_b[pair_of], slot),
        )

    def _narrow_batch(self, kind_a: int, kind_b: int, qa: np.ndarray, qb: np.ndarray):
        """Пакетная узкая фаза пар видов ``(kind_a, kind_b)`` или ``None``.

        Args:
            kind_a: Вид формы первого коллайдера (``KIND_*``).
            kind_b: Вид формы второго коллайдера.
            qa: Строки ``ColliderStore`` первых коллайдеров.
            qb: Строки ``ColliderStore`` вторых коллайдеров.

        Returns:
            ``(hit, normal, count, points, depths)`` или ``None``, если для вида
            пары нет пакетной версии (считается скалярным ``collide``).
        """
        cs = self._colliders
        if kind_a == KIND_BOX and kind_b == KIND_BOX:
            return box_box_batch(
                cs.world_pos[qa], cs.world_rot[qa], cs.half[qa],
                cs.world_pos[qb], cs.world_rot[qb], cs.half[qb],
            )
        if not self._batch_round:
            return None
        return collide_batch(
            kind_a, cs.world_pos[qa], cs.world_rot[qa], cs.half[qa],
            kind_b, cs.world_pos[qb], cs.world_rot[qb], cs.half[qb],
        )

    def _scalar_contacts(self, idx: np.ndarray, ra: np.ndarray, rb: np.ndarray):
        """Скалярная узкая фаза пар ``idx`` (позиции в ``solid``): колонки контактов."""
        items = self._colliders.items
        pos: list[int] = []
        slot: list[int] = []
        normal: list[np.ndarray] = []
        point: list[np.ndarray] = []
        depth: list[float] = []
        for p in idx.tolist():
            manifold = self._narrow(items[ra[p]], items[rb[p]])
            if manifold is None:
                continue
            for k in range(len(manifold.points)):
                pos.append(p)
                slot.append(k)
                normal.append(manifold.normal)
                point.append(manifold.points[k])
                depth.append(float(manifold.depths[k]))
        return (
            np.array(pos, dtype=np.intp), np.array(slot, dtype=np.intp),
            np.array(normal).reshape(-1, 3), np.array(point).reshape(-1, 3),
            np.array(depth),
        )

    def _driven_handles(self, touching: set[tuple[int, int]]) -> set[int]:
        """Handle тел, касающихся движущихся кинематических тел."""
        driven: set[int] = set()
        for ca, cb in touching:
            ea, eb = self._collider_owner[ca][0], self._collider_owner[cb][0]
            for kin, other in ((ea, eb), (eb, ea)):
                if kin.body.is_kinematic and self._is_mover(kin):
                    driven.add(other.handle)
        return driven

    @staticmethod
    def _is_sleeping(entry: _Entry) -> bool:
        return entry.body.sleeping

    @staticmethod
    def _is_mover(entry: _Entry) -> bool:
        b = entry.body
        if b.is_dynamic:
            return not b.sleeping
        if b.is_kinematic:
            return bool(np.any(b.linear_velocity) or np.any(b.angular_velocity))
        return False

    @staticmethod
    def _overlap(col_a: Collider, col_b: Collider) -> bool:
        return overlaps(
            col_a.shape, col_a.world_position, col_a.world_rotation,
            col_b.shape, col_b.world_position, col_b.world_rotation,
        )

    @staticmethod
    def _narrow(col_a: Collider, col_b: Collider):
        return collide(
            col_a.shape, col_a.world_position, col_a.world_rotation,
            col_b.shape, col_b.world_position, col_b.world_rotation,
        )

    def _integrate(self, movers: list[_Entry], dt: float) -> None:
        bs = self._bodies
        rows = np.fromiter((e.body._row for e in movers), dtype=np.intp,
                           count=len(movers))
        vel = bs.vel[rows]
        lin = vel[:, :3]
        ang = vel[:, 3:]
        speed = np.linalg.norm(lin, axis=1)
        over = speed > MAX_LINEAR_SPEED
        if over.any():
            lin[over] *= (MAX_LINEAR_SPEED / speed[over])[:, None]
        aspeed = np.linalg.norm(ang, axis=1)
        over = aspeed > MAX_ANGULAR_SPEED
        if over.any():
            ang[over] *= (MAX_ANGULAR_SPEED / aspeed[over])[:, None]
        quat = quats_integrate(bs.rotation[rows], ang, dt)
        mats = quats_to_matrices(quat)
        bs.position[rows] = bs.position[rows] + lin * dt
        bs.rotation[rows] = quat
        bs.vel[rows] = vel
        bs.inv_inertia_world[rows] = (
            mats @ bs.inv_inertia_local[rows] @ mats.transpose(0, 2, 1)
        )
        # Коллайдеры сдвинувшихся тел: поза и AABB одним пакетом.
        cs = self._colliders
        moving = np.zeros(bs.count, dtype=bool)
        moving[rows] = True
        n = cs.count
        body_row = bs.row_of_handle[cs.owner_handle[:n]]
        crows = np.flatnonzero(moving[body_row])
        if crows.size:
            brows = body_row[crows]
            cs.update_world(
                crows, bs.position[brows], quats_to_matrices(bs.rotation[brows])
            )

    def _emit_pair(self, cls, key: tuple[int, int]) -> None:
        ea = self._collider_owner.get(key[0])
        eb = self._collider_owner.get(key[1])
        if ea is None or eb is None:
            return
        if cls in (TriggerEnter, TriggerStay, TriggerExit) and not ea[1].is_trigger:
            ea, eb = eb, ea  # в событии триггера ``a`` — всегда сам триггер
        self.events.append(cls(ea[0].handle, eb[0].handle, ea[0].user_data,
                               eb[0].user_data))

    def _diff_events(self, now, before, enter_cls, stay_cls, exit_cls) -> None:
        for key in sorted(now - before):
            self._emit_pair(enter_cls, key)
        if stay_cls is not None:
            for key in sorted(now & before):
                self._emit_pair(stay_cls, key)
        for key in sorted(before - now):
            self._emit_pair(exit_cls, key)

    def pop_events(self) -> list[PhysicsEvent]:
        """Возвращает накопленные события и очищает очередь."""
        events, self.events = self.events, []
        return events

    # ---------------------------------------------------------------- запросы

    def _accept(self, col: Collider, layer_mask: int, include_triggers: bool) -> bool:
        if col.is_trigger and not include_triggers:
            return False
        return bool(col.layer & layer_mask)

    def raycast(
        self,
        origin,
        direction,
        max_distance: float = math.inf,
        layer_mask: int = ALL_LAYERS,
        include_triggers: bool = False,
        ignore: int | None = None,
    ) -> RaycastHit | None:
        """Ближайшее пересечение луча с коллайдерами."""
        ox, oy, oz = origin.tolist() if hasattr(origin, "tolist") else (
            float(v) for v in origin)
        dx, dy, dz = direction.tolist() if hasattr(direction, "tolist") else (
            float(v) for v in direction)
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-12:
            return None
        dx, dy, dz = dx / length, dy / length, dz / length
        best: RaycastHit | None = None
        if self._collider_owner:
            best = self._raycast_colliders(
                ox, oy, oz, dx, dy, dz, max_distance, layer_mask,
                include_triggers, ignore,
            )
        if self._meshes:
            best = self._raycast_meshes(
                ox, oy, oz, dx, dy, dz, max_distance, layer_mask,
                include_triggers, ignore, best,
            )
        return best

    def raycast_batch(
        self,
        origins,
        directions,
        max_distance: float = math.inf,
        layer_mask: int = ALL_LAYERS,
        include_triggers: bool = False,
        ignore: int | None = None,
    ) -> RaycastBatch:
        """Пакет лучей: ближайшее попадание на каждый луч.

        Mesh-коллайдеры обрабатываются пакетно (векторное отсечение по AABB и
        обход BVH без объектов на луч); обычные коллайдеры - по одному лучу
        (они и так быстрые), поэтому результат совпадает с :meth:`raycast`.

        Args:
            origins: Начала ``(n, 3)``.
            directions: Направления ``(n, 3)`` или ``(1, 3)`` (нормируются).
            max_distance: Предельное расстояние.
            layer_mask: Маска слоёв.
            include_triggers: Учитывать триггеры.
            ignore: Handle, который пропускается.

        Returns:
            :class:`RaycastBatch`.

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
        lengths = np.linalg.norm(dirs, axis=1)
        valid = lengths >= 1e-12
        unit = dirs / np.where(valid, lengths, 1.0)[:, None]
        dist = np.full(count, np.inf)
        handle = np.full(count, -1, dtype=np.int64)
        point = np.full((count, 3), np.nan)
        normal = np.full((count, 3), np.nan)
        if self._collider_owner:
            o_list, d_list = orig.tolist(), unit.tolist()
            for i in np.flatnonzero(valid).tolist():
                hit = self._raycast_colliders(
                    *o_list[i], *d_list[i], max_distance, layer_mask,
                    include_triggers, ignore,
                )
                if hit is not None:
                    dist[i], handle[i] = hit.distance, hit.handle
                    point[i], normal[i] = hit.point, hit.normal
        for mesh in self._meshes.values():
            if mesh.handle == ignore or not self._mesh_accept(mesh, layer_mask,
                                                              include_triggers):
                continue
            idx = np.flatnonzero(valid)
            m_dist, m_normal = mesh.raycast_batch(orig[idx], unit[idx], max_distance)
            better = m_dist < dist[idx]
            if better.any():
                sel = idx[better]
                dist[sel] = m_dist[better]
                handle[sel] = mesh.handle
                normal[sel] = m_normal[better]
                point[sel] = orig[sel] + unit[sel] * dist[sel][:, None]
        return RaycastBatch(dist, handle, point, normal)

    def _raycast_colliders(
        self, ox, oy, oz, dx, dy, dz, max_distance, layer_mask, include_triggers,
        ignore,
    ) -> RaycastHit | None:
        o = np.array((ox, oy, oz))
        d = np.array((dx, dy, dz))
        if math.isinf(max_distance):
            candidates = sorted(self._collider_owner)
        else:
            end = o + d * max_distance
            candidates = sorted(
                self.broadphase.query_aabb((np.minimum(o, end), np.maximum(o, end)))
            )
        best: RaycastHit | None = None
        for cid in candidates:
            entry, col = self._collider_owner[cid]
            if entry.handle == ignore or not self._accept(col, layer_mask,
                                                          include_triggers):
                continue
            lo, hi = col.world_aabb
            if not ray_aabb(o, d, lo, hi, max_distance):
                continue
            hit = ray_shape(o, d, col.shape, col.world_position, col.world_rotation)
            if hit is None or hit[0] > max_distance:
                continue
            if best is None or hit[0] < best.distance:
                best = RaycastHit(entry.handle, col, o + d * hit[0], hit[1], hit[0],
                                  entry.user_data)
        return best

    def _raycast_meshes(
        self, ox, oy, oz, dx, dy, dz, max_distance, layer_mask, include_triggers,
        ignore, best: RaycastHit | None,
    ) -> RaycastHit | None:
        for mesh in self._meshes.values():
            if mesh.handle == ignore or not self._mesh_accept(mesh, layer_mask,
                                                              include_triggers):
                continue
            limit = max_distance if best is None else min(max_distance, best.distance)
            hit = mesh.raycast_floats(ox, oy, oz, dx, dy, dz, limit)
            if hit is not None and (best is None or hit[0] < best.distance):
                t = hit[0]
                best = RaycastHit(
                    mesh.handle, mesh,
                    np.array((ox + dx * t, oy + dy * t, oz + dz * t)),
                    np.asarray(hit[1]), t, mesh.user_data,
                )
        return best

    def overlap(
        self,
        shape: Shape,
        position,
        rotation=IDENTITY_QUAT,
        layer_mask: int = ALL_LAYERS,
        include_triggers: bool = False,
    ) -> list[int]:
        """Handle тел, чьи коллайдеры пересекают форму (по возрастанию)."""
        pos = vec3(position)
        rot = as_matrix(rotation)
        found: set[int] = set()
        for cid in self.broadphase.query_aabb(shape.aabb(pos, rot)):
            entry, col = self._collider_owner[cid]
            if not self._accept(col, layer_mask, include_triggers):
                continue
            if collide(shape, pos, rot, col.shape, col.world_position,
                       col.world_rotation) is not None:
                found.add(entry.handle)
        for mesh in self._meshes.values():
            if self._mesh_accept(mesh, layer_mask, include_triggers) and mesh.overlaps(
                shape, pos, rot
            ):
                found.add(mesh.handle)
        return sorted(found)

    def overlap_sphere(self, center, radius: float, layer_mask: int = ALL_LAYERS,
                       include_triggers: bool = False) -> list[int]:
        """Overlap шаром (удобная обёртка)."""
        from engine.physics.shape import SphereShape

        return self.overlap(SphereShape(radius), center, IDENTITY_QUAT, layer_mask,
                            include_triggers)

    def region_fingerprint(self, lo, hi, ignore: int | None = None) -> bytes | None:
        """Отпечаток коллайдеров, чьи AABB пересекают область ``[lo, hi]``.

        Равные отпечатки значат: в области тот же состав коллайдеров с теми же
        формами, слоями и мировыми позами, - любой запрос (``shapecast``,
        ``raycast``, ``collide``) внутри области даст тот же ответ. Скорости и сон
        тел не входят: запросы их не читают. Цена - один проход по AABB без
        сортировки широкой фазы.

        Args:
            lo: Нижний угол области.
            hi: Верхний угол области.
            ignore: Handle тела, чьи коллайдеры не учитываются (сам персонаж).

        Returns:
            Байтовая строка или ``None``, если в области есть коллайдер вида
            ``KIND_OTHER`` (его форма вне хранилища, отпечаток был бы неполным).
        """
        store = self._colliders
        n = store.count
        inside = (store.aabb_lo[:n] <= hi).all(axis=1)
        inside &= (store.aabb_hi[:n] >= lo).all(axis=1)
        if ignore is not None:
            inside &= store.owner_handle[:n] != ignore
        rows = np.flatnonzero(inside)
        kind = store.kind[rows]
        if (kind == KIND_OTHER).any():
            return None
        meshes = b"".join(
            m.key_bytes() for m in self._meshes.values()
            if m.handle != ignore
            and (m.world_aabb[0] <= hi).all() and (m.world_aabb[1] >= lo).all()
        )
        return b"".join((
            meshes,
            store.cid[rows].tobytes(), kind.tobytes(), store.half[rows].tobytes(),
            store.layer[rows].tobytes(), store.mask[rows].tobytes(),
            store.is_trigger[rows].tobytes(), store.world_pos[rows].tobytes(),
            store.world_rot[rows].tobytes(),
        ))

    def shapecast(
        self,
        shape: Shape,
        position,
        rotation,
        direction,
        max_distance: float,
        layer_mask: int = ALL_LAYERS,
        include_triggers: bool = False,
        ignore: int | None = None,
    ) -> RaycastHit | None:
        """Переносит форму вдоль ``direction`` и возвращает первое касание.

        Поиск шагами не длиннее четверти минимального размера формы с уточнением
        бисекцией (без полноценного CCD: тонкие препятствия тоньше шага могут быть
        пропущены, но шаг мал по сравнению с размером самой формы).
        """
        pos = vec3(position)
        rot = as_matrix(rotation)
        d = vec3(direction)
        length = float(np.linalg.norm(d))
        if length < 1e-12 or max_distance <= 0.0:
            return None
        d = d / length
        lo0, hi0 = shape.aabb(pos, rot)
        end = pos + d * max_distance
        lo1, hi1 = shape.aabb(end, rot)
        size = float(np.min(hi0 - lo0))
        step = max(size * 0.25, 1e-3)
        cands = []
        for cid in sorted(self.broadphase.query_aabb(
                (np.minimum(lo0, lo1), np.maximum(hi0, hi1)))):
            entry, col = self._collider_owner[cid]
            if entry.handle == ignore or not self._accept(col, layer_mask,
                                                          include_triggers):
                continue
            cands.append((entry, col))
        best: RaycastHit | None = None
        for entry, col in cands:
            limit = max_distance if best is None else min(max_distance, best.distance)
            hit = self._sweep_one(shape, pos, rot, d, limit, step, col, lo0, hi0)
            if hit is not None and (best is None or hit[0] < best.distance):
                t, normal, point = hit
                best = RaycastHit(entry.handle, col, point, normal, t, entry.user_data)
        for mesh in self._meshes.values():
            if mesh.handle == ignore or not self._mesh_accept(mesh, layer_mask,
                                                              include_triggers):
                continue
            limit = max_distance if best is None else min(max_distance, best.distance)
            prepared = mesh.prepare_sweep(shape, pos, rot, d, limit, lo0, hi0)
            if prepared is None:
                continue
            overlap, contact = prepared
            t = _first_overlap(overlap, limit, step)
            if t is None:
                continue
            found = contact(t)
            if found is not None and (best is None or t < best.distance):
                best = RaycastHit(mesh.handle, mesh, found[2], found[1], t,
                                  mesh.user_data)
        return best

    @staticmethod
    def _sweep_one(shape, pos, rot, d, limit, step, col, lo0, hi0):
        # AABB формы в точке t = [lo0, hi0] + d t; пока он не пересекает AABB
        # коллайдера, дорогую узкую фазу не вызываем.
        c_lo, c_hi = col.world_aabb
        lo_t = (lo0 - c_hi).tolist()
        hi_t = (hi0 - c_lo).tolist()
        dx, dy, dz = d.tolist()

        def test(t):
            if (
                lo_t[0] + dx * t > 0.0 or hi_t[0] + dx * t < 0.0
                or lo_t[1] + dy * t > 0.0 or hi_t[1] + dy * t < 0.0
                or lo_t[2] + dz * t > 0.0 or hi_t[2] + dz * t < 0.0
            ):
                return None
            return collide(shape, pos + d * t, rot, col.shape, col.world_position,
                           col.world_rotation)

        overlap = _pair_overlap(shape, pos, rot, d, col, lo_t, hi_t)
        if overlap is not None:
            # Марш и бисекция идут по дешёвому предикату, полный контакт строится
            # один раз в найденной точке. Если предикат и ``collide`` разошлись
            # (граничный случай), берём прежний путь.
            found = _first_overlap(overlap, limit, step)
            if found is None:
                return None
            m = collide(shape, pos + d * found, rot, col.shape, col.world_position,
                        col.world_rotation)
            if m is not None:
                return found, -m.normal, m.point

        m = test(0.0)
        if m is not None:
            return 0.0, -m.normal, m.point
        t_prev = 0.0
        t = 0.0
        while t < limit:
            t = min(t + step, limit)
            m = test(t)
            if m is not None:
                lo, hi = t_prev, t
                for _ in range(_BISECT_ITERATIONS):
                    mid = 0.5 * (lo + hi)
                    mid_m = test(mid)
                    if mid_m is None:
                        lo = mid
                    else:
                        hi, m = mid, mid_m
                return hi, -m.normal, m.point
            t_prev = t
        return None
