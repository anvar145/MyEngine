"""Контроллер персонажа: капсула, скользящее движение, ступени, склоны, прыжок.

Персонаж не участвует в солвере: он двигается собственными запросами
``shapecast`` по миру (collide-and-slide) и затем переносит кинематический
двойник в мире (``move_kinematic``), поэтому динамические тела и триггеры
реагируют на него как на обычное тело.
"""

from __future__ import annotations

import math

import numpy as np

from engine.physics.body import RigidBody
from engine.physics.collider import Collider
from engine.physics.mathutil import vec3
from engine.physics.shape import BoxShape, CapsuleShape
from engine.physics.world import ALL_LAYERS, PhysicsWorld, RaycastHit

_UP = np.array([0.0, 1.0, 0.0])
_EYE = np.eye(3)
MAX_SLIDES = 4
# Точность свипа (бисекция ``shapecast``, до ≈1 мм): продвижение меньше этого
# значения считается «упёрся».
_SWEEP_EPS = 1e-3
# Остаток скольжения короче ``_SLIDE_EPS`` не свипается; нормали с косинусом
# выше ``_SAME_NORMAL`` считаются одной плоскостью.
_SLIDE_EPS = 1e-5
_SAME_NORMAL = 0.9999
_STEP_OK, _STEP_BLOCKED, _STEP_VOID, _STEP_STEEP, _STEP_HIGH = range(5)
_ZERO = np.zeros(3)
# Частота, при которой ``air_control`` — доля сближения скоростей за один кадр;
# для другого ``dt`` доля пересчитывается, чтобы не зависеть от частоты шага.
_REFERENCE_FPS = 60.0
# Запас вокруг капсулы (м) для отпечатка окрестности покоящегося шага: сверх
# спуска на ступень, пробы земли и пути гравитации за кадр.
_REST_MARGIN = 0.05


def _require(name: str, value: float, low: float | None = None,
             high: float | None = None, open_low: bool = False) -> None:
    """Проверяет числовой параметр движения; ``ValueError`` с именем параметра."""
    ok = math.isfinite(value)
    if ok and low is not None:
        ok = value > low if open_low else value >= low
    if ok and high is not None:
        ok = value <= high
    if not ok:
        bounds = f"{'>' if open_low else '>='} {low}" if low is not None else ""
        if high is not None:
            bounds += f"{', ' if bounds else ''}<= {high}"
        raise ValueError(f"{name} must be {bounds} (got {name}={value})")


class CharacterController:
    """Капсульный контроллер. ``position`` — центр капсулы.

    Осознанные ограничения kinematic-контроллера (не баги):

    * персонаж **не толкает динамические тела** — для него они такая же
      преграда, как стена;
    * персонаж **не ездит на горизонтально движущихся платформах** (вертикальное
      движение платформы он повторяет, горизонтальное — нет).

    Args:
        world: Физический мир.
        position: Центр капсулы.
        radius: Радиус капсулы, строго больше 0.
        height: Полная высота капсулы, строго больше ``2 * radius``.
        step_offset: Максимальная высота ступени, не меньше 0.
        slope_limit: Предельный угол проходимого склона в градусах.
        skin_width: Зазор до поверхностей, строго больше 0.
        layer: Слой двойника персонажа.
        mask: Маска слоёв, с которыми персонаж сталкивается.
        user_data: Произвольные данные, привязанные к двойнику.
        move_speed: Скорость ходьбы для ``move_input``, м/с, не меньше 0.
        sprint_multiplier: Множитель скорости бега в ``move_input``, не меньше 1.
        jump_velocity: Скорость прыжка для ``jump()`` без аргумента, м/с, > 0.
        use_gravity: Применять ли гравитацию; иначе вертикальной скорости нет,
            и прыжок невозможен.
        gravity_scale: Множитель гравитации мира, не меньше 0.
        ground_check_distance: Дальность пробы земли сверх ``skin_width``, > 0.
        air_control: Доля управляемости в воздухе для ``move_input``, [0, 1].
        snap_to_ground: Прилипать к земле при спуске по склону и ступеням.

    Raises:
        ValueError: При недопустимых ``radius``, ``height``, ``skin_width``,
            ``step_offset`` или числовых параметрах движения.
    """

    def __init__(
        self,
        world: PhysicsWorld,
        position=(0.0, 0.0, 0.0),
        radius: float = 0.4,
        height: float = 1.8,
        step_offset: float = 0.3,
        slope_limit: float = 45.0,
        skin_width: float = 0.02,
        layer: int = 1,
        mask: int = ALL_LAYERS,
        user_data=None,
        move_speed: float = 4.5,
        sprint_multiplier: float = 1.6,
        jump_velocity: float = 5.0,
        use_gravity: bool = True,
        gravity_scale: float = 1.0,
        ground_check_distance: float = 0.03,
        air_control: float = 0.3,
        snap_to_ground: bool = True,
    ) -> None:
        if radius <= 0.0:
            raise ValueError(f"radius must be > 0 (got radius={radius})")
        if height <= 2.0 * radius:
            raise ValueError(
                f"height must be > 2*radius (got height={height}, radius={radius})"
            )
        if skin_width <= 0.0:
            raise ValueError(f"skin_width must be > 0 (got skin_width={skin_width})")
        if step_offset < 0.0:
            raise ValueError(
                f"step_offset must be >= 0 (got step_offset={step_offset})"
            )
        _require("move_speed", move_speed, low=0.0)
        _require("sprint_multiplier", sprint_multiplier, low=1.0)
        _require("jump_velocity", jump_velocity, low=0.0, open_low=True)
        _require("gravity_scale", gravity_scale, low=0.0)
        _require("ground_check_distance", ground_check_distance, low=0.0,
                 open_low=True)
        _require("air_control", air_control, low=0.0, high=1.0)
        self.world = world
        self.radius = float(radius)
        self.height = float(height)
        self.step_offset = float(step_offset)
        self.slope_limit = float(slope_limit)
        self.skin_width = float(skin_width)
        self.mask = int(mask)
        self.move_speed = float(move_speed)
        self.sprint_multiplier = float(sprint_multiplier)
        self.jump_velocity = float(jump_velocity)
        self.use_gravity = bool(use_gravity)
        self.gravity_scale = float(gravity_scale)
        self.ground_check_distance = float(ground_check_distance)
        self.air_control = float(air_control)
        self.snap_to_ground = bool(snap_to_ground)
        self.shape = CapsuleShape(self.radius, self.height / 2.0 - self.radius)
        self.position = vec3(position).copy()
        self.vertical_velocity = 0.0
        self.horizontal_velocity = np.zeros(3)
        self.grounded = False
        self._landed_hit: RaycastHit | None = None
        self._rest_key: tuple | None = None  # отпечаток покоя, см. ``step``
        self.ground_normal = _UP.copy()
        body = RigidBody(body_type="kinematic", position=self.position)
        self.handle = world.add_body(
            body, Collider(self.shape, layer=layer, mask=mask), user_data
        )
        self._update_ground()

    # ------------------------------------------------------------- свойства

    @property
    def slope_limit(self) -> float:
        return self._slope_limit

    @slope_limit.setter
    def slope_limit(self, degrees: float) -> None:
        self._slope_limit = float(degrees)
        self._min_ground_y = math.cos(math.radians(self._slope_limit))

    @property
    def is_grounded(self) -> bool:
        return self.grounded

    def remove(self) -> None:
        """Убирает двойника персонажа из мира."""
        if self.handle in self.world:
            self.world.remove_body(self.handle)

    def refresh_ground(self) -> None:
        """Пересчитывает ``grounded`` по текущему миру, не двигая персонажа.

        Нужен, когда мир изменился вокруг стоящего персонажа (например, статику
        добавили после его создания): сам ``grounded`` обновляется только при
        движении и телепорте.
        """
        self._update_ground()

    def teleport(self, position) -> None:
        """Мгновенный перенос (без проверки столкновений)."""
        self.position = vec3(position).copy()
        self.vertical_velocity = 0.0
        self.horizontal_velocity = np.zeros(3)
        self.world.set_pose(self.handle, self.position)
        self._update_ground()

    # --------------------------------------------------------------- запросы

    def _sweep(self, direction: np.ndarray, distance: float) -> RaycastHit | None:
        return self.world.shapecast(
            self.shape, self.position, _EYE, direction, distance, self.mask,
            ignore=self.handle,
        )

    def _walkable(self, normal: np.ndarray) -> bool:
        return bool(normal[1] >= self._min_ground_y)

    def _depenetrate(self) -> None:
        """Выталкивает капсулу из геометрии, в которую она попала."""
        from engine.physics.narrowphase import (
            _capsule_box_overlap,
            _capsule_capsule_overlap,
            collide,
        )

        for _ in range(4):
            push = np.zeros(3)
            lo, hi = self.shape.aabb(self.position, _EYE)
            hits = 0
            for cid in sorted(self.world.broadphase.query_aabb((lo, hi))):
                entry, col = self.world._collider_owner[cid]
                if entry.handle == self.handle or col.is_trigger:
                    continue
                if not col.layer & self.mask:
                    continue
                # дешёвый предикат отсекает боксы, которых капсула не касается
                kind = type(col.shape)
                if kind is BoxShape:
                    quick = _capsule_box_overlap
                elif kind is CapsuleShape:
                    quick = _capsule_capsule_overlap
                else:
                    quick = None
                if quick is not None and not quick(
                    self.shape, self.position, _EYE, col.shape,
                    col.world_position, col.world_rotation,
                ):
                    continue
                m = collide(self.shape, self.position, _EYE, col.shape,
                            col.world_position, col.world_rotation)
                if m is not None and m.depth > 1e-5:
                    push -= m.normal * m.depth
                    hits += 1
            # mesh: берём самый глубокий треугольник (соседние грани одной
            # поверхности дали бы двойной толчок); остальное дожмут итерации
            contacts = self.world.mesh_contacts(self.shape, self.position, _EYE,
                                                self.mask)
            if contacts:
                normal, depth = max(contacts, key=lambda c: c[1])
                if depth > 1e-5:
                    push += normal * depth
                    hits += 1
            if not hits:
                return
            self.position = self.position + push

    def _update_ground(self, known_hit: RaycastHit | None = None) -> None:
        probe = self.skin_width + self.ground_check_distance
        # приземление в _move_vertical уже дало проходимый хит у этой позиции
        hit = known_hit if known_hit is not None else self._sweep(-_UP, probe)
        self.grounded = False
        self.ground_normal = _UP.copy()
        if hit is None:
            return
        normal = hit.normal
        if not self._walkable(normal):
            # на ребре ступени нормаль капсулы наклонена: смотрим луч вниз
            foot = self.position - _UP * (self.height / 2.0 - 0.05)
            ray = self.world.raycast(foot, -_UP, probe + 0.05, self.mask,
                                     ignore=self.handle)
            if ray is None or not self._walkable(ray.normal):
                return
            normal = ray.normal
        self.grounded = True
        self.ground_normal = normal.copy()

    # ------------------------------------------------------------- движение

    def move(self, motion, dt: float | None = None) -> np.ndarray:
        """Двигает персонажа на ``motion`` с учётом столкновений.

        Сначала движение в плоскости XZ (со скольжением и подъёмом на ступени),
        затем по Y. Возвращает фактическое смещение.
        """
        motion = vec3(motion)
        start = self.position.copy()
        self._depenetrate()
        was_grounded = self.grounded
        horizontal = np.array([motion[0], 0.0, motion[2]])
        if float(horizontal @ horizontal) > 1e-18:
            self._slide(horizontal, allow_step=was_grounded)
        self._landed_hit = None
        landed = self._move_vertical(float(motion[1]))
        if self.snap_to_ground and was_grounded and motion[1] <= 1e-9 and not landed:
            self._snap_to_ground()
        self._update_ground(self._landed_hit if landed else None)
        self.world.move_kinematic(self.handle, self.position, dt=dt)
        return self.position - start

    def _slide(self, vector: np.ndarray, allow_step: bool) -> None:
        remaining = vector
        contacts: list[np.ndarray] = []
        for _ in range(MAX_SLIDES):
            dist = float(np.linalg.norm(remaining))
            if dist < _SLIDE_EPS:
                return
            direction = remaining / dist
            hit = self._sweep(direction, dist + self.skin_width)
            if hit is None or hit.distance - self.skin_width >= dist:
                self.position = self.position + remaining
                return
            advance = max(hit.distance - self.skin_width, 0.0)
            self.position = self.position + direction * advance
            left = dist - advance
            normal = hit.normal
            if any(float(normal @ seen) > _SAME_NORMAL for seen in contacts):
                return  # повторный контакт с той же плоскостью: скользить некуда
            contacts.append(normal)
            walkable = self._walkable(normal)
            if allow_step and not walkable and abs(normal[1]) < 0.7:
                flat = np.array([direction[0], 0.0, direction[2]])
                length = float(np.linalg.norm(flat))
                if length > 1e-9 and self._try_step(flat / length, dist * length):
                    return
            rest = direction * left
            slide = rest - normal * float(rest @ normal)
            if not walkable and slide[1] > 0.0:
                slide[1] = 0.0
            remaining = slide

    def _try_step(self, direction: np.ndarray, distance: float) -> bool:
        """Подъём на ступень: вверх, вперёд, вниз; откат, если не вышло.

        ``distance`` — полное горизонтальное смещение кадра. Сначала шаг вперёд
        на него; если за ребром нашлась лишь непроходимая поверхность (нижняя
        полусфера села на ребро), повтор на ``max(distance, radius / 2)``.
        Рывок вперёд случается только во втором, редком случае.
        """
        origin = self.position.copy()
        up = self._sweep(_UP, self.step_offset + self.skin_width)
        raise_by = (
            self.step_offset if up is None
            else max(up.distance - self.skin_width, 0.0)
        )
        if raise_by < _SWEEP_EPS:
            return False
        ground_y = float(origin[1]) - self.height / 2.0 - self.skin_width
        travels = [distance]
        if self.radius * 0.5 > distance:
            travels.append(self.radius * 0.5)
        for travel in travels:
            self.position = origin + _UP * raise_by
            status = self._step_forward_down(direction, travel, raise_by, ground_y)
            if status == _STEP_OK:
                return True
            if status != _STEP_STEEP:
                break  # препятствие спереди или пустота: повтор не поможет
        self.position = origin
        return False

    def _step_forward_down(self, direction: np.ndarray, travel: float,
                           raise_by: float, ground_y: float) -> int:
        """Шаг вперёд с поднятой позиции и посадка вниз (см. ``_try_step``)."""
        forward = self._sweep(direction, travel + self.skin_width)
        advance = (
            travel if forward is None
            else max(forward.distance - self.skin_width, 0.0)
        )
        if advance < _SWEEP_EPS:
            return _STEP_BLOCKED
        self.position = self.position + direction * advance
        down = self._sweep(-_UP, raise_by + self.skin_width + 0.02)
        if down is None:
            return _STEP_VOID
        if float(down.point[1]) - ground_y > self.step_offset + 1e-3:
            return _STEP_HIGH  # посадка на ребро выше step_offset: не ступень
        if not self._walkable(down.normal):
            # ребро ступени: проверяем истинную нормаль поверхности лучом
            foot = self.position - _UP * (self.height / 2.0 - 0.05)
            ray = self.world.raycast(foot, -_UP, raise_by + 0.1, self.mask,
                                     ignore=self.handle)
            if ray is None or not self._walkable(ray.normal):
                return _STEP_STEEP
            # луч должен попасть в поверхность под самой посадкой, а не в пол
            # перед ступенью, лежащий ниже нижнего полюса
            landed_pole = (self.position[1] - max(down.distance - self.skin_width, 0.0)
                           - self.height / 2.0)
            if abs(float(ray.point[1]) - landed_pole) > self.skin_width + 0.08:
                return _STEP_STEEP
        self.position = self.position - _UP * max(
            down.distance - self.skin_width, 0.0
        )
        return _STEP_OK

    def _move_vertical(self, dy: float) -> bool:
        """Движение по Y. Возвращает ``True``, если упёрлись в землю."""
        if abs(dy) < 1e-9:
            return False
        sign = 1.0 if dy > 0.0 else -1.0
        dist = abs(dy)
        hit = self._sweep(_UP * sign, dist + self.skin_width)
        if hit is None or hit.distance - self.skin_width >= dist:
            self.position = self.position + _UP * dy
            return False
        advance = max(hit.distance - self.skin_width, 0.0)
        self.position = self.position + _UP * (sign * advance)
        if sign < 0.0:
            if not self._walkable(hit.normal):
                # крутой склон: скользим вниз вдоль него
                left = _UP * (-(dist - advance))
                slide = left - hit.normal * float(left @ hit.normal)
                self._slide(slide, allow_step=False)
                return False
            self._landed_hit = hit
            return True
        return False

    def _snap_to_ground(self) -> None:
        """Прилипание к земле при спуске по склону или лестнице."""
        hit = self._sweep(-_UP, self.step_offset + self.skin_width)
        if hit is None or not self._walkable(hit.normal):
            return
        self.position = self.position - _UP * max(hit.distance - self.skin_width, 0.0)

    def simple_move(self, velocity, dt: float | None = None) -> np.ndarray:
        """Движение с гравитацией: ``velocity`` — горизонтальная скорость (XZ).

        Вертикальная скорость хранится в ``vertical_velocity`` (гравитация мира
        × ``gravity_scale``, прыжок). Скорость берётся как есть: бег и
        ``air_control`` относятся только к ``move_input``. Возвращает фактическое
        смещение.
        """
        dt = self.world.fixed_dt if dt is None else float(dt)
        v = vec3(velocity)
        self.horizontal_velocity = np.array([v[0], 0.0, v[2]])
        return self._advance(self.horizontal_velocity, dt)

    def step(self, dt: float | None = None) -> np.ndarray:
        """Шаг без ввода: только гравитация (и нулевая горизонтальная скорость).

        Адаптер зовёт его для персонажей, которых за этот физический шаг не
        двигал скрипт, - так персонаж, заспавненный в воздухе, падает на землю.

        Стоящий персонаж возвращается этим шагом в ту же точку, поэтому полный
        расчёт (``shapecast`` и ``move_kinematic``) выполняется один раз: шаг,
        после которого состояние не изменилось, запоминается под отпечатком
        входов (поза, скорости, параметры, гравитация, ``dt``, окрестность
        капсулы), и пока отпечаток тот же, шаг пропускается и возвращает нули.
        Результат при этом побитово тот же, что у полного расчёта: шаг зависит
        только от этих входов. Явные ``move``/``simple_move``/``move_input``
        быстрого пути не имеют.
        """
        dt = self.world.fixed_dt if dt is None else float(dt)
        key = self._rest_fingerprint(dt) if self.vertical_velocity == 0.0 else None
        if key is not None and key == self._rest_key:
            return np.zeros(3)
        actual = self.simple_move(_ZERO, dt)
        resting = key is not None and key[0] == self._state_key()
        self._rest_key = key if resting else None
        return actual

    def _state_key(self) -> tuple:
        """Изменяемое состояние, которое читает и пишет шаг."""
        return (
            self.position.tobytes(), self.vertical_velocity,
            self.horizontal_velocity.tobytes(), self.grounded,
        )

    def _rest_fingerprint(self, dt: float) -> tuple | None:
        """Отпечаток входов пассивного шага; ``None`` - окрестность не опознать."""
        gravity_y = float(self.world.gravity[1])
        reach = (
            self.step_offset + self.skin_width + self.ground_check_distance
            + _REST_MARGIN + abs(gravity_y) * self.gravity_scale * dt * dt
        )
        lo, hi = self.shape.aabb(self.position, _EYE)
        region = self.world.region_fingerprint(lo - reach, hi + reach, self.handle)
        if region is None:
            return None
        params = (
            self.skin_width, self.step_offset, self._slope_limit, self.mask,
            self.snap_to_ground, self.ground_check_distance, self.use_gravity,
            self.gravity_scale, gravity_y, dt,
        )
        return (self._state_key(), params, region)

    def move_input(self, direction, dt: float | None = None, *, sprint: bool = False,
                   air_control: float | None = None) -> np.ndarray:
        """Движение по направлению ввода: ``direction * move_speed`` и гравитация.

        Args:
            direction: Направление в мировых осях; важны X и Z, длина больше 1
                сжимается до 1 (диагональ не быстрее), меньше 1 - частичный отклон.
            dt: Шаг, секунды; по умолчанию ``world.fixed_dt``.
            sprint: Умножить скорость на ``sprint_multiplier``.
            air_control: Переопределяет поле ``air_control`` на этот вызов.

        Returns:
            Фактическое смещение. На земле скорость равна желаемой сразу; в
            воздухе она сближается с желаемой на долю ``air_control`` за кадр
            (при другом ``dt`` доля пересчитывается), остальное - инерция.
        """
        dt = self.world.fixed_dt if dt is None else float(dt)
        d = vec3(direction)
        scale = self.move_speed * (self.sprint_multiplier if sprint else 1.0)
        length = math.hypot(float(d[0]), float(d[2]))
        if length > 1.0:
            scale /= length
        wanted = np.array([d[0] * scale, 0.0, d[2] * scale])
        if not self.grounded:
            control = self.air_control if air_control is None else air_control
            control = min(max(float(control), 0.0), 1.0)
            blend = 1.0 - (1.0 - control) ** (dt * _REFERENCE_FPS)
            current = self.horizontal_velocity
            wanted = current + (wanted - current) * blend
        self.horizontal_velocity = wanted
        return self._advance(wanted, dt)

    def _advance(self, velocity: np.ndarray, dt: float) -> np.ndarray:
        """Шаг с гравитацией - общая часть ``simple_move`` и ``move_input``."""
        if self.grounded and self.vertical_velocity < 0.0:
            self.vertical_velocity = 0.0
        if self.use_gravity:
            self.vertical_velocity += (
                float(self.world.gravity[1]) * self.gravity_scale * dt
            )
        else:
            self.vertical_velocity = 0.0
        wanted = np.array(
            [velocity[0] * dt, self.vertical_velocity * dt, velocity[2] * dt]
        )
        actual = self.move(wanted, dt)
        if self.vertical_velocity > 0.0 and actual[1] < wanted[1] - 1e-6:
            self.vertical_velocity = 0.0  # упёрлись головой
        elif self.vertical_velocity < 0.0 and self.grounded:
            self.vertical_velocity = 0.0
        return actual

    def jump(self, speed: float | None = None) -> bool:
        """Прыжок с начальной вертикальной скоростью (только с земли).

        Двойной и воздушный прыжок запрещены: в воздухе вызов ничего не делает.
        Без гравитации (``use_gravity=False``) прыжка тоже нет.

        Args:
            speed: Начальная скорость вверх, м/с; строго больше 0. ``None`` -
                ``jump_velocity``.

        Returns:
            True, если прыжок начат; False, если персонаж не на земле или
            гравитация выключена.

        Raises:
            ValueError: Если ``speed <= 0``.
        """
        speed = self.jump_velocity if speed is None else float(speed)
        if speed <= 0.0:
            raise ValueError(f"jump speed must be > 0 (got speed={speed})")
        if not self.grounded or not self.use_gravity:
            return False
        self.vertical_velocity = float(speed)
        self.grounded = False
        return True

