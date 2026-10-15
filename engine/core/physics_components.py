"""Физические компоненты: данные, которые превращаются в тела мира физики.

Компоненты не содержат симуляции — их читает :class:`~engine.core.physics_adapter.
ScenePhysicsAdapter` (``scene.physics``) и создаёт по ним тела
:class:`~engine.physics.world.PhysicsWorld`. Изменение публичного поля у
работающего компонента помечает тело на пересоздание (поза и скорости
сохраняются).

Соглашения:

* трансформ объекта — центр тела; размеры коллайдеров умножаются на мировой
  масштаб объекта (как у обычного меша: куб размера 1 с масштабом ``(10, 0.2, 10)``
  даёт пол);
* объект с ``ColliderComponent`` без ``RigidBodyComponent`` — статика;
* коллайдеры дочерних объектов не объединяются с телом родителя.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from engine.core.component import Component
from engine.core.reflection import FieldKind, component_fields, declare, registry

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.game_object import GameObject
    from engine.core.resources.asset_ref import AssetRef

__all__ = [
    "BODY_TYPES",
    "COMBINE_MODES",
    "SHAPE_TYPES",
    "CharacterControllerComponent",
    "ColliderComponent",
    "MeshColliderComponent",
    "RigidBodyComponent",
    "clamp_character_field",
    "has_character_rigidbody_conflict",
]

_log = logging.getLogger("engine.core.physics_components")

BODY_TYPES: tuple[str, ...] = ("static", "kinematic", "dynamic")
SHAPE_TYPES: tuple[str, ...] = ("box", "sphere", "capsule")
_ALL_LAYERS = 0xFFFFFFFF
_IDENTITY_QUAT: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
COMBINE_MODES: tuple[str, ...] = ("multiply", "min", "max", "average")



def _pfield(name: str, default: Any, **options: Any) -> Any:
    """Поле физического компонента: по умолчанию правка пересоздаёт тело."""
    options.setdefault("live", False)
    return declare(name, default, **options)


def _finish(cls: type) -> type:
    """Регистрирует поля класса и выводит из них набор «живых» полей."""
    component_fields(cls)
    cls._live_fields = frozenset(  # type: ignore[attr-defined]
        d.name for d in registry.fields(cls) if d.live
    )
    return cls


def clamp_character_field(name: str, value: float) -> float:
    """Приводит числовое поле персонажа к допустимому диапазону.

    Допустимые значения возвращаются без изменений. Значение вне диапазона
    заменяется ближайшей границей (для открытого минимума - ``1e-3``), а
    нечисловое (``nan``/``inf``) - значением по умолчанию. Замена тихая: о ней
    предупреждает вызывающий код.

    Args:
        name: Имя поля (``move_speed``, ``sprint_multiplier``, ``jump_velocity``,
            ``gravity_scale``, ``ground_check_distance``, ``air_control``).
        value: Исходное значение.

    Returns:
        Допустимое значение поля.
    """
    return float(
        registry.clamp(CharacterControllerComponent, name, float(value), warn=False)
    )


def _checked_character_field(name: str, value: float) -> float:
    """Значение поля персонажа; вне диапазона - ``ValueError`` с именем поля."""
    return float(
        registry.clamp(CharacterControllerComponent, name, float(value), strict=True)
    )


def _vec(value: Any, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    return tuple(float(v) for v in value)


class _PhysicsComponent(Component):
    """База физических компонентов: правка поля помечает тело на пересоздание."""

    __slots__ = ()

    # Публичные поля, правка которых не пересоздаёт тело: адаптер читает их сам.
    _live_fields: frozenset[str] = frozenset()

    def __setattr__(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)
        # ``enabled`` - свойство базы: о нём сообщают ``on_enable``/``on_disable``
        if name[0] != "_" and name != "enabled":
            if name in self._live_fields:
                self._notify_live()
            else:
                self._notify_changed()

    def _notify_changed(self) -> None:
        owner = self.game_object
        scene = None if owner is None else owner.scene
        physics = None if scene is None else scene.physics_if_exists
        if physics is not None:
            physics.mark_dirty(owner)

    def _notify_live(self) -> None:
        owner = self.game_object
        scene = None if owner is None else owner.scene
        physics = None if scene is None else scene.physics_if_exists
        if physics is not None:
            physics.mark_live(owner)

    def _adapter(self):
        scene = self.scene
        if scene is None:
            raise RuntimeError(f"{type(self).__name__} не добавлен в сцену")
        return scene.physics

    def on_enable(self) -> None:
        # Физика создаётся лениво: сцена без физических компонентов её не платит.
        scene = self.scene
        if scene is not None:
            scene.physics.mark_dirty(self.game_object)

    def on_disable(self) -> None:
        self._notify_changed()


@_finish
class RigidBodyComponent(_PhysicsComponent):
    """Твёрдое тело.

    Args:
        body_type: ``static``, ``kinematic`` (двигается трансформом) или
            ``dynamic`` (двигается физикой).
        mass: Масса динамического тела, кг (> 0).
        linear_damping: Линейное затухание, 1/с.
        angular_damping: Угловое затухание, 1/с.
        gravity_scale: Множитель гравитации.
        use_gravity: Действует ли гравитация.
        freeze_rotation: Запретить вращение (инерция бесконечна).
        can_sleep: Разрешить засыпание.
    """

    __slots__ = (
        "angular_damping",
        "body_type",
        "can_sleep",
        "freeze_rotation",
        "gravity_scale",
        "linear_damping",
        "mass",
        "use_gravity",
    )

    __field_specs__ = (
        _pfield("body_type", "dynamic", kind=FieldKind.ENUM, choices=BODY_TYPES),
        _pfield("mass", 1.0, tooltip="Масса динамического тела, кг (> 0)"),
        _pfield("linear_damping", 0.05),
        _pfield("angular_damping", 0.05),
        _pfield("gravity_scale", 1.0),
        _pfield("use_gravity", True),
        _pfield("freeze_rotation", False),
        _pfield("can_sleep", True),
    )

    def __init__(
        self,
        body_type: str = "dynamic",
        mass: float = 1.0,
        linear_damping: float = 0.05,
        angular_damping: float = 0.05,
        gravity_scale: float = 1.0,
        use_gravity: bool = True,
        freeze_rotation: bool = False,
        can_sleep: bool = True,
    ) -> None:
        super().__init__()
        self.body_type = str(body_type)
        self.mass = float(mass)
        self.linear_damping = float(linear_damping)
        self.angular_damping = float(angular_damping)
        self.gravity_scale = float(gravity_scale)
        self.use_gravity = bool(use_gravity)
        self.freeze_rotation = bool(freeze_rotation)
        self.can_sleep = bool(can_sleep)

    # ------------------------------------------------------------ рантайм-API

    @property
    def handle(self) -> int | None:
        """Handle тела в мире физики (``None``, пока тело не создано)."""
        return self._adapter().handle_of(self.game_object)

    @property
    def linear_velocity(self) -> tuple[float, float, float]:
        """Линейная скорость, м/с."""
        return self._adapter().get_velocity(self.game_object)

    @linear_velocity.setter
    def linear_velocity(self, value) -> None:
        self._adapter().set_velocity(self.game_object, linear=value)

    @property
    def angular_velocity(self) -> tuple[float, float, float]:
        """Угловая скорость, рад/с."""
        return self._adapter().get_velocity(self.game_object, angular=True)

    @angular_velocity.setter
    def angular_velocity(self, value) -> None:
        self._adapter().set_velocity(self.game_object, angular=value)

    def apply_impulse(self, impulse, point=None) -> None:
        """Мгновенный импульс (в центр масс или в мировую точку ``point``)."""
        self._adapter().apply_impulse(self.game_object, impulse, point)

    def to_dict(self) -> dict[str, object]:
        """Сериализовать компонент (для сцены)."""
        return registry.serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RigidBodyComponent:
        """Восстановить компонент; отсутствующие поля берутся по умолчанию."""
        return registry.deserialize(cls, data)


@_finish
class ColliderComponent(_PhysicsComponent):
    """Форма столкновений.

    Политика валидации: компонент хранит значения как есть (приводит только
    типы) и ничего не отвергает. Некорректные числа исправляет адаптер при
    создании тела: ``size`` берётся по модулю, размеры не меньше 1e-3,
    ``friction`` не меньше 0, ``restitution`` зажимается в 0..1, нулевой
    кватернион ``rotation_offset`` считается единичным. Ошибки, которые
    исправить нельзя (неизвестная ``shape`` или режим смешивания, ``size``
    короче трёх чисел), дают ``ValueError`` при создании тела: объект
    пропускается с предупреждением, остальная сцена живёт. Сама физика
    (``engine.physics``) ничего не исправляет и бросает ``ValueError``.

    Args:
        shape: ``box``, ``sphere`` или ``capsule`` (ось капсулы — локальный Y).
        size: Полные размеры бокса по осям (до умножения на масштаб объекта).
        radius: Радиус шара/капсулы.
        height: Полная высота капсулы (с полусферами).
        offset: Смещение центра формы от трансформа объекта.
        is_trigger: Триггер — не толкает тела, только шлёт события.
        friction: Трение материала.
        restitution: Упругость материала (0..1).
        layer: Битовая маска слоя коллайдера.
        mask: Слои, с которыми он взаимодействует.
        rotation_offset: Поворот формы относительно тела, кватернион
            ``(x, y, z, w)``. Смещение ``offset`` им не поворачивается.
        friction_combine: Режим смешивания трения пары: ``multiply``,
            ``min``, ``max`` или ``average``.
        restitution_combine: То же для упругости.

    Режим смешивания пары выбирается симметрично, «главного» коллайдера нет:
    побеждает режим, стоящий позже в порядке ``multiply < min < max <
    average``. Пример: ``multiply`` против ``max`` даёт ``max``.
    """

    __slots__ = (
        "friction",
        "friction_combine",
        "height",
        "is_trigger",
        "layer",
        "mask",
        "offset",
        "radius",
        "restitution",
        "restitution_combine",
        "rotation_offset",
        "shape",
        "size",
    )

    __field_specs__ = (
        _pfield("shape", "box", kind=FieldKind.ENUM, choices=SHAPE_TYPES),
        _pfield("size", (1.0, 1.0, 1.0), kind=FieldKind.VEC3),
        _pfield("radius", 0.5),
        _pfield("height", 2.0),
        _pfield("offset", (0.0, 0.0, 0.0), kind=FieldKind.VEC3),
        _pfield("is_trigger", False),
        _pfield("friction", 0.5),
        _pfield("restitution", 0.0),
        _pfield("layer", 1),
        _pfield("mask", _ALL_LAYERS, kind=FieldKind.MASK32),
        _pfield(
            "rotation_offset", _IDENTITY_QUAT, kind=FieldKind.QUAT,
            omit_if_default=True,
        ),
        _pfield(
            "friction_combine", "multiply", kind=FieldKind.ENUM,
            choices=COMBINE_MODES, omit_if_default=True,
        ),
        _pfield(
            "restitution_combine", "max", kind=FieldKind.ENUM,
            choices=COMBINE_MODES, omit_if_default=True,
        ),
    )

    def __init__(
        self,
        shape: str = "box",
        size: tuple[float, float, float] = (1.0, 1.0, 1.0),
        radius: float = 0.5,
        height: float = 2.0,
        offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        is_trigger: bool = False,
        friction: float = 0.5,
        restitution: float = 0.0,
        layer: int = 1,
        mask: int = _ALL_LAYERS,
        rotation_offset: tuple[float, float, float, float] = _IDENTITY_QUAT,
        friction_combine: str = "multiply",
        restitution_combine: str = "max",
    ) -> None:
        super().__init__()
        self.shape = str(shape)
        self.size = _vec(size, (1.0, 1.0, 1.0))
        self.radius = float(radius)
        self.height = float(height)
        self.offset = _vec(offset, (0.0, 0.0, 0.0))
        self.is_trigger = bool(is_trigger)
        self.friction = float(friction)
        self.restitution = float(restitution)
        self.layer = int(layer)
        self.mask = int(mask)
        self.rotation_offset = _vec(rotation_offset, _IDENTITY_QUAT)
        self.friction_combine = str(friction_combine)
        self.restitution_combine = str(restitution_combine)

    def to_dict(self) -> dict[str, object]:
        """Сериализовать компонент.

        ``rotation_offset`` и режимы смешивания со значениями по умолчанию
        не пишутся.
        """
        return registry.serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ColliderComponent:
        """Восстановить компонент; отсутствующие поля и ``null`` дают значения
        по умолчанию."""
        return registry.deserialize(cls, data)


@_finish
class MeshColliderComponent(_PhysicsComponent):
    """Статический коллайдер по геометрии меша (уровень, стены, рельеф).

    Берётся LOD0 cooked-меша ассета (все его части склеиваются), масштаб
    объекта запекается в вершины, BVH строится при загрузке в мир. В общий
    список коллайдеров, широкую фазу и солвер mesh не входит: его видят лучи,
    свипы персонажа (``CharacterController``) и ``overlap``. Динамическое тело
    сквозь mesh проваливается, событий столкновений нет. Для динамики берите
    ``ColliderComponent`` (box/sphere/capsule).

    Args:
        mesh: Ссылка на меш-ассет (``.obj``/``.glb``/``.fbx``).
        is_trigger: Триггер: виден запросам только при ``include_triggers``.
        friction: Трение материала.
        restitution: Упругость материала (0..1).
        layer: Битовая маска слоя.
        mask: Слои, с которыми он взаимодействует.
    """

    __slots__ = ("friction", "is_trigger", "layer", "mask", "mesh", "restitution")

    __field_specs__ = (
        _pfield("mesh", None, kind=FieldKind.ASSET_REF, asset_kind="mesh",
                tooltip="Меш-ассет, по геометрии которого строится коллайдер"),
        _pfield("is_trigger", False),
        _pfield("friction", 0.5),
        _pfield("restitution", 0.0),
        _pfield("layer", 1),
        _pfield("mask", _ALL_LAYERS, kind=FieldKind.MASK32),
    )

    def __init__(
        self,
        mesh: AssetRef | None = None,
        is_trigger: bool = False,
        friction: float = 0.5,
        restitution: float = 0.0,
        layer: int = 1,
        mask: int = _ALL_LAYERS,
    ) -> None:
        super().__init__()
        self.mesh = mesh
        self.is_trigger = bool(is_trigger)
        self.friction = float(friction)
        self.restitution = float(restitution)
        self.layer = int(layer)
        self.mask = int(mask)

    def inheritable_mesh(self) -> AssetRef | None:
        """Ссылка на меш из ``MeshRenderer`` объекта (``None`` - брать нечего).

        Годится только ссылка на ассет (с GUID): встроенный примитив без GUID
        меш-коллайдером не читается.
        """
        from engine.core.components import MeshRenderer

        owner = self.game_object
        if owner is None:
            return None
        for renderer in owner.get_components(MeshRenderer):
            ref = renderer.mesh_ref
            if ref is not None and ref.guid:
                return ref
        return None

    def on_attach(self) -> None:
        """Пустой ``mesh`` при создании берётся из ``MeshRenderer`` объекта.

        Копия делается один раз: дальше коллайдер за рендером не следует.
        """
        if self.mesh is None:
            ref = self.inheritable_mesh()
            if ref is not None:
                self.mesh = ref

    def to_dict(self) -> dict[str, object]:
        """Сериализовать компонент (``mesh`` - ссылка ``{guid, path}``)."""
        return registry.serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MeshColliderComponent:
        """Восстановить компонент; отсутствующие поля дают значения по умолчанию."""
        return registry.deserialize(cls, data)


@_finish
class CharacterControllerComponent(_PhysicsComponent):
    """Капсульный контроллер персонажа (трансформ — центр капсулы).

    Args:
        radius: Радиус капсулы.
        height: Полная высота капсулы.
        step_offset: Максимальная высота ступени.
        slope_limit: Максимальный проходимый уклон, градусы.
        skin_width: Зазор до препятствий.
        layer: Слой двойника персонажа в мире.
        mask: Слои, с которыми персонаж сталкивается.
        move_speed: Скорость ходьбы для ``simple_move()`` без аргумента, м/с (>= 0).
        sprint_multiplier: Множитель скорости при ``sprint(True)`` (>= 1).
        jump_velocity: Скорость прыжка для ``jump()`` без аргумента, м/с (> 0).
        use_gravity: Применять ли гравитацию (иначе вертикальная скорость 0).
        gravity_scale: Множитель гравитации мира (>= 0).
        ground_check_distance: Дальность пробы земли сверх ``skin_width`` (> 0).
        air_control: Доля управляемости в воздухе для ввода с клавиатуры [0, 1].
        snap_to_ground: Прилипать к земле при спуске по склону и ступеням.
        alignment_to_ground: Наклонять трансформ по нормали земли (рыскание
            сохраняется, физическая капсула остаётся вертикальной).

    Поля скорости, гравитации, пробы земли и флаги - «живые»: их правка на
    работающем компоненте не пересоздаёт двойник и не сбрасывает вертикальную
    скорость. Недопустимые значения в конструкторе - ``ValueError``; присвоенные
    позже приводятся к диапазону адаптером (с предупреждением в лог).
    """

    __slots__ = (
        "_air_override",
        "_sprinting",
        "air_control",
        "alignment_to_ground",
        "gravity_scale",
        "ground_check_distance",
        "height",
        "jump_velocity",
        "layer",
        "mask",
        "move_speed",
        "radius",
        "skin_width",
        "slope_limit",
        "snap_to_ground",
        "sprint_multiplier",
        "step_offset",
        "use_gravity",
    )

    __field_specs__ = (
        _pfield("radius", 0.4),
        _pfield("height", 1.8),
        _pfield("step_offset", 0.3),
        _pfield("slope_limit", 45.0),
        _pfield("skin_width", 0.02),
        _pfield("layer", 1),
        _pfield("mask", _ALL_LAYERS, kind=FieldKind.MASK32),
        _pfield("move_speed", 4.5, range=(0.0, math.inf), live=True),
        _pfield("sprint_multiplier", 1.6, range=(1.0, math.inf), live=True),
        _pfield(
            "jump_velocity", 5.0, range=(0.0, math.inf), exclusive_min=True,
            live=True,
        ),
        _pfield("use_gravity", True, live=True),
        _pfield("gravity_scale", 1.0, range=(0.0, math.inf), live=True),
        _pfield(
            "ground_check_distance", 0.03, range=(0.0, math.inf),
            exclusive_min=True, live=True,
        ),
        _pfield("air_control", 0.3, range=(0.0, 1.0), live=True),
        _pfield("snap_to_ground", True, live=True),
        _pfield("alignment_to_ground", False, live=True),
    )

    def on_enable(self) -> None:
        """Размораживает персонажа: двойник остаётся, не пересоздаётся."""
        scene = self.scene
        if scene is not None:
            scene.physics.mark_toggled()

    def on_disable(self) -> None:
        """Замораживает персонажа: двойник остаётся в мире, но не двигается."""
        scene = self.scene
        physics = None if scene is None else scene.physics_if_exists
        if physics is not None:
            physics.mark_toggled()

    def __init__(
        self,
        radius: float = 0.4,
        height: float = 1.8,
        step_offset: float = 0.3,
        slope_limit: float = 45.0,
        skin_width: float = 0.02,
        layer: int = 1,
        mask: int = _ALL_LAYERS,
        move_speed: float = 4.5,
        sprint_multiplier: float = 1.6,
        jump_velocity: float = 5.0,
        use_gravity: bool = True,
        gravity_scale: float = 1.0,
        ground_check_distance: float = 0.03,
        air_control: float = 0.3,
        snap_to_ground: bool = True,
        alignment_to_ground: bool = False,
    ) -> None:
        super().__init__()
        self.radius = float(radius)
        self.height = float(height)
        self.step_offset = float(step_offset)
        self.slope_limit = float(slope_limit)
        self.skin_width = float(skin_width)
        self.layer = int(layer)
        self.mask = int(mask)
        self.move_speed = _checked_character_field("move_speed", move_speed)
        self.sprint_multiplier = _checked_character_field(
            "sprint_multiplier", sprint_multiplier
        )
        self.jump_velocity = _checked_character_field("jump_velocity", jump_velocity)
        self.use_gravity = bool(use_gravity)
        self.gravity_scale = _checked_character_field("gravity_scale", gravity_scale)
        self.ground_check_distance = _checked_character_field(
            "ground_check_distance", ground_check_distance
        )
        self.air_control = _checked_character_field("air_control", air_control)
        self.snap_to_ground = bool(snap_to_ground)
        self.alignment_to_ground = bool(alignment_to_ground)
        # Рантайм-состояние скрипта (не сериализуется, не пересоздаёт двойник).
        self._sprinting = False
        self._air_override: float | None = None

    # ------------------------------------------------------------ рантайм-API

    @property
    def is_grounded(self) -> bool:
        """Стоит ли персонаж на проходимой поверхности."""
        return self._adapter().is_grounded(self.game_object)

    @property
    def is_sprinting(self) -> bool:
        """Включён ли бег (``sprint(True)``)."""
        return self._sprinting

    @property
    def air_control_override(self) -> float | None:
        """Значение ``set_air_control`` или ``None``, если действует поле."""
        return self._air_override

    def move(self, motion) -> tuple[float, float, float]:
        """Двигает на ``motion`` (метры) с коллизиями; возвращает смещение."""
        return self._adapter().move_character(self.game_object, motion)

    def simple_move(self, velocity) -> tuple[float, float, float]:
        """Движение со скоростью (XZ) и гравитацией за один физический шаг.

        Скорость берётся как есть: бег и ``air_control`` не применяются
        (для них - :meth:`move_input`).
        """
        return self._adapter().update_character(self.game_object, velocity)

    def move_input(self, direction) -> tuple[float, float, float]:
        """Движение по направлению ввода со скоростью ``move_speed``.

        Длина XZ-вектора больше 1 сжимается до 1; бег (:meth:`sprint`) умножает
        скорость на ``sprint_multiplier``, в воздухе работает ``air_control``
        (или значение :meth:`set_air_control`). Возвращает смещение.
        """
        return self._adapter().move_input_character(
            self.game_object, direction,
            sprint=self._sprinting, air_control=self._air_override,
        )

    def jump(self, speed: float | None = None) -> bool:
        """Прыжок с вертикальной скоростью ``speed`` (только с земли).

        Без аргумента - ``jump_velocity``. ``speed <= 0`` - ``ValueError``; в
        воздухе (и при ``use_gravity=False``) возвращает ``False``.
        """
        return self._adapter().jump_character(self.game_object, speed)

    def sprint(self, on: bool = True) -> None:
        """Включает/выключает бег для :meth:`move_input` (до следующего вызова)."""
        self._sprinting = bool(on)

    def set_air_control(self, value: float | None) -> None:
        """Переопределяет ``air_control`` (поле не меняется); ``None`` - снять.

        Raises:
            ValueError: Если значение вне ``[0, 1]`` или не число.
        """
        if value is None:
            self._air_override = None
            return
        value = float(value)
        if not 0.0 <= value <= 1.0:  # nan тоже не проходит
            raise ValueError(f"air_control={value!r}: допустимо значение от 0 до 1")
        self._air_override = value

    def to_dict(self) -> dict[str, object]:
        """Сериализовать компонент (для сцены)."""
        return registry.serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CharacterControllerComponent:
        """Восстановить компонент; отсутствующие поля берутся по умолчанию.

        Недопустимые значения новых числовых полей (например, отрицательная
        скорость) приводятся к диапазону с предупреждением в лог, а не обрывают
        загрузку сцены.
        """
        return registry.deserialize(cls, data)


def has_character_rigidbody_conflict(game_object: GameObject) -> bool:
    """Управляют ли движением объекта сразу контроллер персонажа и твёрдое тело.

    Конфликт - это **активные** оба компонента (``Component.is_active``: включён
    сам компонент и объект активен в иерархии). Выключенный компонент не
    конфликтует - так делается «смерть/рэгдолл»: контроллер выключают, и объектом
    управляет только тело. Одно и то же правило использует адаптер физики (объект
    в мир не попадает) и инспектор редактора (подсветка секций).

    Args:
        game_object: Проверяемый объект.

    Returns:
        ``True``, если у объекта есть активные ``CharacterControllerComponent`` и
        ``RigidBodyComponent``.
    """
    return any(
        c.is_active for c in game_object.get_components(CharacterControllerComponent)
    ) and any(c.is_active for c in game_object.get_components(RigidBodyComponent))
