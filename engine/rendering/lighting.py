"""Освещение MyEngine: источники света, менеджер и упаковка данных для GPU.

Модуль решает четыре задачи:

* **Описание источников** — :class:`DirectionalLight`, :class:`PointLight`,
  :class:`SpotLight` и :class:`AmbientLight`; это чистые данные без OpenGL и Qt,
  поэтому их одинаково используют движок, редактор и тесты;
* **Привязка к сцене** — :class:`LightComponent` вешает свет на
  :class:`~engine.core.game_object.GameObject` и тянет позицию и направление
  из его трансформа (как камера — см. :mod:`engine.rendering.camera`);
* **Сбор и отсечение** — :class:`LightManager` собирает активные источники
  сцены, отсекает их по радиусу влияния и выбирает для каждого объекта не
  больше ``max_lights_per_object`` самых важных;
* **Упаковка под шейдер** — :class:`LightBuffer` раскладывает параметры в
  непрерывные numpy-массивы и загружает их одним вызовом на массив
  (``glUniform3fv``), а не по одной переменной.

Оптимизация под слабое железо (Intel HD и подобное) — сквозная идея модуля:

* свет отсекается **до** загрузки uniform-переменных: объект получает только
  те источники, чья сфера влияния его действительно накрывает;
* набор света для объекта кэшируется по ревизии сцены и габаритам объекта,
  поэтому неподвижная сцена не пересобирает списки каждый кадр;
* результаты упаковки кэшируются по «ключу набора», и повторяющиеся наборы
  (частый случай: одна и та же пара ламп освещает сотни объектов) не
  загружаются в драйвер повторно;
* при ``dynamic=False`` (запечённый свет) загрузка массивов пропускается
  целиком, и шейдер берёт освещение из вершинных цветов или лайтмапы.

Example:
    >>> manager = LightManager(max_lights_per_object=8)
    >>> component = cube.add_component(LightComponent, light=PointLight())
    >>> manager.collect(scene.objects)
    >>> manager.active_lights(component.light, object_bounds)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from engine.core.component import Component
from engine.core.math3d import EPSILON, Quaternion, Vector3

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from collections.abc import Iterable, Sequence

    from engine.core.game_object import GameObject
    from engine.rendering.shader import ShaderProgram

__all__ = [
    "ALL_LAYERS_MASK",
    "AMBIENT_TYPE",
    "DIRECTIONAL_TYPE",
    "DEFAULT_MAX_LIGHTS",
    "DEFAULT_SHADOW_MAP_SIZE",
    "LAYERS_ALL",
    "LAYER_COUNT",
    "LIGHT_TYPES",
    "AmbientLight",
    "DirectionalLight",
    "Light",
    "LightBuffer",
    "LightCacheStats",
    "LightComponent",
    "LightManager",
    "LightSample",
    "LightSet",
    "POINT_TYPE",
    "PointLight",
    "SPOT_TYPE",
    "SpotLight",
    "ambient_light",
    "create_light",
    "directional_light",
    "light_from_dict",
    "point_light",
    "spot_light",
]

_log = logging.getLogger("engine.rendering.lighting")

#: Сколько источников одновременно учитывается для одного объекта.
DEFAULT_MAX_LIGHTS = 8

#: Разрешение карты теней по умолчанию: 512×512 — компромисс для слабых GPU.
DEFAULT_SHADOW_MAP_SIZE = 512

#: Сколько слоёв адресует маска света (бит на слой).
LAYER_COUNT = 32

#: Маска по умолчанию: свет действует на все 32 слоя.
LAYERS_ALL = (1 << LAYER_COUNT) - 1

#: Синоним :data:`LAYERS_ALL` — то же имя, что и у маски отсечения камеры.
ALL_LAYERS_MASK = LAYERS_ALL

#: Идентификаторы типов источника (порядок задаёт приоритет в наборе).
DIRECTIONAL_TYPE = 0
POINT_TYPE = 1
SPOT_TYPE = 2
AMBIENT_TYPE = 3

#: Поддерживаемые типы источников.
LIGHT_TYPES: tuple[str, ...] = ("directional", "point", "spot", "ambient")

#: Имена типов для сериализации: идентификатор → строка.
_TYPE_NAMES: dict[int, str] = {
    DIRECTIONAL_TYPE: "directional",
    POINT_TYPE: "point",
    SPOT_TYPE: "spot",
    AMBIENT_TYPE: "ambient",
}

#: Обратное отображение: строка → идентификатор.
_TYPE_IDS: dict[str, int] = {name: key for key, name in _TYPE_NAMES.items()}


# --------------------------------------------------------------------------- #
# Источники света
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Light:
    """Общая часть источника света.

    Args:
        color: Цвет света (линейный, ``0..1``).
        intensity: Яркость (умножается на цвет).
        enabled: Учитывать ли источник в рендере.
        cast_shadow: Участвует ли источник в карте теней (только солнце).
        layers: Битовая маска слоёв: объект освещается, если его слой в маске.
        name: Имя источника (для UI и отладочных сообщений).

    Attributes:
        type_id: Числовой идентификатор типа (нужен шейдеру и упаковке).
    """

    type_id: int = field(default=DIRECTIONAL_TYPE, init=False)
    color: Vector3 = field(
        default_factory=lambda: Vector3(1.0, 0.98, 0.94)
    )
    intensity: float = 1.0
    enabled: bool = True
    cast_shadow: bool = False
    layers: int = LAYERS_ALL
    name: str = "Light"

    # ------------------------------------------------------------------ #
    # Общие операции
    # ------------------------------------------------------------------ #
    @property
    def type_name(self) -> str:
        """Строковое имя типа источника."""
        return _TYPE_NAMES.get(self.type_id, "light")

    @property
    def kind(self) -> str:
        """Псевдоним :attr:`type_name` (короткое имя для UI и тестов)."""
        return self.type_name

    @property
    def rgb(self) -> Vector3:
        """Цвет, умноженный на яркость и обрезанный до разумных значений."""
        level = max(0.0, float(self.intensity))
        return Vector3(
            max(0.0, self.color.x) * level,
            max(0.0, self.color.y) * level,
            max(0.0, self.color.z) * level,
        )

    @property
    def is_ambient(self) -> bool:
        """``True``, если источник не зависит от геометрии."""
        return self.type_id == AMBIENT_TYPE

    def affects_layer(self, layer: int) -> bool:
        """Проверить слой объекта по маске источника."""
        layer = int(layer)
        if not 0 <= layer < 32:
            return True
        return bool((1 << layer) & (int(self.layers) & LAYERS_ALL))

    def world_direction(self) -> Vector3:
        """Направление, **куда** светит источник (для точечного — ``forward``)."""
        return Vector3(0.0, -1.0, 0.0)

    def world_position(self) -> Vector3:
        """Позиция источника в мире (у направленного — начало координат)."""
        return Vector3.zero()

    def reach(self) -> float:
        """Радиус влияния в мировых единицах (``inf`` для направленного)."""
        return math.inf

    def bucket_key(self) -> tuple:
        """Ключ состояния источника: меняется при любой правке параметров."""
        return (
            self.type_id,
            round(self.color.x, 5),
            round(self.color.y, 5),
            round(self.color.z, 5),
            round(self.intensity, 5),
            bool(self.enabled),
            int(self.layers),
        )

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать источник в словарь."""
        return {
            "type": self.type_name,
            "color": list(self.color.to_tuple()),
            "intensity": self.intensity,
            "enabled": self.enabled,
            "cast_shadow": self.cast_shadow,
            "layers": int(self.layers),
            "name": self.name,
        }

    def copy(self) -> Light:
        """Создать независимую копию источника."""
        return light_from_dict(self.to_dict())

    def __repr__(self) -> str:
        state = "on" if self.enabled else "off"
        return (
            f"<{type(self).__name__} '{self.name}' {state} "
            f"color={self.color} intensity={self.intensity:.2f}>"
        )


@dataclass(slots=True)
class DirectionalLight(Light):
    """Солнце: бесконечно удалённый источник с параллельными лучами.

    Args:
        direction: Направление, **куда** светит источник (нормализуется).
        shadow_distance: До какого расстояния от наблюдателя строятся тени.
        shadow_map_size: Разрешение карты теней (512 — значение для слабых GPU).
        shadow_bias: Смещение по глубине против «акне» теней.
        normal_bias: Смещение по нормали (помогает на наклонных поверхностях).

    Attributes:
        type_id: ``DIRECTIONAL_TYPE``.
    """

    direction: Vector3 = field(
        default_factory=lambda: Vector3(-0.45, -1.0, -0.35)
    )
    shadow_distance: float = 30.0
    shadow_map_size: int = DEFAULT_SHADOW_MAP_SIZE
    shadow_bias: float = 0.0015
    normal_bias: float = 0.02
    ambient: float = 0.22
    #: Подсказка: ``ambient`` — запасной фон, когда в сцене нет AmbientLight.

    def __post_init__(self) -> None:
        self.type_id = DIRECTIONAL_TYPE
        if self.name == "Light":
            self.name = "Directional Light"

    @property
    def unit_direction(self) -> Vector3:
        """Нормализованное направление света."""
        return self.direction.normalized

    @property
    def vector_to_light(self) -> Vector3:
        """Направление от поверхности **к** свету (то, что нужно шейдеру)."""
        return -self.unit_direction

    def world_direction(self) -> Vector3:
        """Направление, куда светит источник."""
        return self.unit_direction

    def reach(self) -> float:
        """Направленный свет влияет на всю сцену."""
        return math.inf

    @property
    def ambient_color(self) -> Vector3:
        """Запасная фоновая подсветка как цвет.

        Используется менеджером света, когда в сцене нет явного
        :class:`AmbientLight`: сцена с одним солнцем сохраняет привычную
        фоновую засветку.
        """
        level = max(0.0, float(self.ambient))
        return Vector3(level, level, level)

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать источник."""
        data = Light.to_dict(self)
        data.update(
            {
                "direction": list(self.unit_direction.to_tuple()),
                "shadow_distance": self.shadow_distance,
                "shadow_map_size": int(self.shadow_map_size),
                "shadow_bias": self.shadow_bias,
                "normal_bias": self.normal_bias,
                "ambient": self.ambient,
            }
        )
        return data


@dataclass(slots=True)
class PointLight(Light):
    """Точечный источник: светит во все стороны с затуханием.

    Args:
        position: Позиция в мире (обновляется трансформом компонента).
        range: Радиус влияния: за его пределами вклад равен нулю, поэтому свет
            можно отсекать по сфере.
        falloff: Показатель затухания внутри радиуса (1 — линейное).

    Attributes:
        type_id: ``POINT_TYPE``.
    """

    position: Vector3 = field(default_factory=Vector3.zero)
    range: float = 8.0
    falloff: float = 2.0

    def __post_init__(self) -> None:
        self.type_id = POINT_TYPE
        if self.name == "Light":
            self.name = "Point Light"
        self.range = max(EPSILON, float(self.range))

    def world_position(self) -> Vector3:
        """Позиция источника в мире."""
        return Vector3(self.position.x, self.position.y, self.position.z)

    def world_direction(self) -> Vector3:
        """У точечного источника направления нет: берём ``-Y``."""
        return Vector3(0.0, -1.0, 0.0)

    def reach(self) -> float:
        """Радиус влияния."""
        return self.range

    def bucket_key(self) -> tuple:
        """Ключ состояния с учётом позиции и радиуса."""
        return (
            *Light.bucket_key(self),
            round(self.position.x, 4),
            round(self.position.y, 4),
            round(self.position.z, 4),
            round(self.range, 4),
            round(self.falloff, 4),
        )

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать источник."""
        data = Light.to_dict(self)
        data.update(
            {
                "position": list(self.position.to_tuple()),
                "range": self.range,
                "falloff": self.falloff,
            }
        )
        return data


@dataclass(slots=True)
class SpotLight(Light):
    """Прожектор: конус света с затуханием по расстоянию и по краю конуса.

    Args:
        position: Позиция в мире.
        direction: Направление, **куда** светит конус.
        range: Радиус влияния (дальность конуса).
        inner_angle: Угол полного света (градусы, половина раствора).
        outer_angle: Угол, на котором свет гаснет (градусы).
        falloff: Показатель затухания по расстоянию.

    Attributes:
        type_id: ``SPOT_TYPE``.
    """

    position: Vector3 = field(default_factory=Vector3.zero)
    direction: Vector3 = field(default_factory=lambda: Vector3(0.0, -1.0, 0.0))
    range: float = 12.0
    inner_angle: float = 18.0
    outer_angle: float = 28.0
    falloff: float = 2.0

    def __post_init__(self) -> None:
        self.type_id = SPOT_TYPE
        if self.name == "Light":
            self.name = "Spot Light"
        self.range = max(EPSILON, float(self.range))
        self.inner_angle = max(0.0, float(self.inner_angle))
        # Внешний угол обязан быть больше внутреннего, иначе конус вырожден.
        self.outer_angle = max(self.inner_angle + 0.1, float(self.outer_angle))
        self.outer_angle = min(self.outer_angle, 89.9)

    @property
    def unit_direction(self) -> Vector3:
        """Нормализованное направление конуса."""
        return self.direction.normalized

    @property
    def inner_cosine(self) -> float:
        """Косинус внутреннего угла (шейдеру нужен именно он)."""
        return math.cos(math.radians(self.inner_angle))

    @property
    def outer_cosine(self) -> float:
        """Косинус внешнего угла."""
        return math.cos(math.radians(self.outer_angle))

    def world_position(self) -> Vector3:
        """Позиция источника в мире."""
        return Vector3(self.position.x, self.position.y, self.position.z)

    def world_direction(self) -> Vector3:
        """Направление конуса."""
        return self.unit_direction

    def reach(self) -> float:
        """Радиус влияния (дальность конуса)."""
        return self.range

    def bucket_key(self) -> tuple:
        """Ключ состояния с учётом позиции, направления и углов."""
        return (
            *Light.bucket_key(self),
            round(self.position.x, 4),
            round(self.position.y, 4),
            round(self.position.z, 4),
            round(self.direction.x, 4),
            round(self.direction.y, 4),
            round(self.direction.z, 4),
            round(self.range, 4),
            round(self.inner_angle, 3),
            round(self.outer_angle, 3),
            round(self.falloff, 4),
        )

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать источник."""
        data = Light.to_dict(self)
        data.update(
            {
                "position": list(self.position.to_tuple()),
                "direction": list(self.unit_direction.to_tuple()),
                "range": self.range,
                "inner_angle": self.inner_angle,
                "outer_angle": self.outer_angle,
                "falloff": self.falloff,
            }
        )
        return data


@dataclass(slots=True)
class AmbientLight(Light):
    """Фоновая подсветка: равномерный свет без направления и затухания.

    В сцене достаточно одного такого источника; если их несколько, вклад
    суммируется, а в наборе света они идут последними.
    """

    def __post_init__(self) -> None:
        self.type_id = AMBIENT_TYPE
        self.intensity = max(0.0, float(self.intensity))
        if self.name == "Light":
            self.name = "Ambient Light"

    def world_direction(self) -> Vector3:
        """У фонового света направления нет."""
        return Vector3.zero()

    def reach(self) -> float:
        """Фоновый свет влияет на всю сцену."""
        return math.inf

    def bucket_key(self) -> tuple:
        """Ключ состояния (позиция и направление не важны)."""
        return Light.bucket_key(self)

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать источник."""
        return Light.to_dict(self)


def create_light(kind: str, **kwargs: Any) -> Light:
    """Создать источник света по имени типа.

    Args:
        kind: ``"directional"``, ``"point"``, ``"spot"`` или ``"ambient"``.
        **kwargs: Параметры конструктора нужного класса.

    Returns:
        Новый источник света.

    Raises:
        ValueError: Если тип неизвестен.
    """
    factories = {
        "directional": DirectionalLight,
        "point": PointLight,
        "spot": SpotLight,
        "ambient": AmbientLight,
    }
    factory = factories.get(str(kind).lower())
    if factory is None:
        raise ValueError(
            f"Неизвестный тип света: {kind!r} (ожидался один из {sorted(factories)})"
        )
    if factory is AmbientLight:
        # У фонового света нет ни направления, ни радиуса — лишние ключи
        # молча выкидываем, чтобы копирование сцены не падало.
        allowed = {"color", "intensity", "enabled", "layers", "name", "cast_shadow"}
        kwargs = {key: value for key, value in kwargs.items() if key in allowed}
    return factory(**kwargs)


def light_from_dict(data: dict[str, Any]) -> Light:
    """Восстановить источник света из словаря.

    Args:
        data: Словарь, полученный из :meth:`Light.to_dict`.

    Returns:
        Новый источник света.

    Raises:
        ValueError: Если тип источника неизвестен.
    """
    common: dict[str, Any] = {
        "color": Vector3.from_iterable(data.get("color") or (1.0, 1.0, 1.0)),
        "intensity": float(data.get("intensity", 1.0)),
        "enabled": bool(data.get("enabled", True)),
        "cast_shadow": bool(data.get("cast_shadow", False)),
        "layers": int(data.get("layers", LAYERS_ALL)),
        "name": str(data.get("name", "Light")),
    }
    kind = str(data.get("type", "directional"))
    if kind == "directional":
        return DirectionalLight(
            direction=Vector3.from_iterable(
                data.get("direction") or (-0.45, -1.0, -0.35)
            ),
            shadow_distance=float(data.get("shadow_distance", 30.0)),
            shadow_map_size=int(data.get("shadow_map_size", DEFAULT_SHADOW_MAP_SIZE)),
            shadow_bias=float(data.get("shadow_bias", 0.0015)),
            normal_bias=float(data.get("normal_bias", 0.02)),
            ambient=float(data.get("ambient", 0.22)),
            **common,
        )
    if kind == "point":
        return PointLight(
            position=Vector3.from_iterable(data.get("position") or (0.0, 0.0, 0.0)),
            range=float(data.get("range", 8.0)),
            falloff=float(data.get("falloff", 2.0)),
            **common,
        )
    if kind == "spot":
        return SpotLight(
            position=Vector3.from_iterable(data.get("position") or (0.0, 0.0, 0.0)),
            direction=Vector3.from_iterable(
                data.get("direction") or (0.0, -1.0, 0.0)
            ),
            range=float(data.get("range", 12.0)),
            inner_angle=float(data.get("inner_angle", 18.0)),
            outer_angle=float(data.get("outer_angle", 28.0)),
            falloff=float(data.get("falloff", 2.0)),
            **common,
        )
    if kind == "ambient":
        return AmbientLight(**common)
    raise ValueError(
        f"Неизвестный тип света: {kind!r} (ожидался один из {sorted(LIGHT_TYPES)})"
    )


def directional_light(**kwargs: Any) -> DirectionalLight:
    """Солнце со значениями по умолчанию."""
    return DirectionalLight(**kwargs)


def point_light(**kwargs: Any) -> PointLight:
    """Точечный источник со значениями по умолчанию."""
    return PointLight(**kwargs)


def spot_light(**kwargs: Any) -> SpotLight:
    """Прожектор со значениями по умолчанию."""
    return SpotLight(**kwargs)


def ambient_light(intensity: float = 0.2, **kwargs: Any) -> AmbientLight:
    """Фоновая подсветка со значениями по умолчанию."""
    return AmbientLight(intensity=intensity, **kwargs)


# --------------------------------------------------------------------------- #
# Компонент света
# --------------------------------------------------------------------------- #
class LightComponent(Component):
    """Компонент источника света на игровом объекте.

    Позиция и направление берутся из мирового трансформа владельца, поэтому
    источник движется вместе с объектом (и по иерархии). Дополнительные
    смещение и доворот позволяют отвязать визуальную модель от направления
    света — например, повесить фонарь на модель и направить его вниз.

    Args:
        light: Готовый источник; ``None`` — создать точечный по умолчанию.
        local_offset: Смещение источника в локальных осях объекта.
        local_rotation: Доворот направления относительно объекта.
        follow_owner: Синхронизировать ли источник с трансформом владельца.

    Attributes:
        light: Источник света.
    """

    __slots__ = ("follow_owner", "light", "local_offset", "local_rotation")

    def __init__(
        self,
        light: Light | None = None,
        *,
        local_offset: Vector3 | None = None,
        local_rotation: Quaternion | None = None,
        follow_owner: bool = True,
    ) -> None:
        super().__init__()
        self.light: Light = light or PointLight()
        self.local_offset = local_offset or Vector3.zero()
        self.local_rotation = local_rotation or Quaternion.identity()
        self.follow_owner = bool(follow_owner)

    # ------------------------------------------------------------------ #
    # Идентичность
    # ------------------------------------------------------------------ #
    @property
    def display_name(self) -> str:
        """Имя источника для инспектора и списков."""
        owner = self.game_object
        if owner is not None:
            return owner.name
        return self.light.name

    @property
    def kind(self) -> str:
        """Строковый тип источника."""
        return self.light.type_name

    def world_position(self) -> Vector3:
        """Мировая позиция источника (с учётом локального смещения)."""
        return self.light.world_position()

    def world_direction(self) -> Vector3:
        """Мировое направление источника."""
        return self.light.world_direction()

    # ------------------------------------------------------------------ #
    # Синхронизация с трансформом
    # ------------------------------------------------------------------ #
    def sync_to_light(self) -> Light:
        """Перенести мировой трансформ объекта в параметры источника.

        Returns:
            Обновлённый источник.
        """
        if not self.follow_owner or not self.is_attached:
            return self.light

        light = self.light
        if light.is_ambient:
            # Фоновому свету позиция и направление не нужны.
            return light

        transform = self.transform
        origin, rotation = _world_origin_rotation(transform)
        offset = rotation.rotate_vector(self.local_offset)
        world_position = origin + offset
        # Направление — это локальная ось +Z объекта, повёрнутая мировым
        # вращением и доворотом компонента. Кватернионы перемножаются ДО
        # поворота вектора: rotate_vector возвращает вектор, а не кватернион.
        orientation = (rotation * self.local_rotation).normalized
        world_direction = orientation.rotate_vector(Vector3.forward())

        if isinstance(light, PointLight):
            light.position = world_position
        elif isinstance(light, SpotLight):
            light.position = world_position
            light.direction = world_direction
        elif isinstance(light, DirectionalLight):
            light.direction = world_direction
        return light

    def on_attach(self) -> None:
        """Сразу подтянуть трансформ владельца."""
        self.sync_to_light()

    def on_update(self, dt: float) -> None:
        """Обновлять источник каждый кадр: он часть трансформа объекта."""
        self.sync_to_light()

    def on_late_update(self, dt: float) -> None:
        """Позднее обновление: после логики объект мог сдвинуться."""
        self.sync_to_light()

    # ------------------------------------------------------------------ #
    # Сериализация (для сохранения сцены)
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, object]:
        """Сериализовать компонент вместе с источником света."""
        return {
            "light": self.light.to_dict(),
            "local_offset": list(self.local_offset.to_tuple()),
            "local_rotation": list(self.local_rotation.to_tuple()),
            "follow_owner": self.follow_owner,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> LightComponent:
        """Восстановить компонент из словаря.

        Args:
            data: Словарь, полученный из :meth:`to_dict`.

        Returns:
            Новый компонент света.
        """
        payload = data.get("light")
        light = (
            light_from_dict(payload)
            if isinstance(payload, dict)
            else DirectionalLight()
        )
        offset = data.get("local_offset") or (0.0, 0.0, 0.0)
        rotation = data.get("local_rotation") or (0.0, 0.0, 0.0, 1.0)
        return cls(
            light,
            local_offset=Vector3.from_iterable(offset),  # type: ignore[arg-type]
            local_rotation=Quaternion(*[float(value) for value in rotation]),  # type: ignore[arg-type]
            follow_owner=bool(data.get("follow_owner", True)),
        )

    def __repr__(self) -> str:
        owner = self.game_object
        target = owner.name if owner is not None else "<свободен>"
        return f"<LightComponent '{target}' {self.light!r}>"


def _world_origin_rotation(transform: Any) -> tuple[Vector3, Quaternion]:
    """Взять мировые позицию и вращение трансформа.

    Args:
        transform: :class:`~engine.core.transform.Transform`.

    Returns:
        Пару «мировая позиция, мировое вращение».

    Note:
        Обращение к ``world_matrix`` уже синхронизирует кэш иерархии, а
        разложение на позицию и кватернион делается теми же функциями, что и в
        камере, — отдельного пересчёта углов не требуется.
    """
    from engine.core.math3d import extract_translation, matrix_to_quaternion

    matrix = np.asarray(transform.world_matrix, dtype=float).reshape(4, 4)
    return extract_translation(matrix), matrix_to_quaternion(matrix)


# --------------------------------------------------------------------------- #
# Упаковка данных для шейдера
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class LightSample:
    """Один источник в упакованном виде (для отладки и тестов)."""

    type_id: int
    position: tuple[float, float, float]
    direction: tuple[float, float, float]
    color: tuple[float, float, float]
    range: float
    inner_cosine: float
    outer_cosine: float
    falloff: float

    @classmethod
    def from_light(cls, light: Light) -> LightSample:
        """Упаковать источник света."""
        position = light.world_position()
        direction = light.world_direction()
        level = light.rgb
        inner = 1.0
        outer = 1.0
        falloff = 1.0
        distance = 0.0
        if isinstance(light, SpotLight):
            inner = light.inner_cosine
            outer = light.outer_cosine
            falloff = light.falloff
            distance = light.range
        elif isinstance(light, PointLight):
            falloff = light.falloff
            distance = light.range
        elif isinstance(light, AmbientLight):
            # Фоновый свет едет в тех же массивах: важна только яркость.
            position = Vector3.zero()
            direction = Vector3.zero()
            distance = 0.0
        return cls(
            type_id=light.type_id,
            position=(position.x, position.y, position.z),
            direction=(direction.x, direction.y, direction.z),
            color=(level.x, level.y, level.z),
            range=distance,
            inner_cosine=inner,
            outer_cosine=outer,
            falloff=falloff,
        )

    @staticmethod
    def type_name(type_id: int) -> str:
        """Имя типа источника по идентификатору."""
        return _TYPE_NAMES.get(int(type_id), "?")


class LightBuffer:
    """Непрерывные массивы параметров света для загрузки в шейдер.

    Данные лежат «структурой массивов»: отдельный массив на каждый атрибут.
    Так их можно отдать драйверу одним вызовом (``glUniform3fv`` на весь
    массив), а не по одной uniform-переменной на источник — на слабых
    драйверах это заметно дешевле.

    Args:
        capacity: Сколько источников помещается в буфер.

    Attributes:
        position: Массив ``(N, 3)`` позиций.
        direction: Массив ``(N, 3)`` направлений (куда светит источник).
        color: Массив ``(N, 3)`` цвета с учётом яркости.
        params: Массив ``(N, 4)``: ``(радиус, cos внутреннего, cos внешнего,
            затухание)``.
    """

    __slots__ = (
        "capacity",
        "color",
        "count",
        "direction",
        "params",
        "position",
        "type_ids",
    )

    def __init__(self, capacity: int = DEFAULT_MAX_LIGHTS) -> None:
        self.capacity = max(1, int(capacity))
        self.count = 0
        self.position = np.zeros((self.capacity, 3), dtype=np.float32)
        self.direction = np.zeros((self.capacity, 3), dtype=np.float32)
        self.color = np.zeros((self.capacity, 3), dtype=np.float32)
        self.params = np.zeros((self.capacity, 4), dtype=np.float32)
        self.type_ids = np.zeros(self.capacity, dtype=np.int32)

    # ------------------------------------------------------------------ #
    # Заполнение
    # ------------------------------------------------------------------ #
    def fill(self, lights: Sequence[Light]) -> int:
        """Разложить источники по массивам.

        Args:
            lights: Источники в порядке приоритета (важные — первыми).

        Returns:
            Сколько источников реально поместилось.
        """
        count = min(len(lights), self.capacity)
        for index in range(count):
            sample = LightSample.from_light(lights[index])
            self.position[index] = sample.position
            self.direction[index] = sample.direction
            self.color[index] = sample.color
            self.params[index] = (
                sample.range,
                sample.inner_cosine,
                sample.outer_cosine,
                sample.falloff,
            )
            self.type_ids[index] = sample.type_id
        self.count = count
        return count

    def key(self) -> tuple:
        """Ключ содержимого буфера: годится для кэша загрузки в драйвер."""
        if self.count == 0:
            return (0,)
        # Массивы небольшие (до 8 элементов), поэтому хеш по байтам дёшев и
        # точен: он меняется ровно тогда, когда меняются параметры света.
        return (
            self.count,
            self.type_ids[: self.count].tobytes(),
            self.position[: self.count].tobytes(),
            self.direction[: self.count].tobytes(),
            self.color[: self.count].tobytes(),
            self.params[: self.count].tobytes(),
        )

    def describe(self) -> str:
        """Краткое описание содержимого (для логов и статистики)."""
        names = [
            LightSample.type_name(int(value))
            for value in self.type_ids[: self.count]
        ]
        return f"{self.count}/{self.capacity}: {', '.join(names)}"

    def __len__(self) -> int:
        return self.count

    def __repr__(self) -> str:
        return f"<LightBuffer {self.count}/{self.capacity}>"


# --------------------------------------------------------------------------- #
# Менеджер света
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class LightSet:
    """Набор света для одного объекта: источники и ключ их состояния.

    Набор готовит :meth:`LightManager.prepare`; рендерер передаёт его в
    :meth:`LightManager.bind_set`. Ключ позволяет пропустить повторную
    загрузку массивов, когда один и тот же набор освещает сотни объектов.

    Attributes:
        lights: Источники, влияющие на объект (не длиннее потолка).
        key: Ключ содержимого буфера.
        count: Число источников в наборе.
    """

    lights: tuple[Light, ...]
    key: tuple
    count: int

    def __bool__(self) -> bool:
        """``True``, если в наборе есть источники."""
        return self.count > 0

    def __len__(self) -> int:
        return self.count

    def describe(self) -> str:
        """Краткое описание набора для отладки."""
        names = [light.type_name for light in self.lights]
        return f"{self.count}: {', '.join(names)}" if names else "пусто"


@dataclass(slots=True)
class LightCacheStats:
    """Счётчики работы менеджера света за кадр."""

    collected: int = 0
    culled: int = 0
    selections: int = 0
    selection_hits: int = 0
    uploads: int = 0
    upload_hits: int = 0

    def reset(self) -> None:
        """Обнулить счётчики."""
        self.collected = 0
        self.culled = 0
        self.selections = 0
        self.selection_hits = 0
        self.uploads = 0
        self.upload_hits = 0

    def describe(self) -> str:
        """Краткое описание для строки статуса."""
        return (
            f"lights {self.collected} (culled {self.culled}), "
            f"sets {self.selection_hits}/{self.selections}, "
            f"uploads {self.upload_hits}/{self.uploads}"
        )


class LightManager:
    """Сбор источников сцены, отсечение по радиусу и кэширование наборов.

    Args:
        max_lights_per_object: Сколько источников учитывается для одного
            объекта (настраиваемый потолок; по умолчанию 8).
        dynamic: Динамическое освещение включено. При ``False`` параметры
            света не загружаются в шейдер, и освещение берётся из вершинных
            цветов или лайтмапы (запечённый свет).
        cast_shadows: Разрешены ли карты теней (нужны ещё и у источника).
        shadow_map_size: Разрешение карты теней.
    """

    __slots__ = (
        "_ambient",
        "_ambient_version",
        "_buffer",
        "_cache",
        "_directional",
        "_lights",
        "_max_lights",
        "_selection_cache",
        "_collect_signature",
        "_stats",
        "_version",
        "cast_shadows",
        "dynamic",
        "fallback_ambient",
        "max_lights_per_object",
        "shadow_map_size",
    )

    def __init__(
        self,
        *,
        max_lights_per_object: int = DEFAULT_MAX_LIGHTS,
        dynamic: bool = True,
        cast_shadows: bool = True,
        shadow_map_size: int = DEFAULT_SHADOW_MAP_SIZE,
        fallback_ambient: Vector3 | None = None,
    ) -> None:
        self.max_lights_per_object = max(1, int(max_lights_per_object))
        self.dynamic = bool(dynamic)
        self.cast_shadows = bool(cast_shadows)
        self.shadow_map_size = max(64, int(shadow_map_size))
        #: Фоновая подсветка, когда в сцене нет явного AmbientLight.
        self.fallback_ambient = fallback_ambient or Vector3(0.22, 0.22, 0.22)
        self._lights: list[Light] = []
        self._directional: DirectionalLight | None = None
        self._ambient: AmbientLight | None = None
        self._buffer = LightBuffer(self.max_lights_per_object)
        # Ключ набора, который СЕЙЧАС загружен в программу (id программы → ключ).
        self._cache: dict[int, tuple] = {}
        self._selection_cache: dict[tuple, tuple[int, ...]] = {}
        self._stats = LightCacheStats()
        self._version = 0
        self._collect_signature: tuple | None = None
        self._ambient_version = 0

    # ------------------------------------------------------------------ #
    # Состояние
    # ------------------------------------------------------------------ #
    @property
    def stats(self) -> LightCacheStats:
        """Счётчики работы менеджера."""
        return self._stats

    @property
    def version(self) -> int:
        """Номер ревизии списка источников (растёт при сборе)."""
        return self._version

    @property
    def lights(self) -> tuple[Light, ...]:
        """Собранные источники в порядке приоритета."""
        return tuple(self._lights)

    @property
    def buffer(self) -> LightBuffer:
        """Буфер параметров последнего загруженного набора."""
        return self._buffer

    @property
    def max_lights(self) -> int:
        """Текущий потолок источников на объект."""
        return self._buffer.capacity

    @max_lights.setter
    def max_lights(self, value: int) -> None:
        capacity = max(1, int(value))
        self.max_lights_per_object = capacity
        if self._buffer.capacity != capacity:
            self._buffer = LightBuffer(capacity)
            self.clear_cache()

    @property
    def directional(self) -> DirectionalLight | None:
        """Главное (первое) солнце сцены или ``None``."""
        return self._directional

    @property
    def ambient(self) -> AmbientLight | None:
        """Первый фоновый источник сцены или ``None``."""
        return self._ambient

    @property
    def ambient_color(self) -> Vector3:
        """Суммарный фоновый свет (для uniform-переменной шейдера).

        Если в сцене нет явного :class:`AmbientLight`, берётся запасное
        значение :attr:`fallback_ambient` — так сцена с одним солнцем
        сохраняет привычную фоновую подсветку. Явный :class:`AmbientLight` с
        нулевой яркостью даёт честный ноль.
        """
        red = green = blue = 0.0
        has_ambient = False
        for light in self._lights:
            if isinstance(light, AmbientLight):
                has_ambient = True
                level = light.rgb
                red += level.x
                green += level.y
                blue += level.z
        if not has_ambient:
            # Явного фонового источника нет: берём засветку солнца, если она
            # задана, иначе общее запасное значение.
            if self._directional is not None:
                return self._directional.ambient_color
            return self.fallback_ambient
        return Vector3(red, green, blue)

    def use_fallback_ambient(self, light: DirectionalLight) -> None:
        """Взять запасную фоновую подсветку из направленного источника.

        Нужно для совместимости: сцена может задать фоновую засветку прямо у
        солнца (:attr:`DirectionalLight.ambient`), не создавая отдельного
        :class:`AmbientLight`.

        Args:
            light: Источник, чей :attr:`ambient` берётся как запасной.
        """
        self.fallback_ambient = light.ambient_color

    @property
    def shadow_caster(self) -> DirectionalLight | None:
        """Источник, для которого строится карта теней (или ``None``)."""
        if not self.cast_shadows or not self.dynamic:
            return None
        for light in self._lights:
            if isinstance(light, DirectionalLight) and light.cast_shadow:
                return light
        return None

    def describe(self) -> str:
        """Строка для панели статистики."""
        return (
            f"свет {len(self._lights)} (макс {self.max_lights}/объект), "
            f"{self._stats.describe()}"
        )

    # ------------------------------------------------------------------ #
    # Сбор источников
    # ------------------------------------------------------------------ #
    def collect(self, objects: Iterable[GameObject]) -> int:
        """Собрать включённые источники со всех объектов.

        Порядок важен для приоритета: солнце (и фоновый свет) идут первыми,
        потому что влияют на всю сцену, а точечные и прожекторы — по убыванию
        силы влияния, чтобы при переполнении набора отбросить наименее важные.

        Args:
            objects: Игровые объекты сцены.

        Returns:
            Сколько источников собрано.
        """
        lights: list[Light] = []
        for game_object in objects:
            if game_object.is_destroyed or not game_object.active_in_hierarchy:
                continue
            component = game_object.get_component(LightComponent)
            if component is None or not component.enabled:
                continue
            light = component.light
            if not light.enabled:
                continue
            # Трансформ мог измениться после обновления компонентов: редактор не
            # крутит игровой цикл, поэтому синхронизируемся перед сбором.
            component.sync_to_light()
            lights.append(light)

        lights.sort(key=_light_priority)
        # Статичный свет не должен сбрасывать кэши выбора каждый кадр: при
        # неизменном наборе и параметрах версию не трогаем.
        signature = tuple(
            (
                id(item),
                item.bucket_key(),
                item.world_position().to_tuple(),
                item.world_direction().to_tuple(),
                item.reach(),
                bool(item.cast_shadow),
            )
            for item in lights
        )
        if signature == self._collect_signature:
            self._stats.collected = len(lights)
            return len(lights)
        self._collect_signature = signature
        self._lights = lights
        self._directional = next(
            (item for item in lights if isinstance(item, DirectionalLight)), None
        )
        self._ambient = next(
            (item for item in lights if isinstance(item, AmbientLight)), None
        )
        self._version += 1
        self._stats.collected = len(lights)
        self.clear_cache()
        return len(lights)

    def set_lights(self, lights: Sequence[Light]) -> int:
        """Задать источники напрямую (плеер, тесты, процедурные сцены).

        Args:
            lights: Источники света.

        Returns:
            Сколько источников принято.
        """
        ordered = sorted(lights, key=_light_priority)
        self._lights = list(ordered)
        self._collect_signature = None
        self._directional = next(
            (item for item in ordered if isinstance(item, DirectionalLight)), None
        )
        self._ambient = next(
            (item for item in ordered if isinstance(item, AmbientLight)), None
        )
        self._version += 1
        self._stats.collected = len(ordered)
        self.clear_cache()
        return len(ordered)

    def add(self, light: Light) -> Light:
        """Добавить источник к собранному набору."""
        self._lights.append(light)
        self._collect_signature = None
        self._lights.sort(key=_light_priority)
        if isinstance(light, DirectionalLight) and self._directional is None:
            self._directional = light
        if isinstance(light, AmbientLight) and self._ambient is None:
            self._ambient = light
        self._version += 1
        self.clear_cache()
        return light

    def clear(self) -> None:
        """Забыть все источники (например, при закрытии сцены)."""
        self._lights.clear()
        self._collect_signature = None
        self._directional = None
        self._ambient = None
        self._version += 1
        self.clear_cache()

    def clear_cache(self) -> None:
        """Сбросить кэши наборов и загрузок."""
        self._cache.clear()
        self._selection_cache.clear()

    # ------------------------------------------------------------------ #
    # Отсечение и выбор источников
    # ------------------------------------------------------------------ #
    def influences(
        self,
        light: Light,
        center: Vector3,
        radius: float,
        *,
        layer: int | None = None,
    ) -> bool:
        """Проверить, влияет ли источник на сферу объекта.

        Направленный свет и фоновая подсветка влияют всегда; точечный и
        прожектор отсекаются по сфере радиуса влияния (для прожектора
        дополнительно проверяется конус — но грубо, по сфере: точный тест на
        конус стоил бы больше, чем экономил на слабом GPU).

        Args:
            light: Источник света.
            center: Центр габаритов объекта в мире.
            radius: Радиус габаритов объекта.
            layer: Слой объекта (учитывается маска источника).

        Returns:
            ``True``, если источник может осветить объект.
        """
        if layer is not None and not light.affects_layer(layer):
            return False
        if light.is_ambient or isinstance(light, DirectionalLight):
            return True
        reach = light.reach() + max(0.0, radius)
        if isinstance(light, SpotLight):
            # Конус расширяется с расстоянием: на дальнем конце радиус равен
            # range * tan(outer_angle). Учитываем это, иначе свет «пропадал» бы
            # на краю пятна.
            spread = light.range * math.tan(math.radians(light.outer_angle))
            reach = light.range + spread + max(0.0, radius)
        return (
            light.world_position() - center
        ).length_squared <= reach * reach

    def select(
        self,
        center: Vector3,
        radius: float,
        *,
        layer: int | None = None,
        dynamic: bool | None = None,
    ) -> tuple[Light, ...]:
        """Выбрать источники, влияющие на объект, с кэшированием.

        Кэш ключуется по позиции и радиусу габаритов: набор источников меняется
        только при смене ревизии списка или геометрии, поэтому неподвижная сцена
        не пересобирает списки каждый кадр.

        Args:
            center: Центр габаритов объекта.
            radius: Радиус габаритов объекта.
            layer: Слой объекта.
            dynamic: Переопределить режим динамического освещения.

        Returns:
            Кортеж источников в порядке приоритета (не длиннее потолка).
        """
        return self.prepare(center, radius, layer=layer, dynamic=dynamic).lights

    def prepare(
        self,
        center: Vector3,
        radius: float,
        *,
        layer: int | None = None,
        dynamic: bool | None = None,
    ) -> LightSet:
        """Подготовить набор света для объекта и упаковать его в буфер.

        Метод делает две вещи: выбирает влияющие источники (с кэшем) и
        раскладывает их по массивам буфера. Вместе с ключом набора это
        позволяет рендереру решить, нужно ли вообще трогать драйвер.

        Args:
            center: Центр габаритов объекта.
            radius: Радиус габаритов объекта.
            layer: Слой объекта.
            dynamic: Переопределить режим динамического освещения.

        Returns:
            Готовый набор света.
        """
        if dynamic is None:
            dynamic = self.dynamic
        self._stats.selections += 1
        key = (
            self._version,
            round(center.x, 3),
            round(center.y, 3),
            round(center.z, 3),
            round(radius, 3),
            -1 if layer is None else int(layer),
            bool(dynamic),
        )
        cached = self._selection_cache.get(key)
        if cached is not None:
            self._stats.selection_hits += 1
            indices, set_key = cached
            lights = self._cached_lights(indices)
            return LightSet(lights=lights, key=set_key, count=len(lights))

        selected: list[int] = []
        rejected = 0
        for index, light in enumerate(self._lights):
            if len(selected) >= self.max_lights_per_object:
                rejected += 1
                continue
            if self.influences(light, center, radius, layer=layer):
                selected.append(index)
            else:
                rejected += 1
        if not dynamic:
            # Запечённый свет: динамические источники не передаются вовсе,
            # остаётся только фоновая составляющая.
            selected = [
                index
                for index in selected
                if isinstance(self._lights[index], AmbientLight)
            ]
        self._stats.culled += rejected

        lights = self._cached_lights(tuple(selected))
        self._buffer.fill(lights)
        # Ключ набора включает содержимое буфера: если два объекта освещены
        # одинаково, второй не будет загружать массивы повторно.
        set_key = (self._buffer.capacity, self._buffer.key())
        if len(self._selection_cache) > 4096:
            # Записи живут кадр; чистка нужна лишь для долгого вращения камеры
            # на одной точке, когда ключи перестают повторяться.
            self._selection_cache.clear()
        self._selection_cache[key] = (tuple(selected), set_key)
        return LightSet(lights=lights, key=set_key, count=len(lights))

    def _cached_lights(self, indices: tuple[int, ...]) -> tuple[Light, ...]:
        """Собрать кортеж источников по индексам (список не менялся)."""
        return tuple(self._lights[index] for index in indices)

    def bind_set(
        self,
        program: ShaderProgram,
        light_set: LightSet,
        *,
        force: bool = False,
    ) -> int:
        """Загрузить готовый набор света в программу.

        Args:
            program: Шейдерная программа.
            light_set: Набор из :meth:`prepare`.
            force: Загрузить, даже если этот набор уже загружался.

        Returns:
            Сколько источников загружено.
        """
        if not self.dynamic:
            self._stats.uploads += 1
            program.set_uniform_array("u_light_count", self._counts(0))
            self._apply_ambient(program)
            return 0
        self._stats.uploads += 1
        if not force and self._cache.get(id(program)) == light_set.key:
            self._stats.upload_hits += 1
            return light_set.count
        # Буфер мог быть перезаполнен другим объектом (набор взят из кэша
        # выбора): перед загрузкой кладём в него именно этот набор.
        self._buffer.fill(light_set.lights)
        count = light_set.count
        program.set_uniform_array("u_light_count", self._counts(count))
        program.set_uniform_array("u_light_position", self._buffer.position)
        program.set_uniform_array("u_light_direction", self._buffer.direction)
        program.set_uniform_array("u_light_color", self._buffer.color)
        program.set_uniform_array("u_light_params", self._buffer.params)
        program.set_uniform_array("u_light_type", self._buffer.type_ids)
        program.set_uniform("u_max_lights", int(self._buffer.capacity))
        self._apply_ambient(program)
        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[id(program)] = light_set.key
        return count

    def _apply_ambient(self, program: ShaderProgram) -> None:
        """Передать суммарную фоновую подсветку."""
        ambient = self.ambient_color
        program.set_uniform("u_ambient", Vector3(ambient.x, ambient.y, ambient.z))

    def bind(
        self,
        program: ShaderProgram,
        lights: Sequence[Light] | None = None,
        *,
        force: bool = False,
    ) -> int:
        """Загрузить свет в программу без готового набора (простой путь).

        Удобно для тестов, инструментов и сцен без отсечения. В горячем цикле
        рендерер использует :meth:`prepare` + :meth:`bind_set`: они позволяют
        пропустить работу, когда набор света у объекта не изменился.

        Args:
            program: Шейдерная программа.
            lights: Источники; ``None`` — все собранные.
            force: Загрузить, даже если набор уже загружался.

        Returns:
            Сколько источников загружено (``0`` при запечённом свете).
        """
        if not self.dynamic:
            self._stats.uploads += 1
            program.set_uniform_array("u_light_count", self._counts(0))
            self._apply_ambient(program)
            return 0
        source = self._lights if lights is None else list(lights)
        self._buffer.fill(source)
        light_set = LightSet(
            lights=tuple(source[: self._buffer.count]),
            key=(self._buffer.capacity, self._buffer.key()),
            count=self._buffer.count,
        )
        return self.bind_set(program, light_set, force=force)

    def _counts(self, count: int) -> np.ndarray:
        """Счётчик активных источников как массив из одного int."""
        return np.array([count], dtype=np.int32)

    # ------------------------------------------------------------------ #
    # Запекание
    # ------------------------------------------------------------------ #
    def bake_to_vertex_colors(
        self, mesh: Any, *, ambient_only: bool = True
    ) -> np.ndarray:
        """Посчитать простой запечённый свет для вершин меша.

        Освещение считается по нормалям вершин и выдаётся как цвета
        ``(N, 4)`` — их можно положить в вершинный буфер, чтобы не тратить
        кадр на динамический свет (типичный приём для слабых устройств).

        Args:
            mesh: :class:`~engine.core.resources.asset.MeshAsset` или
                ``GpuMesh`` с атрибутами ``positions`` и ``normals``.
            ambient_only: Учитывать только фоновую подсветку (максимально
                дешёвый вариант) или все собранные источники.

        Returns:
            Массив ``(N, 4)`` float32: RGB-освещение и альфа ``1``.

        Raises:
            ValueError: Если у меша нет нормалей.
        """
        positions, normals = _mesh_arrays(mesh)
        if normals is None:
            raise ValueError("Для запекания света нужны нормали вершин")
        light = np.zeros_like(positions, dtype=np.float32)
        ambient = self.ambient_color
        light += np.array((ambient.x, ambient.y, ambient.z), dtype=np.float32)
        if not ambient_only:
            for source in self._lights:
                if isinstance(source, AmbientLight):
                    continue
                light += _bake_light(source, positions, normals)
        colors = np.ones((positions.shape[0], 4), dtype=np.float32)
        colors[:, :3] = np.clip(light, 0.0, 1.0)
        return colors

    def __repr__(self) -> str:
        return (
            f"<LightManager lights={len(self._lights)} "
            f"max={self.max_lights_per_object} dynamic={self.dynamic}>"
        )


def _light_priority(light: Light) -> tuple[int, int, float]:
    """Ключ сортировки источников: сначала важные для всей сцены.

    Порядок: фоновый свет и солнце, затем остальные по убыванию силы. Так при
    переполнении набора отбрасываются наименее яркие местные источники, а
    глобальные остаются при любом количестве объектов.
    """
    if isinstance(light, DirectionalLight):
        return (0, 0, 0.0)
    if isinstance(light, AmbientLight):
        return (1, 0, 0.0)
    level = light.rgb
    strength = level.x + level.y + level.z
    return (2, 0, -strength)


def _mesh_arrays(mesh: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """Достать позиции и нормали из ассета меша или GPU-меша."""
    positions = getattr(mesh, "positions", None)
    normals = getattr(mesh, "normals", None)
    if positions is not None:
        return np.asarray(positions, dtype=np.float32), (
            None if normals is None else np.asarray(normals, dtype=np.float32)
        )
    # GPU-меш держит упакованные вершины; для запекания достаточно того, что
    # вызывающий код передал ассет.
    raise ValueError("Запекание света требует меш-ассет с массивами вершин")


def _bake_light(
    light: Light, positions: np.ndarray, normals: np.ndarray
) -> np.ndarray:
    """Посчитать вклад одного источника в вершины (векторизованно).

    Расчёт идёт массивами numpy без питон-цикла по вершинам: запекание света
    прогоняется один раз, но на больших мешах разница заметна.
    """
    level = light.rgb
    radiance = np.array((level.x, level.y, level.z), dtype=np.float32)
    if isinstance(light, DirectionalLight):
        direction = light.vector_to_light
        to_light = np.array(
            (direction.x, direction.y, direction.z), dtype=np.float32
        )
        lambert = np.clip(normals @ to_light, 0.0, None)
        return radiance * lambert[:, None]

    origin = light.world_position()
    to_light = (
        np.array((origin.x, origin.y, origin.z), dtype=np.float32) - positions
    )
    distances = np.linalg.norm(to_light, axis=1, keepdims=True)
    distances[distances < EPSILON] = EPSILON
    directions = to_light / distances
    lambert = np.clip(np.sum(normals * directions, axis=1), 0.0, None)

    reach = max(light.range, EPSILON)
    ratio = np.clip(distances[:, 0] / reach, 0.0, 1.0)
    attenuation = np.clip(1.0 - ratio, 0.0, 1.0) ** max(light.falloff, EPSILON)

    if isinstance(light, SpotLight):
        direction = light.unit_direction
        cone_axis = np.array(
            (direction.x, direction.y, direction.z), dtype=np.float32
        )
        cosine = directions @ cone_axis
        denominator = max(light.inner_cosine - light.outer_cosine, 1e-4)
        cone = np.clip((cosine - light.outer_cosine) / denominator, 0.0, 1.0)
        attenuation = attenuation * cone

    return radiance * (lambert * attenuation)[:, None]


#: Обратная совместимость: раньше источники света описывал один класс.
AmbientLightComponent = LightComponent
