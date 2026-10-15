"""Физика для скриптов: ``self.physics`` (простые типы на входе и выходе).

Фасад над :class:`~engine.core.physics_adapter.ScenePhysicsAdapter`. Векторы
принимаются как ``Vector3`` или кортеж/список из трёх чисел; результаты —
``dict``/кортежи/списки объектов, без внутренних типов физики.

Example:
    >>> hit = self.physics.raycast(self.transform.world_position, (0, -1, 0), 10)
    >>> if hit: self.log(hit["object"].name, hit["distance"])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from engine.core.physics_components import CharacterControllerComponent

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.game_object import GameObject
    from engine.scripting.script_component import ScriptComponent

__all__ = ["ScriptPhysics", "as_tuple3"]

_ALL_LAYERS = 0xFFFFFFFF


def as_tuple3(value: Any) -> tuple[float, float, float]:
    """Привести ``Vector3``/последовательность к кортежу из трёх float."""
    if hasattr(value, "to_tuple"):
        value = value.to_tuple()
    x, y, z = value
    return (float(x), float(y), float(z))


class ScriptPhysics:
    """Физический API скрипта (``self.physics``).

    Методы, действующие «на себя» (``is_grounded``, ``move_character``,
    ``apply_impulse``), работают с объектом скрипта; у ``apply_impulse`` можно
    передать другой объект через ``target``.
    """

    __slots__ = ("_component",)

    def __init__(self, component: ScriptComponent) -> None:
        self._component = component

    # ------------------------------------------------------------------ #
    def _adapter(self):
        scene = self._component.scene
        if scene is None:
            raise RuntimeError("Скрипт не добавлен в сцену")
        return scene.physics

    def _owner(self) -> GameObject:
        owner = self._component.game_object
        if owner is None:
            raise RuntimeError("Скрипт не привязан к объекту")
        return owner

    # ------------------------------------------------------------------ запросы
    def raycast(
        self,
        origin: Any,
        direction: Any,
        max_distance: float = 1000.0,
        layer_mask: int = _ALL_LAYERS,
        include_triggers: bool = False,
    ) -> dict[str, Any] | None:
        """Луч в мир. ``dict(object, point, normal, distance)`` или ``None``."""
        return self._adapter().raycast(
            as_tuple3(origin), as_tuple3(direction), float(max_distance),
            int(layer_mask), bool(include_triggers),
        )

    def overlap_sphere(
        self,
        center: Any,
        radius: float,
        layer_mask: int = _ALL_LAYERS,
        include_triggers: bool = False,
    ) -> list[GameObject]:
        """Объекты, чьи коллайдеры пересекают шар."""
        return self._adapter().overlap_sphere(
            as_tuple3(center), float(radius), int(layer_mask), bool(include_triggers)
        )

    # ------------------------------------------------------------------ мир
    def set_gravity(self, *gravity: Any) -> None:
        """Задаёт гравитацию: ``set_gravity(0, -9.81, 0)`` или ``set_gravity(vec)``."""
        value = gravity[0] if len(gravity) == 1 else gravity
        self._adapter().set_gravity(as_tuple3(value))

    def get_gravity(self) -> tuple[float, float, float]:
        """Текущая гравитация."""
        return as_tuple3(self._adapter().world.gravity)

    # ------------------------------------------------------------------ тела
    def apply_impulse(
        self, impulse: Any, point: Any = None, target: GameObject | None = None
    ) -> None:
        """Импульс телу (по умолчанию — телу своего объекта)."""
        self._adapter().apply_impulse(
            target or self._owner(), as_tuple3(impulse),
            None if point is None else as_tuple3(point),
        )

    def set_velocity(self, velocity: Any, target: GameObject | None = None) -> None:
        """Задаёт линейную скорость тела."""
        self._adapter().set_velocity(target or self._owner(), as_tuple3(velocity))

    def get_velocity(
        self, target: GameObject | None = None
    ) -> tuple[float, float, float]:
        """Линейная скорость тела."""
        return self._adapter().get_velocity(target or self._owner())

    # ------------------------------------------------------------------ персонаж
    def is_grounded(self) -> bool:
        """Стоит ли персонаж этого объекта на земле."""
        return self._adapter().is_grounded(self._owner())

    def move_character(self, motion: Any) -> tuple[float, float, float]:
        """Сдвиг персонажа на ``motion`` с коллизиями; возвращает фактический."""
        return self._adapter().move_character(self._owner(), as_tuple3(motion))

    def _character_component(self) -> CharacterControllerComponent:
        owner = self._owner()
        components = owner.get_components(CharacterControllerComponent)
        for comp in components:
            if comp.is_active:
                return comp
        if components:
            # выключенный (замороженный) персонаж: методы движения - no-op адаптера
            return components[0]
        raise RuntimeError(
            f"У объекта '{owner.name}' нет CharacterControllerComponent"
        )

    def simple_move(self, velocity: Any = None) -> tuple[float, float, float]:
        """Шаг персонажа с гравитацией; зовите из ``on_fixed_update``.

        Аргумент ``velocity`` трактуется по-разному:

        * **явный** - *скорость* (XZ) в м/с, как и раньше: ``move_speed``, бег и
          ``air_control`` не применяются;
        * **без аргумента** - *направление* из клавиш WASD (W = -Z, D = +X в мировых
          осях; диагональ сжимается до длины 1). Скорость берётся из поля
          ``move_speed``, умножается на ``sprint_multiplier`` после
          ``sprint(True)``; в воздухе действует ``air_control``.

        Returns:
            Фактическое смещение за шаг.
        """
        if velocity is not None:
            return self._adapter().update_character(self._owner(), as_tuple3(velocity))
        keys = self._component.input
        dx = float(keys.is_key_down("d")) - float(keys.is_key_down("a"))
        dz = float(keys.is_key_down("s")) - float(keys.is_key_down("w"))
        return self._character_component().move_input((dx, 0.0, dz))

    def jump(self, speed: float | None = None) -> bool:
        """Прыжок персонажа с начальной скоростью ``speed`` (только с земли).

        Без аргумента - ``jump_velocity`` компонента. Двойной прыжок в воздухе
        запрещён (вернёт False), как и прыжок при ``use_gravity=False``.
        ``speed <= 0`` - ошибка.
        """
        return self._adapter().jump_character(
            self._owner(), None if speed is None else float(speed)
        )

    def sprint(self, on: bool = True) -> None:
        """Включает или выключает бег для ``simple_move()`` без аргумента.

        Состояние хранится до следующего вызова: ``sprint(False)`` возвращает
        обычную скорость. Явный ``simple_move(velocity)`` бег не применяет.
        """
        self._character_component().sprint(on)

    def set_air_control(self, value: float | None) -> None:
        """Переопределяет ``air_control`` на время работы скрипта (поле цело).

        ``None`` возвращает значение поля компонента. Значение вне ``[0, 1]`` -
        ``ValueError``.
        """
        self._character_component().set_air_control(value)
