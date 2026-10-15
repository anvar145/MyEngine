"""Базовый класс компонента игрового объекта.

Компонент — единица поведения и данных, которую можно добавить объекту
несколько раз (несколько компонентов одного типа допускаются и хранятся в
порядке добавления). Логика компонента вызывается сценой: компонент никогда
не «тикает» сам, что делает порядок обновления детерминированным.

Example:
    >>> class Spinner(Component):
    ...     def __init__(self, speed: float = 90.0) -> None:
    ...         super().__init__()
    ...         self.speed = speed
    ...
    ...     def on_update(self, dt: float) -> None:
    ...         self.transform.rotate(Vector3.up(), self.speed * dt)
"""

from __future__ import annotations

import logging
import weakref
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.game_object import GameObject
    from engine.core.scene import Scene
    from engine.core.transform import Transform

__all__ = ["Component"]

_log = logging.getLogger("engine.core.component")


class Component:
    """База компонента: владелец, активность и хуки жизненного цикла.

    Хуки вызываются движком в таком порядке:

    * :meth:`on_attach` / :meth:`on_detach` — добавление и удаление компонента;
    * :meth:`on_enable` / :meth:`on_disable` — включение/выключение компонента
      или его игрового объекта;
    * :meth:`on_start` — один раз перед первым обновлением активного компонента;
    * :meth:`on_fixed_update` — фиксированный шаг (физика);
    * :meth:`on_update` / :meth:`on_late_update` — переменный шаг логики;
    * :meth:`on_destroy` — перед окончательным удалением.

    Наследники могут объявлять ``__slots__`` со своими полями, но обязаны
    либо вызвать ``super().__init__()``, либо сохранить внутренние атрибуты
    базового класса.
    """

    __slots__ = ("_enabled", "_game_object_ref", "_started", "__weakref__")

    def __init__(self) -> None:
        self._game_object_ref: weakref.ref[GameObject] | None = None
        self._enabled = True
        self._started = False

    # ------------------------------------------------------------------ #
    # Владелец
    # ------------------------------------------------------------------ #
    @property
    def game_object(self) -> GameObject | None:
        """Игровой объект-владелец или ``None``, если компонент не добавлен."""
        if self._game_object_ref is None:
            return None
        return self._game_object_ref()

    @property
    def transform(self) -> Transform:
        """Трансформ владельца.

        Raises:
            RuntimeError: Если компонент ещё не добавлен объекту.
        """
        owner = self.game_object
        if owner is None:
            raise RuntimeError(
                f"{type(self).__name__} не привязан к игровому объекту"
            )
        return owner.transform

    @property
    def scene(self) -> Scene | None:
        """Сцена владельца или ``None`` для объекта вне сцены."""
        owner = self.game_object
        return None if owner is None else owner.scene

    @property
    def is_attached(self) -> bool:
        """``True``, если компонент добавлен живому игровому объекту."""
        owner = self.game_object
        return owner is not None and not owner.is_destroyed

    # ------------------------------------------------------------------ #
    # Активность
    # ------------------------------------------------------------------ #
    @property
    def enabled(self) -> bool:
        """Включён ли компонент (независимо от активности объекта)."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        value = bool(value)
        if value == self._enabled:
            return
        self._enabled = value
        scene = self.scene
        if scene is not None:
            scene._on_component_enabled_changed(self)  # noqa: SLF001 - сцена владеет активностью

    @property
    def is_active(self) -> bool:
        """``True``, если компонент включён и его объект активен в иерархии."""
        owner = self.game_object
        if owner is None or not self._enabled:
            return False
        return owner.active_in_hierarchy

    # ------------------------------------------------------------------ #
    # Хуки жизненного цикла (переопределяются наследниками)
    # ------------------------------------------------------------------ #
    def on_attach(self) -> None:
        """Вызывается сразу после добавления компонента объекту."""

    def on_detach(self) -> None:
        """Вызывается перед удалением компонента с объекта."""

    def on_enable(self) -> None:
        """Вызывается при активации компонента."""

    def on_disable(self) -> None:
        """Вызывается при деактивации компонента."""

    def on_start(self) -> None:
        """Вызывается один раз перед первым обновлением активного компонента."""

    def on_fixed_update(self, dt: float) -> None:
        """Обновление с фиксированным шагом (обычно физика).

        Args:
            dt: Фиксированный шаг в секундах.
        """

    def on_update(self, dt: float) -> None:
        """Обновление с переменным шагом (игровая логика).

        Args:
            dt: Время с прошлого кадра в секундах.
        """

    def on_late_update(self, dt: float) -> None:
        """Позднее обновление: камеры, IK, зависимые от логики вычисления."""

    def on_destroy(self) -> None:
        """Вызывается перед уничтожением компонента."""

    def on_spawn(self) -> None:
        """Объект взят из пула (вызывается после включения объекта)."""

    def on_despawn(self) -> None:
        """Объект возвращается в пул (вызывается до выключения объекта)."""

    # ------------------------------------------------------------------ #
    # Служебный интерфейс для GameObject и Scene
    # ------------------------------------------------------------------ #
    def _attach(self, game_object: GameObject) -> None:
        """Привязать компонент к объекту (вызывает GameObject)."""
        self._game_object_ref = weakref.ref(game_object)
        self._enabled = True
        self._started = False
        self.on_attach()

    def _detach(self) -> None:
        """Отвязать компонент от объекта (вызывает GameObject)."""
        self.on_detach()
        self._game_object_ref = None
        self._started = False

    def _mark_started(self) -> None:
        """Отметить, что ``on_start`` уже вызван."""
        self._started = True

    @property
    def _is_started(self) -> bool:
        """Был ли уже вызван :meth:`on_start`."""
        return self._started

    def __repr__(self) -> str:
        owner = self.game_object
        target = owner.name if owner is not None else "<свободен>"
        state = "on" if self._enabled else "off"
        return f"<{type(self).__name__} объекта '{target}' {state}>"
