"""Базовый класс системы: пакетная обработка данных сцены.

Система отличается от компонента тем, что не принадлежит одному объекту, а
обрабатывает множество объектов за раз (поиск, физика, анимация, ИИ). Порядок
вызова систем задаётся их :attr:`System.priority`: чем больше значение, тем
раньше система обновляется в своей фазе.

Example:
    >>> class MoveSystem(System):
    ...     priority = 100
    ...
    ...     def on_update(self, scene, dt):
    ...         for obj in scene.query(MoveComponent):
    ...             obj.transform.translate(Vector3(0, 0, dt))
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.game_object import GameObject
    from engine.core.scene import Scene

__all__ = ["System"]


class System:
    """База системы сцены.

    Наследники переопределяют только нужные хуки. Система может быть
    выключена (:attr:`enabled`), оставаясь зарегистрированной в сцене.
    """

    __slots__ = ("_scene_ref", "enabled")

    #: Приоритет внутри фазы: большие значения обновляются раньше.
    priority: int = 0

    def __init__(self) -> None:
        self.enabled = True
        self._scene_ref: weakref.ref[Scene] | None = None

    # ------------------------------------------------------------------ #
    # Регистрация
    # ------------------------------------------------------------------ #
    @property
    def scene(self) -> Scene | None:
        """Сцена, которой принадлежит система, или ``None``."""
        if self._scene_ref is None:
            return None
        return self._scene_ref()

    @property
    def is_attached(self) -> bool:
        """``True``, если система добавлена в сцену."""
        return self._scene_ref is not None and self._scene_ref() is not None

    def query(self, *component_types: type) -> list[GameObject]:
        """Найти объекты сцены с указанными компонентами.

        Raises:
            RuntimeError: Если система ещё не добавлена в сцену.
        """
        scene = self.scene
        if scene is None:
            raise RuntimeError(f"{type(self).__name__} не добавлена в сцену")
        return scene.query(*component_types)

    # ------------------------------------------------------------------ #
    # Хуки жизненного цикла
    # ------------------------------------------------------------------ #
    def on_added(self, scene: Scene) -> None:
        """Вызывается после добавления системы в сцену."""

    def on_removed(self, scene: Scene) -> None:
        """Вызывается после удаления системы из сцены."""

    def on_fixed_update(self, scene: Scene, dt: float) -> None:
        """Обновление с фиксированным шагом.

        Args:
            scene: Сцена, которой принадлежит система.
            dt: Фиксированный шаг в секундах.
        """

    def on_update(self, scene: Scene, dt: float) -> None:
        """Обновление с переменным шагом."""

    def on_late_update(self, scene: Scene, dt: float) -> None:
        """Позднее обновление (после логики кадра)."""

    def __repr__(self) -> str:
        state = "on" if self.enabled else "off"
        return f"<{type(self).__name__} priority={self.priority} {state}>"
