"""Экранные эффекты для скриптов: ``self.screen`` (фасад над ``Scene.screen``).

Example:
    >>> self.screen.fade_to((0, 0, 0), 1.0, 0.5)   # затемнить за 0.5 с
    >>> self.screen.flash((1, 1, 1), 0.8, 0.2)     # белая вспышка
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.scripting.script_component import ScriptComponent

__all__ = ["ScriptScreen"]


class ScriptScreen:
    """Fade и flash всего экрана (рисуются поверх кадра и UI-сцены не трогают)."""

    __slots__ = ("_component",)

    def __init__(self, component: ScriptComponent) -> None:
        self._component = component

    def _fx(self) -> Any:
        return self._component.scene.screen

    def fade_to(self, color: Any, alpha: float, duration: float = 0.5) -> None:
        """Плавно довести затемнение до ``alpha`` (0 — кадр виден, 1 — сплошной цвет).

        Args:
            color: ``Vector3`` или ``(r, g, b)`` в диапазоне 0..1.
            alpha: Целевая непрозрачность.
            duration: Длительность в секундах (реальное время, не зависит от паузы).
        """
        self._fx().fade_to(color, alpha, duration)

    def flash(
        self,
        color: Any = (1.0, 1.0, 1.0),
        intensity: float = 1.0,
        duration: float = 0.25,
    ) -> None:
        """Короткая вспышка, затухающая за ``duration`` секунд."""
        self._fx().flash(color, intensity, duration)

    def clear(self) -> None:
        """Убрать fade и flash немедленно."""
        self._fx().clear()

    @property
    def fade_alpha(self) -> float:
        """Текущая непрозрачность затемнения."""
        return self._fx().fade_alpha
