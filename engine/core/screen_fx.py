"""Экранные эффекты поверх кадра: затемнение (fade) и вспышка (flash).

Модуль не зависит от Qt и OpenGL: это состояние и таймеры, которые двигает
хост (плеер) раз в кадр через :meth:`ScreenFx.update`, а рендерер читает
итоговый цвет и непрозрачность. Скрипты видят его как ``self.screen``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ScreenFx"]

Color3 = tuple[float, float, float]


def _color(value: Any) -> Color3:
    """Привести ``Vector3``/``Color``/кортеж к ``(r, g, b)`` в диапазоне 0..1."""
    try:
        red, green, blue = (float(value[i]) for i in range(3))
    except (TypeError, IndexError, KeyError):
        red, green, blue = (
            float(getattr(value, "x", getattr(value, "r", 0.0))),
            float(getattr(value, "y", getattr(value, "g", 0.0))),
            float(getattr(value, "z", getattr(value, "b", 0.0))),
        )
    return (
        min(1.0, max(0.0, red)),
        min(1.0, max(0.0, green)),
        min(1.0, max(0.0, blue)),
    )


class ScreenFx:
    """Затемнение и вспышка экрана с таймерами.

    Fade держит цвет и целевую непрозрачность: ``fade_to(цвет, 1.0, 0.5)``
    плавно затемняет, ``fade_to(цвет, 0.0, 0.5)`` возвращает кадр. Flash — разовая
    вспышка, затухающая за ``duration``; рисуется поверх fade.
    """

    __slots__ = (
        "_fade_color",
        "_fade_alpha",
        "_fade_from",
        "_fade_to",
        "_fade_time",
        "_fade_duration",
        "_flash_color",
        "_flash_peak",
        "_flash_time",
        "_flash_duration",
    )

    def __init__(self) -> None:
        self._fade_color: Color3 = (0.0, 0.0, 0.0)
        self._fade_alpha = 0.0
        self._fade_from = 0.0
        self._fade_to = 0.0
        self._fade_time = 0.0
        self._fade_duration = 0.0
        self._flash_color: Color3 = (1.0, 1.0, 1.0)
        self._flash_peak = 0.0
        self._flash_time = 0.0
        self._flash_duration = 0.0

    # ------------------------------------------------------------------ #
    # Управление
    # ------------------------------------------------------------------ #
    def fade_to(self, color: Any, alpha: float, duration: float = 0.5) -> None:
        """Плавно привести затемнение к ``alpha`` заданным цветом.

        Args:
            color: ``Vector3``/``Color``/``(r, g, b)`` в диапазоне 0..1.
            alpha: Целевая непрозрачность 0..1 (1 — экран залит цветом).
            duration: Время перехода в секундах; ``<= 0`` — мгновенно.
        """
        self._fade_color = _color(color)
        target = min(1.0, max(0.0, float(alpha)))
        if duration <= 0.0:
            self._fade_alpha = self._fade_from = self._fade_to = target
            self._fade_time = self._fade_duration = 0.0
            return
        self._fade_from = self._fade_alpha
        self._fade_to = target
        self._fade_time = 0.0
        self._fade_duration = float(duration)

    def flash(
        self,
        color: Any = (1.0, 1.0, 1.0),
        intensity: float = 1.0,
        duration: float = 0.25,
    ) -> None:
        """Запустить вспышку, затухающую за ``duration`` секунд.

        Args:
            color: Цвет вспышки.
            intensity: Начальная непрозрачность 0..1.
            duration: Время затухания; ``<= 0`` — вспышки нет.
        """
        self._flash_color = _color(color)
        self._flash_peak = min(1.0, max(0.0, float(intensity)))
        self._flash_time = 0.0
        self._flash_duration = max(0.0, float(duration))
        if self._flash_duration <= 0.0:
            self._flash_peak = 0.0

    def clear(self) -> None:
        """Сбросить fade и flash (кадр без эффектов)."""
        self.fade_to((0.0, 0.0, 0.0), 0.0, 0.0)
        self._flash_peak = 0.0
        self._flash_duration = 0.0

    def update(self, dt: float) -> None:
        """Продвинуть таймеры на ``dt`` секунд реального времени."""
        dt = max(0.0, float(dt))
        if self._fade_duration > 0.0:
            self._fade_time += dt
            ratio = min(1.0, self._fade_time / self._fade_duration)
            span = self._fade_to - self._fade_from
            self._fade_alpha = self._fade_from + span * ratio
            if ratio >= 1.0:
                self._fade_alpha = self._fade_to
                self._fade_duration = 0.0
        if self._flash_peak > 0.0:
            self._flash_time += dt
            if self._flash_time >= self._flash_duration:
                self._flash_peak = 0.0

    # ------------------------------------------------------------------ #
    # Состояние
    # ------------------------------------------------------------------ #
    @property
    def fade_alpha(self) -> float:
        """Текущая непрозрачность затемнения 0..1."""
        return self._fade_alpha

    @property
    def fade_color(self) -> Color3:
        """Цвет затемнения."""
        return self._fade_color

    @property
    def flash_alpha(self) -> float:
        """Текущая непрозрачность вспышки (квадратичное затухание)."""
        if self._flash_peak <= 0.0 or self._flash_duration <= 0.0:
            return 0.0
        remaining = 1.0 - min(1.0, self._flash_time / self._flash_duration)
        return self._flash_peak * remaining * remaining

    @property
    def flash_color(self) -> Color3:
        """Цвет вспышки."""
        return self._flash_color

    @property
    def is_fading(self) -> bool:
        """``True``, пока идёт переход fade."""
        return self._fade_duration > 0.0

    @property
    def active(self) -> bool:
        """``True``, если что-то видно на экране или анимируется."""
        return self._fade_alpha > 0.0 or self.flash_alpha > 0.0 or self.is_fading
