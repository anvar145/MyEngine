"""Отладочная отрисовка (C1): ``debug.line``, ``debug.sphere`` и оверлеи.

Скрипт рисует через модуль, без доступа к рендеру::

    from engine import debug

    def on_update(self, dt):
        debug.line(self.transform.position, target, (1, 0, 0))
        debug.sphere(target, 0.3)
        debug.text(target, "цель")

Всё попадает в общий буфер :data:`draw`, а ``DebugPass`` рисует его поверх
кадра, когда включены оверлеи (F3 в плеере, «Вид → Оверлеи отладки» в
редакторе). Пока оверлеи выключены, вызовы копятся только на один кадр и
ничего не рисуют.

Модуль не зависит от GL и Qt: тесты и headless-логика работают без контекста.
"""

from __future__ import annotations

from typing import Any

from engine.debug.draw import DEFAULT_CHANNEL, DebugDraw, DebugText
from engine.debug.overlays import BUILTIN_OVERLAYS, Overlay, OverlayRegistry

__all__ = [
    "BUILTIN_OVERLAYS",
    "DEFAULT_CHANNEL",
    "DebugDraw",
    "DebugText",
    "Overlay",
    "OverlayRegistry",
    "arrow",
    "box",
    "circle",
    "cone",
    "draw",
    "line",
    "overlays",
    "path",
    "sphere",
    "text",
]

#: Общий буфер процесса (плеер и редактор рисуют из него).
draw = DebugDraw()
#: Общий реестр оверлеев процесса.
overlays = OverlayRegistry()


def line(start: Any, end: Any, color=(1.0, 1.0, 1.0), **options: Any) -> None:
    """Отрезок (см. :meth:`DebugDraw.line`)."""
    draw.line(start, end, color, **options)


def arrow(start: Any, end: Any, color=(1.0, 0.85, 0.2), **options: Any) -> None:
    """Стрелка (см. :meth:`DebugDraw.arrow`)."""
    draw.arrow(start, end, color, **options)


def box(center: Any, half_extents: Any, color=(0.3, 1.0, 0.4), **options: Any) -> None:
    """Каркас коробки (см. :meth:`DebugDraw.box`)."""
    draw.box(center, half_extents, color, **options)


def sphere(center: Any, radius: float, color=(0.3, 0.8, 1.0), **options: Any) -> None:
    """Каркас сферы (см. :meth:`DebugDraw.sphere`)."""
    draw.sphere(center, radius, color, **options)


def circle(center: Any, radius: float, normal=(0.0, 1.0, 0.0), color=(0.3, 0.8, 1.0),
           **options: Any) -> None:
    """Окружность (см. :meth:`DebugDraw.circle`)."""
    draw.circle(center, radius, normal, color, **options)


def cone(apex: Any, direction: Any, angle_deg: float, distance: float,
         color=(1.0, 0.4, 0.3), **options: Any) -> None:
    """Конус зрения (см. :meth:`DebugDraw.cone`)."""
    draw.cone(apex, direction, angle_deg, distance, color, **options)


def path(points, color=(0.2, 0.8, 1.0), **options: Any) -> None:
    """Ломаная (см. :meth:`DebugDraw.path`)."""
    draw.path(points, color, **options)


def text(position: Any, value: str, color=(1.0, 1.0, 1.0), **options: Any) -> None:
    """Подпись в мире (см. :meth:`DebugDraw.text`)."""
    draw.text(position, value, color, **options)
