"""Лёгкие анимации интерфейса: прозрачность и геометрия, до 180 мс.

Анимируются только ``windowOpacity`` и ``maximumWidth``/``maximumHeight``:
никаких графических эффектов (тени, размытие) — они дороги на слабых iGPU.
В offscreen-режиме (тесты) и при ``MYENGINE_REDUCE_MOTION=1`` анимации
отключены: состояние применяется сразу, без таймеров.
"""

from __future__ import annotations

import os

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QVariantAnimation
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QWidget

__all__ = [
    "DURATION_MS",
    "animate_height",
    "animate_width",
    "fade_in",
    "motion_enabled",
    "variant",
]

#: Длительность стандартной анимации, мс.
DURATION_MS = 160


def motion_enabled() -> bool:
    """Включены ли анимации (выключены в offscreen и по переменной окружения)."""
    if os.environ.get("MYENGINE_REDUCE_MOTION") == "1":
        return False
    return QGuiApplication.platformName() not in {"offscreen", "minimal"}


def fade_in(widget: QWidget, duration: int = DURATION_MS) -> QPropertyAnimation | None:
    """Плавно показать окно верхнего уровня (``windowOpacity`` 0 → 1).

    Args:
        widget: Окно (диалог, лаунчер).
        duration: Длительность, мс.

    Returns:
        Анимация (уже запущена) или ``None``, если анимации выключены.
    """
    if not motion_enabled():
        return None
    widget.setWindowOpacity(0.0)
    animation = QPropertyAnimation(widget, b"windowOpacity", widget)
    animation.setDuration(duration)
    animation.setStartValue(0.0)
    animation.setEndValue(1.0)
    animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    animation.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
    return animation


def _animate_extent(
    widget: QWidget, prop: bytes, start: int, end: int, duration: int
) -> QPropertyAnimation | None:
    """Анимировать ``maximumWidth``/``maximumHeight`` виджета от start до end."""
    if not motion_enabled():
        return None
    animation = QPropertyAnimation(widget, prop, widget)
    animation.setDuration(duration)
    animation.setStartValue(start)
    animation.setEndValue(end)
    animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    animation.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
    return animation


def animate_width(
    widget: QWidget, start: int, end: int, duration: int = DURATION_MS
) -> QPropertyAnimation | None:
    """Плавно изменить максимальную ширину виджета (сворачивание панели)."""
    return _animate_extent(widget, b"maximumWidth", start, end, duration)


def animate_height(
    widget: QWidget, start: int, end: int, duration: int = DURATION_MS
) -> QPropertyAnimation | None:
    """Плавно изменить максимальную высоту виджета (сворачивание секции)."""
    return _animate_extent(widget, b"maximumHeight", start, end, duration)


def variant(
    start: float, end: float, apply, duration: int = DURATION_MS
) -> QVariantAnimation | None:
    """Анимировать число и вызывать ``apply(value)`` на каждом шаге."""
    if not motion_enabled():
        apply(end)
        return None
    animation = QVariantAnimation()
    animation.setDuration(duration)
    animation.setStartValue(start)
    animation.setEndValue(end)
    animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    animation.valueChanged.connect(apply)
    animation.start(QVariantAnimation.DeletionPolicy.DeleteWhenStopped)
    return animation
