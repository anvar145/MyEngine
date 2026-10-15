"""Секция «Небо» для инспектора.

Небо — свойство сцены (:class:`~engine.rendering.sky.SkySettings` в
``SceneDocument.sky``). Значения применяются сразу; если запущен Play Mode,
изменение дополнительно отправляется плееру командой ``set_setting``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from editor.widgets import ColorButton, FormSection
from engine.core.math3d import Vector3

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from editor.context import EditorContext

__all__ = ["SKY_COLORS", "SkySection"]

#: Цветовые параметры неба: имя поля → ключ перевода.
SKY_COLORS: tuple[tuple[str, str], ...] = (
    ("zenith_color", "sky.zenith"),
    ("horizon_color", "sky.horizon"),
    ("ground_color", "sky.ground"),
    ("sun_color", "sky.sun_color"),
)


class SkySection(QWidget):
    """Параметры процедурного неба активной сцены.

    Args:
        context: Контекст редактора.
        parent: Родительский виджет.

    Signals:
        changed: Параметр неба изменён (нужно перерисовать кадр).
    """

    changed = Signal()

    def __init__(self, context: EditorContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.context = context
        self._sky = context.active_document.sky
        self.color_buttons: dict[str, ColorButton] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        section = FormSection(self.tr("sky.title"), self)
        self.enabled_box = section.add_check(
            self.tr("sky.enabled"),
            self._sky.enabled,
            lambda value: self._apply("enabled", value),
        )
        for name, caption in SKY_COLORS:
            button = ColorButton(Vector3(*getattr(self._sky, name)), section)
            button.set_label(self.tr(caption))
            button.color_changed.connect(
                lambda color, field=name: self._apply(
                    field, (color.x, color.y, color.z)
                )
            )
            section.form.addRow(self.tr(caption), button)
            self.color_buttons[name] = button
        self.spins = {
            "sun_size": section.add_spin(
                self.tr("sky.sun_size"),
                self._sky.sun_size,
                lambda value: self._apply("sun_size", value),
                minimum=0.1,
                maximum=30.0,
            ),
            "sun_glow": section.add_spin(
                self.tr("sky.sun_glow"),
                self._sky.sun_glow,
                lambda value: self._apply("sun_glow", value),
                minimum=1.0,
                maximum=200.0,
                step=1.0,
            ),
            "exposure": section.add_spin(
                self.tr("sky.exposure"),
                self._sky.exposure,
                lambda value: self._apply("exposure", value),
                minimum=0.05,
                maximum=10.0,
            ),
            "intensity": section.add_spin(
                self.tr("sky.intensity"),
                self._sky.intensity,
                lambda value: self._apply("intensity", value),
                minimum=0.0,
                maximum=10.0,
            ),
        }
        layout.addWidget(section)

    def tr(self, key: str, **kwargs: object) -> str:  # noqa: A003 - привычное имя Qt
        """Перевести строку через переводчик редактора."""
        return self.context.tr(key, **kwargs)

    def _apply(self, name: str, value: Any) -> None:
        """Записать параметр в небо сцены и в запущенный плеер."""
        self.context.set_document_setting("sky", name, value)
        play_mode = getattr(self.context, "play_mode", None)
        if play_mode is not None and play_mode.is_active:
            play_mode.send_setting(f"sky.{name}", value)
        self.changed.emit()
