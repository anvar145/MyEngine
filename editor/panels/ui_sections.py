"""Секция UI-компонента в инспекторе (панель, картинка, кнопка, текст).

Одна секция на любой тип: общие поля :class:`~engine.core.ui_components.UIElement`
(якорь, позиция, размер, слой, непрозрачность) плюс поля конкретного виджета.
Для ``UIText`` свойства текста правит вложенная :class:`TextSection`.
Значения применяются сразу; об изменении сообщает сигнал ``changed``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLineEdit, QWidget

from editor.widgets import ColorButton, FormSection
from engine.core.math3d import Vector3
from engine.core.ui_components import (
    UI_ANCHORS,
    UIButton,
    UIElement,
    UIImage,
    UIPanel,
)

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from editor.context import EditorContext

__all__ = ["UISection", "UI_TYPES"]

#: Типы UI-компонентов (порядок — порядок поиска у объекта).
UI_TYPES: tuple[str, ...] = ("UIPanel", "UIImage", "UIButton", "UIText")

_COORD = 10_000.0


class UISection(FormSection):
    """Свойства UI-виджета.

    Args:
        context: Контекст редактора.
        component: Редактируемый компонент (потомок ``UIElement``).
        parent: Родительский виджет.

    Signals:
        changed: Параметр изменён.
    """

    changed = Signal()

    def __init__(
        self,
        context: EditorContext,
        component: UIElement,
        parent: QWidget | None = None,
    ) -> None:
        kind = type(component).__name__
        super().__init__(context.tr(f"inspector.section.{_key(kind)}"), parent)
        self.context = context
        self._ui = component
        tr = context.tr

        self.anchor = self.add_combo(
            tr("inspector.ui.anchor"),
            [(tr(f"inspector.ui.anchor.{a}"), a) for a in UI_ANCHORS],
            component.anchor,
            lambda value: self._set("anchor", str(value)),
        )
        self.pos_x, self.pos_y = self._pair(
            "inspector.ui.position", "position", component.position, ("X", "Y")
        )
        self.size_w, self.size_h = self._pair(
            "inspector.ui.size", "size", component.size, ("W", "H"), minimum=0.0
        )
        self.pivot_x, self.pivot_y = self._pair(
            "inspector.ui.pivot",
            "pivot",
            component.pivot,
            ("X", "Y"),
            minimum=0.0,
            maximum=1.0,
            step=0.05,
        )
        self.opacity = self.add_spin(
            tr("inspector.ui.opacity"),
            component.opacity,
            lambda value: self._set("opacity", float(value)),
            maximum=1.0,
            step=0.05,
        )
        self.layer = self.add_int(
            tr("inspector.ui.sorting_layer"),
            component.sorting_layer,
            lambda value: self._set("sorting_layer", int(value)),
            minimum=-1000,
            maximum=1000,
        )
        self.visible = self.add_check(
            tr("inspector.ui.visible"),
            component.visible,
            lambda value: self._set("visible", bool(value)),
        )
        self.clip = self.add_check(
            tr("inspector.ui.clip"),
            component.clip,
            lambda value: self._set("clip", bool(value)),
        )
        if isinstance(component, UIPanel):
            self._build_panel(component)
        elif isinstance(component, UIImage):
            self._build_image(component)
        elif isinstance(component, UIButton):
            self._build_button(component)

    @property
    def component(self) -> UIElement:
        """Редактируемый компонент."""
        return self._ui

    # ------------------------------------------------------------------ #
    # Поля по типам
    # ------------------------------------------------------------------ #
    def _build_panel(self, panel: UIPanel) -> None:
        tr = self.context.tr
        self.color, self.color_alpha = self._rgba(
            "inspector.ui.color", "color", panel.color
        )
        self.border_color, self.border_alpha = self._rgba(
            "inspector.ui.border_color", "border_color", panel.border_color
        )
        self.border_width = self.add_spin(
            tr("inspector.ui.border_width"),
            panel.border_width,
            lambda value: self._set("border_width", float(value)),
            maximum=256.0,
            step=0.5,
        )

    def _build_image(self, image: UIImage) -> None:
        tr = self.context.tr
        self.texture = QLineEdit(image.texture_path, self)
        self.texture.setPlaceholderText(tr("inspector.ui.texture_hint"))
        self.texture.editingFinished.connect(
            lambda: self._set("texture_path", self.texture.text().strip())
        )
        self.form.addRow(tr("inspector.ui.texture"), self.texture)
        self.color, self.color_alpha = self._rgba(
            "inspector.ui.tint", "color", image.color
        )

    def _build_button(self, button: UIButton) -> None:
        tr = self.context.tr
        self.label = QLineEdit(button.text, self)
        self.label.textChanged.connect(lambda value: self._set("text", value))
        self.form.addRow(tr("inspector.ui.label"), self.label)
        self.normal_color, self.normal_alpha = self._rgba(
            "inspector.ui.normal_color", "normal_color", button.normal_color
        )
        self.hover_color, self.hover_alpha = self._rgba(
            "inspector.ui.hover_color", "hover_color", button.hover_color
        )
        self.pressed_color, self.pressed_alpha = self._rgba(
            "inspector.ui.pressed_color", "pressed_color", button.pressed_color
        )
        self.text_color, self.text_alpha = self._rgba(
            "inspector.ui.text_color", "text_color", button.text_color
        )
        self.font_size = self.add_int(
            tr("inspector.ui.font_size"),
            button.size_px,
            lambda value: self._set("size_px", int(value)),
            minimum=8,
            maximum=256,
        )
        self.on_click = QLineEdit(button.on_click, self)
        self.on_click.setPlaceholderText("event:name | load_scene:path")
        self.on_click.setToolTip(tr("inspector.ui.on_click_hint"))
        self.on_click.editingFinished.connect(self._apply_click)
        self.form.addRow(tr("inspector.ui.on_click"), self.on_click)
        self.interactive = self.add_check(
            tr("inspector.ui.interactive"),
            button.interactive,
            lambda value: self._set("interactive", bool(value)),
        )

    # ------------------------------------------------------------------ #
    # Помощники
    # ------------------------------------------------------------------ #
    def _pair(
        self,
        label_key: str,
        attribute: str,
        value: tuple[float, float],
        suffixes: tuple[str, str],
        *,
        minimum: float = -_COORD,
        maximum: float = _COORD,
        step: float = 1.0,
    ):
        """Два числовых поля (X и Y / ширина и высота) для кортежа."""
        current = list(value)

        def apply(index: int):
            def slot(number: float) -> None:
                current[index] = float(number)
                self._set(attribute, (current[0], current[1]))

            return slot

        label = self.context.tr(label_key)
        first = self.add_spin(
            f"{label} {suffixes[0]}",
            current[0],
            apply(0),
            minimum=minimum,
            maximum=maximum,
            step=step,
        )
        second = self.add_spin(
            f"{label} {suffixes[1]}",
            current[1],
            apply(1),
            minimum=minimum,
            maximum=maximum,
            step=step,
        )
        return first, second

    def _rgba(self, label_key: str, attribute: str, value: tuple[float, ...]):
        """Цвет RGB кнопкой выбора плюс отдельное поле альфы."""
        label = self.context.tr(label_key)
        current = list(value)
        button = ColorButton(Vector3(current[0], current[1], current[2]), self)
        button.set_label(label)

        def apply_rgb(color: Vector3) -> None:
            current[:3] = [color.x, color.y, color.z]
            self._set(attribute, tuple(current))

        def apply_alpha(alpha: float) -> None:
            current[3] = float(alpha)
            self._set(attribute, tuple(current))

        button.color_changed.connect(apply_rgb)
        self.form.addRow(label, button)
        alpha = self.add_spin(
            f"{label} α", current[3], apply_alpha, maximum=1.0, step=0.05
        )
        return button, alpha

    def _apply_click(self) -> None:
        """Применить действие кнопки; неверный формат возвращает прежнее значение."""
        try:
            self._set("on_click", self.on_click.text().strip())
        except ValueError:
            self.on_click.setText(self._ui.on_click)  # type: ignore[attr-defined]

    def _set(self, attribute: str, value: object) -> None:
        self.context.set_attr(self._ui, attribute, value)
        self.changed.emit()

    def __repr__(self) -> str:
        return f"<UISection {type(self._ui).__name__}>"


def _key(kind: str) -> str:
    """Ключ i18n для типа: ``UIPanel`` → ``ui_panel``."""
    return {
        "UIPanel": "ui_panel",
        "UIImage": "ui_image",
        "UIButton": "ui_button",
        "UIText": "ui_text",
    }[kind]
