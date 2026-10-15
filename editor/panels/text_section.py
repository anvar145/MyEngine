"""Секция «Текст» инспектора: правка :class:`~engine.core.components.TextRenderer`.

Значения применяются сразу (без перестройки инспектора); об изменении сообщает
сигнал ``changed``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QPlainTextEdit, QWidget

from editor.widgets import ColorButton, FormSection
from engine.core.components import TEXT_ALIGNS, TEXT_VERTICAL_ALIGNS, TextRenderer
from engine.core.math3d import Vector3
from engine.rendering.text import DEFAULT_FONT

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from editor.context import EditorContext

__all__ = ["TextSection", "list_fonts"]

_FONT_SUFFIXES = {".ttf", ".otf"}


def list_fonts() -> list[str]:
    """Имена файлов шрифтов из ``assets/fonts`` (по алфавиту)."""
    folder = DEFAULT_FONT.parent
    if not folder.is_dir():
        return []
    return sorted(
        path.name for path in folder.iterdir() if path.suffix.lower() in _FONT_SUFFIXES
    )


class _TextEdit(QPlainTextEdit):
    """Многострочное поле: прокрутка колесом только при фокусе."""

    def wheelEvent(self, event) -> None:  # noqa: N802 - API Qt
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class TextSection(FormSection):
    """Свойства текстового компонента.

    Args:
        context: Контекст редактора.
        component: Редактируемый компонент.
        parent: Родительский виджет.

    Signals:
        changed: Параметр текста изменён (нужно перерисовать кадр).
    """

    changed = Signal()

    def __init__(
        self,
        context: EditorContext,
        component: TextRenderer,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(context.tr("inspector.section.text"), parent)
        self.context = context
        self._text = component
        tr = context.tr

        self.edit = _TextEdit(self)
        self.edit.setPlainText(component.text)
        self.edit.setFixedHeight(72)
        self.edit.textChanged.connect(self._apply_text)
        self.form.addRow(tr("inspector.text.text"), self.edit)

        fonts = [(tr("inspector.text.font_default"), "")]
        fonts += [(name, f"assets/fonts/{name}") for name in list_fonts()]
        known = {data for _, data in fonts}
        if component.font_path and component.font_path not in known:
            fonts.append((component.font_path, component.font_path))
        self.font = self.add_combo(
            tr("inspector.text.font"), fonts, component.font_path, self._apply_font
        )
        self.size = self.add_int(
            tr("inspector.text.size"),
            component.size_px,
            self._apply_size,
            minimum=8,
            maximum=256,
        )
        self.color = ColorButton(Vector3(*component.color), self)
        self.color.set_label(tr("inspector.text.color"))
        self.color.color_changed.connect(self._apply_color)
        self.form.addRow(tr("inspector.text.color"), self.color)
        self.alpha = self.add_spin(
            tr("inspector.text.alpha"),
            component.alpha,
            self._apply_alpha,
            maximum=1.0,
            step=0.05,
        )
        self.align = self.add_combo(
            tr("inspector.text.align"),
            [(tr(f"inspector.text.align.{a}"), a) for a in TEXT_ALIGNS],
            component.align,
            self._apply_align,
        )
        self.vertical_align = self.add_combo(
            tr("inspector.text.vertical_align"),
            [(tr(f"inspector.text.valign.{a}"), a) for a in TEXT_VERTICAL_ALIGNS],
            component.vertical_align,
            self._apply_vertical_align,
        )
        self.wrap = self.add_spin(
            tr("inspector.text.wrap_width"),
            component.wrap_width,
            self._apply_wrap,
            maximum=10_000.0,
            step=0.5,
            tooltip=tr("inspector.text.wrap_hint"),
        )
        self.line_spacing = self.add_spin(
            tr("inspector.text.line_spacing"),
            component.line_spacing,
            self._apply_line_spacing,
            minimum=0.5,
            maximum=3.0,
            step=0.05,
        )
        self.bold = self.add_check(
            tr("inspector.text.bold"), component.bold, self._apply_bold
        )
        self.italic = self.add_check(
            tr("inspector.text.italic"), component.italic, self._apply_italic
        )
        self.outline_color = ColorButton(Vector3(*component.outline_color), self)
        self.outline_color.set_label(tr("inspector.text.outline_color"))
        self.outline_color.color_changed.connect(self._apply_outline_color)
        self.form.addRow(tr("inspector.text.outline_color"), self.outline_color)
        self.outline_width = self.add_spin(
            tr("inspector.text.outline_width"),
            component.outline_width,
            self._apply_outline_width,
            maximum=16.0,
            step=0.5,
            tooltip=tr("inspector.text.outline_hint"),
        )
        self.billboard = self.add_check(
            tr("inspector.text.billboard"), component.billboard, self._apply_billboard
        )
        self.units = self.add_spin(
            tr("inspector.text.units_per_pixel"),
            component.units_per_pixel,
            self._apply_units,
            minimum=0.001,
            maximum=1.0,
            step=0.001,
            decimals=3,
        )
        self.visible = self.add_check(
            tr("inspector.text.visible"), component.visible, self._apply_visible
        )
        self.layer = self.add_int(
            tr("inspector.text.sorting_layer"),
            component.sorting_layer,
            self._apply_layer,
            minimum=-1000,
            maximum=1000,
        )

    @property
    def component(self) -> TextRenderer:
        """Редактируемый компонент."""
        return self._text

    def _set(self, field: str, value: object) -> None:
        """Записать поле текста командой с undo."""
        self.context.set_attr(self._text, field, value)

    def _touch(self) -> None:
        self.changed.emit()

    def _apply_text(self) -> None:
        self._set("text", self.edit.toPlainText())
        self._touch()

    def _apply_font(self, value: object) -> None:
        self._set("font_path", str(value or ""))
        self._touch()

    def _apply_size(self, value: int) -> None:
        self._set("size_px", int(value))
        self._touch()

    def _apply_color(self, color: Vector3) -> None:
        self._set("color", (color.x, color.y, color.z))
        self._touch()

    def _apply_alpha(self, value: float) -> None:
        self._set("alpha", float(value))
        self._touch()

    def _apply_align(self, value: object) -> None:
        self._set("align", str(value))
        self._touch()

    def _apply_vertical_align(self, value: object) -> None:
        self._set("vertical_align", str(value))
        self._touch()

    def _apply_wrap(self, value: float) -> None:
        self._set("wrap_width", float(value))
        self._touch()

    def _apply_line_spacing(self, value: float) -> None:
        self._set("line_spacing", float(value))
        self._touch()

    def _apply_bold(self, value: bool) -> None:
        self._set("bold", bool(value))
        self._touch()

    def _apply_italic(self, value: bool) -> None:
        self._set("italic", bool(value))
        self._touch()

    def _apply_outline_color(self, color: Vector3) -> None:
        self._set("outline_color", (color.x, color.y, color.z))
        self._touch()

    def _apply_outline_width(self, value: float) -> None:
        self._set("outline_width", float(value))
        self._touch()

    def _apply_billboard(self, value: bool) -> None:
        self._set("billboard", bool(value))
        self._touch()

    def _apply_units(self, value: float) -> None:
        self._set("units_per_pixel", float(value))
        self._touch()

    def _apply_visible(self, value: bool) -> None:
        self._set("visible", bool(value))
        self._touch()

    def _apply_layer(self, value: int) -> None:
        self._set("sorting_layer", int(value))
        self._touch()

    def __repr__(self) -> str:
        return f"<TextSection текст={self._text.text!r}>"
