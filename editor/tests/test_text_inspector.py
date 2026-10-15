"""Текст как компонент объекта: создание, секция инспектора, дублирование."""

from __future__ import annotations

from editor.panels.text_section import TextSection
from engine.core.components import TextRenderer


def _make(window):
    game_object = window.create_text()
    return game_object, game_object.get_component(TextRenderer)


def test_text_is_created_from_menu_action(window) -> None:
    count = len(window.context.scene)
    game_object, text = _make(window)
    assert len(window.context.scene) == count + 1
    assert game_object.name == "Text"
    assert text.text == "Text" and text.size_px == 32
    assert window.context.selection.primary is game_object


def test_inspector_shows_text_section(window) -> None:
    _, text = _make(window)
    section = window.inspector.text_section
    assert isinstance(section, TextSection)
    assert section.component is text


def test_edits_apply_instantly_without_rebuild(window) -> None:
    _, text = _make(window)
    section = window.inspector.text_section
    section.edit.setPlainText("Привет")
    section.size.setValue(64)
    section.alpha.setValue(0.5)
    section.wrap.setValue(4.0)
    section.line_spacing.setValue(1.5)
    section.units.setValue(0.02)
    section.outline_width.setValue(2.0)
    section.bold.setChecked(True)
    section.billboard.setChecked(False)
    section.align.setCurrentIndex(section.align.findData("center"))
    section.vertical_align.setCurrentIndex(section.vertical_align.findData("middle"))
    assert (text.text, text.size_px, text.alpha) == ("Привет", 64, 0.5)
    assert (text.wrap_width, text.line_spacing) == (4.0, 1.5)
    assert text.units_per_pixel == 0.02 and text.outline_width == 2.0
    assert text.bold and not text.billboard
    assert (text.align, text.vertical_align) == ("center", "middle")
    assert window.inspector.text_section is section  # секция не пересобиралась


def test_units_per_pixel_range(window) -> None:
    _make(window)
    spin = window.inspector.text_section.units
    assert spin.minimum() == 0.001 and spin.maximum() == 1.0
    assert window.inspector.text_section.size.minimum() == 8
    assert window.inspector.text_section.size.maximum() == 256


def test_duplicate_copies_text(window) -> None:
    game_object, text = _make(window)
    text.text = "Копия"
    clone = window.context.duplicate_object(game_object)
    copied = clone.get_component(TextRenderer)
    assert copied is not text and copied.text == "Копия"
