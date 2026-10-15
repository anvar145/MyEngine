"""Авто-секция: header «Включён/Удалить», QUAT, LIST (editor.panels.auto_section)."""

from __future__ import annotations

import pytest

from editor.panels.auto_section import AutoSection, ListEditor
from editor.widgets import VectorEditor
from engine.core.component import Component
from engine.core.math3d import Quaternion
from engine.core.reflection import (
    FieldKind,
    InvalidPolicy,
    component_fields,
    field,
    registry,
)


@component_fields
class ExtProbe(Component):
    """Компонент с кватернионом и списками."""

    rotation: tuple[float, float, float, float] = field(
        (0.0, 0.0, 0.0, 1.0), kind=FieldKind.QUAT
    )
    numbers: list = field([1.0, 2.0], kind=FieldKind.LIST)
    points: list = field([(0.0, 0.0, 0.0)], kind=FieldKind.LIST)
    names: list = field([], kind=FieldKind.LIST)
    mode: str = field("a", kind=FieldKind.ENUM, choices=("a", "b"))
    strict_mode: str = field(
        "a", kind=FieldKind.ENUM, choices=("a", "b"), on_invalid=InvalidPolicy.ERROR
    )


def _section(context):
    comp = ExtProbe()
    return comp, AutoSection(context, comp)


def test_header_controls_toggle_enabled(context) -> None:
    comp, section = _section(context)
    assert section.enabled_check is not None and section.remove_button is not None
    assert section.enabled_check.isChecked()
    seen: list[int] = []
    section.changed.connect(lambda: seen.append(1))
    section.enabled_check.click()
    assert comp.enabled is False
    assert section.header.property("frozen") is True
    assert seen == [1]


def test_header_can_be_disabled(context) -> None:
    section = AutoSection(context, ExtProbe(), header=False)
    assert section.enabled_check is None


def test_remove_button_emits_signal(context) -> None:
    _, section = _section(context)
    hits: list[int] = []
    section.remove_requested.connect(lambda: hits.append(1))
    section.remove_button.click()
    assert hits == [1]


def test_quat_edited_as_euler(context) -> None:
    comp, section = _section(context)
    editor = section.widgets["rotation"]
    assert isinstance(editor, VectorEditor)
    from engine.core.math3d import Vector3

    editor.set_value(Vector3(0.0, 90.0, 0.0), emit=True)
    expected = Quaternion.from_euler(0.0, 90.0, 0.0).to_tuple()
    assert comp.rotation == pytest.approx(expected, abs=1e-6)


def test_quat_shows_stored_rotation(context) -> None:
    comp = ExtProbe()
    comp.rotation = Quaternion.from_euler(0.0, 45.0, 0.0).to_tuple()
    editor = AutoSection(context, comp).widgets["rotation"]
    assert editor.value().y == pytest.approx(45.0, abs=1e-3)


def test_list_widget_and_operations(context) -> None:
    comp, section = _section(context)
    editor = section.widgets["numbers"]
    assert isinstance(editor, ListEditor)
    assert len(editor.rows) == 2
    seen: list[int] = []
    section.changed.connect(lambda: seen.append(1))
    editor.add_item()
    assert comp.numbers == [1.0, 2.0, 0.0] and len(editor.rows) == 3
    editor.move_item(2, -1)
    assert comp.numbers == [1.0, 0.0, 2.0]
    editor.move_item(0, -1)  # за границу не выходит
    assert comp.numbers == [1.0, 0.0, 2.0]
    editor.remove_item(0)
    assert comp.numbers == [0.0, 2.0]
    editor.set_item(1, 9.5)
    assert comp.numbers == [0.0, 9.5]
    assert len(seen) == 4  # холостой move_item не считается


def test_list_of_vector3_and_empty(context) -> None:
    comp, section = _section(context)
    points = section.widgets["points"]
    points.add_item()
    assert comp.points == [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    names = section.widgets["names"]
    names.add_item()
    assert comp.names == [""]


def test_list_edit_marks_scene_modified_via_changed(context) -> None:
    _, section = _section(context)
    hits: list[int] = []
    section.changed.connect(lambda: hits.append(1))
    section.widgets["numbers"].add_item()
    assert hits


def test_list_serializes_through_registry() -> None:
    comp = ExtProbe()
    comp.numbers = [3.0, 4.0]
    comp.points = [(1.0, 2.0, 3.0)]
    data = registry.serialize(comp)
    assert data["numbers"] == [3.0, 4.0]
    restored = registry.deserialize(ExtProbe, data)
    assert restored.numbers == [3.0, 4.0]
    assert restored.points == [(1.0, 2.0, 3.0)]


def test_enum_invalid_value_falls_back_with_warning(caplog) -> None:
    desc = registry.for_name(ExtProbe, "mode")
    with caplog.at_level("WARNING"):
        value = registry.clamp_descriptor("ExtProbe", desc, "zzz")
    assert value == "a"
    assert "zzz" in caplog.text
    assert registry.clamp_descriptor("ExtProbe", desc, "b") == "b"


def test_enum_invalid_value_raises_under_error_policy() -> None:
    desc = registry.for_name(ExtProbe, "strict_mode")
    with pytest.raises(ValueError, match="strict_mode"):
        registry.clamp_descriptor("ExtProbe", desc, "zzz")
