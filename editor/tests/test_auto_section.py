"""Авто-секция инспектора по дескрипторам полей (editor.panels.auto_section)."""

from __future__ import annotations

import math

from PySide6.QtWidgets import QCheckBox, QLineEdit

from editor.panels.auto_section import AutoSection, has_auto_section
from editor.panels.physics_sections import MaskEditor
from editor.widgets import FocusComboBox, FocusDoubleSpinBox, VectorEditor
from engine.core.component import Component
from engine.core.reflection import FieldKind, component_fields, field


@component_fields
class AutoProbe(Component):
    """Компонент без собственной секции: все виды полей."""

    speed: float = field(4.5, range=(0.0, 10.0), step=0.5, tooltip="Скорость")
    power: float = field(5.0, range=(0.0, math.inf), exclusive_min=True)
    count: int = field(3, range=(0, 20))
    flag: bool = True
    label: str = "abc"
    mode: str = field("one", kind=FieldKind.ENUM, choices=("one", "two"))
    pos: tuple[float, float, float] = (0.0, 1.0, 2.0)
    mask: int = field(0xFF, kind=FieldKind.MASK32)
    fixed: float = field(1.0, readonly=True)
    extra: float = field(2.0, visible_if="flag == True")
    hidden: float = field(9.0, serialize=False)


def _section(context):
    comp = AutoProbe()
    section = AutoSection(context, comp)
    return comp, section


def test_widget_per_kind(context) -> None:
    _, section = _section(context)
    w = section.widgets
    assert isinstance(w["speed"], FocusDoubleSpinBox)
    assert isinstance(w["count"], FocusDoubleSpinBox) and w["count"].decimals() == 0
    assert isinstance(w["flag"], QCheckBox)
    assert isinstance(w["label"], QLineEdit)
    assert isinstance(w["mode"], FocusComboBox)
    assert isinstance(w["pos"], VectorEditor)
    assert isinstance(w["mask"], MaskEditor)
    assert section.title.text() == "AutoProbe"


def test_range_step_and_tooltip_from_descriptor(context) -> None:
    _, section = _section(context)
    speed = section.widgets["speed"]
    assert (speed.minimum(), speed.maximum(), speed.singleStep()) == (0.0, 10.0, 0.5)
    assert speed.toolTip() == "Скорость"
    power = section.widgets["power"]
    assert power.minimum() == 1e-3  # открытый минимум
    assert power.maximum() > 1e5  # inf заменён большим числом


def test_editing_writes_component_and_emits(context) -> None:
    comp, section = _section(context)
    seen: list[int] = []
    section.changed.connect(lambda: seen.append(1))
    section.widgets["speed"].setValue(7.0)
    section.widgets["flag"].setChecked(False)
    section.widgets["label"].setText("xyz")
    section.widgets["label"].editingFinished.emit()
    combo = section.widgets["mode"]
    combo.setCurrentIndex(combo.findData("two"))
    section.widgets["pos"].value_changed.emit(
        type("V", (), {"x": 1.0, "y": 2.0, "z": 3.0})()
    )
    assert (comp.speed, comp.flag, comp.label, comp.mode) == (7.0, False, "xyz", "two")
    assert comp.pos == (1.0, 2.0, 3.0)
    assert len(seen) == 5


def test_mask_and_int_roundtrip(context) -> None:
    comp, section = _section(context)
    section.widgets["mask"].set_value(0xFFFFFFFF)
    section.widgets["mask"].value_changed.emit(0xFFFFFFFF)
    section.widgets["count"].setValue(7)
    assert comp.mask == 0xFFFFFFFF and comp.count == 7
    assert isinstance(comp.count, int)


def test_readonly_and_unserialized(context) -> None:
    _, section = _section(context)
    assert not section.widgets["fixed"].isEnabled()
    assert "hidden" in section.widgets  # не сериализуется, но видно в инспекторе


def test_visible_if_follows_other_field(context) -> None:
    _, section = _section(context)
    extra = section.widgets["extra"]
    assert not extra.isHidden()
    section.widgets["flag"].setChecked(False)
    assert extra.isHidden()
    assert section.labels["extra"].isHidden()
    section.widgets["flag"].setChecked(True)
    assert not extra.isHidden()


def test_has_auto_section() -> None:
    assert has_auto_section(AutoProbe)
    assert not has_auto_section(Component)


def test_inspector_builds_auto_section_but_keeps_manual(window) -> None:
    from engine.core.physics_components import CharacterControllerComponent

    go = window.context.scene.create_game_object("Probe")
    go.add_component(AutoProbe)
    go.add_component(CharacterControllerComponent)
    window.context.selection.select(go)
    window.inspector.refresh()
    sections = window.inspector.auto_sections
    assert [type(s.component) for s in sections] == [AutoProbe]
    assert window.inspector.character_section is not None
