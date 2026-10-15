"""Инспектор CharacterController: маска слоёв (32 бита), нижняя граница skin_width."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from editor.panels.physics_sections import CharacterControllerSection, MaskEditor
from engine.core.math3d import Vector3
from engine.core.physics_components import CharacterControllerComponent

I18N = Path(__file__).resolve().parents[1] / "i18n"

FIELD_KEYS = (
    "radius", "height", "step_offset", "step_offset_hint", "slope_limit",
    "skin_width", "layer", "mask", "mask_hint", "mask_all", "mask_none",
    "character.move_speed", "character.move_speed_hint",
    "character.sprint_multiplier", "character.sprint_multiplier_hint",
    "character.jump_velocity", "character.jump_velocity_hint",
    "character.use_gravity", "character.use_gravity_hint",
    "character.gravity_scale", "character.gravity_scale_hint",
    "character.ground_check_distance", "character.ground_check_distance_hint",
    "character.air_control", "character.air_control_hint",
    "character.snap_to_ground", "character.snap_to_ground_hint",
    "character.alignment_to_ground", "character.alignment_to_ground_hint",
)


def _select(window, **kwargs):
    go = window.context.scene.create_game_object("Hero", position=Vector3(0, 1, 0))
    comp = go.add_component(CharacterControllerComponent, **kwargs)
    window.context.selection.select(go)
    window.inspector.refresh()
    return go, comp, window.inspector.character_section


def _type_bits(editor, text: str) -> None:
    editor.line_edit.setText(text)
    editor.line_edit.editingFinished.emit()


def _type_mask(section, text: str) -> None:
    section.mask.line_edit.setText(text)
    section.mask.line_edit.editingFinished.emit()


def test_mask_is_mask_editor_not_int_spin(window) -> None:
    _, _, section = _select(window)
    assert isinstance(section, CharacterControllerSection)
    assert isinstance(section.mask, MaskEditor)


def test_mask_reaches_bit_31_through_ui_and_is_kept(window) -> None:
    _, comp, section = _select(window, mask=6)
    assert section.mask.value() == 6
    _type_mask(section, "FFFFFFFF")
    assert comp.mask == 0xFFFFFFFF
    assert comp.mask & (1 << 31)
    assert section.mask.line_edit.text() == "0xFFFFFFFF"
    _type_mask(section, "0x80000000")
    assert comp.mask == 0x80000000


def test_mask_all_and_none_buttons(window) -> None:
    _, comp, section = _select(window, mask=6)
    section.mask.all_button.click()
    assert comp.mask == 0xFFFFFFFF
    section.mask.none_button.click()
    assert comp.mask == 0


def test_default_mask_is_shown_in_full_and_not_rewritten(window) -> None:
    """Открытие инспектора не урезает маску 0xFFFFFFFF до 2147483647."""
    _, comp, section = _select(window)
    assert comp.mask == 0xFFFFFFFF
    assert section.mask.value() == 0xFFFFFFFF
    assert section.mask.line_edit.text() == "0xFFFFFFFF"


def test_mask_with_bit_31_survives_scene_round_trip(window) -> None:
    _, comp, section = _select(window)
    _type_mask(section, "0x80000001")
    restored = CharacterControllerComponent.from_dict(comp.to_dict())
    assert restored.mask == 0x80000001


def test_skin_width_widget_has_positive_minimum(window) -> None:
    _, comp, section = _select(window)
    assert section.skin_width.minimum() == pytest.approx(0.001)
    section.skin_width.setValue(0.0)
    assert comp.skin_width >= 0.001


def _tables() -> tuple[dict, dict]:
    ru = json.loads((I18N / "ru.json").read_text(encoding="utf-8"))
    en = json.loads((I18N / "en.json").read_text(encoding="utf-8"))
    return ru, en


@pytest.mark.parametrize("key", FIELD_KEYS)
def test_character_i18n_keys_exist_in_both_languages(key: str) -> None:
    ru, en = _tables()
    full = f"inspector.physics.{key}"
    assert ru.get(full), f"нет {full} в ru"
    assert en.get(full), f"нет {full} в en"


# --- п.14: поля, запись в компонент, неприменимые поля ----------------------

EDITABLE = ("radius", "height", "step_offset", "slope_limit", "skin_width",
            "layer", "mask", "move_speed", "sprint_multiplier", "jump_velocity",
            "use_gravity", "gravity_scale", "ground_check_distance", "air_control",
            "snap_to_ground", "alignment_to_ground")
NOT_APPLICABLE = ("gravity",)  # гравитация - свойство мира, а не персонажа


def test_every_component_field_has_a_widget_and_writes_component(window) -> None:
    _, comp, section = _select(window)
    section.radius.setValue(0.5)
    section.height.setValue(2.0)
    section.step_offset.setValue(0.4)
    section.slope_limit.setValue(50.0)
    section.skin_width.setValue(0.05)
    _type_bits(section.layer, "3")
    _type_mask(section, "0x00000006")
    section.move_speed.setValue(6.0)
    section.sprint_multiplier.setValue(2.0)
    section.jump_velocity.setValue(7.5)
    section.use_gravity.setChecked(False)
    section.gravity_scale.setValue(0.5)
    section.ground_check_distance.setValue(0.1)
    section.air_control.setValue(0.8)
    section.snap_to_ground.setChecked(False)
    section.alignment_to_ground.setChecked(True)
    assert comp.to_dict() == {
        "radius": 0.5, "height": 2.0, "step_offset": 0.4, "slope_limit": 50.0,
        "skin_width": 0.05, "layer": 3, "mask": 6,
        "move_speed": 6.0, "sprint_multiplier": 2.0, "jump_velocity": 7.5,
        "use_gravity": False, "gravity_scale": 0.5, "ground_check_distance": 0.1,
        "air_control": 0.8, "snap_to_ground": False, "alignment_to_ground": True,
    }


@pytest.mark.parametrize("name", EDITABLE)
def test_section_exposes_widget_for_each_field(window, name: str) -> None:
    _, _, section = _select(window)
    assert getattr(section, name) is not None


@pytest.mark.parametrize("name", NOT_APPLICABLE)
def test_not_applicable_fields_have_no_widget_or_component_slot(
    window, name: str
) -> None:
    """``gravity`` - не поле контроллера: мировая гравитация задаётся сценой."""
    _, comp, section = _select(window)
    assert not hasattr(section, name)
    assert name not in CharacterControllerComponent.__slots__
    assert name not in comp.to_dict()


def test_widgets_show_component_values(window) -> None:
    _, _, section = _select(
        window, radius=0.5, height=2.0, step_offset=0.4, slope_limit=50.0,
        skin_width=0.05, layer=3, move_speed=6.0, sprint_multiplier=2.0,
        jump_velocity=7.5, use_gravity=False, gravity_scale=0.5,
        ground_check_distance=0.1, air_control=0.8, snap_to_ground=False,
        alignment_to_ground=True,
    )
    assert section.radius.value() == pytest.approx(0.5)
    assert section.height.value() == pytest.approx(2.0)
    assert section.step_offset.value() == pytest.approx(0.4)
    assert section.slope_limit.value() == pytest.approx(50.0)
    assert section.skin_width.value() == pytest.approx(0.05)
    assert section.layer.value() == 3
    assert section.move_speed.value() == pytest.approx(6.0)
    assert section.sprint_multiplier.value() == pytest.approx(2.0)
    assert section.jump_velocity.value() == pytest.approx(7.5)
    assert section.use_gravity.isChecked() is False
    assert section.gravity_scale.value() == pytest.approx(0.5)
    assert section.ground_check_distance.value() == pytest.approx(0.1)
    assert section.air_control.value() == pytest.approx(0.8)
    assert section.snap_to_ground.isChecked() is False
    assert section.alignment_to_ground.isChecked() is True


def test_new_defaults_are_shown_as_in_the_component(window) -> None:
    _, _, section = _select(window)
    assert section.move_speed.value() == pytest.approx(4.5)
    assert section.sprint_multiplier.value() == pytest.approx(1.6)
    assert section.jump_velocity.value() == pytest.approx(5.0)
    assert section.use_gravity.isChecked() is True
    assert section.gravity_scale.value() == pytest.approx(1.0)
    assert section.ground_check_distance.value() == pytest.approx(0.03)
    assert section.air_control.value() == pytest.approx(0.3)
    assert section.snap_to_ground.isChecked() is True
    assert section.alignment_to_ground.isChecked() is False


def test_opening_inspector_does_not_rewrite_component(window) -> None:
    """Показ секции не меняет значения (в т.ч. slope_limit=90 при максимуме 89)."""
    _, comp, _section = _select(window, slope_limit=90.0, step_offset=12.0)
    assert comp.slope_limit == 90.0
    assert comp.step_offset == 12.0


def test_widget_limits_match_adapter_guarantees(window) -> None:
    _, _, section = _select(window)
    assert section.radius.minimum() > 0.0
    assert section.skin_width.minimum() > 0.0
    assert section.height.minimum() > 0.0
    assert section.slope_limit.minimum() >= 0.0
    assert section.slope_limit.maximum() <= 90.0
    assert section.step_offset.minimum() >= 0.0


def test_new_widget_limits_match_component_ranges(window) -> None:
    """Границы виджетов совпадают с допустимыми значениями компонента."""
    _, _, section = _select(window)
    assert section.move_speed.minimum() == 0.0
    assert section.sprint_multiplier.minimum() == 1.0
    assert section.jump_velocity.minimum() > 0.0
    assert section.gravity_scale.minimum() == 0.0
    assert section.ground_check_distance.minimum() > 0.0
    assert (section.air_control.minimum(), section.air_control.maximum()) == (0.0, 1.0)


@pytest.mark.parametrize(
    ("name", "bad"),
    [("sprint_multiplier", 0.5), ("jump_velocity", 0.0), ("air_control", 3.0),
     ("ground_check_distance", 0.0), ("move_speed", -2.0)],
)
def test_out_of_range_input_never_reaches_the_component(window, name, bad) -> None:
    """Виджет зажимает ввод: компонент остаётся в допустимой области."""
    from engine.core.physics_components import clamp_character_field

    _, comp, section = _select(window)
    getattr(section, name).setValue(bad)
    assert getattr(comp, name) == clamp_character_field(name, getattr(comp, name))


def test_new_edits_reach_a_registered_controller_without_rebuild(window) -> None:
    """Правка скорости в инспекторе действует на живой контроллер сразу."""
    go, comp, section = _select(window)
    window.context.scene.physics.sync_in()
    before = window.context.scene.physics._bindings[go.id].handle
    section.move_speed.setValue(9.0)
    section.alignment_to_ground.setChecked(True)
    window.context.scene.physics.sync_in()
    after = window.context.scene.physics._bindings[go.id].handle
    assert before == after
    assert comp.move_speed == 9.0


def test_labels_are_translated_not_raw_keys(window) -> None:
    from PySide6.QtWidgets import QLabel

    _, _, section = _select(window)
    texts = [label.text() for label in section.findChildren(QLabel)]
    assert texts
    assert not any("inspector.physics" in text for text in texts)


# --- слой: тот же MaskEditor, что у mask (32 бита) -------------------------------

def test_layer_is_mask_editor_and_reaches_bit_31(window) -> None:
    _, comp, section = _select(window)
    assert isinstance(section.layer, MaskEditor)
    assert section.layer.value() == 1
    _type_bits(section.layer, "0x80000001")
    assert comp.layer == 0x80000001
    assert section.layer.line_edit.text() == "0x80000001"


def test_layer_all_and_none_buttons(window) -> None:
    _, comp, section = _select(window)
    section.layer.all_button.click()
    assert comp.layer == 0xFFFFFFFF
    section.layer.none_button.click()
    assert comp.layer == 0


def test_layer_and_mask_are_independent(window) -> None:
    _, comp, section = _select(window)
    _type_bits(section.layer, "4")
    _type_mask(section, "2")
    assert (comp.layer, comp.mask) == (4, 2)


def test_layer_with_bit_31_survives_scene_round_trip(window) -> None:
    _, comp, section = _select(window)
    _type_bits(section.layer, "0x80000000")
    restored = CharacterControllerComponent.from_dict(comp.to_dict())
    assert restored.layer == 0x80000000


def test_slope_limit_widget_reaches_90_like_the_adapter(window) -> None:
    _, comp, section = _select(window)
    assert section.slope_limit.maximum() == 90.0
    section.slope_limit.setValue(90.0)
    assert comp.slope_limit == 90.0
    section.slope_limit.setValue(120.0)
    assert comp.slope_limit == 90.0
