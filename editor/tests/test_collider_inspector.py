"""Верификация Collider: поля инспектора, живая правка и i18n."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from editor.panels.physics_sections import ColliderSection
from engine.core.math3d import Vector3
from engine.core.physics_components import (
    SHAPE_TYPES,
    ColliderComponent,
    RigidBodyComponent,
)

DT = 1.0 / 60.0
I18N = Path(__file__).resolve().parents[1] / "i18n"

FIELD_KEYS = (
    "shape", "shape.box", "shape.sphere", "shape.capsule", "size", "radius",
    "height", "offset", "is_trigger", "is_trigger_hint", "friction",
    "restitution", "layer", "layer_hint", "mask", "mask_hint",
    "rotation_offset", "rotation_offset_hint", "friction_combine",
    "restitution_combine", "combine_hint", "combine.multiply", "combine.min",
    "combine.max", "combine.average", "mask_all", "mask_none",
)


def _type_bits(editor, text: str) -> None:
    """Вводит текст в hex-поле ``MaskEditor`` (layer или mask) и подтверждает."""
    editor.line_edit.setText(text)
    editor.line_edit.editingFinished.emit()


def _type_mask(section, text: str) -> None:
    """Вводит текст в hex-поле маски и подтверждает ввод."""
    _type_bits(section.mask, text)


def _floor(scene) -> None:
    scene.create_game_object(
        "Floor", position=Vector3(0, -5, 0), scale=Vector3(100, 10, 100)
    ).add_component(ColliderComponent, shape="box", size=(1, 1, 1))


def _select(window, y: float = 3.0, body: bool = True, **collider):
    scene = window.context.scene
    go = scene.create_game_object("Obj", position=Vector3(0, y, 0))
    comp = go.add_component(ColliderComponent, **collider)
    if body:
        go.add_component(RigidBodyComponent, can_sleep=False)
    window.context.selection.select(go)
    window.inspector.refresh()
    return go, comp, window.inspector.collider_section


def _run(scene, steps: int) -> None:
    for _ in range(steps):
        scene.fixed_update(DT)
        scene.update(DT)


def _body(scene, go):
    return scene.physics.world.get_body(scene.physics.handle_of(go))


# --- поля ---------------------------------------------------------------------------

def test_every_field_has_a_widget_and_writes_component(window) -> None:
    _, comp, section = _select(window)
    section.shape.setCurrentIndex(section.shape.findData("capsule"))
    section.size.set_value(Vector3(2.0, 3.0, 4.0), emit=True)
    section.radius.setValue(0.75)
    section.height.setValue(3.5)
    section.offset.set_value(Vector3(0.5, -1.0, 2.0), emit=True)
    section.is_trigger.setChecked(True)
    section.friction.setValue(1.25)
    section.restitution.setValue(0.4)
    _type_bits(section.layer, "8")
    _type_mask(section, "6")
    data = comp.to_dict()
    assert data["shape"] == "capsule"
    assert data["radius"] == pytest.approx(0.75)
    assert data["height"] == pytest.approx(3.5)
    assert data["is_trigger"] is True
    assert data["friction"] == pytest.approx(1.25)
    assert data["restitution"] == pytest.approx(0.4)
    assert (data["layer"], data["mask"]) == (8, 6)
    assert data["size"] == pytest.approx([2.0, 3.0, 4.0])
    assert data["offset"] == pytest.approx([0.5, -1.0, 2.0])


def test_section_shows_component_values(window) -> None:
    _, _, section = _select(
        window, shape="capsule", size=(2, 3, 4), radius=0.3, height=2.5,
        offset=(1, 2, 3), is_trigger=True, friction=0.8, restitution=0.6,
        layer=4, mask=12)
    assert section.shape.currentData() == "capsule"
    assert section.radius.value() == pytest.approx(0.3)
    assert section.height.value() == pytest.approx(2.5)
    assert section.is_trigger.isChecked()
    assert section.friction.value() == pytest.approx(0.8)
    assert section.restitution.value() == pytest.approx(0.6)
    assert (section.layer.value(), section.mask.value()) == (4, 12)
    size, offset = section.size.value(), section.offset.value()
    assert (size.x, size.y, size.z) == pytest.approx((2, 3, 4))
    assert (offset.x, offset.y, offset.z) == pytest.approx((1, 2, 3))


def test_shape_combo_lists_exactly_the_supported_shapes(window) -> None:
    _, _, section = _select(window)
    items = [section.shape.itemData(i) for i in range(section.shape.count())]
    assert items == list(SHAPE_TYPES)


def test_section_switches_with_selection(window) -> None:
    _, _, first = _select(window, radius=0.2, shape="sphere")
    second_go = window.context.scene.create_game_object("Second")
    second_go.add_component(ColliderComponent, shape="sphere", radius=0.9)
    window.context.selection.select(second_go)
    window.inspector.refresh()
    assert window.inspector.collider_section.radius.value() == pytest.approx(0.9)
    assert first is not window.inspector.collider_section


def test_no_collider_section_without_component(window) -> None:
    go = window.context.scene.create_game_object("Bare")
    window.context.selection.select(go)
    window.inspector.refresh()
    assert getattr(window.inspector, "collider_section", None) is None
    assert window.inspector.findChildren(ColliderSection) == []


# --- доступность полей по форме -----------------------------------------------------

@pytest.mark.parametrize(("shape", "size", "radius", "height"), [
    ("box", True, False, False),
    ("sphere", False, True, False),
    ("capsule", False, True, True),
])
def test_shape_selects_available_fields(window, shape, size, radius, height) -> None:
    _, _, section = _select(window, shape=shape)
    assert section.size.isEnabled() is size
    assert section.radius.isEnabled() is radius
    assert section.height.isEnabled() is height


def test_changing_shape_in_combo_updates_available_fields(window) -> None:
    _, comp, section = _select(window, shape="box")
    for shape, size, radius, height in (
        ("sphere", False, True, False),
        ("capsule", False, True, True),
        ("box", True, False, False),
    ):
        section.shape.setCurrentIndex(section.shape.findData(shape))
        assert comp.shape == shape
        assert (section.size.isEnabled(), section.radius.isEnabled(),
                section.height.isEnabled()) == (size, radius, height)


def test_fields_common_to_all_shapes_are_always_enabled(window) -> None:
    _, _, section = _select(window)
    for shape in SHAPE_TYPES:
        section.shape.setCurrentIndex(section.shape.findData(shape))
        for widget in (section.offset, section.is_trigger, section.friction,
                       section.restitution, section.layer, section.mask):
            assert widget.isEnabled()


# --- пределы виджетов ------------------------------------------------------------

def test_widget_limits_protect_component_from_invalid_values(window) -> None:
    _, comp, section = _select(window, shape="capsule")
    section.radius.setValue(-5.0)
    assert comp.radius == pytest.approx(0.001)
    section.height.setValue(-5.0)
    assert comp.height == pytest.approx(0.001)
    section.restitution.setValue(7.0)
    assert comp.restitution == 1.0
    section.restitution.setValue(-1.0)
    assert comp.restitution == 0.0
    section.friction.setValue(-1.0)
    assert comp.friction == 0.0
    section.friction.setValue(99.0)
    assert comp.friction == 10.0
    _type_bits(section.layer, "FFFFFFFF")
    assert comp.layer == 0xFFFFFFFF


def test_mask_widget_reaches_bit_31_through_hex_input(window) -> None:
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
    assert section.mask.line_edit.text() == "0xFFFFFFFF"
    section.mask.none_button.click()
    assert comp.mask == 0
    assert section.mask.line_edit.text() == "0x00000000"


def test_mask_hex_input_rejects_garbage(window) -> None:
    _, comp, section = _select(window, mask=6)
    edit = section.mask.line_edit
    edit.clear()
    for char in "xyz!12G345678901":
        edit.insert(char)
    assert all(c in "0123456789abcdefABCDEFxX" for c in edit.text())
    _type_mask(section, "")  # пустой ввод возвращает прежнее значение
    assert comp.mask == 6
    assert edit.text() == "0x00000006"


def test_mask_widget_shows_full_32_bit_value(window) -> None:
    _, _, section = _select(window, mask=0xDEADBEEF)
    assert section.mask.value() == 0xDEADBEEF
    assert section.mask.line_edit.text() == "0xDEADBEEF"


def test_rotation_offset_field_writes_quaternion(window) -> None:
    _, comp, section = _select(window)
    assert tuple(comp.rotation_offset) == (0.0, 0.0, 0.0, 1.0)
    section.rotation_offset.set_value(Vector3(0.0, 90.0, 0.0), emit=True)
    x, y, z, w = comp.rotation_offset
    assert (x, y, z, w) == pytest.approx((0.0, 0.7071067811865476, 0.0,
                                          0.7071067811865476))
    shown = section.rotation_offset.value()
    assert (shown.x, shown.y, shown.z) == pytest.approx((0.0, 90.0, 0.0))


def test_section_shows_rotation_offset_as_euler_angles(window) -> None:
    s = 0.7071067811865476
    _, _, section = _select(window, rotation_offset=(0.0, 0.0, s, s))
    shown = section.rotation_offset.value()
    assert (shown.x, shown.y, shown.z) == pytest.approx((0.0, 0.0, 90.0), abs=1e-3)


def test_combine_combos_list_modes_and_write_component(window) -> None:
    _, comp, section = _select(window)
    modes = [section.friction_combine.itemData(i)
             for i in range(section.friction_combine.count())]
    assert modes == ["multiply", "min", "max", "average"]
    assert section.friction_combine.currentData() == "multiply"
    assert section.restitution_combine.currentData() == "max"
    section.friction_combine.setCurrentIndex(section.friction_combine.findData("min"))
    section.restitution_combine.setCurrentIndex(
        section.restitution_combine.findData("average"))
    assert (comp.friction_combine, comp.restitution_combine) == ("min", "average")


def test_opening_inspector_does_not_rewrite_rotation_offset(window) -> None:
    _, comp, _ = _select(window, rotation_offset=(0.1, 0.2, 0.3, 0.9))
    assert tuple(comp.rotation_offset) == (0.1, 0.2, 0.3, 0.9)


def test_opening_inspector_does_not_rewrite_out_of_range_values(window) -> None:
    """Виджет показывает зажатое значение, но компонент не трогает."""
    _, comp, section = _select(window, friction=50.0, restitution=1.5)
    assert section.friction.value() == 10.0
    assert section.restitution.value() == 1.0
    assert comp.friction == 50.0 and comp.restitution == 1.5
    assert comp.mask == 0xFFFFFFFF  # виджет показывает 2147483647


def test_friction_widget_rounds_to_two_decimals(window) -> None:
    _, comp, section = _select(window)
    section.friction.setValue(0.123)
    assert comp.friction == pytest.approx(0.12)


def test_same_value_does_not_emit_changed(window) -> None:
    _, comp, section = _select(window)
    emitted: list[int] = []
    section.changed.connect(lambda: emitted.append(1))
    section.friction.setValue(comp.friction)
    section.is_trigger.setChecked(False)
    assert emitted == []
    section.friction.setValue(1.0)
    assert emitted == [1]


# --- живая правка влияет на симуляцию ---------------------------------------------

def test_live_edit_is_trigger_releases_resting_body(window) -> None:
    scene = window.context.scene
    _floor(scene)
    go, _, section = _select(window, y=1.0, shape="sphere", radius=0.25)
    _run(scene, 90)
    assert _body(scene, go).position[1] == pytest.approx(0.25, abs=0.03)
    section.is_trigger.setChecked(True)
    _run(scene, 120)
    assert _body(scene, go).position[1] < -1.0


def test_live_edit_size_changes_rest_height(window) -> None:
    scene = window.context.scene
    _floor(scene)
    go, _, section = _select(window, y=3.0, shape="box", size=(1, 1, 1))
    go.get_component(RigidBodyComponent).freeze_rotation = True
    _run(scene, 120)
    assert _body(scene, go).position[1] == pytest.approx(0.5, abs=0.03)
    section.size.set_value(Vector3(1.0, 3.0, 1.0), emit=True)
    _run(scene, 120)
    assert _body(scene, go).position[1] == pytest.approx(1.5, abs=0.03)


def test_live_edit_radius_changes_rest_height(window) -> None:
    scene = window.context.scene
    _floor(scene)
    go, _, section = _select(window, y=3.0, shape="sphere", radius=0.25)
    _run(scene, 120)
    section.radius.setValue(1.0)
    _run(scene, 120)
    assert _body(scene, go).position[1] == pytest.approx(1.0, abs=0.03)


def test_live_edit_height_changes_rest_height(window) -> None:
    scene = window.context.scene
    _floor(scene)
    go, _, section = _select(window, y=3.0, shape="capsule", radius=0.5, height=2.0)
    go.get_component(RigidBodyComponent).freeze_rotation = True
    _run(scene, 120)
    section.height.setValue(4.0)
    _run(scene, 120)
    assert _body(scene, go).position[1] == pytest.approx(2.0, abs=0.03)


def test_live_edit_shape_switch_changes_rest_height(window) -> None:
    scene = window.context.scene
    _floor(scene)
    go, _, section = _select(window, y=3.0, shape="box", size=(1, 1, 1))
    go.get_component(RigidBodyComponent).freeze_rotation = True
    _run(scene, 120)
    assert _body(scene, go).position[1] == pytest.approx(0.5, abs=0.03)
    section.shape.setCurrentIndex(section.shape.findData("sphere"))
    section.radius.setValue(0.25)
    _run(scene, 120)
    assert _body(scene, go).position[1] == pytest.approx(0.25, abs=0.03)


def test_live_edit_offset_moves_collider(window) -> None:
    scene = window.context.scene
    go, _, section = _select(window, y=5.0, shape="sphere", radius=0.5)
    go.get_component(RigidBodyComponent).use_gravity = False
    section.offset.set_value(Vector3(3.0, 0.0, 0.0), emit=True)
    scene.fixed_update(DT)
    lo, hi = scene.physics.world.get_colliders(
        scene.physics.handle_of(go))[0].world_aabb
    assert (lo[0] + hi[0]) / 2 == pytest.approx(3.0, abs=1e-6)


def test_live_edit_restitution_makes_ball_bounce(window) -> None:
    scene = window.context.scene
    _floor(scene)
    go, _, section = _select(window, y=2.0, shape="sphere", radius=0.25)
    section.restitution.setValue(0.9)
    peak_after_bounce = 0.0
    seen_fall = False
    for _ in range(240):
        _run(scene, 1)
        vy = float(_body(scene, go).linear_velocity[1])
        seen_fall = seen_fall or vy < -3.0
        if seen_fall and vy > 0.5:
            peak_after_bounce = max(peak_after_bounce, vy)
    assert peak_after_bounce > 3.0


def test_live_edit_friction_changes_sliding_distance(window) -> None:
    scene = window.context.scene
    _floor(scene)
    go, _, section = _select(window, y=0.5, shape="box", size=(1, 1, 1))
    go.get_component(RigidBodyComponent).freeze_rotation = True
    _run(scene, 30)
    section.friction.setValue(0.0)
    _run(scene, 2)
    _body(scene, go).linear_velocity[0] = 2.0
    _run(scene, 60)
    far = _body(scene, go).position[0]
    section.friction.setValue(2.0)
    _run(scene, 120)
    stopped = _body(scene, go).position[0]
    assert far > 1.5
    assert stopped - far < 0.5  # на трении 2.0 скользит куда меньше


def test_live_edit_layer_lets_body_pass_through(window) -> None:
    scene = window.context.scene
    wall = scene.create_game_object("Wall", position=Vector3(0, 0, 0))
    wall.add_component(ColliderComponent, shape="box", size=(1, 4, 4))
    go, _, section = _select(window, y=0.0, shape="sphere", radius=0.25)
    go.transform.position = Vector3(-4, 0, 0)
    go.get_component(RigidBodyComponent).use_gravity = False
    scene.fixed_update(DT)
    _type_bits(section.layer, "2")
    _type_mask(section, "4")  # слой стены (1) в маске не значится
    scene.fixed_update(DT)
    _body(scene, go).linear_velocity[0] = 3.0
    _run(scene, 180)
    assert _body(scene, go).position[0] > 3.0


# --- i18n ---------------------------------------------------------------------------

def _tables() -> tuple[dict, dict]:
    ru = json.loads((I18N / "ru.json").read_text(encoding="utf-8"))
    en = json.loads((I18N / "en.json").read_text(encoding="utf-8"))
    return ru, en


@pytest.mark.parametrize("key", FIELD_KEYS)
def test_collider_i18n_keys_exist_in_both_languages(key: str) -> None:
    ru, en = _tables()
    full = f"inspector.physics.{key}"
    assert ru.get(full), f"нет {full} в ru"
    assert en.get(full), f"нет {full} в en"


def test_collider_section_title_key_exists() -> None:
    ru, en = _tables()
    for table in (ru, en):
        assert table["inspector.physics.collider"]
        assert table["inspector.physics.add.collider"]


def test_collider_labels_differ_between_languages() -> None:
    ru, en = _tables()
    for key in FIELD_KEYS:
        full = f"inspector.physics.{key}"
        assert ru[full] != en[full], full


def test_section_uses_translated_labels(window) -> None:
    _, _, section = _select(window)
    labels = {
        section.form.itemAt(row, section.form.ItemRole.LabelRole).widget().text()
        for row in range(section.form.rowCount())
        if section.form.itemAt(row, section.form.ItemRole.LabelRole) is not None
        and section.form.itemAt(row, section.form.ItemRole.LabelRole).widget()
    }
    tr = window.context.tr
    for key in ("shape", "size", "radius", "height", "offset", "friction",
                "restitution", "layer", "mask"):
        assert tr(f"inspector.physics.{key}") in labels, key


def test_layer_widget_is_mask_editor_and_reaches_bit_31(window) -> None:
    from editor.panels.physics_sections import MaskEditor

    _, comp, section = _select(window)
    assert isinstance(section.layer, MaskEditor)
    _type_bits(section.layer, "0x80000001")
    assert comp.layer == 0x80000001
    section.layer.none_button.click()
    assert comp.layer == 0
    section.layer.all_button.click()
    assert comp.layer == 0xFFFFFFFF
