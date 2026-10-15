"""Верификация RigidBody: поля инспектора и их эффект на симуляцию."""

from __future__ import annotations

import pytest

from engine.core.physics_components import ColliderComponent, RigidBodyComponent

DT = 1.0 / 60.0


def _select(window, **kw):
    scene = window.context.scene
    go = scene.create_game_object("Phys")
    go.add_component(ColliderComponent)
    rb = go.add_component(RigidBodyComponent, **kw)
    go.transform.position = go.transform.position.__class__(0, 50, 0)
    window.context.selection.select(go)
    window.inspector.refresh()
    return go, rb, window.inspector.rigidbody_section


def _run(scene, steps) -> None:
    for _ in range(steps):
        scene.fixed_update(DT)
        scene.update(DT)


def test_every_field_has_a_widget_and_writes_component(window) -> None:
    _, rb, section = _select(window)
    section.body_type.setCurrentIndex(section.body_type.findData("kinematic"))
    section.mass.setValue(5.5)
    section.linear_damping.setValue(0.4)
    section.angular_damping.setValue(0.6)
    section.gravity_scale.setValue(-2.0)
    section.use_gravity.setChecked(False)
    section.freeze_rotation.setChecked(True)
    section.can_sleep.setChecked(False)
    assert rb.to_dict() == {
        "body_type": "kinematic", "mass": 5.5, "linear_damping": 0.4,
        "angular_damping": 0.6, "gravity_scale": -2.0, "use_gravity": False,
        "freeze_rotation": True, "can_sleep": False,
    }


def test_section_shows_component_values(window) -> None:
    _, _, section = _select(
        window, mass=3.0, linear_damping=0.2, gravity_scale=0.5,
        use_gravity=False, freeze_rotation=True, can_sleep=False,
    )
    assert section.mass.value() == pytest.approx(3.0)
    assert section.linear_damping.value() == pytest.approx(0.2)
    assert section.gravity_scale.value() == pytest.approx(0.5)
    assert not section.use_gravity.isChecked()
    assert section.freeze_rotation.isChecked()
    assert not section.can_sleep.isChecked()


def test_live_edit_gravity_scale_changes_fall(window) -> None:
    go, _, section = _select(window, linear_damping=0.0)
    scene = window.context.scene
    scene.physics.world.set_gravity((0, -9.81, 0))
    _run(scene, 30)
    y0 = go.transform.world_position.y
    section.gravity_scale.setValue(0.0)
    _run(scene, 30)
    y1 = go.transform.world_position.y
    section.gravity_scale.setValue(0.0)  # без изменений
    _run(scene, 30)
    # при g_scale=0 тело летит с набранной скоростью, но без ускорения
    assert y1 < y0
    v = (y1 - y0) / 0.5
    assert (go.transform.world_position.y - y1) / 0.5 == pytest.approx(v, rel=0.02)


def test_live_edit_use_gravity_checkbox(window) -> None:
    go, _, section = _select(window, linear_damping=0.0)
    section.use_gravity.setChecked(False)
    _run(window.context.scene, 60)
    assert go.transform.world_position.y == pytest.approx(50.0, abs=1e-4)
    section.use_gravity.setChecked(True)
    _run(window.context.scene, 60)
    assert go.transform.world_position.y < 50.0 - 1.0


def test_runtime_body_type_switch_from_inspector(window) -> None:
    go, _, section = _select(window)
    scene = window.context.scene
    _run(scene, 30)
    y = go.transform.world_position.y
    assert y < 50.0 - 0.5
    section.body_type.setCurrentIndex(section.body_type.findData("static"))
    _run(scene, 60)
    assert go.transform.world_position.y == pytest.approx(y, abs=1e-4)
    section.body_type.setCurrentIndex(section.body_type.findData("dynamic"))
    _run(scene, 30)
    assert go.transform.world_position.y < y - 0.5


def test_mass_widget_cannot_set_zero_mass(window) -> None:
    _, rb, section = _select(window)
    section.mass.setValue(0.0)
    assert rb.mass > 0.0  # минимум спинбокса 0.001 защищает от ValueError
