"""Тесты переиспользуемых виджетов редактора."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QLabel

from editor.widgets import FocusComboBox, FocusDoubleSpinBox, VectorEditor


def _wheel(widget) -> QWheelEvent:
    return QWheelEvent(
        QPointF(2, 2),
        QPointF(2, 2),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def test_wheel_ignored_without_focus(qapp) -> None:
    spin = FocusDoubleSpinBox()
    spin.setValue(1.0)
    event = _wheel(spin)
    spin.wheelEvent(event)
    assert spin.value() == 1.0
    assert not event.isAccepted()


def test_wheel_changes_value_with_focus(qapp, monkeypatch) -> None:
    spin = FocusDoubleSpinBox()
    spin.setValue(1.0)
    monkeypatch.setattr(spin, "hasFocus", lambda: True)
    spin.wheelEvent(_wheel(spin))
    assert spin.value() > 1.0


def test_combo_wheel_ignored_without_focus(qapp) -> None:
    combo = FocusComboBox()
    combo.addItems(["a", "b"])
    combo.wheelEvent(_wheel(combo))
    assert combo.currentIndex() == 0


def test_vector_editor_axis_letters_are_separate_labels(qapp) -> None:
    editor = VectorEditor()
    letters = [label.text() for label in editor.findChildren(QLabel)]
    assert letters == ["X", "Y", "Z"]
    assert all(spin.prefix() == "" for spin in editor._spins)


def test_vector_editor_fields_have_equal_width(qapp) -> None:
    from engine.core.math3d import Vector3

    editor = VectorEditor()
    editor.resize(300, 24)
    editor.show()
    for value in (Vector3(0, 0, 0), Vector3(-123.456, 0.5, 1e6)):
        editor.set_value(value)
        qapp.processEvents()
        widths = [spin.width() for spin in editor._spins]
        assert max(widths) - min(widths) <= 1, widths  # округление до пикселя
    editor.resize(420, 24)
    qapp.processEvents()
    widths = [spin.width() for spin in editor._spins]
    assert max(widths) - min(widths) <= 1, widths


def test_click_selects_whole_field(qapp) -> None:
    from PySide6.QtTest import QTest

    spin = FocusDoubleSpinBox()
    spin.setValue(12.345)
    spin.show()
    spin.activateWindow()
    qapp.processEvents()
    spin.clearFocus()
    QTest.mouseClick(spin, Qt.MouseButton.LeftButton, pos=spin.rect().center())
    QTest.qWait(30)
    assert spin.hasFocus()
    assert spin.lineEdit().selectedText() == spin.lineEdit().text()


def test_inspector_fields_fit_narrow_panel(window) -> None:
    """Минимум контента инспектора не шире узкой панели (ru, все секции)."""
    from PySide6.QtWidgets import QApplication

    from engine.core.math3d import Vector3
    from engine.core.physics_components import (
        CharacterControllerComponent,
        ColliderComponent,
        RigidBodyComponent,
    )

    window.set_language("ru")
    window.show()
    inspector = window.inspector
    inspector.show()
    inspector.resize(280, 900)
    scene = window.context.scene
    body = scene.create_game_object("Hero", position=Vector3(0, 0, 0))
    body.add_component(RigidBodyComponent)
    body.add_component(ColliderComponent)
    character = scene.create_game_object("Char", position=Vector3(0, 0, 0))
    character.add_component(CharacterControllerComponent)
    objects = [
        window.create_camera(),
        window.create_primitive("cube"),
        window.create_light("point"),
        window.create_text(),
        body,
        character,
    ]
    for obj in objects:
        window.context.selection.select(obj)
        inspector.refresh()
        QApplication.processEvents()
        content = inspector._content
        content.layout().activate()
        assert content.minimumSizeHint().width() <= 280, obj.name
