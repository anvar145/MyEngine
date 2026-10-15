"""Инспектор: поля Behaviour (export) в ScriptFieldsForm, выбор класса, undo/redo."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QComboBox, QLabel

from editor.panels.script_fields import ReferenceField, ScriptFieldsForm
from editor.widgets import FocusComboBox, FocusDoubleSpinBox, VectorEditor
from engine.scripting import ScriptComponent

SOURCE = '''
from engine.core.math3d import Vector3
from engine.core.reflection import FieldKind
from engine.scripting import Behaviour, export

class Mover(Behaviour):
    speed: float = export(2.0, range=(0.0, 10.0), step=0.5, tooltip="м/с")
    mode: str = export("walk", choices=("walk", "run"), kind=FieldKind.ENUM)
    offset: Vector3 = export(Vector3(0, 1, 0))
    target: GameObject = export(None)
    tint: tuple = export((1.0, 0.0, 0.0), kind=FieldKind.COLOR)
'''

TWO = '''
from engine.scripting import Behaviour, export

class First(Behaviour):
    a: float = export(1.0)

class Second(Behaviour):
    b: float = export(2.0)
'''


def _commit(window) -> None:
    """Команды пишутся сразу: фиксировать нечего."""


def _setup(window, tmp_path: Path, source: str = SOURCE):
    path = tmp_path / "mover.py"
    path.write_text(source, "utf-8")
    cube = window.create_primitive("cube")
    script = cube.add_component(
        ScriptComponent(path, runtime=window.context.script_runtime)
    )
    _commit(window)
    window.context.selection.select(cube)
    window.inspector.refresh()
    return cube, script


def _form(window) -> ScriptFieldsForm:
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    forms = window.inspector.findChildren(ScriptFieldsForm)
    assert len(forms) == 1
    return forms[0]


def test_export_fields_get_matching_widgets(window, tmp_path) -> None:
    _setup(window, tmp_path)
    form = _form(window)
    assert form.field_names == ["speed", "mode", "offset", "target", "tint"]
    speed = form.widgets["speed"]
    assert isinstance(speed, FocusDoubleSpinBox)
    assert (speed.minimum(), speed.maximum()) == (0.0, 10.0)
    assert speed.singleStep() == 0.5
    assert speed.toolTip() == "м/с"
    assert isinstance(form.widgets["mode"], FocusComboBox)
    assert isinstance(form.widgets["offset"], VectorEditor)
    assert isinstance(form.widgets["target"], ReferenceField)
    assert isinstance(form.widgets["tint"], QLabel)  # без редактора - только показ
    assert not form.widgets["tint"].isEnabled()


def test_edit_marks_scene_modified_and_undo_redo(window, tmp_path) -> None:
    _, script = _setup(window, tmp_path)
    form = _form(window)
    form.widgets["speed"].setValue(7.5)
    assert script.behaviour.speed == 7.5
    assert script.fields == {"speed": 7.5}
    assert window.context.active_document.dirty
    _commit(window)

    window.undo()
    restored = window.context.selection.primary.get_component(ScriptComponent)
    assert restored.fields == {}
    assert restored.behaviour.speed == 2.0
    window.redo()
    redone = window.context.selection.primary.get_component(ScriptComponent)
    assert redone.fields == {"speed": 7.5}
    assert redone.behaviour.speed == 7.5


def test_enum_and_vec3_edit(window, tmp_path) -> None:
    _, script = _setup(window, tmp_path)
    form = _form(window)
    combo = form.widgets["mode"]
    combo.setCurrentIndex(combo.findData("run"))
    assert script.behaviour.mode == "run"
    form.widgets["offset"].value_changed.emit(
        type(script.behaviour.offset)(1.0, 2.0, 3.0)
    )
    assert tuple(script.behaviour.offset) == (1.0, 2.0, 3.0)
    assert script.fields["mode"] == "run"


def test_untouched_behaviour_field_is_not_overridden(window, tmp_path) -> None:
    _, script = _setup(window, tmp_path)
    form = _form(window)
    form.widgets["speed"].setValue(2.0)
    assert script.fields == {}


def test_reset_returns_default(window, tmp_path) -> None:
    _, script = _setup(window, tmp_path)
    form = _form(window)
    form.widgets["speed"].setValue(9.0)
    form._reset("speed")
    assert "speed" not in script.fields
    assert script.behaviour.speed == 2.0


def test_class_picker_writes_script_class(window, tmp_path) -> None:
    _, script = _setup(window, tmp_path, TWO)
    assert script.script_class is None
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    pickers = window.inspector.findChildren(QComboBox, "script_class_picker")
    assert len(pickers) == 1
    picker = pickers[0]
    assert [picker.itemText(i) for i in range(picker.count())] == ["First", "Second"]
    picker.setCurrentIndex(picker.findData("Second"))
    assert script.script_class == "Second"
    assert type(script.behaviour).__name__ == "Second"
    assert window.context.active_document.dirty
    assert script.to_dict()["script_class"] == "Second"


def test_single_class_has_no_picker(window, tmp_path) -> None:
    _setup(window, tmp_path)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not window.inspector.findChildren(QComboBox, "script_class_picker")


REQUIRES = '''
from engine.core.physics_components import RigidBodyComponent
from engine.scripting import Behaviour, require_component

@require_component(RigidBodyComponent)
class Pusher(Behaviour):
    pass
'''


def test_unmet_requirement_shows_red_label(window, tmp_path) -> None:
    from engine.core.physics_components import RigidBodyComponent

    cube, _ = _setup(window, tmp_path, REQUIRES)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    labels = window.inspector.findChildren(QLabel, "script_requires_label")
    assert len(labels) == 1
    assert "RigidBodyComponent" in labels[0].text()
    assert labels[0].property("role") == "error"

    cube.add_component(RigidBodyComponent())
    window.inspector.refresh()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not window.inspector.findChildren(QLabel, "script_requires_label")
