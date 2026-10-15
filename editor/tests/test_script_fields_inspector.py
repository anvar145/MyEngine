"""Поля скрипта в инспекторе: показ, правка, сброс, undo/redo, drop."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QCoreApplication, QEvent, QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QCheckBox, QLineEdit

from editor.mime import OBJECT_MIME
from editor.panels.script_fields import ReferenceField, ScriptFieldsForm
from editor.widgets import FocusDoubleSpinBox, FocusSpinBox
from engine.scripting import ScriptComponent

SOURCE = """speed = 3.0
count = 5
label = "hi"
on = True
_hidden = 1
target: GameObject = None
icon: Asset = None
"""


def _commit(window) -> None:
    """Команды пишутся сразу: фиксировать нечего."""


def _setup(window, tmp_path: Path):
    path = tmp_path / "s.py"
    path.write_text(SOURCE, "utf-8")
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


def _drop(widget, mime: QMimeData) -> None:
    event = QDropEvent(
        QPointF(2, 2), Qt.DropAction.CopyAction, mime,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )
    widget.dragEnterEvent(event)
    widget.dropEvent(event)


def test_fields_shown_for_public_literals(window, tmp_path) -> None:
    _setup(window, tmp_path)
    form = _form(window)
    assert form.field_names == ["speed", "count", "label", "on", "target", "icon"]
    assert isinstance(form.widgets["speed"], FocusDoubleSpinBox)
    assert form.widgets["speed"].value() == 3.0
    assert isinstance(form.widgets["count"], FocusSpinBox)
    assert isinstance(form.widgets["label"], QLineEdit)
    assert isinstance(form.widgets["on"], QCheckBox)
    assert isinstance(form.widgets["target"], ReferenceField)


def test_edit_writes_override_and_undo_redo(window, tmp_path) -> None:
    _, script = _setup(window, tmp_path)
    form = _form(window)
    form.widgets["speed"].setValue(9.5)
    form.widgets["count"].setValue(7)
    form.widgets["label"].setText("boss")
    form.widgets["label"].editingFinished.emit()
    form.widgets["on"].setChecked(False)
    assert script.fields == {"speed": 9.5, "count": 7, "label": "boss", "on": False}
    assert script.module.speed == 9.5
    _commit(window)

    # Каждое поле - своя запись истории: четыре правки, четыре откатa.
    for _ in range(4):
        window.undo()
    restored = window.context.selection.primary.get_component(ScriptComponent)
    assert restored.fields == {}
    for _ in range(4):
        window.redo()
    redone = window.context.selection.primary.get_component(ScriptComponent)
    assert redone.fields["speed"] == 9.5


def test_untouched_field_is_not_overridden(window, tmp_path) -> None:
    _, script = _setup(window, tmp_path)
    form = _form(window)
    form.widgets["label"].editingFinished.emit()
    assert script.fields == {}


def test_reset_returns_code_value(window, tmp_path, qapp) -> None:
    _, script = _setup(window, tmp_path)
    form = _form(window)
    form.widgets["speed"].setValue(9.5)
    assert "speed" in script.fields
    form._reset("speed")
    assert "speed" not in script.fields
    assert script.module.speed == 3.0


def test_drop_object_and_asset(window, tmp_path) -> None:
    cube, script = _setup(window, tmp_path)
    other = window.create_primitive("sphere")
    window.context.selection.select(cube)
    window.inspector.refresh()
    form = _form(window)

    mime = QMimeData()
    mime.setData(OBJECT_MIME, str(other.id).encode("ascii"))
    _drop(form.widgets["target"], mime)
    assert script.fields["target"] is other
    assert script.module.target is other
    assert other.name in form.widgets["target"].label.text()

    asset = tmp_path / "pic.png"
    asset.write_bytes(b"x")
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(asset))])
    _drop(form.widgets["icon"], mime)
    assert Path(script.module.icon.path) == asset

    # объект не принимается в поле файла и наоборот
    bad = QMimeData()
    bad.setData(OBJECT_MIME, b"1")
    before = dict(script.fields)
    _drop(form.widgets["icon"], bad)
    assert script.fields == before

    form.widgets["target"].clear_button.click()
    assert script.module.target is None
    assert "target" not in script.fields


def test_field_removed_from_code_disappears_after_refresh(window, tmp_path) -> None:
    _, script = _setup(window, tmp_path)
    Path(script.path).write_text("speed = 3.0\n", "utf-8")
    script.reload()
    window.inspector.refresh()
    assert _form(window).field_names == ["speed"]


def test_narrow_panel_does_not_overflow(window, tmp_path) -> None:
    _setup(window, tmp_path)
    form = _form(window)
    form.widgets["label"].setText("очень-очень-длинное-значение-" * 5)
    assert form.minimumSizeHint().width() <= 260


def test_prefab_field_accepts_only_prefab_files(window, tmp_path) -> None:
    path = tmp_path / "p.py"
    path.write_text("bullet: Prefab = None\n", "utf-8")
    cube = window.create_primitive("cube")
    script = cube.add_component(
        ScriptComponent(path, runtime=window.context.script_runtime)
    )
    _commit(window)
    window.context.selection.select(cube)
    window.inspector.refresh()
    form = _form(window)
    assert form.field_names == ["bullet"]

    other = tmp_path / "pic.png"
    other.write_bytes(b"x")
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(other))])
    _drop(form.widgets["bullet"], mime)
    assert "bullet" not in script.fields

    good = tmp_path / "bullet.prefab.json"
    good.write_text("{}", "utf-8")
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(good))])
    _drop(form.widgets["bullet"], mime)
    assert Path(script.fields["bullet"].path) == good


def test_same_field_edits_merge_into_one_undo(window, tmp_path) -> None:
    _setup(window, tmp_path)
    form = _form(window)
    steps = len(window.context.active_document.commands._undo)  # noqa: SLF001
    for value in (4.0, 5.0, 6.0, 7.0):
        form.widgets["speed"].setValue(value)
    assert len(window.context.active_document.commands._undo) == steps + 1  # noqa: SLF001
    window.undo()
    restored = window.context.selection.primary.get_component(ScriptComponent)
    assert "speed" not in restored.fields
    window.redo()
    redone = window.context.selection.primary.get_component(ScriptComponent)
    assert redone.fields["speed"] == 7.0
