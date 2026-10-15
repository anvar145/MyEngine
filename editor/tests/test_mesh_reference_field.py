"""Поле меша MeshRenderer и поля скриптов: ReferenceField для ``AssetRef``."""

from __future__ import annotations

from PySide6.QtCore import QCoreApplication, QEvent, QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDropEvent

from editor.panels.script_fields import ReferenceField, make_asset_ref
from engine.core.components import MeshRenderer
from engine.core.resources import AssetRef


def _mesh_field(window) -> ReferenceField:
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    fields = window.inspector.findChildren(ReferenceField)
    assert len(fields) == 1
    return fields[0]


def _drop(widget, path) -> None:
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    event = QDropEvent(
        QPointF(2, 2), Qt.DropAction.CopyAction, mime,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )
    widget.dragEnterEvent(event)
    widget.dropEvent(event)


def test_primitive_shown_not_broken(window) -> None:
    field = _mesh_field(window)
    assert field.label.text() == "cube"
    assert field.label.property("role") != "error"


def test_drop_model_sets_ref_and_clear_resets(window, tmp_path) -> None:
    cube = window.context.selection.primary
    model = tmp_path / "x.obj"
    model.write_text("v 0 0 0\n", "utf-8")
    _drop(_mesh_field(window), model)
    ref = cube.get_component(MeshRenderer).mesh_ref
    assert ref.path is not None and ref.path.endswith("x.obj")
    field = _mesh_field(window)
    assert field.label.text() == "x.obj"
    field.cleared.emit()
    assert cube.get_component(MeshRenderer).mesh_ref == AssetRef(path="cube")


def test_non_model_drop_is_ignored(window, tmp_path) -> None:
    cube = window.context.selection.primary
    other = tmp_path / "a.txt"
    other.write_text("x", "utf-8")
    _drop(_mesh_field(window), other)
    assert cube.get_component(MeshRenderer).mesh_ref == AssetRef(path="cube")


def test_broken_ref_is_red_with_guid_tooltip(window) -> None:
    cube = window.context.selection.primary
    guid = "ab" * 16
    cube.get_component(MeshRenderer).mesh_ref = AssetRef(guid=guid, path="Gone/x.obj")
    window.inspector.refresh()
    field = _mesh_field(window)
    assert field.label.text() == "x.obj"
    assert field.label.property("role") == "error"
    assert guid in field.toolTip()


def test_make_asset_ref_without_db() -> None:
    assert make_asset_ref(None, "a/b.png") == AssetRef(path="a/b.png")
