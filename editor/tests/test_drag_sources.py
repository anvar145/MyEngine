"""Источники перетаскивания: иерархия отдаёт id объекта, ассеты — uri-list."""

from __future__ import annotations

from PySide6.QtWidgets import QAbstractItemView

from editor.mime import OBJECT_MIME


def test_hierarchy_mime_has_object_id(window) -> None:
    cube = window.create_primitive("cube")
    model = window.hierarchy.model
    mime = model.mimeData([model.index_for(cube)])
    assert OBJECT_MIME in model.mimeTypes()
    assert bytes(mime.data(OBJECT_MIME)).decode() == str(cube.id)


def test_hierarchy_drags_and_accepts_drops(window) -> None:
    tree = window.hierarchy.tree
    assert tree.dragEnabled() and tree.acceptDrops()
    # Перестановка идёт через собственный dropEvent (uid-MIME), не InternalMove.
    assert tree.dragDropMode() == QAbstractItemView.DragDropMode.DragDrop


def test_assets_panel_drags_files(window) -> None:
    tree = window.assets.tree if hasattr(window, "assets") else None
    assert tree is not None and tree.dragEnabled()


def test_double_click_on_py_opens_script_editor(window, tmp_path) -> None:
    path = tmp_path / "x.py"
    path.write_text("a = 1\n", "utf-8")
    window.assets.script_activated.emit(str(path))
    assert window.script_editor.isVisible()
    assert window.script_editor._editor_for_path(path) is not None
