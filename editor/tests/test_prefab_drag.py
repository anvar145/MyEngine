"""Перетаскивание префабов из панели ассетов в иерархию и вьюпорт."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QMimeData, QPointF, Qt
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QMessageBox

from editor.mime import PREFAB_MIME
from engine.core.math3d import Vector3
from engine.core.prefab_components import PrefabInstance


@pytest.fixture
def prefab_path(window, tmp_path):
    window.context.create_object("Tower")
    window.context.selection.select(window.context.scene.objects[-1])
    path = tmp_path / "tower.prefab.json"
    window.hierarchy.save_selected_as_prefab(str(path))
    window.context.selection.clear()
    return path


def test_assets_model_offers_prefab_mime(window, prefab_path) -> None:
    model = window.assets.fs_model
    index = model.index(str(prefab_path))
    data = model.mimeData([index])
    assert Path(bytes(data.data(PREFAB_MIME)).decode("utf-8")) == prefab_path


def test_assets_model_plain_file_has_no_prefab_mime(window, tmp_path) -> None:
    other = tmp_path / "a.txt"
    other.write_text("x", encoding="utf-8")
    model = window.assets.fs_model
    data = model.mimeData([model.index(str(other))])
    assert not data.hasFormat(PREFAB_MIME)


def test_drop_on_object_makes_child(window, prefab_path) -> None:
    parent = window.context.create_object("Holder")
    instance = window.hierarchy.drop_prefab(str(prefab_path), parent)
    assert instance is not None and instance.parent is parent
    assert instance.get_component(PrefabInstance) is not None
    assert window.context.selection.primary is instance


def test_drop_event_on_tree_creates_root_instance(window, prefab_path) -> None:
    tree = window.hierarchy.tree
    tree.resize(300, 600)
    before = len(window.context.scene.objects)
    mime = QMimeData()
    mime.setData(PREFAB_MIME, str(prefab_path).encode("utf-8"))
    event = QDropEvent(
        QPointF(2, 500),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    tree.dropEvent(event)
    assert len(window.context.scene.objects) > before
    assert window.context.selection.primary.parent is None


def test_drop_broken_prefab_warns(window, tmp_path, monkeypatch) -> None:
    shown: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *a, **k: shown.append(str(a[2]))
    )
    bad = tmp_path / "bad.prefab.json"
    bad.write_text("{не json", encoding="utf-8")
    assert window.hierarchy.drop_prefab(str(bad)) is None
    assert shown


def test_viewport_drop_places_on_ground_plane(window, prefab_path) -> None:
    viewport = window.viewport
    point = viewport.ground_point_at(10, 10)
    assert isinstance(point, Vector3)
    root = viewport.drop_prefab(str(prefab_path), 10, 10)
    assert root is not None
    assert root.get_component(PrefabInstance) is not None
    assert window.context.selection.primary is root


