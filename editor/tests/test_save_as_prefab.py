"""«Сохранить как префаб» из иерархии."""

from __future__ import annotations

import json

import pytest
from PySide6.QtWidgets import QMessageBox

from engine.core.prefab_components import PrefabInstance


@pytest.fixture
def warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    log: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args, **kw: log.append(str(args[2]))
    )
    return log


def _make(window):
    root = window.context.create_object("Tower")
    child = window.context.scene.create_game_object("Top", parent=root)
    window.context.selection.select(root)
    return root, child


def test_save_creates_file_and_instance(window, tmp_path, warnings) -> None:
    root, _child = _make(window)
    uid = root.uid
    path = tmp_path / "tower.prefab.json"
    instance = window.hierarchy.save_selected_as_prefab(str(path))
    assert instance is not None and not warnings
    assert json.loads(path.read_text(encoding="utf-8"))["name"] == "Tower"
    assert instance.get_component(PrefabInstance) is not None
    assert window.context.selection.primary is instance
    assert root.is_destroyed and instance.uid != uid
    assert [c.name for c in instance.children] == ["Top"]


def test_multiple_selection_is_refused(window, tmp_path, warnings) -> None:
    root, child = _make(window)
    window.context.selection.select_many([root, child])
    path = tmp_path / "x.prefab.json"
    assert window.hierarchy.save_selected_as_prefab(str(path)) is None
    assert warnings and not path.exists()


def test_instance_member_is_refused(window, tmp_path, warnings) -> None:
    _make(window)
    instance = window.hierarchy.save_selected_as_prefab(str(tmp_path / "a.prefab.json"))
    window.context.selection.select(instance.children[0])
    assert window.hierarchy.save_selected_as_prefab(str(tmp_path / "b.prefab.json")) is None
    assert warnings and not (tmp_path / "b.prefab.json").exists()


def test_undo_restores_original_object(window, tmp_path, warnings) -> None:
    _make(window)
    window.context.process_pending()
    window.hierarchy.save_selected_as_prefab(str(tmp_path / "t.prefab.json"))
    assert window.context.undo()
    names = sorted(o.name for o in window.context.scene.objects if o.name in ("Tower", "Top"))
    assert names == ["Top", "Tower"]
    assert not any(o.get_component(PrefabInstance) for o in window.context.scene.objects)
