"""Unpack, подсказка и команды экземпляра в иерархии."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from editor.models.scene_tree_model import prefab_display_name
from engine.core.math3d import Vector3
from engine.core.prefab_components import PrefabInstance
from engine.prefab import get_overrides


@pytest.fixture
def instance(window, tmp_path):
    root = window.context.create_object("Tower")
    window.context.scene.create_game_object("Top", parent=root)
    window.context.selection.select(root)
    result = window.hierarchy.save_selected_as_prefab(str(tmp_path / "tower.prefab.json"))
    window.context.process_pending()
    return result


def _answer(monkeypatch, button) -> list[str]:
    asked: list[str] = []

    def question(*args, **kwargs):
        asked.append(str(args[2]))
        return button

    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))
    return asked


def test_unpack_after_confirmation(window, instance, monkeypatch) -> None:
    asked = _answer(monkeypatch, QMessageBox.StandardButton.Yes)
    uids = {o.uid for o in window.context.scene.objects}
    child = instance.children[0]
    child.transform.position = Vector3(0, 7, 0)
    window.context.selection.select(child)  # член экземпляра
    result = window.hierarchy.unpack_selected()
    assert result is instance and asked and "Tower" in asked[0]
    assert instance.get_component(PrefabInstance) is None
    assert {o.uid for o in window.context.scene.objects} == uids
    assert child.transform.position.y == 7
    assert window.context.selection.primary is instance


def test_unpack_cancelled_changes_nothing(window, instance, monkeypatch) -> None:
    _answer(monkeypatch, QMessageBox.StandardButton.No)
    assert window.hierarchy.unpack_selected() is None
    assert instance.get_component(PrefabInstance) is not None


def test_unpack_is_undoable(window, instance, monkeypatch) -> None:
    _answer(monkeypatch, QMessageBox.StandardButton.Yes)
    window.hierarchy.unpack_selected()
    window.context.process_pending()
    assert window.context.undo()
    restored = window.context.scene.find_by_uid(instance.uid)
    assert restored is not None and restored.get_component(PrefabInstance) is not None


def test_tooltip_shows_prefab_name(window, instance) -> None:
    model = window.hierarchy.model
    tip = model.data(model.index_for(instance), Qt.ItemDataRole.ToolTipRole)
    assert "prefab: tower" in tip
    plain = window.context.create_object("Plain")
    assert "prefab:" not in model.data(model.index_for(plain), Qt.ItemDataRole.ToolTipRole)


def test_prefab_display_name() -> None:
    marker = PrefabInstance("g", r"Assets\Prefabs\Enemy.prefab.json")
    assert prefab_display_name(marker) == "Enemy"


def test_revert_all(window, instance) -> None:
    instance.children[0].transform.position = Vector3(0, 9, 0)
    window.context.selection.select(instance)
    assert get_overrides(instance, assets=window.context.asset_db)
    assert window.hierarchy.revert_all_overrides() >= 1
    assert not get_overrides(instance, assets=window.context.asset_db)


def test_apply_all_updates_prefab_and_clears_overrides(window, instance) -> None:
    instance.children[0].transform.position = Vector3(0, 9, 0)
    window.context.selection.select(instance)
    assert window.hierarchy.apply_all_to_prefab() >= 1
    current = window.context.scene.find_by_uid(instance.uid)
    assert not get_overrides(current, assets=window.context.asset_db)
    assert current.children[0].transform.position.y == 9


def test_open_prefab_file_uses_desktop_services(window, instance, monkeypatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(
        "editor.panels.hierarchy.QDesktopServices.openUrl",
        staticmethod(lambda url: opened.append(url.toLocalFile()) or True),
    )
    window.context.selection.select(instance)
    assert window.hierarchy.open_prefab_file()
    assert opened and opened[0].endswith("tower.prefab.json")
