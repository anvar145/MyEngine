"""Панель «Настройки ввода»: правка привязок, сохранение отличий, диалог."""

from __future__ import annotations

import json

import pytest
from PySide6.QtCore import Qt

from editor.context import EditorContext
from editor.panels.input_settings import (
    BindingCaptureDialog,
    InputSettingsPanel,
    binding_text,
)
from engine.core.input import Binding


@pytest.fixture
def panel(qapp, tmp_path):
    context = EditorContext(project_root=tmp_path, language="en")
    return InputSettingsPanel(context)


def _top(panel, prefix: str):
    tree = panel.tree
    items = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
    return next(i for i in items if i.text(0).startswith(prefix))


def test_tree_lists_actions_and_bindings(panel) -> None:
    jump = _top(panel, "Jump")
    assert jump.childCount() == len(panel.input_map.get("Jump").bindings)
    assert binding_text(Binding("keyboard", "w", -1.0, 1)) == "keyboard: w  x-1  (Y)"


def test_edit_via_dialog_result_and_cancel(panel, monkeypatch) -> None:
    monkeypatch.setattr(panel, "ask_binding", lambda: ("keyboard", "enter"))
    panel.edit_binding("Jump", 0)
    assert panel.input_map.get("Jump").bindings[0] == Binding("keyboard", "enter")
    assert _top(panel, "Jump").text(0).endswith("*")
    monkeypatch.setattr(panel, "ask_binding", lambda: None)
    panel.edit_binding("Jump", 0)
    assert panel.input_map.get("Jump").bindings[0] == Binding("keyboard", "enter")


def test_add_remove_reset(panel, monkeypatch) -> None:
    count = len(panel.input_map.get("Fire").bindings)
    monkeypatch.setattr(panel, "ask_binding", lambda: ("mouse", "middle"))
    panel.add_binding("Fire")
    assert len(panel.input_map.get("Fire").bindings) == count + 1
    panel.remove_binding("Fire", 0)
    assert not panel.input_map.is_default("Fire")
    panel.reset_action("Fire")
    assert panel.input_map.is_default("Fire")
    panel.remove_binding("Jump", 0)
    panel.reset_all()
    assert panel.input_map.is_default("Jump")


def test_save_writes_only_changes_and_load_restores(panel, monkeypatch) -> None:
    monkeypatch.setattr(panel, "ask_binding", lambda: ("keyboard", "enter"))
    panel.edit_binding("Jump", 0)
    panel.save()
    data = json.loads(panel.settings_file.read_text("utf-8"))
    assert list(data["actions"]) == ["Jump"]
    panel.reset_all()
    panel.load()
    assert panel.input_map.get("Jump").bindings[0] == Binding("keyboard", "enter")


def test_capture_dialog_key_mouse_escape(qapp) -> None:
    dialog = BindingCaptureDialog("t", "h", "c")
    dialog.capture_key(Qt.Key_Q)
    assert dialog.result_binding == ("keyboard", "q")
    other = BindingCaptureDialog("t", "h", "c")
    other.capture_key(Qt.Key_Escape)
    assert other.result_binding is None
    third = BindingCaptureDialog("t", "h", "c")
    third.capture_mouse("right")
    assert third.result_binding == ("mouse", "right")
