"""Undo/redo в главном окне: действия меню, горячие клавиши, панели."""

from __future__ import annotations

from PySide6.QtGui import QKeySequence

from engine.core.math3d import Vector3


def _commit(window) -> None:
    """Команды пишутся сразу: фиксировать нечего."""


def test_actions_disabled_without_history(window) -> None:
    assert not window.action_undo.isEnabled()
    assert not window.action_redo.isEnabled()


def test_shortcuts(window) -> None:
    assert window.action_undo.shortcut() == QKeySequence("Ctrl+Z")
    redo = {s.toString() for s in window.action_redo.shortcuts()}
    assert {"Ctrl+Y", "Ctrl+Shift+Z"} <= redo


def test_create_undo_redo_updates_panels_and_menu(window) -> None:
    count = len(window.context.scene)
    can_before = window.action_undo.isEnabled()
    window.create_primitive("cube")
    _commit(window)
    assert window.action_undo.isEnabled()
    assert "Cube" in window.action_undo.text()

    window.undo()
    assert len(window.context.scene) == count
    assert window.action_redo.isEnabled()
    assert window.action_undo.isEnabled() == can_before

    window.redo()
    assert len(window.context.scene) == count + 1
    assert window.context.selection.primary.name == "Cube"


def test_inspector_transform_edit_is_undoable(window) -> None:
    cube = window.create_primitive("cube")
    _commit(window)
    before = cube.transform.position.to_tuple()
    index = window.context.scene.objects.index(cube)
    window.context.set_field(cube, cube.transform, "position", Vector3(5.0, 0.0, 0.0))
    _commit(window)
    window.undo()
    restored = window.context.scene.objects[index]
    assert restored.transform.position.to_tuple() == before


def test_label_translates(window) -> None:
    window.create_primitive("cube")
    _commit(window)
    window.context.translator.set_language("en")
    assert window.action_undo.text().startswith("Undo")
    window.context.translator.set_language("ru")
    assert not window.action_undo.text().startswith("Undo")
