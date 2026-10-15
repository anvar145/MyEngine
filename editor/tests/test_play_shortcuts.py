"""Горячие клавиши Play Mode срабатывают в главном окне."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest


@pytest.mark.parametrize(
    ("key", "modifier", "name"),
    [
        (Qt.Key.Key_F5, Qt.KeyboardModifier.NoModifier, "action_play"),
        (Qt.Key.Key_F5, Qt.KeyboardModifier.ShiftModifier, "action_play_stop"),
        (Qt.Key.Key_F6, Qt.KeyboardModifier.NoModifier, "action_play_pause"),
        (Qt.Key.Key_F7, Qt.KeyboardModifier.NoModifier, "action_play_step"),
    ],
)
def test_play_shortcut_triggers_action(window, key, modifier, name) -> None:
    window.show()
    window.activateWindow()
    QTest.qWaitForWindowActive(window, 2000)
    window.setFocus()
    action = getattr(window, name)
    hits: list[bool] = []
    action.triggered.disconnect()
    action.triggered.connect(lambda *_: hits.append(True))
    QTest.keyClick(window, key, modifier)
    assert hits == [True]


def test_control_shortcut_works(window) -> None:
    window.show()
    window.activateWindow()
    QTest.qWaitForWindowActive(window, 2000)
    hits: list[bool] = []
    window.action_deselect.triggered.connect(lambda *_: hits.append(True))
    QTest.keyClick(
        window,
        Qt.Key.Key_A,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
    )
    assert hits == [True]



def test_script_editor_shortcuts_are_scoped(window) -> None:
    editor = window.script_editor
    for action in (editor.action_run, editor.action_stop, editor.action_reload):
        assert (
            action.shortcutContext() == Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
