"""QtInputBackend: клавиши Qt → InputState только при enabled."""

from __future__ import annotations

from PySide6.QtCore import Qt

from editor.play_mode_qt_input import QtInputBackend, qt_key_name
from engine.core.input import InputState


def test_qt_key_names() -> None:
    assert qt_key_name(Qt.Key_W) == "w" and qt_key_name(Qt.Key_Space) == "space"
    assert qt_key_name(Qt.Key_F2) == "f2" and qt_key_name(Qt.Key_Home) is None


def test_ignored_until_enabled_and_autorepeat() -> None:
    state = InputState()
    backend = QtInputBackend(state)
    backend.handle_key(Qt.Key_Space, True)
    assert not state.held("Jump")
    backend.set_enabled(True)
    backend.handle_key(Qt.Key_Space, True)
    backend.handle_key(Qt.Key_Space, True, auto_repeat=True)
    assert state.pressed("Jump")
    backend.set_enabled(False)
    assert not state.held("Jump")
