"""Shift+F5 в окне плеера останавливает игру, а F5 остаётся быстрым сохранением (B5)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from engine.core.input.keys import KEY_F5, KEY_SHIFT_F5, MOD_SHIFT
from engine.player.app import PlayerApp, parse_args
from engine.player.window import GlfwWindow


def _app(tmp_path: Path) -> PlayerApp:
    project = tmp_path / "game"
    project.mkdir(exist_ok=True)
    app = PlayerApp(parse_args(["--project", str(project), "--headless"]))
    app.window = SimpleNamespace(key_events=[])  # type: ignore[assignment]
    return app


def test_window_maps_shift_f5_to_stop_event() -> None:
    window = GlfwWindow()
    window._on_key(None, KEY_F5, 0, 1, MOD_SHIFT)
    window._on_key(None, KEY_F5, 0, 1, 0)
    assert window.key_events == [KEY_SHIFT_F5, KEY_F5]


def test_shift_f5_requests_stop_without_quick_save(tmp_path: Path) -> None:
    app = _app(tmp_path)
    app.window.key_events = [KEY_SHIFT_F5]
    app._pump_keyboard()
    assert app.stop_requested is True
    assert app._pending_quick == []


def test_plain_f5_still_quick_saves(tmp_path: Path) -> None:
    app = _app(tmp_path)
    app.window.key_events = [KEY_F5]
    app._pump_keyboard()
    assert app.stop_requested is False
    assert app._pending_quick == ["save"]
