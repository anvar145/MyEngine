"""Бэкенды ввода: GLFW, Manual, интеграция с плеером (--hold-keys, --input-map)."""

from __future__ import annotations

import json

from engine.core.input import InputState, ManualInputBackend, ui_context
from engine.core.input.keys import glfw_key_name
from engine.player.input_backend import GlfwInputBackend


def test_glfw_backend_converts_keys_and_mouse() -> None:
    state = InputState()
    backend = GlfwInputBackend(state)
    backend.on_key(87, True)  # W
    backend.on_key(32, True)  # space
    backend.on_key(999, True)  # неизвестная - игнор
    assert state.is_key_down("w") and state.pressed("Jump")
    backend.on_key(87, False)
    assert not state.is_key_down("w")
    backend.on_mouse_button("left", True)
    assert state.held("UIClick")
    backend.on_cursor(100, 100)  # первая точка - без дельты
    backend.on_cursor(110, 100)
    assert state.mouse_delta == (10.0, 0.0)
    backend.on_cursor_leave()
    backend.on_cursor(500, 500)
    assert state.mouse_delta == (10.0, 0.0)


def test_pressed_true_only_in_first_frame() -> None:
    state = InputState()
    backend = ManualInputBackend(state)
    backend.feed_key("space", down=True)
    assert state.pressed("Jump")
    state.end_frame()
    assert not state.pressed("Jump") and state.held("Jump")
    backend.feed_key("space", down=False)
    assert state.released("Jump")


def test_manual_hold_is_additive_and_reinjected() -> None:
    state = InputState()
    backend = ManualInputBackend(state)
    state.key_down("space")
    backend.hold(["w"])
    assert state.is_key_down("w") and state.is_key_down("space")
    state.pressed_keys.discard("w")  # сняли снаружи
    backend.hold(["w"])
    assert state.is_key_down("w")
    backend.hold([])
    assert not state.is_key_down("w") and state.is_key_down("space")


def test_glfw_names_still_resolve() -> None:
    assert glfw_key_name(65) == "a" and glfw_key_name(291) == "f2"


def test_pause_passes_through_ui_and_paused_contexts() -> None:
    state = InputState()
    state.key_down("escape")
    state.contexts.push(ui_context())
    assert state.contexts.allows("Pause")
    state.contexts.reset()
    assert state.contexts.names() == ["game"]


def test_player_loads_input_map(tmp_path) -> None:
    from engine.player.app import PlayerApp, parse_args

    path = tmp_path / "map.json"
    path.write_text(json.dumps({"version": 1, "actions": {"Jump": {
        "kind": "button", "bindings": [{"device": "keyboard", "code": "enter"}]}}}),
        "utf-8")
    argv = ["--project", str(tmp_path), "--headless", "--input-map", str(path)]
    args = parse_args(argv)
    app = PlayerApp(args)
    app.manual_input.feed_key("enter")
    assert app.input.held("Jump")
    app.manual_input.feed_key("enter", down=False)
    app.manual_input.feed_key("space")
    assert not app.input.held("Jump")
