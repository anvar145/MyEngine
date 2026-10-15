"""Клавиатура плеера: коды GLFW → имена скриптов."""

from __future__ import annotations

from engine.player.keymap import KEY_F2, glfw_key_name, sync_input
from engine.scripting.script_component import InputState


def test_names():
    assert glfw_key_name(87) == "w"
    assert glfw_key_name(32) == "space"
    assert glfw_key_name(340) == "shift"
    assert glfw_key_name(KEY_F2) == "f2"
    assert glfw_key_name(49) == "1"
    assert glfw_key_name(9999) is None


def test_sync_replaces_state_and_adds_extra():
    state = InputState(["a"])
    sync_input(state, [87, 32], ["Shift"])
    assert state.pressed_keys == {"w", "space", "shift"}
    sync_input(state, [])
    assert state.pressed_keys == set()
