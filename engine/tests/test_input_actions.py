"""Действия ввода: привязки, кадры, оси, контексты, бэкенды."""

from __future__ import annotations

import json

import pytest

from engine.core.input import (
    Binding,
    InputAction,
    InputContext,
    InputContextStack,
    InputMap,
    InputState,
    ManualInputBackend,
    default_input_map,
    game_context,
    paused_context,
    ui_context,
)
from engine.core.math3d import Vector3


@pytest.fixture
def backend() -> ManualInputBackend:
    return ManualInputBackend()


def test_default_map_has_standard_actions() -> None:
    names = set(default_input_map().names())
    assert {"Jump", "Fire", "Move", "Look", "Interact", "Pause", "Menu"} <= names
    jump = default_input_map().get("Jump")
    assert Binding("keyboard", "space") in jump.bindings


def test_jump_reacts_to_both_bindings() -> None:
    state = InputState(input_map=InputMap([
        InputAction("Jump", "button", [Binding("keyboard", "space"),
                                       Binding("keyboard", "w")]),
    ]))
    backend = ManualInputBackend(state)
    backend.key_down("w")
    assert state.held("Jump")
    backend.key_up("w")
    backend.key_down("space")
    assert state.held("Jump")


def test_pressed_held_released_sequence(backend) -> None:
    s = backend.state
    assert not (s.pressed("Jump") or s.held("Jump") or s.released("Jump"))
    backend.key_down("space")
    assert (s.pressed("Jump"), s.held("Jump"), s.released("Jump")) == (True, True, False)
    backend.end_frame()
    assert (s.pressed("Jump"), s.held("Jump"), s.released("Jump")) == (False, True, False)
    backend.key_up("space")
    assert (s.pressed("Jump"), s.held("Jump"), s.released("Jump")) == (False, False, True)
    backend.end_frame()
    assert not (s.pressed("Jump") or s.held("Jump") or s.released("Jump"))


def test_tap_within_one_frame_is_not_lost(backend) -> None:
    s = backend.state
    backend.key_down("space")
    backend.key_up("space")
    assert s.pressed("Jump") and s.released("Jump") and not s.held("Jump")
    backend.end_frame()
    assert not s.pressed("Jump")


def test_axis2_wasd(backend) -> None:
    s = backend.state
    assert s.axis2("Move") == (0.0, 0.0)
    backend.key_down("d")
    assert s.axis2("Move") == (1.0, 0.0)
    backend.key_down("w")
    x, y = s.axis2("Move")
    assert x == pytest.approx(0.70710678) and y == pytest.approx(0.70710678)
    backend.key_up("d", "w")
    backend.hold(["a", "s"])
    x, y = s.axis2("Move")
    assert x < 0 and y < 0
    backend.hold(["a", "d"])
    assert s.axis2("Move") == (0.0, 0.0)  # противоположные клавиши гасятся


def test_axis_deadzone_and_vec2() -> None:
    action = InputAction("Throttle", "axis", [Binding("gamepad", "axis:left_y")], 0.2)
    state = InputState(input_map=InputMap([
        action,
        InputAction("Steer", "axis", [Binding("keyboard", "d"),
                                      Binding("keyboard", "a", scale=-1)]),
    ]))
    backend = ManualInputBackend(state)
    backend.gamepad_axis("left_y", 0.1)
    assert state.axis("Throttle") == 0.0
    backend.gamepad_axis("left_y", 0.6)
    assert state.axis("Throttle") == pytest.approx(0.6)
    backend.gamepad_axis("left_y", 3.0)
    assert state.axis("Throttle") == 1.0
    backend.key_down("a")
    assert state.vec2("Steer", "Throttle") == Vector3(-1.0, 1.0, 0.0)


def test_analog_trigger_acts_as_button(backend) -> None:
    s = backend.state
    backend.gamepad_axis("right_trigger", 0.3)
    assert not s.held("Fire")
    backend.gamepad_axis("right_trigger", 0.9)
    assert s.held("Fire")


def test_mouse_delta_resets_each_frame(backend) -> None:
    s = backend.state
    backend.mouse_move(10, 0)
    backend.mouse_move(10, 0)
    assert s.axis2("Look")[0] == pytest.approx(1.0)
    backend.end_frame()
    assert s.axis2("Look") == (0.0, 0.0)


def test_ui_context_blocks_game_actions(backend) -> None:
    s = backend.state
    backend.key_down("space")
    backend.mouse_button("left", True)
    assert s.held("Jump") and s.held("UIClick")
    s.contexts.push(ui_context())
    assert not s.held("Jump") and not s.pressed("Jump")
    assert s.axis2("Move") == (0.0, 0.0)
    assert s.pressed("UIClick") and s.held("UIClick")
    s.contexts.pop("ui")
    assert s.held("Jump")


def test_context_priority_and_names() -> None:
    stack = InputContextStack()
    stack.push(paused_context())
    stack.push(ui_context())
    assert stack.names() == ["ui", "paused", "game"]
    assert stack.allows("UIClick") and stack.allows("Pause") and not stack.allows("Jump")
    stack.pop("ui")
    assert stack.allows("Pause") and not stack.allows("Jump")
    stack.pop("paused")
    assert stack.allows("Jump")


def test_non_absorbing_context_passes_through() -> None:
    stack = InputContextStack([game_context(), InputContext("hud", 10, frozenset({"Menu"}))])
    assert stack.allows("Jump") and stack.allows("Menu")


def test_old_api_unaffected_by_contexts(backend) -> None:
    s = backend.state
    s.contexts.push(ui_context())
    backend.key_down("W")
    assert s.is_key_down("w") and s.is_key_pressed("W") and "w" in s
    s.pressed_keys.clear()
    s.press("space")
    assert s.is_key_down("space")
    s.release("space")
    assert not s.is_key_down("space")


def test_direct_pressed_keys_mutation_still_works() -> None:
    s = InputState()
    s.pressed_keys.update({"space"})  # как делает sync_input плеера
    assert s.held("Jump") and s.pressed("Jump")
    s.end_frame()
    assert s.held("Jump") and not s.pressed("Jump")


def test_unknown_action_is_neutral(backend, caplog) -> None:
    s = backend.state
    assert not s.held("Nope") and s.axis("Nope") == 0.0 and s.axis2("Nope") == (0, 0)
    assert not s.pressed("Nope") and not s.released("Nope")


def test_two_states_with_same_events_agree() -> None:
    def run(state: InputState) -> list[object]:
        backend = ManualInputBackend(state)
        out: list[object] = []
        for step in range(4):
            if step == 0:
                backend.key_down("space", "d")
            if step == 2:
                backend.key_up("space")
            out.append((state.pressed("Jump"), state.held("Jump"),
                        state.released("Jump"), state.axis2("Move")))
            backend.end_frame()
        return out

    assert run(InputState()) == run(InputState(input_map=default_input_map()))


def test_map_json_roundtrip(tmp_path) -> None:
    original = default_input_map()
    path = tmp_path / "input.json"
    original.save(path)
    loaded = InputMap.load(path)
    assert loaded.to_dict() == original.to_dict()
    assert json.loads(path.read_text("utf-8"))["version"] == 1


def test_user_file_overrides_defaults(tmp_path) -> None:
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"actions": {"Jump": {"kind": "button", "bindings": [
        {"device": "keyboard", "code": "enter"}]}}}), "utf-8")
    loaded = InputMap.load(path)
    assert loaded.get("Jump").bindings == [Binding("keyboard", "enter")]
    assert "Fire" in loaded  # остальное - из дефолта


@pytest.mark.parametrize("bad", [
    {"device": "joystick", "code": "x"},
    {"device": "mouse", "code": "side"},
    {"device": "gamepad", "code": "south"},
    {"device": "keyboard", "code": ""},
    {"device": "keyboard", "code": "a", "axis": 2},
])
def test_invalid_bindings_rejected(bad) -> None:
    with pytest.raises(ValueError):
        Binding.from_dict(bad)


def test_invalid_map_file(tmp_path) -> None:
    path = tmp_path / "x.json"
    path.write_text("{not json", "utf-8")
    with pytest.raises(ValueError):
        InputMap.load(path)
    path.write_text("{}", "utf-8")
    with pytest.raises(ValueError):
        InputMap.load(path)
