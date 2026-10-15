"""Ремаппинг: begin_rebind, колбэки, save/load только отличий, round-trip."""

from __future__ import annotations

import json

import pytest

from engine.core.input import Binding, InputMap, InputState, default_input_map


def test_begin_rebind_then_key_updates_map() -> None:
    state = InputState()
    state.begin_rebind("Jump")
    assert state.rebinding == ("Jump", 0)
    state.key_down("enter")
    assert state.rebinding is None
    assert state.input_map.get("Jump").bindings[0] == Binding("keyboard", "enter")
    assert "enter" not in state.pressed_keys  # нажатие поглощено
    state.key_down("enter")
    assert state.pressed("Jump")


def test_rebind_keeps_axis_direction() -> None:
    state = InputState()
    move = state.input_map.get("Move")
    index = next(i for i, b in enumerate(move.bindings) if b.code == "w")
    old = move.bindings[index]
    state.begin_rebind("Move", index)
    state.key_down("i")
    new = move.bindings[index]
    assert new.code == "i" and new.scale == old.scale and new.axis == old.axis


def test_escape_cancels_and_mouse_gamepad_capture() -> None:
    state = InputState()
    before = list(state.input_map.get("Jump").bindings)
    state.begin_rebind("Jump")
    state.key_down("escape")
    assert state.rebinding is None
    assert state.input_map.get("Jump").bindings == before
    state.begin_rebind("Jump", len(before))  # новая привязка
    state.mouse_button("right", True)
    assert state.input_map.get("Jump").bindings[-1] == Binding("mouse", "right")
    state.begin_rebind("Jump", 0)
    state.gamepad_button("south", True)
    expected = Binding("gamepad", "button:south")
    assert state.input_map.get("Jump").bindings[0] == expected
    state.begin_rebind("Jump")
    state.cancel_rebind()
    assert state.rebinding is None


def test_on_rebind_callback_and_errors() -> None:
    state = InputState()
    seen: list[tuple[str, int, Binding]] = []
    state.on_rebind("Jump", lambda a, i, b: seen.append((a, i, b)))
    state.begin_rebind("Fire")
    state.key_down("f")
    assert seen == []
    state.begin_rebind("Jump")
    state.key_down("j")
    assert seen == [("Jump", 0, Binding("keyboard", "j"))]
    with pytest.raises(KeyError):
        state.begin_rebind("Nope")
    with pytest.raises(IndexError):
        state.begin_rebind("Jump", 99)


def test_map_api_add_unbind() -> None:
    m = default_input_map()
    count = len(m.get("Jump").bindings)
    m.add_binding("Jump", Binding("keyboard", "j"))
    assert len(m.get("Jump").bindings) == count + 1
    assert m.unbind("Jump", count) == Binding("keyboard", "j")
    assert m.is_default("Jump")


def test_save_only_changes_and_roundtrip(tmp_path) -> None:
    state = InputState()
    state.settings_path = tmp_path / "Settings" / "input.json"
    state.begin_rebind("Jump")
    state.key_down("enter")
    state.save_bindings()
    data = json.loads(state.settings_path.read_text("utf-8"))
    assert data["version"] == 1 and list(data["actions"]) == ["Jump"]
    other = InputState()
    other.load_bindings(state.settings_path)
    assert other.input_map.to_dict() == state.input_map.to_dict()
    assert not other.input_map.is_default("Jump")
    assert other.input_map.is_default("Fire")


def test_reset_makes_file_empty_diff(tmp_path) -> None:
    m = default_input_map()
    m.rebind("Jump", 0, Binding("keyboard", "z"))
    m.reset_all()
    path = tmp_path / "i.json"
    m.save(path, only_changes=True)
    assert json.loads(path.read_text("utf-8"))["actions"] == {}
    assert InputMap.load(path).to_dict() == default_input_map().to_dict()


def test_paths_required() -> None:
    with pytest.raises(ValueError):
        InputState().save_bindings()
