"""SaveContext, GameState, ISaveable (A7a п.1)."""

from __future__ import annotations

import pytest

from engine.core.math3d import Vector3
from engine.core.save import GameState, ISaveable, SaveContext
from engine.core.scene import Scene


def test_nested_keys_and_default() -> None:
    ctx = SaveContext()
    ctx.set("a.b.c", 5)
    assert ctx.get("a.b.c") == 5
    assert ctx.get("a.b") == {"c": 5}
    assert ctx.get("a.x", 7) == 7
    assert ctx.has("a.b") and not ctx.has("z")


def test_value_types_round_trip_json() -> None:
    ctx = SaveContext()
    ctx.set("t", (1, 2, (3, 4)))
    ctx.set("v", Vector3(1, 2, 3))
    ctx.set("l", [1, "x", None, True])
    ctx.set("d", {"k": 1.5})
    back = SaveContext.from_dict(ctx.to_dict())
    assert back == ctx
    assert back.get("t") == (1, 2, (3, 4))
    assert isinstance(back.get("v"), Vector3)


@pytest.mark.parametrize("bad", [object(), {1: 2}, {"a.b": 1}, {"$x": 1}, {1, 2}])
def test_rejects_unsupported(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        SaveContext().set("k", bad)


@pytest.mark.parametrize("key", ["", "a..b", ".a", "a.", "$x"])
def test_rejects_bad_keys(key: str) -> None:
    with pytest.raises(ValueError):
        SaveContext().set(key, 1)


def test_set_through_scalar_fails() -> None:
    ctx = SaveContext()
    ctx.set("a", 1)
    with pytest.raises(ValueError):
        ctx.set("a.b", 2)


def test_get_typed() -> None:
    ctx = SaveContext()
    ctx.set("i", 3)
    ctx.set("b", True)
    assert ctx.get_typed("i", int) == 3
    assert ctx.get_typed("i", float) == 3.0
    with pytest.raises(ValueError):
        ctx.get_typed("b", int)
    with pytest.raises(ValueError):
        ctx.get_typed("i", str)
    with pytest.raises(ValueError):
        ctx.get_typed("missing", int)
    assert ctx.get_typed("missing", int, 9) == 9


def test_get_returns_copy() -> None:
    ctx = SaveContext()
    ctx.set("l", [1])
    ctx.get("l").append(2)
    assert ctx.get("l") == [1]


def test_game_state_on_changed() -> None:
    state = GameState()
    seen: list[tuple[str, object]] = []
    off = state.on_changed("quest", lambda k, v: seen.append((k, v)))
    state.set("quest.stage", 1)
    state.set("quest.stage", 1)  # без изменения - тишина
    state.set("other", 1)
    state.set("quest", {"stage": 2})
    assert seen == [("quest.stage", 1), ("quest", {"stage": 2})]
    off()
    state.set("quest.stage", 3)
    assert len(seen) == 2


def test_game_state_load_dict_notifies() -> None:
    state = GameState()
    state.set("a", 1)
    seen: list[str] = []
    state.on_changed("a", lambda k, v: seen.append(k))
    state.load_dict({})
    assert seen == ["a"] and not state.has("a")


def test_scene_state_is_lazy_and_shared() -> None:
    scene = Scene()
    assert scene.state is scene.state


def test_isaveable_protocol() -> None:
    class A:
        def on_save(self, ctx: SaveContext) -> None: ...
        def on_load(self, ctx: SaveContext) -> None: ...

    assert isinstance(A(), ISaveable) and not isinstance(object(), ISaveable)
