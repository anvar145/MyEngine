"""Ядро undo-команд: CommandStack, транзакции, слияние, базовые команды."""

from __future__ import annotations

import pytest

from engine.core.commands import (
    AddComponentCommand,
    CommandStack,
    CreateObjectCommand,
    DeleteObjectCommand,
    RemoveComponentCommand,
    RenameObjectCommand,
    ReparentCommand,
    SetActiveCommand,
    SetFieldCommand,
    SetLayerCommand,
    SetTagCommand,
)
from engine.core.math3d import Vector3
from engine.core.scene import Scene
from engine.rendering.lighting import LightComponent
from engine.scene_io import scene_to_dict


def _scene() -> tuple[Scene, dict[str, object]]:
    scene = Scene("s")
    a = scene.create_game_object("A", position=Vector3(1, 0, 0))
    b = scene.create_game_object("B", parent=a, position=Vector3(0, 2, 0))
    c = scene.create_game_object("C", parent=a, position=Vector3(0, 0, 3))
    d = scene.create_game_object("D", position=Vector3(5, 5, 5))
    d.add_component(LightComponent())
    return scene, {"a": a, "b": b, "c": c, "d": d}


def _state(scene: Scene) -> dict:
    """Состояние сцены без привязки к порядку создания: по uid + место среди братьев."""
    data = scene_to_dict(scene)
    entries = {}
    for entry in data["objects"]:
        entry = dict(entry)
        entry["_index"] = scene.find_by_uid(entry["uid"]).sibling_index()
        entries[entry["uid"]] = entry
    data["objects"] = entries
    return data


def test_do_undo_restores_state_bitwise() -> None:
    scene2, objects2 = _scene()
    stack = CommandStack(scene2)
    # Каждая команда строится непосредственно перед execute.
    steps = [
        lambda o: RenameObjectCommand(o["a"], "A2"),
        lambda o: RenameObjectCommand(o["b"], "B2"),
        lambda o: SetActiveCommand(o["c"], False),
        lambda o: SetTagCommand(o["d"], "Enemy"),
        lambda o: SetLayerCommand(o["d"], 4),
        lambda o: SetFieldCommand(
            o["d"],
            o["d"].get_component(LightComponent),
            "follow_owner",
            True,
            False,
        ),
        lambda o: SetFieldCommand(
            o["d"],
            o["d"].get_component(LightComponent),
            "local_offset",
            o["d"].get_component(LightComponent).local_offset,
            Vector3(1, 2, 3),
        ),
        lambda o: SetFieldCommand(
            o["a"], o["a"].transform, "position", o["a"].transform.position,
            Vector3(9, 8, 7),
        ),
        lambda o: SetFieldCommand(
            o["a"], o["a"].transform, "scale", o["a"].transform.scale,
            Vector3(2, 2, 2),
        ),
        lambda o: ReparentCommand(o["d"], o["a"], 1),
        lambda o: ReparentCommand(o["c"], None, 0),
        lambda o: ReparentCommand(o["b"], o["d"]),
        lambda o: RenameObjectCommand(o["d"], "D2"),
        lambda o: SetTagCommand(o["a"], "Boss"),
        lambda o: SetLayerCommand(o["b"], 8),
        lambda o: SetActiveCommand(o["a"], False),
        lambda o: RemoveComponentCommand(
            o["d"], o["d"].get_component(LightComponent)
        ),
        lambda o: DeleteObjectCommand(o["b"]),
        lambda o: DeleteObjectCommand(o["c"]),
        lambda o: DeleteObjectCommand(o["d"]),
    ]
    assert len(steps) == 20
    snapshots = [_state(scene2)]
    for step in steps:
        stack.execute(step(objects2))
        snapshots.append(_state(scene2))
    for expected in reversed(snapshots[:-1]):
        stack.undo()
        assert _state(scene2) == expected
    assert not stack.can_undo
    for expected in snapshots[1:]:
        stack.redo()
        assert _state(scene2) == expected


def test_transaction_of_five_commands_is_one_undo() -> None:
    scene, o = _scene()
    stack = CommandStack(scene)
    before = _state(scene)
    with stack.begin_transaction("Пять правок"):
        stack.execute(RenameObjectCommand(o["a"], "X"))
        stack.execute(SetTagCommand(o["a"], "T"))
        stack.execute(SetLayerCommand(o["a"], 3))
        stack.execute(SetActiveCommand(o["b"], False))
        stack.execute(ReparentCommand(o["d"], o["a"], 0))
    assert stack.undo_count == 1 and stack.undo_name == "Пять правок"
    stack.undo()
    assert _state(scene) == before
    stack.redo()
    assert o["a"].name == "X" and o["d"].parent is o["a"]


def test_transaction_exception_rolls_back() -> None:
    scene, o = _scene()
    stack = CommandStack(scene)
    before = _state(scene)
    with pytest.raises(RuntimeError), stack.begin_transaction("t"):
        stack.execute(RenameObjectCommand(o["a"], "X"))
        raise RuntimeError("boom")
    assert _state(scene) == before
    assert not stack.can_undo


def test_merge_ten_edits_of_one_field_is_one_record() -> None:
    scene, o = _scene()
    stack = CommandStack(scene)
    d = o["d"]
    light = d.get_component(LightComponent)
    first = light.local_offset
    value = first
    for i in range(10):
        new = Vector3(i + 1, 0, 0)
        stack.execute(
            SetFieldCommand(d, light, "local_offset", value, new, stamp=100.0 + i * 0.1)
        )
        value = new
    assert stack.undo_count == 1
    stack.undo()
    assert light.local_offset == first
    stack.redo()
    assert light.local_offset == Vector3(10, 0, 0)


def test_no_merge_outside_window_or_other_field() -> None:
    scene, o = _scene()
    stack = CommandStack(scene)
    d = o["d"]
    light = d.get_component(LightComponent)
    stack.execute(SetFieldCommand(d, light, "follow_owner", True, False, stamp=1.0))
    stack.execute(SetFieldCommand(d, light, "follow_owner", False, True, stamp=5.0))
    stack.execute(
        SetFieldCommand(d, light, "local_offset", Vector3.zero(), Vector3(1, 1, 1),
                        stamp=5.2)
    )
    assert stack.undo_count == 3


def test_limit_drops_oldest() -> None:
    scene, o = _scene()
    stack = CommandStack(scene, limit=200)
    for i in range(250):
        stack.execute(RenameObjectCommand(o["a"], f"n{i}", stamp=float(i) * 10))
    assert stack.undo_count == 200
    for _ in range(200):
        stack.undo()
    assert not stack.can_undo
    assert o["a"].name == "n49"


def test_new_command_after_undo_truncates_redo() -> None:
    scene, o = _scene()
    stack = CommandStack(scene)
    stack.execute(RenameObjectCommand(o["a"], "X", stamp=0.0))
    stack.undo()
    assert stack.can_redo
    stack.execute(RenameObjectCommand(o["a"], "Y", stamp=50.0))
    assert not stack.can_redo


def test_create_capture_push_and_redo_keeps_uid() -> None:
    scene, o = _scene()
    stack = CommandStack(scene)
    created = scene.create_game_object("New", parent=o["a"])
    created.add_component(LightComponent())
    uid = created.uid
    stack.push(CreateObjectCommand.capture(created))
    stack.undo()
    assert scene.find_by_uid(uid) is None
    stack.redo()
    restored = scene.find_by_uid(uid)
    assert restored is not None and restored.name == "New"
    assert restored.parent is o["a"]
    assert restored.has_component(LightComponent)


def test_delete_restores_subtree_and_position() -> None:
    scene, o = _scene()
    before = _state(scene)
    stack = CommandStack(scene)
    stack.execute(DeleteObjectCommand(o["a"]))
    assert scene.find_by_uid(o["b"].uid) is None
    stack.undo()
    assert _state(scene) == before


def test_add_and_remove_component() -> None:
    scene, o = _scene()
    stack = CommandStack(scene)
    b = o["b"]
    before = _state(scene)
    light = b.add_component(LightComponent())
    stack.push(AddComponentCommand.capture(b, light))
    assert b.has_component(LightComponent)
    stack.undo()
    assert not scene.find_by_uid(b.uid).has_component(LightComponent)
    assert _state(scene) == before
    stack.redo()
    assert scene.find_by_uid(b.uid).has_component(LightComponent)
    stack.undo()
    d = o["d"]
    with_light = _state(scene)
    stack.execute(RemoveComponentCommand(d, d.get_component(LightComponent)))
    assert not d.has_component(LightComponent)
    stack.undo()
    assert _state(scene) == with_light


def test_subscribers_notified() -> None:
    scene, o = _scene()
    stack = CommandStack(scene)
    seen: list[str] = []
    off = stack.subscribe(seen.append)
    stack.execute(RenameObjectCommand(o["a"], "X"))
    stack.undo()
    stack.redo()
    off()
    stack.undo()
    assert seen == ["do", "undo", "redo"]


def test_set_parent_index_and_cycle_guard() -> None:
    scene = Scene("s")
    a = scene.create_game_object("A")
    b = scene.create_game_object("B", parent=a)
    c = scene.create_game_object("C", parent=a)
    d = scene.create_game_object("D", parent=a)
    d.set_parent(a, index=0)
    assert [x.name for x in a.children] == ["D", "B", "C"]
    r1 = scene.create_game_object("R1")
    r2 = scene.create_game_object("R2")
    r2.set_parent(None, index=0)
    assert [x.name for x in scene.root_objects] == ["R2", "A", "R1"]
    b.set_parent(r1, index=5)
    assert [x.name for x in r1.children] == ["B"]
    with pytest.raises(ValueError):
        a.set_parent(c)
    assert c.sibling_index() == 1


def test_reparent_sibling_a_after_c() -> None:
    scene = Scene("s")
    parent = scene.create_game_object("P")
    a = scene.create_game_object("A", parent=parent)
    scene.create_game_object("B", parent=parent)
    c = scene.create_game_object("C", parent=parent)
    stack = CommandStack(scene)
    # A после C: индекс после изъятия A = 2 (B, C, [A])
    stack.execute(ReparentCommand(a, parent, 2))
    assert [x.name for x in parent.children] == ["B", "C", "A"]
    stack.undo()
    assert [x.name for x in parent.children] == ["A", "B", "C"]
    stack.redo()
    assert [x.name for x in parent.children] == ["B", "C", "A"]
    assert c.sibling_index() == 1


def test_reparent_keeps_world_and_restores_local_bitwise() -> None:
    scene = Scene("s")
    p = scene.create_game_object("P", position=Vector3(3, 0, 0))
    p.transform.scale = Vector3(2, 2, 2)
    child = scene.create_game_object("C", position=Vector3(1.1, 2.3, 3.7))
    before_local = child.transform.position
    world = child.transform.world_matrix.copy()
    stack = CommandStack(scene)
    stack.execute(ReparentCommand(child, p))
    assert child.parent is p
    assert abs(child.transform.world_matrix - world).max() < 1e-9
    stack.undo()
    assert child.parent is None
    assert child.transform.position == before_local


def test_sibling_order_survives_save_and_load() -> None:
    from engine.scene_io import scene_from_dict

    scene = Scene("s")
    parent = scene.create_game_object("P")
    a = scene.create_game_object("A", parent=parent)
    scene.create_game_object("B", parent=parent)
    scene.create_game_object("C", parent=parent)
    root_b = scene.create_game_object("R")
    a.set_parent(parent, index=2)
    root_b.set_parent(None, index=0)
    loaded = scene_from_dict(scene_to_dict(scene))
    assert [x.name for x in loaded.root_objects] == ["R", "P"]
    assert [x.name for x in loaded.find("P").children] == ["B", "C", "A"]
