"""Сценарии A12: undo/redo структурных операций редактора одной записью."""

from __future__ import annotations

import json

from PySide6.QtCore import QModelIndex, Qt

from editor.prefab_actions import apply_override_to_prefab, unpack_object
from engine.core.components import MeshRenderer
from engine.core.math3d import Vector3
from engine.core.prefab_components import PrefabInstance
from engine.prefab import get_overrides
from engine.scene_io import scene_to_dict


def _steps(window) -> int:
    return len(window.context.active_document.commands._undo)  # noqa: SLF001


def _state(window) -> dict:
    context = window.context
    return scene_to_dict(context.scene, context.asset_db, context.active_document.sky)


def _entry(local_id: str, parent: str | None = None) -> dict:
    entry: dict = {
        "local_id": local_id,
        "name": local_id,
        "tag": "Untagged",
        "layer": 1,
        "active": True,
        "transform": {
            "position": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "scale": [1.0, 1.0, 1.0],
        },
        "components": [],
    }
    if parent is not None:
        entry["parent"] = {"local": parent}
    return entry


def _write_prefab(tmp_path):
    path = tmp_path / "pair.prefab.json"
    data = {
        "format": "myengine.prefab",
        "version": 1,
        "name": "pair",
        "root": "L0",
        "objects": [_entry("L0"), _entry("L1", "L0")],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _drop(window, objects, parent, row):
    model = window.hierarchy.model
    mime = model.mimeData([model.index_for(o) for o in objects])
    parent_index = model.index_for(parent) if parent is not None else QModelIndex()
    return model.dropMimeData(mime, Qt.MoveAction, row, 0, parent_index)


def test_reparent_fifty_objects_is_one_undo(window) -> None:
    scene = window.context.scene
    target = scene.create_game_object("Target")
    objects = [scene.create_game_object(f"O{i}") for i in range(50)]
    window.context.notify_modified()
    before = _state(window)
    steps = _steps(window)

    assert _drop(window, objects, target, -1)
    assert _steps(window) == steps + 1
    assert all(o.parent is target for o in objects)

    window.undo()
    assert _state(window) == before
    window.redo()
    assert [c.name for c in window.context.scene.find("Target").children] == [
        f"O{i}" for i in range(50)
    ]


def test_remove_component_is_undoable(window) -> None:
    context = window.context
    cube = context.create_object("Cube", mesh="cube")
    renderer = cube.get_component(MeshRenderer)
    context.remove_component(cube, renderer)
    assert cube.get_component(MeshRenderer) is None
    assert context.undo()
    assert context.scene.find_by_uid(cube.uid).get_component(MeshRenderer) is not None
    assert context.redo()
    assert context.scene.find_by_uid(cube.uid).get_component(MeshRenderer) is None


def test_multi_selection_move_is_one_entry(window) -> None:
    context = window.context
    objects = [context.create_object(f"O{i}", mesh="cube") for i in range(3)]
    steps = _steps(window)
    with context.transaction("Переместить"):
        for index, obj in enumerate(objects):
            context.set_field(
                obj, obj.transform, "position", Vector3(float(index + 1), 0.0, 0.0)
            )
    assert _steps(window) == steps + 1
    assert context.undo()
    assert all(o.transform.position.to_tuple() == (0.0, 0.0, 0.0) for o in objects)
    assert context.redo()
    assert [o.transform.position.x for o in objects] == [1.0, 2.0, 3.0]


def test_apply_to_prefab_keeps_sibling_order(window, tmp_path) -> None:
    """5 экземпляров и посторонний объект между ними: порядок не меняется."""
    context = window.context
    path = _write_prefab(tmp_path)
    scene = context.scene
    instances = []
    for index in range(5):
        instances.append(window.create_prefab(str(path)))
        if index == 2:
            scene.create_game_object("Foreign")
    order_before = [o.name for o in scene.root_objects]
    uids_before = [o.uid for o in scene.root_objects]
    base = len(order_before) - 6
    assert order_before.index("Foreign") == base + 3

    target = instances[1]
    next(c for c in target.children if c.name == "L1").transform.position = Vector3(
        4.0, 0.0, 0.0
    )
    record = get_overrides(target, assets=context.asset_db)[0]
    assert apply_override_to_prefab(context, target, record) == 5

    assert [o.uid for o in scene.root_objects] == uids_before
    before_undo = [o.uid for o in context.scene.root_objects]
    assert context.undo()
    assert [o.uid for o in context.scene.root_objects] == before_undo
    assert context.redo()
    assert [o.uid for o in context.scene.root_objects] == uids_before


def test_apply_to_prefab_ten_instances(window, tmp_path) -> None:
    context = window.context
    path = _write_prefab(tmp_path)
    roots = [window.create_prefab(str(path)) for _ in range(10)]
    uids = [o.uid for o in context.scene.root_objects]
    next(c for c in roots[0].children if c.name == "L1").transform.position = Vector3(
        2.0, 0.0, 0.0
    )
    record = get_overrides(roots[0], assets=context.asset_db)[0]
    assert apply_override_to_prefab(context, roots[0], record) == 10
    assert [o.uid for o in context.scene.root_objects] == uids
    for root in (o for o in context.scene.root_objects if o.name == "L0"):
        assert root.get_component(PrefabInstance) is not None
        child = next(c for c in root.children if c.name == "L1")
        assert child.transform.position.to_tuple() == (2.0, 0.0, 0.0)
    assert context.undo()
    assert [o.uid for o in context.scene.root_objects] == uids


def test_unpack_undo_returns_instance(window, tmp_path) -> None:
    root = window.create_prefab(str(_write_prefab(tmp_path)))
    before = _state(window)
    unpack_object(window.context, root)
    assert window.context.scene.find_by_uid(root.uid).get_component(
        PrefabInstance
    ) is None
    assert window.context.undo()
    assert _state(window) == before
    assert window.context.redo()
    assert window.context.scene.find_by_uid(root.uid).get_component(
        PrefabInstance
    ) is None
