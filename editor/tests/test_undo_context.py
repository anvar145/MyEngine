"""Undo/redo на уровне ``EditorContext``: команды и стек документа."""

from __future__ import annotations

from pathlib import Path

import pytest

from editor.context import EditorContext
from engine.core.components import MeshRenderer
from engine.core.math3d import Vector3
from engine.rendering.lighting import LightComponent
from engine.scene_io import scene_to_dict


@pytest.fixture()
def context(qapp, tmp_path: Path) -> EditorContext:
    ctx = EditorContext(project_root=tmp_path, language="ru")
    ctx.mark_clean()
    return ctx


def _state(ctx: EditorContext) -> dict:
    return scene_to_dict(ctx.scene, ctx.asset_db, ctx.active_document.sky)


def _steps(ctx: EditorContext) -> int:
    return len(ctx.active_document.commands._undo)  # noqa: SLF001


def test_nothing_to_undo_initially(context) -> None:
    assert not context.can_undo
    assert not context.can_redo
    assert not context.undo()


def test_create_undo_redo_roundtrip(context) -> None:
    before = _state(context)
    cube = context.create_object("Cube", mesh="cube")
    context.selection.select(cube)
    after = _state(context)
    assert context.can_undo

    assert context.undo()
    assert _state(context) == before
    assert context.selection.is_empty
    assert context.can_redo

    assert context.redo()
    assert _state(context) == after
    assert [o.name for o in context.scene.objects] == ["Cube"]


def test_parameter_change_is_undoable(context) -> None:
    cube = context.create_object("Cube", mesh="cube")
    context.set_field(cube, cube.transform, "position", Vector3(1.0, 2.0, 3.0))
    renderer = cube.get_component(MeshRenderer)
    old_roughness = renderer.roughness
    context.set_field(cube, renderer, "roughness", 0.1)

    assert context.undo()
    assert cube.get_component(MeshRenderer).roughness == old_roughness
    assert context.undo()
    assert cube.transform.position.to_tuple() == (0.0, 0.0, 0.0)
    assert context.redo()
    assert cube.transform.position.to_tuple() == (1.0, 2.0, 3.0)


def test_transaction_is_one_step(context) -> None:
    """Создание объекта вместе с компонентом — один шаг."""
    with context.transaction("Лампа"):
        game_object = context.create_object("Lamp")
        context.add_component(game_object, LightComponent)
    assert _steps(context) == 1
    assert context.undo()
    assert len(context.scene) == 0
    assert context.redo()
    assert context.scene.find_by_uid(game_object.uid).get_component(LightComponent)


def test_delete_restores_components_and_hierarchy(context) -> None:
    parent = context.create_object("Parent", mesh="cube")
    child = context.create_object("Child", mesh="sphere", parent=parent)
    context.set_field(child, child.transform, "position", Vector3(0.0, 1.0, 0.0))
    before = _state(context)

    context.delete_objects([parent])
    assert len(context.scene) == 0

    assert context.undo()
    assert _state(context) == before
    restored_child = next(o for o in context.scene.objects if o.name == "Child")
    assert restored_child.parent is not None
    assert restored_child.parent.name == "Parent"
    assert restored_child.get_component(MeshRenderer).mesh == "sphere"


def test_undo_restores_selection_before_change(context) -> None:
    context.create_object("A", mesh="cube")
    second = context.create_object("B", mesh="cube")
    context.selection.select(second)
    context.delete_objects([second])
    assert context.undo()
    assert context.selection.primary is not None
    assert context.selection.primary.name == "B"


def test_new_edit_drops_redo(context) -> None:
    context.create_object("A", mesh="cube")
    assert context.undo()
    context.create_object("B", mesh="cube")
    assert not context.can_redo
    assert [o.name for o in context.scene.objects] == ["B"]


def test_notify_adds_no_step(context) -> None:
    context.create_object("A", mesh="cube")
    steps = _steps(context)
    context.notify_modified()
    assert _steps(context) == steps


def test_dirty_flag_follows_saved_state(context) -> None:
    document = context.active_document
    context.create_object("A", mesh="cube")
    assert document.dirty
    context.mark_saved(document, document.path or Path("x.scene.json"))
    assert not document.dirty

    context.create_object("B", mesh="cube")
    assert document.dirty
    assert context.undo()
    assert not document.dirty, "после отмены состояние совпало с сохранённым"
    assert context.undo()
    assert document.dirty
    assert context.redo()
    assert not document.dirty


def test_same_field_edits_merge_within_window(context) -> None:
    cube = context.create_object("Cube", mesh="cube")
    steps = _steps(context)
    for x in (1.0, 2.0, 3.0):
        context.set_field(cube, cube.transform, "position", Vector3(x, 0.0, 0.0))
    # первая правка — новый шаг, следующие того же поля сливаются с ней
    assert _steps(context) == steps + 1
    assert context.undo()
    assert cube.transform.position.to_tuple() == (0.0, 0.0, 0.0)


def test_each_document_has_own_history(context) -> None:
    from engine.core.event_bus import EventBus
    from engine.core.scene import Scene

    context.create_object("A", mesh="cube")
    context.open_document(Scene("Other", event_bus=EventBus()))
    assert not context.can_undo
    context.create_object("B", mesh="cube")
    assert context.undo()
    assert len(context.scene) == 0
    context.activate_document(0)
    assert context.can_undo
    assert [o.name for o in context.scene.objects] == ["A"]


def test_undo_blocked_during_play(context) -> None:
    class _Play:
        is_active = True

    context.create_object("A", mesh="cube")
    context.play_mode = _Play()
    assert not context.can_undo
    assert not context.undo()
    context.play_mode = None
    assert context.undo()


def test_labels(context) -> None:
    context.create_object("A", mesh="cube")
    assert "A" in context.undo_text()
    assert context.redo_text() == ""


def test_edit_undo_keeps_selection_by_uid(context) -> None:
    obj = context.create_object("A", mesh="cube")
    other = context.create_object("B", mesh="cube")
    context.selection.select_many([obj, other])
    uids = {obj.uid, other.uid}
    context.set_field(obj, obj.transform, "position", Vector3(3, 0, 0))
    assert context.undo()
    assert {o.uid for o in context.selection.objects} == uids


def test_delete_undo_restores_object_and_selection(context) -> None:
    keep = context.create_object("Keep", mesh="cube")
    gone = context.create_object("Gone", mesh="cube")
    gone_uid, keep_uid = gone.uid, keep.uid
    context.selection.select(gone)
    context.delete_objects([gone])
    assert context.scene.find_by_uid(gone_uid) is None
    assert context.undo()
    restored = context.scene.find_by_uid(gone_uid)
    assert restored is not None and restored.name == "Gone"
    assert [o.uid for o in context.selection.objects] == [gone_uid]
    assert context.scene.find_by_uid(keep_uid) is not None


def test_hundred_objects_undo_redo(context) -> None:
    for index in range(100):
        context.create_object(f"O{index}", mesh="cube")
    assert len(context.scene) == 100
    for _ in range(100):
        assert context.undo()
    assert len(context.scene) == 0
    for _ in range(100):
        assert context.redo()
    assert [o.name for o in context.scene.objects] == [f"O{i}" for i in range(100)]


def test_scene_modified_subscribers_see_destroyed_as_such(context) -> None:
    """Во время сигнала удаления уничтоженные объекты помечены `is_destroyed`."""
    obj = context.create_object("A", mesh="cube")
    context.selection.select(obj)
    seen: list[bool] = []
    context.scene_modified.connect(
        lambda: seen.extend(o.is_destroyed for o in context.selection.objects)
    )
    context.delete_objects([obj])
    assert seen and all(seen)
    assert context.selection.is_empty
