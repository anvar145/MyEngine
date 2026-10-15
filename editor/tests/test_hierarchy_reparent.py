"""Перестановка объектов в иерархии: drag-and-drop через модель и дерево."""

from __future__ import annotations

from PySide6.QtCore import QModelIndex, Qt

from editor.mime import OBJECT_MIME, OBJECT_UID_MIME


def _commit(window) -> None:
    """Команды пишутся сразу: фиксировать нечего."""


def _names(objects) -> list[str]:
    return [o.name for o in objects]


def _drop(window, objects, parent, row):
    model = window.hierarchy.model
    mime = model.mimeData([model.index_for(o) for o in objects])
    parent_index = model.index_for(parent) if parent is not None else QModelIndex()
    return model.dropMimeData(mime, Qt.MoveAction, row, 0, parent_index)


def _three_siblings(window):
    scene = window.context.scene
    parent = scene.create_game_object("P")
    a = scene.create_game_object("A", parent=parent)
    b = scene.create_game_object("B", parent=parent)
    c = scene.create_game_object("C", parent=parent)
    window.context.notify_modified()
    _commit(window)
    return parent, a, b, c


def test_mime_carries_uid_and_keeps_object_id(window) -> None:
    cube = window.create_primitive("cube")
    model = window.hierarchy.model
    mime = model.mimeData([model.index_for(cube)])
    assert OBJECT_MIME in model.mimeTypes() and OBJECT_UID_MIME in model.mimeTypes()
    assert bytes(mime.data(OBJECT_MIME)).decode() == str(cube.id)
    assert bytes(mime.data(OBJECT_UID_MIME)).decode() == cube.uid


def test_tree_accepts_internal_moves(window) -> None:
    from PySide6.QtWidgets import QAbstractItemView

    tree = window.hierarchy.tree
    assert tree.dragEnabled() and tree.acceptDrops()
    assert tree.dragDropMode() == QAbstractItemView.DragDropMode.DragDrop


def test_drop_under_new_parent_and_undo(window) -> None:
    scene = window.context.scene
    a = scene.create_game_object("A")
    b = scene.create_game_object("B")
    window.context.notify_modified()
    _commit(window)
    assert _drop(window, [a], b, -1)
    _commit(window)
    assert a.parent is b
    window.undo()
    restored = window.context.scene
    assert restored.find("A").parent is None
    window.redo()
    assert window.context.scene.find("A").parent.name == "B"


def test_sibling_a_after_c_then_undo(window) -> None:
    parent, a, _b, _c = _three_siblings(window)
    # «A после C»: бросок между C и концом списка, строка = 3
    assert _drop(window, [a], parent, 3)
    _commit(window)
    assert _names(window.context.scene.find("P").children) == ["B", "C", "A"]
    window.undo()
    assert _names(window.context.scene.find("P").children) == ["A", "B", "C"]
    window.redo()
    assert _names(window.context.scene.find("P").children) == ["B", "C", "A"]


def test_move_to_root_keeps_world_position(window) -> None:
    from engine.core.math3d import Vector3

    scene = window.context.scene
    p = scene.create_game_object("P", position=Vector3(5, 0, 0))
    child = scene.create_game_object("C", parent=p, position=Vector3(1, 0, 0))
    world = child.transform.world_matrix.copy()
    assert _drop(window, [child], None, -1)
    assert child.parent is None
    assert abs(child.transform.world_matrix - world).max() < 1e-9
    assert scene.root_objects[-1] is child


def test_drop_under_descendant_is_rejected(window) -> None:
    scene = window.context.scene
    a = scene.create_game_object("A")
    b = scene.create_game_object("B", parent=a)
    c = scene.create_game_object("C", parent=b)
    window.context.notify_modified()
    _commit(window)
    messages: list[str] = []
    window.context.status_message.connect(messages.append)
    before = [(o.name, o.parent.name if o.parent else None) for o in scene.objects]
    assert not _drop(window, [a], c, -1)
    assert not _drop(window, [a], a, -1)
    assert messages and "потомка" in messages[0]
    after = [(o.name, o.parent.name if o.parent else None) for o in scene.objects]
    assert before == after


def test_multi_drop_keeps_relative_order(window) -> None:
    parent, a, b, _c = _three_siblings(window)
    # A и B в конец: C, A, B
    assert _drop(window, [a, b], parent, 3)
    assert _names(parent.children) == ["C", "A", "B"]
    window.context.reparent_objects([parent.children[2], parent.children[0]], parent, 0)
    assert _names(parent.children) == ["B", "C", "A"]


def test_noop_drop_does_not_touch_scene(window) -> None:
    parent, _a, b, _c = _three_siblings(window)
    assert not _drop(window, [b], parent, 1)
    assert not _drop(window, [b], parent, 2)
    assert _names(parent.children) == ["A", "B", "C"]


def test_tree_view_refreshes_after_reorder(window) -> None:
    parent, a, _b, _c = _three_siblings(window)
    _drop(window, [a], parent, 3)
    window.hierarchy.refresh()
    model = window.hierarchy.model
    parent_index = model.index_for(parent)
    rows = [model.data(model.index(r, 0, parent_index)) for r in range(3)]
    assert rows == ["B", "C", "A"]


def test_tree_drop_event_reorders_and_ignores_action(window) -> None:
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QDragMoveEvent, QDropEvent

    window.show()
    _parent, a, _b, c = _three_siblings(window)
    window.hierarchy.refresh()
    tree = window.hierarchy.tree
    tree.expandAll()
    model = window.hierarchy.model
    mime = model.mimeData([model.index_for(a)])
    c_index = window.hierarchy.proxy.mapFromSource(model.index_for(c))
    rect = tree.visualRect(c_index)
    # нижняя четверть строки C - «под C»
    point = QPointF(rect.center().x(), rect.bottom() - 1)
    move = QDragMoveEvent(
        point.toPoint(), Qt.MoveAction, mime, Qt.LeftButton, Qt.NoModifier
    )
    tree.dragMoveEvent(move)
    drop = QDropEvent(point, Qt.MoveAction, mime, Qt.LeftButton, Qt.NoModifier)
    tree.dropEvent(drop)
    assert drop.dropAction() == Qt.IgnoreAction
    assert _names(window.context.scene.find("P").children) == ["B", "C", "A"]
