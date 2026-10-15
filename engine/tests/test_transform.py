"""Тесты иерархии трансформов и пакетного пересчёта мировых матриц."""

from __future__ import annotations

import numpy as np
import pytest

from engine.core.game_object import GameObject
from engine.core.math3d import Quaternion, Vector3
from engine.core.scene import Scene
from engine.core.transform import TransformStore


class TestTransformStore:
    def test_create_and_local_values(self) -> None:
        store = TransformStore(initial_capacity=2)
        index = store.create(position=Vector3(1, 2, 3))
        assert store.is_alive(index)
        assert store.alive_count == 1
        assert store.local_position(index).to_tuple() == (1.0, 2.0, 3.0)

        store.set_local_position(index, Vector3(4, 5, 6))
        assert store.local_position(index).to_tuple() == (4.0, 5.0, 6.0)
        store.set_local_rotation(index, Quaternion.from_euler(0, 90, 0))
        store.set_local_scale(index, Vector3(2, 2, 2))
        assert store.local_scale(index).to_tuple() == (2.0, 2.0, 2.0)

    def test_growth_and_slot_reuse(self) -> None:
        store = TransformStore(initial_capacity=2, growth_factor=2.0)
        indices = [store.create() for _ in range(5)]
        assert store.capacity >= 5
        assert store.alive_count == 5

        store.release(indices[0])
        assert not store.is_alive(indices[0])
        reused = store.create()
        assert reused == indices[0]  # free-list переиспользует слот
        assert store.alive_count == 5

    def test_release_parent_detaches_children(self) -> None:
        store = TransformStore(initial_capacity=4)
        parent = store.create()
        child = store.create(parent)
        store.release(parent)
        assert store.parent_index(child) == -1
        assert store.is_alive(child)

    def test_discard_ignores_children(self) -> None:
        store = TransformStore(initial_capacity=4)
        parent = store.create()
        child = store.create(parent)
        store.discard(parent)
        assert not store.is_alive(parent)
        assert store.is_alive(child)

    def test_lifecycle_errors(self) -> None:
        store = TransformStore(initial_capacity=2)
        with pytest.raises(ValueError):
            store.create(parent=99)
        index = store.create()
        store.release(index)
        with pytest.raises(ValueError):
            store.local_position(index)


class TestWorldMatrices:
    def test_world_matrix_of_root_is_local(self) -> None:
        store = TransformStore(initial_capacity=4)
        index = store.create(position=Vector3(1, 0, 0))
        matrix = store.world_matrix(index)
        assert np.allclose(matrix, store.local_matrix(index), atol=1e-6)

    def test_child_inherits_parent_transform(self) -> None:
        store = TransformStore(initial_capacity=4)
        parent = store.create(position=Vector3(1, 0, 0))
        child = store.create(parent, position=Vector3(0, 2, 0))
        world = store.world_matrix(child)
        assert world[0, 3] == pytest.approx(1.0)
        assert world[1, 3] == pytest.approx(2.0)

    def test_grandchild_and_batch_update(self) -> None:
        store = TransformStore(initial_capacity=4)
        root = store.create(position=Vector3(1, 0, 0))
        middle = store.create(root, position=Vector3(0, 1, 0))
        leaf = store.create(middle, position=Vector3(0, 0, 1))
        updated = store.update_world_matrices()
        assert updated == 3

        position = store.world_matrix(leaf)[:3, 3]
        assert position == pytest.approx([1.0, 1.0, 1.0])

    def test_no_update_when_nothing_changed(self) -> None:
        store = TransformStore(initial_capacity=2)
        store.create(position=Vector3(1, 0, 0))
        assert store.update_world_matrices() == 1
        assert store.update_world_matrices() == 0
        assert store.dirty_count == 0

    def test_local_change_recomputes_subtree_only(self) -> None:
        store = TransformStore(initial_capacity=8)
        moved = store.create(position=Vector3(0, 0, 0))
        child = store.create(moved)
        other = store.create(position=Vector3(5, 0, 0))
        store.update_world_matrices()

        store.set_local_position(moved, Vector3(1, 0, 0))
        updated = store.update_world_matrices()
        assert updated == 2  # сам объект и его ребёнок
        assert store.world_matrix(child)[0, 3] == pytest.approx(1.0)
        assert store.world_matrix(other)[0, 3] == pytest.approx(5.0)

    def test_mark_all_dirty_recomputes_everything(self) -> None:
        store = TransformStore(initial_capacity=4)
        store.create()
        store.create()
        store.update_world_matrices()
        store.mark_all_dirty()
        assert store.update_world_matrices() == 2


class TestReparenting:
    def test_keep_world_preserves_position(self) -> None:
        store = TransformStore(initial_capacity=4)
        parent = store.create(position=Vector3(10, 0, 0))
        child = store.create(position=Vector3(3, 0, 0))
        before = store.world_matrix(child)[:3, 3].copy()

        store.set_parent(child, parent, keep_world=True)
        after = store.world_matrix(child)[:3, 3]
        assert after == pytest.approx(before)
        assert store.local_position(child).x == pytest.approx(-7.0)

    def test_keep_world_false_keeps_local_values(self) -> None:
        store = TransformStore(initial_capacity=4)
        parent = store.create(position=Vector3(10, 0, 0))
        child = store.create(position=Vector3(3, 0, 0))
        store.set_parent(child, parent, keep_world=False)
        assert store.local_position(child).x == pytest.approx(3.0)
        assert store.world_matrix(child)[0, 3] == pytest.approx(13.0)

    def test_depth_is_refreshed(self) -> None:
        store = TransformStore(initial_capacity=4)
        outer = store.create()
        inner = store.create(outer)
        target = store.create()
        assert store.depth(target) == 0
        store.set_parent(target, inner)
        assert store.depth(target) == 2

    def test_cycle_is_rejected(self) -> None:
        store = TransformStore(initial_capacity=4)
        parent = store.create()
        child = store.create(parent)
        with pytest.raises(ValueError):
            store.set_parent(parent, child)
        with pytest.raises(ValueError):
            store.set_parent(parent, parent)

    def test_children_and_parent_queries(self) -> None:
        store = TransformStore(initial_capacity=4)
        parent = store.create()
        first = store.create(parent)
        second = store.create(parent)
        assert store.child_indices(parent) == (first, second)
        assert store.parent_index(first) == parent


class TestTransformFacade:
    def test_facade_reads_and_writes(self) -> None:
        scene = Scene("Test")
        game_object = scene.create_game_object("Obj", position=Vector3(1, 1, 1))
        transform = game_object.transform
        assert transform.game_object is game_object
        assert transform.is_valid

        transform.translate(Vector3(0, 1, 0))
        assert transform.position.almost_equals(Vector3(1, 2, 1), 1e-6)
        transform.rotate(Vector3.up(), 90)
        assert transform.forward.almost_equals(Vector3(1, 0, 0), 1e-5)

    def test_local_to_world_and_back(self) -> None:
        scene = Scene("Test")
        parent = scene.create_game_object("Parent", position=Vector3(5, 0, 0))
        child = scene.create_game_object(
            "Child", parent=parent, position=Vector3(0, 1, 0)
        )
        world = child.transform.local_to_world(Vector3(1, 0, 0))
        assert world.almost_equals(Vector3(6, 1, 0), 1e-5)
        back = child.transform.world_to_local(world)
        assert back.almost_equals(Vector3(1, 0, 0), 1e-4)

    def test_parent_children_and_set_parent(self) -> None:
        scene = Scene("Test")
        parent = scene.create_game_object("Parent", position=Vector3(2, 0, 0))
        child = scene.create_game_object("Child")
        child.transform.set_parent(parent.transform)
        assert child.transform.parent == parent.transform
        assert parent.transform.children[0] == child.transform
        child.transform.set_parent(None)
        assert child.transform.parent is None

    def test_foreign_scene_parent_rejected(self) -> None:
        first = Scene("A")
        second = Scene("B")
        a = first.create_game_object("A")
        b = second.create_game_object("B")
        with pytest.raises(ValueError):
            a.transform.set_parent(b.transform)

    def test_transform_equality_and_reset(self) -> None:
        scene = Scene("Test")
        game_object = scene.create_game_object("Obj")
        transform = game_object.transform
        assert transform == game_object.transform
        assert hash(transform) == hash(game_object.transform)
        transform.position = Vector3(9, 9, 9)
        transform.reset_local()
        assert transform.position == Vector3.zero()
        assert "Obj" in repr(transform)

    def test_orphan_transform_is_valid(self) -> None:
        free = GameObject("Free", position=Vector3(1, 1, 1))
        assert free.transform.is_valid
        assert free.transform.world_position.almost_equals(Vector3(1, 1, 1), 1e-6)
