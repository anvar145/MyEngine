"""Тесты сцены: объекты, компоненты, системы, запросы и уничтожение."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from engine.core.component import Component
from engine.core.event_bus import Event
from engine.core.game_object import GameObject
from engine.core.math3d import Vector3
from engine.core.scene import Scene
from engine.core.system import System


@dataclass(slots=True)
class ValueEvent(Event):
    """Тестовое событие с числовым значением."""

    value: int


class Recorder(Component):
    """Компонент, записывающий все вызовы хуков в общий журнал."""

    def __init__(self, log: list[str], name: str = "rec") -> None:
        super().__init__()
        self.log = log
        self.label = name
        self.updates = 0

    def _note(self, hook: str) -> None:
        self.log.append(f"{self.label}:{hook}")

    def on_attach(self) -> None:
        self._note("attach")

    def on_detach(self) -> None:
        self._note("detach")

    def on_enable(self) -> None:
        self._note("enable")

    def on_disable(self) -> None:
        self._note("disable")

    def on_start(self) -> None:
        self._note("start")

    def on_update(self, dt: float) -> None:
        self.updates += 1

    def on_fixed_update(self, dt: float) -> None:
        self._note("fixed")

    def on_late_update(self, dt: float) -> None:
        self._note("late")

    def on_destroy(self) -> None:
        self._note("destroy")


class Marker(Component):
    """Пустой компонент-метка для проверки запросов."""


class OrderedSystem(System):
    def __init__(self, name: str, log: list[str], priority: int = 0) -> None:
        super().__init__()
        self.name = name
        self.log = log
        self.priority = priority

    def on_added(self, scene: Scene) -> None:
        self.log.append(f"{self.name}:added")

    def on_update(self, scene: Scene, dt: float) -> None:
        self.log.append(f"{self.name}:update")

    def on_fixed_update(self, scene: Scene, dt: float) -> None:
        self.log.append(f"{self.name}:fixed")


class TestGameObjects:
    def test_create_and_ids(self) -> None:
        scene = Scene("Test")
        first = scene.create_game_object("A")
        second = scene.create_game_object("B")
        assert first.id != second.id
        assert scene.get_game_object(first.id) is first
        assert len(scene) == 2
        assert scene.root_objects == (first, second)

    def test_hierarchy_world_position(self) -> None:
        scene = Scene("Test")
        parent = scene.create_game_object("Parent", position=Vector3(1, 0, 0))
        child = scene.create_game_object(
            "Child", parent=parent, position=Vector3(0, 2, 0)
        )
        assert child.parent is parent
        assert parent.children == (child,)
        assert child.transform.world_position.almost_equals(Vector3(1, 2, 0), 1e-6)
        assert child in parent.root.children
        assert scene.root_objects == (parent,)
        assert parent.find_child("Child") is child

    def test_reparent_keeps_world_position(self) -> None:
        scene = Scene("Test")
        first = scene.create_game_object("First", position=Vector3(10, 0, 0))
        second = scene.create_game_object("Second")
        assert scene.root_objects == (first, second)

        second.set_parent(first)
        assert second in first.children
        assert second not in scene.root_objects
        # keep_world=True по умолчанию: мировая позиция сохраняется,
        # поэтому локальная становится отрицательной.
        assert second.transform.world_position.almost_equals(Vector3(0, 0, 0), 1e-5)
        assert second.transform.position.almost_equals(Vector3(-10, 0, 0), 1e-5)

        second.transform.position = Vector3(5, 0, 0)
        second.set_parent(first, keep_world=False)
        assert second.transform.world_position.almost_equals(Vector3(15, 0, 0), 1e-5)

        second.set_parent(None)
        assert second.parent is None
        assert second in scene.root_objects

    def test_deferred_destroy_of_subtree(self) -> None:
        scene = Scene("Test")
        parent = scene.create_game_object("Parent")
        child = scene.create_game_object("Child", parent=parent)
        log: list[str] = []
        parent.add_component(Recorder, log=log, name="parent")
        child.add_component(Recorder, log=log, name="child")

        parent.destroy()
        assert parent.is_destroyed
        assert len(scene) == 2  # до обработки очереди объекты ещё живы

        scene.update(1.0 / 60.0)
        assert len(scene) == 0
        assert "child:destroy" in log
        assert "parent:destroy" in log
        assert scene.stats.total_destroyed == 2

    def test_find_and_tags(self) -> None:
        scene = Scene("Test")
        scene.create_game_object("Enemy", tag="Monster")
        scene.create_game_object("Enemy")
        assert scene.find("Enemy") is not None
        assert len(scene.find_all("Enemy")) == 2
        assert scene.find_with_tag("Monster") is not None
        assert len(scene.find_all_with_tag("Monster")) == 1

    def test_free_object_added_to_scene(self) -> None:
        scene = Scene("Test")
        free = GameObject("Free", position=Vector3(0, 5, 0))
        child = GameObject("FreeChild")
        child.set_parent(free, keep_world=False)
        assert free.scene is None

        scene.add_game_object(free)
        assert free.scene is scene
        assert free.id != 0
        assert child.scene is scene
        assert child.transform.world_position.almost_equals(Vector3(0, 5, 0), 1e-6)
        assert scene.find("FreeChild") is child

    def test_cannot_add_same_object_twice(self) -> None:
        scene = Scene("Test")
        other = Scene("Other")
        free = GameObject("Free")
        scene.add_game_object(free)
        with pytest.raises(ValueError):
            other.add_game_object(free)

    def test_object_outside_scene_destroy(self) -> None:
        free = GameObject("Free")
        free.destroy()
        assert free.is_destroyed
        assert not free.is_valid


class TestComponents:
    def test_lifecycle_order(self) -> None:
        scene = Scene("Test")
        log: list[str] = []
        game_object = scene.create_game_object("Obj")
        recorder = game_object.add_component(Recorder, log=log, name="r")
        assert log == ["r:attach", "r:enable", "r:start"]

        scene.update(0.016)
        assert recorder.updates == 1
        assert scene.active_components == (recorder,)

        game_object.remove_component(recorder)
        assert log[-1] == "r:detach"
        assert len(scene.active_components) == 0

    def test_disable_component_stops_updates(self) -> None:
        scene = Scene("Test")
        log: list[str] = []
        game_object = scene.create_game_object("Obj")
        recorder = game_object.add_component(Recorder, log=log, name="r")
        recorder.enabled = False
        assert "r:disable" in log
        scene.update(0.016)
        assert recorder.updates == 0
        recorder.enabled = True
        scene.update(0.016)
        assert recorder.updates == 1
        assert log.count("r:enable") == 2
        assert log.count("r:start") == 1  # on_start вызывается один раз

    def test_inactive_object_stops_components(self) -> None:
        scene = Scene("Test")
        log: list[str] = []
        game_object = scene.create_game_object("Obj")
        recorder = game_object.add_component(Recorder, log=log, name="r")
        game_object.active_self = False
        assert "r:disable" in log
        scene.update(0.016)
        assert recorder.updates == 0
        assert not recorder.is_active
        game_object.active_self = True
        scene.update(0.016)
        assert recorder.updates == 1

    def test_child_of_inactive_parent_is_inactive(self) -> None:
        scene = Scene("Test")
        log: list[str] = []
        parent = scene.create_game_object("Parent")
        child = scene.create_game_object("Child", parent=parent)
        recorder = child.add_component(Recorder, log=log, name="c")
        parent.active_self = False
        assert not child.active_in_hierarchy
        assert not recorder.is_active
        scene.update(0.016)
        assert recorder.updates == 0

    def test_multiple_components_of_same_type(self) -> None:
        scene = Scene("Test")
        game_object = scene.create_game_object("Obj")
        first = game_object.add_component(Marker)
        second = game_object.add_component(Marker)
        assert first is not second
        assert game_object.get_component(Marker) is first
        assert game_object.get_components(Marker) == [first, second]
        assert game_object.has_component(Marker)
        assert game_object.remove_component(second) is True
        assert game_object.get_components(Marker) == [first]

    def test_duplicate_instance_rejected(self) -> None:
        scene = Scene("Test")
        game_object = scene.create_game_object("Obj")
        marker = Marker()
        game_object.add_component(marker)
        with pytest.raises(ValueError):
            game_object.add_component(marker)

    def test_invalid_component_type(self) -> None:
        game_object = GameObject("Obj")
        with pytest.raises(TypeError):
            game_object.add_component(object)  # type: ignore[arg-type]

    def test_fixed_and_late_phases(self) -> None:
        scene = Scene("Test")
        log: list[str] = []
        game_object = scene.create_game_object("Obj")
        game_object.add_component(Recorder, log=log, name="r")
        scene.fixed_update(0.02)
        scene.update(0.02)
        scene.late_update(0.02)
        assert "r:fixed" in log
        assert "r:late" in log


class TestQueries:
    def test_query_returns_matching_objects(self) -> None:
        scene = Scene("Test")
        with_marker = scene.create_game_object("A")
        with_marker.add_component(Marker)
        scene.create_game_object("B")
        assert scene.query(Marker) == (with_marker,)

    def test_query_cache_invalidated_by_structure_change(self) -> None:
        scene = Scene("Test")
        game_object = scene.create_game_object("A")
        assert scene.query(Marker) == ()
        game_object.add_component(Marker)
        assert scene.query(Marker) == (game_object,)
        game_object.remove_component(Marker)
        assert scene.query(Marker) == ()

    def test_query_without_types_returns_all(self) -> None:
        scene = Scene("Test")
        scene.create_game_object("A")
        scene.create_game_object("B")
        assert len(scene.query()) == 2

    def test_query_result_is_immutable_and_cached(self) -> None:
        scene = Scene("Test")
        game_object = scene.create_game_object("A")
        game_object.add_component(Marker)
        first = scene.query(Marker)
        assert isinstance(first, tuple)
        assert scene.query(Marker) is first  # кэш отдаёт тот же объект без копий


class TestSystems:
    def test_systems_run_in_priority_order(self) -> None:
        scene = Scene("Test")
        log: list[str] = []
        scene.add_system(OrderedSystem("low", log, priority=0))
        scene.add_system(OrderedSystem("high", log, priority=100))
        scene.update(0.016)
        assert log == ["low:added", "high:added", "high:update", "low:update"]
        assert [type(system).__name__ for system in scene.systems] == [
            "OrderedSystem",
            "OrderedSystem",
        ]

    def test_system_enabled_flag_and_removal(self) -> None:
        scene = Scene("Test")
        log: list[str] = []
        system = scene.add_system(OrderedSystem("sys", log, priority=0))
        system.enabled = False
        scene.update(0.016)
        assert "sys:update" not in log
        assert scene.remove_system(system) is True
        assert scene.remove_system(system) is False
        assert scene.get_system(OrderedSystem) is None

    def test_system_query_helper(self) -> None:
        scene = Scene("Test")
        log: list[str] = []

        class QuerySystem(System):
            def on_update(self, scene: Scene, dt: float) -> None:
                log.append(f"found:{len(self.query(Marker))}")

        scene.create_game_object("A").add_component(Marker)
        scene.add_system(QuerySystem())
        scene.update(0.016)
        assert log == ["found:1"]

    def test_duplicate_system_rejected(self) -> None:
        scene = Scene("Test")
        system = OrderedSystem("sys", [], priority=0)
        scene.add_system(system)
        with pytest.raises(ValueError):
            scene.add_system(system)


class TestSceneLifecycle:
    def test_events_flushed_in_update(self) -> None:
        scene = Scene("Test")
        seen: list[int] = []
        scene.events.subscribe(ValueEvent, lambda event: seen.append(event.value))
        scene.events.emit_deferred(ValueEvent(7))
        scene.update(0.016)
        assert seen == [7]

    def test_clear_and_shutdown(self) -> None:
        scene = Scene("Test")
        scene.create_game_object("A")
        scene.create_game_object("B")
        scene.clear()
        assert len(scene) == 0

        scene.create_game_object("C")
        scene.shutdown()
        assert len(scene) == 0
        with pytest.raises(RuntimeError):
            scene.create_game_object("D")

    def test_stats(self) -> None:
        scene = Scene("Test")
        game_object = scene.create_game_object("A")
        game_object.add_component(Marker)
        stats = scene.stats
        assert stats.game_objects == 1
        assert stats.components == 1
        assert stats.active_components == 1
        assert stats.transforms_alive == 1
        assert stats.total_created == 1

    def test_repr_contains_name(self) -> None:
        scene = Scene("Level")
        assert "Level" in repr(scene)
