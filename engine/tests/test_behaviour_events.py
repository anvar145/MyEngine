"""Behaviour: ``@event_handler`` - подписки на шину событий сцены."""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.core.scene import Scene
from engine.physics.events import PhysicsEvent, TriggerEnter, TriggerExit
from engine.scripting import Behaviour, ScriptComponent, ScriptRuntime, event_handler

SINGLE = '''
from engine.physics.events import TriggerEnter
from engine.scripting import Behaviour, event_handler

LOG = []

class Door(Behaviour):
    @event_handler(TriggerEnter)
    def on_enter(self, event):
        LOG.append(("enter", event.a))

    def on_destroy(self):
        LOG.append(("destroy", self.scene.events.handler_count(TriggerEnter)))
'''

PRIORITY = '''
from engine.physics.events import TriggerEnter
from engine.scripting import Behaviour, event_handler

LOG = []

class Door(Behaviour):
    @event_handler(TriggerEnter, priority=1)
    def low(self, event):
        LOG.append("low")

    @event_handler(TriggerEnter, priority=5)
    def high(self, event):
        LOG.append("high")

    @event_handler(TriggerEnter)
    def middle(self, event):
        LOG.append("middle")
'''

TWO_EVENTS = '''
from engine.physics.events import TriggerEnter, TriggerExit
from engine.scripting import Behaviour, event_handler

LOG = []

class Door(Behaviour):
    @event_handler(TriggerEnter)
    @event_handler(TriggerExit)
    def either(self, event):
        LOG.append(type(event).__name__)
'''

BASE_EVENT = '''
from engine.physics.events import PhysicsEvent
from engine.scripting import Behaviour, event_handler

LOG = []

class Door(Behaviour):
    @event_handler(PhysicsEvent)
    def any_physics(self, event):
        LOG.append(type(event).__name__)
'''

BROKEN = '''
from engine.physics.events import TriggerEnter
from engine.scripting import Behaviour, event_handler

LOG = []

class Door(Behaviour):
    @event_handler(TriggerEnter, priority=9)
    def broken(self, event):
        raise ValueError("крах хендлера")

    @event_handler(TriggerEnter)
    def fine(self, event):
        LOG.append("fine")
'''


def _write(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def _attach(scene: Scene, runtime: ScriptRuntime, path: Path, name: str = "O"):
    obj = scene.create_game_object(name)
    return obj, obj.add_component(ScriptComponent(path, runtime=runtime))


def _setup(tmp_path: Path, source: str):
    runtime = ScriptRuntime(project_root=tmp_path)
    scene = Scene("T")
    obj, component = _attach(scene, runtime, _write(tmp_path, "door.py", source))
    return scene, obj, component


def _log(component: ScriptComponent) -> list:
    return component.handle.module.LOG


def test_single_subscriber_receives_event(tmp_path: Path) -> None:
    scene, _, component = _setup(tmp_path, SINGLE)
    assert scene.events.handler_count(TriggerEnter) == 1
    assert scene.events.emit(TriggerEnter(7, 8)) == 1
    assert _log(component) == [("enter", 7)]


def test_subscribed_at_attach_before_start(tmp_path: Path) -> None:
    runtime = ScriptRuntime(project_root=tmp_path)
    scene = Scene("T")
    obj = scene.create_game_object("O")
    obj.active_self = False  # on_start не будет вызван, подписка уже стоит
    component = obj.add_component(
        ScriptComponent(_write(tmp_path, "door.py", SINGLE), runtime=runtime)
    )
    assert scene.events.handler_count(TriggerEnter) == 1
    scene.events.emit(TriggerEnter(1, 2))
    assert _log(component) == [("enter", 1)]


def test_several_subscribers_all_receive(tmp_path: Path) -> None:
    runtime = ScriptRuntime(project_root=tmp_path)
    scene = Scene("T")
    path = _write(tmp_path, "door.py", SINGLE)
    components = [_attach(scene, runtime, path, f"D{i}")[1] for i in range(3)]
    assert scene.events.emit(TriggerEnter(5, 6)) == 3
    assert [_log(c) for c in components] == [[("enter", 5)]] * 3


def test_one_method_on_two_event_types(tmp_path: Path) -> None:
    scene, _, component = _setup(tmp_path, TWO_EVENTS)
    scene.events.emit(TriggerEnter(1, 2))
    scene.events.emit(TriggerExit(1, 2))
    assert _log(component) == ["TriggerEnter", "TriggerExit"]


def test_base_event_type_receives_subclasses(tmp_path: Path) -> None:
    scene, _, component = _setup(tmp_path, BASE_EVENT)
    scene.events.emit(TriggerEnter(1, 2))
    scene.events.emit(TriggerExit(1, 2))
    assert _log(component) == ["TriggerEnter", "TriggerExit"]


def test_priority_order(tmp_path: Path) -> None:
    scene, _, component = _setup(tmp_path, PRIORITY)
    scene.events.emit(TriggerEnter(1, 2))
    assert _log(component) == ["high", "low", "middle"]


def test_remove_component_unsubscribes_before_on_destroy(tmp_path: Path) -> None:
    scene, obj, component = _setup(tmp_path, SINGLE)
    log = _log(component)
    obj.remove_component(component)
    assert scene.events.handler_count() == 0
    scene.events.emit(TriggerEnter(1, 2))
    assert log == [("destroy", 0)]  # на момент on_destroy подписки уже сняты


def test_destroy_game_object_unsubscribes(tmp_path: Path) -> None:
    scene, obj, component = _setup(tmp_path, SINGLE)
    log = _log(component)
    scene.destroy_game_object(obj, immediate=True)
    assert scene.events.handler_count() == 0
    assert log == [("destroy", 0)]


def test_handler_exception_does_not_break_others(tmp_path: Path, caplog) -> None:
    scene, _, component = _setup(tmp_path, BROKEN)
    with caplog.at_level("ERROR"):
        delivered = scene.events.emit(TriggerEnter(1, 2))
    assert _log(component) == ["fine"]
    assert delivered == 2  # ошибка перехвачена скриптом, шина её не видит
    assert component.handle.errors == 1
    assert "крах хендлера" in caplog.text


def test_subscription_survives_hot_reload(tmp_path: Path) -> None:
    scene, _, component = _setup(tmp_path, SINGLE)
    old = component.behaviour
    _write(tmp_path, "door.py", SINGLE.replace('"enter"', '"enter2"'))
    component.reload()
    assert component.behaviour is not old
    assert scene.events.handler_count(TriggerEnter) == 1  # старая снята, новая стоит
    scene.events.emit(TriggerEnter(3, 4))
    assert _log(component) == [("enter2", 3)]


def test_manual_script_reload_without_start_keeps_single_subscription(
    tmp_path: Path,
) -> None:
    scene, _, component = _setup(tmp_path, SINGLE)
    for _ in range(3):
        component.reload()
    assert scene.events.handler_count(TriggerEnter) == 1


def test_inherited_and_overridden_handlers() -> None:
    class Base(Behaviour):
        @event_handler(TriggerEnter)
        def a(self, event):
            pass

        @event_handler(TriggerExit, priority=2)
        def b(self, event):
            pass

    class Child(Base):
        def b(self, event):  # переопределён без декоратора - подписка снята
            pass

        @event_handler(PhysicsEvent)
        def c(self, event):
            pass

    assert Base.__event_handlers__ == (
        ("a", TriggerEnter, 0),
        ("b", TriggerExit, 2),
    )
    assert Child.__event_handlers__ == (
        ("a", TriggerEnter, 0),
        ("c", PhysicsEvent, 0),
    )


def test_event_handler_rejects_non_class() -> None:
    with pytest.raises(TypeError):
        event_handler("TriggerEnter")  # type: ignore[arg-type]


def test_handlers_do_not_run_when_no_scene(tmp_path: Path) -> None:
    runtime = ScriptRuntime(project_root=tmp_path)
    from engine.core.game_object import GameObject

    obj = GameObject("Free")
    component = obj.add_component(
        ScriptComponent(_write(tmp_path, "door.py", SINGLE), runtime=runtime)
    )
    assert component.behaviour._subscriptions == []  # noqa: SLF001
    scene = Scene("T")
    scene.add_game_object(obj)  # on_enable при попадании в сцену ставит подписки
    scene.events.emit(TriggerEnter(1, 2))
    assert _log(component) == [("enter", 1)]
