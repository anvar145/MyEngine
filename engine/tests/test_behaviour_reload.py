"""Behaviour: горячая перезагрузка с сохранением состояния."""

from __future__ import annotations

from pathlib import Path

from engine.core.scene import Scene
from engine.physics.events import TriggerEnter
from engine.scripting import ScriptComponent, ScriptRuntime

V1 = '''
from engine.physics.events import TriggerEnter
from engine.scripting import Behaviour, event_handler, export

LOG = []

class Door(Behaviour):
    speed: float = export(1.0)

    def __init__(self):
        super().__init__()
        self._timer = 0
        self.items = []

    def on_start(self):
        LOG.append("start")

    def on_update(self, delta_time):
        self._timer += 1

    @event_handler(TriggerEnter)
    def on_enter(self, event):
        LOG.append("v1:enter")

    def on_reload(self):
        LOG.append("reload1")

    def on_destroy(self):
        LOG.append("destroy1")
'''

V2 = V1.replace("v1:enter", "v2:enter").replace("reload1", "reload2").replace(
    "destroy1", "destroy2"
).replace(
    "self.items = []", "self.items = []\n        self._new_in_new_class = 'fresh'"
)


def _setup(tmp_path: Path, fields: dict | None = None):
    runtime = ScriptRuntime(project_root=tmp_path)
    scene = Scene("T")
    obj = scene.create_game_object("O")
    path = tmp_path / "door.py"
    path.write_text(V1, encoding="utf-8")
    component = ScriptComponent(path, runtime=runtime)
    if fields:
        component._fields.update(fields)  # noqa: SLF001
    obj.add_component(component)
    return scene, component, path


def _reload(component: ScriptComponent, path: Path, source: str = V2) -> None:
    path.write_text(source, encoding="utf-8")
    component.reload()


def test_private_state_is_kept(tmp_path: Path) -> None:
    scene, component, path = _setup(tmp_path)
    scene.update(0.016)
    component.behaviour._timer = 42  # noqa: SLF001
    _reload(component, path)
    assert component.behaviour._timer == 42  # noqa: SLF001


def test_new_attribute_created_in_new_init(tmp_path: Path) -> None:
    scene, component, path = _setup(tmp_path)
    scene.update(0.016)
    _reload(component, path)
    assert component.behaviour._new_in_new_class == "fresh"  # noqa: SLF001


def test_export_field_comes_from_scene_not_old_state(tmp_path: Path) -> None:
    scene, component, path = _setup(tmp_path, {"speed": 5.0})
    scene.update(0.016)
    component.behaviour.speed = 7.0  # правка в рантайме, в сцену не записана
    _reload(component, path)
    assert component.behaviour.speed == 5.0


def test_event_subscriptions_are_moved(tmp_path: Path) -> None:
    scene, component, path = _setup(tmp_path)
    scene.update(0.016)
    old = component.behaviour
    _reload(component, path)
    assert component.behaviour is not old
    assert scene.events.handler_count(TriggerEnter) == 1
    scene.events.emit(TriggerEnter(1, 2))
    log = component.handle.module.LOG
    assert "v2:enter" in log
    assert "v1:enter" not in log


def test_on_destroy_old_on_reload_new_no_second_start(tmp_path: Path) -> None:
    scene, component, path = _setup(tmp_path)
    scene.update(0.016)
    _reload(component, path)
    log = component.handle.module.LOG
    assert log == ["reload2"]  # on_start не повторяется
    # on_destroy старой версии ушёл в журнал старого модуля
    assert component.behaviour.__class__.__module__ == component.handle.module.__name__


def test_old_instance_gets_on_destroy(tmp_path: Path) -> None:
    scene, component, path = _setup(tmp_path)
    scene.update(0.016)
    old_log = component.handle.module.LOG
    _reload(component, path)
    assert "destroy1" in old_log


def test_reload_before_first_start_calls_on_start(tmp_path: Path) -> None:
    runtime = ScriptRuntime(project_root=tmp_path)
    scene = Scene("T")
    obj = scene.create_game_object("O")
    obj.active_self = False  # on_start ещё не вызывался
    path = tmp_path / "door.py"
    path.write_text(V1, encoding="utf-8")
    component = obj.add_component(ScriptComponent(path, runtime=runtime))
    path.write_text(V2, encoding="utf-8")
    component.reload()
    obj.active_self = True
    scene.update(0.016)
    assert component.handle.module.LOG.count("start") == 1


def test_internal_attrs_are_not_in_dict_transfer(tmp_path: Path) -> None:
    scene, component, path = _setup(tmp_path)
    scene.update(0.016)
    assert not {"scene", "game_object", "transform"} & set(component.behaviour.__dict__)
    _reload(component, path)
    assert component.behaviour._component is component  # noqa: SLF001


def test_type_change_warns_once(tmp_path: Path, caplog) -> None:
    scene, component, path = _setup(tmp_path)
    scene.update(0.016)
    changed = V2.replace("self.items = []", "self.items = {}")
    with caplog.at_level("WARNING"):
        _reload(component, path, changed)
    warnings = [r for r in caplog.records if "изменился тип" in r.getMessage()]
    assert len(warnings) == 1
    assert "items" in warnings[0].getMessage()
