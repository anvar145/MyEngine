"""ScriptComponent и Behaviour: режимы functions/behaviour, жизненный цикл, сцена."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.core.scene import Scene
from engine.scene_io import scene_from_dict, scene_to_dict
from engine.scripting import ScriptComponent, ScriptRuntime, set_default_runtime

BEHAVIOUR = '''
from engine.scripting import Behaviour, export

LOG = []

class Spinner(Behaviour):
    speed: float = export(90.0, range=(0, 360))
    target: GameObject = export(None)

    def on_attach(self):
        LOG.append("attach")

    def on_enable(self):
        LOG.append("enable")

    def on_start(self):
        LOG.append("start:" + str(self.speed))

    def on_update(self, delta_time):
        LOG.append("update")

    def on_late_update(self, dt):
        LOG.append("late")

    def on_fixed_update(self, dt):
        LOG.append("fixed")

    def on_destroy(self):
        LOG.append("destroy")
'''

FUNCTIONS = '''
LOG = []
speed = 5.0

def on_start(self):
    LOG.append("start")

def on_update(self, delta_time):
    LOG.append("update")

def on_destroy(self):
    LOG.append("destroy")
'''

TWO = '''
from engine.scripting import Behaviour

class First(Behaviour):
    def on_update(self, delta_time):
        self.transform.position  # noqa

class Second(Behaviour):
    def on_update(self, delta_time):
        pass
'''

MIXED = '''
from engine.scripting import Behaviour

CALLS = []

class Only(Behaviour):
    def on_update(self, delta_time):
        CALLS.append("class")

def on_update(self, delta_time):
    CALLS.append("function")
'''

BATCH = '''
from engine.scripting import Behaviour

BATCHES = []

class Patrol(Behaviour):
    @classmethod
    def on_fixed_update_batch(cls, items, dt):
        BATCHES.append(
            (cls.__name__, len(items), all(hasattr(i, "transform") for i in items))
        )

    def on_fixed_update(self, dt):
        raise AssertionError("пакетный хук заменяет поштучный")
'''


def _write(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def _attach(tmp_path: Path, name: str, source: str, **kwargs):
    runtime = ScriptRuntime(project_root=tmp_path)
    scene = Scene("T")
    obj = scene.create_game_object("O")
    component = obj.add_component(
        ScriptComponent(_write(tmp_path, name, source), runtime=runtime, **kwargs)
    )
    return scene, obj, component


def _log(component: ScriptComponent) -> list[str]:
    return component.handle.module.LOG


def test_module_with_behaviour_is_behaviour_kind(tmp_path: Path) -> None:
    _, _, component = _attach(tmp_path, "spin.py", BEHAVIOUR)
    assert component.script_kind == "behaviour"
    assert type(component.behaviour).__name__ == "Spinner"


def test_module_with_functions_is_functions_kind(tmp_path: Path) -> None:
    _, _, component = _attach(tmp_path, "fn.py", FUNCTIONS)
    assert component.script_kind == "functions"
    assert component.behaviour is None


def test_behaviour_lifecycle_in_scene(tmp_path: Path) -> None:
    scene, obj, component = _attach(tmp_path, "spin.py", BEHAVIOUR)
    # объект сразу активен: attach -> enable -> start при добавлении
    assert _log(component) == ["attach", "enable", "start:90.0"]
    scene.update(0.016)
    scene.update(0.016)
    scene.late_update(0.016)
    scene.fixed_update(0.02)
    assert _log(component)[3:] == ["update", "update", "late", "fixed"]
    module = component.handle.module
    obj.remove_component(component)
    assert module.LOG[-1] == "destroy"


def test_functions_lifecycle_in_scene(tmp_path: Path) -> None:
    scene, obj, component = _attach(tmp_path, "fn.py", FUNCTIONS)
    scene.update(0.016)
    scene.update(0.016)
    module = component.handle.module
    obj.remove_component(component)
    assert module.LOG == ["start", "update", "update", "destroy"]


def test_destroy_game_object_calls_behaviour_on_destroy(tmp_path: Path) -> None:
    scene, obj, component = _attach(tmp_path, "spin.py", BEHAVIOUR)
    scene.update(0.016)
    module = component.handle.module
    scene.destroy_game_object(obj)
    scene.update(0.016)
    assert module.LOG.count("destroy") == 1


def test_behaviour_wins_over_functions(tmp_path: Path) -> None:
    scene, _, component = _attach(tmp_path, "mixed.py", MIXED)
    scene.update(0.016)
    scene.update(0.016)
    assert component.script_kind == "behaviour"
    assert component.handle.module.CALLS == ["class", "class"]


def test_several_behaviours_first_wins_and_warns(tmp_path: Path, caplog) -> None:
    with caplog.at_level("WARNING"):
        _, _, component = _attach(tmp_path, "two.py", TWO)
    assert type(component.behaviour).__name__ == "First"
    assert "Second" in caplog.text


def test_explicit_script_class_selects_behaviour(tmp_path: Path) -> None:
    _, _, component = _attach(tmp_path, "two.py", TWO, script_class="Second")
    assert type(component.behaviour).__name__ == "Second"
    assert component.script_class == "Second"


def test_unknown_script_class_falls_back_to_functions(tmp_path: Path, caplog) -> None:
    with caplog.at_level("WARNING"):
        _, _, component = _attach(tmp_path, "two.py", TWO, script_class="Nope")
    assert component.script_kind == "functions"
    assert "Nope" in caplog.text


def test_batch_hook_called_once_with_instances(tmp_path: Path) -> None:
    runtime = ScriptRuntime(project_root=tmp_path)
    path = _write(tmp_path, "patrol.py", BATCH)
    scene = Scene("T")
    components = [
        scene.create_game_object(f"P{i}").add_component(
            ScriptComponent(path, runtime=runtime)
        )
        for i in range(3)
    ]
    scene.update(0.016)
    scene.fixed_update(0.02)
    scene.fixed_update(0.02)
    # у каждого компонента свой модуль; пакет зовёт класс первого компонента
    assert components[0].handle.module.BATCHES == [("Patrol", 3, True)] * 2
    assert components[1].handle.module.BATCHES == []


def test_fields_applied_before_on_start(tmp_path: Path) -> None:
    runtime = ScriptRuntime(project_root=tmp_path)
    path = _write(tmp_path, "spin.py", BEHAVIOUR)
    scene = Scene("T")
    obj = scene.create_game_object("O")
    component = ScriptComponent(path, runtime=runtime)
    component.fields["speed"] = 45.0
    obj.add_component(component)
    scene.update(0.016)
    assert "start:45.0" in _log(component)


# ----------------------------------------------------------- сериализация


def test_to_dict_has_no_script_class_for_functions(tmp_path: Path) -> None:
    _, _, component = _attach(tmp_path, "fn.py", FUNCTIONS)
    assert "script_class" not in component.to_dict()


def test_to_dict_omits_implicit_class_and_writes_explicit(tmp_path: Path) -> None:
    _, _, component = _attach(tmp_path, "spin.py", BEHAVIOUR)
    assert "script_class" not in component.to_dict()
    _, _, explicit = _attach(tmp_path, "two.py", TWO, script_class="Second")
    assert explicit.to_dict()["script_class"] == "Second"


def test_scene_round_trip_with_script_class(tmp_path: Path) -> None:
    runtime = ScriptRuntime(project_root=tmp_path)
    path = _write(tmp_path, "two.py", TWO)
    scene = Scene("S")
    scene.create_game_object("O").add_component(
        ScriptComponent(path, runtime=runtime, script_class="Second")
    )
    saved = json.loads(json.dumps(scene_to_dict(scene)))
    entry = saved["objects"][0]["components"][0]
    assert entry["script_class"] == "Second"

    set_default_runtime(runtime)
    try:
        loaded = scene_from_dict(saved)
        script = loaded.find("O").get_component(ScriptComponent)
        assert type(script.behaviour).__name__ == "Second"
        # сцена v1 без script_class: берётся первый класс модуля
        del entry["script_class"]
        legacy = scene_from_dict(saved).find("O").get_component(ScriptComponent)
        assert type(legacy.behaviour).__name__ == "First"
    finally:
        set_default_runtime(None)


def test_v1_scene_without_script_class_unchanged(tmp_path: Path) -> None:
    path = _write(tmp_path, "fn.py", FUNCTIONS)
    payload = {"auto_start": True, "path": str(path), "fields": {"speed": 7.5}}
    component = ScriptComponent.from_dict(dict(payload))
    assert json.dumps(component.to_dict()) == json.dumps(payload)


def test_behaviour_fields_round_trip(tmp_path: Path) -> None:
    _, _, component = _attach(tmp_path, "spin.py", BEHAVIOUR)
    assert component.set_field("speed", 120.0)
    assert component.behaviour.speed == 120.0
    assert component.to_dict()["fields"] == {"speed": 120.0}
    assert component.set_field("speed", 1000.0)
    assert component.behaviour.speed == 360.0  # зажим по range
    assert component.clear_field("speed")
    assert component.behaviour.speed == 90.0
    assert not component.set_field("nope", 1.0)


def test_behaviour_object_ref_resolved_after_load(tmp_path: Path) -> None:
    runtime = ScriptRuntime(project_root=tmp_path)
    path = _write(tmp_path, "spin.py", BEHAVIOUR)
    scene = Scene("T")
    holder = scene.create_game_object("H")
    other = scene.create_game_object("Other")
    component = ScriptComponent.from_dict(
        {"path": str(path), "fields": {"target": {"object": 77}}}
    )
    component._runtime = runtime  # noqa: SLF001
    holder.add_component(component)
    component.resolve_refs({77: other})
    assert component.behaviour.target is other


def test_reload_recreates_instance(tmp_path: Path) -> None:
    scene, _, component = _attach(tmp_path, "spin.py", BEHAVIOUR)
    scene.update(0.016)
    old = component.behaviour
    old.custom_state = 1
    component.reload()
    assert component.behaviour is not old  # экземпляр новый
    assert component.behaviour.custom_state == 1  # состояние перенесено
    assert component.script_kind == "behaviour"


def test_reload_keeps_non_export_state(tmp_path: Path) -> None:
    """Приватное состояние переносится, export-поля остаются из сцены."""
    source = BEHAVIOUR.replace(
        "class Spinner(Behaviour):",
        "class Spinner(Behaviour):\n    def __init__(self):\n"
        "        super().__init__()\n        self._x = None\n",
    )
    _, _, component = _attach(tmp_path, "spin.py", source)
    component.set_field("speed", 120.0)
    component.behaviour._x = 42  # noqa: SLF001
    component.reload()
    assert component.behaviour._x == 42  # noqa: SLF001
    assert component.behaviour.speed == 120.0


def test_hook_exception_is_isolated(tmp_path: Path, caplog) -> None:
    source = BEHAVIOUR.replace('LOG.append("update")', "raise ValueError('крах')")
    scene, _, component = _attach(tmp_path, "boom.py", source)
    with caplog.at_level("ERROR"):
        scene.update(0.016)
        scene.update(0.016)
    assert component.handle.errors == 2
    assert "крах" in caplog.text


@pytest.mark.parametrize("name", ["game_object", "scene", "transform", "time"])
def test_behaviour_proxy_reaches_component(tmp_path: Path, name: str) -> None:
    _, _, component = _attach(tmp_path, "spin.py", BEHAVIOUR)
    assert getattr(component.behaviour, name) is getattr(component, name)
