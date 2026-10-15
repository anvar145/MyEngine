"""Behaviour: ``@require_component`` и ``get_component`` в прокси."""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.core.physics_components import ColliderComponent, RigidBodyComponent
from engine.core.scene import Scene
from engine.scripting import (
    Behaviour,
    ScriptComponent,
    ScriptRuntime,
    require_component,
)

NEEDS_RB = '''
from engine.core.physics_components import RigidBodyComponent
from engine.scripting import Behaviour, require_component

LOG = []

@require_component(RigidBodyComponent)
class Pusher(Behaviour):
    def on_start(self):
        LOG.append("start")

    def on_update(self, delta_time):
        LOG.append("update:" + type(self.get_component(RigidBodyComponent)).__name__)
'''

NEEDS_TWO = '''
from engine.core.physics_components import ColliderComponent, RigidBodyComponent
from engine.scripting import Behaviour, require_component

LOG = []

@require_component(RigidBodyComponent)
@require_component(ColliderComponent)
class Pusher(Behaviour):
    def on_start(self):
        LOG.append("start")
'''


def _make(tmp_path: Path, source: str):
    runtime = ScriptRuntime(project_root=tmp_path)
    scene = Scene("T")
    obj = scene.create_game_object("Hero")
    path = tmp_path / "pusher.py"
    path.write_text(source, encoding="utf-8")
    return scene, obj, ScriptComponent(path, runtime=runtime)


def _errors(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]


def test_missing_component_blocks_start_and_logs_error(tmp_path: Path, caplog) -> None:
    scene, obj, script = _make(tmp_path, NEEDS_RB)
    with caplog.at_level("ERROR"):
        obj.add_component(script)
        scene.update(0.016)
        scene.update(0.016)
    log = script.handle.module.LOG
    assert log == []  # ни on_start, ни on_update
    errors = _errors(caplog)
    assert len(errors) == 1  # один ERROR, не по кадру
    assert "RigidBodyComponent" in errors[0]
    assert "Hero" in errors[0]
    assert "Pusher" in errors[0]
    assert script.missing_requirements() == [RigidBodyComponent]
    assert not script.is_running


def test_present_component_runs(tmp_path: Path, caplog) -> None:
    scene, obj, script = _make(tmp_path, NEEDS_RB)
    obj.add_component(RigidBodyComponent())
    obj.add_component(script)
    with caplog.at_level("ERROR"):
        scene.update(0.016)
    assert script.handle.module.LOG == ["start", "update:RigidBodyComponent"]
    assert _errors(caplog) == []
    assert script.missing_requirements() == []


def test_component_order_on_object_does_not_matter(tmp_path: Path, caplog) -> None:
    """Скрипт раньше требуемого компонента (как в сцене) стартует на первом кадре."""
    scene, obj, script = _make(tmp_path, NEEDS_RB)
    obj.add_component(script)
    obj.add_component(RigidBodyComponent())
    with caplog.at_level("ERROR"):
        scene.update(0.016)
    assert script.handle.module.LOG == ["start", "update:RigidBodyComponent"]
    assert _errors(caplog) == []


def test_adding_component_later_unblocks(tmp_path: Path) -> None:
    scene, obj, script = _make(tmp_path, NEEDS_RB)
    obj.add_component(script)
    scene.update(0.016)
    assert script.handle.module.LOG == []
    obj.add_component(RigidBodyComponent())
    scene.update(0.016)
    assert script.handle.module.LOG == ["start", "update:RigidBodyComponent"]


def test_several_requirements_are_combined(tmp_path: Path) -> None:
    scene, obj, script = _make(tmp_path, NEEDS_TWO)
    obj.add_component(RigidBodyComponent())
    obj.add_component(script)
    scene.update(0.016)
    assert script.missing_requirements() == [ColliderComponent]
    assert script.handle.module.LOG == []
    obj.add_component(ColliderComponent())
    scene.update(0.016)
    assert script.handle.module.LOG == ["start"]
    assert type(script.behaviour).__required_components__ == (
        ColliderComponent,
        RigidBodyComponent,
    )


def test_removing_required_component_at_runtime_is_noop(
    tmp_path: Path, caplog
) -> None:
    scene, obj, script = _make(tmp_path, NEEDS_RB)
    body = obj.add_component(RigidBodyComponent())
    obj.add_component(script)
    scene.update(0.016)
    obj.remove_component(body)
    with caplog.at_level("ERROR"):
        scene.update(0.016)
        scene.update(0.016)
    log = script.handle.module.LOG
    assert log[0] == "start"
    # скрипт не остановлен и ничего не логируется про требования; сам он
    # получает None от get_component (обычная ситуация для кода скрипта)
    assert script.is_running
    assert log[-1] == "update:NoneType"
    assert _errors(caplog) == []


def test_requirements_inherited_from_base() -> None:
    @require_component(RigidBodyComponent)
    class Base(Behaviour):
        pass

    @require_component(ColliderComponent)
    class Child(Base):
        pass

    assert Base.__required_components__ == (RigidBodyComponent,)
    assert Child.__required_components__ == (RigidBodyComponent, ColliderComponent)


def test_require_component_rejects_non_component() -> None:
    with pytest.raises(TypeError):
        require_component(int)  # type: ignore[arg-type]


def test_get_components_proxy(tmp_path: Path) -> None:
    _, obj, script = _make(tmp_path, NEEDS_RB)
    obj.add_component(RigidBodyComponent())
    obj.add_component(script)
    behaviour = script.behaviour
    assert len(behaviour.get_components(RigidBodyComponent)) == 1
    assert behaviour.get_component(ColliderComponent) is None
