"""Планировщик в скриптах: self.after/every/tween/start_coroutine."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from engine.core.math3d import Vector3
from engine.core.scene import Scene
from engine.scripting import ScriptComponent, ScriptRuntime

BEHAVIOUR = '''
from engine.scripting import Behaviour

LOG = []

class Sched(Behaviour):
    def on_start(self):
        self.after(0.5, lambda: LOG.append("after"))
        self.every(0.2, lambda: LOG.append("tick"))
        self.start_coroutine(self.routine())

    def routine(self):
        yield self.wait(0.3)
        LOG.append("co1")
        yield self.wait_for_frame()
        LOG.append("co2")

    def on_update(self, delta_time):
        LOG.append("update")

    def on_destroy(self):
        LOG.append("destroy:%d" % self.scene.scheduler.count)
'''

FUNCTIONS = '''
LOG = []

def on_start(self):
    self.after(0.5, lambda: LOG.append("after"))
    self.every(0.2, lambda: LOG.append("tick"))

def on_destroy(self):
    LOG.append("destroy:%d" % self.scene.scheduler.count)
'''


def _setup(tmp_path: Path, source: str):
    runtime = ScriptRuntime(project_root=tmp_path)
    scene = Scene("T")
    obj = scene.create_game_object("O")
    path = tmp_path / "s.py"
    path.write_text(source, encoding="utf-8")
    component = ScriptComponent(path, runtime=runtime)
    obj.add_component(component)
    return scene, obj, component, path


def _log(component: ScriptComponent) -> list[str]:
    return component.handle.module.LOG


def _run(scene: Scene, seconds: float, dt: float = 0.1) -> None:
    for _ in range(round(seconds / dt)):
        scene.update(dt)


def test_after_every_and_coroutine_fire(tmp_path: Path) -> None:
    scene, _, component, _ = _setup(tmp_path, BEHAVIOUR)
    _run(scene, 1.0)
    log = _log(component)
    assert "after" in log and log.count("tick") >= 3
    assert log.index("co1") < log.index("co2")


def test_destroy_cancels_all_before_user_hook(tmp_path: Path) -> None:
    scene, obj, component, _ = _setup(tmp_path, BEHAVIOUR)
    scene.update(0.1)
    log = _log(component)
    assert scene.scheduler.count == 3
    obj.remove_component(component)
    assert scene.scheduler.count == 0
    assert "destroy:0" in log  # задачи отменены до on_destroy
    _run(scene, 1.0)
    assert "after" not in log


def test_functions_script_has_same_api_and_cleanup(tmp_path: Path) -> None:
    scene, obj, component, _ = _setup(tmp_path, FUNCTIONS)
    _run(scene, 1.0)
    log = _log(component)
    assert "after" in log and "tick" in log
    obj.remove_component(component)
    assert scene.scheduler.count == 0 and "destroy:0" in log


@pytest.mark.parametrize("source", [BEHAVIOUR, FUNCTIONS])
def test_reload_cancels_tasks_and_allows_new(tmp_path: Path, source: str) -> None:
    scene, _, component, _ = _setup(tmp_path, source)
    scene.update(0.1)
    assert scene.scheduler.count >= 2
    component.reload()
    # задачи старой версии отменены; у Behaviour on_start не повторяется
    # (состояние перенесено), у функций новый on_start создаёт свои
    assert scene.scheduler.count == (0 if source is BEHAVIOUR else 2)
    before = scene.scheduler.count
    handle = component.after(10.0, lambda: None)
    assert not handle.is_done
    assert scene.scheduler.count == before + 1


def test_tween_moves_transform(tmp_path: Path) -> None:
    source = (
        "from engine.core.math3d import Vector3\n"
        "from engine.scripting import Behaviour\n"
        "class T(Behaviour):\n"
        "    def on_start(self):\n"
        "        self.tween(self.transform, 'position', Vector3(10, 0, 0), 1.0,"
        " ease='linear')\n"
    )
    scene, obj, _, _ = _setup(tmp_path, source)
    _run(scene, 0.5)
    assert 3.0 < obj.transform.position.x < 7.0
    _run(scene, 1.0)
    assert obj.transform.position.x == pytest.approx(10.0)
    assert obj.transform.position == Vector3(10, 0, 0)


def test_timer_exception_does_not_break_scheduler(tmp_path: Path, caplog) -> None:
    scene, _, component, _ = _setup(tmp_path, FUNCTIONS)
    ok: list[int] = []
    component.after(0.1, lambda: 1 / 0)
    component.after(0.2, lambda: ok.append(1))
    with caplog.at_level(logging.ERROR):
        _run(scene, 0.5)
    assert ok == [1]
    assert any("ZeroDivisionError" in r.getMessage() for r in caplog.records)


def test_disabled_component_keeps_tasks_but_skips_update(tmp_path: Path) -> None:
    scene, _, component, _ = _setup(tmp_path, BEHAVIOUR)
    scene.update(0.1)
    log = _log(component)
    component.enabled = False
    updates = log.count("update")
    _run(scene, 1.0)
    assert "after" in log  # таймер сработал на выключенном компоненте
    assert log.count("update") == updates  # on_update - нет


def test_finished_handles_are_pruned(tmp_path: Path) -> None:
    scene, _, component, _ = _setup(tmp_path, FUNCTIONS)
    for _ in range(200):
        component.after(0.0, lambda: None)
        scene.update(0.016)
    assert len(component._tasks) < 100  # noqa: SLF001


def test_cancel_single_and_cancel_all(tmp_path: Path) -> None:
    scene, _, component, _ = _setup(tmp_path, FUNCTIONS)
    fired: list[int] = []
    handle = component.after(0.1, lambda: fired.append(1))
    component.cancel(handle)
    component.after(0.1, lambda: fired.append(2))
    component.cancel_all()
    _run(scene, 0.5)
    assert fired == []


def test_tick_scheduler_ui_runs_ui_clock_only() -> None:
    scene = Scene("T")
    fired: list[str] = []
    scene.scheduler.after(
        0.1, lambda: fired.append("ui"), category=_ui()
    )
    scene.scheduler.after(0.1, lambda: fired.append("game"))
    scene.tick_scheduler_ui(0.2)
    assert fired == ["ui"]
    scene.update(0.2)
    assert fired == ["ui", "game"]


def _ui():
    from engine.core.scheduling import TimeCategory

    return TimeCategory.UI
