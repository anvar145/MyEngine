"""Исключения в задачах планировщика скриптов: лог, троттлинг, продолжение."""

from __future__ import annotations

import logging
from pathlib import Path

from engine.core.scene import Scene
from engine.scripting import ScriptComponent, ScriptRuntime

SOURCE = '''
def on_start(self):
    pass
'''


def _setup(tmp_path: Path):
    runtime = ScriptRuntime(project_root=tmp_path)
    scene = Scene("T")
    obj = scene.create_game_object("O")
    path = tmp_path / "s.py"
    path.write_text(SOURCE, encoding="utf-8")
    component = ScriptComponent(path, runtime=runtime)
    obj.add_component(component)
    return scene, component


def _run(scene: Scene, steps: int, dt: float = 0.1) -> None:
    for _ in range(steps):
        scene.update(dt)


def test_timer_error_logged_with_traceback_and_scheduler_continues(
    tmp_path: Path, caplog
) -> None:
    scene, component = _setup(tmp_path)
    ok: list[int] = []

    def boom() -> None:
        raise ValueError("плохо")

    component.after(0.1, boom)
    component.after(0.2, lambda: ok.append(1))
    with caplog.at_level(logging.ERROR):
        _run(scene, 5)
    assert ok == [1]
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "ValueError" in text and "Traceback" in text
    assert component.handle.errors == 1


def test_coroutine_error_cancels_only_that_coroutine(tmp_path: Path, caplog) -> None:
    scene, component = _setup(tmp_path)
    ticks: list[int] = []

    def bad():
        yield component.wait(0.1)
        raise RuntimeError("корутина")

    handle = component.start_coroutine(bad())
    component.every(0.1, lambda: ticks.append(1))
    with caplog.at_level(logging.ERROR):
        _run(scene, 5)
    assert handle.is_done
    assert len(ticks) >= 3
    assert any("RuntimeError" in r.getMessage() for r in caplog.records)


def test_tween_on_complete_error_logged_and_tween_done(tmp_path: Path, caplog) -> None:
    scene, component = _setup(tmp_path)

    class Box:
        value = 0.0

    box = Box()
    handle = component.tween(box, "value", 1.0, 0.1).on_complete(lambda: 1 / 0)
    with caplog.at_level(logging.ERROR):
        _run(scene, 3)
    assert handle.is_done and box.value == 1.0
    assert any(r.exc_info for r in caplog.records)


def test_repeated_identical_errors_are_throttled(tmp_path: Path, caplog) -> None:
    scene, component = _setup(tmp_path)

    def boom() -> None:
        raise ValueError("одно и то же")

    component.every(0.1, boom)
    with caplog.at_level(logging.ERROR):
        _run(scene, 10)
    reports = [
        r for r in caplog.records if "ошибка в задаче планировщика" in r.getMessage()
    ]
    notices = [r for r in caplog.records if "подавлены" in r.getMessage()]
    assert len(reports) + len(notices) <= 3
    assert component.handle.errors >= 9  # счётчик считает все
    assert component.after(0.0, lambda: None)  # планировщик жив


def test_different_errors_are_logged_separately(tmp_path: Path, caplog) -> None:
    scene, component = _setup(tmp_path)

    def one() -> None:
        raise ValueError("a")

    def two() -> None:
        raise KeyError("b")

    component.after(0.1, one)
    component.after(0.1, two)
    with caplog.at_level(logging.ERROR):
        _run(scene, 3)
    reports = [
        r for r in caplog.records if "ошибка в задаче планировщика" in r.getMessage()
    ]
    assert len(reports) == 2
