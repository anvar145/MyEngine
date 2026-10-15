"""Порционная загрузка сцены: ``SceneLoadOperation`` и ``SceneManager.load_async``."""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.core.jobs import JobSystem
from engine.core.math3d import Vector3
from engine.core.scene import Scene
from engine.core.scene_load import (
    BUILD_BUDGET_MAX_MS,
    BUILD_BUDGET_MIN_MS,
    SceneLoadCancelled,
    adaptive_budget_ms,
)
from engine.core.scene_manager import SceneManager
from engine.core.ui_events import SceneActivated, SceneLoaded, SceneLoadProgress
from engine.scene_io import (
    iter_scene_from_dict,
    load_scene,
    read_scene_data,
    save_scene,
)


def _write_scene(path: Path, count: int) -> Path:
    scene = Scene("Big")
    parent = scene.create_game_object("Root")
    for index in range(count):
        child = scene.create_game_object(f"Obj{index}")
        child.transform.position = Vector3(float(index), 0.0, 0.0)
        child.set_parent(parent, keep_world=False)
    save_scene(scene, path)
    return path


def _manager(jobs: JobSystem | None = None) -> SceneManager:
    return SceneManager(
        load_scene,
        parse=read_scene_data,
        build=lambda data, _path: iter_scene_from_dict(data),
        jobs=jobs,
    )


def test_async_load_matches_sync_and_progress_grows(tmp_path: Path) -> None:
    path = _write_scene(tmp_path / "big.scene.json", 300)
    manager = _manager()
    seen: list[tuple[float, str]] = []
    manager.events.subscribe(
        SceneLoadProgress, lambda e: seen.append((e.progress, e.phase))
    )
    operation = manager.load_async(path, use_jobs=False)
    steps = 0
    while not operation.step(0.5):
        steps += 1
    scene = operation.result()
    reference = load_scene(path)
    assert len(scene) == len(reference) == 301
    assert {o.uid for o in scene.objects} == {o.uid for o in reference.objects}
    assert steps > 3  # бюджет 0.5 мс не вмещает 300 объектов в один шаг
    values = [p for p, _ in seen]
    assert values == sorted(values) and values[-1] == 1.0
    assert {phase for _, phase in seen} >= {"build", "activate"}
    assert manager.active_scene is scene


def test_parse_runs_in_worker_thread(tmp_path: Path) -> None:
    path = _write_scene(tmp_path / "a.scene.json", 20)
    jobs = JobSystem(workers=1)
    try:
        manager = _manager(jobs)
        operation = manager.load_async(path)
        guard = 0
        while not operation.step(5.0):
            guard += 1
            assert guard < 100000
        assert len(operation.result()) == 21
    finally:
        jobs.shutdown()


def test_sync_load_still_uses_loader() -> None:
    manager = SceneManager(lambda path: Scene(path.stem))
    scene = manager.load(Path("a"))
    assert scene.name == "a" and manager.active_scene is scene


def test_cancel_does_not_register_scene(tmp_path: Path) -> None:
    path = _write_scene(tmp_path / "c.scene.json", 200)
    manager = _manager()
    events: list[object] = []
    manager.events.subscribe(SceneLoaded, events.append)
    operation = manager.load_async(path, use_jobs=False)
    operation.step(0.2)
    assert not operation.is_done
    operation.cancel()
    assert operation.is_done and operation.is_cancelled
    assert manager.active_scene is None and not manager.loaded and not events
    with pytest.raises(SceneLoadCancelled):
        operation.result()


def test_bad_json_reports_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.scene.json"
    path.write_text("{ not json", encoding="utf-8")
    manager = _manager()
    operation = manager.load_async(path, use_jobs=False)
    assert operation.step() and isinstance(operation.error, ValueError)
    with pytest.raises(ValueError):
        operation.result()
    assert not manager.loaded


def test_result_before_done_raises(tmp_path: Path) -> None:
    path = _write_scene(tmp_path / "d.scene.json", 100)
    operation = _manager().load_async(path, use_jobs=False)
    operation.step(0.1)
    with pytest.raises(RuntimeError):
        operation.result()


def test_update_loading_advances_and_returns_finished(tmp_path: Path) -> None:
    path = _write_scene(tmp_path / "e.scene.json", 150)
    manager = _manager()
    activated: list[object] = []
    manager.events.subscribe(SceneActivated, activated.append)
    manager.load_async(path, use_jobs=False)
    finished: list[object] = []
    for _ in range(10000):
        finished += manager.update_loading(1.0)
        if finished:
            break
    assert len(finished) == 1 and not manager.loading
    assert len(activated) == 1


def test_loader_fallback_without_async_pair() -> None:
    manager = SceneManager(lambda path: Scene(path.stem))
    operation = manager.load_async(Path("x"))
    while not operation.step(5.0):
        pass
    assert operation.result().name == "x"


def test_adaptive_budget_is_clamped() -> None:
    assert adaptive_budget_ms(100.0) == BUILD_BUDGET_MIN_MS
    assert adaptive_budget_ms(0.0) == BUILD_BUDGET_MAX_MS
    mid = adaptive_budget_ms(10.0)
    assert BUILD_BUDGET_MIN_MS <= mid <= BUILD_BUDGET_MAX_MS
