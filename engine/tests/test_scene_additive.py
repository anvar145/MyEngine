"""Аддитивные сцены: ``SceneManager`` и UI-действия."""

from __future__ import annotations

from pathlib import Path

from engine.core.scene import Scene
from engine.core.scene_manager import SceneManager
from engine.core.ui_events import (
    SceneActivated,
    SceneLoaded,
    SceneLoadRequested,
    SceneUnloaded,
    SceneUnloadRequested,
)
from engine.scene_io import (
    iter_scene_from_dict,
    load_scene,
    read_scene_data,
    save_scene,
)


def _write(path: Path, names: list[str]) -> Path:
    scene = Scene(path.stem)
    for name in names:
        scene.create_game_object(name)
    save_scene(scene, path)
    return path


def _manager() -> SceneManager:
    return SceneManager(
        load_scene,
        parse=read_scene_data,
        build=lambda data, _p: iter_scene_from_dict(data),
    )


def test_additive_scene_lives_next_to_active(tmp_path: Path) -> None:
    main = _write(tmp_path / "main.scene.json", ["A", "B"])
    ui = _write(tmp_path / "ui.scene.json", ["Hud"])
    manager = _manager()
    loaded: list[SceneLoaded] = []
    activated: list[SceneActivated] = []
    manager.events.subscribe(SceneLoaded, loaded.append)
    manager.events.subscribe(SceneActivated, activated.append)

    scene = manager.load(main)
    extra = manager.load(ui, additive=True)

    assert manager.active_scene is scene and extra is not scene
    assert manager.additive_scenes == {ui: extra}
    assert manager.all_scenes == [scene, extra]
    assert [e.additive for e in loaded] == [False, True]
    assert len(activated) == 1  # дополнительная сцена не активируется

    # Каждая сцена обновляется отдельно, объекты не пересекаются.
    ticks: list[str] = []
    for s in manager.all_scenes:
        s.create_game_object("Probe")
        ticks.append(s.name)
    assert ticks == ["main.scene", "ui.scene"]
    assert {o.name for o in scene.objects} == {"A", "B", "Probe"}
    assert {o.name for o in extra.objects} == {"Hud", "Probe"}


def test_unload_additive_keeps_active(tmp_path: Path) -> None:
    main = _write(tmp_path / "main.scene.json", ["A"])
    ui = _write(tmp_path / "ui.scene.json", ["Hud"])
    manager = _manager()
    scene = manager.load(main)
    extra = manager.load(ui, additive=True)
    unloaded: list[SceneUnloaded] = []
    manager.events.subscribe(SceneUnloaded, unloaded.append)

    manager.unload(ui)

    assert manager.active_scene is scene and len(scene) == 1
    assert not manager.additive_scenes and manager.get(ui) is None
    assert len(extra) == 0  # shutdown выгрузил объекты
    assert [(e.path, e.additive) for e in unloaded] == [(str(ui), True)]


def test_additive_before_any_active_becomes_active(tmp_path: Path) -> None:
    ui = _write(tmp_path / "ui.scene.json", ["Hud"])
    manager = _manager()
    scene = manager.load(ui, additive=True)
    assert manager.active_scene is scene
    assert not manager.additive_scenes


def test_set_active_moves_old_to_additive(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.scene.json", ["A"])
    b = _write(tmp_path / "b.scene.json", ["B"])
    manager = _manager()
    scene_a = manager.load(a)
    scene_b = manager.load(b, activate=False, additive=True)
    manager.set_active(b)
    assert manager.active_scene is scene_b
    assert manager.additive_scenes == {a: scene_a}


def test_reload_additive_replaces_scene(tmp_path: Path) -> None:
    main = _write(tmp_path / "main.scene.json", ["A"])
    ui = _write(tmp_path / "ui.scene.json", ["Hud"])
    manager = _manager()
    manager.load(main)
    first = manager.load(ui, additive=True)
    second = manager.load(ui, additive=True)
    assert first is not second and len(first) == 0
    assert manager.additive_scenes == {ui: second}


def test_scenes_do_not_share_uid_refs(tmp_path: Path) -> None:
    main = _write(tmp_path / "main.scene.json", ["A"])
    ui = _write(tmp_path / "ui.scene.json", ["Hud"])
    manager = _manager()
    scene = manager.load(main)
    extra = manager.load(ui, additive=True)
    uid = next(iter(scene.objects)).uid
    assert scene.find_by_uid(uid) is not None
    assert extra.find_by_uid(uid) is None


def _run_transition(manager: SceneManager, transition) -> int:
    frames = 0
    while not transition.is_done:
        manager.update_loading(0.3)
        frames += 1
        assert frames < 100000
    return frames


def test_transition_switches_active_and_unloads_old(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.scene.json", [f"A{i}" for i in range(200)])
    b = _write(tmp_path / "b.scene.json", [f"B{i}" for i in range(200)])
    loading = _write(tmp_path / "loading.scene.json", ["Spinner"])
    manager = _manager()
    old = manager.load(a)
    seen: list[tuple[str, ...]] = []

    transition = manager.transition(b, loading_scene=loading)
    # Пока грузится: старая активна, экран загрузки жив.
    assert manager.active_scene is old and manager.get(loading) is not None
    frames = 0
    while not transition.is_done:
        manager.update_loading(0.3)
        frames += 1
        if not transition.is_done:
            seen.append(tuple(p.name for p in manager.loaded))
    assert frames > 3 and transition.error is None
    new = manager.active_scene
    assert new is not None and new is not old and new.name == "b.scene"
    assert manager.loaded == [b]
    assert len(old) == 0  # старая выгружена
    assert all(loading.name in names for names in seen)


def test_transition_keep_source_as_additive(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.scene.json", ["A"])
    b = _write(tmp_path / "b.scene.json", ["B"])
    manager = _manager()
    old = manager.load(a)
    _run_transition(manager, manager.transition(b, unload_from=False))
    assert manager.active_scene is not old
    assert manager.additive_scenes == {a: old} and len(old) == 1


def test_transition_failure_keeps_old_scene(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.scene.json", ["A"])
    bad = tmp_path / "bad.scene.json"
    bad.write_text("{ nope", encoding="utf-8")
    loading = _write(tmp_path / "loading.scene.json", ["Spinner"])
    manager = _manager()
    old = manager.load(a)
    transition = manager.transition(bad, loading_scene=loading)
    _run_transition(manager, transition)
    assert transition.error is not None
    assert manager.active_scene is old and manager.loaded == [a]


def test_transition_cancel_keeps_old_scene(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.scene.json", ["A"])
    b = _write(tmp_path / "b.scene.json", [f"B{i}" for i in range(300)])
    manager = _manager()
    old = manager.load(a)
    transition = manager.transition(b)
    manager.update_loading(0.2)
    transition.cancel()
    manager.update_loading(0.2)
    assert transition.is_done and transition.error is not None
    assert manager.active_scene is old and manager.loaded == [a]


def test_ui_actions_emit_additive_requests() -> None:
    from engine.core.ui_components import UIButton
    from engine.core.ui_system import UISystem

    scene = Scene("s")
    got: list[object] = []
    scene.events.subscribe(SceneLoadRequested, got.append)
    scene.events.subscribe(SceneUnloadRequested, got.append)
    obj = scene.create_game_object("b")
    button = obj.add_component(UIButton)
    system = UISystem.__new__(UISystem)
    system.scene = scene
    system._buttons = {obj.id: button}
    for action in ("load_scene:a", "load_additive:b", "unload_scene:c"):
        button.on_click = action
        system._run_action(obj.id)
    assert [type(e).__name__ for e in got] == [
        "SceneLoadRequested",
        "SceneLoadRequested",
        "SceneUnloadRequested",
    ]
    assert [e.path for e in got] == ["a", "b", "c"]
    assert [getattr(e, "additive", None) for e in got[:2]] == [False, True]
