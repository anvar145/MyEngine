"""ObjectPool: выдача/возврат, режимы исчерпания, хуки, исключение из сохранения."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from engine.core.component import Component
from engine.core.math3d import Vector3
from engine.core.pool import ObjectPool
from engine.core.scene import Scene
from engine.scene_io import scene_to_dict
from engine.tests.test_prefab_instantiate import Library, _entry, _prefab, _three


def _scene() -> tuple[Scene, object]:
    prefab = _three("p")
    scene = Scene("s")
    scene.prefab_loader = Library(prefab)
    return scene, prefab


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, bool]]:
    log: list[tuple[str, str, bool]] = []

    def make(hook: str):
        def method(self: Component) -> None:
            if not isinstance(self, _Marker):
                return
            log.append((hook, self.game_object.name, self.game_object.active_self))

        return method

    monkeypatch.setattr(Component, "on_spawn", make("spawn"))
    monkeypatch.setattr(Component, "on_despawn", make("despawn"))
    return log


class _Marker(Component):
    pass


def test_get_release_lifo_and_counters() -> None:
    scene, prefab = _scene()
    pool = ObjectPool(scene, prefab, prewarm=2)
    assert (pool.size, pool.free) == (0, 2)
    assert len(scene.objects) == 6
    a = pool.get()
    b = pool.get()
    assert a is not None and b is not None and a is not b
    assert (pool.size, pool.free) == (2, 0)
    assert a.active_self and a.pooled
    pool.release(a)
    pool.release(b)
    assert not a.active_self and (pool.size, pool.free) == (0, 2)
    assert pool.get() is b  # LIFO
    c = pool.get()
    c2 = pool.get()
    assert c is a and c2 not in (a, b)  # третий - новый
    assert len(scene.objects) == 9


def test_get_places_and_release_resets_transform() -> None:
    scene, prefab = _scene()
    pool = ObjectPool(scene, prefab)
    obj = pool.get(position=Vector3(5, 6, 7))
    assert obj.transform.position.to_tuple() == (5.0, 6.0, 7.0)
    obj.children[0].transform.position = Vector3(9, 9, 9)
    pool.release(obj)
    assert obj.transform.position.to_tuple() == (1.0, 0.0, 0.0)
    assert obj.children[0].transform.position.to_tuple() == (1.0, 0.0, 0.0)
    again = pool.get()
    assert again is obj and again.transform.position.to_tuple() == (1.0, 0.0, 0.0)


def test_hook_order(calls: list) -> None:
    scene, prefab = _scene()
    pool = ObjectPool(scene, prefab, prewarm=1)
    root = pool.get()
    for obj in (root, *root.children):
        obj.add_component(_Marker())
    calls.clear()
    pool.release(root)
    assert calls == [
        ("despawn", "L0", True),
        ("despawn", "L1", True),
        ("despawn", "L2", True),
    ]
    calls.clear()
    pool.get()
    assert calls == [
        ("spawn", "L0", True),
        ("spawn", "L1", True),
        ("spawn", "L2", True),
    ]


def test_exception_in_despawn_is_logged_and_object_returned(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    scene, prefab = _scene()
    pool = ObjectPool(scene, prefab)
    root = pool.get()
    root.add_component(_Marker())

    def boom(self: Component) -> None:
        raise RuntimeError("fail")

    monkeypatch.setattr(Component, "on_despawn", boom)
    with caplog.at_level(logging.ERROR, logger="engine.core.pool"):
        assert pool.release(root)
    assert "on_despawn" in caplog.text
    assert not root.active_self and pool.free == 1


def test_release_foreign_object_warns(caplog: pytest.LogCaptureFixture) -> None:
    scene, prefab = _scene()
    pool = ObjectPool(scene, prefab)
    other = scene.create_game_object("x")
    with caplog.at_level(logging.WARNING, logger="engine.core.pool"):
        assert pool.release(other) is False
    assert "не выдан" in caplog.text


def test_destroyed_free_object_is_replaced(caplog: pytest.LogCaptureFixture) -> None:
    scene, prefab = _scene()
    pool = ObjectPool(scene, prefab, prewarm=1)
    victim = pool._free[0]  # noqa: SLF001
    scene.destroy_game_object(victim, immediate=True)
    with caplog.at_level(logging.ERROR, logger="engine.core.pool"):
        fresh = pool.get()
    assert fresh is not None and fresh is not victim and fresh.active_self
    assert "уничтожен" in caplog.text


def test_destroyed_active_object_is_dropped(caplog: pytest.LogCaptureFixture) -> None:
    scene, prefab = _scene()
    pool = ObjectPool(scene, prefab)
    obj = pool.get()
    scene.destroy_game_object(obj, immediate=True)
    with caplog.at_level(logging.ERROR, logger="engine.core.pool"):
        other = pool.get()
    assert other is not obj and pool.size == 1


def test_max_size_grow_returns_none_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    scene, prefab = _scene()
    pool = ObjectPool(scene, prefab, max_size=2)
    assert pool.get() and pool.get()
    with caplog.at_level(logging.WARNING, logger="engine.core.pool"):
        assert pool.get() is None
    assert "max_size" in caplog.text


def test_return_none_mode_is_quiet(caplog: pytest.LogCaptureFixture) -> None:
    scene, prefab = _scene()
    pool = ObjectPool(scene, prefab, max_size=1, on_exhausted="return_none")
    assert pool.get()
    with caplog.at_level(logging.WARNING, logger="engine.core.pool"):
        assert pool.get() is None
    assert caplog.text == ""


def test_reuse_oldest_recycles_first_issued() -> None:
    scene, prefab = _scene()
    pool = ObjectPool(scene, prefab, max_size=2, on_exhausted="reuse_oldest")
    first = pool.get()
    second = pool.get()
    third = pool.get()
    assert third is first and second.active_self
    assert pool.size == 2 and len(scene.objects) == 6


def test_prewarm_warns_over_limit(caplog: pytest.LogCaptureFixture) -> None:
    prefab = _prefab("one", [_entry("L0")])
    scene = Scene("s")
    scene.prefab_loader = Library(prefab)
    with caplog.at_level(logging.WARNING, logger="engine.core.pool"):
        ObjectPool(scene, prefab, prewarm=51)
    assert "Прогрев" in caplog.text


def test_bad_arguments() -> None:
    scene, prefab = _scene()
    with pytest.raises(ValueError):
        ObjectPool(scene, prefab, on_exhausted="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ObjectPool(scene, prefab, max_size=0)


def test_uids_are_distinct_and_deterministic() -> None:
    def run() -> list[str]:
        scene, prefab = _scene()
        pool = ObjectPool(scene, prefab, prewarm=3)
        got = [pool.get() for _ in range(5)]
        return [o.uid for o in got]

    first = run()
    assert len(set(first)) == 5
    assert first == run()


class TestSaveExclusion:
    def test_pool_objects_not_saved_but_manual_instances_are(self) -> None:
        scene, prefab = _scene()
        pool = ObjectPool(scene, prefab, prewarm=2)
        pool.get()  # один активен, один свободен
        manual = scene.instantiate(prefab)
        assert not manual.pooled
        data = scene_to_dict(scene)
        saved = {e["uid"] for e in data["objects"]}
        # вручную созданный экземпляр свернулся в одну запись-маркер
        assert saved == {manual.uid}

    def test_released_object_stays_excluded(self) -> None:
        scene, prefab = _scene()
        pool = ObjectPool(scene, prefab)
        obj = pool.get()
        pool.release(obj)
        assert scene_to_dict(scene)["objects"] == []


BEHAVIOUR = """
from engine.scripting import Behaviour

LOG = []

class Bullet(Behaviour):
    def on_start(self):
        self.after(10.0, lambda: LOG.append("late"))

    def on_spawn(self):
        LOG.append("spawn:%s" % self.game_object.active_self)

    def on_despawn(self):
        LOG.append("despawn:%s" % self.game_object.active_self)
"""


def test_behaviour_hooks_and_cancel_all(tmp_path: Path) -> None:
    from engine.scripting import ScriptComponent, ScriptRuntime

    path = tmp_path / "bullet.py"
    path.write_text(BEHAVIOUR, encoding="utf-8")
    runtime = ScriptRuntime(project_root=tmp_path)
    prefab = _prefab("b", [_entry("L0")])
    scene = Scene("s")
    scene.prefab_loader = Library(prefab)
    pool = ObjectPool(scene, prefab)
    obj = pool.get()
    component = ScriptComponent(path, runtime=runtime)
    obj.add_component(component)
    scene.update(0.1)
    log = component.handle.module.LOG
    assert scene.scheduler.count == 1
    pool.release(obj)
    assert log == ["despawn:True"]
    assert scene.scheduler.count == 0  # задачи отменены
    pool.get()
    assert log == ["despawn:True", "spawn:True"]


def test_rigidbody_velocity_does_not_leak_between_lives() -> None:
    from engine.prefab import prefab_from_dict

    prefab = prefab_from_dict(
        {
            "format": "myengine.prefab",
            "version": 1,
            "name": "b",
            "root": "L0",
            "objects": [
                {
                    "local_id": "L0",
                    "name": "B",
                    "tag": "Untagged",
                    "layer": 1,
                    "active": True,
                    "transform": {
                        "position": [0, 5, 0],
                        "rotation": [0, 0, 0, 1],
                        "scale": [1, 1, 1],
                    },
                    "components": [
                        {"type": "RigidBodyComponent"},
                        {"type": "ColliderComponent"},
                    ],
                }
            ],
        },
        "b",
    )
    scene = Scene("s")
    scene.prefab_loader = lambda guid, path: prefab
    scene.physics  # noqa: B018 - ленивое создание адаптера
    pool = ObjectPool(scene, prefab)

    def fall(obj: object) -> float:
        for _ in range(30):
            scene.fixed_update(1 / 60)
        return obj.transform.position.y  # type: ignore[attr-defined]

    first = pool.get()
    expected = fall(first)
    for _ in range(2):
        pool.release(first)
        again = pool.get()
        assert again is first and again.transform.position.y == 5.0
        assert fall(again) == pytest.approx(expected)
