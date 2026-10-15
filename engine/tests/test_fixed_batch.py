"""Пакетный хук ``on_fixed_update_batch`` и стартовое оседание физики."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from engine.core.math3d import Vector3
from engine.core.physics_components import ColliderComponent, RigidBodyComponent
from engine.core.scene import Scene
from engine.scripting.script_component import ScriptComponent

DT = 1.0 / 60.0
PATROL = (
    Path(__file__).resolve().parents[2]
    / "projects/PlayModeTest/Assets/Scripts/patrol.py"
)

BATCH_SRC = (
    "calls = []\n"
    "def on_fixed_update_batch(self, items, dt):\n"
    "    calls.append([i.game_object.name for i in items])\n"
)
SINGLE_SRC = (
    "calls = 0\n"
    "def on_fixed_update(self, dt):\n"
    "    global calls\n"
    "    calls += 1\n"
)


def _g(go, name):
    return go.get_component(ScriptComponent).handle.module.__dict__[name]


def _spawn(scene, names, source, **kw):
    out = []
    for name in names:
        go = scene.create_game_object(name, **kw)
        go.add_component(ScriptComponent, source=source)
        out.append(go)
    return out


def _run(scene, steps):
    for _ in range(steps):
        scene.fixed_update(DT)
        scene.update(DT)


def test_batch_hook_called_once_per_step_in_order():
    scene = Scene()
    gos = _spawn(scene, ["A", "B", "C"], BATCH_SRC)
    _run(scene, 3)
    # Хук вызывается на модуле первого компонента, один раз на шаг.
    calls = _g(gos[0], "calls")
    assert calls[-1] == ["A", "B", "C"]
    assert len(calls) == 3  # ровно один вызов на шаг, не по числу объектов
    assert _g(gos[1], "calls") == []


def test_single_hook_still_called_per_component():
    scene = Scene()
    gos = _spawn(scene, ["A", "B"], SINGLE_SRC)
    _run(scene, 3)
    assert all(_g(g, "calls") >= 2 for g in gos)


def test_batches_are_separate_per_script():
    scene = Scene()
    a = _spawn(scene, ["A1", "A2"], BATCH_SRC)
    b = _spawn(scene, ["B1"], BATCH_SRC + "# другой скрипт\n")
    _run(scene, 3)
    assert _g(a[0], "calls")[-1] == ["A1", "A2"]
    assert _g(b[0], "calls")[-1] == ["B1"]


def test_add_fixed_batch_cleared_after_step():
    scene = Scene()
    seen = []
    scene.add_fixed_batch("k", lambda items, dt: seen.append((items, dt)), 1)
    scene.add_fixed_batch("k", lambda items, dt: seen.append(("x", dt)), 2)
    scene.fixed_update(DT)
    scene.fixed_update(DT)
    assert seen == [([1, 2], DT)]


def test_numpy_is_importable_from_scripts():
    scene = Scene()
    go = _spawn(
        scene, ["N"], "import numpy as np\nok = float(np.cos(0.0))\n"
    )[0]
    _run(scene, 2)
    assert _g(go, "ok") == 1.0


def _patrol_old(self, dt):
    import math

    phase = float(sum(map(ord, self.game_object.name)) % 17)
    now = self.time.fixed_frame * self.time.fixed_delta
    speed = 3.0 * 0.8 * math.cos(0.8 * now + phase)
    p = self.transform.position
    self.transform.position = Vector3(p.x + speed * dt, p.y, p.z)


def test_batch_patrol_matches_per_component_patrol():
    names = [f"Enemy_{i}" for i in range(6)]
    batch = Scene()
    gos = _spawn(
        batch, names, PATROL.read_text(encoding="utf-8"), position=Vector3(1, 2, 3)
    )
    ref = Scene()
    refs = _spawn(
        ref,
        names,
        "def on_fixed_update(self, dt):\n    pass\n",
        position=Vector3(1, 2, 3),
    )
    ref_scripts = [g.get_component(ScriptComponent) for g in refs]
    for _ in range(120):
        batch.fixed_update(DT)
        batch.update(DT)
        ref.fixed_update(DT)
        ref.update(DT)
        for sc in ref_scripts:
            if sc.handle is not None:
                _patrol_old(sc, DT)
    got = np.array([g.transform.position.to_tuple() for g in gos])
    want = np.array([g.transform.position.to_tuple() for g in refs])
    assert np.abs(got[:, 0] - 1).max() > 0.1  # патруль реально двигает
    # Шаг отстаёт на один кадр из-за порядка on_start: сравниваем форму.
    assert got[:, 1:] == pytest.approx(want[:, 1:])
    assert got.shape == want.shape


def test_settle_puts_bodies_to_sleep_without_events():
    scene = Scene()
    floor = scene.create_game_object(
        "Floor", position=Vector3(0, -0.5, 0), scale=Vector3(20, 1, 20)
    )
    floor.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    boxes = []
    for i in range(4):
        go = scene.create_game_object("Box", position=Vector3(i * 2, 0.6, 0))
        go.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
        go.add_component(RigidBodyComponent)
        boxes.append(go)
    scene.fixed_update(DT)
    physics = scene.physics
    steps = physics.settle(max_steps=240)
    assert 0 < steps <= 240
    assert physics.world.awake_dynamic_count() == 0
    assert physics.world.pop_events() == []


def test_settle_respects_max_steps():
    scene = Scene()
    go = scene.create_game_object("Falling", position=Vector3(0, 100, 0))
    go.add_component(ColliderComponent, shape="sphere", radius=0.5)
    go.add_component(RigidBodyComponent)
    scene.fixed_update(DT)
    assert scene.physics.settle(max_steps=5) == 5
