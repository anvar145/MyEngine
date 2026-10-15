"""Скрипт-API физики: ``self.physics.*`` возвращает простые типы."""

from __future__ import annotations

import pytest

from engine.core.math3d import Vector3
from engine.core.physics_components import (
    CharacterControllerComponent,
    ColliderComponent,
    RigidBodyComponent,
)
from engine.core.scene import Scene
from engine.scripting.script_component import InputState, ScriptComponent

DT = 1.0 / 60.0


def _run(scene: Scene, steps: int) -> None:
    for _ in range(steps):
        scene.fixed_update(DT)
        scene.update(DT)


def _floor(scene: Scene):
    go = scene.create_game_object(
        "Floor", position=Vector3(0, -0.5, 0), scale=Vector3(20, 1, 20)
    )
    go.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    return go


def _g(go, name: str):
    """Глобал модуля скрипта (у ``self`` нет ``__dict__``)."""
    return go.get_component(ScriptComponent).handle.module.__dict__[name]


def _scripted(scene: Scene, source: str, name: str = "Obj", **kw):
    go = scene.create_game_object(name, **kw)
    go.add_component(ScriptComponent, source=source)
    return go


def test_raycast_returns_dict_with_primitives():
    scene = Scene()
    floor = _floor(scene)
    src = (
        "hit = None\n"
        "def on_start(self):\n"
        "    global hit\n"
        "    hit = self.physics.raycast((0, 5, 0), (0, -1, 0), 20)\n"
    )
    go = _scripted(scene, src)
    _run(scene, 2)
    hit = _g(go, "hit")
    assert hit["object"] is floor
    assert hit["distance"] == pytest.approx(5.0, abs=1e-4)
    assert hit["point"][1] == pytest.approx(0.0, abs=1e-4)


def test_raycast_miss_is_none_and_accepts_vector3():
    scene = Scene()
    go = _scripted(scene, "def on_start(self):\n    pass\n")
    _run(scene, 1)
    comp = go.get_component(ScriptComponent)
    assert comp.physics.raycast(Vector3(0, 5, 0), Vector3(0, -1, 0), 10) is None


def test_overlap_sphere_returns_objects():
    scene = Scene()
    box = scene.create_game_object("Box", position=Vector3(1, 0, 0))
    box.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    go = _scripted(scene, "def on_start(self):\n    pass\n")
    _run(scene, 1)
    comp = go.get_component(ScriptComponent)
    assert comp.physics.overlap_sphere((1, 0, 0), 0.5) == [box]
    assert comp.physics.overlap_sphere((9, 0, 0), 0.5) == []


def test_set_gravity_and_get_gravity():
    scene = Scene()
    go = _scripted(scene, "def on_start(self):\n    pass\n")
    comp = go.get_component(ScriptComponent)
    comp.physics.set_gravity(0, -1.0, 0)
    assert comp.physics.get_gravity() == (0.0, -1.0, 0.0)
    comp.physics.set_gravity(Vector3(0, -3, 0))
    assert comp.physics.get_gravity() == (0.0, -3.0, 0.0)


def test_apply_impulse_moves_own_body():
    scene = Scene()
    src = (
        "def on_start(self):\n"
        "    self.physics.set_gravity(0, 0, 0)\n"
        "    self.physics.apply_impulse((2, 0, 0))\n"
    )
    go = scene.create_game_object("Obj", position=Vector3(0, 3, 0))
    go.add_component(ColliderComponent, shape="sphere", radius=0.5)
    go.add_component(RigidBodyComponent, mass=1.0)
    go.add_component(ScriptComponent, source=src)
    _run(scene, 30)
    assert go.transform.world_position.x > 0.5
    vx = go.get_component(ScriptComponent).physics.get_velocity()[0]
    assert vx == pytest.approx(2.0, abs=0.2)


def test_on_fixed_update_hook_is_called_with_dt():
    scene = Scene()
    src = (
        "n = 0\n"
        "def on_fixed_update(self, dt):\n"
        "    global n\n"
        "    n += 1\n"
    )
    go = _scripted(scene, src)
    _run(scene, 10)
    assert _g(go, "n") == 10


def test_character_script_moves_and_jumps():
    scene = Scene()
    _floor(scene)
    src = (
        "grounded = None\n"
        "def on_fixed_update(self, dt):\n"
        "    global grounded\n"
        "    grounded = self.physics.is_grounded()\n"
        "    self.physics.simple_move((2, 0, 0))\n"
    )
    go = _scripted(scene, src, "Hero", position=Vector3(0, 0.93, 0))
    go.add_component(CharacterControllerComponent)
    _run(scene, 60)
    comp = go.get_component(ScriptComponent)
    assert go.transform.world_position.x == pytest.approx(2.0, abs=0.2)
    assert _g(go, "grounded") is True
    assert comp.physics.jump(5.0) is True


def _hero_on_floor() -> tuple[Scene, ScriptComponent]:
    scene = Scene()
    _floor(scene)
    src = (
        "def on_fixed_update(self, dt):\n"
        "    self.physics.simple_move((0, 0, 0))\n"
    )
    go = _scripted(scene, src, "Hero", position=Vector3(0, 0.93, 0))
    go.add_component(CharacterControllerComponent)
    _run(scene, 20)
    comp = go.get_component(ScriptComponent)
    assert comp.physics.is_grounded()
    return scene, comp


def test_script_jump_default_speed_is_five():
    scene, comp = _hero_on_floor()
    assert comp.physics.jump() is True
    ctrl = scene.physics._bindings[comp.game_object.id].character
    assert ctrl.vertical_velocity == 5.0


@pytest.mark.parametrize("speed", [0, -3, -0.5])
def test_script_jump_non_positive_speed_raises(speed):
    _scene, comp = _hero_on_floor()
    with pytest.raises(ValueError, match="speed"):
        comp.physics.jump(speed)


def test_script_jump_in_air_is_false_not_double_jump():
    _scene, comp = _hero_on_floor()
    assert comp.physics.jump(5.0) is True
    assert comp.physics.jump(5.0) is False


def test_bind_input_shares_state():
    scene = Scene()
    src = (
        "w = None\n"
        "def on_update(self, dt):\n"
        "    global w\n"
        "    w = self.input.is_key_down('w')\n"
    )
    go = _scripted(scene, src)
    comp = go.get_component(ScriptComponent)
    shared = InputState()
    comp.bind_input(shared)
    shared.press("w")
    _run(scene, 2)
    assert _g(go, "w") is True
