"""Скриптовый API персонажа: ``self.physics.is_grounded/move_character/simple_move/jump``."""

from __future__ import annotations

import pytest

from engine.core.math3d import Vector3
from engine.core.physics_components import (
    CharacterControllerComponent,
    ColliderComponent,
)
from engine.core.scene import Scene
from engine.scripting.script_component import ScriptComponent

DT = 1.0 / 60.0
_NOOP = "def on_start(self):\n    pass\n"


def _run(scene: Scene, steps: int) -> None:
    for _ in range(steps):
        scene.fixed_update(DT)
        scene.update(DT)


def _scene_with_floor() -> Scene:
    scene = Scene()
    floor = scene.create_game_object(
        "Floor", position=Vector3(0, -0.5, 0), scale=Vector3(40, 1, 40)
    )
    floor.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    return scene


def _hero(scene: Scene, source: str = _NOOP, **cc):
    go = scene.create_game_object("Hero", position=Vector3(0, 0.93, 0))
    go.add_component(ScriptComponent, source=source)
    go.add_component(CharacterControllerComponent, **cc)
    return go, go.get_component(ScriptComponent)


def _standing():
    scene = _scene_with_floor()
    src = "def on_fixed_update(self, dt):\n    self.physics.simple_move((0, 0, 0))\n"
    go, comp = _hero(scene, src)
    _run(scene, 20)
    return scene, go, comp


def test_is_grounded_returns_bool():
    _scene, _go, comp = _standing()
    result = comp.physics.is_grounded()
    assert result is True


def test_is_grounded_false_in_air():
    scene = _scene_with_floor()
    go, comp = _hero(scene)
    go.transform.position = Vector3(0, 6.0, 0)
    comp.physics.move_character((0, 0, 0))
    assert comp.physics.is_grounded() is False


def test_simple_move_returns_tuple_of_floats():
    scene = _scene_with_floor()
    _go, comp = _hero(scene)
    actual = comp.physics.simple_move((1.0, 0.0, 0.0))
    assert isinstance(actual, tuple) and len(actual) == 3
    assert all(isinstance(v, float) for v in actual)
    assert actual[0] == pytest.approx(1.0 * DT, abs=1e-6)


def test_simple_move_accepts_vector3_and_list():
    scene = _scene_with_floor()
    _go, comp = _hero(scene)
    a = comp.physics.simple_move(Vector3(1.0, 0.0, 0.0))
    b = comp.physics.simple_move([1.0, 0.0, 0.0])
    assert a[0] == pytest.approx(b[0], abs=1e-6)


def test_simple_move_writes_pose_to_transform():
    scene = _scene_with_floor()
    go, comp = _hero(scene)
    before = go.transform.world_position.x
    for _ in range(30):
        comp.physics.simple_move((3.0, 0.0, 0.0))
    assert go.transform.world_position.x == pytest.approx(before + 3.0 * 0.5,
                                                          abs=0.05)


def test_simple_move_ignores_y_of_velocity():
    scene = _scene_with_floor()
    go, comp = _hero(scene)
    for _ in range(20):
        comp.physics.simple_move((0.0, 50.0, 0.0))
    assert go.transform.world_position.y < 1.5  # вверх «скоростью» не улететь


def test_simple_move_called_twice_per_step_is_two_steps():
    """Документируем: каждый вызов — отдельный шаг dt (не идемпотентно за кадр)."""
    scene = _scene_with_floor()
    _go, comp = _hero(scene)
    a = comp.physics.simple_move((1.0, 0.0, 0.0))
    b = comp.physics.simple_move((1.0, 0.0, 0.0))
    assert a[0] + b[0] == pytest.approx(2.0 * DT, abs=1e-6)


def test_move_character_moves_by_motion_and_returns_actual():
    scene = _scene_with_floor()
    go, comp = _hero(scene)
    actual = comp.physics.move_character((0.5, 0.0, 0.0))
    assert actual[0] == pytest.approx(0.5, abs=1e-6)
    assert go.transform.world_position.x == pytest.approx(0.5, abs=1e-5)


def test_move_character_does_not_apply_gravity():
    scene = _scene_with_floor()
    go, comp = _hero(scene)
    go.transform.position = Vector3(0, 5.0, 0)
    comp.physics.move_character((0.0, 0.0, 0.0))
    comp.physics.move_character((0.0, 0.0, 0.0))
    assert go.transform.world_position.y == pytest.approx(5.0, abs=1e-5)


def test_jump_returns_true_on_ground_and_false_in_air():
    _scene, _go, comp = _standing()
    assert comp.physics.jump() is True
    assert comp.physics.jump() is False


def test_jump_actually_rises_and_lands():
    scene, go, comp = _standing()
    y0 = go.transform.world_position.y
    comp.physics.jump(5.0)
    peak = y0
    for _ in range(120):
        scene.fixed_update(DT)
        peak = max(peak, go.transform.world_position.y)
    assert peak > y0 + 0.9
    assert go.transform.world_position.y == pytest.approx(y0, abs=0.03)
    assert comp.physics.is_grounded()


@pytest.mark.parametrize(
    "method,args",
    [
        ("is_grounded", ()),
        ("move_character", ((0, 0, 0),)),
        ("simple_move", ((0, 0, 0),)),
        ("jump", ()),
    ],
)
def test_character_methods_without_component_raise_runtime_error(method, args):
    scene = _scene_with_floor()
    go = scene.create_game_object("Plain")
    go.add_component(ScriptComponent, source=_NOOP)
    comp = go.get_component(ScriptComponent)
    with pytest.raises(RuntimeError, match="CharacterControllerComponent"):
        getattr(comp.physics, method)(*args)


def test_invalid_size_reports_registration_failure_to_script():
    scene = _scene_with_floor()
    _go, comp = _hero(scene, radius=0.0)
    with pytest.raises(RuntimeError, match="не зарегистрирован"):
        comp.physics.simple_move((1, 0, 0))


def test_invalid_vector_raises_value_error():
    scene = _scene_with_floor()
    _go, comp = _hero(scene)
    with pytest.raises(ValueError):
        comp.physics.simple_move((1.0, 0.0))  # два числа вместо трёх


def test_script_driven_walk_in_hooks_reaches_target():
    scene = _scene_with_floor()
    src = (
        "def on_fixed_update(self, dt):\n"
        "    self.physics.simple_move((2, 0, 0))\n"
    )
    go, _comp = _hero(scene, src)
    _run(scene, 60)
    assert go.transform.world_position.x == pytest.approx(2.0, abs=0.2)


def test_jump_speed_defaults_to_none_everywhere():
    """``jump(speed=None)`` одинаково у компонента, адаптера и ``self.physics``.

    ``None`` значит ``jump_velocity`` компонента (по умолчанию 5.0).
    """
    import inspect

    from engine.core.physics_adapter import ScenePhysicsAdapter
    from engine.scripting.physics_api import ScriptPhysics

    for func in (CharacterControllerComponent.jump, ScenePhysicsAdapter.jump_character,
                 ScriptPhysics.jump):
        assert inspect.signature(func).parameters["speed"].default is None
    assert CharacterControllerComponent().jump_velocity == 5.0


def test_component_jump_without_argument_uses_default_speed():
    scene, go, _comp = _standing()
    cc = go.get_component(CharacterControllerComponent)
    assert cc.jump() is True
    ctrl = scene.physics._bindings[go.id].character
    assert ctrl.vertical_velocity == 5.0


def test_component_jump_non_positive_speed_raises():
    _scene, go, _comp = _standing()
    with pytest.raises(ValueError, match="speed"):
        go.get_component(CharacterControllerComponent).jump(0)
