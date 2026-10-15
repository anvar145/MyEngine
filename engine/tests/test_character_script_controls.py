"""Скриптовый API персонажа: ``simple_move()`` по WASD, ``sprint``, ``jump()``.

Фасад ``self.physics`` читает поля ``CharacterControllerComponent`` на каждом
вызове: правка ``move_speed``/``jump_velocity`` в инспекторе действует сразу.
"""

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


def _hero(source: str = _NOOP, y: float = 0.93, **cc):
    scene = Scene()
    floor = scene.create_game_object(
        "Floor", position=Vector3(0, -0.5, 0), scale=Vector3(80, 1, 80)
    )
    floor.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    go = scene.create_game_object("Hero", position=Vector3(0, y, 0))
    script = go.add_component(ScriptComponent, source=source)
    comp = go.add_component(CharacterControllerComponent, **cc)
    return scene, go, script, comp


def _step_velocity(scene, script, *keys: str) -> tuple[float, float, float]:
    """Скорость (м/с) за один шаг ``simple_move()`` с зажатыми ``keys``."""
    script.input.clear()
    for key in keys:
        script.input.press(key)
    moved = script.physics.simple_move()
    scene.fixed_update(DT)
    return (moved[0] / DT, moved[1] / DT, moved[2] / DT)


def _grounded_hero(**cc):
    scene, go, script, comp = _hero(**cc)
    _run(scene, 10)
    return scene, go, script, comp


# --- simple_move() без аргумента: WASD ---------------------------------------

def test_simple_move_without_argument_and_no_keys_stands_still():
    scene, go, script, _ = _grounded_hero()
    vx, _, vz = _step_velocity(scene, script)
    assert (vx, vz) == (0.0, 0.0)
    assert go.transform.world_position.x == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    ("key", "axis", "sign"),
    [("d", 0, 1.0), ("a", 0, -1.0), ("s", 2, 1.0), ("w", 2, -1.0)],
)
def test_wasd_directions_use_move_speed(key, axis, sign):
    scene, _, script, _ = _grounded_hero()
    v = _step_velocity(scene, script, key)
    assert v[axis] == pytest.approx(sign * 4.5, rel=0.01)
    assert v[2 - axis] == pytest.approx(0.0, abs=1e-6)


def test_diagonal_is_not_faster():
    scene, _, script, _ = _grounded_hero()
    vx, _, vz = _step_velocity(scene, script, "w", "d")
    assert (vx * vx + vz * vz) ** 0.5 == pytest.approx(4.5, rel=0.01)


def test_opposite_keys_cancel():
    scene, _, script, _ = _grounded_hero()
    vx, _, vz = _step_velocity(scene, script, "a", "d", "w", "s")
    assert (vx, vz) == (0.0, 0.0)


def test_move_speed_edit_is_live_for_simple_move():
    scene, _, script, comp = _grounded_hero()
    assert _step_velocity(scene, script, "d")[0] == pytest.approx(4.5, rel=0.01)
    comp.move_speed = 8.0
    assert _step_velocity(scene, script, "d")[0] == pytest.approx(8.0, rel=0.01)


def test_explicit_none_is_the_same_as_no_argument():
    _, _, script, _ = _grounded_hero()
    script.input.press("d")
    moved = script.physics.simple_move(None)
    assert moved[0] / DT == pytest.approx(4.5, rel=0.01)


def test_explicit_velocity_is_a_velocity_not_a_direction():
    """Явный аргумент - скорость в м/с: ``move_speed`` его не масштабирует."""
    scene, _, script, _ = _grounded_hero(move_speed=1.0)
    script.input.press("d")  # клавиши явный вызов не читает
    moved = script.physics.simple_move((3.0, 0.0, 0.0))
    assert moved[0] / DT == pytest.approx(3.0, rel=0.01)
    scene.fixed_update(DT)


# --- sprint ------------------------------------------------------------------

def test_sprint_multiplies_speed_until_switched_off():
    scene, _, script, _ = _grounded_hero()
    script.physics.sprint(True)
    assert _step_velocity(scene, script, "d")[0] == pytest.approx(4.5 * 1.6, rel=0.01)
    assert _step_velocity(scene, script, "d")[0] == pytest.approx(4.5 * 1.6, rel=0.01)
    script.physics.sprint(False)
    assert _step_velocity(scene, script, "d")[0] == pytest.approx(4.5, rel=0.01)


def test_sprint_without_argument_turns_it_on():
    scene, _, script, _ = _grounded_hero()
    script.physics.sprint()
    assert _step_velocity(scene, script, "d")[0] == pytest.approx(4.5 * 1.6, rel=0.01)


def test_sprint_multiplier_edit_is_live():
    scene, _, script, comp = _grounded_hero()
    script.physics.sprint(True)
    comp.sprint_multiplier = 2.0
    assert _step_velocity(scene, script, "d")[0] == pytest.approx(9.0, rel=0.01)


def test_sprint_does_not_speed_up_explicit_velocity():
    scene, _, script, _ = _grounded_hero()
    script.physics.sprint(True)
    moved = script.physics.simple_move((3.0, 0.0, 0.0))
    assert moved[0] / DT == pytest.approx(3.0, rel=0.01)
    scene.fixed_update(DT)


# --- jump() ------------------------------------------------------------------

def test_jump_without_argument_uses_jump_velocity():
    scene, go, script, _ = _grounded_hero(jump_velocity=6.0)
    assert script.physics.jump() is True
    start = go.transform.world_position.y
    peak = start
    for _ in range(90):
        script.physics.simple_move((0, 0, 0))
        scene.fixed_update(DT)
        peak = max(peak, go.transform.world_position.y)
    assert peak - start == pytest.approx(6.0 ** 2 / (2 * 9.81), abs=0.1)


def test_jump_velocity_edit_is_live():
    scene, _, script, comp = _grounded_hero()
    comp.jump_velocity = 8.0
    assert script.physics.jump() is True
    ctrl = scene.physics._bindings[comp.game_object.id].character
    assert ctrl.vertical_velocity == pytest.approx(8.0)


def test_jump_with_explicit_speed_is_unchanged():
    scene, _, script, comp = _grounded_hero()
    assert script.physics.jump(3.5) is True
    ctrl = scene.physics._bindings[comp.game_object.id].character
    assert ctrl.vertical_velocity == pytest.approx(3.5)


def test_jump_in_the_air_is_refused():
    _, _, script, _ = _hero(y=10.0)
    script.physics.move_character((0, 0, 0))
    assert script.physics.jump() is False


def test_jump_non_positive_speed_raises():
    _, _, script, _ = _grounded_hero()
    with pytest.raises(ValueError, match="speed"):
        script.physics.jump(0)


# --- set_air_control ---------------------------------------------------------

def test_set_air_control_changes_how_fast_the_air_speed_follows_input():
    scene, _, script, comp = _hero(y=40.0)
    script.physics.set_air_control(1.0)
    assert _step_velocity(scene, script, "d")[0] == pytest.approx(4.5, rel=0.01)
    script.physics.set_air_control(None)
    comp.move_speed = 0.0  # ввод пропал: при air_control=0.3 остаётся 70 % скорости
    assert _step_velocity(scene, script, "d")[0] == pytest.approx(4.5 * 0.7, rel=0.02)


def test_set_air_control_does_not_touch_the_component_field():
    _, _, script, comp = _hero()
    script.physics.set_air_control(0.9)
    assert comp.air_control == 0.3
    assert comp.air_control_override == 0.9


@pytest.mark.parametrize("bad", [-0.5, 1.5, float("nan")])
def test_set_air_control_rejects_out_of_range(bad):
    _, _, script, comp = _hero()
    with pytest.raises(ValueError, match="air_control"):
        script.physics.set_air_control(bad)
    assert comp.air_control_override is None


# --- ошибки ------------------------------------------------------------------

@pytest.mark.parametrize(
    "call",
    [
        lambda p: p.simple_move(),
        lambda p: p.sprint(True),
        lambda p: p.set_air_control(0.5),
        lambda p: p.jump(),
    ],
    ids=["simple_move", "sprint", "set_air_control", "jump"],
)
def test_new_methods_without_component_raise_runtime_error(call):
    scene = Scene()
    go = scene.create_game_object("Plain")
    script = go.add_component(ScriptComponent, source=_NOOP)
    with pytest.raises(RuntimeError, match="CharacterControllerComponent"):
        call(script.physics)


def test_simple_move_without_argument_reports_registration_failure():
    _, _, script, _ = _hero(radius=0.0)
    with pytest.raises(RuntimeError, match="не зарегистрирован"):
        script.physics.simple_move()


# --- скрипт целиком ----------------------------------------------------------

_SCRIPT = """\
def on_fixed_update(self, dt):
    keys = self.input
    self.physics.sprint(keys.is_key_down("shift"))
    if keys.is_key_down("space"):
        self.physics.jump()
    self.physics.simple_move()
"""


def test_script_with_all_new_methods_walks_sprints_and_jumps():
    scene, go, script, _ = _hero(_SCRIPT)
    _run(scene, 20)  # встать на землю
    assert script.physics.is_grounded()

    script.input.press("d")
    x0 = go.transform.world_position.x
    _run(scene, 60)
    walked = go.transform.world_position.x - x0
    assert walked == pytest.approx(4.5, rel=0.03)

    script.input.press("shift")
    x1 = go.transform.world_position.x
    _run(scene, 60)
    sprinted = go.transform.world_position.x - x1
    assert sprinted == pytest.approx(4.5 * 1.6, rel=0.03)

    script.input.press("space")
    y0 = go.transform.world_position.y
    _run(scene, 1)
    script.input.release("space")
    assert go.transform.world_position.y > y0
    assert not script.physics.is_grounded()
