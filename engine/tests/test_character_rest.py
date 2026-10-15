"""Быстрый путь покоя пассивного шага ``CharacterController``.

Пассивный шаг (``ScenePhysicsAdapter._step_characters`` -> ``CharacterController.step``)
выполняется для каждого персонажа, которого скрипт не двигал. Стоящий на земле
персонаж при этом возвращается ровно в ту же точку, поэтому, пока ничего вокруг не
изменилось, шаг можно не считать: ни ``shapecast``, ни ``move_kinematic``.

* Покой - побитовая неподвижная точка шага; пропуск допустим только при тех же
  входных данных (поза, скорости, параметры, гравитация, окрестность).
* Окрестность - коллайдеры вблизи капсулы; движение вдали быстрый путь не ломает.
* Эквивалентность: пассивный шаг и явный ``update_character(0)`` каждый кадр дают
  одинаковые позы побитово (второй путь быстрого пути не имеет).

Подробности - ``docs/CORE.md`` §7.2 и ``_work/character_improvements_report.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.core.math3d import Vector3
from engine.core.physics_components import (
    CharacterControllerComponent,
    ColliderComponent,
    RigidBodyComponent,
)
from engine.core.scene import Scene
from engine.physics.world import PhysicsWorld

DT = 1.0 / 60.0
REST_Y = 0.92


@pytest.fixture
def sweeps(monkeypatch):
    """Счётчик вызовов ``PhysicsWorld.shapecast`` (вызов проходит к оригиналу)."""
    calls = [0]
    original = PhysicsWorld.shapecast

    def counting(self, *args, **kwargs):
        calls[0] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PhysicsWorld, "shapecast", counting)
    return calls


def _floor(scene: Scene):
    floor = scene.create_game_object(
        "Floor", position=Vector3(0, -0.5, 0), scale=Vector3(40, 1, 40)
    )
    floor.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    return floor


def _hero(scene: Scene, y: float = REST_Y, **fields):
    go = scene.create_game_object("Hero", position=Vector3(0, y, 0))
    comp = go.add_component(CharacterControllerComponent, **fields)
    return go, comp


def _controller(scene: Scene, hero):
    return scene.physics._bindings[hero.id].character


def _run(scene: Scene, steps: int) -> None:
    for _ in range(steps):
        scene.fixed_update(DT)


# --- пропуск шага ------------------------------------------------------------

def test_resting_character_skips_sweeps(sweeps):
    scene = Scene()
    _floor(scene)
    hero, _ = _hero(scene)
    _run(scene, 30)  # осесть
    ctrl = _controller(scene, hero)
    pose = ctrl.position.tobytes()
    sweeps[0] = 0

    _run(scene, 60)

    assert sweeps[0] == 0
    assert ctrl.position.tobytes() == pose
    assert ctrl.grounded is True


def test_resting_character_ignores_distant_movers(sweeps):
    scene = Scene()
    _floor(scene)
    _hero(scene)
    mover = scene.create_game_object("Mover", position=Vector3(50, 1, 50))
    mover.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    mover.add_component(RigidBodyComponent, body_type="kinematic")
    _run(scene, 30)
    sweeps[0] = 0

    for k in range(60):
        mover.transform.position = Vector3(50 + 0.01 * k, 1, 50)  # далеко от героя
        scene.fixed_update(DT)

    assert sweeps[0] == 0


def test_airborne_character_is_not_skipped(sweeps):
    scene = Scene()
    _floor(scene)
    hero, _ = _hero(scene, y=5.0)
    sweeps[0] = 0

    _run(scene, 20)

    assert sweeps[0] >= 20  # падение считается каждый шаг
    assert hero.transform.world_position.y < 5.0


def test_fast_path_is_deterministic():
    def final_pose() -> bytes:
        scene = Scene()
        _floor(scene)
        hero, _ = _hero(scene, y=2.0)
        _run(scene, 120)
        return _controller(scene, hero).position.tobytes()

    assert final_pose() == final_pose()


# --- эквивалентность полному шагу --------------------------------------------

def _scenario_floor_removed(scene, ctx):
    ctx["floor"] = _floor(scene)
    ctx["hero"], _ = _hero(scene)

    def act(step):
        if step == 40:
            scene.destroy_game_object(ctx["floor"])

    return act


def _scenario_platform_rises(scene, ctx):
    platform = scene.create_game_object(
        "Platform", position=Vector3(0, -0.5, 0), scale=Vector3(6, 1, 6)
    )
    platform.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    platform.add_component(RigidBodyComponent, body_type="kinematic")
    ctx["hero"], _ = _hero(scene)

    def act(step):
        if 40 <= step < 70:
            platform.transform.position = Vector3(0, -0.5 + 0.02 * (step - 39), 0)

    return act


def _scenario_wall_appears(scene, ctx):
    _floor(scene)
    ctx["hero"], _ = _hero(scene)

    def act(step):
        if step == 40:
            wall = scene.create_game_object("Wall", position=Vector3(0.5, 1.0, 0))
            wall.add_component(ColliderComponent, shape="box", size=(1, 2, 1))

    return act


def _scenario_box_dropped(scene, ctx):
    _floor(scene)
    ctx["hero"], _ = _hero(scene)
    box = scene.create_game_object("Box", position=Vector3(0, 4.0, 0))
    box.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    box.add_component(RigidBodyComponent, body_type="dynamic", mass=2.0)
    ctx["box"] = box

    return lambda step: None


def _scenario_gravity_flips(scene, ctx):
    _floor(scene)
    ctx["hero"], _ = _hero(scene)

    def act(step):
        if step == 40:
            scene.physics.world.set_gravity((0.0, 9.81, 0.0))

    return act


def _scenario_hover_then_gravity(scene, ctx):
    _floor(scene)
    ctx["hero"], comp = _hero(scene, y=3.0, use_gravity=False)

    def act(step):
        if step == 40:
            comp.use_gravity = True

    return act


def _scenario_teleport_on_floor(scene, ctx):
    _floor(scene)
    ctx["hero"], _ = _hero(scene)

    def act(step):
        if step == 40:
            ctx["hero"].transform.position = Vector3(5.0, REST_Y, 0)

    return act


def _scenario_teleport_off_floor(scene, ctx):
    _floor(scene)
    ctx["hero"], _ = _hero(scene)

    def act(step):
        if step == 40:
            ctx["hero"].transform.position = Vector3(30.0, REST_Y, 0)

    return act


# сценарий -> проверка, что он действительно что-то менял (иначе тест пуст)
_SCENARIOS = {
    "floor_removed": (_scenario_floor_removed, lambda c: c.position[1] < -1.0),
    "platform_rises": (_scenario_platform_rises, lambda c: c.position[1] > REST_Y + 0.5),
    "wall_appears": (_scenario_wall_appears, lambda c: c.position[0] < -0.05),
    "box_dropped": (_scenario_box_dropped, lambda c: True),
    "gravity_flips": (_scenario_gravity_flips, lambda c: c.position[1] > REST_Y + 1.0),
    "hover_then_gravity": (
        _scenario_hover_then_gravity, lambda c: c.position[1] < 1.0
    ),
    "teleport_on_floor": (_scenario_teleport_on_floor, lambda c: c.position[0] > 4.9),
    "teleport_off_floor": (
        _scenario_teleport_off_floor, lambda c: c.position[1] < -1.0
    ),
}


def _trace(name: str, explicit: bool, steps: int = 140):
    """Позы героя (и коробки) по шагам; ``explicit`` - полный шаг скриптом."""
    build, _ = _SCENARIOS[name]
    scene = Scene()
    ctx: dict = {}
    act = build(scene, ctx)
    hero = ctx["hero"]
    trace = []
    for step in range(steps):
        act(step)
        if explicit:
            # пассивный шаг идёт после sync_in; явному нужен тот же порядок
            scene.physics.sync_in(DT)
            scene.physics.update_character(hero, (0.0, 0.0, 0.0))
        scene.fixed_update(DT)
        ctrl = _controller(scene, hero)
        row = [ctrl.position.tobytes(), ctrl.vertical_velocity, ctrl.grounded]
        if "box" in ctx:
            p = ctx["box"].transform.world_position
            row.append(np.array([p.x, p.y, p.z]).tobytes())
        trace.append(row)
    return trace, ctrl


@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_passive_step_matches_explicit_zero_move(name):
    passive, ctrl = _trace(name, explicit=False)
    explicit, _ = _trace(name, explicit=True)

    assert _SCENARIOS[name][1](ctrl), "сценарий ничего не менял"
    for step, (a, b) in enumerate(zip(passive, explicit, strict=True)):
        assert a == b, f"расхождение на шаге {step}"


# --- состояние, которое шаг обязан перезаписать --------------------------------

def _settled_controller():
    from engine.tests.character_kit import make_char, make_world

    world = make_world()
    ctrl = make_char(world, y=0.92)
    for _ in range(10):
        ctrl.step()
        world.step()
    return world, ctrl


def test_step_clears_leftover_horizontal_velocity():
    _, ctrl = _settled_controller()
    ctrl.horizontal_velocity = np.array([3.0, 0.0, 0.0])  # осталось от simple_move

    ctrl.step()

    assert not ctrl.horizontal_velocity.any()


def test_step_recomputes_ground_flag():
    _, ctrl = _settled_controller()
    ctrl.grounded = False  # внешний сброс

    ctrl.step()

    assert ctrl.grounded is True


def test_skipped_step_returns_fresh_zero_motion():
    _, ctrl = _settled_controller()

    first = ctrl.step()
    first[:] = 5.0  # правка результата не должна влиять на следующие шаги
    second = ctrl.step()

    assert not second.any()
