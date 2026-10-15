"""Верификация Collider: restitution (упругость материала).

Мера — высота первого отскока шара радиуса 0.25 над полом при падении с 1 м.
Упругость пары — ``combine_restitution`` (по умолчанию ``max``: упругость пола
тоже важна, поэтому в большинстве тестов у пола 0).
"""

from __future__ import annotations

import pytest

from engine.core.math3d import Vector3
from engine.core.physics_components import ColliderComponent, RigidBodyComponent
from engine.core.scene import Scene
from engine.physics.body import RigidBody
from engine.physics.collider import Collider
from engine.physics.material import PhysicsMaterial
from engine.physics.shape import BoxShape, SphereShape
from engine.physics.world import PhysicsWorld

DT = 1.0 / 60.0
RADIUS = 0.25


def _first_bounce_apex(step, get_y, get_vy, steps: int = 400) -> float:
    """Высота нижней точки шара в апексе первого отскока (0 — отскока не было)."""
    descended = False
    bounced = False
    apex = 0.0
    for _ in range(steps):
        step()
        vy = get_vy()
        if vy < -0.05:
            descended = True
            if bounced:
                break  # снова падает: первый отскок завершён
        elif descended and vy > 0.05:
            bounced = True
        if bounced:
            apex = max(apex, get_y() - RADIUS)
    return apex


def _drop(e_ball: float, e_floor: float = 0.0, height: float = 1.0,
          ball_modes=("multiply", "max"), floor_modes=("multiply", "max")) -> float:
    world = PhysicsWorld()
    world.add_static(
        Collider(BoxShape((50.0, 0.5, 50.0)),
                 PhysicsMaterial(0.5, e_floor, floor_modes[0], floor_modes[1])),
        (0.0, -0.5, 0.0),
    )
    ball = RigidBody(position=(0.0, height + RADIUS, 0.0), can_sleep=False)
    world.add_body(ball, Collider(
        SphereShape(RADIUS),
        PhysicsMaterial(0.5, e_ball, ball_modes[0], ball_modes[1])))
    return _first_bounce_apex(world.step, lambda: float(ball.position[1]),
                              lambda: float(ball.linear_velocity[1]))


# --- величина отскока ---------------------------------------------------------

def test_zero_restitution_has_no_bounce() -> None:
    assert _drop(0.0) < 0.02


def test_half_restitution_bounces_to_quarter_height() -> None:
    assert _drop(0.5) == pytest.approx(0.25, rel=0.15)


def test_full_restitution_bounces_back_to_drop_height() -> None:
    assert _drop(1.0) == pytest.approx(1.0, rel=0.1)


def test_full_restitution_does_not_gain_energy() -> None:
    assert _drop(1.0) <= 1.05


def test_bounce_height_grows_monotonically_with_restitution() -> None:
    apexes = [_drop(e) for e in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert apexes == sorted(apexes)
    assert apexes[-1] > 10 * max(apexes[0], 0.01)


@pytest.mark.parametrize("e", [0.2, 0.4, 0.6, 0.8])
def test_bounce_height_is_e_squared_times_drop_height(e: float) -> None:
    assert _drop(e) == pytest.approx(e * e, rel=0.2, abs=0.02)


def test_floor_restitution_counts_with_default_max_mode() -> None:
    """Шар e=0 на упругом полу e=0.8 отскакивает (режим max)."""
    assert _drop(0.0, e_floor=0.8) == pytest.approx(0.64, rel=0.15)


# --- режимы смешивания (только уровень PhysicsWorld) -------------------------

@pytest.mark.parametrize(("mode", "e_eff"), [
    ("multiply", 0.16), ("min", 0.2), ("max", 0.8), ("average", 0.5),
])
def test_restitution_combine_mode(mode: str, e_eff: float) -> None:
    """Шар 0.2, пол 0.8 — результат по режиму; высота ≈ e_eff²."""
    apex = _drop(0.2, e_floor=0.8, ball_modes=("multiply", mode),
                 floor_modes=("multiply", mode))
    assert apex == pytest.approx(e_eff * e_eff, rel=0.25, abs=0.02)


def test_restitution_combine_modes_are_all_distinguishable() -> None:
    apex = {
        mode: _drop(0.2, e_floor=0.8, ball_modes=("multiply", mode),
                    floor_modes=("multiply", mode))
        for mode in ("multiply", "min", "max", "average")
    }
    assert apex["max"] > apex["average"] > apex["min"] > apex["multiply"]


@pytest.mark.parametrize(("ball_mode", "floor_mode"), [
    ("multiply", "average"), ("average", "multiply"),
    ("min", "max"), ("max", "min"),
])
def test_mixed_restitution_modes_older_in_order_wins(ball_mode, floor_mode) -> None:
    """При разных режимах побеждает старший в (multiply, min, max, average),
    а не «первый/второй коллайдер»: пары меняются местами — результат тот же."""
    winner = max((ball_mode, floor_mode),
                 key=("multiply", "min", "max", "average").index)
    expected = _drop(0.2, e_floor=0.8, ball_modes=("multiply", winner),
                     floor_modes=("multiply", winner))
    got = _drop(0.2, e_floor=0.8, ball_modes=("multiply", ball_mode),
                floor_modes=("multiply", floor_mode))
    assert got == pytest.approx(expected, rel=1e-6)


# --- порог 1 м/с -----------------------------------------------------------------

def test_drop_from_one_centimeter_does_not_bounce() -> None:
    """v = √(2·9.81·0.01) = 0.44 м/с < 1 м/с."""
    assert _drop(1.0, height=0.01) < 0.002


def test_drop_below_threshold_speed_does_not_bounce() -> None:
    """h = 0.04 м → v = 0.89 м/с."""
    assert _drop(1.0, height=0.04) < 0.003


def test_drop_above_threshold_speed_bounces() -> None:
    """h = 0.08 м → v = 1.25 м/с."""
    assert _drop(1.0, height=0.08) > 0.03


def test_drop_from_one_meter_bounces() -> None:
    assert _drop(1.0, height=1.0) > 0.5


# --- компонент ------------------------------------------------------------------

def _scene_ball(e_ball: float, e_floor: float = 0.0, height: float = 1.0):
    scene = Scene()
    floor = scene.create_game_object("Floor", position=Vector3(0, -0.5, 0),
                                     scale=Vector3(100, 1, 100))
    floor.add_component(ColliderComponent, shape="box", size=(1, 1, 1),
                        restitution=e_floor)
    go = scene.create_game_object("Ball", position=Vector3(0, height + RADIUS, 0))
    col = go.add_component(ColliderComponent, shape="sphere", radius=RADIUS,
                           restitution=e_ball)
    go.add_component(RigidBodyComponent, can_sleep=False)
    scene.fixed_update(DT)
    return scene, go, col


def _scene_apex(scene, go, steps: int = 400) -> float:
    handle = scene.physics.handle_of(go)
    body = scene.physics.world.get_body(handle)
    return _first_bounce_apex(lambda: scene.fixed_update(DT),
                              lambda: float(body.position[1]),
                              lambda: float(body.linear_velocity[1]), steps)


def test_component_restitution_zero_and_one() -> None:
    scene0, go0, _ = _scene_ball(0.0)
    scene1, go1, _ = _scene_ball(1.0)
    assert _scene_apex(scene0, go0) < 0.02
    assert _scene_apex(scene1, go1) == pytest.approx(1.0, rel=0.1)


def test_component_restitution_half() -> None:
    scene, go, _ = _scene_ball(0.5)
    assert _scene_apex(scene, go) == pytest.approx(0.25, rel=0.15)


def test_component_restitution_above_one_is_clamped() -> None:
    scene, go, col = _scene_ball(1.5)
    handle = scene.physics.handle_of(go)
    assert scene.physics.world.get_colliders(handle)[0].material.restitution == 1.0
    assert col.restitution == 1.5  # в компоненте значение не меняется
    assert _scene_apex(scene, go) <= 1.05


def test_component_negative_restitution_is_clamped_to_zero() -> None:
    scene, go, _ = _scene_ball(-0.5)
    handle = scene.physics.handle_of(go)
    assert scene.physics.world.get_colliders(handle)[0].material.restitution == 0.0
    assert _scene_apex(scene, go) < 0.02


def test_physics_material_rejects_out_of_range_restitution() -> None:
    for value in (-0.1, 1.1):
        with pytest.raises(ValueError):
            PhysicsMaterial(restitution=value)


def test_component_floor_restitution_matters_under_default_max() -> None:
    scene, go, _ = _scene_ball(0.0, e_floor=0.8)
    assert _scene_apex(scene, go) == pytest.approx(0.64, rel=0.15)


def test_component_runtime_restitution_change() -> None:
    scene, go, col = _scene_ball(0.0)
    assert _scene_apex(scene, go) < 0.02
    col.restitution = 1.0
    scene.fixed_update(DT)
    handle = scene.physics.handle_of(go)
    body = scene.physics.world.get_body(handle)
    # поднять шар и бросить заново (позу задаёт трансформ, скорость сбрасываем)
    go.transform.position = Vector3(0, 1.0 + RADIUS, 0)
    body.linear_velocity[:] = 0.0
    scene.fixed_update(DT)
    assert _scene_apex(scene, go) == pytest.approx(1.0, rel=0.15)


def _apex_with_modes(ball_mode: str, floor_mode: str) -> float:
    """Шар e=0.2 над полом e=0.8 при заданных режимах упругости."""
    scene, go, col = _scene_ball(0.2, e_floor=0.8)
    col.restitution_combine = ball_mode
    scene.find("Floor").get_component(ColliderComponent).restitution_combine = floor_mode
    scene.fixed_update(DT)
    return _scene_apex(scene, go)


@pytest.mark.parametrize(("mode", "e_eff"), [
    ("multiply", 0.16), ("min", 0.2), ("max", 0.8), ("average", 0.5),
])
def test_component_restitution_combine_modes(mode: str, e_eff: float) -> None:
    assert _apex_with_modes(mode, mode) == pytest.approx(e_eff * e_eff, rel=0.25,
                                                         abs=0.02)


def test_component_restitution_combine_pair_is_symmetric() -> None:
    """Режимы разные: побеждает старший в multiply < min < max < average."""
    a = _apex_with_modes("multiply", "average")
    b = _apex_with_modes("average", "multiply")
    assert a == pytest.approx(b, abs=1e-6)
    assert a == pytest.approx(0.25, rel=0.25)  # average: e = 0.5


def test_component_default_restitution_combine_is_max() -> None:
    assert ColliderComponent().restitution_combine == "max"
    assert ColliderComponent().friction_combine == "multiply"
