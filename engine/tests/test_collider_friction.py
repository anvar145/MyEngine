"""Верификация Collider: friction (трение материала).

Наклонная плоскость моделируется наклоном вектора гравитации над плоским полом
(эквивалентно уклону θ без поворота геометрии): ``g = (g·sinθ, −g·cosθ, 0)``.
Скольжение идёт по оси +X. Эффективное трение пары — ``combine_friction``
(по умолчанию ``multiply``).
"""

from __future__ import annotations

import math

import numpy as np
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
G = 9.81


def _tilted(theta_deg: float) -> tuple[float, float, float]:
    t = math.radians(theta_deg)
    return (G * math.sin(t), -G * math.cos(t), 0.0)


def _incline_world(theta_deg: float, mu_block: float, mu_floor: float = 1.0,
                   *, block_modes=("multiply", "max"), floor_modes=("multiply", "max"),
                   v0: float = 0.0, shape=None):
    world = PhysicsWorld(gravity=_tilted(theta_deg))
    world.add_static(
        Collider(BoxShape((200.0, 0.5, 5.0)),
                 PhysicsMaterial(mu_floor, 0.0, floor_modes[0], floor_modes[1])),
        (0.0, -0.5, 0.0),
    )
    rest_y = 0.5
    if isinstance(shape, BoxShape):
        rest_y = shape.half_extents[1]
    elif shape is not None:
        rest_y = shape.radius
    body = RigidBody(position=(0.0, rest_y, 0.0),
                     linear_velocity=(v0, 0.0, 0.0), linear_damping=0.0,
                     angular_damping=0.0, can_sleep=False)
    material = PhysicsMaterial(mu_block, 0.0, block_modes[0], block_modes[1])
    world.add_body(body, Collider(shape or BoxShape((0.5, 0.5, 0.5)), material))
    return world, body


def _slide(world, body, seconds: float) -> float:
    x0 = float(body.position[0])
    for _ in range(round(seconds / DT)):
        world.step()
    return float(body.position[0]) - x0


# --- наклонная плоскость ------------------------------------------------------

def test_zero_friction_slides_with_g_sin_theta() -> None:
    world, body = _incline_world(30.0, mu_block=0.0)
    d = _slide(world, body, 1.0)
    a = G * math.sin(math.radians(30.0))
    assert d == pytest.approx(0.5 * a * 1.0, rel=0.05)  # +≈1.7 % схема Эйлера
    assert body.linear_velocity[0] == pytest.approx(a, rel=0.01)


def test_full_friction_does_not_slide_below_45_degrees() -> None:
    world, body = _incline_world(30.0, mu_block=1.0)
    assert abs(_slide(world, body, 2.0)) < 0.02


def test_full_friction_slides_above_45_degrees() -> None:
    world, body = _incline_world(60.0, mu_block=1.0)
    assert _slide(world, body, 1.0) > 1.0


def test_intermediate_friction_slides_slower() -> None:
    """μ=0.5 при 30°: a = g(sinθ − μ cosθ) = 0.66 м/с²."""
    t = math.radians(30.0)
    world, body = _incline_world(30.0, mu_block=0.5)
    d = _slide(world, body, 1.0)
    a = G * (math.sin(t) - 0.5 * math.cos(t))
    assert d == pytest.approx(0.5 * a, rel=0.2)
    free_world, free = _incline_world(30.0, mu_block=0.0)
    assert d < 0.2 * _slide(free_world, free, 1.0)


def test_distance_decreases_monotonically_with_friction() -> None:
    dists = []
    for mu in (0.0, 0.2, 0.4, 0.5):
        world, body = _incline_world(30.0, mu_block=mu)
        dists.append(_slide(world, body, 1.0))
    assert dists == sorted(dists, reverse=True)
    assert dists[0] > 3 * dists[-1]


def test_critical_angle_is_atan_of_effective_friction() -> None:
    """Эффективное μ = 0.55·1.0: уклон 28° держит, 32° — нет."""
    hold, body = _incline_world(28.0, mu_block=0.55)
    assert abs(_slide(hold, body, 2.0)) < 0.05
    slide, body2 = _incline_world(32.0, mu_block=0.55)
    assert _slide(slide, body2, 2.0) > 0.3


def test_friction_five_holds_on_steep_slope() -> None:
    """friction > 1 допустим: μ=5 держит плоский блок на 60° (tan 60° = 1.73),
    а μ=1 на том же уклоне скользит. Блок плоский, чтобы не опрокидывался."""
    flat = BoxShape((0.5, 0.1, 0.5))
    world, body = _incline_world(60.0, mu_block=5.0, mu_floor=1.0, shape=flat)
    assert abs(_slide(world, body, 2.0)) < 0.05
    world1, body1 = _incline_world(60.0, mu_block=1.0, mu_floor=1.0, shape=flat)
    assert _slide(world1, body1, 1.0) > 1.0


def test_no_separate_static_and_kinetic_friction() -> None:
    """Один коэффициент Кулона: движущийся блок тормозит при μ·cosθ > sinθ
    (a = g(μ cosθ − sinθ) = 0.19 м/с²) и останавливается; покоящийся при том
    же μ не трогается."""
    resting, rb = _incline_world(30.0, mu_block=0.6)
    assert abs(_slide(resting, rb, 2.0)) < 0.02
    moving, mb = _incline_world(30.0, mu_block=0.6, v0=0.5)
    _slide(moving, mb, 1.0)
    v_mid = float(mb.linear_velocity[0])
    assert 0.2 < v_mid < 0.5  # тормозит (ожидаемо 0.5 − 0.19·1 ≈ 0.31)
    _slide(moving, mb, 5.0)
    assert abs(mb.linear_velocity[0]) < 0.05  # и останавливается


# --- плоский пол: торможение ---------------------------------------------------

def _flat_stop_distance(mu_block: float, mu_floor: float = 1.0, v0: float = 5.0):
    world = PhysicsWorld()
    world.add_static(Collider(BoxShape((500.0, 0.5, 5.0)),
                              PhysicsMaterial(mu_floor)), (0.0, -0.5, 0.0))
    body = RigidBody(position=(0.0, 0.5, 0.0), linear_velocity=(v0, 0.0, 0.0),
                     linear_damping=0.0, can_sleep=False)
    world.add_body(body, Collider(BoxShape((0.5, 0.5, 0.5)),
                                  PhysicsMaterial(mu_block)))
    for _ in range(60 * 10):
        world.step()
    return float(body.position[0]), float(body.linear_velocity[0])


def test_zero_friction_keeps_horizontal_speed() -> None:
    world = PhysicsWorld()
    world.add_static(Collider(BoxShape((500.0, 0.5, 5.0)), PhysicsMaterial(0.0)),
                     (0.0, -0.5, 0.0))
    body = RigidBody(position=(0.0, 0.5, 0.0), linear_velocity=(5.0, 0.0, 0.0),
                     linear_damping=0.0, can_sleep=False)
    world.add_body(body, Collider(BoxShape((0.5, 0.5, 0.5)), PhysicsMaterial(0.0)))
    for _ in range(120):
        world.step()
    assert body.linear_velocity[0] == pytest.approx(5.0, abs=0.01)


def test_stop_distance_matches_coulomb_formula() -> None:
    """μ_эфф = 0.5·1.0: s = v²/(2μg) = 25/9.81 = 2.55 м."""
    x, v = _flat_stop_distance(0.5)
    assert abs(v) < 0.05
    assert x == pytest.approx(25.0 / (2 * 0.5 * G), rel=0.1)


def test_stop_distance_inversely_proportional_to_friction() -> None:
    x_low, _ = _flat_stop_distance(0.25)
    x_high, _ = _flat_stop_distance(1.0)
    assert x_low / x_high == pytest.approx(4.0, rel=0.15)


def test_default_pair_friction_is_product_of_defaults() -> None:
    """Умолчания 0.5 · 0.5 = 0.25 (multiply)."""
    world = PhysicsWorld()
    world.add_static(Collider(BoxShape((500.0, 0.5, 5.0))), (0.0, -0.5, 0.0))
    body = RigidBody(position=(0.0, 0.5, 0.0), linear_velocity=(5.0, 0.0, 0.0),
                     linear_damping=0.0, can_sleep=False)
    world.add_body(body, Collider(BoxShape((0.5, 0.5, 0.5))))
    for _ in range(60 * 10):
        world.step()
    assert body.position[0] == pytest.approx(25.0 / (2 * 0.25 * G), rel=0.1)


# --- режимы смешивания (только уровень PhysicsWorld) --------------------------

_MODES = ["multiply", "min", "max", "average"]


@pytest.mark.parametrize(("mode", "mu_eff"), [
    ("multiply", 0.16), ("min", 0.2), ("max", 0.8), ("average", 0.5),
])
def test_friction_combine_mode_gives_expected_effective_mu(mode, mu_eff) -> None:
    """Блок μ=0.2, пол μ=0.8, уклон 30° (tan = 0.577): по режиму —
    multiply 0.16 / min 0.2 скользят, average 0.5 едва скользит, max 0.8 держит."""
    world, body = _incline_world(
        30.0, mu_block=0.2, mu_floor=0.8,
        block_modes=(mode, "max"), floor_modes=(mode, "max"),
    )
    d = _slide(world, body, 1.0)
    t = math.radians(30.0)
    expected_a = max(G * (math.sin(t) - mu_eff * math.cos(t)), 0.0)
    assert d == pytest.approx(0.5 * expected_a, rel=0.25, abs=0.03)


def test_friction_combine_modes_are_all_distinguishable() -> None:
    dist = {}
    for mode in _MODES:
        world, body = _incline_world(
            30.0, mu_block=0.2, mu_floor=0.8,
            block_modes=(mode, "max"), floor_modes=(mode, "max"),
        )
        dist[mode] = _slide(world, body, 1.0)
    assert dist["multiply"] > dist["min"] > dist["average"] > dist["max"]


def test_friction_user_spec_values_a02_b10() -> None:
    """Значения из задания: A=0.2, B=1.0 (уклон 30°). multiply/min → 0.2 скользит,
    max → 1.0 и average → 0.6 держат (0.6 > tan 30° = 0.577)."""
    out = {}
    for mode in _MODES:
        world, body = _incline_world(
            30.0, mu_block=0.2, mu_floor=1.0,
            block_modes=(mode, "max"), floor_modes=(mode, "max"),
        )
        out[mode] = _slide(world, body, 1.0)
    assert out["multiply"] > 1.0 and out["min"] > 1.0
    assert out["max"] < 0.02 and out["average"] < 0.02


def test_mixed_friction_modes_higher_priority_wins() -> None:
    """Режимы коллайдеров разные: побеждает старший по порядку
    (multiply < min < max < average), независимо от того, кто «первый»."""
    for first, second in (("multiply", "average"), ("average", "multiply")):
        world, body = _incline_world(
            30.0, mu_block=0.2, mu_floor=0.8,
            block_modes=(first, "max"), floor_modes=(second, "max"),
        )
        avg_world, avg = _incline_world(
            30.0, mu_block=0.2, mu_floor=0.8,
            block_modes=("average", "max"), floor_modes=("average", "max"),
        )
        assert _slide(world, body, 1.0) == pytest.approx(
            _slide(avg_world, avg, 1.0), rel=1e-6)


# --- вращение (шар катится при трении) ---------------------------------------

def test_sphere_rolls_only_with_friction() -> None:
    slip, sb = _incline_world(20.0, mu_block=0.0, shape=SphereShape(0.5))
    _slide(slip, sb, 1.0)
    roll, rb = _incline_world(20.0, mu_block=0.8, shape=SphereShape(0.5))
    _slide(roll, rb, 1.0)
    assert abs(sb.angular_velocity[2]) < 0.05
    assert abs(rb.angular_velocity[2]) > 1.0
    # катящийся шар медленнее скользящего (энергия уходит во вращение)
    assert rb.position[0] < sb.position[0]


# --- компонент ------------------------------------------------------------------

def _scene_block(theta_deg: float, mu_block: float, mu_floor: float = 1.0):
    scene = Scene()
    scene.physics.world.set_gravity(_tilted(theta_deg))
    floor = scene.create_game_object("Floor", position=Vector3(0, -0.5, 0),
                                     scale=Vector3(400, 1, 10))
    floor.add_component(ColliderComponent, shape="box", size=(1, 1, 1),
                        friction=mu_floor)
    go = scene.create_game_object("Block", position=Vector3(0, 0.5, 0))
    col = go.add_component(ColliderComponent, shape="box", friction=mu_block)
    go.add_component(RigidBodyComponent, can_sleep=False)
    scene.fixed_update(DT)
    return scene, go, col


def _x(scene, go) -> float:
    return float(scene.physics.world.get_body(scene.physics.handle_of(go)).position[0])


def _vx(scene, go) -> float:
    body = scene.physics.world.get_body(scene.physics.handle_of(go))
    return float(body.linear_velocity[0])


def _run(scene, seconds: float) -> None:
    for _ in range(round(seconds / DT)):
        scene.fixed_update(DT)


def test_component_friction_zero_slides_one_holds() -> None:
    scene0, go0, _ = _scene_block(30.0, 0.0)
    scene1, go1, _ = _scene_block(30.0, 1.0)
    _run(scene0, 1.0)
    _run(scene1, 1.0)
    assert _x(scene0, go0) > 2.0
    assert abs(_x(scene1, go1)) < 0.02


def test_component_negative_friction_is_clamped_to_zero() -> None:
    scene, go, _ = _scene_block(30.0, -3.0)
    _run(scene, 1.0)
    assert _x(scene, go) == pytest.approx(0.5 * G * 0.5, rel=0.05)


def test_component_runtime_friction_change_stops_sliding_block() -> None:
    scene, go, col = _scene_block(30.0, 0.0)
    _run(scene, 1.0)
    assert _vx(scene, go) > 4.0
    col.friction = 1.0
    _run(scene, 4.0)
    assert abs(_vx(scene, go)) < 0.05
    x_stop = _x(scene, go)
    _run(scene, 1.0)
    assert _x(scene, go) == pytest.approx(x_stop, abs=0.01)


def test_component_runtime_friction_to_zero_releases_block() -> None:
    scene, go, col = _scene_block(30.0, 1.0)
    _run(scene, 1.0)
    assert abs(_x(scene, go)) < 0.02
    col.friction = 0.0
    _run(scene, 1.0)
    assert _x(scene, go) > 2.0


def test_component_friction_reaches_world_material() -> None:
    scene, go, _ = _scene_block(30.0, 0.37)
    handle = scene.physics.handle_of(go)
    material = scene.physics.world.get_colliders(handle)[0].material
    assert material.friction == pytest.approx(0.37)
    assert (material.friction_combine, material.restitution_combine) == (
        "multiply", "max")
    assert np.isfinite(_x(scene, go))


# --- режимы смешивания в компоненте ----------------------------------------------

def _slide_distance(block_mode: str, floor_mode: str) -> float:
    """Блок μ=0.2 на полу μ=0.8, уклон 30° — путь за 1 с при режимах пары."""
    scene, go, col = _scene_block(30.0, 0.2, 0.8)
    col.friction_combine = block_mode
    scene.find("Floor").get_component(ColliderComponent).friction_combine = floor_mode
    _run(scene, 1.0)
    return _x(scene, go)


def test_component_friction_combine_reaches_world_material() -> None:
    scene, go, col = _scene_block(30.0, 0.5)
    col.friction_combine = "average"
    scene.fixed_update(DT)
    material = scene.physics.world.get_colliders(scene.physics.handle_of(go))[0].material
    assert material.friction_combine == "average"
    assert material.restitution_combine == "max"


@pytest.mark.parametrize(("mode", "slides"), [
    ("multiply", True), ("min", True), ("max", False),
])
def test_component_friction_combine_modes(mode: str, slides: bool) -> None:
    """Оба коллайдера в одном режиме: multiply 0.16 и min 0.2 скользят, max 0.8 держит."""
    distance = _slide_distance(mode, mode)
    assert (distance > 1.0) is slides


def test_component_friction_combine_pair_is_symmetric_and_later_mode_wins() -> None:
    """«Главного» коллайдера нет: побеждает режим, стоящий позже в
    multiply < min < max < average, независимо от того, чей он."""
    max_on_block = _slide_distance("max", "multiply")
    max_on_floor = _slide_distance("multiply", "max")
    assert max_on_block == pytest.approx(max_on_floor, abs=1e-6)
    assert max_on_block < 0.1  # μ_eff = 0.8: блок держится
    assert _slide_distance("multiply", "multiply") > 1.0  # μ_eff = 0.16: скользит


def test_component_unknown_combine_mode_skips_object(caplog) -> None:
    scene, go, col = _scene_block(30.0, 0.5)
    col.friction_combine = "median"
    with caplog.at_level("WARNING"):
        scene.fixed_update(DT)
    assert scene.physics.handle_of(go) is None
    assert any("пропущен" in r.message for r in caplog.records)
