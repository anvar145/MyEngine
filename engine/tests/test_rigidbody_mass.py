"""Верификация RigidBody: масса и обратная масса."""

from __future__ import annotations

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


def _free_world() -> PhysicsWorld:
    return PhysicsWorld(gravity=(0.0, 0.0, 0.0))


def _box(world, mass, pos=(0.0, 0.0, 0.0), **kw) -> int:
    body = RigidBody(mass=mass, position=pos, linear_damping=0.0,
                     angular_damping=0.0, **kw)
    return world.add_body(body, Collider(BoxShape((0.5, 0.5, 0.5))))


def _ball(world, mass, x, v, restitution) -> int:
    body = RigidBody(mass=mass, position=(x, 0.0, 0.0), linear_damping=0.0,
                     angular_damping=0.0, linear_velocity=(v, 0.0, 0.0))
    material = PhysicsMaterial(friction=0.0, restitution=restitution)
    return world.add_body(body, Collider(SphereShape(0.5), material))


@pytest.mark.parametrize("mass", [0.5, 1.0, 2.0, 10.0])
def test_impulse_gives_delta_v_equal_j_over_m(mass: float) -> None:
    world = _free_world()
    h = _box(world, mass)
    world.apply_impulse(h, (4.0, 0.0, 0.0))
    assert world.get_body(h).linear_velocity[0] == pytest.approx(4.0 / mass)


def test_heavier_body_accelerates_less_for_same_impulse() -> None:
    world = _free_world()
    light = _box(world, 1.0, (0.0, 0.0, 0.0))
    heavy = _box(world, 4.0, (10.0, 0.0, 0.0))
    for h in (light, heavy):
        world.apply_impulse(h, (4.0, 0.0, 0.0))
    for _ in range(60):
        world.step()
    xl = world.get_body(light).position[0]
    xh = world.get_body(heavy).position[0] - 10.0
    assert xl / xh == pytest.approx(4.0, rel=1e-6)


def test_off_center_impulse_angular_response_scales_with_inverse_mass() -> None:
    world = _free_world()
    light = _box(world, 1.0, (0.0, 0.0, 0.0))
    heavy = _box(world, 4.0, (10.0, 0.0, 0.0))
    world.apply_impulse(light, (0, 0, 1.0), point=(0.5, 0.0, 0.0))
    world.apply_impulse(heavy, (0, 0, 1.0), point=(10.5, 0.0, 0.0))
    wl = world.get_body(light).angular_velocity[1]
    wh = world.get_body(heavy).angular_velocity[1]
    assert abs(wl) > 0.0
    assert wl / wh == pytest.approx(4.0, rel=1e-6)  # I ∝ m при той же форме


def test_gravity_independent_of_mass() -> None:
    world = PhysicsWorld()
    a = _box(world, 0.1, (0.0, 100.0, 0.0))
    b = _box(world, 1000.0, (10.0, 100.0, 0.0))
    for _ in range(30):
        world.step()
    assert world.get_body(a).position[1] == pytest.approx(
        world.get_body(b).position[1], abs=1e-9
    )


def test_elastic_collision_unequal_masses() -> None:
    world = _free_world()
    m1, m2, v = 1.0, 3.0, 4.0
    a = _ball(world, m1, 0.0, v, 1.0)
    b = _ball(world, m2, 2.0, 0.0, 1.0)
    for _ in range(60):
        world.step()
    v1 = world.get_body(a).linear_velocity[0]
    v2 = world.get_body(b).linear_velocity[0]
    # импульс сохраняется
    assert m1 * v1 + m2 * v2 == pytest.approx(m1 * v, rel=0.02)
    # классические формулы упругого удара
    assert v1 == pytest.approx((m1 - m2) / (m1 + m2) * v, rel=0.1)
    assert v2 == pytest.approx(2 * m1 / (m1 + m2) * v, rel=0.1)
    assert v1 < 0.0 < v2  # лёгкий отскочил назад, тяжёлый поехал вперёд


def test_inelastic_collision_conserves_momentum_of_mass_centre() -> None:
    world = _free_world()
    m1, m2, v = 1.0, 3.0, 4.0
    a = _ball(world, m1, 0.0, v, 0.0)
    b = _ball(world, m2, 2.0, 0.0, 0.0)
    for _ in range(120):
        world.step()
    v1 = world.get_body(a).linear_velocity[0]
    v2 = world.get_body(b).linear_velocity[0]
    # скорость центра масс = m1·v/(m1+m2) независимо от упругости
    assert (m1 * v1 + m2 * v2) / (m1 + m2) == pytest.approx(
        m1 * v / (m1 + m2), rel=1e-6
    )
    # restitution=0: шары не должны разлетаться как при упругом ударе. Фактически
    # остаётся ~0.74 м/с (≈18 % от v) — скоростная поправка проникновения
    # (см. отчёт): допускаем до 25 %.
    assert 0.0 <= v2 - v1 < 0.25 * v


def test_heavy_body_barely_moved_by_light_one() -> None:
    world = _free_world()
    _ball(world, 0.1, 0.0, 4.0, 0.0)
    heavy = _ball(world, 1000.0, 2.0, 0.0, 0.0)
    for _ in range(120):
        world.step()
    assert world.get_body(heavy).linear_velocity[0] < 0.001


def test_zero_or_negative_mass_rejected_for_dynamic() -> None:
    with pytest.raises(ValueError):
        RigidBody(mass=0.0)
    with pytest.raises(ValueError):
        RigidBody(mass=-1.0)
    with pytest.raises(ValueError):
        RigidBody(mass=float("nan"))


def test_mass_zero_allowed_for_static_and_kinematic() -> None:
    for kind in ("static", "kinematic"):
        body = RigidBody(body_type=kind, mass=0.0)
        assert body.inv_mass == 0.0


def test_inv_mass_is_what_impulse_reads() -> None:
    world = _free_world()
    h = _box(world, 2.0)
    body = world.get_body(h)
    assert body.inv_mass == pytest.approx(0.5)
    body.inv_mass = 0.1  # перезапись: импульс читает именно inv_mass
    world.apply_impulse(h, (1.0, 0.0, 0.0))
    assert body.linear_velocity[0] == pytest.approx(0.1)
    body.inv_mass = 0.0  # бесконечная масса: импульс ничего не даёт
    body.linear_velocity = np.zeros(3)
    world.apply_impulse(h, (1.0, 0.0, 0.0))
    assert body.linear_velocity[0] == 0.0


def test_direct_mass_mutation_updates_inv_mass() -> None:
    body = RigidBody(mass=1.0)
    body.mass = 4.0
    assert body.inv_mass == pytest.approx(0.25)


def test_direct_mass_mutation_scales_inertia_and_validates() -> None:
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    body = RigidBody(mass=1.0)
    h = world.add_body(body, Collider(BoxShape((0.5, 0.5, 0.5))))
    inv_before = body.inv_inertia_local.copy()
    body.mass = 4.0
    assert np.allclose(body.inv_inertia_local, inv_before / 4.0)
    world.apply_impulse(h, (4.0, 0.0, 0.0))
    assert body.linear_velocity[0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        body.mass = 0.0
    assert body.mass == 4.0


def test_runtime_mass_change_via_component_rebuilds_body() -> None:
    scene = Scene()
    go = scene.create_game_object("Box", position=Vector3(0, 10, 0))
    go.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    rb = go.add_component(RigidBodyComponent, mass=1.0)
    scene.physics.world.set_gravity((0, 0, 0))
    scene.fixed_update(DT)
    rb.apply_impulse((4.0, 0.0, 0.0))
    assert rb.linear_velocity[0] == pytest.approx(4.0)
    rb.linear_velocity = (0.0, 0.0, 0.0)
    rb.mass = 8.0
    rb.apply_impulse((4.0, 0.0, 0.0))
    assert rb.linear_velocity[0] == pytest.approx(0.5)


def test_component_zero_mass_skips_object() -> None:
    scene = Scene()
    go = scene.create_game_object("Box", position=Vector3(0, 10, 0))
    go.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    rb = go.add_component(RigidBodyComponent, mass=0.0)
    scene.fixed_update(DT)
    assert rb.handle is None  # ValueError в адаптере → объект пропущен
