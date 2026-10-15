"""Тесты sequential-impulse солвера."""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.physics.body import RigidBody
from engine.physics.shape import BoxShape, SphereShape
from engine.physics.solver import ContactConstraint, color_contacts, solve_island


def _dynamic_sphere(pos, vel=(0, 0, 0), mass=1.0, radius=0.5) -> RigidBody:
    body = RigidBody(mass=mass, position=pos, linear_velocity=vel)
    body.set_inertia(SphereShape(radius).inertia_tensor(mass))
    return body


def _static() -> RigidBody:
    return RigidBody(body_type="static")


def test_color_contacts_no_shared_movable_body() -> None:
    a = [0, 1, 2, 0]
    b = [1, 2, 3, 3]
    colors = color_contacts(a, b, [True] * 4)
    for group in colors:
        bodies = [x for i in group for x in (a[i], b[i])]
        assert len(bodies) == len(set(bodies))
    assert sorted(i for g in colors for i in g) == [0, 1, 2, 3]


def test_color_contacts_static_shared() -> None:
    # тело 0 статичное — все контакты с ним могут лежать в одном цвете
    colors = color_contacts([0, 0, 0], [1, 2, 3], [False, True, True, True])
    assert len(colors) == 1


def test_head_on_elastic_spheres_swap_velocities() -> None:
    a = _dynamic_sphere((0, 0, 0), (2, 0, 0))
    b = _dynamic_sphere((1.0, 0, 0), (-2, 0, 0))
    c = ContactConstraint(0, 1, np.array([1.0, 0, 0]), np.array([0.5, 0, 0]), 0.0,
                          friction=0.0, restitution=1.0)
    solve_island([a, b], [c], 1 / 60)
    assert a.linear_velocity[0] == pytest.approx(-2.0, abs=1e-6)
    assert b.linear_velocity[0] == pytest.approx(2.0, abs=1e-6)


def test_inelastic_collision_stops_relative_motion() -> None:
    a = _dynamic_sphere((0, 0, 0), (2, 0, 0))
    b = _dynamic_sphere((1.0, 0, 0), (0, 0, 0))
    c = ContactConstraint(0, 1, np.array([1.0, 0, 0]), np.array([0.5, 0, 0]), 0.0,
                          friction=0.0)
    solve_island([a, b], [c], 1 / 60)
    assert a.linear_velocity[0] == pytest.approx(1.0, abs=1e-6)
    assert b.linear_velocity[0] == pytest.approx(1.0, abs=1e-6)


def test_restitution_below_threshold_ignored() -> None:
    ball = _dynamic_sphere((0, 0.5, 0), (0, -0.5, 0))
    floor = _static()
    c = ContactConstraint(1, 0, np.array([0, -1.0, 0]), np.array([0, 0, 0]), 0.0,
                          friction=0.0, restitution=1.0)
    # нормаль A→B: A=floor(1)? используем порядок (floor, ball)
    c = ContactConstraint(0, 1, np.array([0, 1.0, 0]), np.array([0, 0, 0]), 0.0,
                          friction=0.0, restitution=1.0)
    solve_island([floor, ball], [c], 1 / 60)
    assert ball.linear_velocity[1] == pytest.approx(0.0, abs=1e-6)


def test_ball_bounces_on_floor() -> None:
    ball = _dynamic_sphere((0, 0.5, 0), (0, -4.0, 0))
    floor = _static()
    c = ContactConstraint(0, 1, np.array([0, 1.0, 0]), np.array([0, 0, 0]), 0.0,
                          friction=0.0, restitution=0.5)
    solve_island([floor, ball], [c], 1 / 60)
    assert ball.linear_velocity[1] == pytest.approx(2.0, abs=1e-6)


def test_friction_coulomb_limit() -> None:
    # ящик скользит по полу, нормальный импульс создаётся давлением (скорость вниз)
    box = RigidBody(mass=1.0, position=(0, 0.5, 0), linear_velocity=(3.0, -1.0, 0))
    box.set_inertia(BoxShape((0.5, 0.5, 0.5)).inertia_tensor(1.0))
    floor = _static()
    cs = [
        ContactConstraint(0, 1, np.array([0, 1.0, 0]), np.array([x, 0, z]), 0.0,
                          friction=0.5)
        for x in (-0.5, 0.5) for z in (-0.5, 0.5)
    ]
    solve_island([floor, box], cs, 1 / 60, iterations=16)
    jn = sum(c.normal_impulse for c in cs)
    assert jn == pytest.approx(1.0, rel=1e-2)
    # потеря горизонтальной скорости ограничена mu * Jn
    assert box.linear_velocity[0] == pytest.approx(3.0 - 0.5 * jn, abs=1e-2)
    assert abs(box.linear_velocity[1]) < 1e-6


def test_static_bodies_not_modified() -> None:
    floor = _static()
    ball = _dynamic_sphere((0, 0.5, 0), (0, -4.0, 0))
    c = ContactConstraint(0, 1, np.array([0, 1.0, 0]), np.array([0, 0, 0]), 0.0)
    solve_island([floor, ball], [c], 1 / 60)
    assert np.all(floor.linear_velocity == 0.0)


def test_baumgarte_pushes_out_penetration() -> None:
    ball = _dynamic_sphere((0, 0.4, 0))
    floor = _static()
    c = ContactConstraint(0, 1, np.array([0, 1.0, 0]), np.array([0, 0, 0]), 0.105,
                          friction=0.0)
    solve_island([floor, ball], [c], 1 / 60)
    expected = 0.2 / (1 / 60) * (0.105 - 0.005)
    assert ball.linear_velocity[1] == pytest.approx(min(expected, 4.0), rel=1e-6)


def test_warm_start_applies_stored_impulse() -> None:
    ball = _dynamic_sphere((0, 0.5, 0))
    floor = _static()
    c = ContactConstraint(0, 1, np.array([0, 1.0, 0]), np.array([0, 0, 0]), 0.0,
                          friction=0.0, normal_impulse=0.1)
    solve_island([floor, ball], [c], 1 / 60, iterations=0)
    assert ball.linear_velocity[1] == pytest.approx(0.1)


def test_stack_of_three_supports_weight() -> None:
    dt = 1 / 60
    boxes = []
    for i in range(3):
        b = RigidBody(mass=1.0, position=(0, 0.5 + i, 0))
        b.set_inertia(BoxShape((0.5, 0.5, 0.5)).inertia_tensor(1.0))
        boxes.append(b)
    bodies = [_static(), *boxes]
    contacts = []
    for i in range(3):
        for x in (-0.5, 0.5):
            for z in (-0.5, 0.5):
                contacts.append(
                    ContactConstraint(i, i + 1, np.array([0, 1.0, 0]),
                                      np.array([x, i, z]), 0.0, friction=0.5)
                )
    for _ in range(60):
        for b in boxes:
            b.linear_velocity = b.linear_velocity + np.array([0, -9.81 * dt, 0])
        solve_island(bodies, contacts, dt)
    for b in boxes:
        assert np.linalg.norm(b.linear_velocity) < 0.05


def test_solver_is_deterministic() -> None:
    def run():
        rng = np.random.default_rng(3)
        bodies = [_static()] + [
            _dynamic_sphere(rng.uniform(-1, 1, 3), rng.uniform(-3, 3, 3)) for _ in range(6)
        ]
        contacts = []
        for _ in range(12):
            a, b = rng.choice(7, 2, replace=False)
            n = rng.normal(size=3)
            n /= np.linalg.norm(n)
            contacts.append(ContactConstraint(int(a), int(b), n, rng.uniform(-1, 1, 3),
                                              0.02, friction=0.4))
        for _ in range(5):
            solve_island(bodies, contacts, 1 / 60)
        return np.array([b.linear_velocity for b in bodies])

    assert np.array_equal(run(), run())


def test_inclined_plane_acceleration() -> None:
    # статичный склон: ящик скользит с a = g (sin - mu cos); проверяем одним шагом
    theta = math.radians(30.0)
    mu = 0.2
    dt = 1 / 60
    normal = np.array([-math.sin(theta), math.cos(theta), 0.0])  # нормаль склона
    down = np.array([-math.cos(theta), -math.sin(theta), 0.0])
    box = RigidBody(mass=1.0, position=(0, 0.5, 0))
    box.set_inertia(BoxShape((0.5, 0.5, 0.5)).inertia_tensor(1.0))
    floor = _static()
    cs = [
        ContactConstraint(0, 1, normal, np.array([0, 0.5, z]) - 0.5 * normal, 0.0,
                          friction=mu)
        for z in (-0.5, 0.5)
    ]
    v = np.zeros(3)
    for _ in range(120):
        box.linear_velocity = box.linear_velocity + np.array([0, -9.81 * dt, 0])
        solve_island([floor, box], cs, dt, iterations=16)
        v = box.linear_velocity.copy()
    expected = 9.81 * (math.sin(theta) - mu * math.cos(theta))
    along = float(v @ down) / (120 * dt)
    assert along == pytest.approx(expected, rel=0.1)
