"""Мир с пакетными парами шар/капсула против скалярной узкой фазы."""

from __future__ import annotations

import numpy as np

from engine.physics.body import RigidBody
from engine.physics.collider import Collider
from engine.physics.mathutil import quat_normalize
from engine.physics.shape import BoxShape, CapsuleShape, SphereShape
from engine.physics.world import PhysicsWorld

STEPS = 120


def _build(batch_round: bool, seed: int = 7) -> tuple[PhysicsWorld, list[int]]:
    """Смешанная сцена: пол, шары, капсулы и ящики со случайными позами."""
    rng = np.random.default_rng(seed)
    world = PhysicsWorld()
    world._batch_round = batch_round
    world.add_static(Collider(BoxShape((30, 0.5, 30))), (0, -0.5, 0))
    world.add_static(Collider(CapsuleShape(0.3, 1.0)), (2.0, 0.6, 0.0))
    handles: list[int] = []
    for i in range(36):
        shape = (SphereShape(0.4), CapsuleShape(0.3, 0.4), BoxShape((0.4,) * 3))[i % 3]
        body = RigidBody(
            mass=1.0,
            position=(rng.uniform(-2, 2), 0.6 + 0.9 * (i // 6), rng.uniform(-2, 2)),
            rotation=quat_normalize(rng.normal(size=4)),
        )
        handles.append(world.add_body(body, Collider(shape)))
    return world, handles


def _run(batch_round: bool) -> np.ndarray:
    world, handles = _build(batch_round)
    for _ in range(STEPS):
        world.step()
    return np.array([
        np.concatenate([world.get_body(h).position, world.get_body(h).rotation])
        for h in handles
    ])


def test_batch_and_scalar_narrowphase_agree_after_simulation() -> None:
    np.testing.assert_allclose(_run(True), _run(False), atol=1e-8)


def test_batch_simulation_is_bitwise_repeatable() -> None:
    np.testing.assert_array_equal(_run(True), _run(True))


def test_batch_contacts_match_scalar_contacts_each_step() -> None:
    fast, _ = _build(True)
    slow, _ = _build(False)
    for _ in range(40):
        fast.step()
        slow.step()
        assert fast.contact_count == slow.contact_count
    assert fast.contact_count > 10
