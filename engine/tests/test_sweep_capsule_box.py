"""Свип капсулы по боксу: быстрый путь (предикат) совпадает с прежним (``collide``)."""

from __future__ import annotations

import numpy as np

from engine.physics.collider import Collider
from engine.physics.mathutil import quat_from_axis_angle
from engine.physics.narrowphase import collide
from engine.physics.shape import BoxShape, CapsuleShape
from engine.physics.world import _BISECT_ITERATIONS, PhysicsWorld


def _reference(shape, pos, rot, d, limit, step, col):
    """Прежняя реализация ``_sweep_one`` (каждый тест — полный ``collide``)."""

    def test(t):
        return collide(shape, pos + d * t, rot, col.shape, col.world_position,
                       col.world_rotation)

    m = test(0.0)
    if m is not None:
        return 0.0, -m.normal, m.point
    t_prev = 0.0
    t = 0.0
    while t < limit:
        t = min(t + step, limit)
        m = test(t)
        if m is not None:
            lo, hi = t_prev, t
            for _ in range(_BISECT_ITERATIONS):
                mid = 0.5 * (lo + hi)
                mid_m = test(mid)
                if mid_m is None:
                    lo = mid
                else:
                    hi, m = mid, mid_m
            return hi, -m.normal, m.point
        t_prev = t
    return None


def test_fast_sweep_is_bit_identical_to_reference() -> None:
    rng = np.random.default_rng(7)
    eye = np.eye(3)
    hits = 0
    for _ in range(3000):
        world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
        half = rng.uniform(0.2, 2.0, 3)
        col = Collider(BoxShape(tuple(half)))
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        rot = quat_from_axis_angle(axis, rng.uniform(0, 3.14))
        world.add_static(col, tuple(rng.uniform(-1, 1, 3)), rot)
        shape = CapsuleShape(rng.uniform(0.2, 0.6), rng.uniform(0.3, 1.0))
        pos = rng.uniform(-4, 4, 3)
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        limit = float(rng.uniform(0.1, 6.0))
        lo, hi = shape.aabb(pos, eye)
        step = max(float(np.min(hi - lo)) * 0.25, 1e-3)
        new = world._sweep_one(shape, pos, eye, d, limit, step, col, lo, hi)
        old = _reference(shape, pos, eye, d, limit, step, col)
        if old is None:
            assert new is None
            continue
        hits += 1
        assert new is not None
        assert new[0] == old[0]
        assert np.array_equal(new[1], old[1])
        assert np.array_equal(new[2], old[2])
    assert hits > 300
