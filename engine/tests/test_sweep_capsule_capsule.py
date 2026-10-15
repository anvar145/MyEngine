"""Капсула против капсулы: предикат, быстрый свип и контакт на чистых float."""

from __future__ import annotations

import numpy as np

from engine.physics.collider import Collider
from engine.physics.mathutil import quat_from_axis_angle
from engine.physics.narrowphase import (
    _capsule_capsule_overlap,
    capsule_capsule,
    closest_segment_segment,
)
from engine.physics.shape import CapsuleShape
from engine.physics.world import PhysicsWorld
from engine.tests.test_sweep_capsule_box import _reference


def _random_pair(rng):
    sa = CapsuleShape(rng.uniform(0.2, 0.6), rng.uniform(0.3, 1.0))
    sb = CapsuleShape(rng.uniform(0.2, 0.6), rng.uniform(0.3, 1.0))
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    rb = quat_from_axis_angle(axis, rng.uniform(0, 3.14))
    return sa, sb, rng.uniform(-1.5, 1.5, 3), rng.uniform(-1.5, 1.5, 3), rb


def _old_capsule_capsule(sa, pa, ra, sb, pb, rb):
    """Прежняя реализация на ``np.cross`` (эталон для сравнения)."""
    from engine.physics.narrowphase import _closest_on_segment, _spheres

    p1, q1 = sa.segment(pa, ra)
    p2, q2 = sb.segment(pb, rb)
    c1, c2 = closest_segment_segment(p1, q1, p2, q2)
    d1, d2 = q1 - p1, q2 - p2
    fallback = np.cross(d1, d2)
    if np.linalg.norm(fallback) < 1e-9:
        fallback = np.cross(d1, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(fallback) < 1e-9:
            fallback = np.cross(d1, np.array([0.0, 0.0, 1.0]))
    fallback = fallback / np.linalg.norm(fallback)
    main = _spheres(c1, sa.radius, c2, sb.radius, fallback)
    if main is None:
        return None
    a = float(d1 @ d1)
    if a > 1e-12 and np.linalg.norm(np.cross(d1, d2)) < 1e-6 * np.sqrt(a) * np.sqrt(
        float(d2 @ d2)
    ):
        s_vals = [min(1.0, max(0.0, float((p - p1) @ d1) / a)) for p in (p2, q2)]
        if abs(s_vals[0] - s_vals[1]) * np.sqrt(a) > 1e-4:
            pts, deps = [], []
            for s in s_vals:
                ca = p1 + d1 * s
                cb = _closest_on_segment(ca, p2, q2)
                m = _spheres(ca, sa.radius, cb, sb.radius, main.normal)
                if m is not None:
                    pts.append(m.points[0])
                    deps.append(m.depths[0])
            if len(pts) == 2:
                return main.__class__(main.normal, np.array(pts), np.array(deps))
    return main


def test_predicate_and_contact_match_reference() -> None:
    rng = np.random.default_rng(11)
    touching = 0
    for _ in range(4000):
        sa, sb, pa, pb, rb = _random_pair(rng)
        eye = np.eye(3)
        old = _old_capsule_capsule(sa, pa, eye, sb, pb, rb)
        new = capsule_capsule(sa, pa, eye, sb, pb, rb)
        assert (old is None) == (new is None)
        assert _capsule_capsule_overlap(sa, pa, eye, sb, pb, rb) == (old is not None)
        if old is None:
            continue
        touching += 1
        assert np.allclose(old.normal, new.normal, atol=1e-9)
        assert np.allclose(old.points, new.points, atol=1e-9)
        assert np.allclose(old.depths, new.depths, atol=1e-9)
    assert touching > 500


def test_parallel_capsules_give_two_points() -> None:
    sa = sb = CapsuleShape(0.4, 0.5)
    m = capsule_capsule(sa, (0.0, 0.0, 0.0), np.eye(3), sb, (0.6, 0.3, 0.0), np.eye(3))
    assert m is not None and len(m.depths) == 2


def test_fast_sweep_is_bit_identical_to_reference() -> None:
    rng = np.random.default_rng(5)
    eye = np.eye(3)
    hits = 0
    for _ in range(2000):
        world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
        col_shape = CapsuleShape(rng.uniform(0.2, 0.6), rng.uniform(0.3, 1.0))
        col = Collider(col_shape)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        world.add_static(
            col,
            tuple(rng.uniform(-1, 1, 3)),
            quat_from_axis_angle(axis, rng.uniform(0, 3.14)),
        )
        shape = CapsuleShape(0.4, 0.5)
        pos = rng.uniform(-2.5, 2.5, 3)
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
    assert hits > 150
