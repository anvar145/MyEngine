"""Тесты широкой фазы."""

from __future__ import annotations

import numpy as np

from engine.physics.broadphase import Broadphase


def _box(x, y=0.0, z=0.0, s=1.0):
    return (np.array([x, y, z]), np.array([x + s, y + s, z + s]))


def test_disjoint_has_no_pairs() -> None:
    bp = Broadphase()
    for i in range(3):
        bp.add(i, _box(i * 5.0))
    assert bp.query_pairs() == []


def test_overlapping_three_pairs() -> None:
    bp = Broadphase()
    for i in range(3):
        bp.add(i, _box(i * 0.1))
    assert bp.query_pairs() == [(0, 1), (0, 2), (1, 2)]


def test_add_update_remove() -> None:
    bp = Broadphase()
    bp.add(1, _box(0))
    bp.add(2, _box(10))
    assert bp.query_pairs() == []
    bp.update(2, _box(0.5))
    assert bp.query_pairs() == [(1, 2)]
    bp.remove(1)
    assert bp.query_pairs() == [] and len(bp) == 1
    bp.clear()
    assert len(bp) == 0


def test_query_aabb() -> None:
    bp = Broadphase()
    bp.add(7, _box(0))
    bp.add(3, _box(10))
    assert bp.query_aabb(_box(0.5)) == [7]
    assert bp.query_aabb(_box(-100, s=1000)) == [3, 7]


def test_matches_brute_force() -> None:
    rng = np.random.default_rng(1)
    bp = Broadphase()
    boxes = []
    for i in range(1000):
        lo = rng.uniform(0, 50, 3)
        hi = lo + rng.uniform(0.2, 2.0, 3)
        boxes.append((lo, hi))
        bp.add(i, (lo, hi))
    expected = [
        (i, j)
        for i in range(1000)
        for j in range(i + 1, 1000)
        if np.all(boxes[i][0] <= boxes[j][1]) and np.all(boxes[j][0] <= boxes[i][1])
    ]
    assert bp.query_pairs() == expected
