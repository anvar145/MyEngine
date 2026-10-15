"""Пакетный ``box_box_batch`` против скалярного эталона ``box_box``."""

from __future__ import annotations

import numpy as np
import pytest

from engine.physics import narrowphase_batch
from engine.physics.mathutil import quat_normalize, quat_to_matrix
from engine.physics.narrowphase import _reduce_points, box_box
from engine.physics.narrowphase_batch import _reduce_points_batch, box_box_batch
from engine.physics.shape import BoxShape

TOL = 1e-4


def _random_pairs(rng: np.random.Generator, n: int, spread: float = 1.6):
    """Пары боксов со случайными позами; часть — соосные и с поворотом вокруг Y."""
    pos_a = rng.uniform(-0.3, 0.3, (n, 3))
    pos_b = pos_a + rng.uniform(-spread, spread, (n, 3))
    half_a = rng.uniform(0.2, 1.0, (n, 3))
    half_b = rng.uniform(0.2, 1.0, (n, 3))
    quats = rng.normal(size=(n, 2, 4))
    kind = rng.integers(0, 4, n)
    for i in range(n):
        if kind[i] == 0:  # соосные
            quats[i] = [[0, 0, 0, 1]] * 2
        elif kind[i] == 1:  # поворот вокруг Y (стопка с поворотом)
            ang = rng.uniform(-1.0, 1.0, 2)
            quats[i] = [[0, np.sin(a / 2), 0, np.cos(a / 2)] for a in ang]
    rots = np.empty((n, 2, 3, 3))
    for i in range(n):
        for k in range(2):
            rots[i, k] = quat_to_matrix(quat_normalize(quats[i, k]))
    return pos_a, rots[:, 0], half_a, pos_b, rots[:, 1], half_b


def _scalar(pos_a, rot_a, half_a, pos_b, rot_b, half_b, i):
    return box_box(BoxShape(half_a[i]), pos_a[i], rot_a[i],
                   BoxShape(half_b[i]), pos_b[i], rot_b[i])


def _assert_matches(args, result, tol: float = TOL) -> int:
    hit, normal, count, points, depths = result
    checked = 0
    for i in range(len(args[0])):
        man = _scalar(*args, i)
        assert bool(hit[i]) == (man is not None), f"пара {i}: hit"
        if man is None:
            continue
        checked += 1
        assert int(count[i]) == len(man.points), f"пара {i}: число точек"
        np.testing.assert_allclose(normal[i], man.normal, atol=tol)
        np.testing.assert_allclose(points[i, : count[i]], man.points, atol=tol)
        np.testing.assert_allclose(depths[i, : count[i]], man.depths, atol=tol)
    return checked


def test_matches_scalar_on_100_random_pairs() -> None:
    args = _random_pairs(np.random.default_rng(1), 100)
    assert _assert_matches(args, box_box_batch(*args)) > 30


def test_matches_scalar_on_many_random_pairs_across_geometries() -> None:
    for seed, spread in [(2, 0.8), (3, 1.2), (4, 2.0)]:
        args = _random_pairs(np.random.default_rng(seed), 400, spread)
        assert _assert_matches(args, box_box_batch(*args)) > 50


def test_deep_and_flat_penetrations() -> None:
    rng = np.random.default_rng(5)
    args = list(_random_pairs(rng, 200, 0.4))
    args[2] = args[2] * np.array([1.0, 0.1, 1.0])  # плоские «плиты»
    assert _assert_matches(tuple(args), box_box_batch(*args)) > 100


def test_aligned_stack_and_floor() -> None:
    eye = np.eye(3)
    pa = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    pb = np.array([[0.0, 0.99, 0.0], [0.3, 0.98, -0.2], [0.0, 1.0, 0.0]])
    half = np.full((3, 3), 0.5)
    rot = np.tile(eye, (3, 1, 1))
    args = (pa, rot, half, pb, rot, half)
    result = box_box_batch(*args)
    assert result[0].tolist() == [True, True, True]  # третья касается (overlap = 0)
    _assert_matches(args, result)
    assert result[2].tolist() == [4, 4, 4]


def test_no_hit_and_empty_batch() -> None:
    args = _random_pairs(np.random.default_rng(6), 50, 6.0)
    result = box_box_batch(*args)
    assert not result[0].all()
    _assert_matches(args, result)
    empty = tuple(a[:0] for a in args)
    hit, normal, count, points, depths = box_box_batch(*empty)
    assert hit.shape == (0,) and normal.shape == (0, 3)
    assert count.shape == (0,) and points.shape == (0, 4, 3)
    assert depths.shape == (0, 4)


def test_pair_result_independent_of_batch_composition() -> None:
    """Результат пары побитово один и тот же при любом составе и порядке пакета."""
    args = _random_pairs(np.random.default_rng(7), 120)
    full = box_box_batch(*args)
    perm = np.random.default_rng(8).permutation(120)
    shuffled = box_box_batch(*(a[perm] for a in args))
    inv = np.argsort(perm)
    for whole, part in zip(full, shuffled, strict=True):
        assert np.array_equal(whole, part[inv])
    sub = slice(10, 17)
    small = box_box_batch(*(a[sub] for a in args))
    for whole, part in zip(full, small, strict=True):
        assert np.array_equal(whole[sub], part)


def _random_polygons(rng: np.random.Generator, w: int, cap: int = 9):
    """Выпуклые полигоны на 5..8 точек в случайных плоскостях, с паддингом нулями."""
    pts = np.zeros((w, cap, 3))
    deps = np.zeros((w, cap))
    kept = rng.integers(5, 9, w)
    for i in range(w):
        k = int(kept[i])
        ang = np.sort(rng.uniform(0.0, 2.0 * np.pi, k))
        flat = np.stack([np.cos(ang) * rng.uniform(0.3, 1.0),
                         np.sin(ang) * rng.uniform(0.3, 1.0),
                         np.zeros(k)], axis=1)
        frame = quat_to_matrix(quat_normalize(rng.normal(size=4)))
        pts[i, :k] = flat @ frame.T + rng.uniform(-1.0, 1.0, 3)
        deps[i, :k] = rng.uniform(0.0, 0.1, k)
        if i % 5 == 0:  # одинаковые глубины и повтор точки: тай-брейки по индексу
            deps[i, :k] = 0.05
            pts[i, k - 1] = pts[i, 0]
    return pts, deps, kept


def test_reduce_points_batch_matches_scalar() -> None:
    pts, deps, kept = _random_polygons(np.random.default_rng(21), 300)
    out_pts, out_dep, count = _reduce_points_batch(pts, deps, kept)
    for i in range(len(kept)):
        want_p, want_d = _reduce_points(
            [tuple(p) for p in pts[i, : kept[i]].tolist()], deps[i, : kept[i]].tolist()
        )
        assert count[i] == len(want_p), f"полигон {i}: число точек"
        assert np.array_equal(out_pts[i, : count[i]], np.array(want_p))
        assert np.array_equal(out_dep[i, : count[i]], np.array(want_d))
        assert not out_pts[i, count[i]:].any()  # хвост нулевой


def test_reduce_points_batch_result_independent_of_batch_composition() -> None:
    pts, deps, kept = _random_polygons(np.random.default_rng(22), 80)
    whole = _reduce_points_batch(pts, deps, kept)
    sub = slice(13, 29)
    part = _reduce_points_batch(pts[sub], deps[sub], kept[sub])
    for a, b in zip(whole, part, strict=True):
        assert np.array_equal(a[sub], b)


def test_pairs_with_clipped_polygon_over_four_points_match_scalar(monkeypatch) -> None:
    """Повёрнутые вокруг Y стопки: клиппинг даёт 5..8 точек, редукция как у эталона."""
    calls = []
    real = _reduce_points_batch

    def spy(pts, deps, kept):
        calls.append(len(kept))
        return real(pts, deps, kept)

    monkeypatch.setattr(narrowphase_batch, "_reduce_points_batch", spy)
    rng = np.random.default_rng(23)
    n = 300
    ang = rng.uniform(0.2, 1.3, (n, 2))
    rots = np.empty((n, 2, 3, 3))
    for i in range(n):
        for k in range(2):
            quat = [0.0, np.sin(ang[i, k] / 2), 0.0, np.cos(ang[i, k] / 2)]
            rots[i, k] = quat_to_matrix(quat_normalize(quat))
    pos_a = np.zeros((n, 3))
    pos_b = np.stack([rng.uniform(-0.2, 0.2, n), rng.uniform(0.85, 0.99, n),
                      rng.uniform(-0.2, 0.2, n)], axis=1)
    half = np.full((n, 3), 0.5)
    args = (pos_a, rots[:, 0], half, pos_b, rots[:, 1], half)
    assert _assert_matches(args, box_box_batch(*args)) > 100
    assert sum(calls) > 20  # ветка редукции действительно проверена


@pytest.mark.parametrize("seed", [11, 12])
def test_repeatable_run_to_run(seed: int) -> None:
    args = _random_pairs(np.random.default_rng(seed), 150)
    first, second = box_box_batch(*args), box_box_batch(*args)
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a, b)
