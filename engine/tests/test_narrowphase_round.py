"""Пакетные пары с шаром и капсулой против скалярных эталонов из ``narrowphase``."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from engine.physics.mathutil import quat_normalize, quat_to_matrix
from engine.physics.narrowphase import collide
from engine.physics.narrowphase_round import (
    capsule_box_batch,
    capsule_capsule_batch,
    collide_batch,
    sphere_box_batch,
    sphere_capsule_batch,
    sphere_sphere_batch,
)
from engine.physics.shape import BoxShape, CapsuleShape, SphereShape
from engine.physics.store import KIND_BOX, KIND_CAPSULE, KIND_SPHERE

TOL = 1e-9
KINDS = (KIND_SPHERE, KIND_BOX, KIND_CAPSULE)
PAIRS = list(itertools.product(KINDS, KINDS))
PAIRS.remove((KIND_BOX, KIND_BOX))  # куб - куб: box_box_batch


def _shape(kind: int, half: np.ndarray):
    if kind == KIND_BOX:
        return BoxShape(half)
    if kind == KIND_SPHERE:
        return SphereShape(float(half[0]))
    return CapsuleShape(float(half[0]), float(half[1]))


def _random_side(rng: np.random.Generator, kind: int, n: int, special: bool):
    """Позы и размеры ``n`` форм вида ``kind`` в формате ``ColliderStore``."""
    quat = rng.normal(size=(n, 4))
    if special:  # часть поз — оси мира, повороты на 90° и малые наклоны
        steps = rng.integers(0, 4, n)
        tilt = rng.choice([0.0, 0.0, 1e-3, 0.2], n)
        axis = rng.integers(0, 3, n)
        for i in range(n):
            a = steps[i] * np.pi / 2 + tilt[i]
            q = np.zeros(4)
            q[axis[i]] = np.sin(a / 2)
            q[3] = np.cos(a / 2)
            quat[i] = q
    rot = np.array([quat_to_matrix(quat_normalize(q)) for q in quat])
    if kind == KIND_BOX:
        half = rng.uniform(0.2, 1.0, (n, 3))
    elif kind == KIND_SPHERE:
        half = np.repeat(rng.uniform(0.2, 0.9, (n, 1)), 3, axis=1)
    else:
        half = np.stack(
            [rng.uniform(0.15, 0.6, n), rng.uniform(0.1, 0.9, n), np.zeros(n)], axis=1
        )
    return half, rot


def _pairs(kind_a: int, kind_b: int, n: int, seed: int, spread: float = 1.6,
           special: bool = False):
    rng = np.random.default_rng(seed)
    half_a, rot_a = _random_side(rng, kind_a, n, special)
    half_b, rot_b = _random_side(rng, kind_b, n, special)
    pos_a = rng.uniform(-0.3, 0.3, (n, 3))
    pos_b = pos_a + rng.uniform(-spread, spread, (n, 3))
    if special:  # часть пар — соосные и с совпадающими центрами
        same_axis = rng.random(n) < 0.3
        pos_b[same_axis, 0] = pos_a[same_axis, 0]
        pos_b[same_axis, 2] = pos_a[same_axis, 2]
        coincident = rng.random(n) < 0.05
        pos_b[coincident] = pos_a[coincident]
    return pos_a, rot_a, half_a, pos_b, rot_b, half_b


def _batch(kind_a: int, kind_b: int, args):
    pos_a, rot_a, half_a, pos_b, rot_b, half_b = args
    return collide_batch(kind_a, pos_a, rot_a, half_a, kind_b, pos_b, rot_b, half_b)


def _scalar(kind_a: int, kind_b: int, args, i: int):
    pos_a, rot_a, half_a, pos_b, rot_b, half_b = args
    return collide(
        _shape(kind_a, half_a[i]), pos_a[i], rot_a[i],
        _shape(kind_b, half_b[i]), pos_b[i], rot_b[i],
    )


def _assert_matches(kind_a: int, kind_b: int, args, result) -> int:
    hit, normal, count, points, depths = result
    checked = 0
    for i in range(len(args[0])):
        man = _scalar(kind_a, kind_b, args, i)
        assert bool(hit[i]) == (man is not None), f"пара {i}: hit"
        if man is None:
            continue
        checked += 1
        assert int(count[i]) == len(man.points), f"пара {i}: число точек"
        np.testing.assert_allclose(normal[i], man.normal, atol=TOL, err_msg=f"{i}")
        np.testing.assert_allclose(points[i, : count[i]], man.points, atol=TOL)
        np.testing.assert_allclose(depths[i, : count[i]], man.depths, atol=TOL)
    return checked


@pytest.mark.parametrize(("kind_a", "kind_b"), PAIRS)
def test_matches_scalar_on_random_pairs(kind_a: int, kind_b: int) -> None:
    checked = 0
    for seed, spread in [(1, 0.8), (2, 1.3), (3, 2.0)]:
        args = _pairs(kind_a, kind_b, 300, seed, spread)
        checked += _assert_matches(kind_a, kind_b, args, _batch(kind_a, kind_b, args))
    assert checked > 80


@pytest.mark.parametrize(("kind_a", "kind_b"), PAIRS)
def test_matches_scalar_on_special_poses(kind_a: int, kind_b: int) -> None:
    """Оси мира, повороты на 90°, малые наклоны, соосные и совпадающие центры."""
    checked = 0
    for seed, spread in [(10, 0.6), (11, 1.2)]:
        args = _pairs(kind_a, kind_b, 400, seed, spread, special=True)
        checked += _assert_matches(kind_a, kind_b, args, _batch(kind_a, kind_b, args))
    assert checked > 80


def test_parallel_capsules_give_two_points() -> None:
    n = 3
    rot = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    pos_a = np.zeros((n, 3))
    pos_b = np.array([[0.4, 0.3, 0.0], [0.5, 0.0, 0.0], [0.3, 0.8, 0.0]])
    radius = np.full(n, 0.3)
    half_h = np.full(n, 0.5)
    hit, normal, count, points, _ = capsule_capsule_batch(
        pos_a, rot, radius, half_h, pos_b, rot, radius, half_h
    )
    assert hit.tolist() == [True, True, True]
    assert count.tolist() == [2, 2, 2]
    np.testing.assert_allclose(normal[:2], [[1, 0, 0]] * 2, atol=1e-12)
    np.testing.assert_allclose(points[0, :, 1], [-0.2, 0.5], atol=1e-12)


def test_capsule_through_box_is_pushed_out() -> None:
    """Ось капсулы пересекает бокс: вынос по ближайшей грани (как скалярный)."""
    args = (
        np.array([[0.0, 0.1, 0.0]]), np.eye(3)[None], np.array([[0.3, 0.5, 0.0]]),
        np.array([[0.0, 0.0, 0.0]]), np.eye(3)[None], np.array([[2.0, 0.5, 2.0]]),
    )
    result = _batch(KIND_CAPSULE, KIND_BOX, args)
    assert result[0][0] and result[2][0] == 1
    assert _assert_matches(KIND_CAPSULE, KIND_BOX, args, result) == 1


def test_sphere_center_inside_box() -> None:
    args = (
        np.array([[0.1, 0.2, 0.0], [0.0, 0.0, 0.0]]), np.broadcast_to(np.eye(3), (2, 3, 3)),
        np.array([[0.4] * 3, [0.4] * 3]),
        np.zeros((2, 3)), np.broadcast_to(np.eye(3), (2, 3, 3)),
        np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]]),
    )
    result = _batch(KIND_SPHERE, KIND_BOX, args)
    assert result[0].all()
    assert _assert_matches(KIND_SPHERE, KIND_BOX, args, result) == 2


def test_coincident_spheres_use_up_normal() -> None:
    pos = np.zeros((1, 3))
    hit, normal, count, _, depths = sphere_sphere_batch(
        pos, np.array([0.5]), pos, np.array([0.25])
    )
    assert hit[0] and count[0] == 1
    np.testing.assert_allclose(normal[0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(depths[0, 0], 0.75)


def test_unsupported_kinds_return_none() -> None:
    args = _pairs(KIND_BOX, KIND_BOX, 3, 0)
    assert _batch(KIND_BOX, KIND_BOX, args) is None


def _compare_pairs(result_a, i: int, result_b, j: int) -> None:
    """Побитово равны результаты пары ``i`` в одном пакете и ``j`` в другом."""
    assert bool(result_a[0][i]) == bool(result_b[0][j])
    if not result_a[0][i]:
        return
    count = int(result_a[2][i])
    assert count == int(result_b[2][j])
    np.testing.assert_array_equal(result_a[1][i], result_b[1][j])
    np.testing.assert_array_equal(result_a[3][i, :count], result_b[3][j, :count])
    np.testing.assert_array_equal(result_a[4][i, :count], result_b[4][j, :count])


def _take(args, index):
    return tuple(a[index] for a in args)


@pytest.mark.parametrize(("kind_a", "kind_b"), PAIRS)
def test_result_does_not_depend_on_batch(kind_a: int, kind_b: int) -> None:
    """Пара даёт побитово тот же результат в любом пакете: целом, перемешанном, одна."""
    args = _pairs(kind_a, kind_b, 200, 21, 1.4, special=True)
    full = _batch(kind_a, kind_b, args)
    perm = np.random.default_rng(5).permutation(200)
    shuffled = _batch(kind_a, kind_b, _take(args, perm))
    half = _batch(kind_a, kind_b, _take(args, np.arange(77, 150)))
    for new, old in enumerate(perm):
        _compare_pairs(shuffled, new, full, int(old))
    for new, old in enumerate(range(77, 150)):
        _compare_pairs(half, new, full, old)
    for i in range(0, 200, 7):
        single = _batch(kind_a, kind_b, _take(args, np.array([i])))
        _compare_pairs(single, 0, full, i)


@pytest.mark.parametrize(("kind_a", "kind_b"), PAIRS)
def test_repeat_is_bitwise_identical(kind_a: int, kind_b: int) -> None:
    args = _pairs(kind_a, kind_b, 150, 33, 1.2, special=True)
    one, two = _batch(kind_a, kind_b, args), _batch(kind_a, kind_b, args)
    for x, y in zip(one, two, strict=True):
        np.testing.assert_array_equal(x, y)


def test_named_functions_match_dispatcher() -> None:
    """Именованные функции и ``collide_batch`` дают одно и то же."""
    args = _pairs(KIND_CAPSULE, KIND_CAPSULE, 50, 4)
    pos_a, rot_a, half_a, pos_b, rot_b, half_b = args
    direct = capsule_capsule_batch(
        pos_a, rot_a, half_a[:, 0], half_a[:, 1], pos_b, rot_b, half_b[:, 0],
        half_b[:, 1],
    )
    for x, y in zip(direct, _batch(KIND_CAPSULE, KIND_CAPSULE, args), strict=True):
        np.testing.assert_array_equal(x, y)
    args = _pairs(KIND_SPHERE, KIND_CAPSULE, 50, 5)
    pos_a, rot_a, half_a, pos_b, rot_b, half_b = args
    direct = sphere_capsule_batch(
        pos_a, half_a[:, 0], pos_b, rot_b, half_b[:, 0], half_b[:, 1]
    )
    for x, y in zip(direct, _batch(KIND_SPHERE, KIND_CAPSULE, args), strict=True):
        np.testing.assert_array_equal(x, y)
    assert callable(sphere_box_batch) and callable(capsule_box_batch)
