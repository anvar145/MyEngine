"""BVH статического меша: построение, лучи, выборки (A11)."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from engine.physics.bvh import Bvh, closest_point_triangle, hit

CUBE_VERTS = np.array(
    [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
)
CUBE_TRIS = np.array([
    [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],  # -x, +x
    [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],  # -y, +y
    [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],  # -z, +z
])


def _grid(n: int, seed: int = 1):
    """Холмистая сетка n x n клеток (2 n^2 треугольников)."""
    rng = np.random.default_rng(seed)
    xs, zs = np.meshgrid(np.arange(n + 1.0), np.arange(n + 1.0), indexing="ij")
    ys = rng.uniform(0.0, 0.5, xs.shape)
    verts = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
    tris = []
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            b = a + 1
            c = a + n + 1
            d = c + 1
            tris += [[a, c, b], [b, c, d]]
    return verts, np.array(tris)


def _brute_ray(verts, tris, origin, direction, max_dist=math.inf):
    """Эталон: перебор всех треугольников (Мёллер - Трамбор в numpy)."""
    o = np.asarray(origin, float)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    best = (max_dist, -1)
    for index, tri in enumerate(tris):
        a, b, c = verts[tri]
        e1, e2 = b - a, c - a
        p = np.cross(d, e2)
        det = e1 @ p
        if abs(det) < 1e-12:
            continue
        s = o - a
        u = (s @ p) / det
        q = np.cross(s, e1)
        v = (d @ q) / det
        t = (e2 @ q) / det
        if u >= 0 and v >= 0 and u + v <= 1 and 0 <= t < best[0]:
            best = (t, index)
    return best


def test_cube_structure_two_levels() -> None:
    bvh = Bvh(CUBE_VERTS, CUBE_TRIS, leaf_size=4)
    assert bvh.triangle_count == 12
    assert bvh.depth == 3  # корень -> 6+6 -> 3+3
    assert bvh.node_total == 7
    leaves = bvh.node_count[bvh.node_count > 0]
    assert leaves.sum() == 12 and leaves.max() <= 4
    lo, hi = bvh.bounds
    assert np.allclose(lo, -1.0) and np.allclose(hi, 1.0)
    # каждый треугольник попал ровно в один лист
    assert sorted(bvh.order.tolist()) == list(range(12))


def test_children_bounds_inside_parent() -> None:
    verts, tris = _grid(12)
    bvh = Bvh(verts, tris)
    for node in range(bvh.node_total):
        if bvh.node_count[node] == 0:
            left = bvh.node_first[node]
            for kid in (left, left + 1):
                assert (bvh.node_lo[kid] >= bvh.node_lo[node] - 1e-12).all()
                assert (bvh.node_hi[kid] <= bvh.node_hi[node] + 1e-12).all()


def test_ray_hits_cube_face_with_normal() -> None:
    bvh = Bvh(CUBE_VERTS, CUBE_TRIS)
    result = bvh.raycast((0.2, 0.3, 5.0), (0.0, 0.0, -1.0))
    assert result is not None
    assert result.distance == pytest.approx(4.0)
    assert result.normal == pytest.approx((0.0, 0.0, 1.0))
    again = hit(bvh, (-5.0, 0.1, 0.1), (2.0, 0.0, 0.0))  # не нормированный
    assert again.distance == pytest.approx(4.0)
    assert again.normal == pytest.approx((-1.0, 0.0, 0.0))


def test_ray_miss_and_max_distance() -> None:
    bvh = Bvh(CUBE_VERTS, CUBE_TRIS)
    assert bvh.raycast((3.0, 3.0, 5.0), (0.0, 0.0, -1.0)) is None
    assert bvh.raycast((0.0, 0.0, 5.0), (0.0, 0.0, 1.0)) is None  # смотрит от куба
    assert bvh.raycast((0.0, 0.0, 5.0), (0.0, 0.0, -1.0), 3.9) is None
    assert bvh.raycast((0.0, 0.0, 5.0), (0.0, 0.0, 0.0)) is None


def test_ray_from_inside_hits_wall_with_inward_normal() -> None:
    bvh = Bvh(CUBE_VERTS, CUBE_TRIS)
    result = bvh.raycast((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert result.distance == pytest.approx(1.0)
    assert result.normal == pytest.approx((0.0, -1.0, 0.0))


def test_ray_matches_brute_force_on_random_rays() -> None:
    verts, tris = _grid(8)
    bvh = Bvh(verts, tris)
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(80):
        origin = rng.uniform((-1, 1, -1), (9, 3, 9))
        direction = rng.normal(size=3)
        direction[1] = -abs(direction[1])
        expected = _brute_ray(verts, tris, origin, direction)
        got = bvh.raycast(origin, direction)
        if expected[1] < 0:
            assert got is None
        else:
            assert got is not None
            assert got.distance == pytest.approx(expected[0], abs=1e-9)
            checked += 1
    assert checked > 20


def test_aabb_overlap_cube() -> None:
    bvh = Bvh(CUBE_VERTS, CUBE_TRIS)
    assert sorted(bvh.query_aabb((-0.1, -0.1, 0.9), (0.1, 0.1, 1.1))) == [10, 11]
    assert bvh.query_aabb((2.0, 2.0, 2.0), (3.0, 3.0, 3.0)) == []
    assert len(bvh.query_aabb((-5, -5, -5), (5, 5, 5))) == 12


def test_aabb_overlap_matches_brute_force() -> None:
    verts, tris = _grid(10)
    bvh = Bvh(verts, tris)
    corners = verts[tris]
    tlo, thi = corners.min(axis=1), corners.max(axis=1)
    rng = np.random.default_rng(3)
    for _ in range(40):
        lo = rng.uniform(-1, 10, 3)
        hi = lo + rng.uniform(0.1, 3.0, 3)
        mask = (tlo <= hi).all(axis=1) & (thi >= lo).all(axis=1)
        assert sorted(bvh.query_aabb(lo, hi)) == np.flatnonzero(mask).tolist()


def test_sphere_query_exact() -> None:
    bvh = Bvh(CUBE_VERTS, CUBE_TRIS)
    near = bvh.query_sphere((0.0, 0.0, 1.5), 0.6)
    assert sorted(near) == [10, 11]  # только грань +z
    assert bvh.query_sphere((0.0, 0.0, 3.0), 0.5) == []
    assert len(bvh.query_sphere((0.0, 0.0, 0.0), 0.2)) == 0  # внутри, граней не касается


def test_closest_point_regions() -> None:
    a, b, c = (0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)
    assert closest_point_triangle((0.5, 0.5, 1.0), a, b, c) == pytest.approx(
        (0.5, 0.5, 0.0)
    )
    assert closest_point_triangle((-1.0, -1.0, 0.0), a, b, c) == pytest.approx(a)
    assert closest_point_triangle((3.0, 0.0, 0.0), a, b, c) == pytest.approx(b)
    assert closest_point_triangle((1.0, -1.0, 0.0), a, b, c) == pytest.approx(
        (1.0, 0.0, 0.0)
    )
    assert closest_point_triangle((2.0, 2.0, 0.0), a, b, c) == pytest.approx(
        (1.0, 1.0, 0.0)
    )


def test_invalid_input_and_empty_mesh() -> None:
    with pytest.raises(ValueError):
        Bvh(CUBE_VERTS, [[0, 1, 99]])
    with pytest.raises(ValueError):
        Bvh([[0.0, 0.0]], [[0, 0, 0]])
    empty = Bvh(np.zeros((0, 3)), np.zeros((0, 3), dtype=int))
    assert empty.raycast((0, 0, 0), (1, 0, 0)) is None
    assert empty.query_aabb((0, 0, 0), (1, 1, 1)) == []


def test_perf_10k_triangles_1000_rays() -> None:
    verts, tris = _grid(71)  # 2 * 71^2 = 10082 треугольника
    started = time.perf_counter()
    bvh = Bvh(verts, tris)
    build = time.perf_counter() - started
    assert bvh.triangle_count > 10000
    rng = np.random.default_rng(5)
    origins = rng.uniform((1, 3, 1), (60, 5, 60), (1000, 3))
    dirs = np.tile((0.1, -1.0, 0.05), (1000, 1))
    started = time.perf_counter()
    hits = sum(bvh.raycast(o, d) is not None for o, d in zip(origins, dirs, strict=True))
    elapsed = time.perf_counter() - started
    assert hits == 1000
    # запас в разы относительно измеренного (≈0.1 с): тест ловит порядок, не микросекунды
    assert build < 3.0
    assert elapsed < 1.5
