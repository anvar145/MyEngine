"""Пакетные лучи и fast-path raycast по BVH/mesh (A11b)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.physics.bvh import Bvh
from engine.physics.collider import Collider
from engine.physics.shape import BoxShape
from engine.physics.world import PhysicsWorld


def _grid(cells: int = 20, size: float = 10.0, amplitude: float = 0.8) -> Bvh:
    xs = np.linspace(-size, size, cells + 1)
    gx, gz = np.meshgrid(xs, xs, indexing="ij")
    gy = amplitude * np.sin(gx * 0.7) * np.cos(gz * 0.5)
    verts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    index = np.arange((cells + 1) ** 2).reshape(cells + 1, cells + 1)
    a, b = index[:-1, :-1].ravel(), index[:-1, 1:].ravel()
    c, d = index[1:, :-1].ravel(), index[1:, 1:].ravel()
    tris = np.concatenate([np.stack([a, b, c], 1), np.stack([b, d, c], 1)])
    return Bvh(verts, tris)


def _brute(bvh: Bvh, origin, direction, limit: float):
    """Эталон: перебор всех треугольников (Мёллер - Трамбор)."""
    o = np.asarray(origin, float)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    best, best_tri = limit, -1
    for i in range(bvh.triangle_count):
        a, b, c = bvh.triangle(i)
        e1, e2 = b - a, c - a
        p = np.cross(d, e2)
        det = float(e1 @ p)
        if abs(det) < 1e-12:
            continue
        s = o - a
        u = float(s @ p) / det
        q = np.cross(s, e1)
        v = float(d @ q) / det
        t = float(e2 @ q) / det
        if u < 0 or v < 0 or u + v > 1 or not (0 <= t < best):
            continue
        best, best_tri = t, i
    return best, best_tri


def _rays(n: int = 300, seed: int = 3):
    rng = np.random.default_rng(seed)
    orig = np.column_stack([rng.uniform(-14, 14, n), rng.uniform(-2, 6, n),
                            rng.uniform(-14, 14, n)])
    dirs = rng.normal(size=(n, 3))
    dirs[: n // 3] = (0.0, -1.0, 0.0)
    return orig, dirs


def test_bvh_raycast_matches_brute_force() -> None:
    bvh = _grid()
    orig, dirs = _rays(200)
    for o, d in zip(orig, dirs, strict=True):
        ref_t, ref_tri = _brute(bvh, o, d, 40.0)
        hit = bvh.raycast(o, d, 40.0)
        if ref_tri < 0:
            assert hit is None
        else:
            assert hit is not None
            assert hit.distance == pytest.approx(ref_t, abs=1e-9)


def test_bvh_batch_equals_single() -> None:
    bvh = _grid()
    orig, dirs = _rays()
    hit_t, hit_tri, hit_point = bvh.raycast_batch(orig, dirs, 40.0)
    assert hit_t.shape == (len(orig),) and hit_point.shape == (len(orig), 3)
    for i, (o, d) in enumerate(zip(orig, dirs, strict=True)):
        hit = bvh.raycast(o, d, 40.0)
        if hit is None:
            assert math.isinf(hit_t[i]) and hit_tri[i] == -1
            assert np.isnan(hit_point[i]).all()
        else:
            assert hit_t[i] == pytest.approx(hit.distance, abs=1e-12)
            assert hit_tri[i] == hit.triangle
            unit = d / np.linalg.norm(d)
            assert hit_point[i] == pytest.approx(o + unit * hit.distance)


def test_bvh_batch_edge_cases() -> None:
    bvh = _grid()
    t, tri, p = bvh.raycast_batch(np.zeros((0, 3)), np.zeros((0, 3)))
    assert len(t) == len(tri) == len(p) == 0
    # нулевое направление - промах; одно направление на все лучи
    t, tri, _ = bvh.raycast_batch([(0, 5, 0), (1, 5, 0)], [(0, 0, 0)])
    assert np.isinf(t).all() and (tri == -1).all()
    t, tri, _ = bvh.raycast_batch([(0, 5, 0), (1, 5, 0)], [(0, -1, 0)])
    assert (tri >= 0).all()
    with pytest.raises(ValueError):
        bvh.raycast_batch([(0, 5, 0), (1, 5, 0)], [(0, -1, 0)] * 3)
    empty = Bvh(np.zeros((0, 3)), np.zeros((0, 3), dtype=int))
    t, tri, _ = empty.raycast_batch([(0, 5, 0)], [(0, -1, 0)])
    assert math.isinf(t[0]) and tri[0] == -1


@pytest.mark.parametrize("angle", [0.0, 0.7, 2.0])
def test_world_batch_equals_single_posed_mesh(angle: float) -> None:
    w = PhysicsWorld()
    q = (0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2))
    pos = (1.5, -0.5, 2.0) if angle else (0.0, 0.0, 0.0)
    w.add_mesh(_grid(), position=pos, rotation=q)
    orig, dirs = _rays()
    batch = w.raycast_batch(orig, dirs, 40.0)
    for i, (o, d) in enumerate(zip(orig, dirs, strict=True)):
        hit = w.raycast(o, d, 40.0)
        if hit is None:
            assert batch.handle[i] == -1 and math.isinf(batch.distance[i])
        else:
            assert batch.handle[i] == hit.handle
            assert batch.distance[i] == pytest.approx(hit.distance, abs=1e-9)
            assert batch.point[i] == pytest.approx(hit.point, abs=1e-9)
            assert batch.normal[i] == pytest.approx(hit.normal, abs=1e-9)


def test_world_batch_mixes_colliders_and_meshes() -> None:
    w = PhysicsWorld()
    w.add_mesh(_grid(amplitude=0.0))  # пол y=0
    wall = w.add_static(Collider(BoxShape((0.5, 2.0, 0.5))),
                        position=(3.0, 1.0, 0.0))
    orig = np.array([[0.0, 5.0, 0.0], [3.0, 5.0, 0.0], [3.0, 1.0, 5.0]])
    dirs = np.array([[0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    batch = w.raycast_batch(orig, dirs, 20.0)
    assert batch.handle[0] != -1 and batch.handle[0] != wall
    assert batch.distance[0] == pytest.approx(5.0)
    assert batch.handle[1] == wall and batch.distance[1] == pytest.approx(2.0)
    assert batch.handle[2] == wall
    for i in range(3):
        hit = w.raycast(orig[i], dirs[i], 20.0)
        assert batch.handle[i] == hit.handle
        assert batch.distance[i] == pytest.approx(hit.distance)


def test_world_batch_layer_mask_and_ignore() -> None:
    w = PhysicsWorld()
    h = w.add_mesh(_grid(amplitude=0.0), layer=2)
    o, d = [(0.0, 5.0, 0.0)], [(0.0, -1.0, 0.0)]
    assert w.raycast_batch(o, d, layer_mask=2).handle[0] == h
    assert w.raycast_batch(o, d, layer_mask=1).handle[0] == -1
    assert w.raycast_batch(o, d, ignore=h).handle[0] == -1
    assert w.raycast_batch(o, d, max_distance=1.0).handle[0] == -1


def test_mesh_fast_raycast_identity_flag() -> None:
    w = PhysicsWorld()
    h = w.add_mesh(_grid())
    mesh = w.get_mesh(h)
    assert mesh.pose_is_identity
    w.set_mesh_pose(h, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    assert mesh.pose_is_identity
    w.set_mesh_pose(h, (1.0, 0.0, 0.0), (0.0, 0.3, 0.0, 0.954))
    assert not mesh.pose_is_identity
    hit = w.raycast((0.0, 5.0, 0.0), (0.0, -1.0, 0.0))
    assert hit is not None and hit.point[1] < 5.0
