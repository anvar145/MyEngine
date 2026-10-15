"""Mesh-коллайдеры в PhysicsWorld: луч, свип, overlap, отпечаток (A11)."""

from __future__ import annotations

import numpy as np
import pytest

from engine.physics.body import RigidBody
from engine.physics.bvh import Bvh
from engine.physics.collider import Collider
from engine.physics.shape import BoxShape, CapsuleShape, SphereShape
from engine.physics.world import PhysicsWorld

IDENT = (0.0, 0.0, 0.0, 1.0)


def _floor(size: float = 10.0, y: float = 0.0) -> Bvh:
    v = np.array([[-size, y, -size], [size, y, -size], [size, y, size], [-size, y, size]])
    return Bvh(v, np.array([[0, 2, 1], [0, 3, 2]]))  # нормаль +Y


def _box(half: float = 1.0) -> Bvh:
    v = np.array([[x, y, z] for x in (-half, half) for y in (-half, half)
                  for z in (-half, half)])
    t = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
         [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    return Bvh(v, np.array(t))


def test_raycast_hits_mesh_floor() -> None:
    w = PhysicsWorld()
    h = w.add_mesh(_floor(), user_data="пол")
    hit = w.raycast((1.0, 5.0, 2.0), (0.0, -1.0, 0.0))
    assert hit is not None and hit.handle == h
    assert hit.distance == pytest.approx(5.0)
    assert hit.point == pytest.approx((1.0, 0.0, 2.0))
    assert hit.normal == pytest.approx((0.0, 1.0, 0.0))
    assert hit.user_data == "пол"
    assert w.raycast((50.0, 5.0, 0.0), (0.0, -1.0, 0.0)) is None


def test_raycast_picks_nearest_between_body_and_mesh() -> None:
    w = PhysicsWorld()
    w.add_mesh(_floor())
    body = w.add_body(RigidBody("static", position=(0.0, 1.0, 0.0)),
                      Collider(BoxShape((0.5, 0.5, 0.5))))
    hit = w.raycast((0.0, 5.0, 0.0), (0.0, -1.0, 0.0))
    assert hit.handle == body and hit.distance == pytest.approx(3.5)
    # тело убрано - луч доходит до меша
    w.remove_body(body)
    assert w.raycast((0.0, 5.0, 0.0), (0.0, -1.0, 0.0)).distance == pytest.approx(5.0)


def test_mesh_pose_and_rotation() -> None:
    w = PhysicsWorld()
    h = w.add_mesh(_floor(), position=(0.0, 2.0, 0.0))
    assert w.raycast((0, 5, 0), (0, -1, 0)).distance == pytest.approx(3.0)
    w.set_mesh_pose(h, (0.0, 0.0, 0.0), IDENT)
    assert w.raycast((0, 5, 0), (0, -1, 0)).distance == pytest.approx(5.0)
    s = np.sqrt(0.5)  # поворот на 90° вокруг Z: пол становится стеной x = 0
    w.set_mesh_pose(h, (3.0, 0.0, 0.0), (0.0, 0.0, s, s))
    hit = w.raycast((10.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    assert hit.distance == pytest.approx(7.0)
    assert abs(hit.normal[0]) == pytest.approx(1.0)


def test_layer_mask_and_trigger_filters() -> None:
    w = PhysicsWorld()
    w.add_mesh(_floor(), layer=2)
    assert w.raycast((0, 5, 0), (0, -1, 0), layer_mask=1) is None
    assert w.raycast((0, 5, 0), (0, -1, 0), layer_mask=2) is not None
    t = PhysicsWorld()
    t.add_mesh(_floor(), is_trigger=True)
    assert t.raycast((0, 5, 0), (0, -1, 0)) is None
    assert t.raycast((0, 5, 0), (0, -1, 0), include_triggers=True) is not None
    assert t.overlap_sphere((0, 0.1, 0), 0.5) == []


def test_overlap_sphere_and_box() -> None:
    w = PhysicsWorld()
    h = w.add_mesh(_floor())
    assert w.overlap_sphere((0.0, 0.3, 0.0), 0.5) == [h]
    assert w.overlap_sphere((0.0, 0.8, 0.0), 0.5) == []
    assert w.overlap(BoxShape((0.5, 0.5, 0.5)), (0.0, 0.4, 0.0)) == [h]
    assert w.overlap(BoxShape((0.5, 0.5, 0.5)), (0.0, 0.6, 0.0)) == []
    assert w.overlap(CapsuleShape(0.3, 0.5), (0.0, 0.7, 0.0)) == [h]


def test_shapecast_capsule_lands_on_floor() -> None:
    w = PhysicsWorld()
    h = w.add_mesh(_floor())
    cap = CapsuleShape(0.3, 0.5)  # от низа до верха: 0.5 + 2 * 0.3 = 1.1
    hit = w.shapecast(cap, (0.0, 3.0, 0.0), IDENT, (0.0, -1.0, 0.0), 10.0)
    assert hit is not None and hit.handle == h
    # центр на высоте half_height + radius = 0.8 в момент касания
    assert 3.0 - hit.distance == pytest.approx(0.8, abs=2e-3)
    assert hit.normal == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)
    assert w.shapecast(cap, (0.0, 3.0, 0.0), IDENT, (1.0, 0.0, 0.0), 5.0) is None


def test_shapecast_hits_wall_of_box_mesh() -> None:
    w = PhysicsWorld()
    w.add_mesh(_box(1.0), position=(5.0, 0.0, 0.0))
    ball = SphereShape(0.25)
    hit = w.shapecast(ball, (0.0, 0.0, 0.0), IDENT, (1.0, 0.0, 0.0), 10.0)
    assert hit is not None
    assert hit.distance == pytest.approx(4.0 - 0.25, abs=2e-3)
    assert hit.normal == pytest.approx((-1.0, 0.0, 0.0), abs=1e-6)


def test_shapecast_box_against_mesh() -> None:
    w = PhysicsWorld()
    w.add_mesh(_floor())
    hit = w.shapecast(BoxShape((0.5, 0.5, 0.5)), (0.0, 4.0, 0.0), IDENT,
                      (0.0, -1.0, 0.0), 10.0)
    assert hit is not None
    assert 4.0 - hit.distance == pytest.approx(0.5, abs=2e-3)
    assert hit.normal[1] > 0.99


def test_mesh_contacts_report_depth_along_normal() -> None:
    w = PhysicsWorld()
    w.add_mesh(_floor())
    contacts = w.mesh_contacts(SphereShape(0.5), (0.0, 0.3, 0.0))
    assert len(contacts) == 2  # два треугольника пола
    best = max(contacts, key=lambda c: c[1])
    assert best[1] == pytest.approx(0.2)
    assert best[0] == pytest.approx((0.0, 1.0, 0.0))
    assert w.mesh_contacts(SphereShape(0.5), (0.0, 2.0, 0.0)) == []


def test_region_fingerprint_changes_with_mesh() -> None:
    w = PhysicsWorld()
    lo, hi = np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])
    empty = w.region_fingerprint(lo, hi)
    h = w.add_mesh(_floor())
    with_mesh = w.region_fingerprint(lo, hi)
    assert with_mesh != empty
    assert w.region_fingerprint(lo, hi) == with_mesh  # стабильно
    w.set_mesh_pose(h, (0.0, -0.5, 0.0))
    assert w.region_fingerprint(lo, hi) != with_mesh
    far = w.region_fingerprint(np.array([100.0, 0, 0]), np.array([101.0, 1, 1]))
    assert far == w.region_fingerprint(np.array([200.0, 0, 0]), np.array([201.0, 1, 1]))


def test_remove_and_clear() -> None:
    w = PhysicsWorld()
    h = w.add_mesh(_floor())
    assert w.has_mesh(h) and w.mesh_count == 1
    w.remove_mesh(h)
    assert not w.has_mesh(h)
    w.add_mesh(_floor())
    w.clear()
    assert w.mesh_count == 0
    assert w.raycast((0, 5, 0), (0, -1, 0)) is None


def test_mesh_does_not_stop_dynamic_body() -> None:
    """Известное ограничение: динамическое тело проходит сквозь mesh."""
    w = PhysicsWorld()
    w.add_mesh(_floor())
    body = w.add_body(RigidBody("dynamic", 1.0, position=(0.0, 2.0, 0.0)),
                      Collider(SphereShape(0.5)))
    for _ in range(120):
        w.step()
    assert w.get_body(body).position[1] < -1.0
