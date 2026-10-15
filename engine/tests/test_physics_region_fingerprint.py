"""``PhysicsWorld.region_fingerprint``: отпечаток коллайдеров в области.

Отпечаток нужен быстрому пути покоя ``CharacterController.step``: равные
отпечатки значит, что любой запрос в области даст тот же ответ.
"""

from __future__ import annotations

import numpy as np

from engine.physics.collider import Collider
from engine.physics.shape import BoxShape
from engine.physics.world import PhysicsWorld

NEAR = ((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))


class _OpaqueShape:
    """Форма вне хранилища (``KIND_OTHER``): известен только её AABB."""

    def aabb(self, position, rotation):
        p = np.asarray(position, dtype=float)
        return p - 1.0, p + 1.0


def _world() -> tuple[PhysicsWorld, int]:
    world = PhysicsWorld()
    box = world.add_static(Collider(BoxShape((0.5, 0.5, 0.5))), (0.0, 0.0, 0.0))
    return world, box


def test_same_state_gives_same_fingerprint():
    world, _ = _world()

    assert world.region_fingerprint(*NEAR) == world.region_fingerprint(*NEAR)


def test_collider_pose_change_inside_changes_fingerprint():
    world, box = _world()
    before = world.region_fingerprint(*NEAR)

    world.set_pose(box, (0.25, 0.0, 0.0))

    assert world.region_fingerprint(*NEAR) != before


def test_collider_added_inside_changes_fingerprint():
    world, _ = _world()
    before = world.region_fingerprint(*NEAR)

    world.add_static(Collider(BoxShape((0.2, 0.2, 0.2))), (1.0, 0.0, 0.0))

    assert world.region_fingerprint(*NEAR) != before


def test_movement_outside_does_not_change_fingerprint():
    world, _ = _world()
    far = world.add_static(Collider(BoxShape((0.5, 0.5, 0.5))), (50.0, 0.0, 0.0))
    before = world.region_fingerprint(*NEAR)

    world.set_pose(far, (60.0, 0.0, 0.0))

    assert world.region_fingerprint(*NEAR) == before


def test_ignored_body_is_not_part_of_fingerprint():
    world, box = _world()
    other = world.add_static(Collider(BoxShape((0.2, 0.2, 0.2))), (1.0, 0.0, 0.0))
    before = world.region_fingerprint(*NEAR, ignore=other)

    world.set_pose(other, (1.5, 0.0, 0.0))

    assert world.region_fingerprint(*NEAR, ignore=other) == before
    assert world.region_fingerprint(*NEAR) != before
    assert world.region_fingerprint(*NEAR, ignore=box) != before


def test_layer_change_changes_fingerprint():
    world = PhysicsWorld()
    collider = Collider(BoxShape((0.5, 0.5, 0.5)), layer=1)
    world.add_static(collider, (0.0, 0.0, 0.0))
    before = world.region_fingerprint(*NEAR)

    collider.layer = 2

    assert world.region_fingerprint(*NEAR) != before


def test_empty_region_has_a_fingerprint():
    world, _ = _world()

    assert world.region_fingerprint((100.0,) * 3, (101.0,) * 3) is not None


def test_opaque_shape_in_region_refuses_fingerprint():
    world, _ = _world()
    world.add_static(Collider(_OpaqueShape()), (0.0, 1.0, 0.0))

    assert world.region_fingerprint(*NEAR) is None
    assert world.region_fingerprint((100.0,) * 3, (101.0,) * 3) is not None
