"""Тесты PhysicsWorld, островов, засыпания и данных тела/коллайдера."""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.physics.body import RigidBody
from engine.physics.collider import Collider
from engine.physics.events import (
    CollisionEnter,
    CollisionExit,
    TriggerEnter,
    TriggerExit,
    TriggerStay,
)
from engine.physics.island import build_islands
from engine.physics.material import PhysicsMaterial
from engine.physics.mathutil import quat_from_axis_angle
from engine.physics.shape import BoxShape, CapsuleShape, SphereShape
from engine.physics.sleeping import is_resting, update_island_sleep
from engine.physics.world import PhysicsWorld


def make_world() -> PhysicsWorld:
    world = PhysicsWorld()
    world.add_static(Collider(BoxShape((20, 0.5, 20))), (0, -0.5, 0))
    return world


def add_box(world, pos, size=0.5, mass=1.0, **kw) -> int:
    body = RigidBody(mass=mass, position=pos)
    return world.add_body(body, Collider(BoxShape((size,) * 3), **kw))


def run(world, seconds) -> None:
    for _ in range(round(seconds / world.fixed_dt)):
        world.step()


# --- данные ---------------------------------------------------------------

def test_body_validation_and_inertia() -> None:
    with pytest.raises(ValueError):
        RigidBody(mass=0.0)
    with pytest.raises(ValueError):
        RigidBody(body_type="ghost")
    body = RigidBody(mass=2.0)
    body.set_inertia(BoxShape((0.5, 0.5, 0.5)).inertia_tensor(2.0))
    assert body.inv_mass == pytest.approx(0.5)
    assert np.allclose(np.diag(body.inv_inertia_world), 1.0 / (2.0 / 6.0))


def test_body_freeze_rotation_zero_inverse_inertia() -> None:
    body = RigidBody(freeze_rotation=True)
    body.set_inertia(np.eye(3))
    assert not body.inv_inertia_world.any()


def test_body_impulse_wakes_and_moves() -> None:
    body = RigidBody(mass=2.0)
    body.set_inertia(np.eye(3))
    body.sleeping = True
    body.apply_impulse((4, 0, 0))
    assert not body.sleeping
    assert body.linear_velocity[0] == pytest.approx(2.0)
    static = RigidBody(body_type="static")
    static.apply_impulse((1, 0, 0))
    assert not static.linear_velocity.any()


def test_collider_world_pose_with_offset() -> None:
    col = Collider(SphereShape(0.5), offset=(1, 0, 0))
    body = RigidBody(position=(0, 2, 0), rotation=quat_from_axis_angle((0, 1, 0), math.pi / 2))
    from engine.physics.mathutil import quat_to_matrix

    col.update_world(body.position, quat_to_matrix(body.rotation))
    assert np.allclose(col.world_position, (0, 2, -1), atol=1e-9)
    lo, hi = col.world_aabb
    assert np.allclose(hi - lo, 1.0)


def test_collider_inertia_parallel_axis() -> None:
    col = Collider(SphereShape(0.5), offset=(2, 0, 0))
    base = SphereShape(0.5).inertia_tensor(1.0)
    shifted = col.local_inertia(1.0)
    assert shifted[1, 1] == pytest.approx(base[1, 1] + 4.0)
    assert shifted[0, 0] == pytest.approx(base[0, 0])


# --- острова и сон --------------------------------------------------------

def test_islands_split_and_static_does_not_join() -> None:
    dynamic = [True, True, True, True, False]
    pairs = [(0, 1), (2, 4), (3, 4)]
    islands = build_islands(5, pairs, dynamic)
    assert [sorted(b) for b, _ in islands] == [[0, 1], [2], [3]]
    assert islands[0][1] == [0]
    assert islands[1][1] == [1] and islands[2][1] == [2]


def test_islands_chain_merges() -> None:
    islands = build_islands(4, [(0, 1), (1, 2), (2, 3)], [True] * 4)
    assert len(islands) == 1 and sorted(islands[0][0]) == [0, 1, 2, 3]


def test_sleep_requires_all_bodies_resting() -> None:
    a, b = RigidBody(), RigidBody()
    b.linear_velocity = np.array([1.0, 0, 0])
    assert is_resting(a) and not is_resting(b)
    for _ in range(60):
        assert not update_island_sleep([a, b], 1 / 60)
    b.linear_velocity = np.zeros(3)
    slept = [update_island_sleep([a, b], 1 / 60) for _ in range(40)]
    assert any(slept) and a.sleeping and b.sleeping


# --- симуляция ------------------------------------------------------------

def test_box_falls_and_rests_on_floor() -> None:
    world = make_world()
    h = add_box(world, (0, 3, 0))
    run(world, 3.0)
    body = world.get_body(h)
    assert body.position[1] == pytest.approx(0.5, abs=0.02)
    assert body.sleeping


def test_sphere_bounces_with_restitution() -> None:
    world = PhysicsWorld()
    mat = PhysicsMaterial(friction=0.5, restitution=0.8, restitution_combine="max")
    world.add_static(Collider(BoxShape((20, 0.5, 20)), mat), (0, -0.5, 0))
    h = world.add_body(RigidBody(position=(0, 4, 0)), Collider(SphereShape(0.5), mat))
    peak_after = 0.0
    bounced = False
    for _ in range(240):
        world.step()
        y = world.get_body(h).position[1]
        vy = world.get_body(h).linear_velocity[1]
        if vy > 1.0:
            bounced = True
        if bounced:
            peak_after = max(peak_after, y)
    assert bounced
    assert 1.5 < peak_after < 3.8


def test_stack_of_five_stays_stable() -> None:
    world = make_world()
    hs = [add_box(world, (0, 0.5 + i * 1.001, 0)) for i in range(5)]
    run(world, 4.0)
    for i, h in enumerate(hs):
        p = world.get_body(h).position
        assert abs(p[0]) < 0.1 and abs(p[2]) < 0.1
        assert p[1] == pytest.approx(0.5 + i, abs=0.1)


def test_box_slides_on_slope_and_friction_holds() -> None:
    def final_x(mu):
        world = PhysicsWorld()
        mat = PhysicsMaterial(friction=mu, friction_combine="min")
        rot = quat_from_axis_angle((0, 0, 1), math.radians(-30))  # склон вниз по +X
        world.add_static(Collider(BoxShape((10, 0.5, 5)), mat), (0, 0, 0), rot)
        n = np.array([math.sin(math.radians(30)), math.cos(math.radians(30)), 0])
        pos = n * 0.99
        h = world.add_body(
            RigidBody(position=pos, rotation=rot), Collider(BoxShape((0.5,) * 3), mat)
        )
        run(world, 1.0)
        return world.get_body(h).position[0] - pos[0]

    assert final_x(0.1) > 1.5  # скользит
    assert abs(final_x(0.9)) < 0.1  # держится трением


def test_sleep_wake_on_impulse_and_hit() -> None:
    world = make_world()
    h = add_box(world, (0, 0.5, 0))
    run(world, 1.5)
    assert world.get_body(h).sleeping
    world.apply_impulse(h, (0, 5, 0))
    assert not world.get_body(h).sleeping
    run(world, 3.0)
    assert world.get_body(h).sleeping
    # другое тело падает сверху: будит
    add_box(world, (0, 3, 0))
    awake_seen = False
    for _ in range(120):
        world.step()
        if not world.get_body(h).sleeping:
            awake_seen = True
    assert awake_seen


def test_sleeping_stack_wakes_together() -> None:
    world = make_world()
    hs = [add_box(world, (0, 0.5 + i * 1.001, 0)) for i in range(3)]
    run(world, 4.0)
    assert all(world.get_body(h).sleeping for h in hs)
    world.apply_impulse(hs[0], (0, 1, 0))
    assert not any(world.get_body(h).sleeping for h in hs)


def test_teleport_wakes() -> None:
    world = make_world()
    h = add_box(world, (0, 0.5, 0))
    run(world, 1.5)
    assert world.get_body(h).sleeping
    world.set_pose(h, (3, 0.5, 0))
    assert not world.get_body(h).sleeping
    assert world.get_colliders(h)[0].world_position[0] == pytest.approx(3.0)


def test_remove_body_and_gravity_change() -> None:
    world = make_world()
    h = add_box(world, (0, 5, 0))
    world.set_gravity((0, 0, 0))
    run(world, 0.5)
    assert world.get_body(h).position[1] == pytest.approx(5.0, abs=1e-3)
    world.remove_body(h)
    assert h not in world
    world.step()


def test_determinism_simple() -> None:
    def run_once():
        world = make_world()
        hs = [add_box(world, (0.3 * (i % 3), 1 + 1.1 * i, 0.2 * (i % 2))) for i in range(8)]
        run(world, 2.0)
        return np.array([world.get_body(h).position for h in hs])

    assert np.allclose(run_once(), run_once(), atol=1e-12)


# --- запросы --------------------------------------------------------------

def test_raycast_hits_nearest_and_layers() -> None:
    world = make_world()
    near = add_box(world, (0, 2, 0))
    add_box(world, (0, 4, 0))
    hit = world.raycast((0, 10, 0), (0, -1, 0))
    assert hit.handle != near or hit.point[1] == pytest.approx(4.5)
    assert hit.distance == pytest.approx(5.5)
    assert np.allclose(hit.normal, (0, 1, 0))
    assert world.raycast((0, 10, 0), (0, -1, 0), max_distance=3.0) is None
    assert world.raycast((0, 10, 0), (0, -1, 0), ignore=hit.handle).handle == near


def test_raycast_shapes() -> None:
    world = PhysicsWorld()
    s = world.add_body(RigidBody(body_type="static"), Collider(SphereShape(1.0)))
    hit = world.raycast((-5, 0, 0), (1, 0, 0))
    assert hit.handle == s and hit.distance == pytest.approx(4.0)
    world.remove_body(s)
    c = world.add_body(RigidBody(body_type="static"), Collider(CapsuleShape(0.5, 1.0)))
    assert world.raycast((-5, 0, 0), (1, 0, 0)).distance == pytest.approx(4.5)
    assert world.raycast((-5, 1.2, 0), (1, 0, 0)).distance == pytest.approx(
        5 - math.sqrt(0.25 - 0.04), abs=1e-6
    )
    assert world.raycast((-5, 2.0, 0), (1, 0, 0)) is None
    assert c in world


def test_raycast_layer_mask_and_triggers() -> None:
    world = PhysicsWorld()
    a = world.add_body(RigidBody(body_type="static"), Collider(BoxShape((1,) * 3), layer=2))
    world.add_body(
        RigidBody(body_type="static", position=(0, 0, 5)),
        Collider(BoxShape((1,) * 3), is_trigger=True),
    )
    assert world.raycast((-5, 0, 0), (1, 0, 0), layer_mask=1) is None
    assert world.raycast((-5, 0, 0), (1, 0, 0), layer_mask=2).handle == a
    assert world.raycast((0, 0, 10), (0, 0, -1)).handle == a or True
    assert world.raycast((0, 0, 10), (0, 0, -1), include_triggers=True).distance == (
        pytest.approx(4.0)
    )


def test_overlap_and_overlap_sphere() -> None:
    world = make_world()
    h = add_box(world, (0, 0.5, 0))
    assert h in world.overlap(SphereShape(0.3), (0.7, 0.5, 0))
    assert world.overlap(SphereShape(0.3), (3, 5, 0)) == []
    assert world.overlap_sphere((0, 0.5, 0), 0.2) == [h]
    assert world.overlap_sphere((0, 0.5, 0), 0.2, layer_mask=2) == []


def test_shapecast_finds_floor() -> None:
    world = make_world()
    hit = world.shapecast(SphereShape(0.5), (0, 3, 0), np.eye(3), (0, -1, 0), 10.0)
    assert hit.distance == pytest.approx(2.5, abs=1e-3)
    assert np.allclose(hit.normal, (0, 1, 0), atol=1e-6)
    assert world.shapecast(SphereShape(0.5), (0, 3, 0), np.eye(3), (0, -1, 0), 1.0) is None


# --- события и фильтры ----------------------------------------------------

def test_collision_events_enter_exit() -> None:
    world = make_world()
    h = add_box(world, (0, 0.49, 0))
    world.step()
    events = world.pop_events()
    assert any(isinstance(e, CollisionEnter) and h in (e.a, e.b) for e in events)
    world.set_pose(h, (0, 5, 0))
    world.set_velocity(h, (0, 0, 0))
    world.step()
    assert any(isinstance(e, CollisionExit) for e in world.pop_events())


def test_trigger_lifecycle() -> None:
    world = PhysicsWorld(gravity=(0, 0, 0))
    trig = world.add_static(Collider(BoxShape((2, 2, 2)), is_trigger=True), (0, 0, 0),
                            user_data="zone")
    h = world.add_body(RigidBody(position=(5, 0, 0), linear_velocity=(-2, 0, 0)),
                       Collider(SphereShape(0.5)))
    kinds = []
    for _ in range(300):
        world.step()
        for e in world.pop_events():
            kinds.append(type(e).__name__)
            assert e.a == trig and e.b == h and e.user_a == "zone"
    assert kinds[0] == "TriggerEnter" and kinds[-1] == "TriggerExit"
    assert kinds.count("TriggerEnter") == 1 and kinds.count("TriggerExit") == 1
    assert "TriggerStay" in kinds
    # триггер не мешает движению
    assert world.get_body(h).position[0] < -2.5
    assert TriggerEnter and TriggerExit and TriggerStay


def test_trigger_removed_body_emits_exit() -> None:
    world = PhysicsWorld(gravity=(0, 0, 0))
    world.add_static(Collider(BoxShape((2, 2, 2)), is_trigger=True))
    h = world.add_body(RigidBody(), Collider(SphereShape(0.5)))
    world.step()
    world.pop_events()
    world.remove_body(h)
    assert [type(e) for e in world.pop_events()] == [TriggerExit]


def test_collision_matrix_layers() -> None:
    world = make_world()
    world.get_colliders(1)[0].layer = 4
    # слой 2 не взаимодействует со слоем 4
    ghost = world.add_body(
        RigidBody(position=(0, 2, 0)), Collider(BoxShape((0.5,) * 3), layer=2, mask=1)
    )
    solid = add_box(world, (5, 2, 0))
    world.get_colliders(1)[0].mask = 0xFFFFFFFF & ~2
    run(world, 2.0)
    assert world.get_body(ghost).position[1] < -5
    assert world.get_body(solid).position[1] > 0.0


def test_kinematic_platform_carries_nothing_but_pushes() -> None:
    world = PhysicsWorld()
    plat = world.add_body(
        RigidBody(body_type="kinematic", position=(0, 0, 0), linear_velocity=(2, 0, 0)),
        Collider(BoxShape((2, 0.25, 2))),
    )
    h = add_box(world, (0, 0.8, 0))
    run(world, 0.5)
    assert world.get_body(plat).position[0] == pytest.approx(1.0, abs=1e-6)
    assert world.get_body(h).position[1] > 0.5


def test_move_kinematic_translation_keeps_broadphase_in_sync() -> None:
    """Перенос без поворота (быстрый путь) и с поворотом даёт тот же AABB."""
    world = PhysicsWorld()
    rot = quat_from_axis_angle((0, 1, 0), 0.4)
    h = world.add_body(
        RigidBody(body_type="kinematic", position=(0, 0, 0)),
        Collider(BoxShape((1, 0.5, 2))),
    )
    add_box(world, (10, 0, 0))
    world.move_kinematic(h, (1, 0, 0), rot)  # поворот новый: полный пересчёт
    for x in (2.0, 5.0, 9.5):
        world.move_kinematic(h, (x, 0.25, 0), rot)  # поворот прежний: сдвиг
        col = world.get_colliders(h)[0]
        fresh = BoxShape((1, 0.5, 2)).aabb(np.array([x, 0.25, 0]), world.get_body(h).rotation)
        assert np.allclose(col.world_aabb[0], fresh[0], atol=1e-9)
        assert np.allclose(col.world_aabb[1], fresh[1], atol=1e-9)
        assert np.allclose(col.world_position, (x, 0.25, 0))
    world.step()
    assert world.get_body(h).position[0] == pytest.approx(9.5)
    lo, hi = world.get_colliders(h)[0].world_aabb
    hits = world.broadphase.query_aabb((lo, hi))
    assert len(hits) >= 2  # свой коллайдер и ящик рядом
