"""Верификация Collider: is_trigger (триггер пропускает тела и шлёт Trigger-события)."""

from __future__ import annotations

import pytest

from engine.core.math3d import Vector3
from engine.core.physics_components import ColliderComponent, RigidBodyComponent
from engine.core.scene import Scene
from engine.physics.body import RigidBody
from engine.physics.collider import Collider
from engine.physics.events import (
    CollisionEnter,
    CollisionExit,
    TriggerEnter,
    TriggerExit,
    TriggerStay,
)
from engine.physics.shape import BoxShape, SphereShape
from engine.physics.world import PhysicsWorld

DT = 1.0 / 60.0


def _kinds(events) -> list[str]:
    return [type(e).__name__ for e in events]


def _count(events, kind) -> int:
    return sum(1 for e in events if type(e) is kind)


# --- уровень PhysicsWorld ------------------------------------------------------

def _slab_world(is_trigger: bool):
    """Плита 4×1×4 на y=0 и шар, падающий на неё с y=3."""
    world = PhysicsWorld()
    slab = world.add_static(
        Collider(BoxShape((2.0, 0.5, 2.0)), is_trigger=is_trigger),
        (0.0, 0.0, 0.0),
    )
    ball = RigidBody(position=(0.0, 3.0, 0.0), can_sleep=False)
    ball_h = world.add_body(ball, Collider(SphereShape(0.25)))
    return world, slab, ball, ball_h


def test_solid_collider_stops_body_and_sends_collision_enter() -> None:
    world, _, ball, _ = _slab_world(False)
    events = []
    for _ in range(180):
        world.step()
        events += world.pop_events()
    assert ball.position[1] == pytest.approx(0.5 + 0.25, abs=0.03)
    assert _count(events, CollisionEnter) == 1
    assert _count(events, TriggerEnter) == 0
    assert _count(events, TriggerStay) == 0


def test_trigger_collider_lets_body_through() -> None:
    world, _, ball, _ = _slab_world(True)
    for _ in range(180):
        world.step()
    assert ball.position[1] < -2.0  # упало сквозь плиту


def test_trigger_sends_enter_then_exit_and_no_collision_events() -> None:
    world, slab, _, ball_h = _slab_world(True)
    events = []
    for _ in range(180):
        world.step()
        events += world.pop_events()
    assert _count(events, CollisionEnter) == 0
    assert _count(events, CollisionExit) == 0
    kinds = [k for k in _kinds(events) if k != "TriggerStay"]
    assert kinds == ["TriggerEnter", "TriggerExit"]
    enter = next(e for e in events if type(e) is TriggerEnter)
    assert enter.a == slab  # в Trigger-событии a — всегда триггер
    assert enter.b == ball_h


def test_trigger_sends_stay_every_step_while_inside() -> None:
    world, _, _, _ = _slab_world(True)
    events = []
    for _ in range(180):
        world.step()
        events += world.pop_events()
    enter_i = _kinds(events).index("TriggerEnter")
    exit_i = _kinds(events).index("TriggerExit")
    stays = _kinds(events)[enter_i + 1:exit_i]
    assert len(stays) >= 3
    assert set(stays) == {"TriggerStay"}


def test_trigger_does_not_push_horizontally_moving_body() -> None:
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    world.add_static(Collider(BoxShape((1.0, 1.0, 1.0)), is_trigger=True))
    body = RigidBody(position=(-4.0, 0.0, 0.0), linear_velocity=(3.0, 0.0, 0.0),
                     linear_damping=0.0, can_sleep=False)
    world.add_body(body, Collider(SphereShape(0.5)))
    for _ in range(120):
        world.step()
    assert body.linear_velocity[0] == pytest.approx(3.0, abs=1e-6)
    assert body.position[0] == pytest.approx(-4.0 + 3.0 * 2.0, abs=0.01)


def test_solid_collider_blocks_the_same_horizontal_body() -> None:
    """Контроль к тесту выше: без is_trigger тело останавливается."""
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    world.add_static(Collider(BoxShape((1.0, 1.0, 1.0))))
    body = RigidBody(position=(-4.0, 0.0, 0.0), linear_velocity=(3.0, 0.0, 0.0),
                     linear_damping=0.0, can_sleep=False)
    world.add_body(body, Collider(SphereShape(0.5)))
    for _ in range(120):
        world.step()
    assert body.position[0] < -1.0


def test_kinematic_body_passes_through_trigger() -> None:
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    trig = world.add_static(Collider(BoxShape((1.0, 1.0, 1.0)), is_trigger=True))
    kin = RigidBody(body_type="kinematic", position=(-4.0, 0.0, 0.0),
                    linear_velocity=(3.0, 0.0, 0.0))
    kin_h = world.add_body(kin, Collider(SphereShape(0.5)))
    events = []
    for _ in range(120):
        world.step()
        events += world.pop_events()
    assert kin.position[0] == pytest.approx(2.0, abs=0.05)
    enters = [e for e in events if type(e) is TriggerEnter]
    exits = [e for e in events if type(e) is TriggerExit]
    assert len(enters) == 1 and len(exits) == 1
    assert enters[0].a == trig and enters[0].b == kin_h


def test_kinematic_solid_blocks_dynamic_but_kinematic_trigger_does_not() -> None:
    def run(is_trigger: bool) -> float:
        world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
        world.add_body(
            RigidBody(body_type="kinematic", position=(-4.0, 0.0, 0.0),
                      linear_velocity=(2.0, 0.0, 0.0)),
            Collider(BoxShape((0.5, 0.5, 0.5)), is_trigger=is_trigger))
        ball = RigidBody(position=(0.0, 0.0, 0.0), can_sleep=False)
        world.add_body(ball, Collider(SphereShape(0.25)))
        for _ in range(180):
            world.step()
        return float(ball.position[0])

    assert run(False) > 1.0  # твёрдая кинематика расталкивает
    assert run(True) == pytest.approx(0.0, abs=1e-6)  # триггер не трогает


def test_dynamic_trigger_with_own_body_falls_through_floor_and_reports() -> None:
    world = PhysicsWorld()
    floor = world.add_static(Collider(BoxShape((20.0, 0.5, 20.0))),
                             (0.0, -0.5, 0.0))
    zone = RigidBody(position=(0.0, 3.0, 0.0), can_sleep=False)
    zone_h = world.add_body(zone, Collider(SphereShape(0.5), is_trigger=True))
    events = []
    for _ in range(180):
        world.step()
        events += world.pop_events()
    assert zone.position[1] < -2.0  # триггерное тело не опирается на пол
    enters = [e for e in events if type(e) is TriggerEnter]
    assert [(e.a, e.b) for e in enters] == [(zone_h, floor)]
    assert _count(events, CollisionEnter) == 0


def test_kinematic_trigger_over_resting_body_does_not_push_it() -> None:
    world = PhysicsWorld()
    world.add_static(Collider(BoxShape((20.0, 0.5, 20.0))), (0.0, -0.5, 0.0))
    ball = RigidBody(position=(0.0, 0.25, 0.0), can_sleep=False)
    ball_h = world.add_body(ball, Collider(SphereShape(0.25)))
    trig = world.add_body(
        RigidBody(body_type="kinematic", position=(-3.0, 1.0, 0.0),
                  linear_velocity=(2.0, 0.0, 0.0)),
        Collider(BoxShape((0.5, 0.6, 0.5)), is_trigger=True))
    events = []
    for _ in range(180):
        world.step()
        events += world.pop_events()
    assert abs(ball.position[0]) < 0.01
    assert ball.position[1] == pytest.approx(0.25, abs=0.02)
    enter = next(e for e in events if type(e) is TriggerEnter)
    assert (enter.a, enter.b) == (trig, ball_h)


def test_two_triggers_do_not_report_each_other() -> None:
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    world.add_static(Collider(BoxShape((1.0, 1.0, 1.0)), is_trigger=True))
    world.add_body(RigidBody(position=(0.5, 0.0, 0.0)),
                   Collider(SphereShape(0.5), is_trigger=True))
    for _ in range(10):
        world.step()
        assert world.pop_events() == []


def test_trigger_does_not_wake_sleeping_body() -> None:
    world = PhysicsWorld()
    world.add_static(Collider(BoxShape((20.0, 0.5, 20.0))), (0.0, -0.5, 0.0))
    ball = RigidBody(position=(0.0, 0.25, 0.0))
    world.add_body(ball, Collider(SphereShape(0.25)))
    for _ in range(180):
        world.step()
    assert ball.sleeping
    world.add_body(
        RigidBody(body_type="kinematic", position=(-3.0, 1.0, 0.0),
                  linear_velocity=(2.0, 0.0, 0.0)),
        Collider(BoxShape((0.5, 0.6, 0.5)), is_trigger=True))
    events = []
    for _ in range(120):
        world.step()
        events += world.pop_events()
    assert ball.sleeping  # проход триггера сон не прерывает
    assert _count(events, TriggerEnter) == 1  # но событие о входе приходит


def test_trigger_does_not_collide_with_body_that_stays_inside() -> None:
    """Тело, уложенное внутри триггера, остаётся на месте (триггер не выталкивает)."""
    world = PhysicsWorld()
    world.add_static(Collider(BoxShape((20.0, 0.5, 20.0))), (0.0, -0.5, 0.0))
    world.add_static(Collider(BoxShape((1.0, 1.0, 1.0)), is_trigger=True),
                     (0.0, 1.0, 0.0))
    ball = RigidBody(position=(0.0, 1.0, 0.0), can_sleep=False)
    world.add_body(ball, Collider(SphereShape(0.25)))
    for _ in range(120):
        world.step()
    assert ball.position[1] == pytest.approx(0.25, abs=0.02)
    assert abs(ball.position[0]) < 1e-6


# --- компонент -------------------------------------------------------------------

def _scene_with_slab(is_trigger: bool, ball_y: float = 3.0):
    scene = Scene()
    slab = scene.create_game_object("Slab", position=Vector3(0, 0, 0))
    slab_col = slab.add_component(ColliderComponent, shape="box", size=(4, 1, 4),
                                  is_trigger=is_trigger)
    ball = scene.create_game_object("Ball", position=Vector3(0, ball_y, 0))
    ball.add_component(ColliderComponent, shape="sphere", radius=0.25)
    rb = ball.add_component(RigidBodyComponent, can_sleep=False)
    return scene, slab, slab_col, ball, rb


def _body(scene, go) -> RigidBody:
    return scene.physics.world.get_body(scene.physics.handle_of(go))


def _run(scene, steps: int) -> None:
    for _ in range(steps):
        scene.fixed_update(DT)
        scene.update(DT)  # доставка отложенных событий подписчикам


def test_component_default_is_not_trigger() -> None:
    scene = Scene()
    go = scene.create_game_object("A")
    col = go.add_component(ColliderComponent, shape="box")
    assert col.is_trigger is False


def test_component_solid_collides_and_publishes_collision_enter() -> None:
    scene, _, _, ball, _ = _scene_with_slab(False)
    seen: list = []
    for kind in (CollisionEnter, TriggerEnter):
        scene.events.subscribe(kind, seen.append)
    _run(scene, 180)
    assert _body(scene, ball).position[1] == pytest.approx(0.75, abs=0.03)
    assert _kinds(seen) == ["CollisionEnter"]


def test_component_trigger_passes_through_and_publishes_trigger_events() -> None:
    scene, slab, _, ball, _ = _scene_with_slab(True)
    seen: list = []
    for kind in (CollisionEnter, TriggerEnter, TriggerExit):
        scene.events.subscribe(kind, seen.append)
    _run(scene, 180)
    assert _body(scene, ball).position[1] < -2.0
    assert _kinds(seen) == ["TriggerEnter", "TriggerExit"]
    assert seen[0].user_a is slab and seen[0].user_b is ball


def test_component_runtime_switch_to_trigger_releases_resting_body() -> None:
    scene, _, col, ball, _ = _scene_with_slab(False, ball_y=1.0)
    _run(scene, 120)
    assert _body(scene, ball).position[1] == pytest.approx(0.75, abs=0.03)
    col.is_trigger = True
    _run(scene, 120)
    assert _body(scene, ball).position[1] < -1.0  # провалился сквозь плиту


def test_component_runtime_switch_to_solid_catches_body() -> None:
    scene, _, col, ball, _ = _scene_with_slab(True, ball_y=8.0)
    _run(scene, 5)
    col.is_trigger = False
    _run(scene, 180)
    assert _body(scene, ball).position[1] == pytest.approx(0.75, abs=0.03)


def test_component_runtime_switch_publishes_events_of_the_new_mode() -> None:
    scene, _, col, _, _ = _scene_with_slab(False, ball_y=1.0)
    seen: list = []
    for kind in (CollisionEnter, CollisionExit, TriggerEnter, TriggerExit):
        scene.events.subscribe(kind, seen.append)
    _run(scene, 120)
    seen.clear()
    col.is_trigger = True
    _run(scene, 120)
    kinds = set(_kinds(seen))
    assert "TriggerEnter" in kinds
    assert "CollisionEnter" not in kinds


def test_component_trigger_with_rigidbody_is_kinematic_zone() -> None:
    scene = Scene()
    scene.create_game_object("Floor", position=Vector3(0, -0.5, 0),
                             scale=Vector3(50, 1, 50)).add_component(
        ColliderComponent, shape="box", size=(1, 1, 1))
    ball = scene.create_game_object("Ball", position=Vector3(0, 0.25, 0))
    ball.add_component(ColliderComponent, shape="sphere", radius=0.25)
    ball.add_component(RigidBodyComponent, can_sleep=False)
    zone = scene.create_game_object("Zone", position=Vector3(-3, 1.0, 0))
    zone.add_component(ColliderComponent, shape="box", size=(1, 1.2, 1),
                       is_trigger=True)
    zone.add_component(RigidBodyComponent, body_type="kinematic")
    seen: list = []
    scene.events.subscribe(TriggerEnter, seen.append)
    for i in range(180):  # кинематику двигает трансформ: 2 м/с вдоль X
        zone.transform.position = Vector3(-3.0 + 2.0 * DT * i, 1.0, 0.0)
        _run(scene, 1)
    assert zone.transform.position.x > 2.0
    assert abs(_body(scene, ball).position[0]) < 0.01
    assert len(seen) == 1 and seen[0].user_a is zone and seen[0].user_b is ball
