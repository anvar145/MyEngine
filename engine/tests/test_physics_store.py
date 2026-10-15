"""SoA-хранилища физики: фасады тел/коллайдеров, swap-remove, детерминизм."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from engine.physics.body import RigidBody
from engine.physics.collider import Collider
from engine.physics.shape import BoxShape, CapsuleShape, SphereShape
from engine.physics.world import PhysicsWorld

FLOOR = BoxShape((60.0, 0.5, 60.0))


def _world() -> PhysicsWorld:
    world = PhysicsWorld()
    world.add_static(Collider(FLOOR), (0.0, -0.5, 0.0))
    return world


def _box(world: PhysicsWorld, pos) -> tuple[int, RigidBody]:
    body = RigidBody(mass=1.0, position=pos)
    return world.add_body(body, Collider(BoxShape((0.5,) * 3))), body


def _scene_a(world: PhysicsWorld) -> list[int]:
    """100 ящиков (сцена A бенча)."""
    handles = []
    for i in range(100):
        col, layer = i % 20, i // 20
        pos = (
            (col % 10) * 1.2 - 6.0 + 0.05 * (layer % 3),
            0.5 + (layer + col // 10 * 3) * 1.05,
            (col // 10) * 1.2 - 3.0,
        )
        handles.append(_box(world, pos)[0])
    return handles


def _scene_c(world: PhysicsWorld) -> list[int]:
    """Стопка из 20 ящиков (сцена C бенча)."""
    return [
        _box(world, (0.02 * (i % 3), 0.5 + 1.05 * i, 0.0))[0] for i in range(20)
    ]


def _digest(world: PhysicsWorld, handles: list[int]) -> str:
    poses = np.array(
        [np.concatenate([world.get_body(h).position, world.get_body(h).rotation])
         for h in handles]
    )
    return hashlib.sha256(poses.tobytes()).hexdigest()


def _run(build, steps: int) -> str:
    world = _world()
    handles = build(world)
    for _ in range(steps):
        world.step()
    return _digest(world, handles)


# ----------------------------------------------------------------- фасад тела


def test_standalone_body_has_private_store() -> None:
    body = RigidBody(position=(1.0, 2.0, 3.0), linear_velocity=(0.0, 4.0, 0.0))
    assert body.position.tolist() == [1.0, 2.0, 3.0]
    assert body.linear_velocity.tolist() == [0.0, 4.0, 0.0]
    body.position = (5.0, 6.0, 7.0)
    body.angular_velocity = (0.0, 0.0, 1.0)
    assert body.position.tolist() == [5.0, 6.0, 7.0]
    assert body.angular_velocity.tolist() == [0.0, 0.0, 1.0]
    assert body.linear_velocity.tolist() == [0.0, 4.0, 0.0]  # соседние поля целы


def test_body_view_writes_go_to_row() -> None:
    body = RigidBody()
    body.position[1] = 9.0
    body.linear_velocity[0] = 2.0
    assert body.position[1] == 9.0
    assert body.linear_velocity.tolist() == [2.0, 0.0, 0.0]


def test_adopt_keeps_python_object_and_state() -> None:
    world = _world()
    body = RigidBody(position=(1.0, 2.0, 3.0), linear_velocity=(1.0, 0.0, 0.0),
                     mass=2.0, linear_damping=0.3)
    handle = world.add_body(body, Collider(BoxShape((0.5,) * 3)))
    assert world.get_body(handle) is body
    assert body.position.tolist() == [1.0, 2.0, 3.0]
    assert body.linear_velocity.tolist() == [1.0, 0.0, 0.0]
    assert body.mass == 2.0
    assert body.inv_mass == pytest.approx(0.5)
    assert body.linear_damping == pytest.approx(0.3)


def test_body_cannot_join_two_worlds() -> None:
    world = _world()
    _, body = _box(world, (0.0, 3.0, 0.0))
    with pytest.raises(ValueError, match="уже добавлено"):
        world.add_body(body, Collider(BoxShape((0.5,) * 3)))


def test_stale_view_does_not_corrupt_world_after_growth() -> None:
    """Устаревший view не должен менять других тел и не должен влиять на мир.

    Рост ёмкости (больше 64 тел) переносит строки в новые массивы: view, взятый
    до этого, указывает на старый буфер; запись в него не доходит до мира.
    """
    world = _world()
    handle, body = _box(world, (0.0, 3.0, 0.0))
    stale = body.position  # view до роста хранилища
    others = [_box(world, (10.0 + i, 3.0, 0.0))[0] for i in range(130)]
    stale[0] = 999.0
    assert body.position[0] == 0.0  # мир не пострадал
    assert world.get_body(handle).position[0] == 0.0
    for i, h in enumerate(others):
        assert world.get_body(h).position[0] == 10.0 + i
    fresh = body.position  # свежий view работает
    fresh[0] = 5.0
    assert world.get_body(handle).position[0] == 5.0


def test_stale_view_after_swap_remove_is_not_the_bodys_row() -> None:
    """После swap-remove тело переезжает: заново взятый view указывает верно."""
    world = _world()
    _, first = _box(world, (0.0, 3.0, 0.0))
    removed, _ = _box(world, (1.0, 3.0, 0.0))
    _, last = _box(world, (2.0, 3.0, 0.0))
    stale_last = last.position
    world.remove_body(removed)  # последнее тело встаёт на место удалённого
    assert last.position.tolist() == [2.0, 3.0, 0.0]
    assert first.position.tolist() == [0.0, 3.0, 0.0]
    # старый view указывает на прежнюю строку: запись в него не должна
    # менять тело, переехавшее в чужую строку
    stale_last[0] = 999.0
    assert last.position[0] == 2.0
    last.position[0] = 4.0  # свежий view рабочий
    assert last.position[0] == 4.0
    assert first.position[0] == 0.0


# ------------------------------------------------------- add/remove/clear тел


def test_remove_body_returns_standalone_body_with_state() -> None:
    world = _world()
    handles = [_box(world, (float(i), 3.0, 0.0)) for i in range(5)]
    h2, body2 = handles[2]
    body2.linear_velocity = (1.0, 2.0, 3.0)
    world.remove_body(h2)
    assert h2 not in world
    assert body2.position.tolist() == [2.0, 3.0, 0.0]
    assert body2.linear_velocity.tolist() == [1.0, 2.0, 3.0]
    body2.position = (7.0, 7.0, 7.0)  # тело ещё живо как автономное
    for i, (h, _) in enumerate(handles):
        if h != h2:
            assert world.get_body(h).position.tolist() == [float(i), 3.0, 0.0]
    for _ in range(10):
        world.step()
    assert body2.position.tolist() == [7.0, 7.0, 7.0]  # мир его больше не двигает
    world.add_body(body2, Collider(BoxShape((0.5,) * 3)))  # можно добавить заново


def test_remove_then_add_keeps_row_mapping_consistent() -> None:
    world = _world()
    items = [_box(world, (float(i), 5.0, 0.0)) for i in range(20)]
    for h, _ in items[::3]:
        world.remove_body(h)
    kept = [(h, b) for k, (h, b) in enumerate(items) if k % 3]
    extra = [_box(world, (100.0 + i, 5.0, 0.0)) for i in range(10)]
    for h, body in kept + extra:
        assert world.get_body(h) is body
        assert world._bodies.row_of_handle[h] == body._row
        assert world._bodies.items[body._row] is body
        assert world._bodies.handle[body._row] == h
    world.step()  # шаг не падает на «дырявой» истории


def test_clear_detaches_bodies_and_colliders() -> None:
    world = _world()
    handle, body = _box(world, (1.0, 3.0, 0.0))
    col = world.get_colliders(handle)[0]
    lo, hi = (a.copy() for a in col.world_aabb)
    world.clear()
    assert len(world) == 0
    assert body.position.tolist() == [1.0, 3.0, 0.0]
    got_lo, got_hi = col.world_aabb
    assert got_lo.tolist() == lo.tolist() and got_hi.tolist() == hi.tolist()
    assert world._bodies.count == 0 and world._colliders.count == 0


# ------------------------------------------------------------------ коллайдер


def test_collider_world_state_follows_store_rows() -> None:
    world = _world()
    handle, body = _box(world, (2.0, 3.0, 4.0))
    col = world.get_colliders(handle)[0]
    assert col.world_position.tolist() == [2.0, 3.0, 4.0]
    lo, hi = col.world_aabb
    assert lo.tolist() == [1.5, 2.5, 3.5] and hi.tolist() == [2.5, 3.5, 4.5]
    world.set_pose(handle, (0.0, 10.0, 0.0))
    assert col.world_position.tolist() == [0.0, 10.0, 0.0]
    # ещё одно тело и удаление первого: коллайдер переехал, свойства верны
    _box(world, (50.0, 3.0, 0.0))
    world.remove_body(world.handles()[0])
    assert col.world_position.tolist() == [0.0, 10.0, 0.0]
    assert col.world_aabb[0].tolist() == [-0.5, 9.5, -0.5]
    assert body.position.tolist() == [0.0, 10.0, 0.0]


def test_collider_store_aabb_matches_shape_aabb_for_all_kinds() -> None:
    world = _world()
    rot = (0.2, 0.3, 0.1, 0.9)
    shapes = [BoxShape((0.5, 0.2, 0.8)), SphereShape(0.7), CapsuleShape(0.3, 0.6)]
    for i, shape in enumerate(shapes):
        body = RigidBody(position=(3.0 * i, 4.0, 1.0), rotation=rot)
        col = Collider(shape, offset=(0.1, 0.2, -0.3))
        handle = world.add_body(body, col)
        want_lo, want_hi = shape.aabb(col.world_position, col.world_rotation)
        lo, hi = world.get_colliders(handle)[0].world_aabb
        assert np.allclose(lo, want_lo, atol=1e-12)
        assert np.allclose(hi, want_hi, atol=1e-12)


def test_layer_mask_trigger_writes_reach_the_store() -> None:
    """layer/mask/is_trigger меняются и после add_body (фильтр читает хранилище)."""
    world = _world()
    handle, _ = _box(world, (0.0, 3.0, 0.0))
    col = world.get_colliders(handle)[0]
    store, row = world._colliders, col._row
    col.layer, col.mask = 1 << 40, -1  # 64 бита и «все слои» через отрицательное
    col.is_trigger = True
    assert int(store.layer[row]) == 1 << 40
    assert int(store.mask[row]) == (1 << 64) - 1
    assert bool(store.is_trigger[row])
    col.is_trigger = False
    assert not store.is_trigger[row]
    assert col.layer == 1 << 40 and col.mask == -1  # Python-значения сохраняются


# ---------------------------------------------------------------- детерминизм


@pytest.mark.parametrize("build, steps", [(_scene_a, 120), (_scene_c, 200)])
def test_bitwise_repeatable_run_to_run(build, steps) -> None:
    assert _run(build, steps) == _run(build, steps)


def test_result_does_not_depend_on_row_history() -> None:
    """Те же тела при другой истории add/remove (иные строки) — те же позы."""

    def straight() -> str:
        world = _world()
        return _after(world, _scene_c(world))

    def shuffled() -> str:
        world = _world()
        junk = [_box(world, (200.0 + 3 * i, 40.0, 0.0))[0] for i in range(7)]
        handles = _scene_c(world)
        for h in junk[::2]:  # дыры, swap-remove двигает строки тел сцены
            world.remove_body(h)
        for i in range(3):
            _box(world, (-200.0 - 3 * i, 40.0, 0.0))
        return _after(world, handles)

    def _after(world: PhysicsWorld, handles: list[int]) -> str:
        for _ in range(150):
            world.step()
        return _digest(world, handles)

    assert straight() == shuffled()
