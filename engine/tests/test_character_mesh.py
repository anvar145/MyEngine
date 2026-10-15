"""CharacterController на mesh-коллайдерах (A11)."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from character_kit import WALK, foot_y, make_char, settle, walk

from engine.physics.bvh import Bvh
from engine.physics.world import PhysicsWorld


def _quad(size: float, y: float = 0.0) -> Bvh:
    v = np.array([[-size, y, -size], [size, y, -size], [size, y, size], [-size, y, size]])
    return Bvh(v, np.array([[0, 2, 1], [0, 3, 2]]))  # нормаль +Y


def _grid_floor(cells: int = 40, size: float = 40.0) -> Bvh:
    """Плоский пол из cells² клеток (2 cells² треугольников)."""
    xs = np.linspace(-size, size, cells + 1)
    gx, gz = np.meshgrid(xs, xs, indexing="ij")
    verts = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], axis=1)
    tris = []
    for i in range(cells):
        for j in range(cells):
            a = i * (cells + 1) + j
            b, c, d = a + 1, a + cells + 1, a + cells + 2
            tris += [[a, b, c], [b, d, c]]
    return Bvh(verts, np.array(tris))


def _mesh_world(bvh: Bvh) -> PhysicsWorld:
    world = PhysicsWorld()
    world.add_mesh(bvh)
    return world


def test_character_stands_on_mesh_floor() -> None:
    char = make_char(_mesh_world(_quad(50.0)), y=3.0)
    settle(char, 90)
    assert char.grounded
    assert foot_y(char) == pytest.approx(0.0, abs=0.03)
    assert np.allclose(char.ground_normal, (0.0, 1.0, 0.0), atol=1e-6)


def test_character_does_not_fall_through_mesh_over_time() -> None:
    char = make_char(_mesh_world(_quad(50.0)), y=0.95)
    settle(char, 600)
    assert foot_y(char) > -0.03 and char.grounded


def test_character_walks_on_mesh_floor() -> None:
    char = make_char(_mesh_world(_grid_floor()), y=1.0)
    settle(char, 30)
    start = char.position.copy()
    walk(char, (WALK, 0.0, 0.0), 2.0)
    assert char.position[0] - start[0] == pytest.approx(2 * WALK, rel=0.05)
    assert foot_y(char) == pytest.approx(0.0, abs=0.03)
    assert char.grounded


def test_character_blocked_by_mesh_wall() -> None:
    world = PhysicsWorld()
    world.add_mesh(_quad(50.0))
    # стена x = 5 из двух треугольников
    wall = np.array([[5.0, 0.0, -10.0], [5.0, 4.0, -10.0], [5.0, 4.0, 10.0],
                     [5.0, 0.0, 10.0]])
    world.add_mesh(Bvh(wall, np.array([[0, 1, 2], [0, 2, 3]])))
    char = make_char(world)
    settle(char, 30)
    walk(char, (WALK, 0.0, 0.0), 3.0)
    assert char.position[0] < 5.0 - char.radius + 0.01
    assert char.position[0] > 5.0 - char.radius - 0.1


def test_character_climbs_gentle_mesh_slope() -> None:
    """Пандус 20° из треугольников: персонаж поднимается и остаётся на земле."""
    t = math.radians(20.0)
    length = 10.0
    v = np.array([[2.0, 0.0, -5.0], [2.0 + length * math.cos(t), length * math.sin(t), -5.0],
                  [2.0 + length * math.cos(t), length * math.sin(t), 5.0],
                  [2.0, 0.0, 5.0]])
    world = PhysicsWorld()
    world.add_mesh(_quad(50.0))
    world.add_mesh(Bvh(v, np.array([[0, 2, 1], [0, 3, 2]])))
    char = make_char(world)
    settle(char, 30)
    walk(char, (WALK, 0.0, 0.0), 2.0)
    assert foot_y(char) > 0.5
    assert char.grounded


def test_depenetrate_pushes_out_of_mesh() -> None:
    char = make_char(_mesh_world(_quad(50.0)), y=0.5)  # по пояс в полу
    char.move((0.0, 0.0, 0.0))
    assert foot_y(char) >= -0.03


def test_rest_fast_path_still_wakes_when_mesh_moves() -> None:
    world = _mesh_world(_quad(50.0))
    char = make_char(world)
    settle(char, 60)
    y0 = char.position[1]
    handle = next(iter(world._meshes))
    world.set_mesh_pose(handle, (0.0, -2.0, 0.0))  # пол уехал вниз
    settle(char, 120)
    assert char.position[1] < y0 - 1.5


def test_shapecast_cost_on_big_mesh_is_small() -> None:
    """Свип капсулы по BVH из ~3200 треугольников - единицы миллисекунд."""
    char = make_char(_mesh_world(_grid_floor(40)), y=1.0)
    settle(char, 30)
    started = time.perf_counter()
    for _ in range(100):
        char._sweep(np.array([1.0, 0.0, 0.0]), 0.1)
        char._sweep(np.array([0.0, -1.0, 0.0]), 0.1)
    per_call = (time.perf_counter() - started) / 200
    assert per_call < 5e-3
