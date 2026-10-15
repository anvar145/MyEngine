"""Заснувшее тело попадает в трансформ (B6): нет «висящих в воздухе» объектов."""

from __future__ import annotations

from engine.core.math3d import Vector3
from engine.core.physics_adapter import ScenePhysicsAdapter
from engine.core.physics_components import ColliderComponent, RigidBodyComponent
from engine.core.scene import Scene


def _scene() -> tuple[Scene, list]:
    scene = Scene("t")
    floor = scene.create_game_object(
        "Floor", position=Vector3(0, -0.5, 0), scale=Vector3(20, 1, 20)
    )
    floor.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    cubes = []
    for i, x in enumerate((-2.0, 2.0)):
        cube = scene.create_game_object(f"C{i}", position=Vector3(x, 3.0, 0))
        cube.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
        cube.add_component(RigidBodyComponent, body_type="dynamic")
        cubes.append(cube)
    return scene, cubes


def test_settle_puts_cubes_on_the_floor() -> None:
    scene, cubes = _scene()
    adapter = ScenePhysicsAdapter(scene)
    adapter.settle(240)
    for cube in cubes:
        assert abs(cube.transform.position.y - 0.5) < 0.05


def test_sleepers_are_written_to_transform_without_settle() -> None:
    scene, cubes = _scene()
    adapter = ScenePhysicsAdapter(scene)
    for _ in range(400):
        adapter.step(1 / 60)
    bodies = [adapter._fast[k].body for k in adapter._fast_dynamic.tolist()]
    assert all(b.sleeping for b in bodies)
    for cube, body in zip(cubes, bodies, strict=True):
        assert abs(cube.transform.position.y - float(body.position[1])) < 1e-4
        assert cube.transform.position.y < 1.0
