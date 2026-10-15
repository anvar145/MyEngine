"""MeshColliderComponent в адаптере физики сцены (A11)."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from engine.core import physics_adapter
from engine.core.math3d import Vector3
from engine.core.physics_components import (
    CharacterControllerComponent,
    MeshColliderComponent,
)
from engine.core.resources.asset_ref import AssetRef
from engine.core.scene import Scene

DT = 1.0 / 60.0
DOWN = (0.0, -1.0, 0.0)
REF = AssetRef(guid="a" * 32, path="Meshes/floor.obj")


@pytest.fixture
def quad(monkeypatch):
    """Подмена загрузчика геометрии: квадрат 2x2 в плоскости y=0, нормаль +Y."""
    calls: list[AssetRef] = []

    def loader(assets, ref):
        calls.append(ref)
        v = np.array([[-1.0, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]])
        return v, np.array([[0, 2, 1], [0, 3, 2]])

    monkeypatch.setattr(physics_adapter, "load_mesh_geometry", loader)
    return calls


def _scene(**kwargs):
    scene = Scene()
    scene.asset_source = object()
    go = scene.create_game_object("Level", **kwargs)
    comp = go.add_component(MeshColliderComponent, mesh=REF)
    return scene, go, comp


def _ray(scene, x=0.0, z=0.0):
    return scene.physics.world.raycast((x, 5.0, z), DOWN)


def test_mesh_collider_is_hit_by_ray_and_mapped_to_object(quad) -> None:
    scene, go, _ = _scene()
    scene.fixed_update(DT)
    hit = _ray(scene)
    assert hit is not None and hit.distance == pytest.approx(5.0)
    assert scene.physics.object_of(hit.handle) is go


def test_scale_is_baked_into_geometry(quad) -> None:
    scene, _, _ = _scene(scale=Vector3(10, 1, 10))
    scene.fixed_update(DT)
    assert _ray(scene, 8.0, 8.0) is not None
    assert _ray(scene, 12.0, 0.0) is None


def test_moving_object_moves_mesh_without_rebuild(quad) -> None:
    scene, go, _ = _scene()
    scene.fixed_update(DT)
    go.transform.position = Vector3(0, -2, 0)
    scene.fixed_update(DT)
    assert _ray(scene).distance == pytest.approx(7.0)
    assert len(quad) == 1  # геометрия читалась один раз


def test_scale_change_rebuilds(quad) -> None:
    scene, go, _ = _scene()
    scene.fixed_update(DT)
    assert _ray(scene, 5.0, 0.0) is None
    go.transform.scale = Vector3(10, 1, 10)
    scene.fixed_update(DT)
    assert _ray(scene, 5.0, 0.0) is not None


def test_two_instances_share_bvh(quad) -> None:
    scene, _, _ = _scene()
    other = scene.create_game_object("Other", position=Vector3(50, 0, 0))
    other.add_component(MeshColliderComponent, mesh=REF)
    scene.fixed_update(DT)
    assert scene.physics.world.mesh_count == 2
    assert len(quad) == 1


def test_field_edit_updates_layer_filter(quad) -> None:
    scene, _, comp = _scene()
    scene.fixed_update(DT)
    comp.layer = 4
    scene.fixed_update(DT)
    assert scene.physics.world.raycast((0, 5, 0), DOWN, layer_mask=1) is None
    assert scene.physics.world.raycast((0, 5, 0), DOWN, layer_mask=4) is not None


def test_disable_and_destroy_remove_mesh(quad) -> None:
    scene, go, comp = _scene()
    scene.fixed_update(DT)
    comp.enabled = False
    scene.fixed_update(DT)
    assert _ray(scene) is None
    comp.enabled = True
    scene.fixed_update(DT)
    assert _ray(scene) is not None
    go.destroy()
    scene.fixed_update(DT)
    assert scene.physics.world.mesh_count == 0


def test_missing_mesh_warns_once_and_creates_nothing(quad, caplog) -> None:
    scene = Scene()  # без asset_source
    scene.create_game_object("Broken").add_component(
        MeshColliderComponent, mesh=REF)
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            scene.fixed_update(DT)
    assert scene.physics.world.mesh_count == 0
    assert sum("MeshCollider" in r.message for r in caplog.records) == 1


def test_character_walks_on_scene_mesh(quad) -> None:
    scene = Scene()
    scene.asset_source = object()
    floor = scene.create_game_object("Floor", scale=Vector3(40, 1, 40))
    floor.add_component(MeshColliderComponent, mesh=REF)
    hero = scene.create_game_object("Hero", position=Vector3(0, 3, 0))
    hero.add_component(CharacterControllerComponent)
    for _ in range(180):
        scene.fixed_update(DT)
    assert scene.physics.is_grounded(hero)
    assert hero.transform.position.y < 1.2


def test_clear_drops_meshes(quad) -> None:
    scene, _, _ = _scene()
    scene.fixed_update(DT)
    scene.physics.clear()
    assert scene.physics.world.mesh_count == 0
    scene.fixed_update(DT)  # пересобирается
    assert scene.physics.world.mesh_count == 1


def test_scene_io_round_trip_keeps_mesh_ref_and_format_version() -> None:
    from engine.scene_io import SCENE_VERSION, scene_from_dict, scene_to_dict

    scene = Scene()
    go = scene.create_game_object("Level")
    go.add_component(MeshColliderComponent, mesh=REF, layer=2, friction=0.9)
    data = scene_to_dict(scene)
    assert data["version"] == SCENE_VERSION == 2
    again = scene_from_dict(data)
    comp = again.find("Level").get_component(MeshColliderComponent)
    assert comp.mesh == REF and comp.layer == 2
    assert comp.friction == pytest.approx(0.9)


def test_debug_lines_show_mesh_outline_and_handles_for_includes_mesh(quad) -> None:
    from engine.rendering.physics_debug import build_debug_lines, mesh_outline

    scene, go, _ = _scene()
    scene.fixed_update(DT)
    world = scene.physics.world
    handle = next(iter(world._meshes))
    assert scene.physics.handles_for([go]) == frozenset({handle})
    # плоский квадрат: только 4 граничных ребра, диагональ (общая) не рисуется
    assert len(mesh_outline(world.get_mesh(handle).bvh)) == 4
    positions, _ = build_debug_lines(world)
    assert positions.shape == (8, 3)
    only_none, _ = build_debug_lines(world, only_handles=())
    assert only_none.shape[0] == 0


def test_box_mesh_outline_has_all_twelve_edges() -> None:
    from engine.physics.bvh import Bvh
    from engine.rendering.physics_debug import mesh_outline

    v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    t = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
         [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    assert len(mesh_outline(Bvh(v, np.array(t)))) == 12
