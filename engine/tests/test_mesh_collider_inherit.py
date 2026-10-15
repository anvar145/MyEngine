"""MeshCollider берёт меш у MeshRenderer при создании (B2)."""

from __future__ import annotations

import logging

from engine.core.components import MeshRenderer
from engine.core.math3d import Vector3
from engine.core.physics_adapter import ScenePhysicsAdapter
from engine.core.physics_components import MeshColliderComponent
from engine.core.resources.asset_ref import AssetRef
from engine.core.scene import Scene

REF = AssetRef("a" * 32, "Models/level.glb")


def test_attach_copies_mesh_ref_from_renderer() -> None:
    scene = Scene("t")
    go = scene.create_game_object("Level")
    go.add_component(MeshRenderer(mesh_ref=REF))
    collider = go.add_component(MeshColliderComponent)
    assert collider.mesh == REF


def test_collider_does_not_follow_renderer_later() -> None:
    scene = Scene("t")
    go = scene.create_game_object("Level")
    renderer = go.add_component(MeshRenderer(mesh_ref=REF))
    collider = go.add_component(MeshColliderComponent)
    renderer.mesh_ref = None
    assert collider.mesh == REF


def test_no_renderer_ref_leaves_mesh_empty() -> None:
    scene = Scene("t")
    go = scene.create_game_object("Level")
    go.add_component(MeshRenderer("cube"))
    collider = go.add_component(MeshColliderComponent)
    assert collider.mesh is None
    assert collider.inheritable_mesh() is None


def test_null_mesh_logs_error_once(caplog) -> None:
    scene = Scene("t")
    go = scene.create_game_object("Level", position=Vector3(0, 0, 0))
    go.add_component(MeshColliderComponent)
    adapter = ScenePhysicsAdapter(scene)
    with caplog.at_level(logging.ERROR):
        for _ in range(3):
            adapter.step(1 / 60)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
