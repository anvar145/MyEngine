"""Тесты снимка сцены Play Mode."""

from __future__ import annotations

from engine.core.components import MeshRenderer
from engine.core.math3d import Vector3
from engine.core.scene import Scene
from engine.player.snapshot import SceneSnapshot
from engine.rendering.sky import SkySettings


def _scene() -> Scene:
    scene = Scene("Level")
    parent = scene.create_game_object("Parent")
    parent.transform.position = Vector3(1.0, 2.0, 3.0)
    child = scene.create_game_object("Child")
    child.tag = "Pickup"
    child.add_component(MeshRenderer())
    child.set_parent(parent, keep_world=False)
    return scene


def test_scene_snapshot_roundtrip() -> None:
    scene = _scene()
    sky = SkySettings(sun_size=3.0)
    first = SceneSnapshot.freeze(scene, sky=sky)
    # Меняем сцену и восстанавливаем в неё же.
    scene.create_game_object("Extra")
    first.restore(scene)
    second = SceneSnapshot.freeze(scene, sky=sky)
    assert first.to_json() == second.to_json()
    assert SceneSnapshot.from_json(first.to_json()).to_json() == first.to_json()
    assert second.scene_name == "Level"
    assert len(second.objects) == 2
