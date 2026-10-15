"""Сериализация физических компонентов в .scene.json."""

from __future__ import annotations

import json

from engine.core.math3d import Vector3
from engine.core.physics_components import (
    CharacterControllerComponent,
    ColliderComponent,
    RigidBodyComponent,
)
from engine.core.scene import Scene
from engine.scene_io import scene_from_dict, scene_to_dict


def test_round_trip_physics_components():
    scene = Scene()
    box = scene.create_game_object("Box", position=Vector3(1, 2, 3))
    box.add_component(ColliderComponent, shape="box", size=(1, 2, 3))
    box.add_component(RigidBodyComponent, mass=4.0)
    hero = scene.create_game_object("Hero")
    hero.add_component(CharacterControllerComponent)
    data = json.loads(json.dumps(scene_to_dict(scene)))
    loaded = scene_from_dict(data)
    objects = {go.name: go for go in loaded}
    collider = objects["Box"].get_component(ColliderComponent)
    body = objects["Box"].get_component(RigidBodyComponent)
    assert collider is not None and body is not None
    assert tuple(collider.size) == (1, 2, 3)
    assert body.mass == 4.0
    assert objects["Hero"].get_component(CharacterControllerComponent) is not None


def test_old_scene_without_physics_loads():
    scene = Scene()
    scene.create_game_object("A")
    data = scene_to_dict(scene)
    assert scene_from_dict(data).find("A") is not None


def test_unknown_component_type_is_skipped():
    scene = Scene()
    scene.create_game_object("A")
    data = scene_to_dict(scene)
    data["objects"][0]["components"].append({"type": "FutureThing"})
    assert scene_from_dict(data).find("A") is not None
