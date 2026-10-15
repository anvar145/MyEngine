"""Тесты сохранения и загрузки сцены, а также модели дерева объектов."""

from __future__ import annotations

import json

import pytest

from editor.scene_io import (
    SCENE_FORMAT,
    SCENE_VERSION,
    load_scene,
    save_scene,
    scene_from_dict,
    scene_to_dict,
)
from engine.core.components import MeshRenderer
from engine.core.math3d import Quaternion, Vector3
from engine.core.scene import Scene


class TestSceneSerialization:
    def test_round_trip_keeps_objects_and_hierarchy(self, tmp_path) -> None:
        scene = Scene("Demo")
        parent = scene.create_game_object("Parent", position=Vector3(1, 2, 3))
        parent.add_component(MeshRenderer, mesh="cube", color=Vector3(1, 0, 0))
        child = scene.create_game_object("Child", parent=parent)
        child.add_component(
            MeshRenderer, mesh="sphere", roughness=0.2, metallic=0.8, visible=False
        )
        child.transform.scale = Vector3(2, 2, 2)

        path = save_scene(scene, tmp_path / "demo.scene.json")
        restored = load_scene(path)

        assert len(restored) == 2
        assert restored.name == "Demo"
        restored_parent = restored.find("Parent")
        restored_child = restored.find("Child")
        assert restored_parent is not None and restored_child is not None
        assert restored_child.parent is restored_parent
        assert restored_child.transform.scale.almost_equals(Vector3(2, 2, 2), 1e-6)

        component = restored_parent.get_component(MeshRenderer)
        assert component is not None
        assert component.mesh == "cube"
        assert component.color.almost_equals(Vector3(1, 0, 0), 1e-6)

        hidden = restored_child.get_component(MeshRenderer)
        assert hidden is not None
        assert hidden.visible is False
        assert hidden.metallic == pytest.approx(0.8)

    def test_transform_rotation_survives(self, tmp_path) -> None:
        scene = Scene("Rotation")
        game_object = scene.create_game_object("Spinner")
        game_object.transform.rotation = Quaternion.from_euler(15, 30, 45)
        game_object.transform.position = Vector3(-1.5, 0.25, 7.0)

        restored = load_scene(save_scene(scene, tmp_path / "rot.scene.json"))
        transform = restored.find("Spinner").transform
        assert transform.position.almost_equals(Vector3(-1.5, 0.25, 7.0), 1e-5)
        assert transform.rotation.almost_equals(
            Quaternion.from_euler(15, 30, 45), 1e-4
        )

    def test_inactive_objects_are_restored(self, tmp_path) -> None:
        scene = Scene("Inactive")
        game_object = scene.create_game_object("Hidden")
        game_object.active_self = False
        restored = load_scene(save_scene(scene, tmp_path / "inactive.scene.json"))
        assert restored.find("Hidden").active_self is False

    def test_file_format_is_readable(self, tmp_path) -> None:
        scene = Scene("Format")
        scene.create_game_object("Cube")
        path = save_scene(scene, tmp_path / "format.scene.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["format"] == SCENE_FORMAT
        assert data["version"] == SCENE_VERSION
        assert data["objects"][0]["name"] == "Cube"
        assert data["objects"][0]["transform"]["scale"] == [1.0, 1.0, 1.0]

    def test_unknown_format_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            scene_from_dict({"format": "something.else", "objects": []})

    def test_newer_version_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            scene_from_dict(
                {"format": SCENE_FORMAT, "version": SCENE_VERSION + 5, "objects": []}
            )

    def test_broken_json_reports_error(self, tmp_path) -> None:
        path = tmp_path / "broken.scene.json"
        path.write_text("{ not json", encoding="utf-8")
        with pytest.raises(ValueError):
            load_scene(path)

    def test_scene_to_dict_lists_components(self) -> None:
        scene = Scene("Dict")
        game_object = scene.create_game_object("Cube")
        game_object.add_component(MeshRenderer, mesh="sphere")
        payload = scene_to_dict(scene)
        component = payload["objects"][0]["components"][0]
        assert component["type"] == "MeshRenderer"
        assert component["mesh_ref"] == {"guid": None, "path": "sphere"}


class TestSceneTreeModel:
    def test_model_mirrors_hierarchy(self, qapp) -> None:
        from editor.models import SceneTreeModel

        scene = Scene("Tree")
        parent = scene.create_game_object("Parent")
        child = scene.create_game_object("Child", parent=parent)
        scene.create_game_object("Other")

        model = SceneTreeModel(scene)
        assert model.rowCount() == 2  # два корня
        parent_index = model.index_for(parent)
        assert parent_index.isValid()
        assert model.data(parent_index) == "Parent"
        assert model.rowCount(parent_index) == 1
        child_index = model.index(0, 0, parent_index)
        assert model.data(child_index) == "Child"
        assert model.parent(child_index) == parent_index
        assert model.object_for(child_index) is child

    def test_model_refresh_after_structure_change(self, qapp) -> None:
        from editor.models import SceneTreeModel

        scene = Scene("Tree")
        model = SceneTreeModel(scene)
        assert model.rowCount() == 0
        scene.create_game_object("New")
        model.refresh()
        assert model.rowCount() == 1

    def test_model_switches_scene(self, qapp) -> None:
        from editor.models import SceneTreeModel

        first = Scene("First")
        first.create_game_object("A")
        second = Scene("Second")
        second.create_game_object("B")
        model = SceneTreeModel(first)
        model.set_scene(second)
        assert model.scene is second
        assert model.data(model.index(0, 0)) == "B"

    def test_invalid_indices_are_safe(self, qapp) -> None:
        from editor.models import SceneTreeModel

        model = SceneTreeModel(Scene("Empty"))
        assert model.data(model.index(5, 0)) is None
        assert not model.index(0, 0).isValid()
        assert model.rowCount() == 0

    def test_index_for_foreign_object_is_invalid(self, qapp) -> None:
        from editor.models import SceneTreeModel

        scene = Scene("Mine")
        other = Scene("Other")
        foreign = other.create_game_object("Foreign")
        assert not SceneTreeModel(scene).index_for(foreign).isValid()
