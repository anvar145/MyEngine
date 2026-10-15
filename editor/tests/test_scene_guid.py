"""GUID мешей в сценах: запись, обратная совместимость, переименование файла."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from editor import scene_io
from engine.core.components import MeshRenderer
from engine.core.event_bus import EventBus
from engine.core.resources import AssetDatabase, ImportManager
from engine.core.scene import Scene

OBJ = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"


@pytest.fixture
def db(tmp_path: Path) -> AssetDatabase:
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "tri.obj").write_text(OBJ, encoding="utf-8")
    database = AssetDatabase(assets, library=tmp_path / "Library", event_bus=EventBus())
    database.scan()
    return database


def _scene_with(mesh: str) -> Scene:
    scene = Scene("S", event_bus=EventBus())
    scene.create_game_object("A").add_component(MeshRenderer, mesh=mesh)
    return scene


def _mesh_of(scene: Scene) -> MeshRenderer:
    return scene.objects[0].get_component(MeshRenderer)


def test_guid_is_written_for_registered_mesh(db: AssetDatabase, tmp_path: Path) -> None:
    path = scene_io.save_scene(_scene_with("tri.obj"), tmp_path / "s.scene.json", db)
    entry = json.loads(path.read_text(encoding="utf-8"))["objects"][0]["components"][0]
    assert entry["mesh_ref"] == {"guid": db.guid_of("tri.obj"), "path": "tri.obj"}


def test_no_guid_without_database_and_for_builtin(
    db: AssetDatabase, tmp_path: Path
) -> None:
    plain = scene_io.scene_to_dict(_scene_with("tri.obj"))
    assert plain["objects"][0]["components"][0]["mesh_ref"]["guid"] is None
    builtin = scene_io.scene_to_dict(_scene_with("cube"), db)
    assert builtin["objects"][0]["components"][0]["mesh_ref"]["guid"] is None


def test_old_scene_without_guid_loads(db: AssetDatabase, tmp_path: Path) -> None:
    data = scene_io.scene_to_dict(_scene_with("tri.obj"))
    restored = scene_io.scene_from_dict(data, db)
    assert _mesh_of(restored).mesh == "tri.obj"
    # GUID дописывается из базы при загрузке
    assert _mesh_of(restored).mesh_guid == db.guid_of("tri.obj")


def test_renamed_file_is_found_by_guid(db: AssetDatabase, tmp_path: Path) -> None:
    data = scene_io.scene_to_dict(_scene_with("tri.obj"), db)
    guid = data["objects"][0]["components"][0]["mesh_ref"]["guid"]
    manager = ImportManager(db)
    manager.open()
    (db.root / "tri.obj").rename(db.root / "renamed.obj")
    (db.root / "tri.obj.meta").rename(db.root / "renamed.obj.meta")
    manager.poll(force=True)
    manager.close()
    assert db.guid_of("renamed.obj") == guid

    restored = scene_io.scene_from_dict(data, db)
    assert _mesh_of(restored).mesh == "renamed.obj"
    assert _mesh_of(restored).mesh_guid == guid


def test_unknown_guid_keeps_path(db: AssetDatabase) -> None:
    data = scene_io.scene_to_dict(_scene_with("tri.obj"), db)
    data["objects"][0]["components"][0]["mesh_guid"] = "0" * 32
    restored = scene_io.scene_from_dict(data, db)
    assert _mesh_of(restored).mesh == "tri.obj"
