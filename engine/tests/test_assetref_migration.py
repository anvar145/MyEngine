"""Миграция ссылок на ассеты в ``AssetRef``: старые формы читаются, пишется новая."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from engine.core.components import MeshRenderer, SpriteRenderer
from engine.core.resources import AssetDatabase, AssetRef
from engine.scene_io import load_scene, save_scene, scene_from_dict, scene_to_dict

OBJ = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
PROJECTS = Path(__file__).resolve().parents[2] / "projects" / "PlayModeTest"


@pytest.fixture()
def project(tmp_path: Path):
    root = tmp_path / "Proj"
    assets = root / "Assets"
    (assets / "Models").mkdir(parents=True)
    (assets / "Models" / "tri.obj").write_text(OBJ, encoding="utf-8")
    with AssetDatabase(assets, library=root / "Library") as db:
        db.scan()
        db.reimport_dirty(wait=True)
        yield root, db


def _old_scene(guid: str) -> dict:
    transform = {"position": [0, 0, 0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]}
    return {
        "format": "myengine.scene",
        "version": 2,
        "name": "old",
        "objects": [
            {
                "uid": "a" * 16,
                "name": "Mesh",
                "transform": transform,
                "components": [
                    {"type": "MeshRenderer", "mesh": "Models/tri.obj", "mesh_guid": guid},
                ],
            },
            {
                "uid": "b" * 16,
                "name": "Cube",
                "transform": transform,
                "components": [{"type": "MeshRenderer", "mesh": "cube"}],
            },
            {
                "uid": "c" * 16,
                "name": "Sprite",
                "transform": transform,
                "components": [
                    {"type": "SpriteRenderer", "sprite": "f0", "atlas_guid": "d" * 32},
                ],
            },
        ],
    }


def test_old_v2_scene_fills_mesh_ref(project) -> None:
    _, db = project
    guid = db.guid_of("Models/tri.obj")
    scene = scene_from_dict(_old_scene(guid), db)
    mesh = scene.objects[0].get_component(MeshRenderer)
    assert mesh.mesh_ref == AssetRef(guid=guid, path="Models/tri.obj")
    assert mesh.mesh_guid == guid and mesh.mesh == "Models/tri.obj"
    cube = scene.objects[1].get_component(MeshRenderer)
    assert cube.mesh_ref == AssetRef(path="cube") and cube.is_builtin
    sprite = scene.objects[2].get_component(SpriteRenderer)
    assert sprite.sprite == "f0" and sprite.atlas_guid == "d" * 32


def test_old_mesh_follows_guid_after_rename(project) -> None:
    _, db = project
    guid = db.guid_of("Models/tri.obj")
    data = _old_scene(guid)
    data["objects"][0]["components"][0]["mesh"] = "Models/old_name.obj"
    mesh = scene_from_dict(data, db).objects[0].get_component(MeshRenderer)
    assert mesh.mesh_ref.path == "Models/tri.obj"


def test_save_writes_only_new_form(project) -> None:
    _, db = project
    scene = scene_from_dict(_old_scene(db.guid_of("Models/tri.obj")), db)
    dumped = json.dumps(scene_to_dict(scene, db))
    assert '"mesh_ref"' in dumped and '"sprite_ref"' in dumped
    for old in ('"mesh_guid"', '"atlas_guid"', '"mesh":'):
        assert old not in dumped


def test_save_load_save_is_stable(project) -> None:
    _, db = project
    scene = scene_from_dict(_old_scene(db.guid_of("Models/tri.obj")), db)
    first = scene_to_dict(scene, db)
    second = scene_to_dict(scene_from_dict(json.loads(json.dumps(first)), db), db)
    assert first == second


def test_load_does_not_rewrite_file(project, tmp_path: Path) -> None:
    _, db = project
    path = tmp_path / "old.scene.json"
    path.write_text(json.dumps(_old_scene(db.guid_of("Models/tri.obj"))), "utf-8")
    before = path.read_bytes()
    load_scene(path, db)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.bak"))


def test_save_scene_roundtrip_file(project, tmp_path: Path) -> None:
    _, db = project
    scene = scene_from_dict(_old_scene(db.guid_of("Models/tri.obj")), db)
    path = tmp_path / "new.scene.json"
    save_scene(scene, path, db)
    first = path.read_bytes()
    save_scene(load_scene(path, db), path, db)
    assert path.read_bytes() == first


@pytest.mark.skipif(not PROJECTS.is_dir(), reason="нет PlayModeTest")
@pytest.mark.parametrize(
    "name", ["main", "main_menu", "physics_demo", "realistic"]
)
def test_playmode_scenes_load_and_resave(name: str, tmp_path: Path) -> None:
    scene_path = PROJECTS / "Assets" / "Scenes" / f"{name}.scene.json"
    digest = hashlib.sha256(scene_path.read_bytes()).hexdigest()
    with AssetDatabase(PROJECTS / "Assets", library=PROJECTS / "Library", read_only=True) as db:
        scene = load_scene(scene_path, db)
        assert scene.objects
        out = tmp_path / "out.scene.json"
        save_scene(scene, out, db)
        again = tmp_path / "again.scene.json"
        save_scene(load_scene(out, db), again, db)
        assert out.read_bytes() == again.read_bytes()
    assert hashlib.sha256(scene_path.read_bytes()).hexdigest() == digest


def test_five_kinds_of_refs_roundtrip(tmp_path: Path) -> None:
    """Меш, текстура, префаб, материал, аудио: save -> load -> save идентично."""
    import struct
    import wave

    import numpy as np

    from engine.core.resources.cookers import encode_png

    root = tmp_path / "P"
    assets = root / "Assets"
    (assets / "Scripts").mkdir(parents=True)
    (assets / "Scripts" / "s.py").write_text(
        "mesh: Asset = None\ntex: Asset = None\npre: Prefab = None\n"
        "mat: Asset = None\nsnd: Asset = None\n",
        "utf-8",
    )
    (assets / "m.obj").write_text(OBJ, "utf-8")
    (assets / "t.png").write_bytes(encode_png(np.full((4, 4, 3), 90, np.uint8)))
    (assets / "b.prefab.json").write_text(
        json.dumps({"format": "myengine.prefab", "version": 1, "name": "b",
                    "objects": []}), "utf-8")
    (assets / "k.mat.json").write_text(json.dumps({"name": "K"}), "utf-8")
    with wave.open(str(assets / "a.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(struct.pack("<8h", *range(8)))
    with AssetDatabase(assets, library=root / "Library") as db:
        db.scan()
        refs = {
            "mesh": "m.obj", "tex": "t.png", "pre": "b.prefab.json",
            "mat": "k.mat.json", "snd": "a.wav",
        }
        transform = {"position": [0, 0, 0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]}
        data = {
            "format": "myengine.scene", "version": 2, "name": "r",
            "objects": [{
                "uid": "e" * 16, "name": "O", "transform": transform,
                "components": [{
                    "type": "ScriptComponent", "path": "Assets/Scripts/s.py",
                    "auto_start": False,
                    "fields": {k: {"guid": None, "path": v} for k, v in refs.items()},
                }],
            }],
        }
        first = scene_to_dict(scene_from_dict(data, db), db)
        stored = first["objects"][0]["components"][0]["fields"]
        for name, path in refs.items():
            assert stored[name]["path"] == path
            assert stored[name]["guid"] == db.guid_of(path)
        second = scene_to_dict(scene_from_dict(json.loads(json.dumps(first)), db), db)
        assert first == second
