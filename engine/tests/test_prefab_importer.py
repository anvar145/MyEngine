"""Импортёр ``*.prefab.json``: артефакт, ошибки, вложенные префабы."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.core.resources import AssetDatabase, ImporterRegistry
from engine.core.resources.importers import PrefabImporter, default_importers

PREFABS = Path("Assets") / "Prefabs"


def prefab(*objects: dict, root: str = "L0", **extra) -> dict:
    return {
        "format": "myengine.prefab",
        "version": 1,
        "name": "P",
        "root": root,
        "objects": list(objects),
        **extra,
    }


def obj(local_id: str, parent: str | None = None, components=()) -> dict:
    entry: dict = {
        "local_id": local_id,
        "name": local_id,
        "tag": "Untagged",
        "layer": 1,
        "active": True,
        "transform": {
            "position": [0, 0, 0],
            "rotation": [0, 0, 0, 1],
            "scale": [1, 1, 1],
        },
        "components": list(components),
    }
    if parent is not None:
        entry["parent"] = {"local": parent}
    return entry


def instance(guid: str, path: str) -> dict:
    return {
        "type": "PrefabInstance",
        "prefab_guid": guid,
        "prefab_path": path,
        "overrides": [],
    }


def simple() -> dict:
    return prefab(obj("L0"), obj("L1", "L0"), obj("L2", "L0"))


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / PREFABS).mkdir(parents=True)
    return root


def write(project: Path, name: str, data) -> Path:
    path = project / PREFABS / f"{name}.prefab.json"
    text = data if isinstance(data, str) else json.dumps(data, indent=2)
    path.write_text(text, encoding="utf-8")
    return path


def rel(name: str) -> str:
    return (PREFABS / f"{name}.prefab.json").as_posix()


def import_one(project: Path, path: Path):
    with AssetDatabase(project / "Assets", library=project / "Library") as db:
        db.scan()
        db.reimport_dirty(wait=True)
        guid = db.guid_of(path)
        return db, guid, db.resolve(guid)


class TestRegistry:
    def test_extension(self) -> None:
        registry = ImporterRegistry(default_importers())
        assert isinstance(registry.importer_for("a/enemy.prefab.json"), PrefabImporter)
        assert registry.importer_for("enemy.json") is None


class TestImport:
    def test_valid_prefab_makes_artifact(self, project: Path) -> None:
        path = write(project, "enemy", simple())
        with AssetDatabase(project / "Assets", library=project / "Library") as db:
            db.scan()
            db.reimport_dirty(wait=True)
            handle = db.resolve(db.guid_of(path))
            assert handle.error is None, handle.error
            artifact = handle.artifact("prefab")
            assert artifact is not None
            assert artifact.info["root"] == "L0"
            assert artifact.info["object_count"] == 3
            copy = handle.artifact_path("prefab")
            assert copy.name == "prefab.json"
            # копия без нормализации
            assert copy.read_bytes() == path.read_bytes()

    def test_inspect(self, project: Path) -> None:
        path = write(project, "enemy", simple())
        preview = PrefabImporter().inspect(path)
        assert preview.kind == "prefab"
        assert preview.info == {"root": "L0", "object_count": 3}

    @pytest.mark.parametrize(
        ("content", "text"),
        [
            ("{не json", "JSON"),
            (prefab(obj("L0"), version=2), "версия"),
            (prefab(obj("L0"), root="L9"), "root"),
            (prefab(obj("L0"), obj("L1", "L7")), "несуществующ"),
            (prefab(obj("L0", components=[{"type": "X", "t": {"local": "L4"}}])), "ссылка"),
        ],
    )
    def test_invalid_is_import_error(self, project: Path, content, text: str) -> None:
        path = write(project, "bad", content)
        _, _, handle = import_one(project, path)
        assert handle.error is not None and text in handle.error
        assert not handle.imported


class TestNested:
    def _ids(self, project: Path):
        with AssetDatabase(project / "Assets", library=project / "Library") as db:
            db.scan()
            return {
                p.name.removesuffix(".prefab.json"): db.guid_of(p)
                for p in (project / PREFABS).glob("*.prefab.json")
            }

    def test_nested_registers_dependency(self, project: Path) -> None:
        inner = write(project, "inner", simple())
        with AssetDatabase(project / "Assets", library=project / "Library") as db:
            db.scan()
            inner_guid = db.guid_of(inner)
        outer = write(
            project,
            "outer",
            prefab(obj("L0"), obj("L1", "L0", [instance(inner_guid, rel("inner"))])),
        )
        with AssetDatabase(project / "Assets", library=project / "Library") as db:
            db.scan()
            db.reimport_dirty(wait=True)
            handle = db.resolve(db.guid_of(outer))
            assert handle.error is None, handle.error
            assert inner_guid in db.dependencies(handle.guid)
            assert handle.artifact("prefab").info["nested"] == [inner_guid]

    def test_missing_nested_file(self, project: Path) -> None:
        outer = write(
            project,
            "outer",
            prefab(obj("L0", components=[instance("a" * 32, rel("ghost"))])),
        )
        _, _, handle = import_one(project, outer)
        assert handle.error is not None and "не найден" in handle.error

    def test_self_cycle(self, project: Path) -> None:
        path = write(project, "loop", simple())
        with AssetDatabase(project / "Assets", library=project / "Library") as db:
            db.scan()
            guid = db.guid_of(path)
        write(
            project,
            "loop",
            prefab(obj("L0", components=[instance(guid, rel("loop"))])),
        )
        _, _, handle = import_one(project, path)
        assert handle.error is not None and "цикл" in handle.error

    def test_cycle_a_b_a(self, project: Path) -> None:
        a = write(project, "a", simple())
        b = write(project, "b", simple())
        with AssetDatabase(project / "Assets", library=project / "Library") as db:
            db.scan()
            ga, gb = db.guid_of(a), db.guid_of(b)
        write(project, "a", prefab(obj("L0", components=[instance(gb, rel("b"))])))
        write(project, "b", prefab(obj("L0", components=[instance(ga, rel("a"))])))
        _, _, handle = import_one(project, a)
        assert handle.error is not None and "цикл" in handle.error

    def test_depth_limit(self, project: Path) -> None:
        paths = [write(project, f"p{i}", simple()) for i in range(10)]
        with AssetDatabase(project / "Assets", library=project / "Library") as db:
            db.scan()
            guids = [db.guid_of(p) for p in paths]
        # p0 -> p1 -> ... -> p9: в цепочке 10 префабов
        for i in range(9):
            write(
                project,
                f"p{i}",
                prefab(obj("L0", components=[instance(guids[i + 1], rel(f"p{i + 1}"))])),
            )
        _, _, handle = import_one(project, paths[0])
        assert handle.error is not None and "глубже" in handle.error
