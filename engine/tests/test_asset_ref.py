"""``AssetRef``: сериализация, разрешение по GUID/пути, примитивы."""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.core.reflection import FieldDescriptor, FieldKind, coerce_value, to_plain
from engine.core.resources import (
    AssetDatabase,
    AssetRef,
    LibraryVFS,
    PakBuilder,
    PakVFS,
    WorkdirVFS,
)

OBJ = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
GUID = "ab" * 16


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


def test_to_dict_from_dict_roundtrip() -> None:
    ref = AssetRef(guid=GUID, path="Models/tri.obj")
    assert ref.to_dict() == {"guid": GUID, "path": "Models/tri.obj"}
    assert AssetRef.from_dict(ref.to_dict()) == ref
    only_path = AssetRef(path="a/b.png")
    assert only_path.to_dict() == {"guid": None, "path": "a/b.png"}
    assert AssetRef.from_dict(only_path.to_dict()) == only_path


def test_requires_guid_or_path_and_empty_is_none() -> None:
    with pytest.raises(ValueError):
        AssetRef()
    with pytest.raises(ValueError):
        AssetRef(guid="", path="")
    with pytest.raises(ValueError):
        AssetRef.from_dict({})
    ref = AssetRef(guid="", path="x.png")
    assert ref.guid is None and ref.path == "x.png"
    assert AssetRef(path="a\\b.png").path == "a/b.png"


def test_reflection_serialization() -> None:
    ref = AssetRef(guid=GUID, path="m.obj")
    desc = FieldDescriptor(name="mesh_ref", kind=FieldKind.ASSET_REF, default=None)
    plain = to_plain(ref)
    assert plain == {"guid": GUID, "path": "m.obj"}
    assert coerce_value(desc, plain) == ref
    assert coerce_value(desc, None) is None
    assert coerce_value(desc, {"guid": None, "path": None}) is None


def test_resolve_by_guid_in_vfs(project) -> None:
    root, db = project
    guid = db.guid_of("Models/tri.obj")
    ref = AssetRef(guid=guid, path="moved/elsewhere.obj")
    assert ref.resolve(LibraryVFS(root / "Library")) == guid
    assert ref.resolve(db) == guid


def test_resolve_by_path_through_database(project) -> None:
    _, db = project
    guid = db.guid_of("Models/tri.obj")
    assert AssetRef(path="Models/tri.obj").resolve(db) == guid
    stale = AssetRef(guid="0" * 32, path="Models/tri.obj")
    assert stale.resolve(db) == guid
    assert AssetRef(path="Models/none.obj").resolve(db) is None
    assert AssetRef(guid="0" * 32).resolve(db) is None


def test_resolve_by_path_via_library_vfs_and_workdir(project) -> None:
    root, db = project
    guid = db.guid_of("Models/tri.obj")
    ref = AssetRef(path="Models/tri.obj")
    assert ref.resolve(LibraryVFS(root / "Library")) == guid
    assert ref.resolve(WorkdirVFS(root / "Assets")) == guid


def test_resolve_by_meta_next_to_file(project, tmp_path: Path) -> None:
    root, db = project
    guid = db.guid_of("Models/tri.obj")
    empty = tmp_path / "Empty"
    empty.mkdir()
    # База без записей: GUID не найти, но .meta лежит рядом с файлом.
    other = AssetDatabase(root / "Assets", library=empty, read_only=True)
    assert AssetRef(path="Models/tri.obj").resolve(other) == guid


def test_resolve_in_pak_by_guid(project, tmp_path: Path) -> None:
    _, db = project
    guid = db.guid_of("Models/tri.obj")
    pak = PakBuilder(db).add_all().build(tmp_path / "g.pak")
    with PakVFS(pak) as vfs:
        assert AssetRef(guid=guid, path="Models/tri.obj").resolve(vfs) == guid
        assert AssetRef(guid="0" * 32, path="x.obj").resolve(vfs) is None


def test_primitives_are_not_assets(project) -> None:
    _, db = project
    for name in ("cube", "sphere", "plane", "cylinder"):
        ref = AssetRef(guid=None, path=name)
        assert ref.is_primitive
        assert ref.resolve(db) is None
        assert AssetRef.from_dict(ref.to_dict()) == ref
    assert not AssetRef(path="Models/tri.obj").is_primitive
    # Имя примитива с GUID - это уже ассет.
    assert not AssetRef(guid=GUID, path="cube").is_primitive
