"""VFS: LibraryVFS, PakVFS, OverlayVFS, WorkdirVFS, автодетект и совместимость load_*."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from engine.core.resources import (
    VFS,
    AssetDatabase,
    LibraryVFS,
    NoBuildError,
    OverlayVFS,
    PakBuilder,
    PakVFS,
    VfsAssets,
    WorkdirVFS,
    as_assets,
    detect_vfs,
    load_material,
    load_sprite_atlas,
)
from engine.core.resources.cookers import encode_png, read_cooked_mesh

OBJ = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"


@pytest.fixture()
def project(tmp_path: Path):
    """Проект: ``Assets/`` (obj, png, mat, atlas) + импортированный ``Library/``."""
    root = tmp_path / "Proj"
    assets = root / "Assets"
    assets.mkdir(parents=True)
    (assets / "tri.obj").write_text(OBJ, encoding="utf-8")
    (assets / "red.png").write_bytes(
        encode_png(np.full((4, 4, 3), (200, 40, 40), np.uint8))
    )
    (assets / "paint.mat.json").write_text(
        json.dumps({"name": "Paint", "textures": {"albedo": "red.png"}}),
        encoding="utf-8",
    )
    grid = np.zeros((8, 16, 4), dtype=np.uint8)
    grid[:, :8] = (10, 20, 30, 255)
    (assets / "grid.png").write_bytes(encode_png(grid))
    (assets / "grid.atlas.json").write_text(
        json.dumps({"mode": "grid", "image": "grid.png", "cell": [8, 8], "prefix": "t"}),
        encoding="utf-8",
    )
    with AssetDatabase(assets, library=root / "Library") as db:
        db.scan()
        db.reimport_dirty(wait=True)
        yield root, db


def test_library_vfs_reads_artifact(project) -> None:
    root, db = project
    guid = db.guid_of("tri.obj")
    vfs = LibraryVFS(root / "Library")
    assert isinstance(vfs, VFS)
    assert vfs.exists(guid, "mesh") and not vfs.exists(guid, "texture")
    assert "mesh" in vfs.list_artifacts(guid)
    assert set(vfs.list_guids()) == set(db.guids)
    expected = db.resolve(guid).artifact_path("mesh").read_bytes()
    assert vfs.read(guid, "mesh") == expected
    assert vfs.resolve(guid, "mesh").read_bytes() == expected
    assert vfs.read("0" * 32, "mesh") is None and vfs.resolve("bad", "mesh") is None
    assert len(read_cooked_mesh(vfs, guid).lods) >= 1


def test_library_record_keeps_source_path(project) -> None:
    root, db = project
    handle = LibraryVFS(root / "Library").handle(db.guid_of("tri.obj"))
    assert handle.path.as_posix() == "tri.obj"


def test_pak_vfs_equals_library(project, tmp_path: Path) -> None:
    root, db = project
    pak_path = PakBuilder(db).add_all().build(tmp_path / "game.pak")
    library = LibraryVFS(root / "Library")
    with PakVFS(pak_path) as pak:
        assert set(pak.list_guids()) == set(library.list_guids())
        for guid in library.list_guids():
            assert pak.list_artifacts(guid) == library.list_artifacts(guid)
            for kind in library.list_artifacts(guid):
                assert pak.read(guid, kind) == library.read(guid, kind)
                assert pak.exists(guid, kind)
                assert pak.resolve(guid, kind).read_bytes() == library.read(guid, kind)
        assert pak.read(db.guid_of("tri.obj"), "texture") is None


def test_overlay_patch_overrides_base(project, tmp_path: Path) -> None:
    root, db = project
    guid = db.guid_of("tri.obj")
    base = LibraryVFS(root / "Library")
    patch_dir = tmp_path / "Patch"
    shard = patch_dir / guid[:2] / guid
    shard.mkdir(parents=True)
    (shard / "mesh.bin").write_bytes(b"PATCHED")
    (shard / "artifacts.json").write_text(
        json.dumps(
            {
                "path": "tri.obj",
                "importer": "obj",
                "version": 1,
                "artifacts": [{"kind": "mesh", "name": "mesh.bin", "info": {}}],
            }
        ),
        encoding="utf-8",
    )
    overlay = OverlayVFS(base, LibraryVFS(patch_dir))
    assert overlay.read(guid, "mesh") == b"PATCHED"
    other = db.guid_of("red.png")
    assert overlay.read(other, "texture") == base.read(other, "texture")
    assert set(overlay.list_guids()) == set(base.list_guids())


def test_load_material_by_vfs_equals_database(project, tmp_path: Path) -> None:
    root, db = project
    guid = db.guid_of("paint.mat.json")
    expected = load_material(db, guid)
    assert expected.texture_guids  # albedo -> GUID red.png
    pak = PakBuilder(db).add_all().build(tmp_path / "m.pak")
    with PakVFS(pak) as pak_vfs:
        sources = [LibraryVFS(root / "Library"), pak_vfs, WorkdirVFS(root / "Assets")]
        for vfs in sources:
            got = load_material(vfs, guid)
            assert got.params == expected.params
            assert got.texture_guids == expected.texture_guids, type(vfs).__name__
            assert got.missing == expected.missing


def test_load_sprite_atlas_by_vfs(project) -> None:
    root, db = project
    guid = db.guid_of("grid.atlas.json")
    expected = load_sprite_atlas(db, guid)
    got = load_sprite_atlas(LibraryVFS(root / "Library"), guid)
    assert got.frames == expected.frames


def test_vfs_assets_guid_of_relative_and_absolute(project) -> None:
    root, db = project
    assets = as_assets(LibraryVFS(root / "Library"), root / "Assets")
    assert isinstance(assets, VfsAssets)
    guid = db.guid_of("red.png")
    assert assets.guid_of("red.png") == guid
    assert assets.guid_of(root / "Assets" / "red.png") == guid
    assert assets.guid_of("nope.png") is None
    assert as_assets(db) is db


def test_detect_prefers_pak_then_library(project, tmp_path: Path) -> None:
    root, db = project
    vfs = detect_vfs(root)
    assert isinstance(vfs, WorkdirVFS)
    pak = PakBuilder(db).add_all().build(root / "Build" / f"{root.name}.pak")
    auto = detect_vfs(root)
    assert isinstance(auto, PakVFS)
    auto.close()
    assert isinstance(detect_vfs(root, use_library=True), WorkdirVFS)
    forced = detect_vfs(root, pak=pak)
    assert isinstance(forced, PakVFS)
    forced.close()
    with pytest.raises(FileNotFoundError):
        detect_vfs(root, pak=root / "none.pak")


def test_detect_library_only_and_error(tmp_path: Path) -> None:
    only_lib = tmp_path / "Lib"
    (only_lib / "Library").mkdir(parents=True)
    assert isinstance(detect_vfs(only_lib), LibraryVFS)
    empty = tmp_path / "Empty"
    empty.mkdir()
    with pytest.raises(NoBuildError, match="нет собранного пака"):
        detect_vfs(empty)
