"""Тесты пакета ассетов: сборка, чтение, проверка целостности, совместимость с load_*."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from engine.core.resources import (
    AssetDatabase,
    ResourceFormatError,
    ResourceNotFoundError,
    load_sprite_atlas,
)
from engine.core.resources.cookers import encode_png
from engine.core.resources.pak import PAK_VERSION, PakBuilder, PakReader

OBJ = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"


@pytest.fixture()
def db(tmp_path: Path):
    root = tmp_path / "assets"
    root.mkdir()
    (root / "tri.obj").write_text(OBJ, encoding="utf-8")
    grid = np.zeros((8, 16, 4), dtype=np.uint8)
    grid[:, :8] = (10, 20, 30, 255)
    (root / "grid.png").write_bytes(encode_png(grid))
    (root / "grid.atlas.json").write_text(
        json.dumps(
            {"mode": "grid", "image": "grid.png", "cell": [8, 8], "prefix": "t"}
        ),
        encoding="utf-8",
    )
    with AssetDatabase(root, library=tmp_path / "Library") as database:
        database.scan()
        database.reimport_dirty(wait=True)
        yield database


def test_roundtrip_artifacts(db: AssetDatabase, tmp_path: Path) -> None:
    guid = db.guid_of("tri.obj")
    builder = PakBuilder(db).add_all()
    assert len(builder) == 3
    pak_path = builder.build(tmp_path / "out" / "game.pak")
    assert pak_path.is_file() and not pak_path.with_name("game.pak.tmp").exists()

    handle = db.resolve(guid)
    with PakReader(pak_path) as pak:
        assert set(pak.guids()) == set(db.guids)
        assert guid in pak and len(pak) == 3
        assert pak.guid_of("tri.obj") == guid
        for artifact in handle.artifacts:
            expected = handle.artifact_path(artifact).read_bytes()
            assert pak.read(guid, artifact.name) == expected


def test_pak_acts_as_database_source(db: AssetDatabase, tmp_path: Path) -> None:
    guid = db.guid_of("grid.atlas.json")
    pak_path = PakBuilder(db).add(guid).build(tmp_path / "a.pak", compress=False)
    original = load_sprite_atlas(db, guid)
    with PakReader(pak_path) as pak:
        restored = load_sprite_atlas(pak, guid)
        assert restored.frames == original.frames
        assert restored.texture().width == original.texture().width
        assert pak.resolve(guid).imported
        cache = pak.cache_dir
    assert not cache.exists()  # временный кэш удалён


def test_add_pulls_dependencies(db: AssetDatabase, tmp_path: Path) -> None:
    guid = db.guid_of("grid.atlas.json")
    builder = PakBuilder(db).add(guid)
    assert len(builder) >= 1
    only = PakBuilder(db).add(guid, with_dependencies=False)
    assert len(only) == 1 <= len(builder)


def test_missing_and_unimported(db: AssetDatabase, tmp_path: Path) -> None:
    pak_path = PakBuilder(db).add_all().build(tmp_path / "p.pak")
    with PakReader(pak_path) as pak:
        with pytest.raises(ResourceNotFoundError):
            pak.resolve("0" * 32)
        assert pak.get("0" * 32) is None
        with pytest.raises(ResourceNotFoundError):
            pak.read(db.guid_of("tri.obj"), "нет.bin")
    with pytest.raises(ResourceNotFoundError):
        PakBuilder(db).add("1" * 32)


def test_corruption_detected(db: AssetDatabase, tmp_path: Path) -> None:
    guid = db.guid_of("tri.obj")
    pak_path = PakBuilder(db).add(guid).build(tmp_path / "c.pak", compress=False)
    with PakReader(pak_path) as pak:
        name = db.resolve(guid).artifacts[0].name
        offset = pak._entry(guid, name)["offset"]  # noqa: SLF001
    raw = bytearray(pak_path.read_bytes())
    raw[offset] ^= 0xFF
    pak_path.write_bytes(bytes(raw))
    with PakReader(pak_path) as pak, pytest.raises(ResourceFormatError, match="сумма"):
        pak.read(guid, name)


def test_bad_files(tmp_path: Path) -> None:
    short = tmp_path / "short.pak"
    short.write_bytes(b"MEPK")
    with pytest.raises(ResourceFormatError):
        PakReader(short)
    foreign = tmp_path / "foreign.pak"
    foreign.write_bytes(b"XXXX" + b"\0" * 20)
    with pytest.raises(ResourceFormatError, match="не является"):
        PakReader(foreign)
    future = tmp_path / "future.pak"
    future.write_bytes(struct.pack("<4sIQ", b"MEPK", PAK_VERSION + 1, 16))
    with pytest.raises(ResourceFormatError, match="Версия"):
        PakReader(future)
    unfinished = tmp_path / "unfinished.pak"
    unfinished.write_bytes(struct.pack("<4sIQ", b"MEPK", PAK_VERSION, 0))
    with pytest.raises(ResourceFormatError, match="не дописан"):
        PakReader(unfinished)


def test_files_section_roundtrip(db: AssetDatabase, tmp_path: Path) -> None:
    big = b"scene " * 5000
    builder = PakBuilder(db).add_all()
    builder.add_file(r"Assets\Scenes\main.scene.json", big)
    builder.add_file("./game.project.json", b"{}")
    assert builder.files == ("Assets/Scenes/main.scene.json", "game.project.json")
    pak_path = builder.build(tmp_path / "files.pak")
    with PakReader(pak_path) as pak:
        assert pak.read_file("Assets/Scenes/main.scene.json") == big
        assert pak.read_file("game.project.json") == b"{}"
        assert pak.read_file("nope.txt") is None
        assert pak.has_file("game.project.json")
        assert pak.list_files("Assets/") == ["Assets/Scenes/main.scene.json"]
        assert len(pak.list_files()) == 2


def test_files_crc_detected(db: AssetDatabase, tmp_path: Path) -> None:
    pak_path = PakBuilder(db).add_file("a.txt", b"abcdefgh" * 4).build(
        tmp_path / "crc.pak"
    )
    with PakReader(pak_path) as pak:
        entry = pak._files["a.txt"]  # noqa: SLF001
        offset = int(entry["offset"])
    raw = bytearray(pak_path.read_bytes())
    raw[offset] ^= 0xFF
    pak_path.write_bytes(bytes(raw))
    with PakReader(pak_path) as pak, pytest.raises(ResourceFormatError):
        pak.read_file("a.txt")


def test_add_file_rejects_bad_path(db: AssetDatabase) -> None:
    builder = PakBuilder(db)
    with pytest.raises(ValueError):
        builder.add_file("../x", b"1")
    with pytest.raises(ValueError):
        builder.add_file("", b"1")


def test_v1_pak_is_readable(db: AssetDatabase, tmp_path: Path) -> None:
    pak_path = PakBuilder(db).add_all().build(tmp_path / "old.pak")
    raw = bytearray(pak_path.read_bytes())
    raw[4:8] = struct.pack("<I", 1)
    pak_path.write_bytes(bytes(raw))
    with PakReader(pak_path) as pak:
        assert len(pak) == 3
        assert pak.read_file("any") is None
        assert pak.list_files() == []


def test_pak_vfs_files(db: AssetDatabase, tmp_path: Path) -> None:
    from engine.core.resources.vfs import OverlayVFS, PakVFS

    pak_path = PakBuilder(db).add_file("Assets/Scripts/a.py", b"x = 1").build(
        tmp_path / "vfs.pak"
    )
    other = PakBuilder(db).add_file("Assets/Scripts/b.py", b"y = 2").build(
        tmp_path / "vfs2.pak"
    )
    with PakVFS(pak_path) as base, PakVFS(other) as patch:
        assert base.read_file("Assets/Scripts/a.py") == b"x = 1"
        assert base.list_files("Assets/Scripts/") == ["Assets/Scripts/a.py"]
        merged = OverlayVFS(base, patch)
        assert merged.read_file("Assets/Scripts/b.py") == b"y = 2"
        assert merged.list_files("Assets/") == [
            "Assets/Scripts/a.py",
            "Assets/Scripts/b.py",
        ]


def test_workdir_vfs_files(db: AssetDatabase, tmp_path: Path) -> None:
    from engine.core.resources.vfs import WorkdirVFS

    (tmp_path / "Settings").mkdir()
    (tmp_path / "Settings" / "input.json").write_text("{}", encoding="utf-8")
    vfs = WorkdirVFS(db.root, database=db)
    assert vfs.read_file("Settings/input.json") == b"{}"
    assert vfs.read_file("Settings/none.json") is None
    assert vfs.list_files("Settings/") == ["Settings/input.json"]
    assert "Settings/input.json" in vfs.list_files("Sett")
