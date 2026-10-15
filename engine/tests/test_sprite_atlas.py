"""Тесты спрайт-атласов: нарезка, упаковка, импорт и загрузка."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from engine.core.resources import (
    AssetDatabase,
    AssetImportError,
    ResourceNotFoundError,
    SpriteAtlas,
    load_sprite_atlas,
)
from engine.core.resources.cookers import (
    Sprite,
    encode_png,
    pack_sprites,
    parse_atlas_spec,
    slice_alpha,
    slice_grid,
)
from engine.core.resources.importers import SpriteAtlasImporter, default_importers


def solid(width: int, height: int, color=(255, 0, 0, 255)) -> np.ndarray:
    return np.tile(np.array(color, dtype=np.uint8), (height, width, 1))


def sheet_with_blobs() -> np.ndarray:
    """Лист 16x12 с тремя спрайтами, один из них с диагональным соединением."""
    sheet = np.zeros((12, 16, 4), dtype=np.uint8)
    sheet[1:4, 1:4] = (255, 0, 0, 255)  # красный 3x3
    sheet[2:5, 8:14] = (0, 255, 0, 255)  # зелёный 6x3
    sheet[8:10, 2:4] = (0, 0, 255, 255)  # синий 2x2
    sheet[10, 4] = (0, 0, 255, 255)  # диагональный сосед синего
    sheet[0, 15] = (255, 255, 255, 255)  # шум 1x1
    return sheet


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "sprites").mkdir(parents=True)
    (root / "sheet.png").write_bytes(encode_png(sheet_with_blobs()))
    grid = np.zeros((8, 16, 4), dtype=np.uint8)
    grid[0:8, 0:8] = (10, 20, 30, 255)
    grid[0:8, 8:16] = (0, 0, 0, 0)  # пустая ячейка
    (root / "grid.png").write_bytes(encode_png(grid))
    for name, (w, h, color) in {
        "a": (5, 7, (255, 0, 0, 255)),
        "b": (9, 4, (0, 255, 0, 255)),
        "c": (3, 3, (0, 0, 255, 255)),
    }.items():
        (root / "sprites" / f"{name}.png").write_bytes(
            encode_png(solid(w, h, color))
        )
    return root


def write_spec(root: Path, name: str, spec: dict) -> Path:
    path = root / name
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


class TestSpec:
    def test_modes_validated(self) -> None:
        with pytest.raises(AssetImportError, match="mode"):
            parse_atlas_spec('{"mode": "hex"}')
        with pytest.raises(AssetImportError, match="image"):
            parse_atlas_spec('{"mode": "alpha"}')
        with pytest.raises(AssetImportError, match="cell"):
            parse_atlas_spec('{"mode": "grid", "image": "a.png"}')
        with pytest.raises(AssetImportError, match="sources"):
            parse_atlas_spec('{"mode": "pack"}')
        with pytest.raises(AssetImportError, match="JSON"):
            parse_atlas_spec("{")

    def test_defaults(self) -> None:
        spec = parse_atlas_spec('{"mode": "pack", "folder": "s"}')
        assert spec.padding == 2 and spec.max_size == 2048 and spec.power_of_two

    def test_duplicate_names(self) -> None:
        with pytest.raises(AssetImportError, match="уникальны"):
            parse_atlas_spec(
                '{"mode": "alpha", "image": "a.png", "names": ["x", "x"]}'
            )


class TestSlice:
    def test_grid_skips_empty_and_names_by_index(self) -> None:
        sheet = np.zeros((8, 24, 4), dtype=np.uint8)
        sheet[:, 0:8, 3] = 255
        sheet[:, 16:24, 3] = 255
        sprites = slice_grid(sheet, (8, 8), prefix="walk")
        assert [s.name for s in sprites] == ["walk_00", "walk_02"]
        assert [s.origin for s in sprites] == [(0, 0), (16, 0)]
        assert all(s.pixels.shape == (8, 8, 4) for s in sprites)

    def test_grid_margin_and_spacing(self) -> None:
        sheet = np.full((13, 13, 4), 255, dtype=np.uint8)
        sprites = slice_grid(sheet, (4, 4), margin=1, spacing=1)
        assert len(sprites) == 4 and sprites[3].origin == (6, 6)

    def test_alpha_regions(self) -> None:
        sprites = slice_alpha(sheet_with_blobs(), min_size=2)
        assert [s.origin for s in sprites] == [(1, 1), (8, 2), (2, 8)]
        assert [s.pixels.shape[:2] for s in sprites] == [(3, 3), (3, 6), (3, 3)]

    def test_alpha_diagonal_is_connected(self) -> None:
        sprites = slice_alpha(sheet_with_blobs(), min_size=2)
        assert sprites[2].pixels.shape[:2] == (3, 3)  # 2x2 + пиксель по диагонали

    def test_alpha_threshold(self) -> None:
        sheet = np.zeros((4, 4, 4), dtype=np.uint8)
        sheet[1:3, 1:3, 3] = 10
        assert slice_alpha(sheet, threshold=20) == []
        assert len(slice_alpha(sheet, threshold=5, min_size=1)) == 1


class TestPack:
    def sprites(self) -> list[Sprite]:
        return [
            Sprite("a", solid(5, 7)),
            Sprite("b", solid(9, 4, (0, 255, 0, 255))),
            Sprite("c", solid(3, 3, (0, 0, 255, 255))),
        ]

    def test_frames_do_not_overlap_and_hold_pixels(self) -> None:
        built = pack_sprites(self.sprites(), padding=2)
        assert built.width & (built.width - 1) == 0
        assert built.height & (built.height - 1) == 0
        taken = np.zeros(built.pixels.shape[:2], dtype=bool)
        for frame, sprite in zip(built.frames, self.sprites(), strict=True):
            region = (slice(frame.y, frame.y + frame.h), slice(frame.x, frame.x + frame.w))
            assert not taken[region].any()
            taken[region] = True
            assert (built.pixels[region] == sprite.pixels).all()

    def test_padding_is_extruded(self) -> None:
        built = pack_sprites([Sprite("a", solid(4, 4))], padding=2)
        frame = built.frames[0]
        assert (built.pixels[frame.y - 1, frame.x] == (255, 0, 0, 255)).all()
        assert (built.pixels[frame.y - 2, frame.x - 2] == (255, 0, 0, 255)).all()

    def test_trim(self) -> None:
        pixels = np.zeros((8, 8, 4), dtype=np.uint8)
        pixels[2:5, 3:7] = 255
        frame = pack_sprites([Sprite("t", pixels)], trim=True).frames[0]
        assert (frame.w, frame.h) == (4, 3)
        assert frame.offset == (3, 2) and frame.source_size == (8, 8)

    def test_too_big(self) -> None:
        with pytest.raises(AssetImportError, match="max_size"):
            pack_sprites([Sprite("x", solid(40, 40))], max_size=32)
        many = [Sprite(f"s{i}", solid(30, 30)) for i in range(40)]
        with pytest.raises(AssetImportError, match="не помещ"):
            pack_sprites(many, max_size=64)

    def test_duplicate_names(self) -> None:
        with pytest.raises(AssetImportError, match="повторяются"):
            pack_sprites([Sprite("x", solid(2, 2)), Sprite("x", solid(2, 2))])

    def test_uv(self) -> None:
        built = pack_sprites([Sprite("a", solid(4, 4))], padding=0)
        assert built.frames[0].uv(built.width, built.height) == (0.0, 0.0, 1.0, 1.0)


@pytest.fixture()
def db(project: Path):
    with AssetDatabase(project) as database:
        database.scan()
        yield database


def import_spec(db: AssetDatabase, root: Path, name: str, spec: dict):
    write_spec(root, name, spec)
    db.scan()
    guid = db.guid_of(root / name)
    db.reimport(guid, wait=True)
    return guid


class TestImport:
    def test_registered_by_default(self) -> None:
        assert any(isinstance(item, SpriteAtlasImporter) for item in default_importers())

    def test_grid_atlas(self, db: AssetDatabase, project: Path) -> None:
        guid = import_spec(
            db,
            project,
            "grid.atlas.json",
            {"mode": "grid", "image": "grid.png", "cell": [8, 8], "prefix": "tile"},
        )
        handle = db.resolve(guid)
        assert handle.imported, handle.error
        info = handle.artifact("sprite_atlas").info
        assert info["names"] == ["tile_00"] and info["width"] == 16
        atlas = load_sprite_atlas(db, guid)
        assert isinstance(atlas, SpriteAtlas)
        assert atlas.uv("tile_00") == (0.0, 0.0, 0.5, 1.0)
        assert atlas.texture().width == 16

    def test_alpha_atlas_with_names(self, db: AssetDatabase, project: Path) -> None:
        guid = import_spec(
            db,
            project,
            "blobs.atlas.json",
            {"mode": "alpha", "image": "sheet.png", "names": ["red", "green"]},
        )
        atlas = load_sprite_atlas(db, guid)
        assert list(atlas.frames) == ["red", "green", "sprite_02"]
        assert (atlas.frame("green").x, atlas.frame("green").w) == (8, 6)

    def test_pack_folder(self, db: AssetDatabase, project: Path) -> None:
        guid = import_spec(
            db,
            project,
            "ui.atlas.json",
            {"mode": "pack", "folder": "sprites", "padding": 1},
        )
        atlas = load_sprite_atlas(db, guid)
        assert list(atlas.frames) == ["a", "b", "c"]
        texture = atlas.texture()
        assert (texture.width, texture.height) == (atlas.width, atlas.height)
        assert len(texture.mips) == 1  # мипы выключены по умолчанию

    def test_source_change_makes_dirty(self, db: AssetDatabase, project: Path) -> None:
        guid = import_spec(
            db,
            project,
            "ui.atlas.json",
            {"mode": "pack", "sources": ["sprites/a.png"]},
        )
        assert guid not in db.dirty()
        (project / "sprites" / "a.png").write_bytes(encode_png(solid(6, 6)))
        assert guid in db.dirty()

    def test_reimport_is_idempotent(self, db: AssetDatabase, project: Path) -> None:
        guid = import_spec(
            db, project, "ui.atlas.json", {"mode": "pack", "folder": "sprites"}
        )
        first = db.resolve(guid).artifact_path("sprite_atlas")
        db.reimport(guid, wait=True)
        assert sorted(p.name for p in first.parent.iterdir()) == [
            "artifacts.json",
            "atlas.json",
            "texture.bin",
        ]

    def test_errors_fail_import(self, db: AssetDatabase, project: Path) -> None:
        guid = import_spec(
            db, project, "bad.atlas.json", {"mode": "alpha", "image": "nope.png"}
        )
        handle = db.resolve(guid)
        assert not handle.imported and "nope.png" in (handle.error or "")

    def test_path_escape_rejected(self, db: AssetDatabase, project: Path) -> None:
        guid = import_spec(
            db,
            project,
            "bad.atlas.json",
            {"mode": "alpha", "image": "../outside.png"},
        )
        assert not db.resolve(guid).imported

    def test_load_without_artifact(self, db: AssetDatabase, project: Path) -> None:
        write_spec(project, "x.atlas.json", {"mode": "pack", "folder": "sprites"})
        db.scan()
        with pytest.raises(ResourceNotFoundError):
            load_sprite_atlas(db, db.guid_of(project / "x.atlas.json"))
