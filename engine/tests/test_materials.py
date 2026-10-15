"""Тесты PBR-материалов: разбор .mat.json, упаковка ORM, импорт, загрузка, glTF."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_gltf_import import glb_bytes, quad_binary, quad_document

from engine.core.resources import (
    AssetDatabase,
    AssetImportError,
    ImporterRegistry,
    ResourceNotFoundError,
    load_material,
    load_material_artifact,
)
from engine.core.resources.cookers import (
    MaterialSpec,
    encode_png,
    pack_orm,
    parse_material,
    read_cooked_texture,
)
from engine.core.resources.importers import MaterialImporter, default_importers


def gray(value: int, size: tuple[int, int] = (2, 2)) -> np.ndarray:
    return np.full((*size, 1), value, dtype=np.uint8)


def rgb(r: int, g: int, b: int, size: tuple[int, int] = (2, 2)) -> np.ndarray:
    return np.broadcast_to(np.array([r, g, b], np.uint8), (*size, 3)).copy()


class TestSpec:
    def test_defaults(self) -> None:
        spec = parse_material("{}")
        assert spec.shader == "pbr" and spec.base_color == (1, 1, 1, 1)
        assert spec.metallic == 0 and spec.roughness == 1
        assert spec.alpha_mode == "opaque"

    def test_values_and_aliases(self) -> None:
        spec = parse_material(
            json.dumps(
                {
                    "metallic": 0.25,
                    "emissive": [2, 1, 0],
                    "alpha_mode": "MASK",
                    "textures": {"albedo": "a.png", "occlusion": "o.png"},
                }
            )
        )
        assert spec.metallic == 0.25 and spec.emissive == (2, 1, 0)
        assert spec.alpha_mode == "mask"
        assert spec.textures == {"base_color": "a.png", "ao": "o.png"}
        assert spec.packed_sources == {"ao": "o.png"}

    @pytest.mark.parametrize(
        ("data", "message"),
        [
            ({"colour": [1, 1, 1, 1]}, "Неизвестные поля"),
            ({"alpha_mode": "glass"}, "alpha_mode"),
            ({"roughness": 1.5}, "0..1"),
            ({"base_color": [1, 1, 1]}, "массив из 4"),
            ({"base_color": [-1, 1, 1, 1]}, "отрицательными"),
            ({"metallic": float("nan")}, "конечное"),
            ({"textures": {"gloss": "g.png"}}, "неизвестный слот"),
            ({"textures": {"albedo": "a.png", "base_color": "b.png"}}, "дважды"),
            ({"textures": {"normal": ""}}, "путь"),
            ({"textures": []}, "объект"),
            ({"version": 7}, "версия"),
        ],
    )
    def test_invalid(self, data: dict[str, Any], message: str) -> None:
        with pytest.raises(AssetImportError, match=message):
            MaterialSpec.from_mapping(data)

    def test_sorting_layer_read_and_serialized(self) -> None:
        assert MaterialSpec.from_mapping({}).sorting_layer == 0
        spec = MaterialSpec.from_mapping({"sorting_layer": 2})
        assert spec.sorting_layer == 2
        assert spec.to_dict()["sorting_layer"] == 2

    def test_not_json(self) -> None:
        with pytest.raises(AssetImportError, match="JSON"):
            parse_material("{oops")
        with pytest.raises(AssetImportError, match="объектом"):
            parse_material("[1]")


class TestPackOrm:
    def test_channel_placement_and_defaults(self) -> None:
        packed = pack_orm(ao=gray(10), roughness=gray(20))
        assert packed[0, 0].tolist() == [10, 20, 255, 255]

    def test_metallic_roughness_then_explicit_override(self) -> None:
        mr = rgb(0, 100, 200)
        assert pack_orm(metallic_roughness=mr)[0, 0].tolist() == [255, 100, 200, 255]
        packed = pack_orm(metallic_roughness=mr, roughness=gray(7), ao=gray(5))
        assert packed[0, 0].tolist() == [5, 7, 200, 255]

    def test_packed_orm_texture_as_ao_and_mr_source(self) -> None:
        orm = rgb(30, 100, 200)
        packed = pack_orm(ao=orm, metallic_roughness=orm)
        assert packed[0, 0].tolist() == [30, 100, 200, 255]

    def test_different_sizes_use_largest_nearest(self) -> None:
        big = np.arange(16, dtype=np.uint8).reshape(4, 4, 1)
        packed = pack_orm(ao=gray(9, (1, 1)), roughness=big)
        assert packed.shape == (4, 4, 4)
        assert (packed[:, :, 0] == 9).all()
        assert np.array_equal(packed[:, :, 1], big[:, :, 0])

    def test_errors(self) -> None:
        with pytest.raises(AssetImportError, match="Нечего"):
            pack_orm()
        with pytest.raises(AssetImportError, match="3 каналов"):
            pack_orm(metallic_roughness=gray(1))


class TestRegistry:
    def test_compound_extension(self) -> None:
        registry = ImporterRegistry(default_importers())
        assert isinstance(registry.importer_for("a/brick.mat.json"), MaterialImporter)
        assert isinstance(registry.importer_for("A.B.MAT.JSON"), MaterialImporter)
        assert registry.importer_for("settings.json") is None
        assert registry.importer_for("noextension") is None
        assert registry.importer_for("x.obj").name == "obj"


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "tex").mkdir(parents=True)
    (root / "tex" / "albedo.png").write_bytes(encode_png(rgb(200, 100, 50)))
    (root / "tex" / "normal.png").write_bytes(encode_png(rgb(128, 128, 255)))
    (root / "tex" / "rough.png").write_bytes(encode_png(gray(90)))
    (root / "tex" / "ao.png").write_bytes(encode_png(gray(40, (4, 4))))
    (root / "brick.mat.json").write_text(
        json.dumps(
            {
                "name": "Brick",
                "roughness": 0.8,
                "textures": {
                    "albedo": "tex/albedo.png",
                    "normal": "tex/normal.png",
                    "roughness": "tex/rough.png",
                    "ao": "tex/ao.png",
                },
            }
        )
    )
    return root


@pytest.fixture()
def db(project: Path):
    with AssetDatabase(project) as database:
        yield database


def read_orm(db: AssetDatabase, guid: str):
    return read_cooked_texture(db.resolve(guid).artifact_path("orm_texture").read_bytes())


class TestImport:
    def _import(self, db: AssetDatabase, project: Path):
        db.scan()
        db.reimport_dirty(wait=True)
        guid = db.guid_of(project / "brick.mat.json")
        return guid, db.resolve(guid)

    def test_material_and_orm_artifacts(self, db, project) -> None:
        guid, handle = self._import(db, project)
        assert handle.imported, handle.error
        text = handle.artifact_path("material").read_text(encoding="utf-8")
        document = json.loads(text)
        assert document["name"] == "Brick" and document["roughness"] == 0.8
        assert document["textures"]["base_color"] == {"source": "tex/albedo.png"}
        assert document["textures"]["orm"] == {"artifact": "material_orm.bin"}
        orm = read_orm(db, guid)
        assert (orm.width, orm.height) == (4, 4) and not orm.srgb
        assert orm.pixels(0)[0, 0].tolist() == [40, 90, 255, 255]
        assert len(orm.mips) == 3

    def test_dependencies_and_load_material(self, db, project) -> None:
        guid, _ = self._import(db, project)
        names = {db.resolve(g).path.name for g in db.dependencies(guid)}
        assert names == {"albedo.png", "normal.png", "rough.png", "ao.png"}
        material = load_material(db, guid)
        assert material.name == "Brick" and material.params["roughness"] == 0.8
        assert material.texture_guids["base_color"] == db.guid_of(
            project / "tex" / "albedo.png"
        )
        assert set(material.texture_guids) == {"base_color", "normal"}
        assert material.orm_path is not None and material.orm_path.is_file()
        assert material.missing == ()

    def test_packed_input_change_makes_dirty_but_linked_does_not(
        self, db, project
    ) -> None:
        guid, _ = self._import(db, project)
        db.reimport_dirty(wait=True)
        assert guid not in db.dirty()
        (project / "tex" / "albedo.png").write_bytes(encode_png(rgb(1, 2, 3)))
        assert guid not in db.dirty()
        (project / "tex" / "rough.png").write_bytes(encode_png(gray(200)))
        assert guid in db.dirty()
        db.reimport(guid, wait=True)
        assert read_orm(db, guid).pixels(0)[0, 0, 1] == 200
        assert guid not in db.dirty()

    def test_baked_record_survives_restart(self, db, project) -> None:
        guid, _ = self._import(db, project)
        db.close()
        with AssetDatabase(project) as second:
            second.scan()
            assert guid not in second.dirty()
            (project / "tex" / "ao.png").write_bytes(encode_png(gray(1, (4, 4))))
            assert guid in second.dirty()

    def test_settings_control_orm_texture(self, db, project) -> None:
        guid, _ = self._import(db, project)
        db.set_settings(guid, {"generate_mips": False, "max_texture_size": 2})
        db.reimport(guid, wait=True)
        orm = read_orm(db, guid)
        assert (orm.width, len(orm.mips)) == (2, 1)

    def test_material_without_pbr_maps_has_no_orm(self, db, project) -> None:
        source = {"base_color": [1, 0, 0, 1], "textures": {"albedo": "tex/albedo.png"}}
        (project / "plain.mat.json").write_text(json.dumps(source))
        db.scan()
        db.reimport_dirty(wait=True)
        handle = db.resolve(db.guid_of(project / "plain.mat.json"))
        assert handle.imported and handle.artifact("orm_texture") is None
        assert load_material(db, handle.guid).orm_path is None

    @pytest.mark.parametrize(
        "textures",
        [
            {"normal": "tex/missing.png"},
            {"roughness": "tex/missing.png"},
            {"normal": "/abs.png"},
        ],
    )
    def test_missing_or_absolute_texture_fails_cleanly(
        self, db, project, textures
    ) -> None:
        (project / "bad.mat.json").write_text(json.dumps({"textures": textures}))
        guid = db.register(project / "bad.mat.json")
        db.reimport(guid, wait=True)
        handle = db.resolve(guid)
        assert handle.error and not handle.imported
        assert not list(db.library.rglob("*.tmp"))

    def test_unregistered_texture_is_reported_missing(self, db, project) -> None:
        guid, _ = self._import(db, project)
        (project / "tex" / "albedo.png").unlink()
        (project / "tex" / "albedo.png.meta").unlink()
        with AssetDatabase(project) as reopened:
            reopened.scan()
            assert "base_color" in load_material(reopened, guid).missing

    def test_load_material_artifact_unknown_name(self, db, project) -> None:
        guid, _ = self._import(db, project)
        with pytest.raises(ResourceNotFoundError):
            load_material_artifact(db, guid, "nope.json")

    def test_inspect(self, project) -> None:
        preview = MaterialImporter().inspect(project / "brick.mat.json")
        assert preview.kind == "material" and "normal" in preview.info["textures"]


class TestGltfMaterials:
    @pytest.fixture(autouse=True)
    def _require(self) -> None:
        pytest.importorskip("pygltflib")

    def _scene(self, root: Path) -> Path:
        root.mkdir(exist_ok=True)
        (root / "albedo.png").write_bytes(encode_png(rgb(255, 0, 0)))
        (root / "normal.png").write_bytes(encode_png(rgb(128, 128, 255)))
        (root / "orm.png").write_bytes(encode_png(rgb(10, 120, 240)))
        document = quad_document(
            materials=[
                {
                    "name": "metal",
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [0.5, 0.5, 0.5, 1],
                        "baseColorTexture": {"index": 0},
                        "metallicFactor": 0.9,
                        "roughnessFactor": 0.4,
                        "metallicRoughnessTexture": {"index": 2},
                    },
                    "normalTexture": {"index": 1, "scale": 2.0},
                    "occlusionTexture": {"index": 2, "strength": 0.5},
                    "alphaMode": "MASK",
                    "alphaCutoff": 0.3,
                    "doubleSided": True,
                }
            ],
            textures=[{"source": 0}, {"source": 1}, {"source": 2}],
            images=[{"uri": "albedo.png"}, {"uri": "normal.png"}, {"uri": "orm.png"}],
        )
        path = root / "model.glb"
        path.write_bytes(glb_bytes(document, quad_binary()))
        return path

    def test_materials_extracted_once_and_linked_from_meshes(self, tmp_path) -> None:
        model = self._scene(tmp_path / "gl")
        with AssetDatabase(tmp_path / "gl") as db:
            db.scan()
            db.reimport_dirty(wait=True)
            guid = db.guid_of(model)
            handle = db.resolve(guid)
            assert handle.imported, handle.error
            meshes = [a for a in handle.artifacts if a.kind == "mesh"]
            assert len(meshes) == 2
            linked = {m.info["material_artifact"] for m in meshes}
            assert linked == {"material_000.json"}
            assert len([a for a in handle.artifacts if a.kind == "material"]) == 1

            material = load_material_artifact(db, guid, "material_000.json")
            assert material.name == "metal"
            params = material.params
            assert params["metallic"] == 0.9 and params["roughness"] == 0.4
            assert params["alpha_mode"] == "mask" and params["alpha_cutoff"] == 0.3
            assert params["double_sided"] and params["normal_scale"] == 2.0
            assert params["occlusion_strength"] == 0.5
            assert set(material.texture_guids) == {"base_color", "normal"}
            orm = read_cooked_texture(material.orm_path.read_bytes())
            assert orm.pixels(0)[0, 0].tolist() == [10, 120, 240, 255]

    def test_orm_change_marks_model_dirty_but_albedo_does_not(self, tmp_path) -> None:
        model = self._scene(tmp_path / "gl")
        with AssetDatabase(tmp_path / "gl") as db:
            db.scan()
            db.reimport_dirty(wait=True)
            guid = db.guid_of(model)
            (tmp_path / "gl" / "albedo.png").write_bytes(encode_png(rgb(0, 255, 0)))
            assert guid not in db.dirty()
            (tmp_path / "gl" / "orm.png").write_bytes(encode_png(rgb(1, 2, 3)))
            assert guid in db.dirty()

    def test_extract_materials_can_be_disabled(self, tmp_path) -> None:
        model = self._scene(tmp_path / "gl")
        with AssetDatabase(tmp_path / "gl") as db:
            guid = db.register(model)
            db.set_settings(guid, {"extract_materials": False})
            db.reimport(guid, wait=True)
            handle = db.resolve(guid)
            assert handle.imported
            assert all(a.kind == "mesh" for a in handle.artifacts)
            assert handle.artifacts[0].info["material_artifact"] is None

    def test_broken_material_does_not_break_the_model(self, tmp_path) -> None:
        model = self._scene(tmp_path / "gl")
        (tmp_path / "gl" / "orm.png").write_bytes(b"not a png")
        with AssetDatabase(tmp_path / "gl") as db:
            guid = db.register(model)
            db.reimport(guid, wait=True)
            handle = db.resolve(guid)
            assert handle.imported, handle.error
            assert handle.artifact("material") is None
            assert handle.artifacts[0].info["material_artifact"] is None
