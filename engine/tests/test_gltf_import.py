"""Тесты импортёра glTF/GLB (сборка модели — вручную, без pygltflib)."""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from engine.core.resources import (
    AssetDatabase,
    AssetImportError,
    ImportContext,
    new_guid,
)
from engine.core.resources.cookers import read_cooked_mesh
from engine.core.resources.importers import GltfImporter

QUAD_POSITIONS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
QUAD_NORMALS = np.tile(np.array([0, 0, 1], np.float32), (4, 1))
QUAD_UVS = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
QUAD_INDICES = np.array([0, 1, 2, 0, 2, 3], np.uint16)


def quad_binary(indices: np.ndarray = QUAD_INDICES) -> bytes:
    return (
        QUAD_POSITIONS.tobytes()
        + QUAD_NORMALS.tobytes()
        + QUAD_UVS.tobytes()
        + indices.astype("<u2").tobytes()
    )


def quad_document(index_count: int = 6, **overrides: Any) -> dict[str, Any]:
    binary_size = 48 + 48 + 32 + index_count * 2
    document: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "test"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [
            {"mesh": 0, "name": "moved", "translation": [10, 0, 0]},
            {"mesh": 0, "name": "scaled", "scale": [2, 2, 2]},
        ],
        "meshes": [
            {
                "name": "quad",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                        "indices": 3,
                        "material": 0,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "red",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1, 0, 0, 1],
                    "baseColorTexture": {"index": 0},
                },
            }
        ],
        "textures": [{"source": 0}],
        "images": [{"uri": "albedo.png"}],
        "buffers": [{"byteLength": binary_size}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 48},
            {"buffer": 0, "byteOffset": 48, "byteLength": 48},
            {"buffer": 0, "byteOffset": 96, "byteLength": 32},
            {"buffer": 0, "byteOffset": 128, "byteLength": index_count * 2},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC2"},
            {
                "bufferView": 3,
                "componentType": 5123,
                "count": index_count,
                "type": "SCALAR",
            },
        ],
    }
    document.update(overrides)
    return document


def glb_bytes(document: dict[str, Any], binary: bytes) -> bytes:
    text = json.dumps(document).encode()
    text += b" " * (-len(text) % 4)
    binary += b"\0" * (-len(binary) % 4)
    total = 12 + 8 + len(text) + 8 + len(binary)
    return (
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(text), 0x4E4F534A)
        + text
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )


@pytest.fixture()
def glb(tmp_path: Path) -> Path:
    path = tmp_path / "quad.glb"
    path.write_bytes(glb_bytes(quad_document(), quad_binary()))
    return path


def run_import(path: Path, tmp_path: Path, **settings: Any):
    importer = GltfImporter()
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    ctx = ImportContext(
        new_guid(), path, {**importer.default_settings(), **settings}, out
    )
    return importer.import_asset(ctx), ctx, out


def test_inspect_works_without_pygltflib(glb: Path) -> None:
    preview = GltfImporter().inspect(glb)
    assert preview.kind == "model"
    assert dict(preview.info) == {
        "meshes": 1,
        "primitives": 1,
        "nodes": 2,
        "materials": 1,
        "generator": "test",
    }


def test_inspect_gltf_text_and_garbage(tmp_path: Path) -> None:
    path = tmp_path / "a.gltf"
    path.write_text(json.dumps(quad_document()), encoding="utf-8")
    assert GltfImporter().inspect(path).info["nodes"] == 2
    path.write_text("{oops", encoding="utf-8")
    with pytest.raises(AssetImportError):
        GltfImporter().inspect(path)
    broken = tmp_path / "b.glb"
    broken.write_bytes(b"NOPE" + bytes(30))
    with pytest.raises(AssetImportError, match="GLB"):
        GltfImporter().inspect(broken)


def test_missing_pygltflib_gives_actionable_error(
    glb: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setitem(__import__("sys").modules, "pygltflib", None)
    with pytest.raises(AssetImportError, match=r"\[gltf\]"):
        run_import(glb, tmp_path)


class TestWithPygltflib:
    @pytest.fixture(autouse=True)
    def _require(self) -> None:
        pytest.importorskip("pygltflib")

    def test_one_artifact_per_node_primitive_with_baked_transforms(
        self, glb: Path, tmp_path: Path
    ) -> None:
        artifacts, _ctx, out = run_import(glb, tmp_path)
        assert [a.name for a in artifacts] == ["mesh_000.bin", "mesh_001.bin"]
        moved, scaled = artifacts
        assert (moved.info["node"], scaled.info["node"]) == ("moved", "scaled")
        assert moved.info["bounds_min"] == [10.0, 0.0, 0.0]
        assert moved.info["bounds_max"] == [11.0, 1.0, 0.0]
        assert scaled.info["bounds_max"] == [2.0, 2.0, 0.0]
        assert moved.info["triangles"] == 2

        cooked = read_cooked_mesh(out / moved.name)
        assert cooked.positions()[:, 0].min() == pytest.approx(10.0, abs=1e-3)
        assert np.allclose(cooked.normals(), [0, 0, 1], atol=1e-3)
        assert cooked.lods[0].uvs is not None

    def test_scale_setting(self, glb: Path, tmp_path: Path) -> None:
        artifacts, _, _ = run_import(glb, tmp_path, scale=0.5)
        assert artifacts[0].info["bounds_max"] == [5.5, 0.5, 0.0]

    def test_material_and_texture_dependency(self, glb: Path, tmp_path: Path) -> None:
        (tmp_path / "albedo.png").write_bytes(b"x")
        artifacts, ctx, _ = run_import(glb, tmp_path)
        material = artifacts[0].info["material"]
        assert material["name"] == "red" and material["base_color"] == [1, 0, 0, 1]
        assert material["base_color_texture"] == "albedo.png"
        assert ctx.dependencies == [tmp_path / "albedo.png"]

    def test_missing_texture_file_is_not_a_dependency(
        self, glb: Path, tmp_path: Path
    ) -> None:
        _, ctx, _ = run_import(glb, tmp_path)
        assert ctx.dependencies == []

    def test_gltf_with_data_uri_and_external_bin(self, tmp_path: Path) -> None:
        binary = quad_binary()
        document = quad_document()
        document["buffers"][0]["uri"] = "data:application/octet-stream;base64," + (
            base64.b64encode(binary).decode()
        )
        embedded = tmp_path / "embedded.gltf"
        embedded.write_text(json.dumps(document), encoding="utf-8")
        assert len(run_import(embedded, tmp_path)[0]) == 2

        (tmp_path / "geo.bin").write_bytes(binary)
        document["buffers"][0]["uri"] = "geo.bin"
        external = tmp_path / "external.gltf"
        external.write_text(json.dumps(document), encoding="utf-8")
        assert len(run_import(external, tmp_path)[0]) == 2

    def test_triangle_strip_and_missing_normals_generated(self, tmp_path: Path) -> None:
        document = quad_document(index_count=4)
        document["meshes"][0]["primitives"][0]["mode"] = 5
        del document["meshes"][0]["primitives"][0]["attributes"]["NORMAL"]
        path = tmp_path / "strip.glb"
        path.write_bytes(
            glb_bytes(document, quad_binary(np.array([0, 1, 3, 2], np.uint16)))
        )
        artifacts, _, out = run_import(path, tmp_path)
        assert artifacts[0].info["triangles"] == 2
        assert artifacts[0].info["has_normals"] is True
        # strip 0,1,3,2 → (0,1,3) и (1,2,3): оба смотрят в +Z
        normals = read_cooked_mesh(out / "mesh_000.bin").normals()
        assert (normals[:, 2] > 0.9).all()

    def test_mirrored_node_flips_winding(self, tmp_path: Path) -> None:
        document = quad_document()
        document["nodes"] = [{"mesh": 0, "scale": [-1, 1, 1]}]
        document["scenes"] = [{"nodes": [0]}]
        path = tmp_path / "mirror.glb"
        path.write_bytes(glb_bytes(document, quad_binary()))
        artifacts, _, out = run_import(path, tmp_path, generate_normals=False)
        cooked = read_cooked_mesh(out / artifacts[0].name)
        triangle = cooked.positions()[cooked.lods[0].indices[:3].astype(int)]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        assert normal[2] > 0  # после отражения грань по-прежнему лицевая

    def test_quaternion_rotation(self, tmp_path: Path) -> None:
        half = np.sqrt(0.5)
        document = quad_document()
        document["nodes"] = [{"mesh": 0, "rotation": [0, 0, half, half]}]  # +90° по Z
        document["scenes"] = [{"nodes": [0]}]
        path = tmp_path / "rot.glb"
        path.write_bytes(glb_bytes(document, quad_binary()))
        artifacts, _, _ = run_import(path, tmp_path)
        assert artifacts[0].info["bounds_min"] == pytest.approx([-1, 0, 0], abs=1e-6)
        assert artifacts[0].info["bounds_max"] == pytest.approx([0, 1, 0], abs=1e-6)

    def test_child_inherits_parent_transform(self, tmp_path: Path) -> None:
        document = quad_document()
        document["nodes"] = [
            {"translation": [0, 5, 0], "children": [1]},
            {"mesh": 0, "translation": [1, 0, 0]},
        ]
        document["scenes"] = [{"nodes": [0]}]
        path = tmp_path / "tree.glb"
        path.write_bytes(glb_bytes(document, quad_binary()))
        artifacts, _, _ = run_import(path, tmp_path)
        assert artifacts[0].info["bounds_min"] == [1.0, 5.0, 0.0]

    def test_cyclic_hierarchy_is_rejected(self, tmp_path: Path) -> None:
        document = quad_document()
        document["nodes"] = [{"children": [1]}, {"children": [0], "mesh": 0}]
        document["scenes"] = [{"nodes": [0]}]
        path = tmp_path / "cycle.glb"
        path.write_bytes(glb_bytes(document, quad_binary()))
        with pytest.raises(AssetImportError, match="зациклена"):
            run_import(path, tmp_path)

    def test_truncated_buffer_and_bad_index(self, tmp_path: Path) -> None:
        path = tmp_path / "short.glb"
        path.write_bytes(glb_bytes(quad_document(), quad_binary()[:-20]))
        with pytest.raises(AssetImportError):
            run_import(path, tmp_path)

        bad = tmp_path / "badindex.glb"
        bad.write_bytes(
            glb_bytes(quad_document(), quad_binary(np.array([0, 1, 9, 0, 2, 3], np.uint16)))
        )
        with pytest.raises(AssetImportError, match="за пределы"):
            run_import(bad, tmp_path)

    def test_scene_without_meshes_fails(self, tmp_path: Path) -> None:
        document = quad_document()
        document["nodes"] = [{}]
        document["scenes"] = [{"nodes": [0]}]
        path = tmp_path / "empty.glb"
        path.write_bytes(glb_bytes(document, quad_binary()))
        with pytest.raises(AssetImportError, match="нет треугольных"):
            run_import(path, tmp_path)

    def test_sparse_accessor_rejected(self, tmp_path: Path) -> None:
        document = quad_document()
        document["accessors"][0]["sparse"] = {
            "count": 1,
            "indices": {"bufferView": 3, "componentType": 5123},
            "values": {"bufferView": 0},
        }
        path = tmp_path / "sparse.glb"
        path.write_bytes(glb_bytes(document, quad_binary()))
        with pytest.raises(AssetImportError, match="Sparse"):
            run_import(path, tmp_path)

    def test_strided_and_normalized_accessors(self, tmp_path: Path) -> None:
        # интерливинг: [pos(3 f32) | uv(2 uint16 normalized)] с шагом 16 байт
        rows = b"".join(
            struct.pack("<3f", *p) + struct.pack("<2H", int(u * 65535), int(v * 65535))            for p, (u, v) in zip(QUAD_POSITIONS.tolist(), QUAD_UVS.tolist(), strict=True)
        )
        binary = rows + QUAD_INDICES.tobytes()
        document = quad_document()
        document["buffers"][0]["byteLength"] = len(binary)
        document["bufferViews"] = [
            {"buffer": 0, "byteOffset": 0, "byteLength": 64, "byteStride": 16},
            {"buffer": 0, "byteOffset": 64, "byteLength": 12},
        ]
        document["accessors"] = [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3"},
            {
                "bufferView": 0,
                "byteOffset": 12,
                "componentType": 5123,
                "normalized": True,
                "count": 4,
                "type": "VEC2",
            },
            {"bufferView": 1, "componentType": 5123, "count": 6, "type": "SCALAR"},
        ]
        document["meshes"][0]["primitives"][0]["attributes"] = {
            "POSITION": 0,
            "TEXCOORD_0": 1,
        }
        document["meshes"][0]["primitives"][0]["indices"] = 2
        path = tmp_path / "strided.glb"
        path.write_bytes(glb_bytes(document, binary))
        artifacts, _, out = run_import(path, tmp_path)
        cooked = read_cooked_mesh(out / artifacts[0].name)
        assert cooked.lods[0].uvs.max() == pytest.approx(1.0)
        assert artifacts[0].info["bounds_max"] == [11.0, 1.0, 0.0]

    def test_pipeline_end_to_end_with_database(self, glb: Path, tmp_path: Path) -> None:
        assets = tmp_path / "Assets"
        assets.mkdir()
        (assets / "quad.glb").write_bytes(glb.read_bytes())
        (assets / "albedo.png").write_bytes(
            (Path(__file__).parent / "fixtures" / "tiny.png").read_bytes()
        )
        with AssetDatabase(assets) as db:
            model = db.register(assets / "quad.glb")
            db.reimport(model, wait=True)
            handle = db.resolve(model)
            assert handle.error is None and len(handle.artifacts) == 3  # 2 меша + материал
            texture = db.guid_of(assets / "albedo.png")
            assert db.dependencies(model) == [texture]
            assert db.dependents(texture) == [model]
            db.reimport(texture, wait=True)
            assert read_cooked_mesh(handle.artifact_path(handle.artifacts[1])).lods
            assert db.dirty() == []
