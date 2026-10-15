"""Тесты mesh cooker: LOD, переупорядочивание, квантизация и формат mesh.bin."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from engine.core.resources import AssetImportError, ResourceFormatError, uv_sphere_mesh
from engine.core.resources.cookers import (
    MeshCookSettings,
    cook_mesh,
    read_cooked_mesh,
    reorder_vertices,
    simplify_clustering,
)
from engine.core.resources.cookers.mesh_format import HEADER_SIZE, LOD_ENTRY_SIZE


@pytest.fixture(scope="module")
def sphere():
    return uv_sphere_mesh(radius=2.0, segments=48, rings=32)


def cook(mesh, **kwargs):
    return cook_mesh(
        mesh.positions, mesh.indices, mesh.normals, mesh.uvs, MeshCookSettings(**kwargs)
    )


class TestQuantization:
    def test_positions_roundtrip_within_quantization_step(self, sphere) -> None:
        cooked = read_cooked_mesh(cook(sphere, generate_lods=False))
        lod = cooked.lods[0]
        assert lod.vertex_count == len(np.unique(sphere.indices))
        # порядок вершин изменился, но облако то же: сравниваем отсортированные
        original = sphere.positions[np.unique(sphere.indices)]
        restored = cooked.positions(0)
        step = 2 * cooked.half_extent.max() / 65534
        for axis in range(3):
            assert np.abs(np.sort(restored[:, axis]) - np.sort(original[:, axis])).max() <= step

    def test_normals_are_unit_after_dequantization(self, sphere) -> None:
        normals = read_cooked_mesh(cook(sphere)).normals(0)
        assert np.abs(np.linalg.norm(normals, axis=1) - 1).max() < 1e-3

    def test_uvs_are_kept_exactly(self, sphere) -> None:
        cooked = read_cooked_mesh(cook(sphere, generate_lods=False))
        used = np.unique(sphere.indices)
        assert sorted(map(tuple, cooked.lods[0].uvs.tolist())) == sorted(
            map(tuple, sphere.uvs[used].tolist())
        )

    def test_flat_axis_does_not_divide_by_zero(self) -> None:
        positions = np.array([[0, 0, 5], [1, 0, 5], [0, 1, 5]], dtype=np.float32)
        cooked = read_cooked_mesh(cook_mesh(positions, np.array([0, 1, 2])))
        assert np.allclose(cooked.positions()[:, 2], 5.0)

    def test_geometry_is_triangle_connectivity_preserving(self, sphere) -> None:
        cooked = read_cooked_mesh(cook(sphere, generate_lods=False))
        assert cooked.lods[0].triangle_count == sphere.triangle_count


class TestLods:
    def test_lods_shrink_towards_ratios(self, sphere) -> None:
        cooked = read_cooked_mesh(cook(sphere))
        counts = [lod.triangle_count for lod in cooked.lods]
        assert len(counts) == 3
        assert counts[0] == sphere.triangle_count
        assert counts[1] <= counts[0] * 0.5 and counts[2] <= counts[0] * 0.25
        assert counts[2] > 0
        assert counts[0] > counts[1] > counts[2]

    def test_lods_stay_inside_original_bounds(self, sphere) -> None:
        cooked = read_cooked_mesh(cook(sphere))
        radius = np.linalg.norm(cooked.positions(2), axis=1)
        assert radius.max() <= 2.0 + 1e-3
        assert radius.min() > 1.0  # не схлопнулось в центр

    def test_lod_indices_are_valid_and_vertices_compact(self, sphere) -> None:
        for lod in read_cooked_mesh(cook(sphere)).lods:
            assert lod.indices.max() == lod.vertex_count - 1
            assert len(np.unique(lod.indices)) == lod.vertex_count
            assert lod.indices.size % 3 == 0

    def test_disabled_lods(self, sphere) -> None:
        assert len(read_cooked_mesh(cook(sphere, generate_lods=False)).lods) == 1

    def test_tiny_mesh_has_no_redundant_lods(self) -> None:
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        assert len(read_cooked_mesh(cook_mesh(positions, np.array([0, 1, 2]))).lods) == 1

    def test_simplify_never_returns_empty_mesh(self, sphere) -> None:
        _, indices, _, _ = simplify_clustering(
            sphere.positions, sphere.indices, sphere.normals, sphere.uvs, 1
        )
        assert indices.size >= 3

    @pytest.mark.parametrize("ratios", [(1.0,), (0.0,), (0.25, 0.5), (-0.1,)])
    def test_bad_ratios_rejected(self, ratios) -> None:
        with pytest.raises(AssetImportError, match="lod_ratios"):
            MeshCookSettings(lod_ratios=ratios)

    def test_settings_from_mapping(self) -> None:
        settings = MeshCookSettings.from_mapping(
            {"generate_lods": False, "lod_ratios": [0.4]}
        )
        assert settings == MeshCookSettings(False, (0.4,))
        with pytest.raises(AssetImportError):
            MeshCookSettings.from_mapping({"lod_ratios": "oops"})


class TestReorder:
    def test_vertices_follow_first_use_and_unused_are_dropped(self) -> None:
        positions = np.arange(15, dtype=np.float32).reshape(5, 3)
        indices = np.array([3, 1, 4, 4, 1, 3], dtype=np.uint32)
        new_pos, new_idx, _, _ = reorder_vertices(positions, indices)
        assert new_idx.tolist() == [0, 1, 2, 2, 1, 0]
        assert new_pos[:, 0].tolist() == [9, 3, 12]


class TestFormat:
    def test_sections_are_16_byte_aligned(self, sphere) -> None:
        blob = cook(sphere)
        lods = struct.unpack_from("<I", blob, 8)[0]
        for index in range(lods):
            entry = struct.unpack_from("<IIIIQQQQ", blob, HEADER_SIZE + index * LOD_ENTRY_SIZE)
            assert all(offset % 16 == 0 for offset in entry[4:])

    def test_large_mesh_uses_32bit_indices(self) -> None:
        side = 260
        xs, ys = np.meshgrid(np.arange(side), np.arange(side))
        positions = np.stack([xs.ravel(), ys.ravel(), np.zeros(side * side)], axis=1)
        grid = np.arange(side * side).reshape(side, side)
        a, b, c, d = grid[:-1, :-1], grid[:-1, 1:], grid[1:, 1:], grid[1:, :-1]
        indices = np.stack([a, b, c, a, c, d], axis=-1).reshape(-1)
        cooked = read_cooked_mesh(cook_mesh(positions.astype(np.float32), indices))
        assert cooked.lods[0].indices.dtype.itemsize == 4
        assert cooked.lods[0].vertex_count == side * side

    def test_point_cloud_has_single_lod_without_indices(self) -> None:
        cooked = read_cooked_mesh(cook_mesh(np.random.rand(10, 3).astype(np.float32), None))
        assert cooked.lods[0].vertex_count == 10
        assert cooked.lods[0].indices.size == 0

    def test_optional_attributes_are_omitted(self) -> None:
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        lod = read_cooked_mesh(cook_mesh(positions, np.array([0, 1, 2]))).lods[0]
        assert lod.normals_q is None and lod.uvs is None

    def test_reads_from_memoryview_without_copy(self, sphere) -> None:
        blob = bytearray(cook(sphere))
        cooked = read_cooked_mesh(memoryview(blob))
        before = cooked.lods[0].positions_q[0, 0]
        struct.pack_into("<h", blob, struct.unpack_from("<Q", blob, HEADER_SIZE + 16)[0], before ^ 1)
        assert cooked.lods[0].positions_q[0, 0] == before ^ 1

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda b: b"XXXX" + b[4:], "сигнатура"),
            (lambda b: b[:4] + struct.pack("<I", 99) + b[8:], "версия"),
            (lambda b: b[:30], "короче"),
            (lambda b: b[:-100], "вне файла"),
            (lambda b: b[:8] + struct.pack("<I", 0) + b[12:], "таблица"),
        ],
    )
    def test_corrupted_files_are_rejected(self, sphere, mutate, message) -> None:
        with pytest.raises(ResourceFormatError, match=message):
            read_cooked_mesh(mutate(cook(sphere)))


class TestInputValidation:
    def test_index_out_of_range(self) -> None:
        positions = np.zeros((3, 3), dtype=np.float32)
        with pytest.raises(AssetImportError, match="выходит"):
            cook_mesh(positions, np.array([0, 1, 3]))

    def test_non_triangle_index_count(self) -> None:
        with pytest.raises(AssetImportError, match="кратно"):
            cook_mesh(np.zeros((4, 3), dtype=np.float32), np.array([0, 1, 2, 3]))

    def test_mismatched_attribute_shapes(self) -> None:
        positions = np.zeros((3, 3), dtype=np.float32)
        with pytest.raises(AssetImportError, match="normals"):
            cook_mesh(positions, np.array([0, 1, 2]), normals=np.zeros((2, 3)))

    def test_nan_and_empty(self) -> None:
        with pytest.raises(AssetImportError, match="NaN"):
            cook_mesh(np.full((3, 3), np.nan, dtype=np.float32), np.array([0, 1, 2]))
        with pytest.raises(AssetImportError, match="вершин"):
            cook_mesh(np.zeros((0, 3), dtype=np.float32), None)
