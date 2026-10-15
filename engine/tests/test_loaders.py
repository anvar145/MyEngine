"""Тесты встроенных загрузчиков: OBJ, изображения, GLSL с включениями."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from engine.core.resources import (
    AssetLoader,
    GlslShaderLoader,
    ImageTextureLoader,
    LoadContext,
    ObjMeshLoader,
    ResourceFormatError,
    ResourceManager,
    ShaderStage,
)

TRIANGLE_OBJ = """# тестовый треугольник
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
vn 0 0 1
f 1/1/1 2/2/1 3/3/1
"""

QUAD_OBJ = """v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
"""


@pytest.fixture()
def manager() -> ResourceManager:
    """Менеджер со стандартными загрузчиками (для контекста загрузки)."""
    return ResourceManager()


def context(path: Path, manager: ResourceManager, **options: object) -> LoadContext:
    """Собрать контекст загрузки для прямого вызова загрузчика."""
    return LoadContext(path, manager, options)


class TestObjLoader:
    def test_triangle_with_uv_and_normals(self, tmp_path: Path) -> None:
        path = tmp_path / "triangle.obj"
        path.write_text(TRIANGLE_OBJ, encoding="utf-8")

        asset = ObjMeshLoader().load(context(path, ResourceManager()))
        assert asset.vertex_count == 3
        assert asset.triangle_count == 1
        assert asset.indices is not None
        assert asset.indices.tolist() == [0, 1, 2]
        assert asset.normals is not None
        assert asset.uvs is not None
        minimum, maximum = asset.bounds
        assert minimum.to_tuple() == (0.0, 0.0, 0.0)
        assert maximum.to_tuple() == (1.0, 1.0, 0.0)

    def test_quad_is_triangulated(self, tmp_path: Path) -> None:
        path = tmp_path / "quad.obj"
        path.write_text(QUAD_OBJ, encoding="utf-8")
        asset = ObjMeshLoader().load(context(path, ResourceManager()))
        assert asset.vertex_count == 4
        assert asset.triangle_count == 2
        assert asset.indices is not None
        assert asset.indices.tolist() == [0, 1, 2, 0, 2, 3]
        assert asset.normals is None
        assert asset.uvs is None

    def test_negative_indices(self, tmp_path: Path) -> None:
        path = tmp_path / "neg.obj"
        path.write_text(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n", encoding="utf-8"
        )
        asset = ObjMeshLoader().load(context(path, ResourceManager()))
        assert asset.indices is not None
        assert asset.indices.tolist() == [0, 1, 2]

    def test_scale_option(self, tmp_path: Path) -> None:
        path = tmp_path / "scale.obj"
        path.write_text("v 1 1 1\nv 2 2 2\nv 3 3 3\nf 1 2 3\n", encoding="utf-8")
        asset = ObjMeshLoader().load(context(path, ResourceManager(), scale=2.0))
        assert asset.positions[0].tolist() == [2.0, 2.0, 2.0]

    def test_duplicate_vertices_are_merged(self, tmp_path: Path) -> None:
        path = tmp_path / "merged.obj"
        path.write_text(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\nf 3 2 1\n", encoding="utf-8"
        )
        asset = ObjMeshLoader().load(context(path, ResourceManager()))
        assert asset.vertex_count == 3
        assert asset.triangle_count == 2

    def test_points_without_faces(self, tmp_path: Path) -> None:
        path = tmp_path / "points.obj"
        path.write_text("v 0 0 0\nv 1 1 1\n", encoding="utf-8")
        asset = ObjMeshLoader().load(context(path, ResourceManager()))
        assert asset.vertex_count == 2
        assert asset.indices is None

    def test_index_out_of_range(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.obj"
        path.write_text("v 0 0 0\nf 1 2 3\n", encoding="utf-8")
        with pytest.raises(ResourceFormatError):
            ObjMeshLoader().load(context(path, ResourceManager()))

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.obj"
        path.write_text("# только комментарий\n", encoding="utf-8")
        with pytest.raises(ResourceFormatError):
            ObjMeshLoader().load(context(path, ResourceManager()))

    def test_short_vertex_line(self, tmp_path: Path) -> None:
        path = tmp_path / "short.obj"
        path.write_text("v 0 0\n", encoding="utf-8")
        with pytest.raises(ResourceFormatError):
            ObjMeshLoader().load(context(path, ResourceManager()))


def _write_png(path: Path, width: int, height: int, color_type: int = 6) -> None:
    """Записать минимальный PNG: сигнатура и блок IHDR."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr)


def _write_dds(path: Path, width: int, height: int, four_cc: bytes = b"DXT1") -> None:
    """Записать минимальный DDS: заголовок 128 байт."""
    header = bytearray(128)
    header[0:4] = b"DDS "
    struct.pack_into("<I", header, 4, 124)
    struct.pack_into("<I", header, 12, height)
    struct.pack_into("<I", header, 16, width)
    struct.pack_into("<I", header, 28, 5)
    header[84:88] = four_cc
    path.write_bytes(bytes(header))


def _write_bmp(path: Path, width: int, height: int, bits: int = 24) -> None:
    """Записать минимальный BMP: BITMAPFILEHEADER и BITMAPINFOHEADER."""
    header = bytearray(54)
    header[0:2] = b"BM"
    struct.pack_into("<i", header, 18, width)
    struct.pack_into("<i", header, 22, height)
    struct.pack_into("<H", header, 28, bits)
    path.write_bytes(bytes(header))


class TestImageLoader:
    def test_png_header(self, tmp_path: Path) -> None:
        path = tmp_path / "image.png"
        _write_png(path, 64, 32)
        asset = ImageTextureLoader().load(
            context(path, ResourceManager(), decode=False)
        )
        assert (asset.width, asset.height) == (64, 32)
        assert asset.channels == 4
        assert not asset.compressed
        assert not asset.is_decoded

    def test_png_grayscale(self, tmp_path: Path) -> None:
        path = tmp_path / "gray.png"
        _write_png(path, 8, 8, color_type=0)
        asset = ImageTextureLoader().load(
            context(path, ResourceManager(), decode=False)
        )
        assert asset.channels == 1

    def test_dds_block_compressed(self, tmp_path: Path) -> None:
        path = tmp_path / "texture.dds"
        _write_dds(path, 512, 256)
        asset = ImageTextureLoader().load(
            context(path, ResourceManager(), decode=False)
        )
        assert (asset.width, asset.height) == (512, 256)
        assert asset.compressed is True
        assert asset.pixel_format == "bc1"
        assert asset.mip_levels == 5

    def test_bmp_header(self, tmp_path: Path) -> None:
        path = tmp_path / "image.bmp"
        _write_bmp(path, 20, 10)
        asset = ImageTextureLoader().load(
            context(path, ResourceManager(), decode=False)
        )
        assert (asset.width, asset.height) == (20, 10)
        assert asset.channels == 3

    def test_unknown_format(self, tmp_path: Path) -> None:
        path = tmp_path / "fake.png"
        path.write_bytes(b"not an image at all")
        with pytest.raises(ResourceFormatError):
            ImageTextureLoader().load(context(path, ResourceManager()))

    def test_broken_png(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.png"
        payload = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IDAT" + b"\x00" * 13
        path.write_bytes(payload)
        with pytest.raises(ResourceFormatError):
            ImageTextureLoader().load(context(path, ResourceManager()))


class TestShaderLoader:
    def test_stage_from_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "simple.frag"
        path.write_text("void main() {}\n", encoding="utf-8")
        asset = GlslShaderLoader().load(context(path, ResourceManager()))
        assert asset.stage is ShaderStage.FRAGMENT
        assert asset.includes == ()

    def test_stage_from_directive(self, tmp_path: Path) -> None:
        path = tmp_path / "generic.glsl"
        path.write_text("#stage vertex\nvoid main() {}\n", encoding="utf-8")
        asset = GlslShaderLoader().load(context(path, ResourceManager()))
        assert asset.stage is ShaderStage.VERTEX

    def test_include_is_resolved(self, tmp_path: Path) -> None:
        (tmp_path / "common.glsl").write_text(
            "float helper() { return 1.0; }\n", encoding="utf-8"
        )
        path = tmp_path / "main.frag"
        path.write_text(
            '#include "common.glsl"\nvoid main() { float x = helper(); }\n',
            encoding="utf-8",
        )
        asset = GlslShaderLoader().load(context(path, ResourceManager()))
        assert "float helper()" in asset.source
        assert "void main()" in asset.source
        assert len(asset.includes) == 1
        assert Path(asset.includes[0]).name == "common.glsl"

    def test_include_from_extra_directory(self, tmp_path: Path) -> None:
        shared = tmp_path / "shared"
        shared.mkdir()
        (shared / "lib.glsl").write_text("// lib\n", encoding="utf-8")
        path = tmp_path / "main.vert"
        path.write_text('#include "lib.glsl"\nvoid main() {}\n', encoding="utf-8")
        asset = GlslShaderLoader().load(
            context(path, ResourceManager(), include_dirs=[shared])
        )
        assert "// lib" in asset.source

    def test_missing_include(self, tmp_path: Path) -> None:
        path = tmp_path / "main.frag"
        path.write_text('#include "gone.glsl"\n', encoding="utf-8")
        with pytest.raises(ResourceFormatError):
            GlslShaderLoader().load(context(path, ResourceManager()))

    def test_cyclic_include(self, tmp_path: Path) -> None:
        first = tmp_path / "a.glsl"
        second = tmp_path / "b.glsl"
        first.write_text('#include "b.glsl"\n', encoding="utf-8")
        second.write_text('#include "a.glsl"\n', encoding="utf-8")
        with pytest.raises(ResourceFormatError):
            GlslShaderLoader().load(context(first, ResourceManager()))

    def test_defines_are_prepended(self, tmp_path: Path) -> None:
        path = tmp_path / "main.frag"
        path.write_text("void main() {}\n", encoding="utf-8")
        asset = GlslShaderLoader().load(
            context(path, ResourceManager(), defines={"USE_SHADOW": 1})
        )
        assert asset.source.startswith("#define USE_SHADOW 1")

    def test_pragma_once_skips_repeat_include(self, tmp_path: Path) -> None:
        (tmp_path / "once.glsl").write_text(
            "#pragma once\nfloat value = 1.0;\n", encoding="utf-8"
        )
        path = tmp_path / "main.frag"
        path.write_text(
            '#include "once.glsl"\n#include "once.glsl"\n', encoding="utf-8"
        )
        asset = GlslShaderLoader().load(context(path, ResourceManager()))
        assert asset.source.count("float value = 1.0;") == 1
        assert len(asset.includes) == 1


class TestLoaderRegistryIntegration:
    def test_manager_loads_all_default_formats(
        self, manager: ResourceManager, tmp_path: Path
    ) -> None:
        obj_path = tmp_path / "cube.obj"
        obj_path.write_text(QUAD_OBJ, encoding="utf-8")
        png_path = tmp_path / "tex.png"
        _write_png(png_path, 4, 4)
        frag_path = tmp_path / "shader.frag"
        frag_path.write_text("void main() {}\n", encoding="utf-8")

        mesh = manager.load(obj_path)
        texture = manager.load(png_path, options={"decode": False})
        shader = manager.load(frag_path)

        assert mesh.asset_type.value == "mesh"
        assert texture.asset_type.value == "texture"
        assert shader.asset_type.value == "shader"
        assert manager.stats.loaded_total == 3

    def test_custom_loader_registration(self, tmp_path: Path) -> None:
        class MiscLoader(AssetLoader):
            extensions = (".misc",)

            def load(self, context: LoadContext):  # type: ignore[no-untyped-def]
                from engine.core.resources import Asset
                from engine.core.resources import AssetType as Kind

                return Asset(context.path, Kind.UNKNOWN)

        path = tmp_path / "data.misc"
        path.write_text("x", encoding="utf-8")
        manager = ResourceManager(loaders=[])
        manager.register_loader(MiscLoader())
        asset = manager.load(path)
        assert asset.asset_type.value == "unknown"


class TestAudioProbe:
    """Проверка заголовков WAV/OGG для импортёра аудио (без декодирования)."""

    fixtures = Path(__file__).parent / "fixtures"

    def test_wav_header(self) -> None:
        from engine.core.resources.importers import probe_audio

        info = probe_audio(self.fixtures / "tiny.wav")
        assert (info["format"], info["channels"], info["sample_rate"]) == ("wav", 1, 8000)
        assert info["duration"] == pytest.approx(0.1)

    def test_ogg_vorbis_header(self) -> None:
        from engine.core.resources.importers import probe_audio

        info = probe_audio(self.fixtures / "tiny.ogg")
        assert (info["codec"], info["channels"], info["sample_rate"]) == (
            "vorbis",
            2,
            44100,
        )
        assert info["duration"] == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "payload",
        [b"", b"RIFF\x00\x00\x00\x00WAVE", b"RIFF\x00\x00\x00\x00AVI ", b"OggS", b"MP3"],
    )
    def test_broken_audio_is_rejected(self, tmp_path: Path, payload: bytes) -> None:
        from engine.core.resources import AssetImportError
        from engine.core.resources.importers import probe_audio

        path = tmp_path / "bad.wav"
        path.write_bytes(payload)
        with pytest.raises(AssetImportError):
            probe_audio(path)
