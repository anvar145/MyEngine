"""Тесты texture cooker: PNG-декодер, мипы, sRGB, формат texture.bin, BC-сжатие."""

from __future__ import annotations

import logging
import struct
import zlib

import numpy as np
import pytest

from engine.core.resources import AssetDatabase, AssetImportError, ResourceFormatError
from engine.core.resources.cookers import (
    TextureCookSettings,
    cook_texture,
    decode_image,
    decode_png,
    encode_png,
    mip_count,
    read_cooked_texture,
)
from engine.core.resources.cookers.texture_format import HEADER_SIZE, mip_byte_size


def make_png(
    values: np.ndarray,
    color_type: int,
    depth: int = 8,
    filter_type: int = 0,
    palette: bytes | None = None,
    trns: bytes | None = None,
    interlace: int = 0,
) -> bytes:
    """Собрать PNG вручную: произвольный тип/глубина и выбранный фильтр строки.

    ``values`` — отсчёты ``(H, W, samples)``, уже в целевой глубине.
    """
    height, width, samples = values.shape
    bits = samples * depth
    stride = (width * bits + 7) // 8
    bpp = max(1, bits // 8)
    if depth == 16:
        raw_rows = np.stack([(values >> 8), (values & 255)], axis=3).astype(np.uint8)
        raw_rows = raw_rows.reshape(height, -1)
    elif depth == 8:
        raw_rows = values.reshape(height, -1).astype(np.uint8)
    else:
        flat = values.reshape(height, -1).astype(np.uint8)
        pad = stride * 8 // depth - flat.shape[1]
        flat = np.pad(flat, ((0, 0), (0, pad)))
        shifts = np.arange(depth - 1, -1, -1)
        bitmap = ((flat[:, :, None] >> shifts) & 1).astype(np.uint8).reshape(height, -1)
        raw_rows = np.packbits(bitmap, axis=1)
    out = bytearray()
    previous = np.zeros(stride, dtype=np.int32)
    for y in range(height):
        row = raw_rows[y].astype(np.int32)
        left = np.concatenate([np.zeros(bpp, dtype=np.int32), row[:-bpp]])
        up_left = np.concatenate([np.zeros(bpp, dtype=np.int32), previous[:-bpp]])
        if filter_type == 0:
            filtered = row
        elif filter_type == 1:
            filtered = row - left
        elif filter_type == 2:
            filtered = row - previous
        elif filter_type == 3:
            filtered = row - ((left + previous) >> 1)
        else:
            p = left + previous - up_left
            pa, pb, pc = abs(p - left), abs(p - previous), abs(p - up_left)
            pred = np.where((pa <= pb) & (pa <= pc), left, np.where(pb <= pc, previous, up_left))
            filtered = row - pred
        out.append(filter_type)
        out += (filtered & 255).astype(np.uint8).tobytes()
        previous = row

    def chunk(kind: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(kind + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)

    data = b"\x89PNG\r\n\x1a\n"
    data += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, color_type, 0, 0, interlace))
    if palette:
        data += chunk(b"PLTE", palette)
    if trns:
        data += chunk(b"tRNS", trns)
    data += chunk(b"IDAT", zlib.compress(bytes(out)))
    return data + chunk(b"IEND", b"")


@pytest.fixture(scope="module")
def noise() -> np.ndarray:
    return np.random.default_rng(7).integers(0, 256, (5, 7, 4), dtype=np.uint8)


class TestPngDecoder:
    @pytest.mark.parametrize("filter_type", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize(("color_type", "samples"), [(0, 1), (2, 3), (4, 2), (6, 4)])
    def test_all_filters_and_color_types(self, noise, filter_type, color_type, samples) -> None:
        values = noise[:, :, :samples]
        decoded = decode_png(make_png(values, color_type, filter_type=filter_type))
        assert np.array_equal(decoded, values)

    def test_encoder_roundtrip(self, noise) -> None:
        assert np.array_equal(decode_png(encode_png(noise)), noise)
        assert np.array_equal(decode_png(encode_png(noise[:, :, :3])), noise[:, :, :3])

    def test_16_bit_keeps_high_byte(self) -> None:
        values = np.array([[[0x1234, 0xFF00], [0x00FF, 0x8000]]], dtype=np.uint16)
        decoded = decode_png(make_png(values.reshape(1, 2, 1, 2).reshape(1, 2, 2), 4, 16))
        assert decoded.tolist() == [[[0x12, 0xFF], [0x00, 0x80]]]

    def test_gray_low_depth_is_scaled(self) -> None:
        values = np.array([[[0], [1], [2], [3], [3]]], dtype=np.uint8)
        decoded = decode_png(make_png(values, 0, depth=2))
        assert decoded[:, :, 0].tolist() == [[0, 85, 170, 255, 255]]

    def test_palette_with_transparency(self) -> None:
        indices = np.array([[[0], [1], [2]]], dtype=np.uint8)
        palette = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255])
        png = make_png(indices, 3, depth=8, palette=palette, trns=bytes([255, 128]))
        decoded = decode_png(png)
        assert decoded.tolist() == [
            [[255, 0, 0, 255], [0, 255, 0, 128], [0, 0, 255, 255]]
        ]

    def test_packed_palette_without_transparency(self) -> None:
        indices = np.array([[[0], [1], [1], [0], [1]]], dtype=np.uint8)
        palette = bytes([10, 20, 30, 200, 210, 220])
        decoded = decode_png(make_png(indices, 3, depth=1, palette=palette))
        assert decoded.shape == (1, 5, 3)
        assert decoded[0, 1].tolist() == [200, 210, 220]
        assert decoded[0, 3].tolist() == [10, 20, 30]

    def test_colour_key_transparency(self) -> None:
        values = np.array([[[10, 20, 30], [1, 2, 3]]], dtype=np.uint8)
        key = struct.pack(">HHH", 1, 2, 3)
        decoded = decode_png(make_png(values, 2, trns=key))
        assert decoded[:, :, 3].tolist() == [[255, 0]]

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda b: b"nope" + b[4:], "сигнатура"),
            (lambda b: b[:40], "нет данных|обрезан"),
            (lambda b: b[:8], "IHDR"),
        ],
    )
    def test_corrupted_png_is_rejected(self, noise, mutate, message) -> None:
        with pytest.raises(AssetImportError, match=message):
            decode_png(mutate(encode_png(noise)))

    def test_interlaced_is_rejected_clearly(self, noise) -> None:
        with pytest.raises(AssetImportError, match="Чересстроч"):
            decode_png(make_png(noise, 6, interlace=1))

    def test_broken_deflate_stream(self, noise) -> None:
        png = bytearray(encode_png(noise))
        index = png.index(b"IDAT") + 8
        png[index : index + 4] = b"\xff\xff\xff\xff"
        with pytest.raises(AssetImportError, match="Повреждены"):
            decode_png(bytes(png))


class TestMips:
    def test_full_chain_to_1x1(self, noise) -> None:
        cooked = read_cooked_texture(cook_texture(noise))
        assert [(m.width, m.height) for m in cooked.mips] == [(7, 5), (3, 2), (1, 1)]
        assert len(cooked.mips) == mip_count(7, 5) == 3

    def test_mips_disabled(self, noise) -> None:
        cooked = read_cooked_texture(cook_texture(noise, TextureCookSettings(generate_mips=False)))
        assert len(cooked.mips) == 1

    def test_non_square_chain_clamps_to_one(self) -> None:
        pixels = np.zeros((1, 16, 4), dtype=np.uint8)
        sizes = [(m.width, m.height) for m in read_cooked_texture(cook_texture(pixels)).mips]
        assert sizes == [(16, 1), (8, 1), (4, 1), (2, 1), (1, 1)]

    def test_box_filter_average_in_linear_alpha_plane(self) -> None:
        pixels = np.zeros((2, 2, 4), dtype=np.uint8)
        pixels[:, :, 3] = [[0, 100], [200, 255]]
        mip = read_cooked_texture(cook_texture(pixels)).pixels(1)
        assert mip[0, 0, 3] == (0 + 100 + 200 + 255 + 2) >> 2

    def test_srgb_average_is_brighter_than_naive(self) -> None:
        pixels = np.zeros((2, 2, 4), dtype=np.uint8)
        pixels[:, :, 3] = 255
        pixels[0, 0, :3] = 255
        pixels[1, 1, :3] = 255
        srgb = read_cooked_texture(cook_texture(pixels)).pixels(1)[0, 0, 0]
        linear = read_cooked_texture(
            cook_texture(pixels, TextureCookSettings(srgb=False))
        ).pixels(1)[0, 0, 0]
        assert linear == 128
        assert 185 <= srgb <= 190  # sRGB(0.5 линейного) ≈ 188

    def test_srgb_flag_follows_settings(self, noise) -> None:
        assert read_cooked_texture(cook_texture(noise)).srgb
        assert not read_cooked_texture(cook_texture(noise, TextureCookSettings(srgb=False))).srgb

    def test_constant_color_survives_all_levels(self) -> None:
        pixels = np.full((8, 8, 4), 77, dtype=np.uint8)
        for level in range(4):
            assert (read_cooked_texture(cook_texture(pixels)).pixels(level) == 77).all()

    def test_normal_map_mips_stay_unit_length(self) -> None:
        rng = np.random.default_rng(3)
        vectors = rng.normal(size=(16, 16, 3)) + np.array([0, 0, 3])
        vectors /= np.linalg.norm(vectors, axis=2, keepdims=True)
        pixels = np.rint((vectors + 1) * 127.5).astype(np.uint8)
        settings = TextureCookSettings.from_mapping({"normal_map": True})
        assert not settings.srgb
        cooked = read_cooked_texture(cook_texture(pixels, settings))
        mip = cooked.pixels(2)[:, :, :3].astype(np.float32) / 127.5 - 1
        assert np.abs(np.linalg.norm(mip, axis=2) - 1).max() < 0.02

    def test_max_size_downscales(self) -> None:
        pixels = np.zeros((64, 32, 4), dtype=np.uint8)
        cooked = read_cooked_texture(cook_texture(pixels, TextureCookSettings(max_size=16)))
        assert (cooked.width, cooked.height) == (8, 16)
        assert len(cooked.mips) == 5

    @pytest.mark.parametrize("channels", [1, 2, 3, 4])
    def test_channel_layouts(self, channels) -> None:
        pixels = np.full((4, 4, channels), 9, dtype=np.uint8)
        cooked = read_cooked_texture(cook_texture(pixels))
        expected = {1: "r8", 2: "rg8", 3: "rgba8", 4: "rgba8"}[channels]
        assert cooked.pixel_format == expected
        if channels == 3:
            assert (cooked.pixels(0)[:, :, 3] == 255).all()

    def test_bad_pixel_arrays(self) -> None:
        with pytest.raises(AssetImportError):
            cook_texture(np.zeros((2, 2, 4), dtype=np.float32))
        with pytest.raises(AssetImportError):
            cook_texture(np.zeros((0, 2, 4), dtype=np.uint8))


class TestSettings:
    def test_defaults_and_mapping(self) -> None:
        assert TextureCookSettings.from_mapping({}) == TextureCookSettings()
        assert TextureCookSettings.from_mapping({"max_size": "256"}).max_size == 256

    @pytest.mark.parametrize(
        "bad", [{"compression": "zip"}, {"max_size": -1}, {"max_size": "big"}]
    )
    def test_invalid_values(self, bad) -> None:
        with pytest.raises(AssetImportError):
            TextureCookSettings.from_mapping(bad)


class TestFormat:
    def test_sections_are_aligned_and_zero_copy(self, noise) -> None:
        blob = bytearray(cook_texture(noise))
        cooked = read_cooked_texture(memoryview(blob))
        for index in range(len(cooked.mips)):
            offset = struct.unpack_from("<Q", blob, HEADER_SIZE + index * 24 + 8)[0]
            assert offset % 16 == 0
        before = int(cooked.pixels(0)[0, 0, 0])
        blob[struct.unpack_from("<Q", blob, HEADER_SIZE + 8)[0]] = before ^ 1
        assert cooked.pixels(0)[0, 0, 0] == before ^ 1

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda b: b"XXXX" + b[4:], "сигнатура"),
            (lambda b: b[:4] + struct.pack("<I", 9) + b[8:], "версия"),
            (lambda b: b[:20], "короче"),
            (lambda b: b[:-40], "вне файла"),
            (lambda b: b[:20] + struct.pack("<I", 77) + b[24:], "формат"),
            (lambda b: b[:16] + struct.pack("<I", 0) + b[20:], "мипов"),
        ],
    )
    def test_corrupted_files_are_rejected(self, noise, mutate, message) -> None:
        with pytest.raises(ResourceFormatError, match=message):
            read_cooked_texture(mutate(cook_texture(noise)))

    def test_mip_sizes(self) -> None:
        assert mip_byte_size("rgba8", 3, 2) == 24
        assert mip_byte_size("bc1", 1, 1) == 8
        assert mip_byte_size("bc3", 5, 5) == 64


class FakeCompressor:
    """Подставной компрессор: возвращает блоки правильного размера."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, int, bool]] = []
        self.fail = fail

    def compress(self, pixels, pixel_format, levels, srgb):
        self.calls.append((pixel_format, levels, srgb))
        if self.fail:
            raise AssetImportError("сломан")
        width, height = pixels.shape[1], pixels.shape[0]
        blocks = []
        for _ in range(levels):
            blocks.append(bytes(mip_byte_size(pixel_format, width, height)))
            width, height = max(1, width // 2), max(1, height // 2)
        return blocks


class TestCompression:
    def test_auto_picks_bc1_for_opaque_and_bc3_for_alpha(self, noise) -> None:
        opaque = noise.copy()
        opaque[:, :, 3] = 255
        auto = TextureCookSettings(compression="auto")
        assert read_cooked_texture(cook_texture(opaque, auto, FakeCompressor())).pixel_format == "bc1"
        assert read_cooked_texture(cook_texture(noise, auto, FakeCompressor())).pixel_format == "bc3"

    def test_auto_picks_bc5_for_normal_maps(self, noise) -> None:
        settings = TextureCookSettings.from_mapping({"compression": "auto", "normal_map": True})
        compressor = FakeCompressor()
        cooked = read_cooked_texture(cook_texture(noise, settings, compressor))
        assert cooked.pixel_format == "bc5" and cooked.compressed
        assert compressor.calls == [("bc5", 3, False)]

    def test_compressed_texture_has_block_sized_mips(self, noise) -> None:
        cooked = read_cooked_texture(
            cook_texture(noise, TextureCookSettings(compression="bc3"), FakeCompressor())
        )
        assert [len(m.data) for m in cooked.mips] == [64, 16, 16]
        with pytest.raises(ResourceFormatError):
            cooked.pixels(0)

    def test_missing_compressor_falls_back_with_warning(self, noise, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            cooked = read_cooked_texture(
                cook_texture(noise, TextureCookSettings(compression="bc1"), None)
            )
        assert cooked.pixel_format == "rgba8"
        assert "texconv" in caplog.text

    def test_failing_compressor_falls_back(self, noise, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            cooked = read_cooked_texture(
                cook_texture(noise, TextureCookSettings(compression="bc1"), FakeCompressor(True))
            )
        assert cooked.pixel_format == "rgba8"
        assert "не удалось" in caplog.text

    def test_single_channel_is_not_compressed(self) -> None:
        compressor = FakeCompressor()
        pixels = np.zeros((4, 4, 1), dtype=np.uint8)
        cooked = read_cooked_texture(
            cook_texture(pixels, TextureCookSettings(compression="bc1"), compressor)
        )
        assert cooked.pixel_format == "r8" and not compressor.calls

    def test_srgb_is_passed_to_compressor(self, noise) -> None:
        compressor = FakeCompressor()
        cook_texture(noise, TextureCookSettings(compression="bc3"), compressor)
        assert compressor.calls[0][2] is True


class TestDecodeImage:
    def test_png_goes_through_builtin_decoder(self, tmp_path, noise) -> None:
        path = tmp_path / "a.png"
        path.write_bytes(encode_png(noise))
        assert np.array_equal(decode_image(path), noise)

    def test_non_png_without_pillow_explains_what_to_install(self, tmp_path) -> None:
        try:
            import PIL  # noqa: F401
        except ImportError:
            path = tmp_path / "a.bmp"
            path.write_bytes(b"BM" + bytes(60))
            with pytest.raises(AssetImportError, match="Pillow"):
                decode_image(path)
        else:
            pytest.skip("Pillow установлен")


class TestImporterIntegration:
    def _database(self, tmp_path):
        return AssetDatabase(tmp_path / "proj")

    def test_png_imports_to_cooked_texture_and_settings_trigger_reimport(
        self, tmp_path, noise
    ) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        (project / "a.png").write_bytes(encode_png(noise))
        db = self._database(tmp_path)
        try:
            db.scan()
            db.reimport_dirty(wait=True)
            handle = db.resolve(db.guid_of(project / "a.png"))
            info = handle.artifact("texture").info
            assert info["cooked"] and info["mip_levels"] == 3 and info["pixel_format"] == "rgba8"
            cooked = read_cooked_texture(handle.artifact_path("texture").read_bytes())
            assert np.array_equal(cooked.pixels(0), noise)

            guid = db.guid_of(project / "a.png")
            db.set_settings(guid, {"max_size": 4, "generate_mips": False})
            assert guid in db.dirty()
            db.reimport(guid, wait=True)
            info = db.resolve(guid).artifact("texture").info
            assert (info["width"], info["height"], info["mip_levels"]) == (3, 2, 1)
            assert info["source_width"] == 7
        finally:
            db.close()

    def test_undecodable_format_keeps_raw_copy(self, tmp_path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        bmp = bytearray(b"BM" + bytes(52))
        struct.pack_into("<iiHH", bmp, 18, 2, 2, 1, 24)
        (project / "a.bmp").write_bytes(bytes(bmp))
        db = self._database(tmp_path)
        try:
            db.scan()
            db.reimport_dirty(wait=True)
            try:
                import PIL  # noqa: F401
            except ImportError:
                info = db.resolve(db.guid_of(project / "a.bmp")).artifact("texture").info
                assert info["cooked"] is False and info["width"] == 2
        finally:
            db.close()
