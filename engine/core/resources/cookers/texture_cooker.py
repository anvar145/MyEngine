"""Texture cooker: декодирование, ограничение размера, мипы и BC-сжатие.

Cooker не знает про GL: на выходе байты ``texture.bin`` (см.
:mod:`~engine.core.resources.cookers.texture_format`), которые графический слой
загружает уровень за уровнем.

Пиксели берутся из собственного PNG-декодера; остальные форматы (JPEG, BMP, GIF,
чересстрочный PNG) читает Pillow, если он установлен. BC-сжатие выполняет
внешний CLI (``texconv``); если его нет, cooker пишет несжатую текстуру и
сообщает об этом в лог — импорт не падает.
"""

from __future__ import annotations

import logging
import shutil
import struct
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol

import numpy as np

from engine.core.resources.cookers.png_codec import SIGNATURE, decode_png, encode_png
from engine.core.resources.cookers.texture_format import (
    ALIGNMENT,
    FLAG_SRGB,
    HEADER_SIZE,
    MAGIC,
    MIP_ENTRY_SIZE,
    PIXEL_FORMATS,
    VERSION,
    mip_byte_size,
)
from engine.core.resources.errors import AssetImportError

__all__ = [
    "TextureCompressor",
    "TextureCookSettings",
    "TexconvCompressor",
    "cook_texture",
    "decode_image",
    "find_texconv",
    "mip_count",
]

_log = logging.getLogger("engine.core.resources.cook.texture")

COMPRESSION_MODES = ("none", "auto", "bc1", "bc3", "bc5")
_SRGB_THRESHOLD = 0.0031308
_UNIT = np.arange(256, dtype=np.float32) / 255
_TO_LINEAR = np.where(
    _UNIT <= 0.04045, _UNIT / 12.92, ((_UNIT + 0.055) / 1.055) ** 2.4
).astype(np.float32)


@dataclass(frozen=True, slots=True)
class TextureCookSettings:
    """Параметры cooker'а текстур.

    Attributes:
        srgb: Цветовая текстура (мипы усредняются в линейном пространстве).
            Для ``normal_map`` отключается.
        generate_mips: Строить цепочку мипов до 1x1.
        max_size: Ограничить большую сторону (0 — без ограничения); размер
            уменьшается вдвое, пока не уложится.
        compression: ``none``, ``auto`` или ``bc1``/``bc3``/``bc5``.
        normal_map: Карта нормалей: после усреднения нормали перенормируются,
            ``auto`` выбирает BC5.
    """

    srgb: bool = True
    generate_mips: bool = True
    max_size: int = 0
    compression: str = "none"
    normal_map: bool = False

    def __post_init__(self) -> None:
        if self.compression not in COMPRESSION_MODES:
            raise AssetImportError(
                f"compression: допустимы {', '.join(COMPRESSION_MODES)}, "
                f"получено '{self.compression}'"
            )
        if self.max_size < 0:
            raise AssetImportError("max_size не может быть отрицательным")

    @classmethod
    def from_mapping(cls, settings: Mapping[str, Any]) -> TextureCookSettings:
        """Собрать настройки из ``.meta`` (лишние ключи игнорируются).

        Raises:
            AssetImportError: Если значения некорректны.
        """
        try:
            normal_map = bool(settings.get("normal_map", False))
            return cls(
                srgb=bool(settings.get("srgb", True)) and not normal_map,
                generate_mips=bool(settings.get("generate_mips", True)),
                max_size=int(settings.get("max_size", 0)),
                compression=str(settings.get("compression", "none")),
                normal_map=normal_map,
            )
        except (TypeError, ValueError) as exc:
            raise AssetImportError(f"Некорректные настройки текстуры: {exc}") from exc


class TextureCompressor(Protocol):
    """Внешний BC-компрессор."""

    def compress(
        self, pixels: np.ndarray, pixel_format: str, levels: int, srgb: bool
    ) -> list[bytes]:
        """Сжать RGBA8 ``(H, W, 4)`` в BC-блоки.

        Returns:
            Блоки каждого из ``levels`` мип-уровней (мип 0 первым).

        Raises:
            AssetImportError: Если сжатие не удалось.
        """
        ...


def find_texconv() -> str | None:
    """Найти ``texconv`` в ``PATH`` (DirectXTex)."""
    return shutil.which("texconv")


class TexconvCompressor:
    """BC-сжатие через ``texconv`` (DirectXTex); мипы строит сам texconv."""

    _FORMATS: ClassVar[dict[str, str]] = {
        "bc1": "BC1_UNORM",
        "bc3": "BC3_UNORM",
        "bc5": "BC5_UNORM",
    }

    def __init__(self, executable: str, work_dir: Path | None = None) -> None:
        self.executable = executable
        self.work_dir = work_dir

    def compress(
        self, pixels: np.ndarray, pixel_format: str, levels: int, srgb: bool
    ) -> list[bytes]:
        """См. :class:`TextureCompressor`."""
        fmt = self._FORMATS[pixel_format] + ("_SRGB" if srgb else "")
        with tempfile.TemporaryDirectory(dir=self.work_dir, prefix="texconv_") as tmp:
            source = Path(tmp) / "in.png"
            source.write_bytes(encode_png(pixels))
            command = [self.executable, "-nologo", "-y", "-f", fmt, "-m", str(levels)]
            if srgb:
                command += ["-srgbi"]
            command += ["-o", tmp, str(source)]
            try:
                result = subprocess.run(  # noqa: S603 - путь к CLI задаёт пользователь
                    command, capture_output=True, text=True, timeout=300, check=False
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise AssetImportError(f"texconv не запустился: {exc}") from exc
            output = Path(tmp) / "in.dds"
            if result.returncode != 0 or not output.is_file():
                raise AssetImportError(
                    f"texconv завершился с ошибкой: {result.stdout[-200:]}"
                )
            return _split_dds(output.read_bytes(), pixel_format, pixels.shape, levels)


def decode_image(path: Path) -> np.ndarray:
    """Прочитать пиксели изображения как ``uint8`` ``(H, W, C)``, ``C`` в 1..4.

    PNG разбирается встроенным декодером; для остальных форматов и
    чересстрочных PNG нужен Pillow.

    Raises:
        AssetImportError: Если формат нельзя декодировать.
    """
    try:
        with path.open("rb") as stream:
            is_png = stream.read(len(SIGNATURE)) == SIGNATURE
    except OSError as exc:
        raise AssetImportError(f"Не удалось прочитать '{path}': {exc}") from exc
    if is_png:
        try:
            return decode_png(path)
        except AssetImportError as exc:
            if "Чересстроч" not in str(exc):
                raise
    return _decode_with_pillow(path)


def _decode_with_pillow(path: Path) -> np.ndarray:
    """Декодировать через Pillow (опциональная зависимость)."""
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:
        raise AssetImportError(
            f"'{path.name}': для этого формата нужен Pillow (pip install pillow)"
        ) from exc
    try:
        with Image.open(path) as image:
            has_alpha = "A" in image.getbands() or "transparency" in image.info
            mode = "RGBA" if has_alpha else "RGB"
            if image.mode in ("L", "LA"):
                mode = "LA" if has_alpha else "L"
            return np.asarray(image.convert(mode), dtype=np.uint8).reshape(
                image.height, image.width, -1
            )
    except Exception as exc:  # noqa: BLE001 - Pillow бросает разные исключения
        raise AssetImportError(f"Не удалось декодировать '{path.name}': {exc}") from exc


def mip_count(width: int, height: int) -> int:
    """Число уровней полной цепочки до 1x1."""
    return max(width, height).bit_length()


def cook_texture(
    pixels: np.ndarray,
    settings: TextureCookSettings | None = None,
    compressor: TextureCompressor | None = None,
    log: logging.Logger | None = None,
) -> bytes:
    """Приготовить ``texture.bin`` из пикселей.

    Args:
        pixels: ``uint8`` ``(H, W, C)``, ``C`` в 1..4.
        settings: Параметры cooker'а.
        compressor: BC-компрессор; без него сжатие пропускается.
        log: Логгер предупреждений.

    Raises:
        AssetImportError: При некорректных пикселях.
    """
    settings = settings or TextureCookSettings()
    log = log or _log
    image = _canonical(pixels)
    pixel_format = {1: "r8", 2: "rg8", 4: "rgba8"}[image.shape[2]]
    srgb = settings.srgb and pixel_format == "rgba8"

    while settings.max_size and max(image.shape[:2]) > settings.max_size:
        image = _halve(image, srgb, settings.normal_map)

    levels = mip_count(image.shape[1], image.shape[0]) if settings.generate_mips else 1
    chain = [image]
    for _ in range(levels - 1):
        chain.append(_halve(chain[-1], srgb, settings.normal_map))

    blocks, final_format = _try_compress(
        image, pixel_format, levels, srgb, settings, compressor, log
    )
    if blocks is None:
        blocks = [np.ascontiguousarray(level).tobytes() for level in chain]
        final_format = pixel_format
    sizes = [(level.shape[1], level.shape[0]) for level in chain]
    return _serialize(final_format, srgb and final_format != "bc5", sizes, blocks)


def _canonical(pixels: np.ndarray) -> np.ndarray:
    """Привести к r8 / rg8 / rgba8 (RGB дополняется непрозрачной альфой)."""
    array = np.asarray(pixels)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3 or array.dtype != np.uint8 or not 1 <= array.shape[2] <= 4:
        raise AssetImportError("Ожидался массив uint8 (H, W, C), C в 1..4")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise AssetImportError("Текстура нулевого размера")
    if array.shape[2] == 3:
        alpha = np.full((*array.shape[:2], 1), 255, dtype=np.uint8)
        array = np.concatenate([array, alpha], axis=2)
    return np.ascontiguousarray(array)


def _halve(image: np.ndarray, srgb: bool, normal_map: bool) -> np.ndarray:
    """Уменьшить вдвое усреднением 2x2 (цвет — в линейном пространстве)."""
    height, width, channels = image.shape
    new_h, new_w = max(1, height // 2), max(1, width // 2)
    ys = _taps(height, new_h)
    xs = _taps(width, new_w)
    quads = [image[y, x] for y in ys for x in xs]

    result = np.empty((new_h, new_w, channels), dtype=np.uint8)
    color = 3 if srgb else 0
    if color:
        linear = sum(_TO_LINEAR[q[:, :, :color]] for q in quads) / 4
        encoded = np.where(
            linear <= _SRGB_THRESHOLD,
            linear * 12.92,
            1.055 * np.power(linear, 1 / 2.4) - 0.055,
        )
        result[:, :, :color] = np.clip(np.rint(encoded * 255), 0, 255).astype(np.uint8)
    if color < channels:
        total = sum(q[:, :, color:].astype(np.uint16) for q in quads)
        result[:, :, color:] = ((total + 2) >> 2).astype(np.uint8)
    if normal_map and channels >= 2:
        _renormalize(result)
    return result


def _taps(size: int, new_size: int) -> tuple[slice, slice]:
    """Срезы чётных/нечётных отсчётов оси; у оси длины 1 оба — единственный."""
    if size == 1:
        return slice(0, 1), slice(0, 1)
    return slice(0, new_size * 2, 2), slice(1, new_size * 2, 2)


def _renormalize(image: np.ndarray) -> None:
    """Перенормировать векторы нормалей, закодированные в первых каналах."""
    channels = min(3, image.shape[2])
    vectors = image[:, :, :channels].astype(np.float32) / 127.5 - 1
    length = np.linalg.norm(vectors, axis=2, keepdims=True)
    limit = 1.0 if channels == 2 else 1e-6
    vectors = vectors / np.maximum(length, limit)
    encoded = np.rint((vectors + 1) * 127.5)
    image[:, :, :channels] = np.clip(encoded, 0, 255).astype(np.uint8)


def _try_compress(
    image: np.ndarray,
    pixel_format: str,
    levels: int,
    srgb: bool,
    settings: TextureCookSettings,
    compressor: TextureCompressor | None,
    log: logging.Logger,
) -> tuple[list[bytes] | None, str]:
    """Попробовать BC-сжатие; ``(None, ...)`` — остаться без сжатия."""
    if settings.compression == "none":
        return None, pixel_format
    if pixel_format != "rgba8":
        log.info("Сжатие пропущено: формат %s не сжимается в BC", pixel_format)
        return None, pixel_format
    target = settings.compression
    if target == "auto":
        if settings.normal_map:
            target = "bc5"
        else:
            target = "bc1" if bool((image[:, :, 3] == 255).all()) else "bc3"
    if compressor is None:
        log.warning(
            "BC-компрессор (texconv) не найден: текстура сохранена без сжатия "
            "(запрошено %s)",
            target,
        )
        return None, pixel_format
    try:
        blocks = compressor.compress(image, target, levels, srgb and target != "bc5")
    except AssetImportError as exc:
        log.warning("BC-сжатие не удалось, текстура без сжатия: %s", exc)
        return None, pixel_format
    return blocks, target


def _split_dds(
    data: bytes, pixel_format: str, shape: tuple[int, ...], levels: int
) -> list[bytes]:
    """Нарезать DDS от texconv на блоки мип-уровней."""
    if len(data) < 128 or not data.startswith(b"DDS "):
        raise AssetImportError("texconv вернул не DDS")
    offset = 128 + (20 if data[84:88] == b"DX10" else 0)
    dds_levels = max(1, struct.unpack_from("<I", data, 28)[0])
    if dds_levels != levels:
        raise AssetImportError(f"texconv дал {dds_levels} мипов вместо {levels}")
    width, height = shape[1], shape[0]
    blocks = []
    for _ in range(levels):
        size = mip_byte_size(pixel_format, width, height)
        chunk = data[offset : offset + size]
        if len(chunk) < size:
            raise AssetImportError("DDS от texconv обрезан")
        blocks.append(bytes(chunk))
        offset += size
        width, height = max(1, width // 2), max(1, height // 2)
    return blocks


def _serialize(
    pixel_format: str,
    srgb: bool,
    sizes: list[tuple[int, int]],
    blocks: list[bytes],
) -> bytes:
    """Собрать файл: заголовок, таблица мипов, выровненные данные."""
    table_end = HEADER_SIZE + len(blocks) * MIP_ENTRY_SIZE
    offset = -(-table_end // ALIGNMENT) * ALIGNMENT
    entries = bytearray()
    body = bytearray()
    for (width, height), block in zip(sizes, blocks, strict=True):
        if len(block) != mip_byte_size(pixel_format, width, height):
            raise AssetImportError("Размер мипа не совпадает с форматом")
        entries += struct.pack("<IIQQ", width, height, offset + len(body), len(block))
        body += block
        body += bytes(-len(body) % ALIGNMENT)
    header = struct.pack(
        "<4sIIIIIII",
        MAGIC,
        VERSION,
        sizes[0][0],
        sizes[0][1],
        len(blocks),
        PIXEL_FORMATS[pixel_format][0],
        FLAG_SRGB if srgb else 0,
        0,
    )
    padding = bytes(offset - table_end)
    return bytes(header + entries + padding + body)
