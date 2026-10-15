"""Бинарный формат ``texture.bin`` (MYTX v1) и его чтение без копирования.

Раскладка (little-endian)::

    заголовок 32 байта: magic 'MYTX', version, width, height, mips, format, flags, 0
    таблица мипов: на каждый уровень width u32, height u32, offset u64, size u64
    данные мипов, каждый уровень выровнен по 16 байт

Уровни идут от самого крупного (мип 0) к самому мелкому.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from engine.core.resources.errors import ResourceFormatError

__all__ = [
    "ALIGNMENT",
    "BufferLike",
    "CookedMip",
    "CookedTexture",
    "FLAG_SRGB",
    "HEADER_SIZE",
    "MAGIC",
    "MIP_ENTRY_SIZE",
    "PIXEL_FORMATS",
    "VERSION",
    "mip_byte_size",
    "read_cooked_texture",
]

MAGIC = b"MYTX"
VERSION = 1
HEADER_SIZE = 32
MIP_ENTRY_SIZE = 24
ALIGNMENT = 16
FLAG_SRGB = 1

BufferLike = bytes | bytearray | memoryview

# имя → (код, байт на пиксель для линейных / байт на блок 4x4 для BC, каналов или 0)
PIXEL_FORMATS: dict[str, tuple[int, int, int]] = {
    "r8": (1, 1, 1),
    "rg8": (2, 2, 2),
    "rgba8": (3, 4, 4),
    "bc1": (4, 8, 0),
    "bc3": (5, 16, 0),
    "bc5": (6, 16, 0),
}
_NAME_BY_CODE = {code: name for name, (code, _, _) in PIXEL_FORMATS.items()}


def mip_byte_size(pixel_format: str, width: int, height: int) -> int:
    """Размер одного мип-уровня в байтах."""
    _, unit, channels = PIXEL_FORMATS[pixel_format]
    if channels:
        return width * height * unit
    return ((width + 3) // 4) * ((height + 3) // 4) * unit


@dataclass(frozen=True, slots=True)
class CookedMip:
    """Один мип-уровень: размеры и представление данных в буфере файла."""

    width: int
    height: int
    data: memoryview


@dataclass(frozen=True, slots=True)
class CookedTexture:
    """Разобранный ``texture.bin``. Данные мипов ссылаются на исходный буфер."""

    width: int
    height: int
    pixel_format: str
    srgb: bool
    mips: tuple[CookedMip, ...]

    @property
    def compressed(self) -> bool:
        """Блочное (BC) сжатие."""
        return PIXEL_FORMATS[self.pixel_format][2] == 0

    def pixels(self, level: int = 0) -> np.ndarray:
        """Пиксели несжатого уровня как ``(H, W, C)`` без копирования.

        Raises:
            ResourceFormatError: Для BC-форматов (их блоки отдаются на GPU как есть).
        """
        if self.compressed:
            raise ResourceFormatError("Сжатая текстура не раскрывается в пиксели")
        mip = self.mips[level]
        channels = PIXEL_FORMATS[self.pixel_format][2]
        array = np.frombuffer(mip.data, dtype=np.uint8)
        return array.reshape(mip.height, mip.width, channels)


def read_cooked_texture(
    buffer: BufferLike | object, guid: str | None = None
) -> CookedTexture:
    """Разобрать ``texture.bin``, не копируя данные мипов.

    Args:
        buffer: Буфер или VFS (тогда нужен ``guid``, читается артефакт
            ``texture`` через ``vfs.resolve(guid, "texture")``).
        guid: GUID ассета при чтении из VFS.

    Raises:
        ResourceFormatError: При неверной сигнатуре/версии или выходе за файл.
    """
    if guid is not None and hasattr(buffer, "list_artifacts"):
        path = buffer.resolve(guid, "texture")  # type: ignore[attr-defined]
        if path is None:
            raise ResourceFormatError(f"У ассета {guid} нет артефакта 'texture'")
        buffer = path.read_bytes()
    view = memoryview(buffer)  # type: ignore[arg-type]
    if view.nbytes < HEADER_SIZE:
        raise ResourceFormatError("texture.bin короче заголовка")
    magic, version, width, height, mips, code, flags, _ = struct.unpack_from(
        "<4sIIIIIII", view, 0
    )
    if magic != MAGIC:
        raise ResourceFormatError("texture.bin: неверная сигнатура")
    if version != VERSION:
        raise ResourceFormatError(f"texture.bin: неподдерживаемая версия {version}")
    pixel_format = _NAME_BY_CODE.get(code)
    if pixel_format is None:
        raise ResourceFormatError(f"texture.bin: неизвестный формат {code}")
    if mips == 0 or width == 0 or height == 0:
        raise ResourceFormatError("texture.bin: пустая таблица мипов")
    if view.nbytes < HEADER_SIZE + mips * MIP_ENTRY_SIZE:
        raise ResourceFormatError("texture.bin: таблица мипов вне файла")

    levels = []
    for index in range(mips):
        w, h, offset, size = struct.unpack_from(
            "<IIQQ", view, HEADER_SIZE + index * MIP_ENTRY_SIZE
        )
        if offset + size > view.nbytes:
            raise ResourceFormatError(f"texture.bin: мип {index} вне файла")
        if size != mip_byte_size(pixel_format, w, h):
            raise ResourceFormatError(f"texture.bin: неверный размер мипа {index}")
        levels.append(CookedMip(w, h, view[offset : offset + size]))
    srgb = bool(flags & FLAG_SRGB)
    return CookedTexture(width, height, pixel_format, srgb, tuple(levels))
