"""Бинарный формат кукованного меша ``mesh.bin`` и его чтение.

Раскладка (little-endian, все секции выровнены по 16 байт — файл пригоден для
``mmap`` и прямой загрузки в буфер GPU без перепаковки)::

    0   заголовок (64 байта): magic "MYMS", version, lod_count, flags,
        center[3] f32, half_extent[3] f32, резерв
    64  таблица LOD: по 48 байт на уровень — vertex_count, index_count,
        index_size (2|4), резерв, смещения секций pos/norm/uv/idx (u64, 0 — нет)
    ... данные секций

Секции вершин:

* позиции — ``int16 x4`` (xyz + 0), snorm: ``p = center + q / 32767 * half_extent``;
* нормали — ``int16 x4`` snorm (xyz + 0);
* UV — ``float32 x2`` (тайлинг за пределами [0, 1] не теряет точность);
* индексы — ``uint16`` или ``uint32`` (если вершин больше 65535).

LOD0 — полная детализация, следующие уровни проще.
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from engine.core.resources.errors import ResourceFormatError

__all__ = [
    "FLAG_NORMALS",
    "FLAG_UVS",
    "FORMAT_VERSION",
    "HEADER_SIZE",
    "LOD_ENTRY_SIZE",
    "MAGIC",
    "BufferLike",
    "CookedLod",
    "CookedMesh",
    "read_cooked_mesh",
]

MAGIC = b"MYMS"
FORMAT_VERSION = 1
HEADER_SIZE = 64
LOD_ENTRY_SIZE = 48
ALIGNMENT = 16
FLAG_NORMALS = 1
FLAG_UVS = 2

_HEADER = struct.Struct("<4sIII3f3f")
_LOD_ENTRY = struct.Struct("<IIIIQQQQ")
SNORM = 32767.0

#: Типы буфера, из которых можно читать без копирования.
BufferLike = bytes | bytearray | memoryview | mmap.mmap


@dataclass(frozen=True, slots=True)
class CookedLod:
    """Один уровень детализации (массивы — представления на исходный буфер).

    Attributes:
        positions_q: ``int16 (N, 4)`` — квантованные позиции.
        normals_q: ``int16 (N, 4)`` или ``None``.
        uvs: ``float32 (N, 2)`` или ``None``.
        indices: ``uint16``/``uint32 (M,)``; пусто для облака точек.
    """

    positions_q: np.ndarray
    normals_q: np.ndarray | None
    uvs: np.ndarray | None
    indices: np.ndarray

    @property
    def vertex_count(self) -> int:
        """Число вершин."""
        return int(self.positions_q.shape[0])

    @property
    def triangle_count(self) -> int:
        """Число треугольников."""
        return int(self.indices.size // 3)


@dataclass(frozen=True, slots=True)
class CookedMesh:
    """Разобранный ``mesh.bin``.

    Attributes:
        center: Центр ограничивающего параллелепипеда (квантование позиций).
        half_extent: Половинные размеры параллелепипеда.
        lods: Уровни детализации от самого детального.
    """

    center: np.ndarray
    half_extent: np.ndarray
    lods: tuple[CookedLod, ...]

    def positions(self, lod: int = 0) -> np.ndarray:
        """Позиции уровня в ``float32 (N, 3)``."""
        quantized = self.lods[lod].positions_q[:, :3].astype(np.float32)
        return self.center + quantized * (self.half_extent / np.float32(SNORM))

    def normals(self, lod: int = 0) -> np.ndarray | None:
        """Нормали уровня в ``float32 (N, 3)`` (или ``None``)."""
        quantized = self.lods[lod].normals_q
        if quantized is None:
            return None
        return quantized[:, :3].astype(np.float32) / np.float32(SNORM)


def read_cooked_mesh(
    source: BufferLike | Path | str | object, guid: str | None = None
) -> CookedMesh:
    """Разобрать ``mesh.bin`` из буфера (в т. ч. ``mmap``), файла или VFS.

    Массивы результата ссылаются на переданный буфер без копирования.

    Args:
        source: Буфер, путь к файлу или VFS (тогда нужен ``guid``: артефакт
            ``mesh`` читается через ``vfs.resolve(guid, "mesh")``).
        guid: GUID ассета при чтении из VFS.

    Raises:
        ResourceFormatError: Если сигнатура, версия или границы секций неверны.
    """
    if guid is not None and hasattr(source, "list_artifacts"):
        source = source.resolve(guid, "mesh")  # type: ignore[attr-defined]
        if source is None:
            raise ResourceFormatError(f"У ассета {guid} нет артефакта 'mesh'")
    if isinstance(source, (str, Path)):
        try:
            data: BufferLike = Path(source).read_bytes()
        except OSError as exc:
            raise ResourceFormatError(
                f"Не удалось прочитать '{source}': {exc}"
            ) from exc
    else:
        data = source
    view = memoryview(data).cast("B")
    if len(view) < HEADER_SIZE:
        raise ResourceFormatError("mesh.bin: файл короче заголовка")
    magic, version, lod_count, flags, *extent = _HEADER.unpack_from(view, 0)
    if magic != MAGIC:
        raise ResourceFormatError("mesh.bin: неверная сигнатура")
    if version != FORMAT_VERSION:
        raise ResourceFormatError(f"mesh.bin: версия {version} не поддержана")
    if lod_count == 0 or HEADER_SIZE + lod_count * LOD_ENTRY_SIZE > len(view):
        raise ResourceFormatError("mesh.bin: повреждена таблица LOD")

    lods = []
    for index in range(lod_count):
        entry = _LOD_ENTRY.unpack_from(view, HEADER_SIZE + index * LOD_ENTRY_SIZE)
        vertices, indices, index_size, _reserved, pos, norm, uv, idx = entry
        if index_size not in (2, 4):
            raise ResourceFormatError(
                f"mesh.bin: LOD{index}: размер индекса {index_size}"
            )
        lods.append(
            CookedLod(
                _section(data, view, pos, "<i2", vertices, 4, index, "позиции"),
                _section(data, view, norm, "<i2", vertices, 4, index, "нормали")
                if flags & FLAG_NORMALS
                else None,
                _section(data, view, uv, "<f4", vertices, 2, index, "UV")
                if flags & FLAG_UVS
                else None,
                _section(
                    data,
                    view,
                    idx,
                    "<u2" if index_size == 2 else "<u4",
                    indices,
                    1,
                    index,
                    "индексы",
                ).reshape(-1),
            )
        )
    return CookedMesh(
        np.asarray(extent[:3], dtype=np.float32),
        np.asarray(extent[3:], dtype=np.float32),
        tuple(lods),
    )


def _section(
    data: BufferLike,
    view: memoryview,
    offset: int,
    dtype: str,
    count: int,
    width: int,
    lod: int,
    what: str,
) -> np.ndarray:
    """Представление секции на буфер с проверкой границ."""
    item = np.dtype(dtype).itemsize
    if count == 0:
        return np.empty((0, width) if width > 1 else (0,), dtype=dtype)
    if offset < HEADER_SIZE or offset + count * width * item > len(view):
        raise ResourceFormatError(f"mesh.bin: LOD{lod}: секция '{what}' вне файла")
    array = np.frombuffer(data, dtype=dtype, count=count * width, offset=offset)
    return array.reshape(count, width) if width > 1 else array
