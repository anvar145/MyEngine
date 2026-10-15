"""Загруженные ресурсы: базовый класс и конкретные типы ассетов.

Ассет — это **данные**, а не файл: он уже прочитан, разобран и готов к
использованию (для меша — numpy-массивы вершин, для текстуры — метаданные
изображения, для шейдера — исходник со вставленными ``#include``).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from engine.core.math3d import Vector3

__all__ = [
    "Asset",
    "AssetType",
    "MeshAsset",
    "ShaderAsset",
    "ShaderStage",
    "TextureAsset",
]


class AssetType(str, Enum):
    """Категория ресурса (используется для статистики и поиска загрузчика)."""

    MESH = "mesh"
    TEXTURE = "texture"
    SHADER = "shader"
    AUDIO = "audio"
    TEXT = "text"
    UNKNOWN = "unknown"


class Asset:
    """Базовый загруженный ресурс.

    Args:
        path: Путь к исходному файлу.
        asset_type: Категория ресурса.
        name: Отображаемое имя; по умолчанию — имя файла.
    """

    __slots__ = ("_ref_count", "asset_type", "name", "path")

    def __init__(
        self, path: Path, asset_type: AssetType, name: str | None = None
    ) -> None:
        self.path = Path(path)
        self.asset_type = asset_type
        self.name = name if name is not None else self.path.stem
        self._ref_count = 0

    # ------------------------------------------------------------------ #
    # Учёт ссылок
    # ------------------------------------------------------------------ #
    @property
    def ref_count(self) -> int:
        """Сколько раз ресурс выдан вызывающему коду (для выгрузки)."""
        return self._ref_count

    @property
    def is_used(self) -> bool:
        """``True``, если ресурс ещё кто-то держит."""
        return self._ref_count > 0

    def _add_ref(self) -> int:
        """Увеличить счётчик ссылок (вызывает менеджер ресурсов)."""
        self._ref_count += 1
        return self._ref_count

    def _remove_ref(self) -> int:
        """Уменьшить счётчик ссылок (вызывает менеджер ресурсов)."""
        self._ref_count = max(0, self._ref_count - 1)
        return self._ref_count

    # ------------------------------------------------------------------ #
    # Данные
    # ------------------------------------------------------------------ #
    @property
    def estimated_bytes(self) -> int:
        """Приблизительный объём данных в памяти (для бюджета ресурсов)."""
        return 0

    def release(self) -> None:
        """Освободить нативные данные ресурса (GL-буферы, файловые дескрипторы).

        Вызывается менеджером перед удалением ресурса из кэша. Реализации,
        владеющие внешними ресурсами (GPU), должны переопределить метод и
        освободить их строго в контексте рендера.
        """

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} '{self.name}' refs={self._ref_count} "
            f"bytes={self.estimated_bytes}>"
        )


class MeshAsset(Asset):
    """Меш: вершинные атрибуты и индексы в виде numpy-массивов.

    Args:
        path: Путь к исходному файлу.
        positions: Массив позиций ``(N, 3)``.
        indices: Массив индексов ``(M,)``; ``None`` — рисуем без индексов.
        normals: Нормали ``(N, 3)``, если есть в файле.
        uvs: Текстурные координаты ``(N, 2)``, если есть в файле.
        name: Имя меша.
    """

    __slots__ = ("indices", "normals", "positions", "uvs")

    def __init__(
        self,
        path: Path,
        positions: np.ndarray,
        indices: np.ndarray | None = None,
        normals: np.ndarray | None = None,
        uvs: np.ndarray | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(path, AssetType.MESH, name)
        self.positions = np.ascontiguousarray(positions, dtype=np.float32)
        self.normals = (
            None if normals is None else np.ascontiguousarray(normals, dtype=np.float32)
        )
        self.uvs = None if uvs is None else np.ascontiguousarray(uvs, dtype=np.float32)
        self.indices = (
            None if indices is None else np.ascontiguousarray(indices, dtype=np.uint32)
        )

    @property
    def vertex_count(self) -> int:
        """Число вершин."""
        return int(self.positions.shape[0])

    @property
    def triangle_count(self) -> int:
        """Число треугольников."""
        if self.indices is not None:
            return int(self.indices.size // 3)
        return self.vertex_count // 3

    @property
    def bounds(self) -> tuple[Vector3, Vector3]:
        """Ограничивающий параллелепипед ``(min, max)``."""
        if self.vertex_count == 0:
            return Vector3.zero(), Vector3.zero()
        minimum = self.positions.min(axis=0)
        maximum = self.positions.max(axis=0)
        return (
            Vector3(float(minimum[0]), float(minimum[1]), float(minimum[2])),
            Vector3(float(maximum[0]), float(maximum[1]), float(maximum[2])),
        )

    @property
    def estimated_bytes(self) -> int:
        """Объём данных: позиции, нормали, UV и индексы."""
        total = int(self.positions.nbytes)
        if self.normals is not None:
            total += int(self.normals.nbytes)
        if self.uvs is not None:
            total += int(self.uvs.nbytes)
        if self.indices is not None:
            total += int(self.indices.nbytes)
        return total


class TextureAsset(Asset):
    """Текстура: метаданные изображения и (опционально) пиксели.

    Пиксели могут отсутствовать: для DDS/BC-форматов распаковка не нужна —
    данные загружаются на GPU как есть, поэтому ``pixels`` остаётся ``None``.

    Args:
        path: Путь к файлу изображения.
        width: Ширина в пикселях.
        height: Высота в пикселях.
        channels: Число каналов (1, 2, 3 или 4).
        pixel_format: Строковое описание формата (``"rgba8"``, ``"bc1"``, ...).
        bit_depth: Разрядность одного канала.
        mip_levels: Число уровней мип-текстуры.
        compressed: ``True`` для блочно-сжатых форматов (DDS/BC).
        pixels: Декодированные пиксели ``(H, W, C)``, если доступны.
    """

    __slots__ = (
        "bit_depth",
        "channels",
        "compressed",
        "height",
        "mip_levels",
        "pixel_format",
        "pixels",
        "width",
    )

    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        *,
        channels: int = 4,
        pixel_format: str = "rgba8",
        bit_depth: int = 8,
        mip_levels: int = 1,
        compressed: bool = False,
        pixels: np.ndarray | None = None,
    ) -> None:
        super().__init__(path, AssetType.TEXTURE)
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.pixel_format = str(pixel_format)
        self.bit_depth = int(bit_depth)
        self.mip_levels = int(mip_levels)
        self.compressed = bool(compressed)
        self.pixels = pixels

    @property
    def is_decoded(self) -> bool:
        """``True``, если пиксели доступны в памяти."""
        return self.pixels is not None

    @property
    def estimated_bytes(self) -> int:
        """Объём пиксельных данных (0, если текстура не декодирована)."""
        if self.pixels is None:
            return 0
        return int(self.pixels.nbytes)

    def __repr__(self) -> str:
        return (
            f"<TextureAsset '{self.name}' {self.width}x{self.height} "
            f"{self.pixel_format} decoded={self.is_decoded}>"
        )


class ShaderStage(str, Enum):
    """Стадия шейдера."""

    VERTEX = "vertex"
    FRAGMENT = "fragment"
    GEOMETRY = "geometry"
    COMPUTE = "compute"
    UNKNOWN = "unknown"


class ShaderAsset(Asset):
    """Шейдер: исходник одной стадии с разрешёнными ``#include``.

    Args:
        path: Путь к файлу шейдера.
        stage: Стадия шейдера.
        source: Итоговый исходник.
        includes: Пути всех включённых файлов (по порядку).
    """

    __slots__ = ("includes", "source", "stage")

    def __init__(
        self,
        path: Path,
        stage: ShaderStage,
        source: str,
        includes: tuple[str, ...] = (),
    ) -> None:
        super().__init__(path, AssetType.SHADER)
        self.stage = stage
        self.source = source
        self.includes = includes

    @property
    def estimated_bytes(self) -> int:
        """Объём исходного текста шейдера."""
        return len(self.source.encode("utf-8"))

    def __repr__(self) -> str:
        return (
            f"<ShaderAsset '{self.name}' stage={self.stage.value} "
            f"includes={len(self.includes)}>"
        )


def describe(asset: Asset) -> dict[str, Any]:
    """Краткое описание ресурса (для панели ресурсов в редакторе)."""
    return {
        "name": asset.name,
        "type": asset.asset_type.value,
        "path": str(asset.path),
        "refs": asset.ref_count,
        "bytes": asset.estimated_bytes,
    }
