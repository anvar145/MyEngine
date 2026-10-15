"""Mesh cooker: LOD, переупорядочивание вершин и квантизация (офлайн, без GL).

Конвейер для каждого уровня детализации:

1. **LOD** — упрощение *vertex clustering*: вершины попадают в ячейки равномерной
   сетки, ячейка схлопывается в одну вершину (среднее позиций, нормалей и UV),
   вырожденные треугольники отбрасываются. Размер ячейки подбирается бинарным
   поиском под целевое число треугольников. Метод быстрый и предсказуемый,
   но не сохраняет швы UV — для LOD1+ это допустимо.
2. **Переупорядочивание вершин** — по порядку первого использования в индексном
   буфере (улучшает локальность чтения вершин, неиспользуемые отбрасываются).
3. **Квантизация** — позиции и нормали в ``int16`` (см. :mod:`mesh_format`).

Cooker знает только про байты и numpy; загрузка в GPU — задача ``engine/rendering``.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from engine.core.resources.cookers.mesh_format import (
    ALIGNMENT,
    FLAG_NORMALS,
    FLAG_UVS,
    FORMAT_VERSION,
    HEADER_SIZE,
    LOD_ENTRY_SIZE,
    MAGIC,
    SNORM,
)
from engine.core.resources.errors import AssetImportError

__all__ = [
    "MeshCookSettings",
    "cook_mesh",
    "reorder_vertices",
    "simplify_clustering",
]

_MAX_GRID = 2048
#: Новый LOD должен быть заметно проще предыдущего, иначе он не нужен.
_MIN_REDUCTION = 0.95
_MAX_LODS = 8


@dataclass(frozen=True, slots=True)
class MeshCookSettings:
    """Параметры кука меша.

    Attributes:
        generate_lods: Строить ли LOD1+.
        lod_ratios: Доли треугольников LOD1, LOD2... относительно LOD0, строго
            убывающие, каждая в ``(0, 1)``.
    """

    generate_lods: bool = True
    lod_ratios: tuple[float, ...] = (0.5, 0.25)

    def __post_init__(self) -> None:
        ratios = self.lod_ratios
        if len(ratios) > _MAX_LODS - 1:
            raise AssetImportError(f"lod_ratios: не более {_MAX_LODS - 1} уровней")
        previous = 1.0
        for ratio in ratios:
            if not 0.0 < ratio < previous:
                raise AssetImportError(
                    "lod_ratios должны строго убывать и лежать в (0, 1): "
                    f"{list(ratios)}"
                )
            previous = ratio

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> MeshCookSettings:
        """Собрать настройки из ``.meta`` (``generate_lods``, ``lod_ratios``)."""
        raw = values.get("lod_ratios", cls.lod_ratios)
        if not isinstance(raw, (list, tuple)):
            raise AssetImportError("lod_ratios должен быть списком чисел")
        try:
            ratios = tuple(float(item) for item in raw)
        except (TypeError, ValueError) as exc:
            raise AssetImportError("lod_ratios должен содержать числа") from exc
        return cls(bool(values.get("generate_lods", True)), ratios)


def cook_mesh(
    positions: np.ndarray,
    indices: np.ndarray | None,
    normals: np.ndarray | None = None,
    uvs: np.ndarray | None = None,
    settings: MeshCookSettings | None = None,
) -> bytes:
    """Скукобить меш в бинарный ``mesh.bin``.

    Args:
        positions: ``(N, 3)`` позиции.
        indices: ``(M,)`` индексы треугольников (``M`` кратно 3) или ``None``
            для облака точек (один LOD без индексов).
        normals: ``(N, 3)`` нормали.
        uvs: ``(N, 2)`` текстурные координаты.
        settings: Параметры кука.

    Returns:
        Содержимое файла ``mesh.bin``.

    Raises:
        AssetImportError: Если массивы несогласованы или индексы вне диапазона.
    """
    settings = settings or MeshCookSettings()
    positions = np.ascontiguousarray(positions, dtype=np.float32)
    _check_shapes(positions, indices, normals, uvs)
    if positions.shape[0] == 0:
        raise AssetImportError("Меш не содержит вершин")
    if not np.isfinite(positions).all():
        raise AssetImportError("Позиции содержат NaN или бесконечность")

    levels: list[_Level] = []
    if indices is None or len(indices) == 0:
        levels.append(_Level(positions, np.empty(0, np.uint32), normals, uvs))
    else:
        base_indices = np.ascontiguousarray(indices, dtype=np.uint32)
        full = _Level(*reorder_vertices(positions, base_indices, normals, uvs))
        levels.append(full)
        if settings.generate_lods:
            base_triangles = full.indices.size // 3
            for ratio in settings.lod_ratios:
                target = max(1, int(base_triangles * ratio))
                previous = levels[-1]
                if target >= previous.indices.size // 3 * _MIN_REDUCTION:
                    continue
                simplified = simplify_clustering(
                    full.positions, full.indices, full.normals, full.uvs, target
                )
                level = _Level(*reorder_vertices(*simplified))
                if level.indices.size < previous.indices.size * _MIN_REDUCTION:
                    levels.append(level)

    lows = np.min([level.positions.min(axis=0) for level in levels], axis=0)
    highs = np.max([level.positions.max(axis=0) for level in levels], axis=0)
    center = ((lows + highs) * 0.5).astype(np.float32)
    half = ((highs - lows) * 0.5).astype(np.float32)
    safe_half = np.where(half > 0, half, np.float32(1.0))
    return _serialize(levels, center, half, safe_half)


def reorder_vertices(
    positions: np.ndarray,
    indices: np.ndarray,
    normals: np.ndarray | None = None,
    uvs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Переупорядочить вершины по первому использованию и убрать неиспользуемые.

    Returns:
        ``(positions, indices, normals, uvs)`` уже в новом порядке; порядок
        возвращаемых значений совпадает с аргументами :func:`simplify_clustering`
        и конструктором уровня.
    """
    used, first = np.unique(indices, return_index=True)
    order = used[np.argsort(first, kind="stable")]
    remap = np.zeros(positions.shape[0], dtype=np.uint32)
    remap[order] = np.arange(order.size, dtype=np.uint32)
    return (
        positions[order],
        remap[indices],
        None if normals is None else normals[order],
        None if uvs is None else uvs[order],
    )


def simplify_clustering(
    positions: np.ndarray,
    indices: np.ndarray,
    normals: np.ndarray | None,
    uvs: np.ndarray | None,
    target_triangles: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Упростить меш кластеризацией вершин до ``target_triangles`` и меньше.

    Args:
        positions: ``(N, 3)`` позиции.
        indices: ``(M,)`` индексы треугольников.
        normals: ``(N, 3)`` нормали или ``None``.
        uvs: ``(N, 2)`` UV или ``None``.
        target_triangles: Желаемый максимум треугольников (минимум остаётся 1,
            если меш вообще можно сохранить непустым).

    Returns:
        ``(positions, indices, normals, uvs)`` упрощённого меша.
    """
    triangles = indices.reshape(-1, 3)
    low = positions.min(axis=0)
    extent = float((positions.max(axis=0) - low).max()) or 1.0

    def cluster(resolution: int) -> tuple[np.ndarray, np.ndarray]:
        cell = np.minimum(
            ((positions - low) / (extent / resolution)).astype(np.int64),
            resolution - 1,
        )
        key = cell[:, 0] + resolution * (cell[:, 1] + resolution * cell[:, 2])
        _, inverse = np.unique(key, return_inverse=True)
        mapped = inverse.reshape(-1)[triangles]
        keep = (
            (mapped[:, 0] != mapped[:, 1])
            & (mapped[:, 1] != mapped[:, 2])
            & (mapped[:, 0] != mapped[:, 2])
        )
        return inverse.reshape(-1), mapped[keep]

    low_n, high_n = 1, _MAX_GRID
    while low_n < high_n:  # наибольшая сетка, при которой влезаем в цель
        middle = (low_n + high_n + 1) // 2
        if len(cluster(middle)[1]) <= target_triangles:
            low_n = middle
        else:
            high_n = middle - 1
    resolution = low_n
    inverse, kept = cluster(resolution)
    while len(kept) == 0 and resolution < _MAX_GRID:  # слишком грубо: не теряем меш
        resolution = min(_MAX_GRID, resolution * 2)
        inverse, kept = cluster(resolution)

    count = int(inverse.max()) + 1
    members = np.bincount(inverse, minlength=count).astype(np.float64)

    def average(values: np.ndarray) -> np.ndarray:
        columns = [
            np.bincount(inverse, weights=values[:, axis], minlength=count)
            for axis in range(values.shape[1])
        ]
        return np.stack(columns, axis=1)

    new_positions = (average(positions) / members[:, None]).astype(np.float32)
    new_normals = None
    if normals is not None:
        summed = average(normals)
        length = np.linalg.norm(summed, axis=1, keepdims=True)
        fallback = np.array([[0.0, 0.0, 1.0]])
        safe = np.maximum(length, 1e-12)
        new_normals = np.where(length > 1e-12, summed / safe, fallback)
        new_normals = new_normals.astype(np.float32)
    new_uvs = None
    if uvs is not None:
        new_uvs = (average(uvs) / members[:, None]).astype(np.float32)
    return new_positions, kept.reshape(-1).astype(np.uint32), new_normals, new_uvs


@dataclass(slots=True)
class _Level:
    """Геометрия одного уровня до квантизации."""

    positions: np.ndarray
    indices: np.ndarray
    normals: np.ndarray | None
    uvs: np.ndarray | None


def _check_shapes(
    positions: np.ndarray,
    indices: np.ndarray | None,
    normals: np.ndarray | None,
    uvs: np.ndarray | None,
) -> None:
    """Проверить согласованность входных массивов."""
    count = positions.shape[0] if positions.ndim == 2 else -1
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise AssetImportError(
            f"positions должен быть (N, 3), получено {positions.shape}"
        )
    if normals is not None and np.shape(normals) != (count, 3):
        raise AssetImportError("normals должен иметь форму (N, 3)")
    if uvs is not None and np.shape(uvs) != (count, 2):
        raise AssetImportError("uvs должен иметь форму (N, 2)")
    if indices is not None and len(indices) > 0:
        flat = np.asarray(indices)
        if flat.ndim != 1 or flat.size % 3:
            raise AssetImportError("Число индексов должно быть кратно 3")
        if int(flat.min()) < 0 or int(flat.max()) >= count:
            raise AssetImportError("Индекс вершины выходит за пределы массива позиций")


def _serialize(
    levels: Sequence[_Level],
    center: np.ndarray,
    half: np.ndarray,
    safe_half: np.ndarray,
) -> bytes:
    """Собрать файл: заголовок, таблицу LOD и выровненные секции."""
    first = levels[0]
    flags = (FLAG_NORMALS if first.normals is not None else 0) | (
        FLAG_UVS if first.uvs is not None else 0
    )
    table_end = HEADER_SIZE + len(levels) * LOD_ENTRY_SIZE
    body = bytearray()
    entries = []

    def place(blob: bytes) -> int:
        """Добавить секцию с выравниванием и вернуть её смещение в файле."""
        while (table_end + len(body)) % ALIGNMENT:
            body.append(0)
        offset = table_end + len(body)
        body.extend(blob)
        return offset

    for level in levels:
        count = level.positions.shape[0]
        quantized = np.zeros((count, 4), dtype="<i2")
        quantized[:, :3] = np.rint(
            np.clip((level.positions - center) / safe_half, -1.0, 1.0) * SNORM
        )
        pos = place(quantized.tobytes())
        norm = uv = idx = 0
        if level.normals is not None:
            unit = np.zeros((count, 4), dtype="<i2")
            unit[:, :3] = np.rint(np.clip(level.normals, -1.0, 1.0) * SNORM)
            norm = place(unit.tobytes())
        if level.uvs is not None:
            uv = place(np.ascontiguousarray(level.uvs, dtype="<f4").tobytes())
        index_size = 2 if count <= 0xFFFF else 4
        if level.indices.size:
            idx = place(level.indices.astype(f"<u{index_size}").tobytes())
        entries.append(
            struct.pack(
                "<IIIIQQQQ",
                count,
                level.indices.size,
                index_size,
                0,
                pos,
                norm,
                uv,
                idx,
            )
        )

    header = struct.pack(
        "<4sIII3f3f",
        MAGIC,
        FORMAT_VERSION,
        len(levels),
        flags,
        *center.tolist(),
        *half.tolist(),
    )
    header += bytes(HEADER_SIZE - len(header))
    return header + b"".join(entries) + bytes(body)
