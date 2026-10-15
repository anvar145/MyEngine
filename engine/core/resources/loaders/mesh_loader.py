"""Загрузчик Wavefront OBJ.

Парсер намеренно написан на чистом Python без зависимостей: он рассчитан на
low-poly контент (цель движка — слабое железо), а тяжёлые форматы вроде FBX
подключаются отдельным «extra»-модулем. Поддерживаются:

* ``v``, ``vt``, ``vn``;
* грани ``f v``, ``f v/vt``, ``f v//vn``, ``f v/vt/vn``;
* отрицательные (относительные) индексы;
* полигоны из более чем трёх вершин (триангуляция «веером»);
* комментарии, ``o``/``g``/``s``/``usemtl``/``mtllib`` (игнорируются).

Вершины дедуплицируются по тройке «позиция/UV/нормаль», поэтому индексный
буфер получается компактным.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from engine.core.resources.asset import MeshAsset
from engine.core.resources.errors import ResourceFormatError
from engine.core.resources.loader import AssetLoader, LoadContext

__all__ = ["ObjMeshLoader", "parse_obj"]

_log = logging.getLogger("engine.core.resources.obj")


class ObjMeshLoader(AssetLoader):
    """Загрузчик ``.obj`` в :class:`~engine.core.resources.asset.MeshAsset`."""

    extensions = (".obj",)

    def load(self, context: LoadContext) -> MeshAsset:
        """Разобрать OBJ-файл.

        Поддерживаемые параметры импорта:
            scale (float): Множитель координат вершин, по умолчанию ``1.0``.
            flip_uv (bool): Инвертировать координату ``v`` (для GL-конвенции).
        """
        scale = float(context.option("scale", 1.0))
        flip_uv = bool(context.option("flip_uv", False))
        asset = parse_obj(context.path, scale=scale, flip_uv=flip_uv)
        _log.debug(
            "OBJ загружен: %s (вершин=%d, треугольников=%d)",
            context.path.name,
            asset.vertex_count,
            asset.triangle_count,
        )
        return asset


def parse_obj(path: Path, *, scale: float = 1.0, flip_uv: bool = False) -> MeshAsset:
    """Разобрать OBJ-файл в меш.

    Args:
        path: Путь к файлу ``.obj``.
        scale: Множитель координат вершин.
        flip_uv: Инвертировать координату ``v`` текстурных координат.

    Returns:
        :class:`MeshAsset` с позициями, нормалями, UV и индексами.

    Raises:
        ResourceFormatError: Если файл пуст, повреждён или в нём нет вершин.
    """
    file_positions: list[tuple[float, float, float]] = []
    file_uvs: list[tuple[float, float]] = []
    file_normals: list[tuple[float, float, float]] = []

    out_positions: list[tuple[float, float, float]] = []
    out_uvs: list[tuple[float, float]] = []
    out_normals: list[tuple[float, float, float]] = []
    out_indices: list[int] = []
    vertex_map: dict[tuple[int, int, int], int] = {}
    has_uv = False
    has_normal = False

    try:
        stream = path.open("r", encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ResourceFormatError(f"Не удалось открыть '{path}': {exc}") from exc

    with stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0]

            if tag == "v":
                if len(parts) < 4:
                    raise ResourceFormatError(
                        f"{path.name}:{line_number}: строка 'v' требует 3 координаты"
                    )
                file_positions.append(
                    (
                        float(parts[1]) * scale,
                        float(parts[2]) * scale,
                        float(parts[3]) * scale,
                    )
                )
            elif tag == "vt":
                if len(parts) < 3:
                    raise ResourceFormatError(
                        f"{path.name}:{line_number}: строка 'vt' требует 2 координаты"
                    )
                v = -float(parts[2]) if flip_uv else float(parts[2])
                file_uvs.append((float(parts[1]), v))
            elif tag == "vn":
                if len(parts) < 4:
                    raise ResourceFormatError(
                        f"{path.name}:{line_number}: строка 'vn' требует 3 компоненты"
                    )
                file_normals.append(
                    (float(parts[1]), float(parts[2]), float(parts[3]))
                )
            elif tag == "f":
                corners = parts[1:]
                if len(corners) < 3:
                    raise ResourceFormatError(
                        f"{path.name}:{line_number}: "
                        f"грань должна содержать минимум 3 вершины"
                    )
                triangle_indices: list[int] = []
                for corner in corners:
                    key = _resolve_corner(
                        corner,
                        line_number,
                        path,
                        len(file_positions),
                        len(file_uvs),
                        len(file_normals),
                    )
                    position_index, uv_index, normal_index = key
                    if uv_index >= 0:
                        has_uv = True
                    if normal_index >= 0:
                        has_normal = True
                    vertex_index = vertex_map.get(key)
                    if vertex_index is None:
                        vertex_index = len(out_positions)
                        vertex_map[key] = vertex_index
                        out_positions.append(file_positions[position_index])
                        out_uvs.append(
                            file_uvs[uv_index] if uv_index >= 0 else (0.0, 0.0)
                        )
                        out_normals.append(
                            file_normals[normal_index]
                            if normal_index >= 0
                            else (0.0, 0.0, 0.0)
                        )
                    triangle_indices.append(vertex_index)
                for offset in range(1, len(triangle_indices) - 1):
                    out_indices.append(triangle_indices[0])
                    out_indices.append(triangle_indices[offset])
                    out_indices.append(triangle_indices[offset + 1])
            # Прочие директивы (o, g, s, usemtl, mtllib) для геометрии не нужны.

    if not out_positions:
        if not file_positions:
            raise ResourceFormatError(f"'{path}' не содержит ни одной вершины")
        # Файл без граней: отдаём «облако» вершин без индексов.
        positions = np.asarray(file_positions, dtype=np.float32)
        return MeshAsset(path, positions)

    positions = np.asarray(out_positions, dtype=np.float32)
    indices = np.asarray(out_indices, dtype=np.uint32)
    normals = np.asarray(out_normals, dtype=np.float32) if has_normal else None
    uvs = np.asarray(out_uvs, dtype=np.float32) if has_uv else None
    return MeshAsset(path, positions, indices, normals, uvs)


def _resolve_corner(
    corner: str,
    line_number: int,
    path: Path,
    position_count: int,
    uv_count: int,
    normal_count: int,
) -> tuple[int, int, int]:
    """Разобрать одну ссылку на вершину в грани.

    Args:
        corner: Текст вида ``v``, ``v/vt``, ``v//vn`` или ``v/vt/vn``.
        line_number: Номер строки (для сообщения об ошибке).
        path: Путь к файлу (для сообщения об ошибке).
        position_count: Сколько позиций уже прочитано.
        uv_count: Сколько UV уже прочитано.
        normal_count: Сколько нормалей уже прочитано.

    Returns:
        Тройку ноль-базированных индексов; ``-1`` означает «нет данных».

    Raises:
        ResourceFormatError: Если индекс некорректен.
    """
    chunks = corner.split("/")
    position_index = _parse_index(
        chunks[0], position_count, line_number, path, "позиции"
    )
    uv_index = -1
    normal_index = -1
    if len(chunks) > 1 and chunks[1]:
        uv_index = _parse_index(chunks[1], uv_count, line_number, path, "UV")
    if len(chunks) > 2 and chunks[2]:
        normal_index = _parse_index(
            chunks[2], normal_count, line_number, path, "нормали"
        )
    return position_index, uv_index, normal_index


def _parse_index(
    text: str, count: int, line_number: int, path: Path, what: str
) -> int:
    """Преобразовать индекс OBJ в ноль-базированный.

    Raises:
        ResourceFormatError: Если индекс не число или выходит за пределы файла.
    """
    try:
        value = int(text)
    except ValueError as exc:
        raise ResourceFormatError(
            f"{path.name}:{line_number}: некорректный индекс {what}: '{text}'"
        ) from exc
    index = value - 1 if value > 0 else count + value
    if index < 0 or index >= count:
        raise ResourceFormatError(
            f"{path.name}:{line_number}: "
            f"индекс {what} {value} вне диапазона (0..{count})"
        )
    return index
