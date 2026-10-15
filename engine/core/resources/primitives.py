"""Процедурные примитивы: куб, сфера, плоскость и сетка.

Генераторы возвращают обычные :class:`MeshAsset`, поэтому их можно
использовать и как ассеты (превью в редакторе, отладочная геометрия), и как
источник данных для рендера. Вычисления векторизованы numpy-ом: сфера из
нескольких тысяч вершин собирается за один проход без питон-циклов.
"""

from __future__ import annotations

import numpy as np

from engine.core.math3d import Vector3
from engine.core.resources.asset import MeshAsset

__all__ = [
    "bounds_of",
    "compute_normals",
    "cube_mesh",
    "cylinder_mesh",
    "line_grid_mesh",
    "plane_mesh",
    "uv_sphere_mesh",
    "wire_box_mesh",
]


def compute_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Посчитать нормали по граням и усреднить их по вершинам.

    Args:
        positions: Вершины ``(N, 3)``.
        indices: Индексы ``(M,)``, кратное трём.

    Returns:
        Нормали ``(N, 3)`` (нормализованные; у изолированных вершин — ``+Y``).
    """
    points = np.asarray(positions, dtype=np.float32)
    triangles = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    normals = np.zeros_like(points)

    first = points[triangles[:, 0]]
    second = points[triangles[:, 1]]
    third = points[triangles[:, 2]]
    face_normals = np.cross(second - first, third - first)

    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face_normals)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths < 1e-8] = 1.0
    normals = normals / lengths
    degenerate = np.linalg.norm(normals, axis=1) < 1e-6
    normals[degenerate] = (0.0, 1.0, 0.0)
    return normals.astype(np.float32)


def bounds_of(positions: np.ndarray) -> tuple[Vector3, Vector3]:
    """Вернуть AABB набора вершин."""
    points = np.asarray(positions, dtype=np.float32)
    if points.size == 0:
        return Vector3.zero(), Vector3.zero()
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    return (
        Vector3(float(minimum[0]), float(minimum[1]), float(minimum[2])),
        Vector3(float(maximum[0]), float(maximum[1]), float(maximum[2])),
    )


def cube_mesh(size: float = 1.0, *, name: str = "cube") -> MeshAsset:
    """Собрать куб с раздельными нормалями и UV на каждой грани.

    Args:
        size: Длина ребра.
        name: Имя меша.

    Returns:
        Меш из 24 вершин и 12 треугольников.
    """
    half = float(size) * 0.5
    # Для каждой грани: нормаль и два касательных вектора (U, V).
    faces = (
        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
        ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )
    corners = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    uvs = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))

    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    uvs_out: list[tuple[float, float]] = []
    indices: list[int] = []

    for normal, tangent_u, tangent_v in faces:
        base = len(positions)
        for (u_sign, v_sign), (u, v) in zip(corners, uvs, strict=True):
            offset = np.array(normal, dtype=np.float32) * half
            offset += np.array(tangent_u, dtype=np.float32) * (u_sign * half)
            offset += np.array(tangent_v, dtype=np.float32) * (v_sign * half)
            positions.append((float(offset[0]), float(offset[1]), float(offset[2])))
            normals.append(normal)
            uvs_out.append((u, v))
        indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))

    return _procedural_asset(
        name,
        np.asarray(positions, dtype=np.float32),
        np.asarray(indices, dtype=np.uint32),
        np.asarray(normals, dtype=np.float32),
        np.asarray(uvs_out, dtype=np.float32),
    )


def uv_sphere_mesh(
    radius: float = 0.5,
    *,
    segments: int = 24,
    rings: int = 16,
    name: str = "sphere",
) -> MeshAsset:
    """Собрать UV-сферу.

    Args:
        radius: Радиус сферы.
        segments: Число сегментов по долготе (минимум 3).
        rings: Число колец по широте (минимум 2).
        name: Имя меша.

    Returns:
        Меш с нормалями и UV.

    Raises:
        ValueError: Если сегментов или колец слишком мало.
    """
    if segments < 3 or rings < 2:
        raise ValueError("Сфере нужно минимум 3 сегмента и 2 кольца")

    u = np.linspace(0.0, 1.0, segments + 1, dtype=np.float32)
    v = np.linspace(0.0, 1.0, rings + 1, dtype=np.float32)
    grid_u, grid_v = np.meshgrid(u, v, indexing="ij")

    phi = grid_u * (2.0 * np.pi)
    theta = grid_v * np.pi
    sin_theta = np.sin(theta)
    unit_x = sin_theta * np.sin(phi)
    unit_y = np.cos(theta)
    unit_z = sin_theta * np.cos(phi)

    positions = np.stack(
        [unit_x * radius, unit_y * radius, unit_z * radius], axis=-1
    ).reshape(-1, 3)
    normals = np.stack([unit_x, unit_y, unit_z], axis=-1).reshape(-1, 3)
    uvs = np.stack([grid_u, 1.0 - grid_v], axis=-1).reshape(-1, 2)

    index_grid = np.arange((segments + 1) * (rings + 1), dtype=np.uint32).reshape(
        segments + 1, rings + 1
    )
    quad = np.stack(
        [
            index_grid[:-1, :-1],
            index_grid[1:, :-1],
            index_grid[1:, 1:],
            index_grid[:-1, 1:],
        ],
        axis=-1,
    ).reshape(-1, 4)
    indices = np.concatenate(
        [quad[:, [0, 2, 1]], quad[:, [0, 3, 2]]], axis=0
    ).reshape(-1)

    return _procedural_asset(
        name,
        positions.astype(np.float32),
        indices.astype(np.uint32),
        normals.astype(np.float32),
        uvs.astype(np.float32),
    )


def cylinder_mesh(
    radius: float = 0.5,
    height: float = 1.0,
    *,
    segments: int = 24,
    name: str = "cylinder",
) -> MeshAsset:
    """Собрать цилиндр вдоль оси Y (боковая поверхность и обе крышки).

    Args:
        radius: Радиус основания.
        height: Высота цилиндра.
        segments: Число сегментов по окружности (минимум 3).
        name: Имя меша.

    Returns:
        Меш с нормалями и UV; грани обходятся против часовой стрелки.

    Raises:
        ValueError: Если сегментов меньше трёх.
    """
    if segments < 3:
        raise ValueError("Цилиндру нужно минимум 3 сегмента")

    half = float(height) * 0.5
    angles = np.linspace(0.0, 2.0 * np.pi, segments + 1, dtype=np.float32)
    cos_a = np.cos(angles).astype(np.float32)
    sin_a = np.sin(angles).astype(np.float32)
    ring_x = cos_a * radius
    ring_z = sin_a * radius

    # Боковая поверхность: нижнее кольцо, затем верхнее.
    bottom = np.stack([ring_x, np.full_like(ring_x, -half), ring_z], axis=-1)
    top = np.stack([ring_x, np.full_like(ring_x, half), ring_z], axis=-1)
    side_positions = np.concatenate([bottom, top], axis=0)
    ring_normals = np.stack([cos_a, np.zeros_like(cos_a), sin_a], axis=-1)
    side_normals = np.tile(ring_normals, (2, 1))
    along = np.linspace(0.0, 1.0, segments + 1, dtype=np.float32)
    side_uvs = np.stack(
        [
            np.stack([along, np.zeros_like(along)], axis=-1),
            np.stack([along, np.ones_like(along)], axis=-1),
        ],
        axis=0,
    ).reshape(-1, 2)

    lower = np.arange(segments + 1, dtype=np.uint32)
    upper = lower + np.uint32(segments + 1)
    quads = np.stack([lower[:-1], lower[1:], upper[1:], upper[:-1]], axis=-1)
    side_indices = np.concatenate(
        [quads[:, [0, 3, 2]], quads[:, [0, 2, 1]]], axis=0
    ).reshape(-1)

    # Крышки: центральная вершина плюс то же кольцо; нормали строго по оси Y.
    top_ring = np.stack([ring_x, np.full_like(ring_x, half), ring_z], axis=-1)
    bottom_ring = np.stack([ring_x, np.full_like(ring_x, -half), ring_z], axis=-1)
    top_center = np.array([[0.0, half, 0.0]], dtype=np.float32)
    bottom_center = np.array([[0.0, -half, 0.0]], dtype=np.float32)
    cap_uvs = np.stack([cos_a * 0.5 + 0.5, sin_a * 0.5 + 0.5], axis=-1)
    center_uv = np.array([[0.5, 0.5]], dtype=np.float32)

    positions = np.concatenate(
        [side_positions, top_center, top_ring, bottom_center, bottom_ring], axis=0
    )
    normals = np.concatenate(
        [
            side_normals,
            np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (segments + 2, 1)),
            np.tile(np.array([0.0, -1.0, 0.0], dtype=np.float32), (segments + 2, 1)),
        ],
        axis=0,
    )
    uvs = np.concatenate(
        [
            side_uvs,
            center_uv,
            cap_uvs,
            center_uv,
            np.flip(cap_uvs, axis=0),
        ],
        axis=0,
    )

    base_top = np.uint32(side_positions.shape[0])
    base_bottom = base_top + np.uint32(segments + 2)
    indices = np.concatenate(
        [
            side_indices,
            _cap_indices(segments, base_top, facing_up=True),
            _cap_indices(segments, base_bottom, facing_up=False),
        ],
        axis=0,
    )

    return _procedural_asset(
        name,
        positions.astype(np.float32),
        indices.astype(np.uint32),
        normals.astype(np.float32),
        uvs.astype(np.float32),
    )


def _cap_indices(segments: int, base: np.uint32, *, facing_up: bool) -> np.ndarray:
    """Собрать индексы веера крышки цилиндра.

    Вершины крышки лежат блоком: сначала центр, затем кольцо из ``segments + 1``
    точек. Треугольники обходятся против часовой стрелки со стороны нормали.

    Args:
        segments: Число сегментов по окружности.
        base: Индекс первой вершины блока крышки (центральной).
        facing_up: ``True`` для верхней крышки (нормаль ``+Y``).

    Returns:
        Плоский массив индексов треугольников.
    """
    ring = np.arange(1, segments + 2, dtype=np.uint32) + base
    center = np.full(segments, base, dtype=np.uint32)
    first, second = ring[:-1], ring[1:]
    triangles = (
        np.stack([center, second, first], axis=-1)
        if facing_up
        else np.stack([center, first, second], axis=-1)
    )
    return triangles.reshape(-1).astype(np.uint32)


def plane_mesh(
    size: float = 1.0, *, subdivisions: int = 1, name: str = "plane"
) -> MeshAsset:
    """Собрать горизонтальную плоскость в плоскости XZ.

    Args:
        size: Длина стороны.
        subdivisions: Число разбиений по каждой оси.
        name: Имя меша.

    Returns:
        Меш с нормалями ``+Y`` и UV.

    Raises:
        ValueError: Если subdivisions меньше единицы.
    """
    if subdivisions < 1:
        raise ValueError("subdivisions должно быть не меньше 1")

    half = float(size) * 0.5
    steps = subdivisions + 1
    axis = np.linspace(-half, half, steps, dtype=np.float32)
    grid_x, grid_z = np.meshgrid(axis, axis, indexing="ij")

    positions = np.stack(
        [grid_x, np.zeros_like(grid_x), grid_z], axis=-1
    ).reshape(-1, 3)
    normals = np.tile(
        np.array([0.0, 1.0, 0.0], dtype=np.float32), (positions.shape[0], 1)
    )
    uv_axis = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    grid_u, grid_v = np.meshgrid(uv_axis, uv_axis, indexing="ij")
    uvs = np.stack([grid_u, grid_v], axis=-1).reshape(-1, 2)

    index_grid = np.arange(steps * steps, dtype=np.uint32).reshape(steps, steps)
    quad = np.stack(
        [
            index_grid[:-1, :-1],
            index_grid[1:, :-1],
            index_grid[1:, 1:],
            index_grid[:-1, 1:],
        ],
        axis=-1,
    ).reshape(-1, 4)
    indices = np.concatenate(
        [quad[:, [0, 2, 1]], quad[:, [0, 3, 2]]], axis=0
    ).reshape(-1)

    return _procedural_asset(
        name,
        positions.astype(np.float32),
        indices.astype(np.uint32),
        normals,
        uvs.astype(np.float32),
    )


def line_grid_mesh(
    size: float = 10.0, *, divisions: int = 10, name: str = "grid"
) -> MeshAsset:
    """Собрать сетку из линий (для отрисовки ``GL_LINES``).

    Args:
        size: Длина стороны сетки.
        divisions: Число клеток по каждой оси.
        name: Имя меша.

    Returns:
        Меш без индексов: пары вершин образуют отрезки.

    Raises:
        ValueError: Если divisions меньше единицы.
    """
    if divisions < 1:
        raise ValueError("divisions должно быть не меньше 1")

    half = float(size) * 0.5
    step = float(size) / divisions
    lines: list[tuple[float, float, float]] = []
    for index in range(divisions + 1):
        offset = -half + index * step
        lines.append((-half, 0.0, offset))
        lines.append((half, 0.0, offset))
        lines.append((offset, 0.0, -half))
        lines.append((offset, 0.0, half))

    positions = np.asarray(lines, dtype=np.float32)
    return _procedural_asset(name, positions, None, None, None)


def wire_box_mesh(size: float = 1.0, *, name: str = "wire_box") -> MeshAsset:
    """Собрать каркасный куб из 12 отрезков (рамки выделения, гизмо).

    Args:
        size: Длина ребра (куб строится вокруг начала координат).
        name: Имя меша.

    Returns:
        Меш без индексов: пары вершин образуют отрезки для ``GL_LINES``.
    """
    half = float(size) * 0.5
    corners = (
        (-half, -half, -half),
        (half, -half, -half),
        (half, half, -half),
        (-half, half, -half),
        (-half, -half, half),
        (half, -half, half),
        (half, half, half),
        (-half, half, half),
    )
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),  # нижняя грань
        (4, 5), (5, 6), (6, 7), (7, 4),  # верхняя грань
        (0, 4), (1, 5), (2, 6), (3, 7),  # вертикальные рёбра
    )
    points: list[tuple[float, float, float]] = []
    for start, end in edges:
        points.append(corners[start])
        points.append(corners[end])
    return _procedural_asset(
        name, np.asarray(points, dtype=np.float32), None, None, None
    )


def _procedural_asset(
    name: str,
    positions: np.ndarray,
    indices: np.ndarray | None,
    normals: np.ndarray | None,
    uvs: np.ndarray | None,
) -> MeshAsset:
    """Собрать :class:`MeshAsset` для процедурного меша.

    Процедурные меши не привязаны к файлу, поэтому путь указывается
    виртуальный (``"<procedural>/имя"``), а имя задаётся явно.

    Args:
        name: Имя меша.
        positions: Вершины ``(N, 3)``.
        indices: Индексы или ``None``.
        normals: Нормали или ``None``.
        uvs: Текстурные координаты или ``None``.

    Returns:
        Готовый ассет меша.
    """
    from pathlib import Path

    asset = MeshAsset(
        path=Path(f"<procedural>/{name}"),
        positions=positions,
        indices=indices,
        normals=normals,
        uvs=uvs,
        name=name,
    )
    return asset
