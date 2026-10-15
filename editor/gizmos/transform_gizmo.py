"""Гизмо трансформа: оси для интерактивного перемещения, поворота и масштаба.

Гизмо рисуется у выделенного объекта и сохраняет постоянный размер на экране.
Три режима, как в привычных редакторах:

* ``MOVE`` (клавиша ``W``) — стрелки по мировым осям X, Y, Z;
* ``ROTATE`` (``E``) — кольца вокруг мировых осей;
* ``SCALE`` (``R``) — стержни с кубиками по локальным осям объекта.

Модуль не знает про OpenGL: математика (пикинг, перетаскивание) работает с
лучом камеры и трансформом объекта и проверяется без графики. Геометрия
(:func:`arrow_mesh`, :func:`scale_handle_mesh`, :func:`ring_mesh`) отдаётся как
:class:`~engine.core.resources.MeshAsset`, а вьюпорт редактора сам загружает её
в GPU и рисует поверх сцены.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from engine.core.math3d import EPSILON, Quaternion, Vector3, compose_matrix
from engine.core.resources import MeshAsset

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.game_object import GameObject
    from engine.rendering import Camera

__all__ = [
    "AXIS_COLORS",
    "GizmoMode",
    "TransformGizmo",
    "arrow_mesh",
    "closest_axis_parameter",
    "ring_mesh",
    "scale_handle_mesh",
]


class GizmoMode(str, Enum):
    """Режим гизмо трансформа."""

    MOVE = "move"
    ROTATE = "rotate"
    SCALE = "scale"


#: Цвета осей X, Y, Z и цвет активной (захваченной) оси.
AXIS_COLORS: tuple[Vector3, Vector3, Vector3] = (
    Vector3(0.90, 0.25, 0.25),
    Vector3(0.35, 0.80, 0.30),
    Vector3(0.30, 0.50, 0.95),
)
ACTIVE_COLOR = Vector3(1.0, 0.88, 0.25)

#: Доля расстояния до камеры, которую занимает гизмо (постоянный экранный размер).
SCREEN_FRACTION = 0.17

#: Допуск пикинга оси в пикселях.
PICK_TOLERANCE_PX = 12.0

#: Допуск пикинга кольца: доля радиуса.
RING_TOLERANCE = 0.14

#: Минимальный масштаб, ниже которого объект не сжимается.
MIN_SCALE = 0.01

_AXIS_VECTORS = (Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1))


def _axis_rotation(index: int) -> Quaternion:
    """Поворот, переводящий ось +X в ось с номером ``index``."""
    if index == 1:
        return Quaternion.from_axis_angle(Vector3(0, 0, 1), 90.0)
    if index == 2:
        return Quaternion.from_axis_angle(Vector3(0, 1, 0), -90.0)
    return Quaternion.identity()


# --------------------------------------------------------------------------- #
# Геометрия (единичная длина вдоль +X)
# --------------------------------------------------------------------------- #
def _mesh(name: str, positions: list[list[float]], indices: list[int]) -> MeshAsset:
    return MeshAsset(
        Path(f"<gizmo>/{name}"),
        np.asarray(positions, dtype=np.float32),
        np.asarray(indices, dtype=np.uint32),
        name=name,
    )


def _prism(
    positions: list[list[float]],
    indices: list[int],
    start: float,
    end: float,
    radius: float,
    segments: int = 6,
) -> None:
    """Добавить тонкую призму вдоль +X от ``start`` до ``end``."""
    base = len(positions)
    for step in range(segments):
        angle = 2.0 * math.pi * step / segments
        y, z = radius * math.cos(angle), radius * math.sin(angle)
        positions.append([start, y, z])
        positions.append([end, y, z])
    for step in range(segments):
        a = base + 2 * step
        b = base + 2 * ((step + 1) % segments)
        indices += [a, b, a + 1, b, b + 1, a + 1]


def arrow_mesh() -> MeshAsset:
    """Стрелка перемещения: тонкий стержень и конус на конце (длина 1 вдоль +X)."""
    positions: list[list[float]] = []
    indices: list[int] = []
    _prism(positions, indices, 0.0, 0.78, 0.022)
    segments = 14
    tip = len(positions)
    positions.append([1.0, 0.0, 0.0])
    ring = len(positions)
    for step in range(segments):
        angle = 2.0 * math.pi * step / segments
        positions.append([0.78, 0.075 * math.cos(angle), 0.075 * math.sin(angle)])
    for step in range(segments):
        indices += [tip, ring + step, ring + (step + 1) % segments]
    return _mesh("arrow", positions, indices)


def scale_handle_mesh() -> MeshAsset:
    """Ручка масштаба: стержень и кубик на конце (длина 1 вдоль +X)."""
    positions: list[list[float]] = []
    indices: list[int] = []
    _prism(positions, indices, 0.0, 0.86, 0.022)
    half = 0.07
    centre = 0.93
    base = len(positions)
    for dx in (-half, half):
        for dy in (-half, half):
            for dz in (-half, half):
                positions.append([centre + dx, dy, dz])
    faces = (
        (0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
        (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3),
    )  # fmt: skip
    for a, b, c, d in faces:
        indices += [base + a, base + b, base + c, base + a, base + c, base + d]
    return _mesh("scale_handle", positions, indices)


def ring_mesh(segments: int = 56, tube_segments: int = 5) -> MeshAsset:
    """Кольцо поворота радиуса 1 в плоскости YZ (нормаль — ось +X)."""
    positions: list[list[float]] = []
    indices: list[int] = []
    radius, tube = 1.0, 0.02
    for ring_step in range(segments):
        angle = 2.0 * math.pi * ring_step / segments
        cy, cz = math.cos(angle), math.sin(angle)
        for tube_step in range(tube_segments):
            phi = 2.0 * math.pi * tube_step / tube_segments
            offset = radius + tube * math.cos(phi)
            positions.append([tube * math.sin(phi), offset * cy, offset * cz])
    for ring_step in range(segments):
        nxt = (ring_step + 1) % segments
        for tube_step in range(tube_segments):
            tube_next = (tube_step + 1) % tube_segments
            a = ring_step * tube_segments + tube_step
            b = nxt * tube_segments + tube_step
            c = nxt * tube_segments + tube_next
            d = ring_step * tube_segments + tube_next
            indices += [a, b, c, a, c, d]
    return _mesh("ring", positions, indices)


# --------------------------------------------------------------------------- #
# Математика лучей
# --------------------------------------------------------------------------- #
def closest_axis_parameter(
    ray_origin: Vector3, ray_direction: Vector3, point: Vector3, axis: Vector3
) -> float | None:
    """Параметр точки прямой ``point + t * axis``, ближайшей к лучу.

    Args:
        ray_origin: Начало луча.
        ray_direction: Направление луча (единичное).
        point: Точка на прямой.
        axis: Единичное направление прямой.

    Returns:
        Параметр ``t`` или ``None``, если луч почти параллелен прямой.
    """
    cosine = axis.dot(ray_direction)
    denominator = 1.0 - cosine * cosine
    if denominator < 1e-6:
        return None
    offset = point - ray_origin
    return (cosine * ray_direction.dot(offset) - axis.dot(offset)) / denominator


def _ray_plane(
    ray_origin: Vector3, ray_direction: Vector3, point: Vector3, normal: Vector3
) -> Vector3 | None:
    """Пересечение луча с плоскостью (``None`` — луч параллелен)."""
    denominator = normal.dot(ray_direction)
    if abs(denominator) < 1e-6:
        return None
    t = normal.dot(point - ray_origin) / denominator
    if t < 0.0:
        return None
    return ray_origin + ray_direction * t


def _segment_distance_px(
    px: float, py: float, a: tuple[float, float], b: tuple[float, float]
) -> float:
    """Расстояние от точки до отрезка на экране (в пикселях)."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


@dataclass(slots=True)
class _Drag:
    """Состояние перетаскивания оси."""

    axis: int
    mode: GizmoMode
    origin: Vector3
    direction: Vector3
    start_position: Vector3
    start_rotation: Quaternion
    start_scale: Vector3
    start_parameter: float = 0.0
    start_vector: Vector3 = field(default_factory=Vector3.zero)


class TransformGizmo:
    """Интерактивные оси трансформа выделенного объекта.

    Attributes:
        mode: Текущий режим.
        active_axis: Номер захваченной оси (0, 1, 2) или ``None``.
    """

    def __init__(self, mode: GizmoMode = GizmoMode.MOVE) -> None:
        self.mode = mode
        self._drag: _Drag | None = None

    # ------------------------------------------------------------------ #
    # Состояние
    # ------------------------------------------------------------------ #
    @property
    def active_axis(self) -> int | None:
        """Ось, которую сейчас тянет пользователь."""
        return self._drag.axis if self._drag is not None else None

    @property
    def is_dragging(self) -> bool:
        """``True``, пока идёт перетаскивание."""
        return self._drag is not None

    def set_mode(self, mode: GizmoMode) -> None:
        """Сменить режим (отменяет текущее перетаскивание)."""
        self._drag = None
        self.mode = mode

    # ------------------------------------------------------------------ #
    # Геометрия в мире
    # ------------------------------------------------------------------ #
    @staticmethod
    def origin_of(game_object: GameObject) -> Vector3:
        """Мировая точка, в которой стоит гизмо."""
        return game_object.transform.world_position

    def screen_scale(self, origin: Vector3, camera: Camera) -> float:
        """Мировая длина оси гизмо при постоянном размере на экране."""
        if camera.is_orthographic:
            return max(camera.ortho_height, EPSILON) * SCREEN_FRACTION * 1.2
        distance = max((origin - camera.position).length, 0.05)
        half_fov = math.tan(math.radians(camera.fov) * 0.5)
        return distance * SCREEN_FRACTION * half_fov * 2.0

    def axis_direction(
        self, game_object: GameObject, index: int, mode: GizmoMode
    ) -> Vector3:
        """Мировое направление оси: локальное для масштаба, мировое для остального."""
        axis = _AXIS_VECTORS[index]
        if mode is GizmoMode.SCALE:
            rotation = _world_rotation(game_object)
            return rotation.rotate_vector(axis).normalized
        return axis

    # ------------------------------------------------------------------ #
    # Пикинг
    # ------------------------------------------------------------------ #
    def pick(
        self,
        game_object: GameObject,
        camera: Camera,
        x: float,
        y: float,
        width: int,
        height: int,
    ) -> int | None:
        """Найти ось под курсором.

        Args:
            game_object: Объект с гизмо.
            camera: Камера вьюпорта редактора.
            x: Координата курсора по горизонтали (пиксели).
            y: Координата курсора по вертикали.
            width: Ширина вьюпорта.
            height: Высота вьюпорта.

        Returns:
            Номер оси (0 — X, 1 — Y, 2 — Z) или ``None``.
        """
        origin = self.origin_of(game_object)
        length = self.screen_scale(origin, camera)
        best: tuple[float, int] | None = None
        if self.mode is GizmoMode.ROTATE:
            ray_origin, ray_direction = camera.ray_from_screen(x, y, width, height)
            for index in range(3):
                normal = _AXIS_VECTORS[index]
                if abs(normal.dot(ray_direction)) < 0.08:
                    continue  # кольцо почти ребром к лучу: попасть невозможно
                hit = _ray_plane(ray_origin, ray_direction, origin, normal)
                if hit is None:
                    continue
                error = abs((hit - origin).length - length) / length
                if error < RING_TOLERANCE and (best is None or error < best[0]):
                    best = (error, index)
            return best[1] if best is not None else None

        start = camera.world_to_screen(origin, width, height)
        if not start[2]:
            return None
        for index in range(3):
            direction = self.axis_direction(game_object, index, self.mode)
            end = camera.world_to_screen(origin + direction * length, width, height)
            if not end[2]:
                continue
            distance = _segment_distance_px(
                x, y, (start[0], start[1]), (end[0], end[1])
            )
            if distance < PICK_TOLERANCE_PX and (best is None or distance < best[0]):
                best = (distance, index)
        return best[1] if best is not None else None

    # ------------------------------------------------------------------ #
    # Перетаскивание
    # ------------------------------------------------------------------ #
    def begin(
        self,
        game_object: GameObject,
        camera: Camera,
        x: float,
        y: float,
        width: int,
        height: int,
    ) -> bool:
        """Начать перетаскивание, если курсор над осью.

        Returns:
            ``True``, если ось захвачена (камера редактора вращать не нужно).
        """
        index = self.pick(game_object, camera, x, y, width, height)
        if index is None:
            return False
        transform = game_object.transform
        origin = self.origin_of(game_object)
        direction = self.axis_direction(game_object, index, self.mode)
        ray_origin, ray_direction = camera.ray_from_screen(x, y, width, height)
        drag = _Drag(
            axis=index,
            mode=self.mode,
            origin=origin,
            direction=direction,
            start_position=transform.position,
            start_rotation=transform.rotation,
            start_scale=transform.scale,
        )
        if self.mode is GizmoMode.ROTATE:
            hit = _ray_plane(ray_origin, ray_direction, origin, direction)
            if hit is None:
                return False
            drag.start_vector = (hit - origin).normalized
        else:
            parameter = closest_axis_parameter(
                ray_origin, ray_direction, origin, direction
            )
            if parameter is None:
                return False
            drag.start_parameter = parameter
        self._drag = drag
        return True

    def drag(
        self,
        game_object: GameObject,
        camera: Camera,
        x: float,
        y: float,
        width: int,
        height: int,
    ) -> bool:
        """Применить смещение курсора к трансформу объекта.

        Returns:
            ``True``, если трансформ изменился.
        """
        drag = self._drag
        if drag is None:
            return False
        ray_origin, ray_direction = camera.ray_from_screen(x, y, width, height)
        transform = game_object.transform
        if drag.mode is GizmoMode.MOVE:
            parameter = closest_axis_parameter(
                ray_origin, ray_direction, drag.origin, drag.direction
            )
            if parameter is None:
                return False
            world = drag.origin + drag.direction * (parameter - drag.start_parameter)
            parent = transform.parent
            transform.position = (
                parent.world_to_local(world) if parent is not None else world
            )
            return True
        if drag.mode is GizmoMode.SCALE:
            parameter = closest_axis_parameter(
                ray_origin, ray_direction, drag.origin, drag.direction
            )
            if parameter is None or abs(drag.start_parameter) < 1e-3:
                return False
            factor = max(parameter / drag.start_parameter, 0.0)
            values = [drag.start_scale.x, drag.start_scale.y, drag.start_scale.z]
            values[drag.axis] = max(values[drag.axis] * factor, MIN_SCALE)
            transform.scale = Vector3(*values)
            return True
        hit = _ray_plane(ray_origin, ray_direction, drag.origin, drag.direction)
        if hit is None:
            return False
        current = (hit - drag.origin).normalized
        sine = drag.direction.dot(drag.start_vector.cross(current))
        cosine = drag.start_vector.dot(current)
        angle = math.degrees(math.atan2(sine, cosine))
        delta = Quaternion.from_axis_angle(drag.direction, angle)
        transform.rotation = (delta * drag.start_rotation).normalized
        return True

    def end(self) -> bool:
        """Завершить перетаскивание.

        Returns:
            ``True``, если перетаскивание действительно шло.
        """
        was_dragging = self._drag is not None
        self._drag = None
        return was_dragging

    # ------------------------------------------------------------------ #
    # Отрисовка
    # ------------------------------------------------------------------ #
    def axis_matrices(
        self, game_object: GameObject, camera: Camera
    ) -> list[tuple[int, np.ndarray]]:
        """Матрицы моделей трёх осей: ``(номер оси, матрица)``."""
        origin = self.origin_of(game_object)
        size = self.screen_scale(origin, camera)
        scale = Vector3(size, size, size)
        matrices: list[tuple[int, np.ndarray]] = []
        for index in range(3):
            rotation = _axis_rotation(index)
            if self.mode is GizmoMode.SCALE:
                rotation = _world_rotation(game_object) * rotation
            matrices.append(
                (index, compose_matrix(position=origin, rotation=rotation, scale=scale))
            )
        return matrices

    def color_of(self, index: int) -> Vector3:
        """Цвет оси (захваченная подсвечивается)."""
        if self.active_axis == index:
            return ACTIVE_COLOR
        return AXIS_COLORS[index]


def _world_rotation(game_object: GameObject) -> Quaternion:
    """Мировой поворот объекта (учитывает родителей)."""
    rotation = game_object.transform.rotation
    parent = game_object.transform.parent
    while parent is not None:
        rotation = parent.rotation * rotation
        parent = parent.parent
    return rotation
