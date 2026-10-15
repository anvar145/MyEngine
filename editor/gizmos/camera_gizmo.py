"""Гизмо камеры: иконка, усечённый объём и выбор мышью.

Гизмо строится целиком из данных камеры и не хранит состояние сцены, поэтому
его можно создавать «на лету» для любой камеры — редакторской, игровой или
камеры компонента.

Что рисуется:

* **иконка** — каркасный куб постоянного экранного размера с «объективом»
  (пирамида вдоль оси взгляда камеры), как в привычных 3D-редакторах;
* **фрустум** — отрезки между восемью углами усечённого объёма; при
  выделении камеры он рисуется всегда, иначе — только по флагу
  ``always_show_frustum`` (как гизмо выбранной камеры в Unity).

Всё, что возвращает гизмо, — это элементы очереди отрисовки рендерера
``(меш, материал, матрица, примитив)`` и сами меши линий; GPU-ресурсы
кэшируются, потому что геометрия иконки и фрустума неизменна.

Example:
    >>> gizmo = CameraGizmo(camera, selected=True)
    >>> for entry in gizmo.draw_items(icon_mat, frustum_mat):
    ...     renderer.submit(*entry)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from engine.core.math3d import Vector3, compose_matrix
from engine.rendering.camera import Camera

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.rendering.material import Material
    from engine.rendering.mesh import GpuMesh

__all__ = [
    "CameraGizmo",
    "CameraGizmoStyle",
    "camera_icon_lines",
    "frustum_lines",
]

#: Индексные пары углов усечённого объёма (см. :meth:`Camera.frustum_corners`).
_FRUSTUM_EDGES: tuple[tuple[int, int], ...] = (
    # ближний прямоугольник
    (0, 1), (1, 2), (2, 3), (3, 0),
    # дальний прямоугольник
    (4, 5), (5, 6), (6, 7), (7, 4),
    # рёбра между плоскостями
    (0, 4), (1, 5), (2, 6), (3, 7),
)

#: Рёбра каркасного куба иконки.
_BoxEdge = tuple[tuple[float, float, float], tuple[float, float, float]]
_ICON_BOX_EDGES: tuple[_BoxEdge, ...] = (
    ((-1, -1, -1), (1, -1, -1)),
    ((1, -1, -1), (1, 1, -1)),
    ((1, 1, -1), (-1, 1, -1)),
    ((-1, 1, -1), (-1, -1, -1)),
    ((-1, -1, 1), (1, -1, 1)),
    ((1, -1, 1), (1, 1, 1)),
    ((1, 1, 1), (-1, 1, 1)),
    ((-1, 1, 1), (-1, -1, 1)),
    ((-1, -1, -1), (-1, -1, 1)),
    ((1, -1, -1), (1, -1, 1)),
    ((1, 1, -1), (1, 1, 1)),
    ((-1, 1, -1), (-1, 1, 1)),
)


@dataclass(slots=True)
class CameraGizmoStyle:
    """Оформление гизмо камеры.

    Attributes:
        icon_size: Размер иконки в мировых единицах на референсной дистанции.
        icon_distance: Дистанция, на которой иконка имеет номинальный размер.
        min_icon_size: Минимальный мировой размер иконки (чтобы не исчезала).
        always_show_frustum: Рисовать ли фрустум у невыделенной камеры.
        show_lens: Рисовать ли «объектив» — пирамиду вдоль оси взгляда.
        frustum_alpha: Прозрачность линий фрустума (0..1).
        icon_color: Цвет иконки невыделенной камеры.
        selected_color: Цвет выделенной камеры.
        frustum_color: Цвет линий фрустума.
    """

    icon_size: float = 0.5
    icon_distance: float = 10.0
    min_icon_size: float = 0.05
    always_show_frustum: bool = False
    show_lens: bool = True
    frustum_alpha: float = 0.35
    icon_color: tuple[float, float, float] = (0.62, 0.68, 0.78)
    selected_color: tuple[float, float, float] = (1.0, 0.72, 0.24)
    frustum_color: tuple[float, float, float] = (0.55, 0.62, 0.74)


@dataclass(slots=True)
class CameraGizmo:
    """Гизмо одной камеры: иконка и усечённый объём.

    Args:
        camera: Камера, которую показывает гизмо.
        selected: Выделена ли камера в редакторе.
        style: Оформление (цвета и размеры).
        name: Имя для подписи в сцене (по умолчанию — имя камеры).
    """

    camera: Camera
    selected: bool = False
    style: CameraGizmoStyle = field(default_factory=CameraGizmoStyle)
    name: str | None = None
    _icon_mesh: GpuMesh | None = field(default=None, repr=False, compare=False)
    _frustum_mesh: GpuMesh | None = field(default=None, repr=False, compare=False)
    _frustum_key: tuple = field(default=(), repr=False, compare=False)

    # ------------------------------------------------------------------ #
    # Идентичность и цвет
    # ------------------------------------------------------------------ #
    @property
    def title(self) -> str:
        """Подпись гизмо (для оверлея редактора)."""
        return self.name or self.camera.name

    @property
    def color(self) -> Vector3:
        """Цвет гизмо: акцентный для выделенной камеры, серый для остальных."""
        source = self.style.selected_color if self.selected else self.style.icon_color
        return Vector3(*source)

    @property
    def frustum_color(self) -> Vector3:
        """Цвет линий усечённого объёма."""
        return Vector3(*self.style.frustum_color)

    # ------------------------------------------------------------------ #
    # Геометрия
    # ------------------------------------------------------------------ #
    @property
    def position(self) -> Vector3:
        """Мировая позиция камеры."""
        return Vector3(
            self.camera.position.x, self.camera.position.y, self.camera.position.z
        )

    @property
    def forward(self) -> Vector3:
        """Направление взгляда камеры."""
        return self.camera.forward

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Матрица ориентации камеры без масштаба."""
        return compose_matrix(rotation=self.camera.rotation)

    def icon_scale(self, viewer_position: Vector3) -> float:
        """Подобрать мировой размер иконки для постоянного экранного размера.

        Args:
            viewer_position: Позиция наблюдателя (редакторская камера).

        Returns:
            Масштаб иконки, не меньше ``style.min_icon_size``.
        """
        distance = max((self.position - viewer_position).length, 1e-3)
        scale = self.style.icon_size * distance / max(self.style.icon_distance, 1e-3)
        return max(scale, self.style.min_icon_size)

    def icon_matrix(self, viewer_position: Vector3) -> np.ndarray:
        """Матрица модели иконки камеры.

        Иконка всегда одного экранного размера и разворачивается по
        ориентации камеры, поэтому она читается как «камера», а не как точка.
        """
        scale = self.icon_scale(viewer_position)
        return compose_matrix(
            position=self.position,
            rotation=self.camera.rotation,
            scale=Vector3(scale, scale, scale),
        )

    def lens_matrix(self, viewer_position: Vector3) -> np.ndarray:
        """Матрица модели «объектива» (пирамиды вдоль оси взгляда)."""
        scale = self.icon_scale(viewer_position)
        return compose_matrix(
            position=self.position,
            rotation=self.camera.rotation,
            scale=Vector3(scale, scale, scale),
        )

    def frustum_line_points(self, *, far_scale: float = 1.0) -> np.ndarray:
        """Собрать вершины линий усечённого объёма камеры.

        Args:
            far_scale: Множитель дальней плоскости (доля, если нужно укоротить).

        Returns:
            Массив ``(24, 3)`` float32: 12 отрезков по две вершины.
        """
        corners = np.asarray(self.camera.frustum_corners(), dtype=np.float32)
        if far_scale != 1.0:
            # Масштабируем дальние углы относительно позиции камеры.
            eye = np.array(
                (
                    self.camera.position.x,
                    self.camera.position.y,
                    self.camera.position.z,
                ),
                dtype=np.float32,
            )
            corners[4:] = eye + (corners[4:] - eye) * float(far_scale)
        points = np.empty((len(_FRUSTUM_EDGES) * 2, 3), dtype=np.float32)
        for index, (start, end) in enumerate(_FRUSTUM_EDGES):
            points[index * 2] = corners[start]
            points[index * 2 + 1] = corners[end]
        return points

    # ------------------------------------------------------------------ #
    # Очередь отрисовки
    # ------------------------------------------------------------------ #
    def icon_mesh(self) -> GpuMesh:
        """Меш иконки камеры (кэшируется, геометрия неизменна)."""
        if self._icon_mesh is None:
            from engine.core.resources import MeshAsset
            from engine.rendering.mesh import GpuMesh

            points = camera_icon_lines(1.0, lens=self.style.show_lens)
            asset = MeshAsset(
                path=_procedural_path("camera_icon"),
                positions=points,
                indices=None,
                normals=None,
                uvs=None,
                name="camera_icon",
            )
            self._icon_mesh = GpuMesh.from_mesh_asset(asset)
        return self._icon_mesh

    def frustum_mesh(self, *, far_scale: float = 1.0) -> GpuMesh:
        """Меш линий усечённого объёма.

        Меш пересобирается только при изменении параметров проекции: пока
        камера не двигалась и не меняла FOV, переиспользуется готовая
        геометрия — это заметно на слабых драйверах.
        """
        key = (
            round(self.camera.fov, 4),
            round(self.camera.near, 4),
            round(self.camera.far, 4),
            round(self.camera.ortho_height, 4),
            round(self.camera.effective_aspect, 4),
            self.camera.mode,
            round(float(far_scale), 4),
        )
        if self._frustum_mesh is None or self._frustum_key != key:
            from engine.core.resources import MeshAsset
            from engine.rendering.mesh import GpuMesh

            points = self.frustum_line_points(far_scale=far_scale)
            asset = MeshAsset(
                path=_procedural_path("camera_frustum"),
                positions=points,
                indices=None,
                normals=None,
                uvs=None,
                name="camera_frustum",
            )
            self._frustum_mesh = GpuMesh.from_mesh_asset(asset)
            self._frustum_key = key
        return self._frustum_mesh

    def draw_items(
        self,
        icon_material: Material,
        frustum_material: Material | None = None,
        *,
        viewer_position: Vector3 | None = None,
        show_frustum: bool | None = None,
        far_scale: float = 1.0,
    ) -> list[tuple[GpuMesh, Material, np.ndarray, int]]:
        """Собрать элементы очереди отрисовки для гизмо.

        Args:
            icon_material: Материал иконки (цвет уже задан гизмо).
            frustum_material: Материал линий фрустума; ``None`` — не рисовать.
            viewer_position: Позиция наблюдателя для расчёта размера иконки.
            show_frustum: Показывать ли фрустум; ``None`` — по выделению и
                настройке ``always_show_frustum``.
            far_scale: Множитель дальней плоскости.

        Returns:
            Список ``(меш, материал, матрица модели, примитив GL)``.
        """
        from OpenGL import GL

        viewer = viewer_position or self.position
        items: list[tuple[GpuMesh, Material, np.ndarray, int]] = [
            (self.icon_mesh(), icon_material, self.icon_matrix(viewer), GL.GL_LINES)
        ]
        if show_frustum is None:
            visible = self.selected or self.style.always_show_frustum
        else:
            visible = bool(show_frustum)
        if visible and frustum_material is not None:
            items.append(
                (
                    self.frustum_mesh(far_scale=far_scale),
                    frustum_material,
                    compose_matrix(),
                    GL.GL_LINES,
                )
            )
        return items

    # ------------------------------------------------------------------ #
    # Выбор мышью
    # ------------------------------------------------------------------ #
    def pick(
        self,
        ray_origin: Vector3,
        ray_direction: Vector3,
        viewer_position: Vector3 | None = None,
        *,
        tolerance: float = 1.25,
    ) -> bool:
        """Проверить, попал ли луч выбора в иконку камеры.

        Расстояние считается от центра камеры до ближайшей точки луча, а
        допуск пропорционален экранному размеру иконки: клик по иконке с
        любого расстояния попадает примерно одинаково уверенно.

        Args:
            ray_origin: Начало луча (обычно позиция камеры редактора).
            ray_direction: Направление луча (нормализуется внутри).
            viewer_position: Позиция наблюдателя для масштаба иконки.
            tolerance: Множитель радиуса иконки.

        Returns:
            ``True``, если иконка выбрана.
        """
        direction = ray_direction.normalized
        if direction.is_zero:
            return False
        to_center = self.position - ray_origin
        along = to_center.dot(direction)
        if along < 0.0:
            # Иконка за спиной наблюдателя.
            return False
        closest = ray_origin + direction * along
        radius = self.icon_scale(viewer_position or ray_origin) * tolerance
        return (closest - self.position).length <= radius

    def __repr__(self) -> str:
        state = "выделено" if self.selected else "обычное"
        return f"<CameraGizmo '{self.title}' {state} pos={self.position}>"


# --------------------------------------------------------------------------- #
# Генераторы линий
# --------------------------------------------------------------------------- #
def camera_icon_lines(size: float = 1.0, *, lens: bool = True) -> np.ndarray:
    """Собрать вершины иконки камеры для ``GL_LINES``.

    Иконка состоит из каркасного куба (корпус) и пирамиды вдоль локального
    ``+Z`` (объектив). Геометрия симметрична относительно начала координат,
    поэтому её удобно масштабировать матрицей модели.

    Args:
        size: Длина ребра корпуса.
        lens: Добавлять ли «объектив».

    Returns:
        Массив ``(N, 3)`` float32: пары вершин образуют отрезки.
    """
    half = float(size) * 0.5
    points: list[tuple[float, float, float]] = []
    for start, end in _ICON_BOX_EDGES:
        points.append((start[0] * half, start[1] * half, start[2] * half))
        points.append((end[0] * half, end[1] * half, end[2] * half))

    if lens:
        # Пирамида: вершина впереди, основание — передняя грань корпуса.
        base = (
            (-half, -half, half),
            (half, -half, half),
            (half, half, half),
            (-half, half, half),
        )
        apex = (0.0, 0.0, half * 2.0)
        for corner in base:
            points.append(corner)
            points.append(apex)
        # Диагонали основания делают «объектив» читаемым под любым углом.
        points.append(base[0])
        points.append(base[2])
        points.append(base[1])
        points.append(base[3])

    return np.asarray(points, dtype=np.float32)


def frustum_lines(camera: Camera, *, far_scale: float = 1.0) -> np.ndarray:
    """Собрать вершины линий усечённого объёма камеры для ``GL_LINES``.

    Это удобная обёртка над :meth:`CameraGizmo.frustum_line_points`: она
    подходит и для визуализации фрустума выбранной камеры, и для отладки
    планов отсечения без создания гизмо.

    Args:
        camera: Камера, чей объём нужно нарисовать.
        far_scale: Множитель дальней плоскости.

    Returns:
        Массив ``(24, 3)`` float32: 12 отрезков по две вершины.
    """
    return CameraGizmo(camera).frustum_line_points(far_scale=far_scale)


def _procedural_path(name: str):
    """Виртуальный путь для процедурного меша (как в ``resources.primitives``)."""
    from pathlib import Path

    return Path(f"<procedural>/gizmo/{name}")
