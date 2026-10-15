"""Камеры MyEngine: компонент, стек, усечённый объём и редакторское управление.

Модуль разделён на четыре смысловые части:

* :class:`Camera` — сам компонент камеры: тип проекции, параметры отсечения,
  прямоугольник вьюпорта, цвет очистки, маска отсечения и матрицы
  (``view``/``projection``) с кэшами;
* :class:`Frustum` — усечённый объём, извлекаемый из ``view_projection``; один
  объект переиспользуется камерой, поэтому отсечение не аллоцирует память;
* :class:`CameraComponent` и :class:`CameraStack` — привязка камеры к
  :class:`~engine.core.game_object.GameObject`, несколько камер и активная;
* :class:`EditorCameraController` и :class:`EditorCameraSession` — орбитальное
  управление камерой редактора, фокус на объект и сохранение вида между
  сессиями.

Соглашения движка: камера смотрит вдоль локального ``+Z``, матрицы хранятся
построчно (row-major) со сдвигом в последнем столбце, ``v' = M @ v``.

Example:
    >>> camera = Camera(fov=50.0, near=0.1, far=300.0)
    >>> camera.position = Vector3(4.0, 3.0, 4.0)
    >>> camera.look_at(Vector3.zero())
    >>> camera.get_view_matrix().shape
    (4, 4)
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from engine.core.component import Component
from engine.core.math3d import (
    EPSILON,
    Quaternion,
    Vector3,
    extract_translation,
    look_at_matrix,
    matrix_to_quaternion,
    orthographic_matrix,
    perspective_matrix,
    transform_point,
)

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.transform import Transform
    from engine.rendering.render_graph import RenderTarget
    from engine.rendering.renderer import Renderer, RenderStats

__all__ = [
    "ASPECT_FIXED",
    "ASPECT_VIEWPORT",
    "Camera",
    "CameraComponent",
    "CameraPreview",
    "CameraStack",
    "EditorCameraController",
    "EditorCameraSession",
    "EditorCameraState",
    "Frustum",
    "OrbitCameraController",
    "PLANE_NAMES",
    "PROJECTION_MODES",
    "ProjectionMode",
    "quaternion_from_forward",
]

_log = logging.getLogger("engine.rendering.camera")

#: Отношение сторон следует за размерами области вывода (обычный режим).
ASPECT_VIEWPORT = "viewport"

#: Отношение сторон закреплено за камерой (рендер в текстуру, предпросмотр).
ASPECT_FIXED = "fixed"

#: Порядок плоскостей усечённого объёма (:meth:`Frustum.planes`).
PLANE_NAMES = ("left", "right", "bottom", "top", "near", "far")


#: Сколько слоёв поддерживает маска отсечения (бит на слой).
LAYER_COUNT = 32

#: Маска по умолчанию: пропускает все 32 слоя.
ALL_LAYERS_MASK = (1 << LAYER_COUNT) - 1


class ProjectionMode:
    """Режим проекции камеры."""

    PERSPECTIVE = "perspective"
    ORTHOGRAPHIC = "orthographic"


#: Поддерживаемые режимы проекции (для инспектора и валидации).
PROJECTION_MODES: tuple[str, ...] = (
    ProjectionMode.PERSPECTIVE,
    ProjectionMode.ORTHOGRAPHIC,
)


def quaternion_from_forward(forward: Vector3, up: Vector3 | None = None) -> Quaternion:
    """Построить кватернион ориентации по направлению взгляда.

    Камера смотрит вдоль локальной оси ``+Z`` (как и объекты движка).

    Args:
        forward: Направление взгляда (не обязательно нормализованное).
        up: Вектор «вверх»; по умолчанию мировой ``+Y``.

    Returns:
        Кватернион, поворачивающий ``+Z`` в направление ``forward``.

    Note:
        Для разворота на 180° есть целое семейство решений (любой поворот
        вокруг перпендикулярной оси), и выбор «на глаз» ломает ориентацию:
        базис становится зеркальным — ``+X`` камеры уезжает налево, и вся
        картинка отражается по горизонтали. Ось выбирается из вектора
        «вверх», поэтому базис остаётся правым.
    """
    direction = forward.normalized
    if direction.is_zero:
        return Quaternion.identity()
    up_vector = up or Vector3.up()

    # Look-rotation: базис (right, up, forward) с горизонтальным right, поэтому
    # крен не накапливается (кратчайшая дуга от +Z наклоняла камеру).
    right = up_vector.cross(direction)
    if right.length > 1e-6:
        right = right.normalized
        true_up = direction.cross(right)
        basis = np.array(
            [
                [right.x, true_up.x, direction.x],
                [right.y, true_up.y, direction.y],
                [right.z, true_up.z, direction.z],
            ],
            dtype=np.float64,
        )
        return Quaternion.from_matrix(basis).normalized

    # Взгляд вдоль вектора «вверх»: запасной путь через кратчайшую дугу.
    dot = Vector3.forward().dot(direction)
    if dot > 1.0 - 1e-6:
        return Quaternion.identity()
    if dot < -1.0 + 1e-6:
        # Разворот на 180°: осью берём проекцию «вверх» на плоскость,
        # перпендикулярную направлению взгляда — тогда камера не «перевернётся
        # через бок» (иначе горизонталь отражается зеркально).
        axis = (up_vector - direction * up_vector.dot(direction)).normalized
        if axis.is_zero:
            # Взгляд строго вдоль вектора «вверх»: ось любой ориентации.
            axis = (
                Vector3.right()
                if abs(up_vector.x) < 0.9
                else Vector3.forward()
            )
        return Quaternion.from_axis_angle(axis, 180.0)
    axis = Vector3.forward().cross(direction).normalized
    angle_degrees = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
    return Quaternion.from_axis_angle(axis, angle_degrees)


# --------------------------------------------------------------------------- #
# Усечённый объём
# --------------------------------------------------------------------------- #
class Frustum:
    """Усечённый объём камеры: шесть плоскостей и восемь углов.

    Объект живёт вместе с камерой и обновляется на месте
    (:meth:`Frustum.update`), поэтому отсечение тысяч объектов за кадр не
    создаёт мусора для сборщика.

    Attributes:
        version: Номер ревизии: увеличивается при каждом обновлении.
        planes: Массив ``(6, 4)`` нормализованных плоскостей.
        corners: Массив ``(8, 3)`` углов дальнего и ближнего прямоугольников.
    """

    __slots__ = (
        "_corners",
        "_corners_valid",
        "_dirty",
        "_inverse_view_projection",
        "_planes",
        "_plane_rows",
        "_sphere_buffer",
        "_version",
    )

    def __init__(self) -> None:
        self._planes: np.ndarray = np.zeros((6, 4), dtype=float)
        self._plane_rows: list[list[float]] | None = None
        self._corners: np.ndarray = np.zeros((8, 3), dtype=float)
        self._version = 0
        self._dirty = True
        self._corners_valid = False
        self._inverse_view_projection: np.ndarray | None = None
        self._sphere_buffer = np.empty(4, dtype=float)

    # ------------------------------------------------------------------ #
    # Данные
    # ------------------------------------------------------------------ #
    @property
    def planes(self) -> np.ndarray:
        """Массив ``(6, 4)`` нормализованных плоскостей отсечения."""
        return self._planes

    @property
    def corners(self) -> np.ndarray:
        """Массив ``(8, 3)`` углов объёма (после :meth:`update`)."""
        return self._corners

    @property
    def version(self) -> int:
        """Номер ревизии: растёт при каждом пересчёте плоскостей."""
        return self._version


    # ------------------------------------------------------------------ #
    # Обновление
    # ------------------------------------------------------------------ #
    def invalidate(self) -> None:
        """Пометить объём устаревшим (матрицы камеры изменились)."""
        self._dirty = True
        self._corners_valid = False
        self._inverse_view_projection = None

    def update(
        self, view_projection: np.ndarray, near: float, far: float
    ) -> Frustum:
        """Пересчитать плоскости из матрицы ``view_projection``.

        Args:
            view_projection: Произведение проекции и вида ``(4, 4)``.
            near: Ближняя плоскость (нужна для углов объёма).
            far: Дальняя плоскость (нужна для углов объёма).

        Returns:
            Сам объект (для цепочки вызовов).
        """
        matrix = np.asarray(view_projection, dtype=float).reshape(4, 4)
        # Gribb–Hartmann: строки матрицы дают коэффициенты плоскостей.
        self._planes[0] = matrix[3] + matrix[0]  # left
        self._planes[1] = matrix[3] - matrix[0]  # right
        self._planes[2] = matrix[3] + matrix[1]  # bottom
        self._planes[3] = matrix[3] - matrix[1]  # top
        self._planes[4] = matrix[3] + matrix[2]  # near
        self._planes[5] = matrix[3] - matrix[2]  # far
        lengths = np.linalg.norm(self._planes[:, :3], axis=1, keepdims=True)
        lengths[lengths < EPSILON] = 1.0
        self._planes /= lengths
        self._plane_rows = None

        self._corners_valid = False
        try:
            self._inverse_view_projection = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:  # pragma: no cover - вырожденная матрица
            self._inverse_view_projection = None
        self._dirty = False
        self._version += 1
        self._update_corners(near, far)
        return self

    def _update_corners(self, near: float, far: float) -> None:
        """Посчитать восемь углов объёма через обратную матрицу."""
        inverse = self._inverse_view_projection
        if inverse is None:
            return
        # Углы в NDC: (x, y) — знаки, z — ближняя (-1) и дальняя (+1) плоскости.
        ndc = np.empty((8, 4), dtype=float)
        for index, (x, y) in enumerate(
            ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
        ):
            ndc[index] = (x, y, -1.0, 1.0)
            ndc[index + 4] = (x, y, 1.0, 1.0)
        world = ndc @ inverse.T
        w = world[:, 3:4]
        w[np.abs(w) < EPSILON] = 1.0
        self.corners[:] = world[:, :3] / w
        self._corners_valid = True

    @property
    def is_dirty(self) -> bool:
        """``True``, если объём не пересчитывался после изменения камеры."""
        return self._dirty

    # ------------------------------------------------------------------ #
    # Тесты видимости
    # ------------------------------------------------------------------ #
    def contains_point(self, point: Vector3) -> bool:
        """``True``, если точка внутри всех шести полупространств."""
        data = self._sphere_buffer
        data[0], data[1], data[2], data[3] = point.x, point.y, point.z, 1.0
        return bool(np.all(self.planes @ data >= 0.0))

    def intersects_sphere(self, center: Vector3, radius: float) -> bool:
        """``True``, если сфера видна или пересекает границу объёма.

        Args:
            center: Центр сферы в мире.
            radius: Радиус сферы (отрицательный считается нулевым).
        """
        # Шесть плоскостей на чистых float: для одной сферы numpy-вызовы дороже.
        rows = self._plane_rows
        if rows is None:
            rows = self._plane_rows = self._planes.tolist()
        x, y, z = center.x, center.y, center.z
        limit = -max(0.0, radius)
        return all(a * x + b * y + c * z + d >= limit for a, b, c, d in rows)

    def intersects_aabb(self, minimum: Vector3, maximum: Vector3) -> bool:
        """``True``, если AABB пересекает усечённый объём.

        Реализован как «p-vertex»-тест: для каждой плоскости берётся самая
        дальняя по нормали вершина бокса.
        """
        center = (minimum + maximum) * 0.5
        half = (maximum - minimum) * 0.5
        normals = self.planes[:, :3]
        offsets = self.planes[:, 3]
        # Проекция полуразмеров на нормаль плоскости.
        projected = np.abs(normals) @ np.array((half.x, half.y, half.z), dtype=float)
        signed = normals @ np.array((center.x, center.y, center.z), dtype=float)
        return bool(np.all(signed + projected + offsets >= 0.0))

    def plane_tests(self, center: Vector3, radius: float) -> np.ndarray:
        """Вернуть расстояния центра сферы до шести плоскостей.

        Полезно для отладки отсечения и для гистограмм в панели статистики.
        """
        data = self._sphere_buffer
        data[0], data[1], data[2], data[3] = center.x, center.y, center.z, 1.0
        return self.planes @ data

    def __repr__(self) -> str:
        state = "грязный" if self._dirty else f"v{self.version}"
        return f"<Frustum {state} углов={8 if self._corners_valid else 0}>"


# --------------------------------------------------------------------------- #
# Камера
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Camera:
    """Компонент камеры: проекция, вьюпорт, очистка и матрицы.

    Args:
        fov: Вертикальный угол обзора в градусах (перспектива).
        near: Ближняя плоскость отсечения.
        far: Дальняя плоскость отсечения.
        aspect: Отношение ширины к высоте; :data:`ASPECT_AUTO` использует
            прямоугольник вьюпорта.
        position: Позиция камеры в мире.
        rotation: Ориентация (вперёд — локальный ``+Z``).
        mode: Режим проекции (:class:`ProjectionMode`).
        ortho_height: Высота ортографического объёма в мировых единицах.
        viewport_rect: Прямоугольник вывода ``(x, y, width, height)`` в пикселях.
        clear_color: Цвет очистки фона ``(r, g, b, a)``.
        clear_flags: Что очищать: ``("color", "depth")`` по умолчанию.
        culling_mask: Битовая маска слоёв: объект рисуется, если
            ``layer & culling_mask`` не ноль.
        depth: Порядок в стеке камер (меньше — раньше).
        name: Имя камеры (для UI и сохранения состояния).
        enabled: Участвует ли камера в рендере.

    Attributes:
        frustum: Усечённый объём; обновляется лениво и переиспользуется.
        version: Счётчик изменений — растёт при любой правке, влияющей на
            матрицы (по нему рендерер понимает, что кэш uniform устарел).
    """

    aspect_mode: str = ASPECT_VIEWPORT
    viewport_rect: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    clear_color: tuple[float, float, float, float] = (0.078, 0.075, 0.071, 1.0)
    clear_flags: tuple[str, ...] = ("color", "depth")
    culling_mask: int = ALL_LAYERS_MASK
    depth: int = 0
    name: str = "Main Camera"
    enabled: bool = True
    frustum: Frustum = field(default_factory=Frustum, repr=False, compare=False)
    # ``fov``, ``near``, ``far``, ``aspect``, ``mode`` и ``ortho_height``
    # объявлены ниже свойствами: их фактические значения лежат в приватных
    # слотах, а запись обязана сбрасывать кэш матриц.
    _fov: float = field(default=60.0, repr=False, compare=False)
    _near: float = field(default=0.1, repr=False, compare=False)
    _far: float = field(default=500.0, repr=False, compare=False)
    _mode: str = field(
        default=ProjectionMode.PERSPECTIVE, repr=False, compare=False
    )
    _ortho_height: float = field(default=10.0, repr=False, compare=False)
    _aspect: float = field(default=16.0 / 9.0, repr=False, compare=False)
    _position: Vector3 = field(
        default_factory=Vector3.zero, repr=False, compare=False
    )
    _rotation: Quaternion = field(
        default_factory=Quaternion.identity, repr=False, compare=False
    )
    _viewport_size: tuple[int, int] = field(
        default=(0, 0), repr=False, compare=False
    )
    _view_cache: np.ndarray | None = field(default=None, repr=False, compare=False)
    _projection_cache: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )
    _projection_key: tuple = field(default=(), repr=False, compare=False)
    _view_projection_cache: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )
    _view_projection_inverse: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )
    _version: int = field(default=0, repr=False, compare=False)

    def __init__(
        self,
        fov: float = 60.0,
        near: float = 0.1,
        far: float = 500.0,
        aspect: float = 16.0 / 9.0,
        mode: str = ProjectionMode.PERSPECTIVE,
        ortho_height: float = 10.0,
        viewport_rect: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        clear_color: tuple[float, float, float, float] = (0.078, 0.075, 0.071, 1.0),
        clear_flags: tuple[str, ...] = ("color", "depth"),
        culling_mask: int = ALL_LAYERS_MASK,
        depth: int = 0,
        name: str = "Main Camera",
        enabled: bool = True,
        aspect_mode: str = ASPECT_VIEWPORT,
        **kwargs: Any,
    ) -> None:
        # ``position`` и ``rotation`` остаются именованными аргументами (как
        # раньше), но хранятся в приватных полях: смена значения должна
        # сбрасывать кэш матриц, а за это отвечает сеттер-свойство.
        position = kwargs.pop("position", None)
        rotation = kwargs.pop("rotation", None)
        if kwargs:  # pragma: no cover - защита от опечаток в вызовах
            raise TypeError(
                f"Camera() получила неожидаемые аргументы: {sorted(kwargs)}"
            )

        # Служебные поля заполняются первыми: сеттеры ниже сбрасывают кэши, а
        # ``invalidate`` обращается к усечённому объёму.
        self.frustum = Frustum()
        self._viewport_size = (0, 0)
        self._view_cache = None
        self._projection_cache = None
        self._projection_key = ()
        self._view_projection_cache = None
        self._view_projection_inverse = None
        self._version = 0
        self._position = Vector3.zero()
        self._rotation = Quaternion.identity()
        self._fov = 60.0
        self._near = 0.1
        self._far = 500.0
        self._mode = ProjectionMode.PERSPECTIVE
        self._ortho_height = 10.0

        self.fov = float(fov)
        self.near = float(near)
        self.far = float(far)
        self.aspect = float(aspect)
        self.aspect_mode = str(aspect_mode)
        self.mode = str(mode)
        self.ortho_height = float(ortho_height)
        self.viewport_rect = tuple(viewport_rect)  # type: ignore[assignment]
        self.clear_color = tuple(clear_color)  # type: ignore[assignment]
        self.clear_flags = tuple(str(flag) for flag in clear_flags)
        self.culling_mask = int(culling_mask)
        self.depth = int(depth)
        self.name = str(name)
        self.enabled = bool(enabled)
        if position is not None:
            self.position = Vector3.from_iterable(position)
        if rotation is not None:
            self.rotation = rotation

    # ------------------------------------------------------------------ #
    # Параметры проекции
    #
    # Все они объявлены свойствами: изменение любого параметра обязано
    # сбросить кэш матриц, иначе отсечение и отрисовка работали бы по
    # устаревшей матрице (самая коварная ошибка в горячем цикле).
    # ------------------------------------------------------------------ #
    @property
    def fov(self) -> float:
        """Вертикальный угол обзора в градусах (перспектива)."""
        return self._fov

    @fov.setter
    def fov(self, value: float) -> None:
        self._fov = float(value)
        self.invalidate()

    @property
    def near(self) -> float:
        """Ближняя плоскость отсечения."""
        return self._near

    @near.setter
    def near(self, value: float) -> None:
        self._near = float(value)
        self.invalidate()

    @property
    def far(self) -> float:
        """Дальняя плоскость отсечения."""
        return self._far

    @far.setter
    def far(self, value: float) -> None:
        self._far = float(value)
        self.invalidate()

    @property
    def mode(self) -> str:
        """Режим проекции (:class:`ProjectionMode`)."""
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        self._mode = str(value)
        self.invalidate()

    @property
    def ortho_height(self) -> float:
        """Высота ортографического объёма в мировых единицах."""
        return self._ortho_height

    @ortho_height.setter
    def ortho_height(self, value: float) -> None:
        self._ortho_height = float(value)
        self.invalidate()

    @property
    def aspect(self) -> float:
        """Отношение ширины к высоте, уходящее в матрицу проекции."""
        return self._aspect

    @aspect.setter
    def aspect(self, value: float) -> None:
        self._aspect = float(value)
        self.invalidate()

    def __post_init__(self) -> None:
        """Заглушка: поле ``frustum`` создаётся в :meth:`__init__`."""

    # ------------------------------------------------------------------ #
    # Ориентация
    # ------------------------------------------------------------------ #
    @property
    def position(self) -> Vector3:
        """Позиция камеры в мире."""
        return self._position

    @position.setter
    def position(self, value: Vector3) -> None:
        self._position = Vector3(value.x, value.y, value.z)
        self.invalidate()

    @property
    def rotation(self) -> Quaternion:
        """Ориентация камеры (вперёд — локальный ``+Z``)."""
        return self._rotation

    @rotation.setter
    def rotation(self, value: Quaternion) -> None:
        self._rotation = value
        self.invalidate()

    @property
    def forward(self) -> Vector3:
        """Направление взгляда (мировая ось ``+Z`` камеры)."""
        return self._rotation.rotate_vector(Vector3.forward())

    @property
    def right(self) -> Vector3:
        """Направление вправо (ось ``+X`` камеры)."""
        return self._rotation.rotate_vector(Vector3.right())

    @property
    def up(self) -> Vector3:
        """Направление вверх (ось ``+Y`` камеры)."""
        return self._rotation.rotate_vector(Vector3.up())

    @property
    def version(self) -> int:
        """Счётчик изменений камеры (для кэшей рендерера)."""
        return self._version

    def look_at(self, target: Vector3, up: Vector3 | None = None) -> None:
        """Направить камеру на точку.

        Args:
            target: Точка, куда смотрит камера.
            up: Вектор «вверх»; по умолчанию мировой ``+Y``.
        """
        self.rotation = quaternion_from_forward(target - self._position, up)

    def look_from(
        self, eye: Vector3, target: Vector3, up: Vector3 | None = None
    ) -> None:
        """Поставить камеру в точку и направить её на цель.

        Args:
            eye: Новая позиция камеры.
            target: Точка, куда смотреть.
            up: Вектор «вверх»; по умолчанию мировой ``+Y``.
        """
        self.position = Vector3(eye.x, eye.y, eye.z)
        self.look_at(target, up)

    # ------------------------------------------------------------------ #
    # Вьюпорт
    # ------------------------------------------------------------------ #
    def invalidate(self) -> None:
        """Сбросить кэш матриц (вызывать после изменения параметров)."""
        self._view_cache = None
        self._projection_cache = None
        self._projection_key = ()
        self._view_projection_cache = None
        self._view_projection_inverse = None
        self._version += 1
        frustum = getattr(self, "frustum", None)
        if frustum is not None:
            frustum.invalidate()

    def set_aspect(self, width: int, height: int) -> None:
        """Обновить aspect по размерам вьюпорта.

        В режиме :data:`ASPECT_VIEWPORT` (значение по умолчанию) отношение
        сторон следует за областью вывода: так ведёт себя игровая камера при
        изменении размера окна. В режиме :data:`ASPECT_FIXED` поле ``aspect``
        считается закреплённым и не меняется — это нужно предпросмотру и
        рендеру в текстуру с фиксированным кадром.

        Args:
            width: Ширина области вывода в пикселях.
            height: Высота области вывода в пикселях.
        """
        if height <= 0 or width <= 0:
            return
        self._viewport_size = (int(width), int(height))
        if self.aspect_mode != ASPECT_VIEWPORT:
            return
        aspect = max(1e-4, width / height)
        if abs(aspect - self.aspect) > 1e-9:
            self.aspect = aspect
            self.invalidate()

    def lock_aspect(self) -> Camera:
        """Закрепить текущее отношение сторон за камерой.

        Returns:
            Сам объект (для цепочки вызовов).
        """
        self.aspect_mode = ASPECT_FIXED
        return self

    @property
    def effective_aspect(self) -> float:
        """Отношение сторон, которое реально идёт в матрицу проекции."""
        return max(1e-4, float(self.aspect))

    @property
    def viewport_size(self) -> tuple[int, int]:
        """Последний размер области вывода, о котором знает камера."""
        return self._viewport_size

    def pixel_rect(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Прямоугольник вьюпорта камеры в пикселях заданной области.

        Значения меньше либо равные 1.5 считаются долями области вывода (это
        обычный случай: ``(0, 0, 1, 1)`` — вся область), большие — пикселями.

        Args:
            width: Ширина области вывода.
            height: Высота области вывода.

        Returns:
            Кортеж ``(x, y, width, height)`` в пикселях (y — от нижнего края,
            как принято в OpenGL).
        """
        x, y, w, h = self.viewport_rect
        if abs(w) <= 1.5 and abs(h) <= 1.5:
            return (
                round(x * width),
                round(y * height),
                max(0, round(w * width)),
                max(0, round(h * height)),
            )
        return (
            round(x),
            round(y),
            max(0, round(w)),
            max(0, round(h)),
        )

    @property
    def projection(self) -> str:
        """Псевдоним :attr:`mode` (для сериализации и UI)."""
        return self.mode

    @projection.setter
    def projection(self, value: str) -> None:
        self.mode = value
        self.invalidate()

    @property
    def is_orthographic(self) -> bool:
        """``True``, если включена ортографическая проекция."""
        return self.mode == ProjectionMode.ORTHOGRAPHIC

    def layer_visible(self, layer: int) -> bool:
        """Проверить слой объекта по маске отсечения камеры.

        Слой ``N`` соответствует биту ``N`` маски. Маска трактуется как
        беззнаковая, поэтому значение ``-1`` (или :data:`ALL_LAYERS_MASK`)
        пропускает все 32 слоя.

        Args:
            layer: Номер слоя (0..31).

        Returns:
            ``True``, если объект этого слоя должен рисоваться.
        """
        layer = int(layer)
        if not 0 <= layer < LAYER_COUNT:
            return True
        return bool((1 << layer) & (int(self.culling_mask) & ALL_LAYERS_MASK))

    # ------------------------------------------------------------------ #
    # Матрицы
    # ------------------------------------------------------------------ #
    @property
    def view_matrix(self) -> np.ndarray:
        """Видовая матрица (world → view), кэшируется."""
        if self._view_cache is None:
            forward = self.forward
            up = self.up
            target = self._position + forward
            try:
                self._view_cache = look_at_matrix(self._position, target, up)
            except ValueError:
                # Вырожденный случай (взгляд строго вдоль мирового up):
                # подставляем стабильный «вверх», чтобы кадр не пропал.
                fallback = (
                    Vector3.forward()
                    if abs(up.z) < 0.9
                    else Vector3.right()
                )
                self._view_cache = look_at_matrix(self._position, target, fallback)
        return self._view_cache

    @property
    def projection_matrix(self) -> np.ndarray:
        """Проекционная матрица с учётом текущего режима (кэшируется)."""
        key = (
            self.mode,
            self.effective_aspect,
            self.near,
            self.far,
            self.fov,
            self.ortho_height,
        )
        if self._projection_cache is None or self._projection_key != key:
            if self.mode == ProjectionMode.ORTHOGRAPHIC:
                half_height = max(EPSILON, self.ortho_height * 0.5)
                half_width = half_height * self.effective_aspect
                self._projection_cache = orthographic_matrix(
                    -half_width,
                    half_width,
                    -half_height,
                    half_height,
                    self.near,
                    self.far,
                )
            else:
                self._projection_cache = perspective_matrix(
                    self.fov, self.effective_aspect, self.near, self.far
                )
            self._projection_key = key
        return self._projection_cache

    @property
    def view_projection_matrix(self) -> np.ndarray:
        """Произведение проекции и вида (готово для шейдера)."""
        if self._view_projection_cache is None:
            self._view_projection_cache = self.projection_matrix @ self.view_matrix
        return self._view_projection_cache

    @property
    def inverse_view_projection(self) -> np.ndarray:
        """Обратная матрица ``view_projection`` (world ↔ screen).

        Матрица собирается заново в double и обращается аналитически (см.
        :func:`invert_view_projection`). Так делать необходимо по двум
        причинам:

        * движок хранит матрицы в float32, и произведение ``projection @ view``
          теряет точность на порядке ``1/(far - near)`` — обращение такого
          произведения даёт видимую ошибку (точка «не попадает» под курсор);
        * аналитическая формула не вызывает LAPACK и работает заметно быстрее,
          а результат детерминирован.
        """
        if self._view_projection_inverse is None:
            self._view_projection_inverse = invert_view_projection(
                self._projection_double(), self._view_double()
            )
        return self._view_projection_inverse

    def _projection_double(self) -> np.ndarray:
        """Матрица проекции в double (точная копия формулы движка)."""
        near, far = float(self.near), float(self.far)
        matrix = np.zeros((4, 4), dtype=float)
        if self.is_orthographic:
            half_height = max(EPSILON, self.ortho_height * 0.5)
            half_width = half_height * self.effective_aspect
            matrix[0, 0] = 1.0 / half_width
            matrix[1, 1] = 1.0 / half_height
            matrix[2, 2] = -2.0 / (far - near)
            matrix[2, 3] = -(far + near) / (far - near)
            matrix[3, 3] = 1.0
            return matrix
        focal = 1.0 / math.tan(math.radians(self.fov) * 0.5)
        matrix[0, 0] = focal / self.effective_aspect
        matrix[1, 1] = focal
        matrix[2, 2] = (far + near) / (near - far)
        matrix[2, 3] = (2.0 * far * near) / (near - far)
        matrix[3, 2] = -1.0
        return matrix

    def _view_double(self) -> np.ndarray:
        """Видовая матрица в double (точная копия :func:`look_at_matrix`)."""
        eye = self._position
        forward = self.forward
        if forward.is_zero:
            forward = Vector3.forward()
        up = self.up
        right = forward.cross(up)
        if right.is_zero:
            right = forward.cross(Vector3.right())
        right = right.normalized
        true_up = right.cross(forward)

        matrix = np.identity(4, dtype=float)
        matrix[0, :3] = (right.x, right.y, right.z)
        matrix[1, :3] = (true_up.x, true_up.y, true_up.z)
        matrix[2, :3] = (-forward.x, -forward.y, -forward.z)
        matrix[0, 3] = -right.dot(eye)
        matrix[1, 3] = -true_up.dot(eye)
        matrix[2, 3] = forward.dot(eye)
        return matrix

    # Публичные имена из требований: ``get_*_matrix()`` — те же матрицы.
    def get_view_matrix(self) -> np.ndarray:
        """Вернуть видовую матрицу (world → view).

        Returns:
            numpy-массив ``(4, 4)``.
        """
        return self.view_matrix

    def get_projection_matrix(self) -> np.ndarray:
        """Вернуть проектную матрицу текущего режима.

        Returns:
            numpy-массив ``(4, 4)``.
        """
        return self.projection_matrix

    def get_view_projection_matrix(self) -> np.ndarray:
        """Вернуть произведение проекции и вида.

        Returns:
            numpy-массив ``(4, 4)``.
        """
        return self.view_projection_matrix

    def world_to_view(self, point: Vector3) -> Vector3:
        """Перевести точку из мира в пространство камеры."""
        return transform_point(self.view_matrix, point)

    def world_to_clip(self, point: Vector3) -> tuple[float, float, float, float]:
        """Перевести точку в clip-space (``x, y, z, w``).

        Первые три компоненты принадлежат вызывающему, поэтому возвращается
        кортеж: ``w`` нужно для перспективного деления.
        """
        matrix = self.view_projection_matrix
        vector = np.array([point.x, point.y, point.z, 1.0], dtype=float)
        result = matrix.astype(float) @ vector
        return float(result[0]), float(result[1]), float(result[2]), float(result[3])

    # ------------------------------------------------------------------ #
    # Экран ↔ мир
    # ------------------------------------------------------------------ #
    def world_to_screen(
        self,
        point: Vector3,
        width: int | None = None,
        height: int | None = None,
    ) -> tuple[float, float, bool]:
        """Спроецировать мировую точку на экран.

        Args:
            point: Точка в мире.
            width: Ширина вьюпорта; ``None`` — последний известный размер.
            height: Высота вьюпорта; ``None`` — последний известный размер.

        Returns:
            Пару «координаты в пикселях (начало сверху слева)» и признак
            ``visible``: ``False``, если точка за камерой или вне объёма.
        """
        view_width, view_height = self._resolve_viewport(width, height)
        if view_width <= 0 or view_height <= 0:
            return 0.0, 0.0, False

        matrix = self.view_projection_matrix.astype(float)
        vector = np.array([point.x, point.y, point.z, 1.0], dtype=float)
        clip = matrix @ vector
        w = float(clip[3])
        if abs(w) <= EPSILON:
            return 0.0, 0.0, False
        ndc_x = float(clip[0]) / w
        ndc_y = float(clip[1]) / w

        # Прямоугольник вьюпорта камеры: сдвиг и масштаб внутри области.
        rect_x, rect_y, rect_w, rect_h = self.pixel_rect(view_width, view_height)
        if rect_w <= 0 or rect_h <= 0:
            rect_x, rect_y, rect_w, rect_h = 0, 0, view_width, view_height
        screen_x = rect_x + (ndc_x * 0.5 + 0.5) * rect_w
        # OpenGL считает y снизу, оконные интерфейсы — сверху.
        gl_y = rect_y + (ndc_y * 0.5 + 0.5) * rect_h
        screen_y = view_height - gl_y
        visible = (
            w > 0.0
            and -1.0 <= ndc_x <= 1.0
            and -1.0 <= ndc_y <= 1.0
            and -1.0 <= float(clip[2]) / w <= 1.0
        )
        return screen_x, screen_y, visible

    def screen_to_world(
        self,
        x: float,
        y: float,
        width: int | None = None,
        height: int | None = None,
        *,
        depth: float = 0.0,
    ) -> Vector3:
        """Развернуть точку экрана в мировые координаты.

        Для перспективной камеры точка лежит на луче взгляда на расстоянии,
        соответствующем ``depth`` в NDC (``-1`` — ближняя плоскость, ``+1`` —
        дальняя); по умолчанию берётся середина между плоскостями отсечения.

        Args:
            x: Координата по горизонтали (пиксели, начало сверху слева).
            y: Координата по вертикали.
            width: Ширина вьюпорта; ``None`` — последний известный размер.
            height: Высота вьюпорта; ``None`` — последний известный размер.
            depth: NDC-глубина в диапазоне ``-1..1``.

        Returns:
            Точка в мировых координатах (позиция камеры, если размеры нулевые).
        """
        view_width, view_height = self._resolve_viewport(width, height)
        if view_width <= 0 or view_height <= 0:
            return Vector3(self._position.x, self._position.y, self._position.z)

        rect_x, rect_y, rect_w, rect_h = self.pixel_rect(view_width, view_height)
        if rect_w <= 0 or rect_h <= 0:
            rect_x, rect_y, rect_w, rect_h = 0, 0, view_width, view_height

        ndc_x = ((float(x) - rect_x) / rect_w) * 2.0 - 1.0
        # Возврат из оконных координат (сверху вниз) в NDC.
        gl_y = view_height - float(y)
        ndc_y = ((gl_y - rect_y) / rect_h) * 2.0 - 1.0
        ndc_z = max(-1.0, min(1.0, float(depth)))

        inverse = np.asarray(self.inverse_view_projection, dtype=float)
        clip = np.array([ndc_x, ndc_y, ndc_z, 1.0], dtype=float)
        world = inverse @ clip
        w = float(world[3])
        if abs(w) <= EPSILON:
            return Vector3(self._position.x, self._position.y, self._position.z)
        return Vector3(
            float(world[0]) / w, float(world[1]) / w, float(world[2]) / w
        )

    def _resolve_viewport(
        self, width: int | None, height: int | None
    ) -> tuple[int, int]:
        """Определить размеры вьюпорта: аргументы или последний известный."""
        if width is None or height is None:
            known_width, known_height = self._viewport_size
            width = known_width if width is None else width
            height = known_height if height is None else height
        return int(width or 0), int(height or 0)

    # ------------------------------------------------------------------ #
    # Отсечение
    # ------------------------------------------------------------------ #
    def update_frustum(self) -> Frustum:
        """Пересчитать усечённый объём и вернуть его.

        Обновление выполняется только при изменении матриц камеры, поэтому
        вызывать метод можно каждый кадр и для каждого объекта.
        """
        frustum = self.frustum
        if frustum.is_dirty:
            frustum.update(self.view_projection_matrix, self.near, self.far)
        return frustum

    def frustum_planes(self) -> np.ndarray:
        """Извлечь 6 плоскостей усечённого объёма (нормаль + смещение).

        Returns:
            Массив ``(6, 4)``: первые три компоненты — нормаль, четвёртая —
            смещение. Порядок: left, right, bottom, top, near, far.
        """
        return self.update_frustum().planes

    def is_sphere_visible(self, center: Vector3, radius: float) -> bool:
        """Проверить, попадает ли сфера в усечённый объём камеры.

        Args:
            center: Центр сферы в мире.
            radius: Радиус сферы.

        Returns:
            ``True``, если сфера видна (или пересекает границу объёма).
        """
        return self.update_frustum().intersects_sphere(center, radius)

    def is_aabb_visible(self, minimum: Vector3, maximum: Vector3) -> bool:
        """Проверить видимость AABB (``p``-vertex тест по шести плоскостям)."""
        return self.update_frustum().intersects_aabb(minimum, maximum)

    def frustum_corners(self) -> np.ndarray:
        """Вернуть восемь углов усечённого объёма в мире ``(8, 3)``.

        Первые четыре — ближний прямоугольник, последние четыре — дальний
        (порядок: левый-нижний, правый-нижний, правый-верхний, левый-верхний).

        Returns:
            Копия массива углов: гизмо камеры рисует по нему фрустум.
        """
        frustum = self.update_frustum()
        return frustum.corners.copy()

    def ray_from_screen(
        self, x: float, y: float, width: int, height: int
    ) -> tuple[Vector3, Vector3]:
        """Построить луч из позиции камеры через точку экрана.

        Луч строится через :meth:`screen_to_world`, поэтому он согласован с
        :meth:`world_to_screen` по построению: раньше эти две формулы были
        независимыми, и расхождение (например, тангенс вместо котангенса
        половины угла обзора) приводило к тому, что выбор объектов «не попадал»
        в цель при любом FOV, кроме 90°.

        Args:
            x: Координата курсора по горизонтали (пиксели, начало сверху слева).
            y: Координата курсора по вертикали.
            width: Ширина вьюпорта.
            height: Высота вьюпорта.

        Returns:
            Пару «начало луча, нормализованное направление».
        """
        if width <= 0 or height <= 0:
            return self._position, self.forward

        if self.is_orthographic:
            # В ортографии лучи параллельны, а начало лежит на ближней
            # плоскости: берём две точки на разных глубинах и берём разницу.
            near_point = self.screen_to_world(x, y, width, height, depth=-1.0)
            far_point = self.screen_to_world(x, y, width, height, depth=1.0)
            direction = far_point - near_point
            if direction.is_zero:
                return near_point, self.forward
            return near_point, direction.normalized

        # Перспектива: начало — позиция камеры, направление — на точку
        # развёртки с нулевой глубиной (середина между плоскостями отсечения).
        target = self.screen_to_world(x, y, width, height, depth=0.0)
        direction = target - self._position
        if direction.is_zero:
            return self._position, self.forward
        return self._position, direction.normalized

    def _screen_to_ndc(
        self, x: float, y: float, width: int, height: int
    ) -> tuple[float, float]:
        """Перевести координаты курсора в NDC с учётом прямоугольника вьюпорта.

        Args:
            x: Координата по горизонтали (пиксели, начало сверху слева).
            y: Координата по вертикали.
            width: Ширина области вывода.
            height: Высота области вывода.

        Returns:
            Пару ``(ndc_x, ndc_y)`` в диапазоне ``-1..1`` внутри камеры.
        """
        rect_x, rect_y, rect_w, rect_h = self.pixel_rect(width, height)
        if rect_w <= 0 or rect_h <= 0:
            rect_x, rect_y, rect_w, rect_h = 0, 0, width, height
        ndc_x = ((float(x) - rect_x) / rect_w) * 2.0 - 1.0
        gl_y = height - float(y)
        ndc_y = ((gl_y - rect_y) / rect_h) * 2.0 - 1.0
        return ndc_x, ndc_y

    # ------------------------------------------------------------------ #
    # Привязка к трансформу и сериализация
    # ------------------------------------------------------------------ #
    def sync_from_transform(self, transform: Transform, *, scale: bool = False) -> None:
        """Скопировать позицию и ориентацию из трансформа объекта.

        Args:
            transform: Трансформ игрового объекта.
            scale: Учитывать ли масштаб объекта (для отладки искажений).
        """
        matrix = np.asarray(transform.world_matrix, dtype=float).reshape(4, 4)
        self.position = extract_translation(matrix)
        self.rotation = matrix_to_quaternion(matrix)
        if scale:
            # Масштаб игрового объекта не влияет на камеру, но ортографию
            # удобно растягивать вместе с ним (например, при анимации).
            factor = float(np.linalg.norm(matrix[:3, 1]))
            if factor > EPSILON:
                self.ortho_height = max(EPSILON, self.ortho_height * factor)

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать параметры камеры в словарь."""
        return {
            "name": self.name,
            "mode": self.mode,
            "fov": self.fov,
            "near": self.near,
            "far": self.far,
            "aspect": self.aspect,
            "aspect_mode": self.aspect_mode,
            "ortho_height": self.ortho_height,
            "position": list(self._position.to_tuple()),
            "rotation": list(self._rotation.to_tuple()),
            "viewport_rect": list(self.viewport_rect),
            "clear_color": list(self.clear_color),
            "clear_flags": list(self.clear_flags),
            "culling_mask": int(self.culling_mask),
            "depth": int(self.depth),
            "enabled": bool(self.enabled),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Camera:
        """Восстановить камеру из словаря :meth:`to_dict`."""
        camera = cls(
            fov=float(data.get("fov", 60.0)),
            near=float(data.get("near", 0.1)),
            far=float(data.get("far", 500.0)),
            aspect=float(data.get("aspect", 16.0 / 9.0)),
            mode=str(data.get("mode", ProjectionMode.PERSPECTIVE)),
            ortho_height=float(data.get("ortho_height", 10.0)),
            viewport_rect=_tuple4(data.get("viewport_rect"), (0.0, 0.0, 1.0, 1.0)),
            clear_color=_tuple4(
                data.get("clear_color"), (0.078, 0.075, 0.071, 1.0)
            ),
            clear_flags=tuple(
                str(flag) for flag in data.get("clear_flags", ("color", "depth"))
            ),
            culling_mask=int(data.get("culling_mask", -1)),
            depth=int(data.get("depth", 0)),
            name=str(data.get("name", "Main Camera")),
            enabled=bool(data.get("enabled", True)),
            aspect_mode=str(data.get("aspect_mode", ASPECT_VIEWPORT)),
        )
        position = data.get("position") or (0.0, 0.0, 0.0)
        rotation = data.get("rotation") or (0.0, 0.0, 0.0, 1.0)
        camera.position = Vector3.from_iterable(position)
        camera.rotation = Quaternion(*[float(value) for value in rotation])
        return camera

    def copy(self, *, name: str | None = None) -> Camera:
        """Создать независимую копию камеры.

        Args:
            name: Имя копии; ``None`` — оставить исходное.

        Returns:
            Новая камера с теми же параметрами.
        """
        data = self.to_dict()
        if name is not None:
            data["name"] = name
        return Camera.from_dict(data)

    # ------------------------------------------------------------------ #
    # Служебное
    # ------------------------------------------------------------------ #
    @property
    def clear_color_rgba(self) -> tuple[float, float, float, float]:
        """Цвет очистки, приведённый к четырём компонентам ``0..1``."""
        red, green, blue, alpha = _tuple4(self.clear_color, (0.0, 0.0, 0.0, 1.0))
        clamp = lambda value: max(0.0, min(1.0, float(value)))  # noqa: E731
        return clamp(red), clamp(green), clamp(blue), clamp(alpha)

    @property
    def clears_color(self) -> bool:
        """``True``, если камера очищает цветовой буфер."""
        return "color" in self.clear_flags

    @property
    def clears_depth(self) -> bool:
        """``True``, если камера очищает буфер глубины."""
        return "depth" in self.clear_flags

    def select_objects(self, objects: Iterable[object]) -> list[object]:
        """Отфильтровать объекты сцены по маске отсечения камеры.

        Args:
            objects: Объекты с атрибутом ``layer``.

        Returns:
            Список объектов, чьи слои попадают в маску камеры.
        """
        return [
            item for item in objects if self.layer_visible(getattr(item, "layer", 1))
        ]

    def __repr__(self) -> str:
        return (
            f"<Camera '{self.name}' {self.mode} pos={self._position} "
            f"fov={self.fov:.0f} near={self.near} far={self.far} "
            f"mask=0x{self.culling_mask & 0xFFFFFFFF:X}>"
        )


def _tuple4(value: object, default: tuple[float, float, float, float]):
    """Привести значение к кортежу из четырёх чисел."""
    if value is None:
        return default
    items = tuple(float(item) for item in value)  # type: ignore[arg-type]
    if len(items) != 4:
        return default
    return items


def invert_view_projection(
    projection: np.ndarray, view: np.ndarray
) -> np.ndarray:
    """Аналитически обратить произведение ``projection @ view`` в double.

    Матрица проекции движка «блочно-треугольная»: ненулевые элементы лежат в
    столбцах 0, 1, 2, а последняя строка — либо ``(0, 0, -1, 0)``
    (перспектива), либо ``(0, 0, 0, 1)`` (ортография). Обратную можно собрать
    из двух систем 2x2 без вызова LAPACK — это и быстрее, и точнее.

    Численное обращение произведения целиком даёт заметную ошибку: элементы
    порядка ``1/(far - near)`` и мировой сдвиг камеры попадают в одну матрицу,
    и её обусловленность резко падает. Раздельное обращение вида и проекции
    эту ошибку убирает.

    Args:
        projection: Матрица проекции ``(4, 4)``.
        view: Видовая матрица ``(4, 4)``.

    Returns:
        Матрицу ``inverse(view) @ inverse(projection)`` в double.

    Raises:
        ValueError: Если матрица проекции вырождена.
    """
    matrix = np.asarray(projection, dtype=float).reshape(4, 4)
    a, b = matrix[0, 0], matrix[1, 1]
    c, d = matrix[2, 2], matrix[3, 2]
    e, f = matrix[2, 3], matrix[3, 3]
    if abs(a) < EPSILON or abs(b) < EPSILON or abs(c) < EPSILON:
        raise ValueError("Матрица проекции вырождена: обратной не существует")

    # Нижний блок 2x2 [[d, e], [c, f]] обращается по обычной формуле:
    # определитель равен d*f - e*c, и после умножения на проекцию получается
    # в точности единичная матрица (проверяется тестом).
    determinant = d * f - e * c
    if abs(determinant) < EPSILON:
        raise ValueError("Матрица проекции вырождена: обратной не существует")

    inverse = np.zeros((4, 4), dtype=float)
    inverse[0, 0] = 1.0 / a
    inverse[1, 1] = 1.0 / b
    # Нижний блок 2x2 [[c, e], [d, f]] обращается по обычной формуле.
    inverse[2, 2] = f / determinant
    inverse[3, 2] = -d / determinant
    inverse[2, 3] = -e / determinant
    inverse[3, 3] = c / determinant

    inverse_view = _invert_rigid_double(view)
    return inverse_view @ inverse


def _invert_rigid_double(matrix: np.ndarray) -> np.ndarray:
    """Обратить матрицу ``T * R`` (без масштаба) в double.

    Аналог :func:`engine.core.math3d.invert_rigid`, но с расчётом в double:
    матрица вида содержит мировой сдвиг камеры, и в float32 он теряет
    точность, из-за которой обратное преобразование «уезжает».

    Args:
        matrix: Матрица ``(4, 4)`` без масштабирования.

    Returns:
        Обратную матрицу ``(4, 4)`` в double.
    """
    data = np.asarray(matrix, dtype=float).reshape(4, 4)
    rotation = data[:3, :3]
    translation = data[:3, 3]

    inverse = np.identity(4, dtype=float)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


# --------------------------------------------------------------------------- #
# Камера как компонент игрового объекта
# --------------------------------------------------------------------------- #
class CameraComponent(Component):
    """Компонент камеры: связывает :class:`Camera` с игровым объектом.

    Позиция и ориентация берутся из трансформа владельца, поэтому камера
    движется вместе с объектом (в том числе по иерархии). Локальные
    множители позволяют сместить или повернуть камеру относительно объекта —
    это удобно для «кокпита» и для камеры от третьего лица.

    Args:
        camera: Готовая камера; ``None`` — создать со значениями по умолчанию.
        local_offset: Смещение камеры в локальных осях объекта.
        local_rotation: Дополнительный поворот относительно объекта.
        follow_owner: Синхронизировать ли камеру с трансформом владельца.

    Attributes:
        camera: Объект камеры.
    """

    __slots__ = ("camera", "follow_owner", "local_offset", "local_rotation")

    def __init__(
        self,
        camera: Camera | None = None,
        *,
        local_offset: Vector3 | None = None,
        local_rotation: Quaternion | None = None,
        follow_owner: bool = True,
    ) -> None:
        super().__init__()
        self.camera = camera or Camera()
        self.local_offset = local_offset or Vector3.zero()
        self.local_rotation = local_rotation or Quaternion.identity()
        self.follow_owner = bool(follow_owner)

    # ------------------------------------------------------------------ #
    # Имя для UI
    # ------------------------------------------------------------------ #
    @property
    def display_name(self) -> str:
        """Имя камеры для инспектора и стека."""
        owner = self.game_object
        if self.camera.name and self.camera.name != "Main Camera":
            return self.camera.name
        return owner.name if owner is not None else self.camera.name

    # ------------------------------------------------------------------ #
    # Синхронизация
    # ------------------------------------------------------------------ #
    def sync_to_camera(self) -> Camera:
        """Перенести мировой трансформ объекта в камеру.

        Returns:
            Обновлённую камеру.
        """
        if self.follow_owner and self.is_attached:
            matrix = np.asarray(self.transform.world_matrix, dtype=float).reshape(4, 4)
            origin = extract_translation(matrix)
            rotation = matrix_to_quaternion(matrix)
            # Локальные смещение и доворот применяются в осях объекта.
            offset = rotation.rotate_vector(self.local_offset)
            self.camera.position = origin + offset
            self.camera.rotation = (rotation * self.local_rotation).normalized
        return self.camera

    def on_attach(self) -> None:
        """Сразу подтянуть трансформ владельца."""
        self.sync_to_camera()

    def on_update(self, dt: float) -> None:
        """Обновлять камеру каждый кадр — она часть трансформа объекта."""
        self.sync_to_camera()

    def on_late_update(self, dt: float) -> None:
        """Позднее обновление: после логики объект мог сдвинуться."""
        self.sync_to_camera()

    # ------------------------------------------------------------------ #
    # Сериализация (для сохранения сцены)
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Сериализовать компонент вместе с камерой."""
        return {
            "camera": self.camera.to_dict(),
            "local_offset": list(self.local_offset.to_tuple()),
            "local_rotation": list(self.local_rotation.to_tuple()),
            "follow_owner": self.follow_owner,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraComponent:
        """Восстановить компонент из словаря :meth:`to_dict`."""
        payload = data.get("camera")
        camera = Camera.from_dict(payload) if isinstance(payload, dict) else Camera()
        offset = data.get("local_offset") or (0.0, 0.0, 0.0)
        rotation = data.get("local_rotation") or (0.0, 0.0, 0.0, 1.0)
        return cls(
            camera,
            local_offset=Vector3.from_iterable(offset),
            local_rotation=Quaternion(*[float(value) for value in rotation]),
            follow_owner=bool(data.get("follow_owner", True)),
        )

    def __repr__(self) -> str:
        owner = self.game_object
        target = owner.name if owner is not None else "<свободен>"
        return f"<CameraComponent '{target}' camera={self.camera.name!r}>"


# --------------------------------------------------------------------------- #
# Стек камер
# --------------------------------------------------------------------------- #
class CameraStack:
    """Несколько камер с выбором активной и порядком отрисовки.

    Стек решает две задачи:

    * **Выбор активной камеры** для игрового вида
      (:meth:`set_active`, :attr:`active`);
    * **Отрисовка слоями** — игровая камера, затем UI-камера поверх
      (:meth:`ordered`), каждая со своим прямоугольником вьюпорта, цветом
      очистки и маской отсечения.

    Каждый кадр стек один раз «захватывает» матрицы активной камеры
    (:meth:`begin_frame`), чтобы горячий цикл отсечения не пересчитывал
    ``view_projection`` на объект.

    Args:
        cameras: Начальный набор камер.
        active: Активная камера или её индекс.
    """

    __slots__ = (
        "name",
        "_active",
        "_cameras",
        "_captured",
        "_captured_version",
        "_view_projection",
        "_view",
        "_projection",
    )

    def __init__(
        self,
        cameras: Sequence[Camera] | None = None,
        *,
        active: int | Camera | None = None,
        name: str = "CameraStack",
    ) -> None:
        self.name = name
        self._cameras: list[Camera] = list(cameras or ())
        self._active: Camera | None = None
        self._captured: Camera | None = None
        self._captured_version = -1
        self._view: np.ndarray | None = None
        self._projection: np.ndarray | None = None
        self._view_projection: np.ndarray | None = None
        if cameras:
            self.set_active(active if active is not None else 0)
        elif isinstance(active, Camera):
            self._active = active

    # ------------------------------------------------------------------ #
    # Состав
    # ------------------------------------------------------------------ #
    @property
    def cameras(self) -> tuple[Camera, ...]:
        """Все камеры стека в порядке добавления."""
        return tuple(self._cameras)

    @property
    def enabled_cameras(self) -> tuple[Camera, ...]:
        """Только включённые камеры."""
        return tuple(camera for camera in self._cameras if camera.enabled)

    def __len__(self) -> int:
        return len(self._cameras)

    def __iter__(self) -> Iterator[Camera]:
        return iter(self._cameras)

    def __contains__(self, camera: object) -> bool:
        return camera in self._cameras

    def __getitem__(self, index: int) -> Camera:
        return self._cameras[index]

    def add(self, camera: Camera, *, make_active: bool = False) -> Camera:
        """Добавить камеру в стек.

        Args:
            camera: Камера.
            make_active: Сделать её активной сразу.

        Returns:
            Добавленную камеру.
        """
        if camera not in self._cameras:
            self._cameras.append(camera)
        if make_active or self._active is None:
            self._active = camera
        return camera

    def add_component(
        self, component: CameraComponent, *, make_active: bool = False
    ) -> Camera:
        """Добавить камеру компонента и включить его синхронизацию."""
        component.sync_to_camera()
        return self.add(component.camera, make_active=make_active)

    def remove(self, camera: Camera) -> bool:
        """Убрать камеру из стека; активной станет соседняя."""
        if camera not in self._cameras:
            return False
        index = self._cameras.index(camera)
        self._cameras.pop(index)
        if self._active is camera:
            self._active = (
                self._cameras[min(index, len(self._cameras) - 1)]
                if self._cameras
                else None
            )
        return True

    def clear(self) -> None:
        """Очистить стек."""
        self._cameras.clear()
        self._active = None

    # ------------------------------------------------------------------ #
    # Активная камера
    # ------------------------------------------------------------------ #
    @property
    def active(self) -> Camera | None:
        """Активная камера (``None`` — стек пуст)."""
        if self._active is not None and self._active in self._cameras:
            return self._active
        return self._cameras[0] if self._cameras else None

    @property
    def active_index(self) -> int:
        """Индекс активной камеры (``-1``, если стек пуст)."""
        camera = self.active
        return -1 if camera is None else self._cameras.index(camera)

    def set_active(self, camera: int | Camera | None) -> Camera | None:
        """Выбрать активную камеру.

        Args:
            camera: Камера, её индекс или ``None`` (первая в стеке).

        Returns:
            Новую активную камеру или ``None``, если стек пуст.

        Raises:
            IndexError: Если индекс выходит за границы стека.
        """
        if camera is None:
            self._active = self._cameras[0] if self._cameras else None
        elif isinstance(camera, Camera):
            self.add(camera)
            self._active = camera
        else:
            index = int(camera)
            if not -len(self._cameras) <= index < len(self._cameras):
                raise IndexError(f"Нет камеры с индексом {index} в стеке '{self.name}'")
            self._active = self._cameras[index]
        return self._active

    def set_active_by_name(self, name: str) -> Camera | None:
        """Выбрать активную камеру по имени."""
        for camera in self._cameras:
            if camera.name == name:
                self._active = camera
                return camera
        return None

    def index_of(self, camera: Camera) -> int:
        """Индекс камеры в стеке (``-1``, если её нет)."""
        try:
            return self._cameras.index(camera)
        except ValueError:
            return -1

    # ------------------------------------------------------------------ #
    # Порядок отрисовки
    # ------------------------------------------------------------------ #
    def ordered(self) -> tuple[Camera, ...]:
        """Камеры в порядке отрисовки: по ``depth``, затем по добавлению.

        Камеры с большим ``depth`` рисуются позже — так UI-камера
        (``depth=10``) накладывается поверх игровой (``depth=0``).
        """
        return tuple(
            sorted(
                (camera for camera in self._cameras if camera.enabled),
                key=lambda camera: camera.depth,
            )
        )

    def overlay_cameras(self) -> tuple[Camera, ...]:
        """Камеры, которые не очищают цвет (рисуются поверх нижних слоёв)."""
        ordered = self.ordered()
        if not ordered:
            return ()
        return tuple(camera for camera in ordered[1:] if not camera.clears_color)

    # ------------------------------------------------------------------ #
    # Захват матриц кадра
    # ------------------------------------------------------------------ #
    def begin_frame(self, camera: Camera | None = None) -> Camera | None:
        """Зафиксировать матрицы камеры на кадр.

        Args:
            camera: Камера кадра; ``None`` — активная камера стека.

        Returns:
            Камеру, матрицы которой зафиксированы, или ``None``.
        """
        target = camera or self.active
        if target is None:
            self._captured = None
            return None
        self._captured = target
        self._captured_version = target.version
        self._view = target.view_matrix
        self._projection = target.projection_matrix
        self._view_projection = self._projection @ self._view
        return target

    def _refresh(self) -> None:
        """Обновить матрицы, если камера изменилась после захвата."""
        camera = self._captured
        if camera is None:
            return
        if self._captured_version != camera.version:
            self.begin_frame(camera)

    @property
    def view_matrix(self) -> np.ndarray:
        """Видовая матрица кадра."""
        self._refresh()
        if self._view is None:
            raise ValueError("Стек камер пуст: view_matrix недоступна")
        return self._view

    @property
    def projection_matrix(self) -> np.ndarray:
        """Проекционная матрица кадра."""
        self._refresh()
        if self._projection is None:
            raise ValueError("Стек камер пуст: projection_matrix недоступна")
        return self._projection

    @property
    def view_projection_matrix(self) -> np.ndarray:
        """Произведение проекции и вида для кадра."""
        self._refresh()
        if self._view_projection is None:
            raise ValueError("Стек камер пуст: view_projection недоступна")
        return self._view_projection

    def is_sphere_visible(self, center: Vector3, radius: float) -> bool:
        """Отсечение сферы по активной камере стека."""
        camera = self._captured or self.active
        if camera is None:
            return True
        return camera.is_sphere_visible(center, radius)

    # ------------------------------------------------------------------ #
    # Сервис
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Сериализовать стек камер."""
        return {
            "name": self.name,
            "active": self.active_index,
            "cameras": [camera.to_dict() for camera in self._cameras],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraStack:
        """Восстановить стек камер из словаря."""
        cameras = [Camera.from_dict(entry) for entry in data.get("cameras", [])]
        stack = cls(cameras, name=str(data.get("name", "CameraStack")))
        index = int(data.get("active", 0))
        if cameras:
            stack.set_active(max(0, min(index, len(cameras) - 1)))
        return stack

    def __repr__(self) -> str:
        active = self.active
        label = active.name if active is not None else "—"
        count = len(self._cameras)
        return f"<CameraStack '{self.name}' камер={count} активная='{label}'>"


# --------------------------------------------------------------------------- #
# Редакторская камера
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class EditorCameraState:
    """Сохраняемое состояние редакторской камеры.

    Attributes:
        target: Точка интереса (центр орбиты).
        distance: Дистанция до точки интереса.
        yaw: Поворот вокруг вертикальной оси (градусы).
        pitch: Наклон (градусы).
        fov: Угол обзора камеры.
        near: Ближняя плоскость.
        far: Дальняя плоскость.
        mode: Режим проекции.
        ortho_height: Высота ортографического объёма.
        position: Позиция камеры (нужна для свободного полёта).
        rotation: Ориентация камеры (``x, y, z, w``).
    """

    target: Vector3 = field(default_factory=Vector3.zero)
    distance: float = 6.0
    yaw: float = 45.0
    pitch: float = 25.0
    fov: float = 60.0
    near: float = 0.1
    far: float = 500.0
    mode: str = ProjectionMode.PERSPECTIVE
    ortho_height: float = 10.0
    position: Vector3 = field(default_factory=Vector3.zero)
    rotation: Quaternion = field(default_factory=Quaternion.identity)

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать состояние в словарь JSON."""
        return {
            "target": list(self.target.to_tuple()),
            "distance": self.distance,
            "yaw": self.yaw,
            "pitch": self.pitch,
            "fov": self.fov,
            "near": self.near,
            "far": self.far,
            "mode": self.mode,
            "ortho_height": self.ortho_height,
            "position": list(self.position.to_tuple()),
            "rotation": list(self.rotation.to_tuple()),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EditorCameraState:
        """Восстановить состояние из словаря."""
        return cls(
            target=Vector3.from_iterable(data.get("target") or (0.0, 0.0, 0.0)),
            distance=float(data.get("distance", 6.0)),
            yaw=float(data.get("yaw", 45.0)),
            pitch=float(data.get("pitch", 25.0)),
            fov=float(data.get("fov", 60.0)),
            near=float(data.get("near", 0.1)),
            far=float(data.get("far", 500.0)),
            mode=str(data.get("mode", ProjectionMode.PERSPECTIVE)),
            ortho_height=float(data.get("ortho_height", 10.0)),
            position=Vector3.from_iterable(data.get("position") or (0.0, 0.0, 0.0)),
            rotation=Quaternion(
                *[float(value) for value in (data.get("rotation") or (0, 0, 0, 1))]
            ),
        )


@dataclass(slots=True)
class EditorCameraController:
    """Орбитальное управление камерой редактора.

    Наследует поведение :class:`OrbitCameraController` (ограничения наклона и
    дистанции) и добавляет то, чего требует редактор: различение орбиты с
    модификатором и без, панораму, зум к курсору, фокус на объект и
    сохранение вида между сессиями.

    Раскладка ввода по умолчанию (как в привычных 3D-редакторах):

    * ``Alt + ЛКМ`` — орбита (если ``alt_orbit`` включён);
    * ``ЛКМ`` — орбита (совместимый режим) или выделение;
    * ``СКМ`` — панорама;
    * колесо — зум;
    * ``F`` — фокус на выделенном объекте;
    * ``Home`` — сброс вида.

    Attributes:
        target: Точка интереса.
        distance: Дистанция до точки интереса.
        yaw: Поворот вокруг вертикальной оси (градусы).
        pitch: Наклон (градусы), ограничен ``min_pitch..max_pitch``.
        alt_orbit: Требовать ли ``Alt`` для орбиты левой кнопкой.
    """

    target: Vector3 = field(default_factory=Vector3.zero)
    distance: float = 6.0
    yaw: float = 45.0
    pitch: float = 25.0
    min_distance: float = 0.25
    max_distance: float = 500.0
    min_pitch: float = -89.0
    max_pitch: float = 89.0
    orbit_speed: float = 0.35
    pan_speed: float = 1.0
    zoom_speed: float = 1.12
    min_zoom_step: float = 0.15
    alt_orbit: bool = True
    free_fly: bool = False
    home_target: Vector3 = field(default_factory=Vector3.zero)
    home_distance: float = 6.0
    home_yaw: float = 45.0
    home_pitch: float = 25.0

    # ------------------------------------------------------------------ #
    # Ввод
    # ------------------------------------------------------------------ #
    def orbit(self, dx_pixels: float, dy_pixels: float) -> None:
        """Вращать камеру вокруг точки интереса.

        Args:
            dx_pixels: Смещение мыши по горизонтали.
            dy_pixels: Смещение мыши по вертикали.
        """
        self.yaw = (self.yaw - dx_pixels * self.orbit_speed) % 360.0
        self.pitch = _clamp(
            self.pitch + dy_pixels * self.orbit_speed, self.min_pitch, self.max_pitch
        )

    def pan(
        self, dx_pixels: float, dy_pixels: float, camera: Camera | None = None
    ) -> None:
        """Сместить точку интереса в плоскости экрана.

        Args:
            dx_pixels: Смещение мыши по горизонтали.
            dy_pixels: Смещение мыши по вертикали.
            camera: Камера, задающая оси панорамы; ``None`` — оси по углам.
        """
        if camera is not None:
            # Масштаб подбирается так, чтобы объект под курсором двигался
            # вместе с мышью: на слабых драйверах это заметно экономит нервы.
            scale = (
                self.distance
                * math.tan(math.radians(camera.fov) * 0.5)
                / 400.0
                * self.pan_speed
            )
            self.target = (
                self.target
                - camera.right * (dx_pixels * scale)
                + camera.up * (dy_pixels * scale)
            )
            return
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)
        forward = Vector3(
            math.cos(pitch) * math.sin(yaw),
            -math.sin(pitch),
            -math.cos(pitch) * math.cos(yaw),
        )
        right = forward.cross(Vector3.up()).normalized
        up = right.cross(forward).normalized
        scale = self.distance * 0.01 * self.pan_speed
        self.target = (
            self.target - right * (dx_pixels * scale) + up * (dy_pixels * scale)
        )

    def zoom(self, steps: float, *, anchor: Vector3 | None = None) -> None:
        """Приблизить или отдалить камеру.

        Args:
            steps: Положительное значение приближает (шаги колеса мыши).
            anchor: Точка, остающаяся на месте при зуме (обычно под курсором).
        """
        before = self.distance
        factor = self.zoom_speed ** float(steps)
        if steps > 0.0:
            # Чистая экспонента асимптотически замедляется у цели, поэтому шаг
            # не меньше min_zoom_step; остаток после упора в min_distance
            # уходит в сдвиг цели вперёд (камера «проезжает» сквозь неё).
            wanted = max(
                self.distance - self.distance / factor,
                self.min_zoom_step * float(steps),
            )
            room = max(self.distance - self.min_distance, 0.0)
            self.distance = self.distance - min(wanted, room)
            leftover = wanted - room
            if leftover > 0.0 and anchor is None:
                self.target = self.target + self._forward() * leftover
        else:
            self.distance = _clamp(
                self.distance / factor, self.min_distance, self.max_distance
            )
        if anchor is not None and before > EPSILON:
            # Точка интереса подтягивается к якорю пропорционально зуму: так
            # колесо приближает к тому, что под курсором, а не к центру.
            ratio = 1.0 - (self.distance / before)
            if ratio > 0.0:
                self.target = self.target.lerp(anchor, max(0.0, min(1.0, ratio)))

    def _forward(self) -> Vector3:
        """Единичный вектор от камеры к точке интереса."""
        yaw = math.radians(self.yaw)
        pitch = math.radians(_clamp(self.pitch, self.min_pitch, self.max_pitch))
        return Vector3(
            -math.cos(pitch) * math.sin(yaw),
            -math.sin(pitch),
            -math.cos(pitch) * math.cos(yaw),
        )

    def focus_on(self, point: Vector3, radius: float = 1.0) -> None:
        """Навести камеру на объект с учётом его габаритов.

        Args:
            point: Центр объекта.
            radius: Радиус объекта (половина диагонали AABB).
        """
        self.target = Vector3(point.x, point.y, point.z)
        wanted = max(radius * 3.0, self.min_distance)
        self.distance = _clamp(wanted, self.min_distance, self.max_distance)

    def frame_bounds(self, minimum: Vector3, maximum: Vector3, camera: Camera) -> None:
        """Вписать AABB в кадр: фокус на центре с учётом угла обзора.

        Args:
            minimum: Минимальный угол бокса.
            maximum: Максимальный угол бокса.
            camera: Камера (нужна для учёта FOV и ортографической высоты).
        """
        center = (minimum + maximum) * 0.5
        radius = max((maximum - minimum).length * 0.5, EPSILON)
        if camera.is_orthographic:
            self.target = center
            wanted = max(radius * 4.0, camera.ortho_height)
            self.distance = _clamp(wanted, self.min_distance, self.max_distance)
            return
        half_fov = math.radians(max(1.0, camera.fov) * 0.5)
        wanted = radius / max(math.sin(half_fov), 1e-3) * 1.2
        self.focus_on(center, wanted / 3.0 if wanted > 0 else radius)

    def apply(self, camera: Camera) -> Camera:
        """Перенести состояние контроллера в камеру.

        Args:
            camera: Камера, которую нужно обновить.

        Returns:
            Ту же камеру (для цепочек вызовов).
        """
        yaw = math.radians(self.yaw)
        pitch = math.radians(_clamp(self.pitch, self.min_pitch, self.max_pitch))
        cos_pitch = math.cos(pitch)
        offset = Vector3(
            cos_pitch * math.sin(yaw),
            math.sin(pitch),
            cos_pitch * math.cos(yaw),
        ) * self.distance
        camera.position = self.target + offset
        camera.look_at(self.target)
        return camera

    def reset(self) -> None:
        """Вернуть управление в исходное состояние."""
        self.target = Vector3(
            self.home_target.x, self.home_target.y, self.home_target.z
        )
        self.distance = self.home_distance
        self.yaw = self.home_yaw
        self.pitch = self.home_pitch

    # ------------------------------------------------------------------ #
    # Сохранение между сессиями
    # ------------------------------------------------------------------ #
    def capture(self, camera: Camera) -> EditorCameraState:
        """Снять текущее состояние контроллера и камеры."""
        return EditorCameraState(
            target=Vector3(self.target.x, self.target.y, self.target.z),
            distance=self.distance,
            yaw=self.yaw,
            pitch=self.pitch,
            fov=camera.fov,
            near=camera.near,
            far=camera.far,
            mode=camera.mode,
            ortho_height=camera.ortho_height,
            position=Vector3(
                camera.position.x, camera.position.y, camera.position.z
            ),
            rotation=camera.rotation,
        )

    def restore(self, state: EditorCameraState, camera: Camera) -> Camera:
        """Восстановить состояние контроллера и камеры."""
        self.target = Vector3(state.target.x, state.target.y, state.target.z)
        self.distance = _clamp(state.distance, self.min_distance, self.max_distance)
        self.yaw = state.yaw
        self.pitch = _clamp(state.pitch, self.min_pitch, self.max_pitch)
        camera.fov = state.fov
        camera.near = state.near
        camera.far = state.far
        camera.mode = state.mode
        camera.ortho_height = state.ortho_height
        camera.invalidate()
        self.apply(camera)
        return camera

    def __repr__(self) -> str:
        return (
            f"<EditorCameraController target={self.target} "
            f"distance={self.distance:.2f} yaw={self.yaw:.0f} "
            f"pitch={self.pitch:.0f} alt_orbit={self.alt_orbit}>"
        )


# Совместимость: историческое имя контроллера остаётся рабочим.
OrbitCameraController = EditorCameraController


class EditorCameraSession:
    """Сохранение вида редакторской камеры между сессиями.

    Состояние хранится в JSON-файле (по умолчанию
    ``<project_root>/.myengine/camera.json``) и привязано к имени камеры,
    поэтому переключение между несколькими камерами редактора не теряет их
    положение и ориентацию.

    Args:
        path: Путь к файлу состояния.
        state: Начальное состояние для камеры по умолчанию.
    """

    __slots__ = ("path", "_states")

    #: Имя состояния камеры по умолчанию.
    DEFAULT_KEY = "editor"

    def __init__(
        self, path: Path | str | None = None, *, state: EditorCameraState | None = None
    ) -> None:
        self.path = Path(path) if path is not None else None
        self._states: dict[str, EditorCameraState] = {}
        if state is not None:
            self._states[self.DEFAULT_KEY] = state

    # ------------------------------------------------------------------ #
    # Чтение и запись
    # ------------------------------------------------------------------ #
    @classmethod
    def default_path(cls, project_root: Path | str) -> Path:
        """Путь к файлу состояния по умолчанию для проекта."""
        return Path(project_root) / ".myengine" / "camera.json"

    def get(self, key: str | None = None) -> EditorCameraState | None:
        """Состояние камеры по ключу (``None``, если не сохранено)."""
        return self._states.get(key or self.DEFAULT_KEY)

    def set(self, state: EditorCameraState, key: str | None = None) -> None:
        """Запомнить состояние камеры."""
        self._states[key or self.DEFAULT_KEY] = state

    @property
    def keys(self) -> tuple[str, ...]:
        """Имена камер, для которых есть сохранённое состояние."""
        return tuple(sorted(self._states))

    def load(self) -> bool:
        """Прочитать файл состояния.

        Returns:
            ``True``, если файл существовал и был прочитан.

        Note:
            Повреждённый или недоступный файл не считается ошибкой: редактор
            просто начнёт со вида по умолчанию.
        """
        if self.path is None or not self.path.is_file():
            return False
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _log.warning(
                "Не удалось прочитать состояние камеры '%s': %s", self.path, exc
            )
            return False
        if not isinstance(data, dict):
            return False
        for key, payload in data.items():
            if isinstance(payload, dict):
                try:
                    self._states[str(key)] = EditorCameraState.from_dict(payload)
                except (TypeError, ValueError) as exc:
                    _log.warning("Состояние камеры '%s' пропущено: %s", key, exc)
        return True

    def save(self) -> bool:
        """Записать состояние в файл.

        Returns:
            ``True``, если запись удалась.
        """
        if self.path is None:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                key: state.to_dict() for key, state in sorted(self._states.items())
            }
            self.path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            _log.warning(
                "Не удалось сохранить состояние камеры '%s': %s", self.path, exc
            )
            return False
        return True

    def clear(self) -> None:
        """Забыть все сохранённые состояния."""
        self._states.clear()

    def __len__(self) -> int:
        return len(self._states)

    def __repr__(self) -> str:
        return f"<EditorCameraSession path={self.path} камер={len(self._states)}>"


# --------------------------------------------------------------------------- #
# Предпросмотр игровой камеры (рендер в текстуру)
# --------------------------------------------------------------------------- #
class CameraPreview:
    """Предпросмотр камеры: рендер в текстуру (Render Target).

    Класс связывает камеру, цель вывода и рендерер: он пересчитывает цель при
    смене размера, умеет переключаться между камерами и отдавать пиксели для
    показа в UI.

    Args:
        camera: Камера предпросмотра.
        width: Ширина цели.
        height: Высота цели.
        target: Готовая цель вывода; ``None`` — создать свою.
        name: Имя цели вывода.
    """

    __slots__ = ("camera", "target", "_last_stats")

    def __init__(
        self,
        camera: Camera | None = None,
        *,
        width: int = 320,
        height: int = 180,
        target: RenderTarget | None = None,
        name: str = "camera_preview",
    ) -> None:
        from engine.rendering.render_graph import RenderTarget

        self.camera = camera or Camera(name="Preview Camera")
        self.target = target or RenderTarget(
            name, max(1, width), max(1, height)
        )
        self.camera.set_aspect(self.target.width, self.target.height)
        self._last_stats: RenderStats | None = None

    # ------------------------------------------------------------------ #
    # Состояние
    # ------------------------------------------------------------------ #
    @property
    def size(self) -> tuple[int, int]:
        """Размер цели предпросмотра."""
        return self.target.width, self.target.height

    @property
    def is_created(self) -> bool:
        """``True``, если цель создана в драйвере."""
        return self.target.is_created

    @property
    def last_stats(self) -> RenderStats | None:
        """Статистика последнего кадра предпросмотра."""
        return self._last_stats

    def set_camera(self, camera: Camera) -> Camera:
        """Переключить предпросмотр на другую камеру."""
        self.camera = camera
        self.camera.set_aspect(self.target.width, self.target.height)
        return camera

    def resize(self, width: int, height: int) -> None:
        """Изменить размер цели предпросмотра.

        Args:
            width: Новая ширина.
            height: Новая высота.

        Note:
            Пересоздание буферов требует активного контекста OpenGL: вызывающий
            код (вьюпорт редактора) обязан сделать контекст текущим.
        """
        width = max(1, int(width))
        height = max(1, int(height))
        if (width, height) == self.size and self.target.is_created:
            return
        self.target.resize(width, height)
        self.camera.set_aspect(width, height)

    def update_aspect(self) -> Camera:
        """Пересчитать aspect камеры по текущему размеру цели."""
        self.camera.set_aspect(self.target.width, self.target.height)
        return self.camera

    # ------------------------------------------------------------------ #
    # Рендер
    # ------------------------------------------------------------------ #
    def render(
        self,
        renderer: Renderer,
        items: Iterable[tuple[object, object, np.ndarray]] = (),
        *,
        clear_color: tuple[float, float, float, float] | None = None,
    ) -> RenderStats:
        """Отрисовать кадр предпросмотра в цель вывода.

        Args:
            renderer: Рендерер движка.
            items: Список ``(меш, материал, матрица модели)``; четвёртым
                элементом допускается примитив OpenGL (``GL_LINES``).
            clear_color: Цвет очистки; ``None`` — цвет камеры.

        Returns:
            Статистика кадра предпросмотра.
        """
        width, height = self.size
        self.camera.set_aspect(width, height)
        color = clear_color or self.camera.clear_color_rgba
        renderer.begin_frame(
            self.camera, (width, height), target=self.target, clear_color=color
        )
        for entry in items:
            mesh, material, matrix = entry[0], entry[1], entry[2]
            mode = entry[3] if len(entry) > 3 else None
            renderer.submit(mesh, material, matrix, mode=mode)  # type: ignore[arg-type]
        stats = renderer.end_frame(target=self.target)
        self._last_stats = stats
        return stats

    def read_pixels(self, attachment: int = 0) -> np.ndarray:
        """Прочитать пиксели предпросмотра ``(H, W, 4)`` uint8."""
        return self.target.read_pixels(attachment)

    def dispose(self) -> None:
        """Освободить цель вывода."""
        self.target.dispose()

    def __repr__(self) -> str:
        width, height = self.size
        state = "gpu" if self.is_created else "cpu"
        return (
            f"<CameraPreview '{self.camera.name}' {width}x{height} {state}>"
        )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Ограничить значение диапазоном."""
    return max(minimum, min(maximum, value))
