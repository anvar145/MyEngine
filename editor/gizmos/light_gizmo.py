"""Гизмо источников света: иконки, направление и зона влияния.

Гизмо строится целиком из данных источника и не хранит состояние сцены,
поэтому его можно создать «на лету» для любого света — редакторского, игрового
или компонента объекта.

Что рисуется (и почему именно так):

* **иконка** — каркасный ромб постоянного экранного размера. Свет не имеет
  габаритов, и иконка фиксированного мирового размера терялась бы вдали и
  занимала пол-экрана вблизи;
* **линии направления** — для солнца и прожектора: несколько параллельных
  лучей показывают, куда светит источник;
* **зона влияния** — окружность радиуса действия точечного света и конус
  прожектора. Это главное, что нужно видеть в редакторе: где свет реально
  работает, а где уже нет (там вклад ровно ноль — см.
  :meth:`~engine.rendering.lighting.LightManager.influences`);
* **указатель солнца** — для направленного света рисуется «гребёнка» лучей,
  потому что радиуса у него нет.

Example:
    >>> gizmo = LightGizmo(light, selected=True)
    >>> for entry in gizmo.draw_items(icon_material, wire_material):
    ...     renderer.submit(*entry)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from engine.core.math3d import Vector3, compose_matrix
from engine.rendering.lighting import (
    AmbientLight,
    DirectionalLight,
    Light,
    PointLight,
    SpotLight,
)

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.rendering.material import Material
    from engine.rendering.mesh import GpuMesh

__all__ = [
    "LightGizmo",
    "LightGizmoStyle",
    "direction_lines",
    "light_icon_lines",
    "range_circle_lines",
    "spot_cone_lines",
]

#: Сколько сегментов в окружностях и конусах (компромисс качества и цены).
CIRCLE_SEGMENTS = 32

#: Сколько лучей рисуется у направленного света.
DIRECTION_RAYS = 5


@dataclass(slots=True)
class LightGizmoStyle:
    """Оформление гизмо источника света.

    Attributes:
        icon_size: Размер иконки в мировых единицах на референсной дистанции.
        icon_distance: Дистанция, на которой иконка имеет номинальный размер.
        min_icon_size: Минимальный мировой размер иконки.
        show_range: Рисовать ли зону влияния (окружность или конус).
        show_direction: Рисовать ли лучи направления.
        direction_length: Длина лучей направления в мировых единицах.
        icon_color: Цвет иконки невыделенного источника.
        selected_color: Цвет выделенного источника.
        range_color: Цвет зоны влияния.
        direction_color: Цвет лучей направления.
        ambient_color: Цвет фонового света (у него нет ни позы, ни радиуса).
    """

    icon_size: float = 0.45
    icon_distance: float = 10.0
    min_icon_size: float = 0.08
    show_range: bool = True
    show_direction: bool = True
    direction_length: float = 1.6
    icon_color: tuple[float, float, float] = (0.85, 0.78, 0.45)
    selected_color: tuple[float, float, float] = (1.0, 0.72, 0.24)
    range_color: tuple[float, float, float] = (0.78, 0.72, 0.40)
    direction_color: tuple[float, float, float] = (0.95, 0.85, 0.45)
    ambient_color: tuple[float, float, float] = (0.55, 0.62, 0.78)


@dataclass(slots=True)
class LightGizmo:
    """Гизмо одного источника света.

    Args:
        light: Источник, который показывает гизмо.
        selected: Выделен ли источник в редакторе.
        style: Оформление (цвета и размеры).
        name: Имя для подписи (по умолчанию — имя источника).

    Attributes:
        light: Источник света.
    """

    light: Light
    selected: bool = False
    style: LightGizmoStyle = field(default_factory=LightGizmoStyle)
    name: str | None = None
    _icon_mesh: GpuMesh | None = field(default=None, repr=False, compare=False)
    _wire_cache: dict[tuple, GpuMesh] = field(
        default_factory=dict, repr=False, compare=False
    )

    # ------------------------------------------------------------------ #
    # Идентичность и цвет
    # ------------------------------------------------------------------ #
    @property
    def title(self) -> str:
        """Подпись гизмо."""
        return self.name or self.light.name

    @property
    def kind(self) -> str:
        """Тип источника."""
        return self.light.type_name

    @property
    def color(self) -> Vector3:
        """Цвет гизмо: акцентный для выделенного источника."""
        if self.light.is_ambient:
            return Vector3(*self.style.ambient_color)
        source = self.style.selected_color if self.selected else self.style.icon_color
        return Vector3(*source)

    @property
    def range_color(self) -> Vector3:
        """Цвет линий зоны влияния."""
        return Vector3(*self.style.range_color)

    @property
    def direction_color(self) -> Vector3:
        """Цвет лучей направления."""
        return Vector3(*self.style.direction_color)

    @property
    def position(self) -> Vector3:
        """Мировая позиция источника."""
        return self.light.world_position()

    @property
    def direction(self) -> Vector3:
        """Направление, куда светит источник."""
        return self.light.world_direction()

    # ------------------------------------------------------------------ #
    # Размер и матрицы
    # ------------------------------------------------------------------ #
    def icon_scale(self, viewer_position: Vector3) -> float:
        """Подобрать мировой размер иконки для постоянного экранного размера.

        Args:
            viewer_position: Позиция наблюдателя (камера редактора).

        Returns:
            Масштаб иконки, не меньше ``style.min_icon_size``.
        """
        distance = max((self.position - viewer_position).length, 1e-3)
        scale = self.style.icon_size * distance / max(self.style.icon_distance, 1e-3)
        return max(scale, self.style.min_icon_size)

    def icon_matrix(self, viewer_position: Vector3) -> np.ndarray:
        """Матрица модели иконки источника."""
        scale = self.icon_scale(viewer_position)
        return compose_matrix(
            position=self.position, scale=Vector3(scale, scale, scale)
        )

    # ------------------------------------------------------------------ #
    # Геометрия
    # ------------------------------------------------------------------ #
    def icon_mesh(self) -> GpuMesh:
        """Меш иконки источника (кэшируется)."""
        if self._icon_mesh is None:
            self._icon_mesh = _upload(light_icon_lines(1.0), "light_icon")
        return self._icon_mesh

    def wire_mesh(self, kind: str, key: tuple) -> GpuMesh:
        """Меш линий зоны влияния или направления (кэшируется по параметрам).

        Args:
            kind: ``"range"`` или ``"direction"``.
            key: Параметры геометрии (радиус, углы, длина).

        Returns:
            Готовый меш линий.
        """
        cache_key = (kind, *key)
        mesh = self._wire_cache.get(cache_key)
        if mesh is None:
            mesh = _upload(self._wire_points(kind), f"light_{kind}")
            self._wire_cache[cache_key] = mesh
        return mesh

    def _wire_points(self, kind: str) -> np.ndarray:
        """Собрать вершины линий нужного типа."""
        if kind == "range":
            if isinstance(self.light, PointLight):
                return range_circle_lines(self.light.range)
            if isinstance(self.light, SpotLight):
                return spot_cone_lines(
                    self.light.range, self.light.outer_angle
                )
            return np.zeros((0, 3), dtype=np.float32)
        # Лучи направления есть только у источников, которые куда-то светят:
        # точечный светит во все стороны, у фонового направления нет вовсе.
        if isinstance(self.light, SpotLight):
            return direction_lines(
                self.direction,
                self.style.direction_length,
                offset=self.position,
            )
        if isinstance(self.light, DirectionalLight):
            return direction_lines(self.direction, self.style.direction_length)
        return np.zeros((0, 3), dtype=np.float32)

    def range_key(self) -> tuple:
        """Ключ геометрии зоны влияния."""
        if isinstance(self.light, PointLight):
            return (round(self.light.range, 3),)
        if isinstance(self.light, SpotLight):
            return (
                round(self.light.range, 3),
                round(self.light.outer_angle, 3),
            )
        return ()

    def draw_items(
        self,
        icon_material: Material,
        wire_material: Material | None = None,
        *,
        viewer_position: Vector3 | None = None,
        show_range: bool | None = None,
    ) -> list[tuple[GpuMesh, Material, np.ndarray, int]]:
        """Собрать элементы очереди отрисовки для гизмо.

        Args:
            icon_material: Материал иконки.
            wire_material: Материал линий (зона влияния и направление);
                ``None`` — рисовать только иконку.
            viewer_position: Позиция наблюдателя для размера иконки.
            show_range: Показывать ли зону влияния; ``None`` — по выделению и
                настройке ``style.show_range``.

        Returns:
            Список ``(меш, материал, матрица модели, примитив GL)``.
        """
        from OpenGL import GL

        viewer = viewer_position or self.position
        items: list[tuple[GpuMesh, Material, np.ndarray, int]] = [
            (self.icon_mesh(), icon_material, self.icon_matrix(viewer), GL.GL_LINES)
        ]
        if wire_material is None:
            return items

        # Зона влияния у невыделенного источника только мешала бы: в сцене с
        # десятком ламп это десяток лишних конусов. Показываем её у выбранного
        # источника или по явному запросу.
        visible = self.selected if show_range is None else bool(show_range)
        if (
            visible
            and self.style.show_range
            and not self.light.is_ambient
            and self.range_key()
        ):
            mesh = self.wire_mesh("range", self.range_key())
            if mesh.vertex_count:
                items.append((mesh, wire_material, compose_matrix(), GL.GL_LINES))
        if self.style.show_direction and not self.light.is_ambient:
            direction_key = (
                round(self.direction.x, 3),
                round(self.direction.y, 3),
                round(self.direction.z, 3),
                round(self.style.direction_length, 3),
            )
            mesh = self.wire_mesh("direction", direction_key)
            # Пустой меш рисовать нельзя: у источника без направления (или с
            # нулевым вектором) лучей просто нет.
            if mesh.vertex_count:
                items.append(
                    (mesh, wire_material, compose_matrix(), GL.GL_LINES)
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
        tolerance: float = 1.4,
    ) -> bool:
        """Проверить, попал ли луч выбора в иконку источника.

        Args:
            ray_origin: Начало луча (позиция камеры редактора).
            ray_direction: Направление луча.
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
            return False
        closest = ray_origin + direction * along
        radius = self.icon_scale(viewer_position or ray_origin) * tolerance
        return (closest - self.position).length <= radius

    def __repr__(self) -> str:
        state = "выделен" if self.selected else "обычный"
        return f"<LightGizmo '{self.title}' {self.kind} {state} pos={self.position}>"


# --------------------------------------------------------------------------- #
# Генераторы линий
# --------------------------------------------------------------------------- #
def light_icon_lines(size: float = 1.0) -> np.ndarray:
    """Собрать вершины иконки источника для ``GL_LINES``.

    Иконка — каркасный ромб (октаэдр) с крестовиной: узнаётся под любым углом
    и читается как «источник света», а не как точка.

    Args:
        size: Длина большой диагонали.

    Returns:
        Массив ``(N, 3)`` float32: пары вершин образуют отрезки.
    """
    half = float(size) * 0.5
    top = (0.0, half, 0.0)
    bottom = (0.0, -half, 0.0)
    ring = (
        (half, 0.0, 0.0),
        (0.0, 0.0, half),
        (-half, 0.0, 0.0),
        (0.0, 0.0, -half),
    )
    points: list[tuple[float, float, float]] = []
    # Ромб: вершины соединены с кольцом.
    for corner in ring:
        points.extend((top, corner))
        points.extend((bottom, corner))
    # Крестовина по горизонтали: делает иконку заметной сверху и снизу.
    points.extend((ring[0], ring[2]))
    points.extend((ring[1], ring[3]))
    # Малые «лучи» вокруг: отличают источник света от обычного маркера.
    for index in range(4):
        corner = ring[index]
        length = 1.35
        points.append(corner)
        points.append((corner[0] * length, corner[1], corner[2] * length))
    return np.asarray(points, dtype=np.float32)


def range_circle_lines(radius: float, *, segments: int = CIRCLE_SEGMENTS) -> np.ndarray:
    """Собрать три окружности радиуса влияния (в трёх плоскостях).

    Три кольца вместо одного дают объёмное представление о зоне: по одному
    кольцу непонятно, в какой плоскости оно лежит.

    Args:
        radius: Радиус влияния источника.
        segments: Число сегментов окружности.

    Returns:
        Массив ``(N, 3)`` float32: пары вершин для ``GL_LINES``.
    """
    radius = max(float(radius), 1e-3)
    segments = max(6, int(segments))
    angles = np.linspace(0.0, 2.0 * math.pi, segments + 1)
    cosines = np.cos(angles) * radius
    sines = np.sin(angles) * radius
    zeros = np.zeros_like(cosines)
    rings = (
        np.stack((cosines, sines, zeros), axis=1),  # XY
        np.stack((cosines, zeros, sines), axis=1),  # XZ
        np.stack((zeros, cosines, sines), axis=1),  # YZ
    )
    segments_out: list[np.ndarray] = []
    for ring in rings:
        pairs = np.empty((segments * 2, 3), dtype=np.float32)
        pairs[0::2] = ring[:-1]
        pairs[1::2] = ring[1:]
        segments_out.append(pairs)
    return np.concatenate(segments_out, axis=0).astype(np.float32)


def spot_cone_lines(
    length: float,
    outer_angle: float,
    *,
    segments: int = CIRCLE_SEGMENTS,
) -> np.ndarray:
    """Собрать конус прожектора: основание и боковые рёбра.

    Конус строится вдоль локального ``+Z`` (как смотрит источник), а в мир его
    ставит матрица модели — та же, что у света.

    Args:
        length: Дальность конуса (радиус влияния).
        outer_angle: Угол раствора в градусах (половина полного угла).
        segments: Число сегментов основания.

    Returns:
        Массив ``(N, 3)`` float32: пары вершин для ``GL_LINES``.
    """
    length = max(float(length), 1e-3)
    segments = max(6, int(segments))
    cone_radius = length * math.tan(math.radians(min(max(outer_angle, 0.1), 89.0)))
    angles = np.linspace(0.0, 2.0 * math.pi, segments + 1)
    base = np.stack(
        (
            np.cos(angles) * cone_radius,
            np.sin(angles) * cone_radius,
            np.full_like(angles, length),
        ),
        axis=1,
    )
    points: list[np.ndarray] = []
    # Основание.
    pairs = np.empty((segments * 2, 3), dtype=np.float32)
    pairs[0::2] = base[:-1]
    pairs[1::2] = base[1:]
    points.append(pairs)
    # Боковые рёбра к вершине (начало координат).
    apex = np.zeros(3, dtype=np.float32)
    ribs = np.empty((8 * 2, 3), dtype=np.float32)
    step = max(1, segments // 8)
    selected = base[:-1:step][:8]
    ribs[0::2] = apex
    ribs[1::2] = selected.astype(np.float32)
    points.append(ribs)
    return np.concatenate(points, axis=0).astype(np.float32)


def direction_lines(
    direction: Vector3,
    length: float = 1.6,
    *,
    offset: Vector3 | None = None,
    rays: int = DIRECTION_RAYS,
) -> np.ndarray:
    """Собрать лучи направления света для ``GL_LINES``.

    Рисуется несколько параллельных лучей со стрелкой на конце: один отрезок
    не читается как направление, а «гребёнка» — читается.

    Args:
        direction: Направление, куда светит источник.
        length: Длина лучей.
        offset: Смещение начала лучей (позиция прожектора); ``None`` — начало
            координат (для солнца).
        rays: Сколько лучей нарисовать.

    Returns:
        Массив ``(N, 3)`` float32: пары вершин для ``GL_LINES``.
    """
    forward = direction.normalized
    if forward.is_zero:
        return np.zeros((0, 3), dtype=np.float32)
    origin = offset or Vector3.zero()
    # Перпендикулярные оси для разнесения лучей в стороны.
    side = forward.cross(Vector3.up())
    if side.is_zero:
        side = forward.cross(Vector3.right())
    side = side.normalized
    up = side.cross(forward).normalized

    # Смещения лучей: крест из точек вокруг оси.
    spread = max(0.05, length * 0.22)
    offsets: list[Vector3] = []
    count = max(1, int(rays))
    for index in range(count):
        angle = 2.0 * math.pi * index / count
        offsets.append(
            side * (math.cos(angle) * spread) + up * (math.sin(angle) * spread)
        )

    points: list[tuple[float, float, float]] = []
    arrow = max(0.05, length * 0.18)
    for shift in offsets:
        start = origin + shift
        end = start + forward * length
        points.extend((start.to_tuple(), end.to_tuple()))
        # Стрелка: два коротких отрезка назад и в стороны.
        for sign in (1.0, -1.0):
            back = end - forward * arrow + side * (arrow * sign)
            points.append(end.to_tuple())
            points.append(back.to_tuple())
    # Центральный луч — длиннее прочих, чтобы направление читалось сразу.
    center_start = origin
    center_end = origin + forward * (length * 1.25)
    points.extend((center_start.to_tuple(), center_end.to_tuple()))
    return np.asarray(points, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Служебное
# --------------------------------------------------------------------------- #
def _upload(points: np.ndarray, name: str) -> GpuMesh:
    """Собрать GPU-меш линий из массива вершин.

    Args:
        points: Вершины ``(N, 3)`` (пары образуют отрезки).
        name: Имя меша.

    Returns:
        Меш, готовый к загрузке при первом кадре.
    """
    from pathlib import Path

    from engine.core.resources import MeshAsset
    from engine.rendering.mesh import GpuMesh

    asset = MeshAsset(
        path=Path(f"<procedural>/gizmo/{name}"),
        positions=points,
        indices=None,
        normals=None,
        uvs=None,
        name=name,
    )
    return GpuMesh.from_mesh_asset(asset)


def gizmos_for(scene: Any) -> list[LightGizmo]:
    """Собрать гизмо для всех источников сцены.

    Args:
        scene: :class:`~engine.core.scene.Scene`.

    Returns:
        Список гизмо (по одному на свет).
    """
    from engine.rendering.lighting import LightComponent

    result: list[LightGizmo] = []
    for game_object in scene.objects:
        component = game_object.get_component(LightComponent)
        if component is not None:
            result.append(LightGizmo(component.light, name=game_object.name))
    return result


#: Обратная совместимость: тип источника, у которого нет позы.
AmbientLike = AmbientLight
