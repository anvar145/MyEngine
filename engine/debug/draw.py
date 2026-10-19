"""Накопитель отладочной геометрии: линии, сферы, коробки, стрелки, пути, текст.

Чистые данные на numpy: GL здесь нет, рисует их ``engine.rendering.debug_pass``.
Каждый вызов кладёт отрезки в канал (``channel``): канал совпадает с именем
оверлея в :class:`~engine.debug.overlays.OverlayRegistry`, поэтому выключенный
оверлей скрывает всё, что в него нарисовали.

Время жизни: ``duration=0`` — ровно один отрисованный кадр (вызывать каждый
кадр из ``on_update``); ``duration > 0`` — столько секунд. Из
``on_fixed_update`` фиксированных шагов за кадр бывает 0 или несколько, поэтому
там удобнее ``key``: новый вызов с тем же ключом заменяет прежний, и с
небольшим ``duration`` фигура не мигает и не двоится.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["DEFAULT_CHANNEL", "DebugDraw", "DebugText"]

#: Канал вызовов без явного ``channel`` (скрипты пользователя).
DEFAULT_CHANNEL = "script"
_RING = 24

Color = tuple[float, float, float]


def _vec(value: Any) -> np.ndarray:
    """``Vector3``, кортеж или массив → ``float64 (3,)``."""
    to_tuple = getattr(value, "to_tuple", None)
    if to_tuple is not None:
        value = to_tuple()
    return np.asarray(value, dtype=np.float64).reshape(3)


def _rotation_matrix(rotation: Any) -> np.ndarray:
    """Кватернион ``(x, y, z, w)``/``Quaternion`` или матрица 3×3 → матрица."""
    if rotation is None:
        return np.eye(3)
    array = np.asarray(
        rotation.to_tuple() if hasattr(rotation, "to_tuple") else rotation,
        dtype=np.float64,
    )
    if array.shape == (3, 3):
        return array
    x, y, z, w = array.reshape(4)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _ring(radius: float, axis_a: np.ndarray, axis_b: np.ndarray) -> np.ndarray:
    """Окружность в плоскости ``axis_a``/``axis_b``: отрезки ``(_RING*2, 3)``."""
    angles = np.linspace(0.0, 2.0 * np.pi, _RING + 1)
    points = (
        np.outer(np.cos(angles), axis_a) + np.outer(np.sin(angles), axis_b)
    ) * radius
    return np.stack([points[:-1], points[1:]], axis=1).reshape(-1, 3)


def _basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Два единичных вектора, перпендикулярных ``direction`` и друг другу."""
    helper = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(direction, helper))) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    side = np.cross(direction, helper)
    side /= np.linalg.norm(side)
    up = np.cross(side, direction)
    return side, up


@dataclass(slots=True)
class DebugText:
    """Подпись в мире (рисуется лицом к камере)."""

    position: tuple[float, float, float]
    text: str
    color: Color
    size: float
    channel: str


@dataclass(slots=True)
class _Lines:
    points: np.ndarray  # (2k, 3) пары концов отрезков
    color: Color
    depth_test: bool
    channel: str
    life: float


@dataclass(slots=True)
class _Label:
    text: DebugText
    life: float


class DebugDraw:
    """Буфер отладочной геометрии на кадр (и на ``duration`` секунд).

    Attributes:
        enabled: Выключенный буфер игнорирует вызовы (ноль накладных расходов в
            релизе, если оверлеи никто не включал).
    """

    def __init__(self) -> None:
        self.enabled = True
        self._lines: list[_Lines] = []
        self._keyed: dict[Hashable, _Lines] = {}
        self._labels: list[_Label] = []

    # ------------------------------------------------------------------ #
    # Примитивы
    # ------------------------------------------------------------------ #
    def segments(
        self,
        points: np.ndarray,
        color: Color = (1.0, 1.0, 1.0),
        *,
        duration: float = 0.0,
        depth_test: bool = False,
        channel: str = DEFAULT_CHANNEL,
        key: Hashable | None = None,
    ) -> None:
        """Добавить готовые отрезки: массив ``(2k, 3)`` пар концов.

        Args:
            key: Ключ фигуры: прежняя фигура с тем же ключом заменяется.
        """
        if not self.enabled:
            return
        array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if len(array) < 2:
            return
        array = array[: len(array) // 2 * 2]
        item = _Lines(array, _color(color), bool(depth_test), channel, float(duration))
        if key is not None:
            self._keyed[key] = item
        else:
            self._lines.append(item)

    def line(
        self,
        start: Any,
        end: Any,
        color: Color = (1.0, 1.0, 1.0),
        **options: Any,
    ) -> None:
        """Отрезок ``start`` → ``end``."""
        self.segments(np.stack([_vec(start), _vec(end)]), color, **options)

    def path(
        self,
        points: Sequence[Any],
        color: Color = (0.2, 0.8, 1.0),
        *,
        closed: bool = False,
        channel: str = "paths",
        **options: Any,
    ) -> None:
        """Ломаная по точкам (маршрут, путь A*); ``closed`` — замкнуть."""
        if len(points) < 2:
            return
        array = np.stack([_vec(p) for p in points])
        if closed:
            array = np.concatenate([array, array[:1]])
        pairs = np.stack([array[:-1], array[1:]], axis=1).reshape(-1, 3)
        self.segments(pairs, color, channel=channel, **options)

    def arrow(
        self,
        start: Any,
        end: Any,
        color: Color = (1.0, 0.85, 0.2),
        *,
        head: float = 0.25,
        **options: Any,
    ) -> None:
        """Стрелка с наконечником из четырёх «усов» длиной ``head`` (доля)."""
        a, b = _vec(start), _vec(end)
        shaft = b - a
        length = float(np.linalg.norm(shaft))
        if length < 1e-9:
            return
        direction = shaft / length
        side, up = _basis(direction)
        size = length * head
        back = b - direction * size
        tips = [back + (side * s + up * u) * size * 0.5 for s, u in
                ((1, 0), (-1, 0), (0, 1), (0, -1))]
        points = [a, b]
        for tip in tips:
            points.extend((b, tip))
        self.segments(np.stack(points), color, **options)

    def box(
        self,
        center: Any,
        half_extents: Any,
        color: Color = (0.3, 1.0, 0.4),
        *,
        rotation: Any = None,
        **options: Any,
    ) -> None:
        """Каркас коробки с полуразмерами ``half_extents`` (опц. поворот)."""
        half = _vec(half_extents)
        signs = np.array(
            [[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)], float
        )
        corners = (signs * half) @ _rotation_matrix(rotation).T + _vec(center)
        edges = (
            (0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3),
            (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7),
        )
        self.segments(corners[np.array(edges).reshape(-1)], color, **options)

    def sphere(
        self,
        center: Any,
        radius: float,
        color: Color = (0.3, 0.8, 1.0),
        **options: Any,
    ) -> None:
        """Сфера тремя окружностями по главным плоскостям."""
        x, y, z = np.eye(3)
        rings = np.concatenate(
            [_ring(radius, x, z), _ring(radius, x, y), _ring(radius, y, z)]
        )
        self.segments(rings + _vec(center), color, **options)

    def circle(
        self,
        center: Any,
        radius: float,
        normal: Any = (0.0, 1.0, 0.0),
        color: Color = (0.3, 0.8, 1.0),
        **options: Any,
    ) -> None:
        """Окружность с нормалью ``normal`` (по умолчанию лежит на земле)."""
        n = _vec(normal)
        n = n / max(float(np.linalg.norm(n)), 1e-9)
        side, up = _basis(n)
        self.segments(_ring(radius, side, up) + _vec(center), color, **options)

    def cone(
        self,
        apex: Any,
        direction: Any,
        angle_deg: float,
        distance: float,
        color: Color = (1.0, 0.4, 0.3),
        *,
        channel: str = "sight",
        **options: Any,
    ) -> None:
        """Конус зрения: вершина, направление, полуугол (градусы), дальность."""
        origin = _vec(apex)
        forward = _vec(direction)
        norm = float(np.linalg.norm(forward))
        if norm < 1e-9 or distance <= 0:
            return
        forward /= norm
        side, up = _basis(forward)
        half = np.radians(min(max(angle_deg, 0.0), 89.0))
        radius = np.tan(half) * distance
        base = origin + forward * distance
        ring = _ring(radius, side, up) + base
        rays = [origin, base + side * radius, origin, base - side * radius,
                origin, base + up * radius, origin, base - up * radius]
        points = np.concatenate([ring, np.stack(rays)])
        self.segments(points, color, channel=channel, **options)

    def text(
        self,
        position: Any,
        text: str,
        color: Color = (1.0, 1.0, 1.0),
        *,
        size: float = 0.02,
        duration: float = 0.0,
        channel: str = DEFAULT_CHANNEL,
    ) -> None:
        """Подпись в точке мира; ``size`` — мировых единиц на пиксель шрифта."""
        if not self.enabled or not text:
            return
        label = DebugText(
            tuple(_vec(position).tolist()), str(text), _color(color), size, channel
        )
        self._labels.append(_Label(label, float(duration)))

    # ------------------------------------------------------------------ #
    # Кадр
    # ------------------------------------------------------------------ #
    def collect(
        self, channels: Iterable[str] | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[DebugText]]:
        """Собрать видимое: ``(позиции, цвета, depth_test, подписи)``.

        Args:
            channels: Видимые каналы; ``None`` — все.

        Returns:
            Позиции ``float32 (n, 3)``, цвета ``float32 (n, 3)``, флаг теста
            глубины ``bool (n,)`` на каждую вершину и список подписей.
        """
        allowed = None if channels is None else set(channels)
        lines = [
            ln for ln in self._all_lines() if allowed is None or ln.channel in allowed
        ]
        labels = [
            lb.text for lb in self._labels
            if allowed is None or lb.text.channel in allowed
        ]
        if not lines:
            empty = np.zeros((0, 3), dtype=np.float32)
            return empty, empty.copy(), np.zeros(0, dtype=bool), labels
        positions = np.concatenate([ln.points for ln in lines])
        colors = np.concatenate(
            [np.broadcast_to(np.asarray(ln.color, np.float32), ln.points.shape)
             for ln in lines]
        )
        depth = np.concatenate(
            [np.full(len(ln.points), ln.depth_test, dtype=bool) for ln in lines]
        )
        return positions, colors.astype(np.float32), depth, labels

    def _all_lines(self) -> list[_Lines]:
        return [*self._lines, *self._keyed.values()]

    def advance(self, delta: float) -> None:
        """Конец отрисованного кадра: убрать истёкшее, уменьшить время жизни."""
        for item in self._all_lines():
            item.life -= delta
        for label in self._labels:
            label.life -= delta
        self._lines = [item for item in self._lines if item.life > 0.0]
        self._keyed = {k: v for k, v in self._keyed.items() if v.life > 0.0}
        self._labels = [label for label in self._labels if label.life > 0.0]

    def clear(self, channel: str | None = None) -> None:
        """Убрать всё (или только канал ``channel``)."""
        if channel is None:
            self._lines.clear()
            self._keyed.clear()
            self._labels.clear()
            return
        self._lines = [item for item in self._lines if item.channel != channel]
        self._keyed = {k: v for k, v in self._keyed.items() if v.channel != channel}
        self._labels = [lb for lb in self._labels if lb.text.channel != channel]

    @property
    def line_count(self) -> int:
        """Число накопленных отрезков (все каналы)."""
        return sum(len(item.points) // 2 for item in self._all_lines())

    @property
    def text_count(self) -> int:
        """Число накопленных подписей."""
        return len(self._labels)


def _color(color: Any) -> Color:
    to_tuple = getattr(color, "to_tuple", None)
    values = to_tuple() if to_tuple is not None else color
    r, g, b = (float(c) for c in tuple(values)[:3])
    return (r, g, b)
