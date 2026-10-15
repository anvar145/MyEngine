"""Экран UI: логическое разрешение, режим масштаба и якоря.

UI-координаты — «единицы UI» с началом в левом верхнем углу и осью Y вниз.
Экран переводит их в пиксели окна (``scale`` и ``offset``) и обратно (мышь).

Режимы масштаба:
    * ``fixed`` — холст равен референсному разрешению и целиком масштабируется
      в окно с сохранением пропорций (поля по бокам центрируются);
    * ``scale_with_screen`` — масштаб ``min(sx, sy)``, холст занимает всё окно;
    * ``match_height`` — масштаб по высоте, холст занимает всё окно.

Пример:
    >>> screen = UIScreen((1280, 720))
    >>> screen.set_viewport(1920, 1080)
    >>> screen.mouse_to_ui(960, 540)
    (640.0, 360.0)
"""

from __future__ import annotations

import numpy as np

__all__ = ["ANCHORS", "SCALE_MODES", "UIScreen"]

#: Допустимые значения ``anchor``: доли ширины и высоты родителя.
ANCHORS: dict[str, tuple[float, float]] = {
    "top_left": (0.0, 0.0),
    "top_center": (0.5, 0.0),
    "top_right": (1.0, 0.0),
    "middle_left": (0.0, 0.5),
    "middle_center": (0.5, 0.5),
    "middle_right": (1.0, 0.5),
    "bottom_left": (0.0, 1.0),
    "bottom_center": (0.5, 1.0),
    "bottom_right": (1.0, 1.0),
}

SCALE_MODES = ("fixed", "scale_with_screen", "match_height")


class UIScreen:
    """Логический экран UI.

    Args:
        reference_resolution: Референсное разрешение в единицах UI.
        scale_mode: Один из :data:`SCALE_MODES`.

    Attributes:
        reference_resolution: Референсное разрешение ``(w, h)``.
        scale_mode: Режим масштаба.
        viewport: Размер окна/цели в пикселях.
    """

    def __init__(
        self,
        reference_resolution: tuple[int, int] = (1280, 720),
        scale_mode: str = "fixed",
    ) -> None:
        self.reference_resolution = (1280, 720)
        self.scale_mode = "fixed"
        self.viewport = (1280, 720)
        self.set_reference_resolution(*reference_resolution)
        self.set_scale_mode(scale_mode)

    def set_reference_resolution(self, width: int, height: int) -> None:
        """Задать референсное разрешение.

        Raises:
            ValueError: Если размер не положителен.
        """
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Разрешение UI должно быть положительным: {width}x{height}"
            )
        self.reference_resolution = (int(width), int(height))

    def set_scale_mode(self, mode: str) -> None:
        """Выбрать режим масштаба.

        Raises:
            ValueError: Если режим неизвестен.
        """
        if mode not in SCALE_MODES:
            raise ValueError(f"Неизвестный режим масштаба UI: {mode!r}")
        self.scale_mode = mode

    def set_viewport(self, width: int, height: int) -> None:
        """Сообщить размер окна (или цели вывода) в пикселях."""
        self.viewport = (max(1, int(width)), max(1, int(height)))

    # ------------------------------------------------------------------ #
    # Геометрия
    # ------------------------------------------------------------------ #
    @property
    def scale(self) -> float:
        """Пикселей окна на одну единицу UI."""
        ref_w, ref_h = self.reference_resolution
        view_w, view_h = self.viewport
        if self.scale_mode == "match_height":
            return view_h / ref_h
        return min(view_w / ref_w, view_h / ref_h)

    @property
    def canvas_size(self) -> tuple[float, float]:
        """Размер холста в единицах UI."""
        if self.scale_mode == "fixed":
            ref_w, ref_h = self.reference_resolution
            return (float(ref_w), float(ref_h))
        scale = self.scale
        return (self.viewport[0] / scale, self.viewport[1] / scale)

    @property
    def offset(self) -> tuple[float, float]:
        """Смещение холста в пикселях (центрирование для ``fixed``)."""
        scale = self.scale
        width, height = self.canvas_size
        return (
            (self.viewport[0] - width * scale) * 0.5,
            (self.viewport[1] - height * scale) * 0.5,
        )

    def mouse_to_ui(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Перевести пиксели окна (начало слева сверху) в координаты UI."""
        scale = self.scale
        off_x, off_y = self.offset
        return ((x_px - off_x) / scale, (y_px - off_y) / scale)

    def ui_to_pixels(self, x: float, y: float) -> tuple[float, float]:
        """Перевести координаты UI в пиксели окна."""
        scale = self.scale
        off_x, off_y = self.offset
        return (x * scale + off_x, y * scale + off_y)

    def canvas_rect(self) -> tuple[float, float, float, float]:
        """Прямоугольник холста ``(x, y, w, h)`` — корневой «родитель» виджетов."""
        width, height = self.canvas_size
        return (0.0, 0.0, width, height)

    @staticmethod
    def resolve_rect(
        anchor: str,
        pivot: tuple[float, float],
        position: tuple[float, float],
        size: tuple[float, float],
        parent: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """Прямоугольник виджета ``(x, y, w, h)`` в единицах UI.

        Точка якоря берётся внутри ``parent``; левый верх виджета равен
        ``якорь + position - pivot * size``.
        """
        ax, ay = ANCHORS[anchor]
        px, py, pw, ph = parent
        width, height = size
        x = px + ax * pw + position[0] - pivot[0] * width
        y = py + ay * ph + position[1] - pivot[1] * height
        return (x, y, width, height)

    def projection(self) -> np.ndarray:
        """Матрица для ``text``-программы: единицы UI с осью Y вверх → NDC.

        Текст раскладывается в системе с осью Y вверх; наполнитель подставляет
        ``y = canvas_height - y_ui``. Матрица построчная, как все матрицы ядра:
        ``set_uniform`` сам транспонирует её для OpenGL.
        """
        scale = self.scale
        off_x, off_y = self.offset
        view_w, view_h = self.viewport
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 0] = 2.0 * scale / view_w
        matrix[0, 3] = 2.0 * off_x / view_w - 1.0
        matrix[1, 1] = 2.0 * scale / view_h
        matrix[1, 3] = 2.0 * off_y / view_h - 1.0
        return matrix
