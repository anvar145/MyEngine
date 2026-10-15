"""Пакетная отрисовка текста: квады глифов собираются numpy и рисуются по атласам.

Раскладка (см. :mod:`engine.rendering.text.layout`) кэшируется, поэтому за кадр
на каждый текст приходится несколько append-ов; вершины всех текстов строятся
одним векторным вызовом на атлас (без Python-цикла по глифам). Все атласы идут
в один динамический буфер, на каждый атлас — один ``glDrawElements``.

Порядок: внутри атласа тексты сортируются по ``(sorting_layer, -расстояние)``
(дальние раньше); между атласами порядок — по первому появлению атласа.

Пример:
    >>> batch = TextBatch()
    >>> batch.add_text(atlas, "Привет", (0, 1, 0), scale=0.01, billboard=True)
    >>> renderer.add_pass(TextPass(batch))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from engine.rendering.gl_debug import void_pointer
from engine.rendering.text.layout import layout_text

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.rendering.camera import Camera
    from engine.rendering.text.atlas import TextAtlas

__all__ = ["FLOATS_PER_VERTEX", "TextBatch"]

#: Число float на вершину: позиция (3), UV (2), цвет RGBA (4).
FLOATS_PER_VERTEX = 9

_QUAD_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
_WORLD_RIGHT = (1.0, 0.0, 0.0)
_WORLD_UP = (0.0, 1.0, 0.0)

#: Направления смещённых копий для обводки (8 сторон).
_OUTLINE_DIRECTIONS = tuple(
    (math.cos(i * math.pi / 4.0), math.sin(i * math.pi / 4.0)) for i in range(8)
)


@dataclass
class _Item:
    atlas: TextAtlas
    quads: np.ndarray  # (N, 8) в пикселях атласа
    origin: tuple[float, float, float]
    scale: float
    billboard: bool
    right: tuple[float, float, float]
    up: tuple[float, float, float]
    color: tuple[float, float, float, float]
    layer: int
    outline_px: float
    outline_color: tuple[float, float, float, float]


def _axes(matrix: np.ndarray, scale: float):
    right = tuple((matrix[:3, 0] * scale).tolist())
    up = tuple((matrix[:3, 1] * scale).tolist())
    return right, up


class TextBatch:
    """Накопитель текстов (любого числа атласов).

    Attributes:
        last_draw_calls: Число ``glDrawElements`` в последнем :meth:`draw`.
        last_quads: Число квадов в последнем :meth:`draw`.
    """

    def __init__(self) -> None:
        self._items: list[_Item] = []
        self._vao = 0
        self._vbo = 0
        self._ebo = 0
        self._capacity = 0
        self.last_draw_calls = 0
        self.last_quads = 0
        # Кэш последней сборки: (ключ камеры, группы, число квадов); в GPU-буферах
        # лежат именно эти данные, пока кэш не сброшен.
        self._cache: tuple | None = None
        #: Служебное для :func:`fill_text_batch`: токен последнего заполнения.
        self.fill_token: object = None
        self.fill_count = 0

    # ------------------------------------------------------------------ #
    # Накопление
    # ------------------------------------------------------------------ #
    def clear(self) -> None:
        """Забыть накопленные тексты (в начале кадра)."""
        self._items.clear()
        self._cache = None
        self.fill_token = None
        self.fill_count = 0

    def __len__(self) -> int:
        return len(self._items)

    def add_glyph(
        self,
        atlas: TextAtlas,
        codepoint: int,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: float = 1.0,
        color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        alpha: float = 1.0,
    ) -> None:
        """Добавить один глиф; ``position`` — левая точка базовой линии."""
        glyph = atlas.get_glyph(codepoint)
        if glyph.width_px > 0:
            u0, v0, u1, v1 = glyph.uv_rect
            y1 = glyph.bearing_y
            quads = np.array(
                [
                    [
                        glyph.bearing_x,
                        y1 - glyph.height_px,
                        glyph.bearing_x + glyph.width_px,
                        y1,
                        u0,
                        v0,
                        u1,
                        v1,
                    ]
                ],
                dtype=np.float32,
            )
        else:
            quads = np.zeros((0, 8), dtype=np.float32)
        self._append(atlas, quads, position, scale, color, alpha, False, None, 0, 0.0)

    def add_text(
        self,
        atlas: TextAtlas,
        text: str,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: float = 1.0,
        color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        alpha: float = 1.0,
        align: str = "left",
        wrap_width: float = 0.0,
        *,
        vertical_align: str = "top",
        line_spacing: float = 1.0,
        billboard: bool = False,
        matrix: np.ndarray | None = None,
        sorting_layer: int = 0,
        outline_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        outline_width: float = 0.0,
    ) -> None:
        """Добавить текст.

        Args:
            atlas: Атлас шрифта.
            text: Строка (``\\n`` — перенос).
            position: Точка привязки в мире (задаётся выравниванием).
            scale: Мировых единиц на пиксель атласа.
            color: Цвет RGB.
            alpha: Непрозрачность 0..1.
            align: Горизонтальное выравнивание.
            wrap_width: Ширина переноса в мировых единицах (0 — нет).
            vertical_align: Вертикальное выравнивание.
            line_spacing: Множитель межстрочного интервала.
            billboard: Всегда лицом к камере (оси берутся из камеры при сборке).
            matrix: Мировая матрица 4×4 объекта (оси X/Y); без неё — оси мира.
            sorting_layer: Слой сортировки.
            outline_color: Цвет обводки.
            outline_width: Толщина обводки в пикселях атласа (0 — нет).
        """
        if not text:
            return
        wrap_px = wrap_width / scale if wrap_width > 0 and scale > 0 else 0.0
        layout = layout_text(
            atlas,
            text,
            align=align,
            vertical_align=vertical_align,
            wrap_width=wrap_px,
            line_spacing=line_spacing,
        )
        if len(layout.quads) == 0:
            return
        self._append(
            atlas,
            layout.quads,
            position,
            scale,
            color,
            alpha,
            billboard,
            matrix,
            sorting_layer,
            outline_width,
            outline_color,
        )

    def _append(
        self,
        atlas: TextAtlas,
        quads: np.ndarray,
        position: tuple[float, float, float],
        scale: float,
        color: tuple[float, float, float],
        alpha: float,
        billboard: bool,
        matrix: np.ndarray | None,
        layer: int,
        outline_px: float,
        outline_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        if matrix is None:
            right = tuple(v * scale for v in _WORLD_RIGHT)
            up = tuple(v * scale for v in _WORLD_UP)
        elif billboard:
            # Оси возьмутся из камеры при сборке.
            right = up = _WORLD_RIGHT
        else:
            right, up = _axes(np.asarray(matrix), scale)
        self._cache = None
        self._items.append(
            _Item(
                atlas,
                quads,
                (float(position[0]), float(position[1]), float(position[2])),
                float(scale),
                billboard,
                right,
                up,
                (float(color[0]), float(color[1]), float(color[2]), float(alpha)),
                int(layer),
                float(outline_px),
                (
                    float(outline_color[0]),
                    float(outline_color[1]),
                    float(outline_color[2]),
                    float(alpha),
                ),
            )
        )

    # ------------------------------------------------------------------ #
    # Сборка вершин
    # ------------------------------------------------------------------ #
    def _group_arrays(
        self, items: list[_Item], camera: Camera | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Вершины и индексы (с нуля) для списка текстов одного атласа."""
        if camera is not None:
            cam_pos = camera.position
            # Экранные оси берём из видовой матрицы: ``camera.right`` (+X камеры)
            # у камеры, смотрящей вдоль +Z, направлен влево на экране.
            view = camera.view_matrix
            cam_right = (float(view[0, 0]), float(view[0, 1]), float(view[0, 2]))
            cam_up = (float(view[1, 0]), float(view[1, 1]), float(view[1, 2]))
            cam_xyz: tuple[float, float, float] | None = (
                cam_pos.x,
                cam_pos.y,
                cam_pos.z,
            )
        else:
            cam_right, cam_up, cam_xyz = _WORLD_RIGHT, _WORLD_UP, None

        def sort_key(item: _Item) -> tuple[int, float]:
            far = 0.0 if cam_xyz is None else math.dist(item.origin, cam_xyz)
            return (item.layer, -far)

        quads: list[np.ndarray] = []
        counts: list[int] = []
        origins: list[tuple] = []
        rights: list[tuple] = []
        ups: list[tuple] = []
        colors: list[tuple] = []
        for item in sorted(items, key=sort_key):
            if item.billboard:
                right = tuple(v * item.scale for v in cam_right)
                up = tuple(v * item.scale for v in cam_up)
            else:
                right, up = item.right, item.up
            if item.outline_px > 0:
                for dx, dy in _OUTLINE_DIRECTIONS:
                    shifted = item.quads.copy()
                    shifted[:, (0, 2)] += dx * item.outline_px
                    shifted[:, (1, 3)] += dy * item.outline_px
                    quads.append(shifted)
                    counts.append(len(shifted))
                    origins.append(item.origin)
                    rights.append(right)
                    ups.append(up)
                    colors.append(item.outline_color)
            quads.append(item.quads)
            counts.append(len(item.quads))
            origins.append(item.origin)
            rights.append(right)
            ups.append(up)
            colors.append(item.color)

        data = np.concatenate(quads)
        count = len(data)
        rep = np.asarray(counts)
        origin = np.repeat(np.asarray(origins, dtype=np.float32), rep, axis=0)
        right = np.repeat(np.asarray(rights, dtype=np.float32), rep, axis=0)
        up = np.repeat(np.asarray(ups, dtype=np.float32), rep, axis=0)
        color = np.repeat(np.asarray(colors, dtype=np.float32), rep, axis=0)

        x0, y0, x1, y1 = data[:, 0:1], data[:, 1:2], data[:, 2:3], data[:, 3:4]
        left = origin + right * x0
        right_edge = origin + right * x1
        low = up * y0
        high = up * y1
        vertices = np.empty((count, 4, FLOATS_PER_VERTEX), dtype=np.float32)
        vertices[:, 0, 0:3] = left + high
        vertices[:, 1, 0:3] = right_edge + high
        vertices[:, 2, 0:3] = right_edge + low
        vertices[:, 3, 0:3] = left + low
        uv = vertices[:, :, 3:5]
        uv[:, (0, 3), 0] = data[:, 4:5]
        uv[:, (1, 2), 0] = data[:, 6:7]
        uv[:, (0, 1), 1] = data[:, 5:6]
        uv[:, (2, 3), 1] = data[:, 7:8]
        vertices[:, :, 5:9] = color[:, None, :]
        base = (np.arange(count, dtype=np.uint32) * 4)[:, None]
        indices = (base + _QUAD_INDICES[None, :]).reshape(-1)
        return vertices.reshape(-1, FLOATS_PER_VERTEX), indices

    def _groups(
        self, camera: Camera | None
    ) -> list[tuple[TextAtlas, np.ndarray, np.ndarray]]:
        by_atlas: dict[int, tuple[TextAtlas, list[_Item]]] = {}
        for item in self._items:
            if len(item.quads) == 0:
                continue
            by_atlas.setdefault(id(item.atlas), (item.atlas, []))[1].append(item)
        return [
            (atlas, *self._group_arrays(items, camera))
            for atlas, items in by_atlas.values()
        ]

    @staticmethod
    def _merge(
        groups: list[tuple[TextAtlas, np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray]:
        vertices, indices = [], []
        base = 0
        for _, group_vertices, group_indices in groups:
            vertices.append(group_vertices)
            indices.append(group_indices + np.uint32(base))
            base += len(group_vertices)
        return np.concatenate(vertices), np.concatenate(indices)

    def build(self, camera: Camera | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Собрать вершины и индексы всех текстов (все атласы подряд).

        Args:
            camera: Камера для billboard-ориентации и сортировки по дальности.

        Returns:
            ``(vertices, indices)``: ``float32`` ``(4N, 9)`` и ``uint32`` ``6N``.
        """
        groups = self._groups(camera)
        if not groups:
            return (
                np.zeros((0, FLOATS_PER_VERTEX), dtype=np.float32),
                np.zeros(0, dtype=np.uint32),
            )
        return self._merge(groups)

    # ------------------------------------------------------------------ #
    # GPU
    # ------------------------------------------------------------------ #
    def _ensure_buffers(self, quads: int) -> None:
        from OpenGL import GL

        if not self._vao:
            self._vao = int(GL.glGenVertexArrays(1))
            self._vbo = int(GL.glGenBuffers(1))
            self._ebo = int(GL.glGenBuffers(1))
            stride = FLOATS_PER_VERTEX * 4
            GL.glBindVertexArray(self._vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
            for location, size, offset in ((0, 3, 0), (1, 2, 12), (2, 4, 20)):
                GL.glEnableVertexAttribArray(location)
                GL.glVertexAttribPointer(
                    location,
                    size,
                    GL.GL_FLOAT,
                    GL.GL_FALSE,
                    stride,
                    void_pointer(offset),
                )
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        else:
            GL.glBindVertexArray(self._vao)
        if quads > self._capacity:
            self._capacity = max(quads, self._capacity * 2, 512)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER,
                self._capacity * 4 * FLOATS_PER_VERTEX * 4,
                None,
                GL.GL_DYNAMIC_DRAW,
            )
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
            GL.glBufferData(
                GL.GL_ELEMENT_ARRAY_BUFFER,
                self._capacity * 6 * 4,
                None,
                GL.GL_DYNAMIC_DRAW,
            )

    def draw(
        self,
        renderer,
        state,
        camera: Camera | None = None,
        *,
        projection: np.ndarray | None = None,
    ) -> int:
        """Нарисовать накопленные тексты.

        Args:
            renderer: Рендерер (программа ``text`` и uniform камеры).
            state: Кэш состояний OpenGL.
            camera: Камера для billboard и сортировки (``None`` — без неё).
            projection: Экранный режим (UI): матрица, которой заменяется
                ``u_view_projection``; тест глубины выключается, порядок — по
                ``sorting_layer`` и порядку добавления.

        Returns:
            Число отрисованных квадов.
        """
        from OpenGL import GL

        self.last_draw_calls = 0
        self.last_quads = 0
        key = self._camera_key(camera)
        cache = self._cache
        if cache is not None and cache[0] == key:
            # Ничего не менялось: вершины уже в буфере, пересборка и загрузка не нужны.
            _, groups, total = cache
            uploaded = True
        else:
            groups = self._groups(camera)
            total = sum(len(g[2]) // 6 for g in groups)
            uploaded = False
        if not groups:
            return 0
        program = renderer.program("text")
        self._ensure_buffers(total)
        if not uploaded:
            vertices, indices = self._merge(groups)
            self._cache = (key, groups, total)
        else:
            vertices = indices = None

        state.set_blend(True)
        state.set_depth_write(False)
        state.set_cull(False)
        if projection is None:
            renderer._apply_camera_uniforms(program)  # noqa: SLF001 - тот же пакет
        elif renderer._bind_program(program):  # noqa: SLF001 - тот же пакет
            program.set_uniform("u_view_projection", projection)
        program.set_uniform("u_atlas", 0)
        program.set_uniform("u_alpha", 1.0)
        program.set_uniform("u_model", np.eye(4, dtype=np.float32))

        state.bind_vertex_array(self._vao)
        if vertices is not None:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
            GL.glBufferSubData(GL.GL_ELEMENT_ARRAY_BUFFER, 0, indices.nbytes, indices)
        offset = 0
        for atlas, _, group_indices in groups:
            state.bind_texture(0, GL.GL_TEXTURE_2D, atlas.texture().handle)
            GL.glDrawElements(
                GL.GL_TRIANGLES,
                group_indices.size,
                GL.GL_UNSIGNED_INT,
                void_pointer(offset * 4),
            )
            offset += group_indices.size
            self.last_draw_calls += 1
        self.last_quads = total

        state.set_depth_write(True)
        state.set_blend(False)
        return total

    @staticmethod
    def _camera_key(camera: Camera | None) -> bytes | None:
        """Ключ кэша сборки: положение и ориентация камеры."""
        if camera is None:
            return None
        position = camera.position
        return (
            np.asarray((position.x, position.y, position.z), dtype=np.float64).tobytes()
            + np.asarray(camera.view_matrix).tobytes()
        )

    def dispose(self) -> None:
        """Освободить буферы (текстуры принадлежат атласам)."""
        from OpenGL import GL

        self._cache = None

        if self._vao:
            GL.glDeleteVertexArrays(1, [self._vao])
            GL.glDeleteBuffers(2, [self._vbo, self._ebo])
            self._vao = self._vbo = self._ebo = 0
            self._capacity = 0
