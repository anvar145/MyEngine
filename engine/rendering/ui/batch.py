"""Пакетная отрисовка UI-квадов: панели, рамки, кнопки и картинки.

Как :class:`~engine.rendering.sprite_batch.SpriteBatch`, но без атласа спрайтов:
у каждого квада своя текстура (по умолчанию общий белый пиксель, поэтому цветные
панели и картинки делят один ``glDrawElements`` на текстуру). Вершины строятся
векторно numpy, все сегменты лежат в одном динамическом буфере.

Квады, у которых совпали текстура и прямоугольник отсечения, попадают в один
сегмент; порядок сегментов — по первому появлению.

Пример:
    >>> batch = UIBatch()
    >>> batch.add_quad(10, 10, 100, 40, (1.0, 1.0, 1.0, 1.0))
    >>> batch.draw(renderer, state, screen)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from engine.rendering.gl_debug import void_pointer
from engine.rendering.texture import GpuTexture, TextureFilter

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.rendering.ui.screen import UIScreen

__all__ = ["FLOATS_PER_VERTEX", "UIBatch"]

#: Число float на вершину: позиция (2), UV (2), цвет RGBA (4).
FLOATS_PER_VERTEX = 8

_QUAD_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
_FULL_UV = (0.0, 0.0, 1.0, 1.0)

#: Ключ сегмента: ``(id текстуры или 0, прямоугольник отсечения или None)``.
_Key = tuple[int, "tuple[float, float, float, float] | None"]


class UIBatch:
    """Накопитель UI-квадов.

    Attributes:
        last_draw_calls: Число ``glDrawElements`` в последнем :meth:`draw`.
        last_quads: Число квадов в последнем :meth:`draw`.
    """

    def __init__(self) -> None:
        self._segments: dict[_Key, tuple[GpuTexture | None, list[tuple]]] = {}
        self._count = 0
        self._white: GpuTexture | None = None
        self._vao = 0
        self._vbo = 0
        self._ebo = 0
        self._capacity = 0
        self.last_draw_calls = 0
        self.last_quads = 0
        self._dirty = True
        self._draws: list[tuple[GpuTexture | None, tuple | None, int]] = []
        #: Служебное для наполнителя сцены: токен последнего заполнения.
        self.fill_token: object = None

    def clear(self) -> None:
        """Забыть накопленные квады."""
        self._segments.clear()
        self._count = 0
        self._dirty = True
        self.fill_token = None

    def __len__(self) -> int:
        return self._count

    def add_quad(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        uv: tuple[float, float, float, float] = _FULL_UV,
        texture: GpuTexture | None = None,
        clip: tuple[float, float, float, float] | None = None,
    ) -> None:
        """Добавить квад (единицы UI, начало слева сверху).

        Args:
            x: Левый край.
            y: Верхний край.
            width: Ширина.
            height: Высота.
            color: Множитель цвета RGBA.
            uv: Прямоугольник текстуры ``(u0, v0, u1, v1)``; ``v0`` — верх.
            texture: Текстура; ``None`` — белый пиксель (залитый квад).
            clip: Прямоугольник отсечения ``(x, y, w, h)`` в единицах UI.
        """
        if width <= 0 or height <= 0 or color[3] <= 0.0:
            return
        key = (0 if texture is None else id(texture), clip)
        segment = self._segments.get(key)
        if segment is None:
            segment = self._segments[key] = (texture, [])
        segment[1].append((x, y, x + width, y + height, *uv, *color))
        self._count += 1
        self._dirty = True

    def build(self) -> list[tuple[GpuTexture | None, tuple | None, np.ndarray]]:
        """Вершины сегментов.

        Returns:
            Список ``(текстура, отсечение, вершины (4N, 8))`` в порядке рисования.
        """
        result = []
        for (_, clip), (texture, rows) in self._segments.items():
            data = np.asarray(rows, dtype=np.float32)
            count = len(data)
            vertices = np.empty((count, 4, FLOATS_PER_VERTEX), dtype=np.float32)
            x0, y0, x1, y1 = data[:, 0:1], data[:, 1:2], data[:, 2:3], data[:, 3:4]
            vertices[:, :, 0] = np.concatenate([x0, x1, x1, x0], axis=1)
            vertices[:, :, 1] = np.concatenate([y0, y0, y1, y1], axis=1)
            vertices[:, :, 2] = np.concatenate(
                [data[:, 4:5], data[:, 6:7], data[:, 6:7], data[:, 4:5]], axis=1
            )
            vertices[:, :, 3] = np.concatenate(
                [data[:, 5:6], data[:, 5:6], data[:, 7:8], data[:, 7:8]], axis=1
            )
            vertices[:, :, 4:8] = data[:, None, 8:12]
            result.append((texture, clip, vertices.reshape(-1, FLOATS_PER_VERTEX)))
        return result

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
            for location, size, offset in ((0, 2, 0), (1, 2, 8), (2, 4, 16)):
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
            self._capacity = max(quads, self._capacity * 2, 256)
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

    def _upload(self) -> None:
        """Собрать вершины и индексы и залить в буферы (только при изменении)."""
        from OpenGL import GL

        segments = self.build()
        vertices = np.concatenate([s[2] for s in segments])
        base = (np.arange(self._count, dtype=np.uint32) * 4)[:, None]
        indices = (base + _QUAD_INDICES[None, :]).reshape(-1)
        self._draws = [(s[0], s[1], len(s[2]) // 4) for s in segments]
        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        GL.glBufferSubData(GL.GL_ELEMENT_ARRAY_BUFFER, 0, indices.nbytes, indices)
        self._dirty = False

    def _white_texture(self) -> GpuTexture:
        """Белый пиксель 1×1 для залитых квадов."""
        if self._white is None:
            pixels = np.full((1, 1, 4), 255, dtype=np.uint8)
            self._white = GpuTexture.from_bytes(
                "ui_white",
                1,
                1,
                pixels,
                filter_mode=TextureFilter.NEAREST,
                generate_mips=False,
                wrap_repeat=False,
            )
        if not self._white.is_uploaded:
            self._white.upload()
        return self._white

    def draw(self, renderer, state, screen: UIScreen) -> int:
        """Нарисовать накопленные квады.

        Args:
            renderer: Рендерер (программа ``ui``).
            state: Кэш состояний OpenGL.
            screen: Экран UI (масштаб, смещение, размер окна).

        Returns:
            Число отрисованных квадов.
        """
        from OpenGL import GL

        self.last_draw_calls = 0
        self.last_quads = 0
        if self._count == 0:
            return 0
        program = renderer.program("ui")
        if not renderer._bind_program(program):  # noqa: SLF001 - тот же пакет
            return 0
        self._ensure_buffers(self._count)
        if self._dirty:
            self._upload()
        program.set_uniform("u_screen", np.asarray(screen.viewport, dtype=np.float32))
        program.set_uniform("u_scale", float(screen.scale))
        program.set_uniform("u_offset", np.asarray(screen.offset, dtype=np.float32))
        program.set_uniform("u_texture", 0)
        state.bind_vertex_array(self._vao)

        offset = 0
        scissor_on = False
        for texture, clip, quads in self._draws:
            gpu = self._white_texture() if texture is None else texture
            if not gpu.is_uploaded:
                gpu.upload()
            state.bind_texture(0, GL.GL_TEXTURE_2D, gpu.handle)
            if clip is not None:
                self._apply_scissor(state, screen, clip)
                scissor_on = True
            elif scissor_on:
                state.set_scissor(False)
                scissor_on = False
            GL.glDrawElements(
                GL.GL_TRIANGLES, quads * 6, GL.GL_UNSIGNED_INT, void_pointer(offset * 4)
            )
            offset += quads * 6
            self.last_draw_calls += 1
        if scissor_on:
            state.set_scissor(False)
        self.last_quads = self._count
        return self._count

    @staticmethod
    def _apply_scissor(state, screen: UIScreen, clip: tuple) -> None:
        """Включить scissor по прямоугольнику UI (в пикселях, начало снизу)."""
        x0, y0 = screen.ui_to_pixels(clip[0], clip[1])
        x1, y1 = screen.ui_to_pixels(clip[0] + clip[2], clip[1] + clip[3])
        view_h = screen.viewport[1]
        left, right = round(x0), round(x1)
        top, bottom = round(y0), round(y1)
        state.set_scissor(True)
        state.set_scissor_rect(
            left, view_h - bottom, max(0, right - left), max(0, bottom - top)
        )

    def dispose(self) -> None:
        """Освободить буферы и белую текстуру."""
        from OpenGL import GL

        if self._vao:
            GL.glDeleteVertexArrays(1, [self._vao])
            GL.glDeleteBuffers(2, [self._vbo, self._ebo])
            self._vao = self._vbo = self._ebo = 0
            self._capacity = 0
            self._dirty = True
        if self._white is not None:
            self._white.dispose()
            self._white = None
