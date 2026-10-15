"""Пакетная отрисовка спрайтов из атласа (``SpriteAtlas``).

Вершины квадов собираются на CPU одним векторным вызовом numpy (без Python-цикла
по вершинам) и уходят в один динамический буфер; весь атлас рисуется одним
``glDrawElements``. Спрайты лежат в мировом пространстве и проходят через
камеру сцены, поэтому подходят и для 2D-уровней, и для табличек в 3D.

Пример:
    >>> batch = SpriteBatch(atlas)
    >>> batch.add("hero_idle", (0.0, 1.0, 0.0), rotation=0.3)
    >>> renderer.add_pass(SpritePass(batch))
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from engine.rendering.errors import RenderError
from engine.rendering.gl_debug import void_pointer
from engine.rendering.texture import GpuTexture

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.resources.sprite_atlas import SpriteAtlas
    from engine.rendering.render_graph import FrameContext

__all__ = ["SpriteBatch", "SpritePass"]

_log = logging.getLogger("engine.rendering.sprite_batch")

#: Число float на вершину: позиция (3), UV (2), цвет (4).
FLOATS_PER_VERTEX = 9

#: Столбцы строки спрайта: кадр, позиция xyz, масштаб xy, поворот, flip xy, цвет rgba.
_ROW = 13

#: Порядок вершин квада: левый верх, правый верх, правый низ, левый низ.
_QUAD_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)


class SpriteBatch:
    """Накопитель спрайтов одного атласа.

    Args:
        atlas: Атлас с кадрами и текстурой.
        pixels_per_unit: Сколько пикселей кадра приходится на мировую единицу.
        texture: Готовая GPU-текстура; по умолчанию создаётся из ``texture.bin``
            атласа при первой отрисовке.

    Attributes:
        atlas: Атлас спрайтов.
        pixels_per_unit: Масштаб «пиксели → мир».
        alpha_cutoff: Фрагменты с меньшей альфой отбрасываются.
    """

    def __init__(
        self,
        atlas: SpriteAtlas,
        *,
        pixels_per_unit: float = 100.0,
        texture: GpuTexture | None = None,
    ) -> None:
        if pixels_per_unit <= 0:
            raise RenderError("pixels_per_unit должен быть положительным")
        self.atlas = atlas
        self.pixels_per_unit = float(pixels_per_unit)
        self.alpha_cutoff = 0.01
        self.texture = texture
        self._names = list(atlas.frames)
        self._index = {name: i for i, name in enumerate(self._names)}
        self._table = self._frame_table()
        self._rows: list[tuple[float, ...]] = []
        self._vao = 0
        self._vbo = 0
        self._ebo = 0
        self._capacity = 0

    # ------------------------------------------------------------------ #
    # Подготовка данных
    # ------------------------------------------------------------------ #
    def _frame_table(self) -> np.ndarray:
        """Таблица кадров: углы квада в мировых единицах и UV (строка на кадр).

        Столбцы: ``x0, x1, y_top, y_bottom, u0, v0, u1, v1``. Углы отсчитываются
        от точки привязки (pivot) исходного, необрезанного спрайта, ось Y — вверх.
        """
        table = np.zeros((max(1, len(self._names)), 8), dtype=np.float32)
        ppu = self.pixels_per_unit
        for i, name in enumerate(self._names):
            frame = self.atlas.frames[name]
            source_w, source_h = frame.source_size
            if source_w <= 0 or source_h <= 0:
                source_w, source_h = frame.w, frame.h
            pivot_x = frame.pivot[0] * source_w
            pivot_y = frame.pivot[1] * source_h
            left, top = frame.offset
            table[i, 0] = (left - pivot_x) / ppu
            table[i, 1] = (left + frame.w - pivot_x) / ppu
            table[i, 2] = (pivot_y - top) / ppu
            table[i, 3] = (pivot_y - top - frame.h) / ppu
            table[i, 4:] = frame.uv(self.atlas.width, self.atlas.height)
        return table

    def clear(self) -> None:
        """Забыть накопленные спрайты (вызывать в начале кадра)."""
        self._rows.clear()

    def __len__(self) -> int:
        return len(self._rows)

    def add(
        self,
        name: str,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        *,
        scale: float | tuple[float, float] = 1.0,
        rotation: float = 0.0,
        color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        flip_x: bool = False,
        flip_y: bool = False,
    ) -> None:
        """Добавить спрайт в пакет.

        Args:
            name: Имя кадра в атласе.
            position: Положение точки привязки в мире.
            scale: Масштаб (число или пара ``(x, y)``).
            rotation: Поворот вокруг оси Z в радианах.
            color: Множитель цвета RGBA.
            flip_x: Отразить по горизонтали (вокруг точки привязки).
            flip_y: Отразить по вертикали.

        Raises:
            KeyError: Если кадра с таким именем нет в атласе.
        """
        try:
            index = self._index[name]
        except KeyError:
            raise KeyError(f"В атласе нет спрайта '{name}'") from None
        scale_x, scale_y = (scale, scale) if isinstance(scale, (int, float)) else scale
        self._rows.append(
            (
                float(index),
                *position,
                float(scale_x),
                float(scale_y),
                float(rotation),
                -1.0 if flip_x else 1.0,
                -1.0 if flip_y else 1.0,
                *color,
            )
        )

    def build(self) -> tuple[np.ndarray, np.ndarray]:
        """Собрать вершины и индексы всех накопленных спрайтов.

        Returns:
            Пару ``(vertices, indices)``: массив ``float32`` формы ``(4N, 9)``
            и ``uint32`` длины ``6N``.
        """
        count = len(self._rows)
        if count == 0:
            return (
                np.zeros((0, FLOATS_PER_VERTEX), dtype=np.float32),
                np.zeros(0, dtype=np.uint32),
            )
        rows = np.asarray(self._rows, dtype=np.float32)
        assert rows.shape[1] == _ROW
        info = self._table[rows[:, 0].astype(np.intp)]

        flip_x = rows[:, 7]
        flip_y = rows[:, 8]
        # Углы в локальных осях: (N, 4) для X и Y, порядок TL, TR, BR, BL.
        local_x = np.stack([info[:, 0], info[:, 1], info[:, 1], info[:, 0]], axis=1)
        local_y = np.stack([info[:, 2], info[:, 2], info[:, 3], info[:, 3]], axis=1)
        local_x = local_x * (rows[:, 4] * flip_x)[:, None]
        local_y = local_y * (rows[:, 5] * flip_y)[:, None]

        cos = np.cos(rows[:, 6])[:, None]
        sin = np.sin(rows[:, 6])[:, None]
        world_x = rows[:, 1][:, None] + local_x * cos - local_y * sin
        world_y = rows[:, 2][:, None] + local_x * sin + local_y * cos
        world_z = np.broadcast_to(rows[:, 3][:, None], world_x.shape)

        u0, v0, u1, v1 = info[:, 4], info[:, 5], info[:, 6], info[:, 7]
        # Флип меняет геометрию (знак при масштабе), UV остаются прежними.
        uv_u = np.stack([u0, u1, u1, u0], axis=1)
        uv_v = np.stack([v0, v0, v1, v1], axis=1)

        vertices = np.empty((count, 4, FLOATS_PER_VERTEX), dtype=np.float32)
        vertices[:, :, 0] = world_x
        vertices[:, :, 1] = world_y
        vertices[:, :, 2] = world_z
        vertices[:, :, 3] = uv_u
        vertices[:, :, 4] = uv_v
        vertices[:, :, 5:9] = rows[:, None, 9:13]
        base = (np.arange(count, dtype=np.uint32) * 4)[:, None]
        indices = (base + _QUAD_INDICES[None, :]).reshape(-1)
        return vertices.reshape(-1, FLOATS_PER_VERTEX), indices

    # ------------------------------------------------------------------ #
    # GPU
    # ------------------------------------------------------------------ #
    def _ensure_buffers(self, sprites: int) -> None:
        """Создать VAO/буферы и при необходимости увеличить их ёмкость."""
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
        if sprites > self._capacity:
            self._capacity = max(sprites, self._capacity * 2, 256)
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

    def preload(self) -> None:
        """Залить текстуру атласа в видеопамять заранее (до первого кадра)."""
        self._ensure_texture()

    def _ensure_texture(self) -> GpuTexture:
        """Текстура атласа в видеопамяти (создаётся при первом обращении)."""
        if self.texture is None:
            self.texture = GpuTexture.from_cooked(
                "sprite_atlas",
                self.atlas.texture(),
                wrap_repeat=False,
            )
        if not self.texture.is_uploaded:
            self.texture.upload()
        return self.texture

    def draw(self, renderer, state) -> int:
        """Нарисовать накопленные спрайты.

        Args:
            renderer: Рендерер (даёт программу ``sprite`` и uniform камеры).
            state: Кэш состояний OpenGL.

        Returns:
            Число отрисованных спрайтов.
        """
        from OpenGL import GL

        vertices, indices = self.build()
        sprites = len(self._rows)
        if sprites == 0:
            return 0
        program = renderer.program("sprite")
        texture = self._ensure_texture()
        self._ensure_buffers(sprites)

        state.set_blend(True)
        state.set_depth_write(False)
        state.set_cull(False)
        renderer._apply_camera_uniforms(program)  # noqa: SLF001 - тот же пакет
        state.bind_texture(0, GL.GL_TEXTURE_2D, texture.handle)
        program.set_uniform("u_texture", 0)
        program.set_uniform("u_alpha_cutoff", self.alpha_cutoff)

        state.bind_vertex_array(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        GL.glBufferSubData(GL.GL_ELEMENT_ARRAY_BUFFER, 0, indices.nbytes, indices)
        GL.glDrawElements(GL.GL_TRIANGLES, indices.size, GL.GL_UNSIGNED_INT, None)

        state.set_depth_write(True)
        state.set_blend(False)
        return sprites

    def dispose(self) -> None:
        """Освободить буферы и текстуру атласа."""
        from OpenGL import GL

        if self._vao:
            GL.glDeleteVertexArrays(1, [self._vao])
            GL.glDeleteBuffers(2, [self._vbo, self._ebo])
            self._vao = self._vbo = self._ebo = 0
            self._capacity = 0
        if self.texture is not None:
            self.texture.dispose()
            self.texture = None


class SpritePass:
    """Проход графа кадра: рисует пакеты спрайтов поверх геометрии.

    Args:
        *batches: Пакеты спрайтов (рисуются по порядку).
    """

    def __init__(self, *batches: SpriteBatch) -> None:
        self.batches = list(batches)
        self.enabled = True

    @property
    def name(self) -> str:
        """Имя прохода."""
        return "sprites"

    def execute(self, context: FrameContext) -> None:
        """Нарисовать все пакеты."""
        if not self.enabled or context.renderer is None:
            return
        for batch in self.batches:
            batch.draw(context.renderer, context.state)
