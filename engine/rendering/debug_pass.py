"""Проход графа кадра для отладочной отрисовки (C1): ``engine.debug`` поверх кадра.

Берёт отрезки и подписи из :class:`~engine.debug.draw.DebugDraw` для каналов,
которые включены в :class:`~engine.debug.overlays.OverlayRegistry`, и рисует их
двумя вызовами: сначала линии без теста глубины («сквозь стены»), затем с ним.
Подписи — billboard-текстом шрифта по умолчанию, тоже поверх геометрии.

Проход ставится в конец основной фазы (после прозрачных и каркасов
коллайдеров), поэтому рисует поверх всей геометрии сцены, но под UI.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from engine.debug import DebugDraw, OverlayRegistry
from engine.debug import draw as default_draw
from engine.debug import overlays as default_overlays

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.rendering.render_graph import FrameContext

__all__ = ["DEBUG_TEXT_SIZE_PX", "DebugPass"]

_log = logging.getLogger("engine.rendering.debug_pass")

#: Размер шрифта подписей в пикселях атласа.
DEBUG_TEXT_SIZE_PX = 24


class DebugPass:
    """Рисует буфер отладки для видимых оверлеев.

    Args:
        draw: Буфер; по умолчанию общий ``engine.debug.draw``.
        registry: Реестр оверлеев; по умолчанию общий ``engine.debug.overlays``.
        context: Что передать провайдерам оверлеев (сцена и т. п.); вызываемый
            объект вызывается каждый кадр.
        auto_advance: Сам продвигать время буфера после кадра (редактор: там
            нет игрового цикла, который вызывает ``draw.advance``).

    Attributes:
        line_count: Отрезков в последнем кадре.
        text_count: Подписей в последнем кадре.
    """

    def __init__(
        self,
        draw: DebugDraw | None = None,
        registry: OverlayRegistry | None = None,
        context: Any = None,
        *,
        auto_advance: bool = False,
    ) -> None:
        self.draw = default_draw if draw is None else draw
        self.registry = default_overlays if registry is None else registry
        self.context = context
        self.auto_advance = auto_advance
        self._last_time: float | None = None
        self.line_count = 0
        self.text_count = 0
        self._vao = 0
        self._vbo = 0
        self._capacity = 0
        self._text_batch: Any = None
        # Пока оверлеи скрыты, вызовы debug.* не копят геометрию.
        self._sync_enabled(self.registry)
        self.registry.subscribe(self._sync_enabled)

    @property
    def name(self) -> str:
        """Имя прохода."""
        return "debug"

    def _sync_enabled(self, registry: OverlayRegistry) -> None:
        self.draw.enabled = registry.visible

    # ------------------------------------------------------------------ #
    # Кадр
    # ------------------------------------------------------------------ #
    def gather(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """Вызвать провайдеры и собрать видимое (без GL; для тестов и прохода)."""
        channels = self.registry.active_channels()
        if not channels:
            empty = np.zeros((0, 3), dtype=np.float32)
            return empty, empty.copy(), np.zeros(0, dtype=bool), []
        context = self.context() if callable(self.context) else self.context
        self.registry.run_providers(self.draw, context)
        return self.draw.collect(channels)

    def execute(self, context: FrameContext) -> None:
        """Нарисовать линии и подписи видимых оверлеев."""
        renderer = context.renderer
        self.line_count = self.text_count = 0
        if renderer is None or not self.registry.visible:
            return
        positions, colors, depth, labels = self.gather()
        if len(positions):
            self._draw_lines(context, renderer, positions, colors, depth)
        if labels:
            self._draw_labels(context, renderer, labels)
        if self.auto_advance:
            now = time.perf_counter()
            last, self._last_time = self._last_time, now
            self.draw.advance(0.0 if last is None else now - last)

    def _draw_lines(
        self,
        context: FrameContext,
        renderer: Any,
        positions: np.ndarray,
        colors: np.ndarray,
        depth: np.ndarray,
    ) -> None:
        from OpenGL import GL

        program = self._program(renderer)
        if program is None:
            return
        # Сначала «сквозные» линии, затем с тестом глубины: два диапазона буфера.
        order = np.argsort(depth, kind="stable")
        data = np.ascontiguousarray(
            np.concatenate([positions[order], colors[order]], axis=1), np.float32
        )
        overlay = int(np.count_nonzero(~depth))
        state = context.state
        program.use(state)
        renderer._apply_camera_uniforms(program)  # noqa: SLF001 - тот же пакет
        self._ensure_buffers(len(data))
        state.bind_vertex_array(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, data.nbytes, data)
        state.set_cull(False)
        state.set_blend(False)
        state.set_depth_write(False)
        if overlay:
            state.set_depth_test(False)
            GL.glDrawArrays(GL.GL_LINES, 0, overlay)
        if overlay < len(data):
            state.set_depth_test(True)
            GL.glDrawArrays(GL.GL_LINES, overlay, len(data) - overlay)
        state.set_depth_test(True)
        state.set_depth_write(True)
        state.set_cull(True)
        renderer.invalidate_active_program()
        self.line_count = len(data) // 2
        if context.stats is not None:
            context.stats.draw_calls += int(overlay > 0) + int(overlay < len(data))

    def _draw_labels(self, context: FrameContext, renderer: Any, labels: list) -> None:
        from engine.rendering.text import TextBatch
        from engine.rendering.text.atlas import get_atlas

        try:
            atlas = get_atlas("", DEBUG_TEXT_SIZE_PX)
        except Exception as error:  # noqa: BLE001 - нет шрифта: только линии
            _log.warning("Подписи отладки: шрифт недоступен: %s", error)
            return
        if self._text_batch is None:
            self._text_batch = TextBatch()
        batch = self._text_batch
        batch.clear()
        for label in labels:
            batch.add_text(
                atlas, label.text, label.position, label.size, label.color,
                align="center", vertical_align="bottom", billboard=True,
                outline_width=1.5,
            )
        state = context.state
        state.set_depth_test(False)
        batch.draw(renderer, state, context.camera)
        state.set_depth_test(True)
        renderer.invalidate_active_program()
        self.text_count = len(labels)
        if context.stats is not None:
            context.stats.draw_calls += batch.last_draw_calls

    # ------------------------------------------------------------------ #
    # GL-ресурсы
    # ------------------------------------------------------------------ #
    def _program(self, renderer: Any) -> Any:
        try:
            return renderer.program("debug_line")
        except Exception:  # noqa: BLE001 - регистрируем при первом использовании
            program = renderer._load_program_from_files(  # noqa: SLF001
                "debug_line", "shaders/debug_line.vert", "shaders/debug_line.frag"
            )
            if program is None:
                return None
            return renderer.register_program("debug_line", program)

    def _ensure_buffers(self, vertices: int) -> None:
        from OpenGL import GL

        from engine.rendering.gl_debug import void_pointer

        if not self._vao:
            self._vao = int(GL.glGenVertexArrays(1))
            self._vbo = int(GL.glGenBuffers(1))
            GL.glBindVertexArray(self._vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
            for location, offset in ((0, 0), (1, 12)):
                GL.glEnableVertexAttribArray(location)
                GL.glVertexAttribPointer(
                    location, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, void_pointer(offset)
                )
        if vertices > self._capacity:
            self._capacity = max(vertices, self._capacity * 2, 1024)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER, self._capacity * 24, None, GL.GL_DYNAMIC_DRAW
            )

    def dispose(self) -> None:
        """Освободить буферы и отписаться от реестра."""
        self.registry.unsubscribe(self._sync_enabled)
        if self._text_batch is not None:
            self._text_batch.dispose()
            self._text_batch = None
        if self._vao:
            from OpenGL import GL

            GL.glDeleteVertexArrays(1, [self._vao])
            GL.glDeleteBuffers(1, [self._vbo])
            self._vao = self._vbo = self._capacity = 0
