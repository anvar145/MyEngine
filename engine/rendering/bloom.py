"""Bloom: bright-pass, каскад размытия и аддитивный подъём (A10b).

Каскад работает в линейных значениях ``RGBA16F`` начиная с половинного
разрешения HDR-цели: bright-pass → на каждом уровне два прохода размытия
(горизонтальный из предыдущего уровня, вертикальный) → аддитивное
«всплытие» от меньшего уровня к большему. Итог (уровень 0) складывается с
кадром в fused-проходе tonemap, то есть до FXAA и до сжатия яркости.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from engine.rendering.render_graph import RenderTarget

if TYPE_CHECKING:
    from engine.rendering.renderer import Renderer

__all__ = ["BLOOM_PROGRAMS", "MAX_BLOOM_LEVELS", "BloomChain"]

#: Имена шейдерных программ, нужных каскаду.
BLOOM_PROGRAMS = ("bloom_bright", "blur", "bloom_up")

MAX_BLOOM_LEVELS = 5


class BloomChain:
    """Цели каскада и проходы bloom (цели кэшируются и меняют размер на месте).

    Args:
        renderer: Рендерер (программы, кэш состояния GL).
    """

    def __init__(self, renderer: Renderer) -> None:
        self._renderer = renderer
        #: Пары ``(уровень, временная цель)`` от половинного разрешения вниз.
        self._targets: list[tuple[RenderTarget, RenderTarget]] = []

    @property
    def level_count(self) -> int:
        """Сколько уровней сейчас выделено."""
        return len(self._targets)

    def dispose(self) -> None:
        """Освободить все цели каскада."""
        for level, temp in self._targets:
            level.dispose()
            temp.dispose()
        self._targets.clear()

    def available(self) -> bool:
        """Все шейдеры каскада найдены."""
        from engine.rendering.errors import RenderError

        try:
            for name in BLOOM_PROGRAMS:
                self._renderer.program(name)
        except RenderError:
            return False
        return True

    # ------------------------------------------------------------------ #
    def _ensure_targets(self, width: int, height: int, levels: int) -> None:
        """Подготовить ``levels`` уровней: 1/2, 1/4, 1/8, ... от ``width x height``."""
        while len(self._targets) > levels:
            level, temp = self._targets.pop()
            level.dispose()
            temp.dispose()
        for index in range(levels):
            size_w = max(1, (width // 2) >> index)
            size_h = max(1, (height // 2) >> index)
            if index == len(self._targets):
                pair = tuple(
                    RenderTarget(name, size_w, size_h, depth=False, hdr=True)
                    for name in (f"bloom{index}", f"bloom{index}_tmp")
                )
                self._targets.append(pair)  # type: ignore[arg-type]
            for target in self._targets[index]:
                target.resize(size_w, size_h)
                target.create()

    def run(
        self,
        hdr: RenderTarget,
        *,
        levels: int,
        threshold: float,
        soft_knee: float,
    ) -> RenderTarget | None:
        """Построить bloom для ``hdr``.

        Вызывается внутри пост-пути: фиксированный полноэкранный VAO уже
        привязан, тест глубины, отсечение и смешивание выключены.

        Args:
            hdr: Разрешённая HDR-цель кадра.
            levels: Число уровней (1..:data:`MAX_BLOOM_LEVELS`).
            threshold: Порог яркости (линейные единицы).
            soft_knee: Ширина плавного перехода порога.

        Returns:
            Цель с итоговым свечением в линейных единицах или ``None``, если
            шейдеров нет.
        """
        if not self.available():
            return None
        from OpenGL import GL

        levels = max(1, min(MAX_BLOOM_LEVELS, int(levels)))
        self._ensure_targets(hdr.width, hdr.height, levels)
        renderer = self._renderer
        state = renderer.state

        first = self._targets[0][0]
        first.bind(state)
        program = renderer.program("bloom_bright")
        program.use(state)
        state.bind_texture(0, GL.GL_TEXTURE_2D, hdr.color_textures[0])
        program.set_uniform("u_color", 0)
        program.set_uniform("u_texel", (1.0 / hdr.width, 1.0 / hdr.height))
        program.set_uniform("u_threshold", float(threshold))
        program.set_uniform("u_soft_knee", float(soft_knee))
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)

        blur = renderer.program("blur")
        blur.use(state)
        blur.set_uniform("u_color", 0)
        for index, (level, temp) in enumerate(self._targets):
            source = level if index == 0 else self._targets[index - 1][0]
            self._blur_pass(blur, source, temp, horizontal=True)
            self._blur_pass(blur, temp, level, horizontal=False)

        if levels > 1:
            up = renderer.program("bloom_up")
            up.use(state)
            up.set_uniform("u_color", 0)
            state.set_blend(True, (GL.GL_ONE, GL.GL_ONE))
            for index in range(levels - 1, 0, -1):
                self._targets[index - 1][0].bind(state)
                state.bind_texture(
                    0, GL.GL_TEXTURE_2D, self._targets[index][0].color_textures[0]
                )
                GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
            state.set_blend(False)
        return first

    def _blur_pass(
        self,
        program: object,
        source: RenderTarget,
        destination: RenderTarget,
        *,
        horizontal: bool,
    ) -> None:
        from OpenGL import GL

        state = self._renderer.state
        destination.bind(state)
        state.bind_texture(0, GL.GL_TEXTURE_2D, source.color_textures[0])
        step = (
            (1.0 / destination.width, 0.0)
            if horizontal
            else (0.0, 1.0 / destination.height)
        )
        program.set_uniform("u_direction", step)  # type: ignore[attr-defined]
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
