"""Проход графа кадра: UI поверх всего (последний проход).

Файл назван ``pass_.py``: ``pass`` — зарезервированное слово Python.

Порядок внутри прохода: квады (панели, кнопки, картинки одним батчем), затем
текст через :meth:`TextBatch.draw` в экранном режиме. Второго текстового прохода
нет: текст UI рисует тот же ``text``-шейдер. Чередования «квад — текст — квад»
внутри одного кадра UI нет (текст всегда выше квадов; допущение MVP).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from engine.rendering.text.batch import TextBatch
from engine.rendering.ui.batch import UIBatch
from engine.rendering.ui.screen import UIScreen

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.rendering.render_graph import FrameContext

__all__ = ["UIPass"]


class UIPass:
    """Рисует UI-квады и UI-текст в экранных координатах.

    Args:
        screen: Экран UI; по умолчанию новый (1280×720, ``fixed``).
        batch: Пакет квадов.
        text_batch: Пакет текстов (координаты — единицы UI, Y вверх).

    Attributes:
        enabled: Выключенный проход ничего не рисует.
        after_post: UI рисуется после пост-обработки, в итоговую цель.
    """

    after_post = True

    def __init__(
        self,
        screen: UIScreen | None = None,
        batch: UIBatch | None = None,
        text_batch: TextBatch | None = None,
    ) -> None:
        self.screen = UIScreen() if screen is None else screen
        self.batch = UIBatch() if batch is None else batch
        self.text_batch = TextBatch() if text_batch is None else text_batch
        self.enabled = True

    @property
    def name(self) -> str:
        """Имя прохода."""
        return "ui"

    def execute(self, context: FrameContext) -> None:
        """Нарисовать UI в текущую цель кадра."""
        if not self.enabled or context.renderer is None:
            return
        if not len(self.batch) and not len(self.text_batch):
            return
        state = context.state
        renderer = context.renderer
        self.screen.set_viewport(*context.viewport)
        state.set_depth_test(False)
        state.set_depth_write(False)
        state.set_cull(False)
        state.set_blend(True)
        calls = 0
        if len(self.batch):
            self.batch.draw(renderer, state, self.screen)
            calls += self.batch.last_draw_calls
        if len(self.text_batch):
            self.text_batch.draw(
                renderer, state, None, projection=self.screen.projection()
            )
            calls += self.text_batch.last_draw_calls
        state.set_blend(False)
        state.set_depth_write(True)
        state.set_depth_test(True)
        state.set_cull(True)
        if context.stats is not None:
            context.stats.draw_calls += calls
