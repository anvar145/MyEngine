"""Проход графа кадра: рисует тексты поверх геометрии.

Файл назван ``pass_.py``: ``pass`` — зарезервированное слово Python.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from engine.rendering.text.batch import TextBatch

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.rendering.render_graph import FrameContext

__all__ = ["TextPass"]


class TextPass:
    """Проход, рисующий :class:`TextBatch` (аналог ``SpritePass``).

    Args:
        batch: Пакет текстов; по умолчанию создаётся новый.

    Attributes:
        batch: Пакет текстов.
        enabled: Выключенный проход ничего не рисует.
    """

    def __init__(self, batch: TextBatch | None = None) -> None:
        self.batch = TextBatch() if batch is None else batch
        self.enabled = True

    @property
    def name(self) -> str:
        """Имя прохода."""
        return "text"

    def execute(self, context: FrameContext) -> None:
        """Нарисовать накопленные тексты текущей камерой кадра."""
        if not self.enabled or context.renderer is None:
            return
        self.batch.draw(context.renderer, context.state, context.camera)
        if context.stats is not None:
            context.stats.draw_calls += self.batch.last_draw_calls
