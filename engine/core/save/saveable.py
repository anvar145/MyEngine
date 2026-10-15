"""Контракт компонента, сохраняющего собственные данные."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from engine.core.save.context import SaveContext

__all__ = ["ISaveable"]


@runtime_checkable
class ISaveable(Protocol):
    """Компонент (или ``Behaviour``) с произвольными данными в сохранении.

    ``on_save`` кладёт данные в ``ctx`` (свой контекст компонента на объекте),
    ``on_load`` читает их после загрузки; поля с ``save=True`` применяются
    автоматически до ``on_load``.
    """

    def on_save(self, ctx: SaveContext) -> None:
        """Записать данные в ``ctx``."""

    def on_load(self, ctx: SaveContext) -> None:
        """Восстановить состояние из ``ctx``."""
