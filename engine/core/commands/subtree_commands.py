"""Команда замены поддеревьев: структурные правки, которые не выразить точечно.

Пересборка экземпляров префаба, Unpack, «Сохранить как префаб» меняют сразу
много объектов. Команда хранит записи сцены (формат v2) областей «до» и
«после»: ``do`` убирает области «до» и строит области «после», ``undo`` делает
обратное. Объекты возвращаются с теми же ``uid``; стоимость пропорциональна
размеру затронутых областей, а не всей сцены.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from engine.core.commands.object_commands import (
    _capture_subtree,
    _destroy_subtree,
    _restore_subtree,
)
from engine.core.commands.stack import Command

if TYPE_CHECKING:
    from engine.core.game_object import GameObject
    from engine.core.scene import Scene

#: Область: записи поддерева, ``uid`` родителя (``None`` - корень), позиция.
Region = tuple[list[dict[str, Any]], "str | None", int]


def capture_regions(roots: Iterable[GameObject]) -> list[Region]:
    """Снять области (поддерево каждого корня) в порядке восстановления.

    Области одного родителя упорядочены по позиции: так при построении каждая
    встаёт на своё место, а предыдущие уже на месте.
    """
    regions = [_capture_subtree(root) for root in roots]
    regions.sort(key=lambda region: (region[1] or "", region[2]))
    return regions


class ReplaceSubtreesCommand:
    """Замена набора поддеревьев другим набором.

    Области не должны вкладываться друг в друга. Корни областей «до» и «после»
    могут совпадать по ``uid`` (пересборка на месте).

    Args:
        before: Области до правки (:func:`capture_regions`).
        after: Области после правки.
        name: Название записи.
    """

    def __init__(
        self,
        before: list[Region],
        after: list[Region],
        name: str = "Изменить объекты",
    ) -> None:
        self.name = name
        self.before = before
        self.after = after

    @classmethod
    def around(
        cls,
        before_roots: Iterable[GameObject],
        name: str = "Изменить объекты",
    ) -> PendingReplace:
        """Снять область «до» заранее; ``finish(after_roots)`` соберёт команду."""
        return PendingReplace(capture_regions(before_roots), name)

    def do(self, scene: Scene) -> None:
        _swap(scene, self.before, self.after)

    def undo(self, scene: Scene) -> None:
        _swap(scene, self.after, self.before)

    def merge_with(self, other: Command) -> bool:  # noqa: ARG002
        return False


class PendingReplace:
    """Область «до», снятая перед операцией (``ReplaceSubtreesCommand.around``)."""

    def __init__(self, before: list[Region], name: str) -> None:
        self.before = before
        self.name = name

    def finish(self, after_roots: Iterable[GameObject]) -> ReplaceSubtreesCommand:
        """Снять область «после» и собрать команду."""
        return ReplaceSubtreesCommand(
            self.before, capture_regions(after_roots), self.name
        )


def _swap(scene: Scene, old: list[Region], new: list[Region]) -> None:
    for entries, _parent, _index in old:
        _destroy_subtree(scene, str(entries[0]["uid"]))
    for entries, parent_uid, index in new:
        _restore_subtree(scene, entries, parent_uid, index)
