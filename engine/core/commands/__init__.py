"""Undo/redo командами (headless, без Qt): ``CommandStack`` и базовые команды.

См. докстринг :mod:`engine.core.commands.stack` об инвалидации ссылок на
``GameObject`` после отката создания и удаления.
"""

from __future__ import annotations

from engine.core.commands.component_commands import (
    AddComponentCommand,
    RemoveComponentCommand,
    SetFieldCommand,
    SetScriptFieldCommand,
)
from engine.core.commands.object_commands import (
    CreateObjectCommand,
    DeleteObjectCommand,
    RenameObjectCommand,
    ReparentCommand,
    SetActiveCommand,
    SetLayerCommand,
    SetTagCommand,
    find_object,
    iter_subtree,
)
from engine.core.commands.stack import (
    DEFAULT_LIMIT,
    MERGE_WINDOW,
    Command,
    CommandStack,
    CompositeCommand,
    Transaction,
)
from engine.core.commands.subtree_commands import (
    PendingReplace,
    ReplaceSubtreesCommand,
    capture_regions,
)

__all__ = [
    "DEFAULT_LIMIT",
    "MERGE_WINDOW",
    "AddComponentCommand",
    "Command",
    "CommandStack",
    "CompositeCommand",
    "CreateObjectCommand",
    "DeleteObjectCommand",
    "PendingReplace",
    "RemoveComponentCommand",
    "RenameObjectCommand",
    "ReparentCommand",
    "ReplaceSubtreesCommand",
    "SetActiveCommand",
    "SetFieldCommand",
    "SetLayerCommand",
    "SetScriptFieldCommand",
    "SetTagCommand",
    "Transaction",
    "capture_regions",
    "find_object",
    "iter_subtree",
]
