"""Игровые сохранения: контекст, состояние, слоты, менеджер."""

from __future__ import annotations

from engine.core.save.components import (
    AutoSavePolicy,
    CheckpointComponent,
    SafeZone,
    autosave_before_switch,
)
from engine.core.save.context import GameState, SaveContext
from engine.core.save.manager import SaveManager
from engine.core.save.migrations import (
    MIGRATIONS,
    migrate_chain,
    register_migration,
)
from engine.core.save.paths import (
    default_save_root,
    game_name,
    project_save_version,
)
from engine.core.save.saveable import ISaveable
from engine.core.save.service import (
    AUTOSAVE_SLOT,
    QUICK_SLOT,
    CheckpointReached,
    SaveStatus,
    quick_load,
    quick_save,
    safe_save,
    save_checkpoint,
    state_hash,
)
from engine.core.save.slot import (
    SaveCorruptedError,
    SaveError,
    SaveScene,
    SaveSlot,
    SaveSlotMeta,
    SaveVersionError,
)

__all__ = [
    "AUTOSAVE_SLOT",
    "QUICK_SLOT",
    "AutoSavePolicy",
    "CheckpointComponent",
    "CheckpointReached",
    "SafeZone",
    "SaveStatus",
    "autosave_before_switch",
    "quick_load",
    "quick_save",
    "safe_save",
    "save_checkpoint",
    "state_hash",
    "GameState",
    "MIGRATIONS",
    "ISaveable",
    "SaveContext",
    "SaveCorruptedError",
    "SaveError",
    "SaveManager",
    "SaveScene",
    "SaveSlot",
    "SaveSlotMeta",
    "SaveVersionError",
    "default_save_root",
    "game_name",
    "project_save_version",
    "migrate_chain",
    "register_migration",
]
