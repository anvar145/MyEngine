"""API контрольных точек для скрипта (``self.checkpoint``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from engine.core.save.service import save_checkpoint

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.scripting.script_component import ScriptComponent

__all__ = ["ScriptCheckpoint"]


class ScriptCheckpoint:
    """``self.checkpoint``: записать контрольную точку из скрипта."""

    __slots__ = ("_component",)

    def __init__(self, component: ScriptComponent) -> None:
        self._component = component

    def save(self, slot_name: str = "checkpoint", *, idempotent: bool = False) -> bool:
        """Записать слот и объявить ``CheckpointReached``.

        Args:
            slot_name: Имя слота.
            idempotent: Пропустить запись, если состояние не менялось.

        Returns:
            ``True``, если слот записан (нет ``save_manager`` - ``False``).
        """
        scene = self._component.scene
        if scene is None:
            return False
        return save_checkpoint(scene, slot_name, idempotent=idempotent)
