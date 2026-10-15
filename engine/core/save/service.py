"""Сервис сохранений поверх :class:`SaveManager`: слоты-«соглашения» (A7b).

Слоты с подчёркиванием в начале - служебные: ``_autosave`` (один, перезаписывается),
``_quick`` (F5/F9). Любая запись идёт через :func:`safe_save`: сбой не роняет игру,
повторный вход (запись внутри записи) отбрасывается.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from engine.core.save.objects import collect_objects
from engine.core.save.slot import SaveError, SaveSlot

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.scene import Scene

__all__ = [
    "AUTOSAVE_SLOT",
    "QUICK_SLOT",
    "CheckpointReached",
    "SaveStatus",
    "quick_load",
    "quick_save",
    "safe_save",
    "save_checkpoint",
    "state_hash",
]

_log = logging.getLogger("engine.core.save")

AUTOSAVE_SLOT = "_autosave"
QUICK_SLOT = "_quick"


@dataclass(frozen=True, slots=True)
class CheckpointReached:
    """Контрольная точка записана (``Scene.events``)."""

    slot_name: str


@dataclass(frozen=True, slots=True)
class SaveStatus:
    """Короткое сообщение для HUD («Сохранено», «Загружено», ошибка)."""

    text: str


def state_hash(scene: Scene) -> str:
    """Хеш сохраняемого состояния: ``GameState`` + данные объектов сцены."""
    payload: Any = {"state": scene.state.to_dict(), "objects": collect_objects(scene)}
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_save(
    scene: Scene, slot_name: str, *, playtime_sec: float = 0.0
) -> SaveSlot | None:
    """Записать слот, не бросая исключений.

    Returns:
        Слот или ``None``: нет ``save_manager``, запись уже идёт или произошла
        ошибка (она пишется в журнал).
    """
    manager = scene.save_manager
    if manager is None or manager.busy:
        return None
    manager.busy = True
    try:
        return scene.save(slot_name, playtime_sec=playtime_sec)
    except (OSError, ValueError, SaveError) as error:
        _log.error("Не удалось записать слот %r: %s", slot_name, error)
        return None
    finally:
        manager.busy = False


def save_checkpoint(
    scene: Scene, slot_name: str, *, idempotent: bool = True
) -> bool:
    """Записать контрольную точку и объявить ``CheckpointReached``.

    Args:
        scene: Сцена.
        slot_name: Имя слота.
        idempotent: Не писать повторно, если состояние с прошлой записи этого
            слота не менялось (хеш :func:`state_hash`, слот ещё на диске).

    Returns:
        ``True``, если слот записан.
    """
    manager = scene.save_manager
    if manager is None:
        return False
    if slot_name == AUTOSAVE_SLOT and not manager.autosave_enabled:
        return False
    digest = state_hash(scene) if idempotent else ""
    if (
        idempotent
        and manager.hashes.get(slot_name) == digest
        and manager.slot(slot_name) is not None
    ):
        return False
    if safe_save(scene, slot_name) is None:
        return False
    manager.hashes[slot_name] = digest or state_hash(scene)
    scene.events.emit(CheckpointReached(slot_name))
    return True


def quick_save(scene: Scene) -> SaveSlot | None:
    """Быстрое сохранение в ``_quick`` (F5); статус - событием ``SaveStatus``."""
    slot = safe_save(scene, QUICK_SLOT)
    scene.events.emit(SaveStatus("Сохранено" if slot else "Ошибка сохранения"))
    return slot


def quick_load(scene: Scene) -> SaveSlot | None:
    """Загрузка ``_quick`` (F9), а если его нет - ``_autosave``.

    Returns:
        Загруженный слот или ``None`` (нет слотов либо ошибка).
    """
    manager = scene.save_manager
    if manager is None or manager.busy:
        return None
    name = next((n for n in (QUICK_SLOT, AUTOSAVE_SLOT) if manager.slot(n)), None)
    if name is None:
        scene.events.emit(SaveStatus("Нет сохранения"))
        return None
    manager.busy = True
    try:
        slot = scene.load(name)
    except (OSError, ValueError, SaveError) as error:
        _log.error("Не удалось загрузить слот %r: %s", name, error)
        scene.events.emit(SaveStatus("Ошибка загрузки"))
        return None
    finally:
        manager.busy = False
    scene.events.emit(SaveStatus("Загружено"))
    return slot
