"""Чтение JSON-ассетов звука (``*.cue.json``, ``*.music.json``) из VFS.

Импортёры (:mod:`engine.core.resources.importers.audio_json_importer`) кладут
исходный файл в ``Library`` как артефакт, а плеер читает его по GUID, как и
аудио-клипы: из ``Library``, ``.pak`` или рабочего каталога.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from engine.core.resources.asset_ref import AssetRef

__all__ = ["AudioAssetError", "load_json_asset", "ref_from_json"]

_log = logging.getLogger("engine.audio.asset_json")


class AudioAssetError(ValueError):
    """Файл звукового ассета не соответствует формату."""


def ref_from_json(value: Any, what: str) -> AssetRef:
    """``{"guid", "path"}``/строка/``AssetRef`` -> ``AssetRef``.

    Raises:
        AudioAssetError: Ссылка пуста или неверного вида.
    """
    ref = AssetRef.coerce(value)
    if ref is None:
        raise AudioAssetError(f"{what}: нужна ссылка на ассет")
    return ref


def load_json_asset(
    vfs: Any, ref: AssetRef | str | dict | None, artifact: str
) -> dict[str, Any] | None:
    """Прочитать JSON-артефакт ассета по ссылке.

    Args:
        vfs: ``VFS``, ``AssetDatabase`` или ``PakReader``.
        ref: Ссылка на ассет.
        artifact: Роль артефакта (``sound_cue``/``music``).

    Returns:
        Словарь или ``None`` (причина — WARNING в логе).
    """
    asset_ref = AssetRef.coerce(ref)
    if asset_ref is None or vfs is None:
        return None
    from engine.core.resources.vfs import as_assets, is_vfs

    guid = asset_ref.resolve(vfs)
    if guid is None:
        _log.warning("Звуковой ассет не найден: %s", asset_ref.path or asset_ref.guid)
        return None
    data = vfs.read(guid, artifact) if is_vfs(vfs) else None
    if data is None:
        handle = as_assets(vfs).get(guid)
        path = handle.artifact_path(artifact) if handle else None
        data = path.read_bytes() if path is not None else None
    if data is None:
        _log.warning("В ассете %s нет артефакта %s", asset_ref.path or guid, artifact)
        return None
    try:
        parsed = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        _log.warning("Ассет %s не прочитан: %s", asset_ref.path or guid, exc)
        return None
    if not isinstance(parsed, dict):
        _log.warning("Ассет %s: корень должен быть объектом", asset_ref.path or guid)
        return None
    return parsed
