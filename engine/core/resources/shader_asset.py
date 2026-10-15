"""Импортированный шейдер: артефакт ``shader.glsl`` как :class:`ShaderAsset`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from engine.core.resources.asset import ShaderAsset, ShaderStage
from engine.core.resources.errors import ResourceFormatError, ResourceNotFoundError
from engine.core.resources.vfs import as_assets

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.resources.asset_db import AssetDatabase

__all__ = ["load_shader"]


def load_shader(db: AssetDatabase, guid: str) -> ShaderAsset:
    """Прочитать импортированный шейдер (без ``#include``, готовый к сборке).

    Args:
        db: База ассетов или VFS.
        guid: GUID шейдерного ассета.

    Returns:
        Ассет, который принимает ``ShaderProgram.from_assets``.

    Raises:
        ResourceNotFoundError: Если GUID или артефакта ``shader`` нет.
        ResourceFormatError: Если артефакт не читается.
    """
    handle = as_assets(db).resolve(guid)
    artifact = handle.artifact("shader")
    if artifact is None:
        raise ResourceNotFoundError(f"У ассета {guid} нет артефакта 'shader'")
    path = handle.artifact_path(artifact)
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ResourceFormatError(f"Повреждён шейдер '{path}': {exc}") from exc
    try:
        stage = ShaderStage(str(artifact.info.get("stage", "unknown")))
    except ValueError:
        stage = ShaderStage.UNKNOWN
    return ShaderAsset(handle.path, stage, source)
