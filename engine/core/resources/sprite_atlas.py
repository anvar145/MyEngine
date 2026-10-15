"""Импортированный спрайт-атлас: карта имён → кадры и текстура."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from engine.core.resources.cookers.sprite_atlas_cooker import ATLAS_FORMAT, SpriteFrame
from engine.core.resources.cookers.texture_format import (
    CookedTexture,
    read_cooked_texture,
)
from engine.core.resources.errors import ResourceFormatError, ResourceNotFoundError
from engine.core.resources.vfs import as_assets

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.resources.asset_db import AssetDatabase

__all__ = ["SpriteAtlas", "load_sprite_atlas"]


@dataclass(frozen=True, slots=True)
class SpriteAtlas:
    """Атлас спрайтов.

    Attributes:
        width: Ширина текстуры атласа.
        height: Высота текстуры атласа.
        frames: Кадры по именам (порядок нарезки/упаковки сохранён).
        texture_path: Файл ``texture.bin`` (формат cooker'а текстур).
    """

    width: int
    height: int
    frames: dict[str, SpriteFrame]
    texture_path: Path

    def frame(self, name: str) -> SpriteFrame:
        """Кадр по имени.

        Raises:
            KeyError: Если такого спрайта нет.
        """
        try:
            return self.frames[name]
        except KeyError:
            raise KeyError(f"В атласе нет спрайта '{name}'") from None

    def uv(self, name: str) -> tuple[float, float, float, float]:
        """Нормализованные ``(u0, v0, u1, v1)`` спрайта; ``v`` — сверху вниз."""
        return self.frame(name).uv(self.width, self.height)

    def texture(self) -> CookedTexture:
        """Прочитать текстуру атласа (для ``GpuTexture.from_cooked``).

        Raises:
            ResourceFormatError: Если файл не читается.
        """
        try:
            return read_cooked_texture(self.texture_path.read_bytes())
        except (OSError, ValueError) as exc:
            raise ResourceFormatError(
                f"Повреждена текстура атласа '{self.texture_path}': {exc}"
            ) from exc


def load_sprite_atlas(db: AssetDatabase, guid: str) -> SpriteAtlas:
    """Прочитать импортированный атлас.

    Args:
        db: База ассетов или VFS.
        guid: GUID ассета ``*.atlas.json``.

    Raises:
        ResourceNotFoundError: Если GUID или артефактов атласа нет.
        ResourceFormatError: Если ``atlas.json`` повреждён.
    """
    handle = as_assets(db).resolve(guid)
    frames_artifact = handle.artifact("sprite_atlas")
    texture_artifact = handle.artifact("texture")
    if frames_artifact is None or texture_artifact is None:
        raise ResourceNotFoundError(f"У ассета {guid} нет артефактов атласа")
    path = handle.artifact_path(frames_artifact)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("format") != ATLAS_FORMAT:
            raise ValueError("неизвестный формат")
        width, height = (int(value) for value in document["size"])
        frames = {
            name: SpriteFrame.from_dict(name, data)
            for name, data in document["frames"].items()
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ResourceFormatError(f"Повреждён атлас '{path}': {exc}") from exc
    return SpriteAtlas(width, height, frames, handle.artifact_path(texture_artifact))
