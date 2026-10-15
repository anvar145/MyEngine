"""Готовый к использованию материал: артефакт ``material.json`` + GUID текстур.

Это данные без GL: графический слой берёт отсюда параметры и пути артефактов
и сам создаёт ``Material`` с GPU-текстурами.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engine.core.resources.errors import ResourceFormatError, ResourceNotFoundError
from engine.core.resources.vfs import as_assets

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.resources.asset_db import AssetDatabase

__all__ = ["ResolvedMaterial", "load_material", "load_material_artifact"]


@dataclass(frozen=True, slots=True)
class ResolvedMaterial:
    """Материал после разрешения ссылок на текстуры.

    Attributes:
        guid: GUID ассета материала.
        params: Скалярные параметры из ``material.json`` (``base_color``,
            ``metallic``, ``roughness``, ``alpha_mode``...).
        texture_guids: Слот (``base_color``/``normal``/``emissive``) → GUID
            текстурного ассета; ``None``, если файл не зарегистрирован в базе.
        orm_path: Путь к упакованной ORM-текстуре (``texture.bin``) или ``None``.
        missing: Слоты, чьи файлы базе неизвестны.
    """

    guid: str
    params: dict[str, Any]
    texture_guids: dict[str, str] = field(default_factory=dict)
    orm_path: Path | None = None
    missing: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        """Имя материала."""
        return str(self.params.get("name", ""))


def load_material(db: AssetDatabase, guid: str) -> ResolvedMaterial:
    """Прочитать импортированный материал и сопоставить текстуры с GUID.

    Args:
        db: База ассетов или :class:`~engine.core.resources.vfs.VFS`.
        guid: GUID ассета ``.mat.json`` (у него есть артефакт ``material``).
            Для материалов из glTF см. :func:`load_material_artifact`.

    Raises:
        ResourceNotFoundError: Если GUID или артефакта нет.
        ResourceFormatError: Если ``material.json`` повреждён.
    """
    return _load(db, guid, None)


def load_material_artifact(
    db: AssetDatabase, guid: str, artifact_name: str
) -> ResolvedMaterial:
    """То же, что :func:`load_material`, для материала из набора артефактов ассета.

    Args:
        artifact_name: Имя файла, например ``material_000.json`` (из
            ``info["material_artifact"]`` меша glTF).
    """
    return _load(db, guid, artifact_name)


def _load(
    db: AssetDatabase, guid: str, artifact_name: str | None
) -> ResolvedMaterial:
    db = as_assets(db)
    handle = db.resolve(guid)
    if artifact_name is None:
        path = handle.artifact_path("material")
    else:
        path = next(
            (
                handle.artifact_path(a)
                for a in handle.artifacts
                if a.name == artifact_name
            ),
            None,
        )
        if path is None:
            raise ResourceNotFoundError(
                f"У ассета {guid} нет артефакта '{artifact_name}'"
            )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        textures = dict(document.pop("textures", {}))
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        raise ResourceFormatError(f"Повреждён материал '{path}': {exc}") from exc

    guids: dict[str, str] = {}
    missing: list[str] = []
    orm_path = None
    for slot, ref in textures.items():
        if "artifact" in ref:
            orm_path = handle.directory / ref["artifact"]
            continue
        joined = handle.path.parent / ref["source"]
        normalize = os.path.abspath if joined.is_absolute() else os.path.normpath
        target = Path(normalize(joined))
        texture_guid = db.guid_of(target)
        if texture_guid is None:
            missing.append(slot)
        else:
            guids[slot] = texture_guid
    return ResolvedMaterial(guid, document, guids, orm_path, tuple(sorted(missing)))
