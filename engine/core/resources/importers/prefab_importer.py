"""Импортёр префабов ``.prefab.json``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from engine.core.prefab_data import (
    PrefabError,
    check_prefab_graph,
    nested_instances,
    validate_prefab_data,
)
from engine.core.resources.errors import AssetImportError
from engine.core.resources.importer import Artifact, AssetPreview, ImportContext

__all__ = ["PrefabImporter"]

#: Имя артефакта в ``Library/<xx>/<guid>/``.
ARTIFACT_NAME = "prefab.json"


class PrefabImporter:
    """``.prefab.json`` -> артефакт ``prefab`` (``prefab.json``).

    Файл проверяется (формат, версия, ``root``, ссылки ``local``, иерархия,
    циклы и глубина вложенных префабов) и копируется как есть, без
    нормализации. Вложенные префабы (``PrefabInstance`` внутри объектов)
    регистрируются зависимостями, поэтому изменение вложенного файла делает
    внешний ассет грязным.
    """

    name = "prefab"
    extensions = (".prefab.json",)
    version = 1

    def default_settings(self) -> dict[str, Any]:
        """У префаба нет настроек импорта."""
        return {}

    def inspect(self, path: Path) -> AssetPreview:
        """Краткая сводка: корень и число объектов."""
        data = _read(path)
        return AssetPreview(
            "prefab",
            {"root": data["root"], "object_count": len(data["objects"])},
        )

    def import_asset(self, ctx: ImportContext) -> list[Artifact]:
        """Проверить префаб с вложенными и записать копию в ``Library``."""
        try:
            raw = ctx.source.read_bytes()
        except OSError as exc:
            raise AssetImportError(
                f"Не удалось прочитать '{ctx.source}': {exc}"
            ) from exc
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AssetImportError(
                f"Префаб '{ctx.source.name}': не UTF-8: {exc}"
            ) from exc
        data = _parse(text, ctx.source)
        base = ctx.source.parent

        def resolve(guid: str, rel_path: str) -> dict[str, Any]:
            target = _find_nested(base, rel_path)
            if target is None:
                raise PrefabError(
                    f"вложенный префаб {guid} не найден по пути {rel_path!r}"
                )
            ctx.add_dependency(target)
            return _read(target)

        try:
            check_prefab_graph(data, ctx.guid, resolve)
        except PrefabError as exc:
            raise AssetImportError(f"Префаб '{ctx.source.name}': {exc}") from exc
        # Зависимости, которые не попали в обход (на случай пустого пути),
        # уже зарегистрированы в resolve; здесь только артефакт.
        (ctx.output_dir / ARTIFACT_NAME).write_bytes(raw)
        return [
            ctx.artifact(
                "prefab",
                ARTIFACT_NAME,
                root=data["root"],
                object_count=len(data["objects"]),
                nested=sorted({guid for _, guid, _ in nested_instances(data)}),
            )
        ]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise AssetImportError(f"Не удалось прочитать '{path}': {exc}") from exc


def _parse(text: str, path: Path) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AssetImportError(
            f"Префаб '{path.name}': некорректный JSON: {exc}"
        ) from exc
    try:
        validate_prefab_data(data)
    except PrefabError as exc:
        raise AssetImportError(f"Префаб '{path.name}': {exc}") from exc
    return data


def _read(path: Path) -> dict[str, Any]:
    """Прочитать и проверить префаб; ошибки - :class:`PrefabError`."""
    try:
        return _parse(_read_text(path), path)
    except AssetImportError as exc:
        raise PrefabError(str(exc)) from exc


def _find_nested(start: Path, rel_path: str) -> Path | None:
    """Найти файл вложенного префаба: путь задан от корня проекта.

    Поднимается от каталога исходника вверх, пока не найдётся файл.
    """
    if not rel_path:
        return None
    relative = Path(rel_path)
    if relative.is_absolute():
        return relative if relative.is_file() else None
    for base in (start, *start.parents):
        candidate = base / relative
        if candidate.is_file():
            return candidate
    return None
