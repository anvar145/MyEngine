""".meta-файлы: идентичность и настройки импорта ассета.

``<asset>.meta`` лежит рядом с исходным файлом и коммитится вместе с ним::

    {"guid": "8f3c...", "importer": "obj", "version": 1,
     "settings": {"scale": 1.0, "flip_uv": false}}

``version`` — версия импортёра, с которой ассет был импортирован в последний раз.
Если версия импортёра выросла, ассет считается «грязным» и переимпортируется.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engine.core.resources.errors import InvalidGuidError, MetaFormatError
from engine.core.resources.guid import parse_guid

__all__ = [
    "META_SUFFIX",
    "AssetMeta",
    "is_meta_path",
    "meta_path_for",
    "read_meta",
    "write_meta",
]

META_SUFFIX = ".meta"


@dataclass(slots=True)
class AssetMeta:
    """Содержимое ``.meta``-файла.

    Attributes:
        guid: Стабильный идентификатор ассета.
        importer: Имя импортёра, обрабатывающего ассет.
        version: Версия импортёра при последнем успешном импорте (``0`` — ещё
            не импортировался).
        settings: Пользовательские настройки импорта (поверх умолчаний импортёра).
    """

    guid: str
    importer: str
    version: int = 0
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Сериализуемое представление с фиксированным порядком ключей."""
        return {
            "guid": self.guid,
            "importer": self.importer,
            "version": self.version,
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, data: object, *, source: str = "<meta>") -> AssetMeta:
        """Собрать мету из разобранного JSON.

        Raises:
            MetaFormatError: Если структура или типы полей неверны.
        """
        if not isinstance(data, dict):
            raise MetaFormatError(f"{source}: ожидался JSON-объект")
        try:
            guid = parse_guid(data.get("guid", ""))
        except InvalidGuidError as exc:
            raise MetaFormatError(f"{source}: {exc}") from exc
        importer = data.get("importer")
        if not isinstance(importer, str) or not importer:
            raise MetaFormatError(f"{source}: поле 'importer' должно быть строкой")
        version = data.get("version", 0)
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise MetaFormatError(f"{source}: поле 'version' должно быть int >= 0")
        settings = data.get("settings", {})
        if not isinstance(settings, dict):
            raise MetaFormatError(f"{source}: поле 'settings' должно быть объектом")
        return cls(guid, importer, version, dict(settings))


def meta_path_for(asset_path: Path | str) -> Path:
    """Путь ``.meta``-файла для исходного ассета (``hero.obj`` → ``hero.obj.meta``)."""
    path = Path(asset_path)
    return path.with_name(path.name + META_SUFFIX)


def is_meta_path(path: Path | str) -> bool:
    """``True``, если путь указывает на ``.meta``-файл."""
    return Path(path).suffix.lower() == META_SUFFIX


def read_meta(asset_path: Path | str) -> AssetMeta | None:
    """Прочитать ``.meta`` ассета.

    Args:
        asset_path: Путь к исходному файлу ассета (не к ``.meta``).

    Returns:
        Мету или ``None``, если файла ``.meta`` нет.

    Raises:
        MetaFormatError: Если файл не читается или содержит некорректные данные.
    """
    path = meta_path_for(asset_path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MetaFormatError(f"Не удалось прочитать '{path}': {exc}") from exc
    return AssetMeta.from_dict(data, source=str(path))


def write_meta(asset_path: Path | str, meta: AssetMeta) -> bool:
    """Записать ``.meta`` рядом с ассетом.

    Запись атомарна (временный файл + замена), а если содержимое не изменилось,
    файл не трогается — чтобы не создавать шум в системе контроля версий.

    Args:
        asset_path: Путь к исходному файлу ассета.
        meta: Данные для записи.

    Returns:
        ``True``, если файл был записан.
    """
    path = meta_path_for(asset_path)
    text = json.dumps(meta.to_dict(), ensure_ascii=False, indent=2) + "\n"
    try:
        if path.is_file() and path.read_text(encoding="utf-8") == text:
            return False
    except OSError:
        pass
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temp, path)
    return True
