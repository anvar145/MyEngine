"""Реестр импортёров: расширение файла → импортёр.

Встроенные форматы регистрируются напрямую
(:func:`engine.core.resources.importers.default_importers`), сторонние — через
entry point группы ``myengine.plugins``. Точка входа указывает на функцию
``register(registry: ImporterRegistry) -> None``::

    [project.entry-points."myengine.plugins"]
    my_formats = "my_package.plugin:register"
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from engine.core.resources.importer import AssetImporter

__all__ = ["PLUGIN_GROUP", "ImporterRegistry"]

PLUGIN_GROUP = "myengine.plugins"

_log = logging.getLogger("engine.core.resources.importers")


class ImporterRegistry:
    """Реестр импортёров с поиском по расширению и по имени."""

    __slots__ = ("_by_extension", "_by_name")

    def __init__(self, importers: Iterable[AssetImporter] = ()) -> None:
        self._by_extension: dict[str, AssetImporter] = {}
        self._by_name: dict[str, AssetImporter] = {}
        for importer in importers:
            self.register(importer)

    def register(self, importer: AssetImporter) -> AssetImporter:
        """Зарегистрировать импортёр.

        Raises:
            ValueError: Если имя или расширение уже заняты, версия некорректна
                или расширения не объявлены.
        """
        name = importer.name
        if not name:
            raise ValueError(f"{type(importer).__name__} не объявил имя")
        if name in self._by_name:
            raise ValueError(f"Импортёр '{name}' уже зарегистрирован")
        if not importer.extensions:
            raise ValueError(f"Импортёр '{name}' не объявил ни одного расширения")
        if not isinstance(importer.version, int) or importer.version < 1:
            raise ValueError(f"Импортёр '{name}': version должна быть int >= 1")
        extensions = [extension.lower() for extension in importer.extensions]
        for extension in extensions:
            if not extension.startswith("."):
                raise ValueError(f"Расширение '{extension}' должно начинаться с '.'")
            owner = self._by_extension.get(extension)
            if owner is not None:
                raise ValueError(
                    f"Расширение '{extension}' уже обрабатывает импортёр '{owner.name}'"
                )
        for extension in extensions:
            self._by_extension[extension] = importer
        self._by_name[name] = importer
        return importer

    def unregister(self, name: str) -> bool:
        """Удалить импортёр по имени. Возвращает ``True``, если он был."""
        importer = self._by_name.pop(name, None)
        if importer is None:
            return False
        for extension in importer.extensions:
            self._by_extension.pop(extension.lower(), None)
        return True

    def importer_for(self, path: Path | str) -> AssetImporter | None:
        """Импортёр для файла по расширению или ``None``."""
        suffixes = Path(path).suffixes
        if len(suffixes) >= 2:  # составные расширения, например ``.mat.json``
            compound = "".join(suffixes[-2:]).lower()
            found = self._by_extension.get(compound)
            if found is not None:
                return found
        return self._by_extension.get(Path(path).suffix.lower()) if suffixes else None

    def get(self, name: str) -> AssetImporter | None:
        """Импортёр по имени или ``None``."""
        return self._by_name.get(name)

    @property
    def extensions(self) -> tuple[str, ...]:
        """Все поддерживаемые расширения (отсортированы)."""
        return tuple(sorted(self._by_extension))

    @property
    def importers(self) -> tuple[AssetImporter, ...]:
        """Зарегистрированные импортёры в порядке регистрации."""
        return tuple(self._by_name.values())

    def load_plugins(self, group: str = PLUGIN_GROUP) -> int:
        """Подключить импортёры из entry points сторонних пакетов.

        Ошибка одного плагина логируется и не мешает остальным.

        Args:
            group: Имя группы entry points.

        Returns:
            Число успешно подключённых плагинов.
        """
        from importlib.metadata import entry_points

        loaded = 0
        for entry in entry_points(group=group):
            try:
                register = entry.load()
                register(self)
            except Exception:  # noqa: BLE001 - плагин не должен ронять движок
                _log.exception("Не удалось подключить плагин '%s'", entry.name)
                continue
            loaded += 1
        return loaded

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, Path):
            return self.importer_for(item) is not None
        if isinstance(item, str):
            return item.lower() in self._by_extension or item in self._by_name
        return False

    def __repr__(self) -> str:
        return f"<ImporterRegistry importers={len(self)} extensions={self.extensions}>"
