"""Контракт импортёра ассетов: протокол, контекст и артефакты.

Не путать с :class:`~engine.core.resources.loader.AssetLoader`:

* **загрузчик** читает файл в память во время работы приложения;
* **импортёр** запускается офлайн (в фоновой задаче), читает исходник и кладёт
  готовые артефакты в ``Library/<xx>/<guid>/``.

Метод импорта называется ``import_asset``: ``import`` — ключевое слово Python.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from engine.core.resources.errors import AssetImportError

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.jobs import JobSystem
    from engine.core.resources.resource_manager import ResourceManager

__all__ = ["Artifact", "AssetImporter", "AssetPreview", "ImportContext"]


@dataclass(frozen=True, slots=True)
class AssetPreview:
    """Быстрое описание исходника без полного импорта (для браузера ассетов).

    Attributes:
        kind: Категория (``"mesh"``, ``"texture"``, ``"audio"``...).
        info: Сведения из заголовка файла (размеры, число вершин и т. п.).
    """

    kind: str
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Artifact:
    """Результат импорта — файл в каталоге ассета внутри ``Library/``.

    Attributes:
        kind: Роль артефакта (``"mesh"``, ``"texture"``, ``"audio_clip_bin"``...).
        name: Имя файла относительно ``Library/<xx>/<guid>/``.
        info: Метаданные для ``AssetDatabase`` (JSON-совместимые).
    """

    kind: str
    name: str
    info: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Сериализуемое представление."""
        return {"kind": self.kind, "name": self.name, "info": dict(self.info)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Artifact:
        """Восстановить артефакт из :meth:`to_dict`.

        Raises:
            KeyError: Если нет обязательных полей.
        """
        return cls(str(data["kind"]), str(data["name"]), dict(data.get("info", {})))


@dataclass(slots=True)
class ImportContext:
    """Данные, доступные импортёру во время работы.

    Attributes:
        guid: GUID импортируемого ассета.
        source: Абсолютный путь к исходному файлу.
        settings: Итоговые настройки (умолчания импортёра + ``.meta``).
        output_dir: Каталог, куда импортёр обязан писать артефакты.
        manager: Менеджер ресурсов (опционально). Импорт идёт в рабочем потоке,
            поэтому импортёр не должен менять его состояние.
        jobs: Система задач — для вложенной параллельной работы.
        log: Логгер импорта.
        dependencies: Исходные файлы, от которых зависит ассет.
        baked_inputs: Зависимости, чьё содержимое «запечено» в артефакты:
            их изменение делает ассет грязным (в отличие от простой ссылки).
    """

    guid: str
    source: Path
    settings: Mapping[str, Any]
    output_dir: Path
    manager: ResourceManager | None = None
    jobs: JobSystem | None = None
    log: logging.Logger = field(
        default_factory=lambda: logging.getLogger("engine.core.resources.import")
    )
    dependencies: list[Path] = field(default_factory=list)
    baked_inputs: list[Path] = field(default_factory=list)

    def setting(self, name: str, default: Any = None) -> Any:
        """Прочитать настройку импорта со значением по умолчанию."""
        return self.settings.get(name, default)

    def add_dependency(self, path: Path | str) -> None:
        """Записать зависимость от другого исходного файла (граф ассетов)."""
        dependency = Path(path)
        if dependency not in self.dependencies:
            self.dependencies.append(dependency)

    def add_baked_input(self, path: Path | str) -> None:
        """Зависимость, содержимое которой попало в артефакты (следим за ней)."""
        self.add_dependency(path)
        baked = Path(path)
        if baked not in self.baked_inputs:
            self.baked_inputs.append(baked)

    def artifact(self, kind: str, name: str, **info: Any) -> Artifact:
        """Создать запись об артефакте, проверив, что файл действительно записан.

        Args:
            kind: Роль артефакта.
            name: Имя файла внутри :attr:`output_dir`.
            **info: Метаданные артефакта.

        Raises:
            AssetImportError: Если файла нет или имя выходит за пределы каталога.
        """
        target = (self.output_dir / name).resolve()
        if self.output_dir.resolve() not in target.parents:
            raise AssetImportError(f"Артефакт '{name}' вне каталога импорта")
        if not target.is_file():
            raise AssetImportError(f"Артефакт '{name}' не был записан")
        return Artifact(kind, name, info)


@runtime_checkable
class AssetImporter(Protocol):
    """Протокол импортёра формата.

    Attributes:
        name: Уникальное имя импортёра (пишется в ``.meta``).
        extensions: Расширения в нижнем регистре с точкой.
        version: Версия; её рост вызывает автоматический реимпорт.
    """

    name: str
    extensions: tuple[str, ...]
    version: int

    def default_settings(self) -> dict[str, Any]:
        """Настройки импорта по умолчанию."""
        ...

    def inspect(self, path: Path) -> AssetPreview:
        """Быстро описать исходник по заголовку, не импортируя его."""
        ...

    def import_asset(self, ctx: ImportContext) -> list[Artifact]:
        """Импортировать исходник и записать артефакты в ``ctx.output_dir``.

        Выполняется в рабочем потоке: не трогать состояние главного потока.

        Raises:
            AssetImportError: Если исходник некорректен.
        """
        ...
