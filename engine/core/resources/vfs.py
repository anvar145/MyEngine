"""Виртуальная файловая система ассетов: чтение артефактов по ``(guid, artifact)``.

Игре не важно, откуда лежат импортированные данные: из ``Library/`` проекта, из
``.pak`` или из их наложения. Всё это скрыто за протоколом :class:`VFS`;
``artifact`` - короткое имя роли артефакта (``Artifact.kind``: ``"mesh"``,
``"texture"``, ``"material"``, ``"shader"``, ``"prefab"``...).

Реализации:

* :class:`LibraryVFS` - ``Library/<xx>/<guid>/`` (по ``artifacts.json``);
* :class:`PakVFS` - обёртка над :class:`~engine.core.resources.pak.PakReader`;
* :class:`OverlayVFS` - патч поверх базы (патч первым);
* :class:`WorkdirVFS` - ``Library`` плюс :class:`AssetDatabase` проекта
  (карта путь→GUID для плеера в рабочем каталоге).

Старые загрузчики (``load_pbr_material``, ``load_sprite_atlas``...) принимают
VFS и через :func:`as_assets` получают ``AssetDatabase``-совместимое лицо.

Example:
    >>> vfs = detect_vfs(project)
    >>> data = vfs.read(guid, "mesh")
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, runtime_checkable

from engine.core.resources.asset_db import RECORD_FILE, AssetDatabase, AssetHandle
from engine.core.resources.errors import ResourceNotFoundError
from engine.core.resources.guid import is_valid_guid, library_subdir
from engine.core.resources.importer import Artifact
from engine.core.resources.pak import PakReader, normalize_file_path

__all__ = [
    "VFS",
    "LibraryVFS",
    "NoBuildError",
    "OverlayVFS",
    "PakVFS",
    "VfsAssets",
    "WorkdirVFS",
    "as_assets",
    "detect_vfs",
    "is_vfs",
]


class NoBuildError(RuntimeError):
    """Нет ни собранного пака, ни ``Library/``: игре не из чего читать ассеты."""


@runtime_checkable
class VFS(Protocol):
    """Источник артефактов ассетов."""

    def read(self, guid: str, artifact: str) -> bytes | None:
        """Содержимое артефакта или ``None``, если его нет."""
        ...

    def resolve(self, guid: str, artifact: str) -> Path | None:
        """Путь к файлу артефакта на диске (``.pak`` распаковывает в кэш)."""
        ...

    def exists(self, guid: str, artifact: str) -> bool:
        """Есть ли артефакт."""
        ...

    def list_artifacts(self, guid: str) -> list[str]:
        """Роли артефактов ассета (с повторами для нескольких одной роли)."""
        ...

    def list_guids(self) -> Iterator[str]:
        """GUID всех ассетов источника."""
        ...

    def read_file(self, path: str) -> bytes | None:
        """Файл проекта (сцена, скрипт, настройки) по пути от корня проекта."""
        ...

    def list_files(self, prefix: str = "") -> list[str]:
        """Пути файлов проекта с данным префиксом (отсортированы)."""
        ...


def is_vfs(source: object) -> bool:
    """``True``, если ``source`` - VFS (а не ``AssetDatabase``/``PakReader``)."""
    return hasattr(source, "list_artifacts") and hasattr(source, "list_guids")


def _find(artifacts: tuple[Artifact, ...], kind: str) -> Artifact | None:
    return next((a for a in artifacts if a.kind == kind), None)


class _HandleVFS:
    """База: всё выражено через :meth:`handle` (описание ассета с артефактами)."""

    def handle(self, guid: str) -> AssetHandle | None:
        """Описание ассета (``AssetHandle``) или ``None``."""
        raise NotImplementedError

    def list_guids(self) -> Iterator[str]:
        raise NotImplementedError

    def resolve(self, guid: str, artifact: str) -> Path | None:
        handle = self.handle(guid)
        if handle is None:
            return None
        found = _find(handle.artifacts, artifact)
        return None if found is None else handle.directory / found.name

    def read(self, guid: str, artifact: str) -> bytes | None:
        path = self.resolve(guid, artifact)
        if path is None:
            return None
        try:
            return path.read_bytes()
        except OSError:
            return None

    def exists(self, guid: str, artifact: str) -> bool:
        path = self.resolve(guid, artifact)
        return path is not None and path.is_file()

    def list_artifacts(self, guid: str) -> list[str]:
        handle = self.handle(guid)
        return [] if handle is None else [a.kind for a in handle.artifacts]

    def read_file(self, path: str) -> bytes | None:
        """Файл проекта по пути от корня проекта (по умолчанию его нет)."""
        return None

    def list_files(self, prefix: str = "") -> list[str]:
        """Пути файлов проекта с префиксом (по умолчанию пусто)."""
        return []

    def close(self) -> None:
        """Освободить ресурсы (по умолчанию ничего)."""

    def __enter__(self):  # noqa: ANN204 - вернуть тот же тип
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class LibraryVFS(_HandleVFS):
    """Артефакты из ``Library/<xx>/<guid>/`` (описание - ``artifacts.json``).

    Args:
        root: Каталог ``Library``.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self._cache: dict[str, AssetHandle | None] = {}

    def handle(self, guid: str) -> AssetHandle | None:
        if guid in self._cache:
            return self._cache[guid]
        result = self._load(guid) if is_valid_guid(guid) else None
        self._cache[guid] = result
        return result

    def _load(self, guid: str) -> AssetHandle | None:
        directory = self.root / library_subdir(guid)
        try:
            record = json.loads((directory / RECORD_FILE).read_text("utf-8"))
            artifacts = tuple(Artifact.from_dict(a) for a in record["artifacts"])
            version = int(record["version"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return AssetHandle(
            guid=guid,
            path=Path(str(record.get("path", ""))),
            importer=str(record.get("importer", "")),
            directory=directory,
            imported_version=version,
            artifacts=artifacts,
        )

    def list_guids(self) -> Iterator[str]:
        if not self.root.is_dir():
            return
        for shard in sorted(self.root.iterdir()):
            if not shard.is_dir():
                continue
            for item in sorted(shard.iterdir()):
                if is_valid_guid(item.name) and (item / RECORD_FILE).is_file():
                    yield item.name


class PakVFS(_HandleVFS):
    """Артефакты из ``.pak`` (читает :class:`PakReader`).

    Args:
        pak: Путь к пакету или уже открытый :class:`PakReader`.
        cache_dir: Каталог распаковки для :meth:`resolve`.
        cache_root: Родитель временного каталога распаковки (см. ``PakReader``).
    """

    def __init__(
        self,
        pak: Path | str | PakReader,
        cache_dir: Path | str | None = None,
        cache_root: Path | str | Callable[[], Path | str] | None = None,
    ) -> None:
        self.reader = (
            pak
            if isinstance(pak, PakReader)
            else PakReader(pak, cache_dir, cache_root)
        )

    def handle(self, guid: str) -> AssetHandle | None:
        return self.reader.get(guid)

    def read(self, guid: str, artifact: str) -> bytes | None:
        # Без распаковки на диск: имя файла берём из индекса.
        try:
            record = self.reader._record(guid)
            for entry in record["artifacts"]:
                if entry["kind"] == artifact:
                    return self.reader.read(guid, entry["name"])
        except ResourceNotFoundError:
            pass
        return None

    def exists(self, guid: str, artifact: str) -> bool:
        try:
            record = self.reader._record(guid)
        except ResourceNotFoundError:
            return False
        return any(e["kind"] == artifact for e in record["artifacts"])

    def list_artifacts(self, guid: str) -> list[str]:
        try:
            return [e["kind"] for e in self.reader._record(guid)["artifacts"]]
        except ResourceNotFoundError:
            return []

    def list_guids(self) -> Iterator[str]:
        yield from self.reader.guids()

    def read_file(self, path: str) -> bytes | None:
        return self.reader.read_file(path)

    def list_files(self, prefix: str = "") -> list[str]:
        return self.reader.list_files(prefix)

    def close(self) -> None:
        self.reader.close()


class OverlayVFS(_HandleVFS):
    """Два источника: сначала ``patch``, при отсутствии - ``base``.

    Ассет, который есть в патче, берётся из патча целиком (артефакты двух
    источников не смешиваются).
    """

    def __init__(self, base: Any, patch: Any) -> None:
        self.base = base
        self.patch = patch

    def _owner(self, guid: str) -> Any:
        return self.patch if self.patch.list_artifacts(guid) else self.base

    def handle(self, guid: str) -> AssetHandle | None:
        return self._owner(guid).handle(guid)

    def read(self, guid: str, artifact: str) -> bytes | None:
        return self._owner(guid).read(guid, artifact)

    def resolve(self, guid: str, artifact: str) -> Path | None:
        return self._owner(guid).resolve(guid, artifact)

    def exists(self, guid: str, artifact: str) -> bool:
        return self._owner(guid).exists(guid, artifact)

    def list_artifacts(self, guid: str) -> list[str]:
        return self._owner(guid).list_artifacts(guid)

    def list_guids(self) -> Iterator[str]:
        seen: set[str] = set()
        for source in (self.patch, self.base):
            for guid in source.list_guids():
                if guid not in seen:
                    seen.add(guid)
                    yield guid

    def read_file(self, path: str) -> bytes | None:
        data = self.patch.read_file(path)
        return data if data is not None else self.base.read_file(path)

    def list_files(self, prefix: str = "") -> list[str]:
        return sorted(
            set(self.patch.list_files(prefix)) | set(self.base.list_files(prefix))
        )

    def close(self) -> None:
        for source in (self.patch, self.base):
            source.close()


class WorkdirVFS(_HandleVFS):
    """``Library`` проекта + ``AssetDatabase`` (знает исходные пути и ``.meta``).

    Нужна плееру, запущенному из рабочего каталога проекта: карта
    «путь → GUID» (``assets``) нужна ссылкам по пути и текстурам материалов.

    Args:
        assets_dir: Корень ассетов проекта.
        library: Каталог ``Library`` (по умолчанию ``<assets_dir>/../Library``).
    """

    def __init__(
        self,
        assets_dir: Path | str,
        library: Path | str | None = None,
        *,
        database: AssetDatabase | None = None,
    ) -> None:
        assets_dir = Path(assets_dir)
        if database is None:
            library = Path(library) if library else assets_dir.parent / "Library"
            database = AssetDatabase(assets_dir, library=library, read_only=True)
            database.scan()
        self.assets = database
        self.library = LibraryVFS(database.library)
        self._project_root = assets_dir.parent

    def read_file(self, path: str) -> bytes | None:
        target = self._project_root / normalize_file_path(path)
        try:
            return target.read_bytes() if target.is_file() else None
        except OSError:
            return None

    def list_files(self, prefix: str = "") -> list[str]:
        prefix = normalize_file_path(prefix) if prefix else ""
        base = self._project_root
        folder = prefix if prefix.endswith("/") else str(PurePosixPath(prefix).parent)
        start = base / folder if folder not in ("", ".") else base
        found: list[str] = []
        if not start.is_dir():
            return found
        for item in start.rglob("*"):
            if item.is_file():
                rel = item.relative_to(base).as_posix()
                if rel.startswith(prefix):
                    found.append(rel)
        return sorted(found)

    def handle(self, guid: str) -> AssetHandle | None:
        found = self.assets.get(guid)
        if found is not None and found.imported:
            return found
        return self.library.handle(guid)

    def list_guids(self) -> Iterator[str]:
        seen: set[str] = set()
        for guid in self.assets.guids:
            seen.add(guid)
            yield guid
        for guid in self.library.list_guids():
            if guid not in seen:
                yield guid


class VfsAssets:
    """``AssetDatabase``-совместимое чтение поверх VFS.

    Реализует то, что нужно ``load_*``: ``resolve``/``get``/``guid_of``/``root``.
    Карта «исходный путь → GUID» строится лениво из описаний ассетов.
    """

    def __init__(self, vfs: Any, root: Path | str | None = None) -> None:
        self.vfs = vfs
        self.root = Path(os.path.abspath(root)) if root else Path()
        self._by_path: dict[str, str] | None = None

    def get(self, guid: str) -> AssetHandle | None:
        """Описание ассета или ``None``."""
        return self.vfs.handle(guid)

    def resolve(self, guid: str) -> AssetHandle:
        """Описание ассета.

        Raises:
            ResourceNotFoundError: Если ассета нет.
        """
        handle = self.vfs.handle(guid)
        if handle is None:
            raise ResourceNotFoundError(f"В источнике нет ассета {guid}")
        return handle

    @property
    def guids(self) -> tuple[str, ...]:
        """GUID всех ассетов."""
        return tuple(self.vfs.list_guids())

    def guid_of(self, path: Path | str) -> str | None:
        """GUID по исходному пути (относительному корню ассетов)."""
        if self._by_path is None:
            table: dict[str, str] = {}
            for guid in self.vfs.list_guids():
                handle = self.vfs.handle(guid)
                if handle is not None and str(handle.path):
                    table[PurePosixPath(handle.path.as_posix()).as_posix()] = guid
            self._by_path = table
        path = Path(path)
        if path.is_absolute():
            try:
                path = Path(os.path.abspath(path)).relative_to(self.root)
            except ValueError:
                return None
        key = Path(os.path.normpath(path)).as_posix()
        return self._by_path.get(key)


def as_assets(source: Any, root: Path | str | None = None) -> Any:
    """Привести ``source`` к ``AssetDatabase``-совместимому виду.

    ``AssetDatabase`` и ``PakReader`` возвращаются как есть; ``WorkdirVFS`` даёт
    свою базу; остальные VFS оборачиваются в :class:`VfsAssets`.
    """
    if not is_vfs(source):
        return source
    database = getattr(source, "assets", None)
    if database is not None:
        return database
    return VfsAssets(source, root)


def detect_vfs(
    project: Path | str,
    *,
    pak: Path | str | None = None,
    use_library: bool = False,
    assets_dir: Path | str | None = None,
    library: Path | str | None = None,
    pak_cache_root: Path | str | Callable[[], Path | str] | None = None,
) -> _HandleVFS:
    """Выбрать источник ассетов для плеера.

    Порядок: ``pak`` задан явно → :class:`PakVFS`; ``use_library`` → библиотека;
    иначе ``<проект>/Build/<имя>.pak`` → :class:`PakVFS`; иначе ``Library``
    → библиотека (:class:`WorkdirVFS`, если есть каталог ассетов).

    Args:
        project: Корень проекта.
        pak: Явный путь к ``.pak``.
        use_library: Игнорировать собранный пак.
        assets_dir: Каталог ассетов (по умолчанию ``Assets``/``assets``).
        library: Каталог ``Library`` (по умолчанию ``<проект>/Library``).
        pak_cache_root: Где распаковывать артефакты пака (по умолчанию temp).

    Raises:
        NoBuildError: Нет ни пака, ни ``Library/``.
        FileNotFoundError: Явно заданный пак не найден.
    """
    project = Path(project)
    if pak is not None:
        pak = Path(pak)
        if not pak.is_file():
            raise FileNotFoundError(f"Пак не найден: '{pak}'")
        return PakVFS(pak, cache_root=pak_cache_root)
    library = Path(library) if library else project / "Library"
    if not use_library:
        candidate = project / "Build" / f"{project.name}.pak"
        if candidate.is_file():
            return PakVFS(candidate, cache_root=pak_cache_root)
    if library.is_dir():
        if assets_dir is None:
            assets_dir = project / "Assets"
            if not assets_dir.is_dir():
                assets_dir = project / "assets"
        if Path(assets_dir).is_dir():
            return WorkdirVFS(assets_dir, library)
        return LibraryVFS(library)
    raise NoBuildError(
        f"В '{project}' нет собранного пака (Build/{project.name}.pak) "
        "и нет Library/: нечего загружать"
    )
