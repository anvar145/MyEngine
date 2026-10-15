"""База ассетов: GUID → ассет, граф зависимостей и инкрементальный импорт.

Модель потоков:

* ``AssetDatabase`` принадлежит **главному потоку**: весь её изменяемый стейт
  правится только там;
* в рабочем потоке (``JobSystem``) выполняется лишь функция импортёра, которая
  пишет во временный каталог и не трогает базу;
* результат применяется в главном потоке при :meth:`AssetDatabase.pump` —
  каталог подменяется атомарно, обновляются граф и ``.meta``, публикуется
  ``AssetImported``.

:meth:`AssetDatabase.resolve` — чтение из ``dict`` за O(1) и безопасно для
вызова каждый кадр.

Example:
    >>> db = AssetDatabase(Path("MyGame/Assets"))
    >>> db.scan()
    >>> db.reimport_dirty(wait=True)
    >>> handle = db.resolve(guid)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engine.core.event_bus import Event, EventBus
from engine.core.jobs import Job, JobSystem
from engine.core.resources.errors import (
    AssetImportError,
    ImporterNotFoundError,
    ResourceError,
    ResourceNotFoundError,
)
from engine.core.resources.guid import library_subdir, new_guid, parse_guid
from engine.core.resources.importer import Artifact, AssetImporter, ImportContext
from engine.core.resources.importer_registry import ImporterRegistry
from engine.core.resources.importers import default_importers
from engine.core.resources.meta import (
    AssetMeta,
    is_meta_path,
    read_meta,
    write_meta,
)

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.resources.resource_manager import ResourceManager

__all__ = [
    "RECORD_FILE",
    "AssetDatabase",
    "AssetHandle",
    "AssetImportFailed",
    "AssetImported",
]

_log = logging.getLogger("engine.core.resources.asset_db")

#: Индекс импорта внутри ``Library/<xx>/<guid>/``.
RECORD_FILE = "artifacts.json"
_SKIPPED_DIRS = {"__pycache__", "Library", "_work"}


@dataclass(slots=True)
class AssetImported(Event):
    """Событие: ассет (пере)импортирован, артефакты в ``Library`` актуальны."""

    guid: str
    path: Path


@dataclass(slots=True)
class AssetImportFailed(Event):
    """Событие: импорт ассета завершился ошибкой."""

    guid: str
    path: Path
    error: str


@dataclass(frozen=True, slots=True)
class AssetHandle:
    """Неизменяемое описание ассета в базе.

    Attributes:
        guid: Идентификатор ассета.
        path: Абсолютный путь к исходному файлу.
        importer: Имя импортёра.
        directory: Каталог артефактов в ``Library``.
        imported_version: Версия импортёра при последнем импорте (``0`` — нет).
        artifacts: Артефакты последнего успешного импорта.
        error: Текст ошибки последней неудачной попытки импорта.
    """

    guid: str
    path: Path
    importer: str
    directory: Path
    imported_version: int = 0
    artifacts: tuple[Artifact, ...] = ()
    error: str | None = None

    @property
    def imported(self) -> bool:
        """``True``, если артефакты есть в ``Library``."""
        return self.imported_version > 0 and bool(self.artifacts)

    def artifact(self, kind: str) -> Artifact | None:
        """Первый артефакт указанной роли или ``None``."""
        for artifact in self.artifacts:
            if artifact.kind == kind:
                return artifact
        return None

    def artifact_path(self, artifact: Artifact | str) -> Path:
        """Абсолютный путь артефакта в ``Library`` (по объекту или по роли).

        Raises:
            ResourceNotFoundError: Если артефакта с такой ролью нет.
        """
        if isinstance(artifact, str):
            found = self.artifact(artifact)
            if found is None:
                raise ResourceNotFoundError(
                    f"У ассета {self.guid} нет артефакта '{artifact}'"
                )
            artifact = found
        return self.directory / artifact.name


@dataclass(slots=True)
class _ImportResult:
    """Итог работы импортёра в рабочем потоке (применяется в главном)."""

    temp_dir: Path
    artifacts: list[Artifact]
    dependencies: list[Path]
    importer_version: int
    settings_hash: str
    source: dict[str, Any]
    baked: dict[str, dict[str, Any]]


@dataclass(slots=True)
class _Pending:
    """Запущенный импорт ассета."""

    job: Job
    temp_dir: Path


@dataclass(slots=True)
class _Graph:
    """Граф зависимостей: ассет → от чего зависит и кто зависит от него."""

    deps: dict[str, tuple[str, ...]] = field(default_factory=dict)
    dependents: dict[str, set[str]] = field(default_factory=dict)

    def set(self, guid: str, deps: Iterable[str]) -> None:
        for old in self.deps.get(guid, ()):
            users = self.dependents.get(old)
            if users is not None:
                users.discard(guid)
        unique = tuple(dict.fromkeys(dep for dep in deps if dep != guid))
        self.deps[guid] = unique
        for dep in unique:
            self.dependents.setdefault(dep, set()).add(guid)


class AssetDatabase:
    """Реестр ассетов проекта: GUID, ``.meta``, ``Library`` и зависимости.

    Args:
        root: Каталог исходных ассетов (``Assets``).
        library: Каталог кэша артефактов; по умолчанию ``<root>/../Library``.
        importers: Реестр импортёров; по умолчанию — встроенные форматы.
        jobs: Система задач; по умолчанию создаётся своя и закрывается в
            :meth:`close`.
        event_bus: Шина для ``AssetImported``/``AssetImportFailed``.
        manager: Менеджер ресурсов; загруженный им ассет перечитывается после
            реимпорта.
        read_only: Не писать ``.meta`` рядом с исходниками: GUID живут только в
            памяти, настройки импорта не сохраняются (встроенные ассеты движка).
    """

    def __init__(
        self,
        root: Path | str,
        *,
        library: Path | str | None = None,
        importers: ImporterRegistry | None = None,
        jobs: JobSystem | None = None,
        event_bus: EventBus | None = None,
        manager: ResourceManager | None = None,
        read_only: bool = False,
    ) -> None:
        self.read_only = read_only
        self.root = Path(os.path.abspath(root))
        self.library = Path(
            os.path.abspath(library if library else self.root.parent / "Library")
        )
        self.importers = (
            importers
            if importers is not None
            else ImporterRegistry(default_importers())
        )
        self._owns_jobs = jobs is None
        self.jobs = jobs if jobs is not None else JobSystem(workers=2, name="AssetDb")
        self.events = event_bus
        self.manager = manager

        self._handles: dict[str, AssetHandle] = {}
        self._by_path: dict[str, str] = {}
        self._metas: dict[str, AssetMeta] = {}
        self._unwritable_logged: set[Path] = set()
        self._records: dict[str, dict[str, Any]] = {}
        self._graph = _Graph()
        self._pending: dict[str, _Pending] = {}

    # ------------------------------------------------------------------ #
    # Регистрация и поиск
    # ------------------------------------------------------------------ #
    def register(self, path: Path | str) -> str:
        """Зарегистрировать исходный файл, создав ``.meta`` при необходимости.

        Повторный вызов для того же файла возвращает тот же GUID. Если GUID из
        ``.meta`` уже занят другим файлом (скопировали ассет вместе с ``.meta``),
        новому файлу выдаётся свежий GUID.

        Args:
            path: Путь к исходному файлу (абсолютный или от :attr:`root`).

        Returns:
            GUID ассета.

        Raises:
            ResourceNotFoundError: Файла нет.
            ImporterNotFoundError: Нет импортёра для расширения.
            MetaFormatError: ``.meta`` повреждён (файл не перезаписывается).
        """
        source = self._absolute(path)
        if not source.is_file():
            raise ResourceNotFoundError(f"Файл ассета не найден: '{source}'")
        existing = self._by_path.get(_norm(source))
        if existing is not None:
            return existing
        importer = self.importers.importer_for(source)
        if importer is None:
            raise ImporterNotFoundError(
                f"Нет импортёра для '{source.suffix}' (файл '{source.name}'). "
                f"Доступно: {', '.join(self.importers.extensions) or 'нет'}"
            )

        meta = read_meta(source)
        if meta is not None and meta.guid in self._handles:
            _log.warning(
                "GUID %s из '%s.meta' уже занят '%s': выдан новый GUID",
                meta.guid,
                source.name,
                self._handles[meta.guid].path,
            )
            meta = None
        if meta is None:
            meta = AssetMeta(new_guid(), importer.name)
        elif meta.importer != importer.name:
            meta.importer = importer.name
        self._store_meta(source, meta)

        guid = meta.guid
        directory = self.library / library_subdir(guid)
        self._metas[guid] = meta
        self._by_path[_norm(source)] = guid
        self._handles[guid] = AssetHandle(guid, source, importer.name, directory)
        self._load_record(guid)
        return guid

    def _store_meta(self, source: Path, meta: AssetMeta) -> None:
        """Записать ``.meta``; если каталог недоступен для записи — только лог.

        Ассеты в защищённом каталоге (например, встроенные ``assets/`` в
        установленной копии движка) остаются работоспособными: мета живёт в
        памяти, а GUID при следующем запуске будет выдан заново.
        """
        if self.read_only:
            return
        try:
            write_meta(source, meta)
        except OSError as exc:
            if source.parent not in self._unwritable_logged:
                self._unwritable_logged.add(source.parent)
                _log.warning(
                    "Не удалось записать .meta в '%s': %s. Каталог доступен "
                    "только для чтения, GUID ассетов в нём не сохранятся.",
                    source.parent,
                    exc,
                )

    def scan(self) -> list[str]:
        """Зарегистрировать все поддерживаемые файлы под :attr:`root`.

        Битые ``.meta`` и файлы без импортёра пропускаются с записью в лог.

        Returns:
            GUID найденных ассетов в порядке путей.
        """
        found: list[str] = []
        for path in self.iter_source_files():
            try:
                found.append(self.register(path))
            except ResourceError as exc:
                _log.error("Ассет '%s' пропущен: %s", path, exc)
        return found

    def iter_source_files(self) -> Iterable[Path]:
        """Исходные файлы под :attr:`root`, для которых есть импортёр.

        Служебные каталоги (``Library``, ``_work``, скрытые) и ``.meta`` пропускаются.
        """
        for directory, dirs, files in os.walk(self.root):
            current = Path(directory)
            dirs[:] = sorted(
                name
                for name in dirs
                if not name.startswith(".")
                and name not in _SKIPPED_DIRS
                and not _same_path(current / name, self.library)
            )
            for name in sorted(files):
                path = current / name
                if not is_meta_path(path) and self.importers.importer_for(path):
                    yield path

    def unregister(self, guid: str) -> None:
        """Снять ассет с учёта; файлы, ``.meta`` и ``Library`` не удаляются.

        Нужна, когда исходник удалён: GUID остаётся занятым в ``.meta`` на случай
        возврата файла. Идущий импорт ассета завершится и будет проигнорирован.

        Raises:
            ResourceNotFoundError: GUID неизвестен базе.
        """
        handle = self.resolve(guid)
        self._by_path.pop(_norm(handle.path), None)
        self._handles.pop(guid, None)
        self._metas.pop(guid, None)
        self._records.pop(guid, None)
        self._graph.set(guid, ())
        self._graph.deps.pop(guid, None)
        self._graph.dependents.pop(guid, None)
        pending = self._pending.pop(guid, None)
        if pending is not None:
            shutil.rmtree(pending.temp_dir, ignore_errors=True)

    def invalidate(self, guid: str) -> None:
        """Забыть результат импорта и удалить артефакты ассета из ``Library``.

        Ассет остаётся зарегистрированным, но станет «грязным» (нужен реимпорт).
        Не вызывайте, пока для ассета идёт импорт.

        Raises:
            ResourceNotFoundError: GUID неизвестен базе.
        """
        handle = self.resolve(guid)
        shutil.rmtree(handle.directory, ignore_errors=True)
        self._records.pop(guid, None)
        self._handles[guid] = _replace(
            handle, imported_version=0, artifacts=(), error=None
        )

    def move(self, guid: str, new_path: Path | str) -> None:
        """Сообщить базе, что исходник ассета переехал (GUID остаётся прежним).

        ``.meta`` должен лежать рядом с новым путём (он переезжает вместе с файлом).

        Raises:
            ResourceNotFoundError: GUID неизвестен или нового файла нет.
        """
        handle = self.resolve(guid)
        target = self._absolute(new_path)
        if not target.is_file():
            raise ResourceNotFoundError(f"Файл ассета не найден: '{target}'")
        self._by_path.pop(_norm(handle.path), None)
        self._by_path[_norm(target)] = guid
        self._handles[guid] = _replace(handle, path=target)

    def resolve(self, guid: str) -> AssetHandle:
        """Вернуть ассет по GUID (O(1), безопасно вызывать каждый кадр).

        Raises:
            ResourceNotFoundError: GUID неизвестен базе.
        """
        handle = self._handles.get(guid)
        if handle is None:
            raise ResourceNotFoundError(f"Ассет с GUID {guid!r} не зарегистрирован")
        return handle

    def get(self, guid: str) -> AssetHandle | None:
        """Вернуть ассет по GUID или ``None``."""
        return self._handles.get(guid)

    def guid_of(self, path: Path | str) -> str | None:
        """GUID зарегистрированного файла или ``None``."""
        return self._by_path.get(_norm(self._absolute(path)))

    @property
    def guids(self) -> tuple[str, ...]:
        """GUID всех зарегистрированных ассетов."""
        return tuple(self._handles)

    def settings_of(self, guid: str) -> dict[str, Any]:
        """Итоговые настройки импорта: умолчания импортёра поверх ``.meta``."""
        handle = self.resolve(guid)
        importer = self._importer_of(handle)
        return {**importer.default_settings(), **self._metas[guid].settings}

    def set_settings(self, guid: str, settings: dict[str, Any]) -> None:
        """Заменить пользовательские настройки импорта и записать ``.meta``.

        Ассет станет «грязным»; для применения вызовите :meth:`reimport`.
        """
        handle = self.resolve(guid)
        meta = self._metas[guid]
        meta.settings = dict(settings)
        self._store_meta(handle.path, meta)

    # ------------------------------------------------------------------ #
    # Граф зависимостей
    # ------------------------------------------------------------------ #
    def dependencies(self, guid: str) -> list[str]:
        """GUID ассетов, от которых зависит данный (прямые зависимости)."""
        self.resolve(guid)
        return list(self._graph.deps.get(guid, ()))

    def dependents(self, guid: str) -> list[str]:
        """GUID ассетов, зависящих от данного (прямые)."""
        self.resolve(guid)
        return sorted(self._graph.dependents.get(guid, ()))

    def dependency_closure(self, guid: str) -> list[str]:
        """Все транзитивные зависимости (для состава пака), без самого ассета."""
        self.resolve(guid)
        seen: dict[str, None] = {}
        stack = list(self._graph.deps.get(guid, ()))
        while stack:
            current = stack.pop()
            if current in seen or current == guid:
                continue
            seen[current] = None
            stack.extend(self._graph.deps.get(current, ()))
        return list(seen)

    # ------------------------------------------------------------------ #
    # Импорт
    # ------------------------------------------------------------------ #
    def is_dirty(self, guid: str) -> bool:
        """``True``, если ассет нужно (пере)импортировать.

        Причины: артефактов нет, выросла версия импортёра, изменились настройки
        или исходный файл. Обращается к диску — вызывать не каждый кадр.
        """
        handle = self.resolve(guid)
        record = self._records.get(guid)
        if record is None or not handle.imported:
            return True
        importer = self._importer_of(handle)
        if record.get("version") != importer.version:
            return True
        if record.get("settings_hash") != _hash_json(self.settings_of(guid)):
            return True
        if self._source_changed(handle.path, record.get("source", {})):
            return True
        return any(
            self._source_changed(Path(path), stamp)
            for path, stamp in record.get("baked", {}).items()
        )

    def dirty(self) -> list[str]:
        """GUID всех ассетов, требующих импорта."""
        return [guid for guid in self._handles if self.is_dirty(guid)]

    def reimport(self, guid: str, *, wait: bool = False) -> Job:
        """Запустить (пере)импорт ассета в фоновой задаче.

        Если импорт этого ассета уже идёт, возвращается его задача. Результат
        применяется в главном потоке при :meth:`pump`/``wait=True``.

        Args:
            guid: GUID ассета.
            wait: Дождаться завершения и применить результат.

        Returns:
            Задачу импорта.
        """
        handle = self.resolve(guid)
        pending = self._pending.get(guid)
        if pending is not None:
            if wait:
                self.jobs.wait_idle()
            return pending.job

        importer = self._importer_of(handle)
        settings = self.settings_of(guid)
        temp_dir = handle.directory.with_name(guid + ".tmp")
        job = self.jobs.submit(
            self._run_import,
            guid,
            importer,
            handle.path,
            settings,
            temp_dir,
            on_done=self._on_import_done,
            name=f"import:{handle.path.name}",
        )
        self._pending[guid] = _Pending(job, temp_dir)
        if wait:
            self.jobs.wait_idle()
        return job

    def reimport_dirty(self, *, wait: bool = False) -> list[Job]:
        """Запустить импорт всех «грязных» ассетов."""
        jobs = [self.reimport(guid) for guid in self.dirty()]
        if wait:
            self.jobs.wait_idle()
        return jobs

    def pump(self) -> int:
        """Применить результаты завершённых импортов (вызывать в главном потоке)."""
        return self.jobs.pump()

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Дождаться всех импортов и применить их результаты."""
        return self.jobs.wait_idle(timeout)

    def close(self) -> None:
        """Остановить собственный ``JobSystem`` (чужой не трогаем)."""
        if self._owns_jobs:
            self.jobs.shutdown()

    def __enter__(self) -> AssetDatabase:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Внутреннее: рабочий поток
    # ------------------------------------------------------------------ #
    @staticmethod
    def _run_import(
        guid: str,
        importer: AssetImporter,
        source: Path,
        settings: dict[str, Any],
        temp_dir: Path,
    ) -> _ImportResult:
        """Выполнить импортёр во временный каталог (рабочий поток).

        Не обращается к базе; при любой ошибке убирает за собой.
        """
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True)
        try:
            source_stamp = _stamp(source)
            ctx = ImportContext(guid, source, settings, temp_dir)
            try:
                artifacts = importer.import_asset(ctx)
            except ResourceError:
                raise
            except Exception as exc:  # noqa: BLE001 - оборачиваем ошибки импортёра
                raise AssetImportError(
                    f"Импортёр '{importer.name}' упал на '{source.name}': {exc}"
                ) from exc
            if not artifacts:
                raise AssetImportError(
                    f"Импортёр '{importer.name}' не создал ни одного артефакта"
                )
        except BaseException:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        baked: dict[str, dict[str, Any]] = {}
        for path in ctx.baked_inputs:
            try:
                baked[_norm(path)] = _stamp(path)
            except OSError:
                continue
        return _ImportResult(
            temp_dir,
            list(artifacts),
            list(ctx.dependencies),
            importer.version,
            _hash_json(settings),
            source_stamp,
            baked,
        )

    # ------------------------------------------------------------------ #
    # Внутреннее: главный поток
    # ------------------------------------------------------------------ #
    def _on_import_done(self, job: Job) -> None:
        """Применить итог фонового импорта (главный поток)."""
        guid = next((g for g, p in self._pending.items() if p.job is job), None)
        if guid is None:
            return
        pending = self._pending.pop(guid)
        handle = self._handles[guid]
        error = job.error
        if error is not None:
            shutil.rmtree(pending.temp_dir, ignore_errors=True)
            message = str(error) or type(error).__name__
            _log.error("Импорт '%s' не удался: %s", handle.path.name, message)
            self._handles[guid] = _replace(handle, error=message)
            self._emit(AssetImportFailed(guid, handle.path, message))
            return

        result: _ImportResult = job.result()
        try:
            self._commit(guid, handle, result)
        except OSError as exc:
            shutil.rmtree(result.temp_dir, ignore_errors=True)
            message = f"Не удалось сохранить артефакты: {exc}"
            _log.error("Импорт '%s' не удался: %s", handle.path.name, message)
            self._handles[guid] = _replace(handle, error=message)
            self._emit(AssetImportFailed(guid, handle.path, message))
            return

        if self.manager is not None and self.manager.is_loaded(handle.path):
            try:
                self.manager.reload(handle.path)
            except ResourceError:
                _log.exception("Не удалось перезагрузить '%s'", handle.path.name)
        self._emit(AssetImported(guid, handle.path))

    def _commit(self, guid: str, handle: AssetHandle, result: _ImportResult) -> None:
        """Атомарно заменить каталог артефактов и обновить состояние базы."""
        dependency_guids = self._register_dependencies(guid, result.dependencies)
        record = {
            "guid": guid,
            "importer": handle.importer,
            "path": self._relative_source(handle.path),
            "version": result.importer_version,
            "settings_hash": result.settings_hash,
            "source": result.source,
            "dependencies": dependency_guids,
            "baked": result.baked,
            "artifacts": [artifact.to_dict() for artifact in result.artifacts],
        }
        (result.temp_dir / RECORD_FILE).write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if handle.directory.exists():
            shutil.rmtree(handle.directory)
        os.replace(result.temp_dir, handle.directory)

        self._records[guid] = record
        self._graph.set(guid, dependency_guids)
        meta = self._metas[guid]
        if meta.version != result.importer_version:
            meta.version = result.importer_version
            self._store_meta(handle.path, meta)
        self._handles[guid] = AssetHandle(
            guid,
            handle.path,
            handle.importer,
            handle.directory,
            result.importer_version,
            tuple(result.artifacts),
        )

    def _relative_source(self, path: Path) -> str:
        """Путь исходника относительно корня базы (для VFS без ``AssetDatabase``)."""
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.name

    def _register_dependencies(self, guid: str, paths: list[Path]) -> list[str]:
        """Преобразовать пути зависимостей в GUID, регистрируя новые файлы."""
        guids: list[str] = []
        for dependency in paths:
            try:
                guids.append(self.register(dependency))
            except ResourceError as exc:
                _log.warning(
                    "Зависимость '%s' ассета %s пропущена: %s", dependency, guid, exc
                )
        return guids

    def _load_record(self, guid: str) -> None:
        """Подхватить результат прошлого импорта из ``Library`` (если он цел)."""
        handle = self._handles[guid]
        record_path = handle.directory / RECORD_FILE
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            artifacts = tuple(Artifact.from_dict(a) for a in record["artifacts"])
            dependencies = [parse_guid(d) for d in record.get("dependencies", [])]
            version = int(record["version"])
        except (OSError, ValueError, KeyError, TypeError):
            return
        if not all((handle.directory / a.name).is_file() for a in artifacts):
            return
        self._records[guid] = record
        self._graph.set(guid, dependencies)
        self._handles[guid] = _replace(
            handle, imported_version=version, artifacts=artifacts
        )

    def _importer_of(self, handle: AssetHandle) -> AssetImporter:
        importer = self.importers.importer_for(handle.path)
        if importer is None:
            raise ImporterNotFoundError(f"Для '{handle.path.name}' нет импортёра")
        return importer

    def _source_changed(self, path: Path, stored: dict[str, Any]) -> bool:
        """Сравнить исходник с отпечатком импорта: быстро по stat, затем по хешу."""
        try:
            current = _stat_stamp(path)
        except OSError:
            return False  # исходник удалён — переимпортировать нечего
        if stored.get("size") == current["size"] and stored.get(
            "mtime_ns"
        ) == current["mtime_ns"]:
            return False
        try:
            return _stamp(path)["sha256"] != stored.get("sha256")
        except OSError:
            return False

    def _absolute(self, path: Path | str) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        return Path(os.path.abspath(candidate))

    def _emit(self, event: Event) -> None:
        if self.events is not None:
            self.events.emit(event)

    def __len__(self) -> int:
        return len(self._handles)

    def __contains__(self, guid: object) -> bool:
        return guid in self._handles

    def __repr__(self) -> str:
        return (
            f"<AssetDatabase assets={len(self._handles)} "
            f"pending={len(self._pending)} root='{self.root}'>"
        )


def _norm(path: Path) -> str:
    return os.path.normcase(str(path))


def _same_path(left: Path, right: Path) -> bool:
    return _norm(Path(os.path.abspath(left))) == _norm(right)


def _replace(handle: AssetHandle, **changes: Any) -> AssetHandle:
    """Копия неизменяемого ``AssetHandle`` с заменой полей."""
    values = {
        "guid": handle.guid,
        "path": handle.path,
        "importer": handle.importer,
        "directory": handle.directory,
        "imported_version": handle.imported_version,
        "artifacts": handle.artifacts,
        "error": handle.error,
    }
    values.update(changes)
    return AssetHandle(**values)


def _hash_json(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=repr)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _stat_stamp(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _stamp(path: Path) -> dict[str, Any]:
    """Отпечаток исходника: размер, mtime и SHA-256 содержимого."""
    stamp = _stat_stamp(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    stamp["sha256"] = digest.hexdigest()
    return stamp
