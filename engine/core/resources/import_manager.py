"""Менеджер импортов: очередь, слежение за файлами и сервисные операции.

:class:`ImportManager` надстраивается над :class:`~engine.core.resources.asset_db.
AssetDatabase` и отвечает за «жизнь проекта»:

* ``open()`` сканирует проект и ставит в очередь грязные ассеты;
* ``poll()`` (по таймеру) находит новые, изменённые, удалённые и переехавшие
  файлы — без внешних зависимостей, сравнением ``stat``;
* очередь ограничивает число одновременных импортов, поддерживает приоритет и
  отмену ожидающих; состояние каждого ассета доступно интерфейсу;
* сервисные операции: смена настроек с проверкой по схеме, очистка и сборка
  ``Library``, поиск и удаление «осиротевших» каталогов.

Всё живёт в **главном потоке**: рабочие потоки есть только у ``JobSystem`` базы.

Example:
    >>> manager = ImportManager(AssetDatabase(Path("Game/Assets")))
    >>> manager.open()
    >>> manager.wait_idle()
    >>> manager.poll()          # раз в пол секунды из таймера редактора
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from engine.core.event_bus import Event, EventBus
from engine.core.resources.asset_db import (
    AssetDatabase,
    AssetImported,
    AssetImportFailed,
)
from engine.core.resources.errors import ResourceError, ResourceNotFoundError
from engine.core.resources.guid import is_valid_guid
from engine.core.resources.import_settings import validate_settings
from engine.core.resources.meta import read_meta

__all__ = [
    "AssetMoved",
    "AssetRemoved",
    "ImportBatchFinished",
    "ImportBatchProgress",
    "ImportBatchStarted",
    "ImportManager",
    "ImportState",
]

_log = logging.getLogger("engine.core.resources.import_manager")


class ImportState(str, Enum):
    """Состояние ассета в очереди импорта."""

    IDLE = "idle"
    QUEUED = "queued"
    RUNNING = "running"
    OK = "ok"
    FAILED = "failed"


@dataclass(slots=True)
class ImportBatchStarted(Event):
    """Событие: очередь была пуста, началась новая пачка импортов."""

    total: int


@dataclass(slots=True)
class ImportBatchProgress(Event):
    """Событие: завершился очередной импорт пачки."""

    done: int
    total: int
    guid: str


@dataclass(slots=True)
class ImportBatchFinished(Event):
    """Событие: пачка завершена, очередь пуста."""

    ok: int
    failed: int
    total: int


@dataclass(slots=True)
class AssetRemoved(Event):
    """Событие: исходник ассета удалён, ассет снят с учёта."""

    guid: str
    path: Path


@dataclass(slots=True)
class AssetMoved(Event):
    """Событие: исходник ассета переехал (GUID прежний)."""

    guid: str
    old_path: Path
    new_path: Path


class ImportManager:
    """Очередь импортов и слежение за проектом.

    Args:
        db: База ассетов проекта. Если у неё нет шины событий, менеджер создаёт её.
        poll_interval: Минимальный интервал между обходами диска в секундах.
        max_parallel: Сколько импортов одновременно отдаётся в ``JobSystem``.
        clock: Источник времени (для тестов).
    """

    def __init__(
        self,
        db: AssetDatabase,
        *,
        poll_interval: float = 1.0,
        max_parallel: int = 2,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.db = db
        self.poll_interval = float(poll_interval)
        self.max_parallel = max(1, int(max_parallel))
        self._clock = clock
        if db.events is None:
            db.events = EventBus()
        self.events: EventBus = db.events

        self._states: dict[str, ImportState] = {}
        self._errors: dict[str, str] = {}
        self._failed_stamp: dict[str, tuple[int, int]] = {}
        self._queue: deque[str] = deque()
        self._running: set[str] = set()
        self._again: set[str] = set()
        self._last_poll = float("-inf")
        self._opened = False
        self._reset_batch()
        self._subscriptions = [
            self.events.subscribe(AssetImported, self._on_imported),
            self.events.subscribe(AssetImportFailed, self._on_failed),
        ]

    # ------------------------------------------------------------------ #
    # Жизненный цикл
    # ------------------------------------------------------------------ #
    def open(self) -> int:
        """Просканировать проект и поставить в очередь грязные ассеты.

        Returns:
            Сколько ассетов поставлено в очередь.
        """
        self._opened = True
        self._last_poll = self._clock()
        queued = 0
        for guid in self.db.scan():
            if self.db.is_dirty(guid):
                queued += self._enqueue(guid)
            else:
                self._states[guid] = ImportState.OK
        self._feed()
        return queued

    def close(self) -> None:
        """Отменить ожидающие, дождаться идущих импортов и отписаться."""
        self.cancel()
        self.db.wait_idle()
        for subscription in self._subscriptions:
            subscription.cancel()
        self._subscriptions.clear()
        self._opened = False

    def __enter__(self) -> ImportManager:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Состояние
    # ------------------------------------------------------------------ #
    @property
    def busy(self) -> bool:
        """``True``, если есть ожидающие или идущие импорты."""
        return bool(self._queue or self._running)

    def state(self, guid: str) -> ImportState:
        """Состояние ассета (``IDLE``, если менеджер его ещё не обрабатывал)."""
        return self._states.get(guid, ImportState.IDLE)

    def states(self) -> dict[str, ImportState]:
        """Копия таблицы состояний."""
        return dict(self._states)

    def failures(self) -> dict[str, str]:
        """GUID ассетов с ошибкой последнего импорта → текст ошибки."""
        return dict(self._errors)

    def queued(self) -> list[str]:
        """GUID ожидающих ассетов в порядке выполнения."""
        return list(self._queue)

    def progress(self) -> tuple[int, int]:
        """Прогресс текущей пачки: ``(завершено, всего)``; вне пачки ``(0, 0)``."""
        return self._batch_done, self._batch_total

    # ------------------------------------------------------------------ #
    # Слежение за проектом
    # ------------------------------------------------------------------ #
    def poll(self, *, force: bool = False) -> int:
        """Обработать изменения на диске и продвинуть очередь.

        Args:
            force: Игнорировать ``poll_interval``.

        Returns:
            Число замеченных изменений (новые, изменённые, удалённые, переехавшие).
        """
        self.pump()
        now = self._clock()
        if not force and now - self._last_poll < self.poll_interval:
            return 0
        self._last_poll = now
        changes = self._poll_files()
        changes += self._poll_changed()
        self._feed()
        return changes

    def pump(self) -> int:
        """Применить завершённые импорты и запустить следующие из очереди."""
        handled = self.db.pump()
        self._feed()
        return handled

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Дождаться опустошения очереди (применяя результаты в этом потоке).

        Returns:
            ``True``, если очередь опустела до истечения таймаута.
        """
        deadline = None if timeout is None else self._clock() + timeout
        while self.busy:
            self._feed()
            remaining = None if deadline is None else deadline - self._clock()
            if remaining is not None and remaining <= 0:
                return False
            if not self.db.wait_idle(remaining):
                return False
            if self.busy and self.db.jobs.pending == 0 and not self._queue:
                # В «идущих» остались ассеты без задач (сняты с учёта) — сбросить.
                self._running.clear()
                self._finish_if_idle()
        return True

    # ------------------------------------------------------------------ #
    # Операции
    # ------------------------------------------------------------------ #
    def reimport(self, target: str | Path, *, priority: bool = False) -> str:
        """Принудительно переимпортировать ассет (по GUID или пути).

        Args:
            target: GUID или путь к исходнику (новый файл будет зарегистрирован).
            priority: Поставить в начало очереди.

        Returns:
            GUID ассета.

        Raises:
            ResourceError: Нет файла или импортёра.
        """
        guid = self._guid_for(target)
        self._enqueue(guid, priority=priority)
        self._feed()
        return guid

    def reimport_all(self, *, force: bool = True) -> int:
        """Поставить в очередь все ассеты (``force=False`` — только грязные).

        Returns:
            Сколько ассетов добавлено в очередь.
        """
        return self._enqueue_many(self.db.guids, force=force)

    def reimport_folder(self, folder: Path | str, *, force: bool = True) -> int:
        """Поставить в очередь ассеты внутри каталога (рекурсивно)."""
        base = os.path.normcase(str(self.db._absolute(folder)))  # noqa: SLF001
        guids = [
            guid
            for guid in self.db.guids
            if os.path.normcase(str(self.db.resolve(guid).path)).startswith(
                base + os.sep
            )
        ]
        return self._enqueue_many(guids, force=force)

    def cancel(self, guid: str | None = None) -> int:
        """Отменить ожидающие импорты (идущие дорабатывают).

        Args:
            guid: Конкретный ассет; ``None`` — все ожидающие.

        Returns:
            Сколько ассетов снято с очереди.
        """
        targets = list(self._queue) if guid is None else [guid]
        removed = 0
        for item in targets:
            if item in self._queue:
                self._queue.remove(item)
                self._states[item] = ImportState.IDLE
                self._batch_total -= 1
                removed += 1
        self._again.difference_update(targets)
        self._finish_if_idle()
        return removed

    def set_settings(
        self, guid: str, settings: dict[str, Any], *, reimport: bool = True
    ) -> dict[str, Any]:
        """Проверить настройки по схеме импортёра, записать в ``.meta`` и применить.

        В ``.meta`` попадают только значения, отличные от умолчаний.

        Args:
            guid: GUID ассета.
            settings: Полный набор (или часть) настроек.
            reimport: Сразу поставить ассет в начало очереди.

        Returns:
            Сохранённые (нормализованные) пользовательские настройки.

        Raises:
            AssetImportError: Настройки не прошли проверку.
        """
        handle = self.db.resolve(guid)
        importer = self.db.importers.importer_for(handle.path)
        if importer is None:
            raise ResourceNotFoundError(f"Для '{handle.path.name}' нет импортёра")
        cleaned = validate_settings(importer, settings)
        self.db.set_settings(guid, cleaned)
        if reimport:
            self._enqueue(guid, priority=True)
            self._feed()
        return cleaned

    def clear_library(self) -> int:
        """Удалить артефакты всех ассетов (идущие импорты дожидаются).

        Returns:
            Сколько ассетов сброшено.
        """
        self.cancel()
        self.db.wait_idle()
        guids = self.db.guids
        for guid in guids:
            self.db.invalidate(guid)
            self._states[guid] = ImportState.IDLE
            self._errors.pop(guid, None)
        self.gc_library()
        return len(guids)

    def rebuild_library(self) -> int:
        """Очистить ``Library`` и поставить все ассеты в очередь заново."""
        self.clear_library()
        return self.reimport_all()

    def orphans(self) -> list[Path]:
        """Каталоги ``Library/<xx>/<guid>[.tmp]``, которым не соответствует ассет.

        Посторонние файлы и каталоги (например, кэш шейдеров) не затрагиваются.
        """
        library = self.db.library
        result: list[Path] = []
        if not library.is_dir():
            return result
        for shard in sorted(library.iterdir()):
            if not (shard.is_dir() and len(shard.name) == 2 and _is_hex(shard.name)):
                continue
            for entry in sorted(shard.iterdir()):
                if not entry.is_dir():
                    continue
                name = entry.name.removesuffix(".tmp")
                unknown = not is_valid_guid(name) or name not in self.db
                stale_tmp = entry.name.endswith(".tmp") and name not in self._running
                if unknown or stale_tmp:
                    result.append(entry)
        return result

    def gc_library(self) -> int:
        """Удалить осиротевшие каталоги и пустые шарды.

        Returns:
            Сколько каталогов артефактов удалено.
        """
        orphans = self.orphans()
        for entry in orphans:
            shutil.rmtree(entry, ignore_errors=True)
        library = self.db.library
        if library.is_dir():
            for shard in library.iterdir():
                if shard.is_dir() and len(shard.name) == 2 and _is_hex(shard.name):
                    with contextlib.suppress(OSError):
                        shard.rmdir()
        return len(orphans)

    # ------------------------------------------------------------------ #
    # Внутреннее: обход диска
    # ------------------------------------------------------------------ #
    def _poll_files(self) -> int:
        """Найти новые, удалённые и переехавшие файлы."""
        db = self.db
        missing = {
            guid for guid in db.guids if not db.resolve(guid).path.exists()
        }
        changes = 0
        for path in list(db.iter_source_files()):
            if db.guid_of(path) is not None:
                continue
            try:
                meta = read_meta(path)
                if meta is not None and meta.guid in missing:
                    old = db.resolve(meta.guid).path
                    db.move(meta.guid, path)
                    missing.discard(meta.guid)
                    self.events.emit(AssetMoved(meta.guid, old, path))
                    self._enqueue(meta.guid, force=False)
                else:
                    self._enqueue(db.register(path), force=False)
            except ResourceError as exc:
                _log.error("Ассет '%s' пропущен: %s", path, exc)
                continue
            changes += 1
        for guid in missing:
            path = db.resolve(guid).path
            self.cancel(guid)
            if guid in self._running:
                self._running.discard(guid)
                self._complete(guid, ok=True, notify=False)
            self._again.discard(guid)
            self._states.pop(guid, None)
            self._errors.pop(guid, None)
            self._failed_stamp.pop(guid, None)
            db.unregister(guid)
            self.events.emit(AssetRemoved(guid, path))
            changes += 1
        self._finish_if_idle()
        return changes

    def _poll_changed(self) -> int:
        """Поставить в очередь ассеты, у которых изменился исходник или вход."""
        changes = 0
        for guid in self.db.guids:
            state = self.state(guid)
            if state in (ImportState.QUEUED, ImportState.RUNNING):
                continue
            if state is ImportState.FAILED and (
                self._stamp(guid) == self._failed_stamp.get(guid)
            ):
                continue
            if self.db.is_dirty(guid):
                changes += self._enqueue(guid, force=False)
        return changes

    def _stamp(self, guid: str) -> tuple[int, int] | None:
        try:
            stat = self.db.resolve(guid).path.stat()
        except OSError:
            return None
        return stat.st_size, stat.st_mtime_ns

    # ------------------------------------------------------------------ #
    # Внутреннее: очередь
    # ------------------------------------------------------------------ #
    def _guid_for(self, target: str | Path) -> str:
        if isinstance(target, str) and is_valid_guid(target):
            self.db.resolve(target)
            return target
        guid = self.db.guid_of(target)
        return guid if guid is not None else self.db.register(target)

    def _enqueue_many(self, guids: Iterable[str], *, force: bool) -> int:
        added = sum(self._enqueue(guid, force=force) for guid in list(guids))
        self._feed()
        return added

    def _enqueue(self, guid: str, *, priority: bool = False, force: bool = True) -> int:
        """Поставить ассет в очередь; возвращает 1, если добавлен."""
        state = self.state(guid)
        if state is ImportState.RUNNING:
            self._again.add(guid)  # файл изменился во время импорта
            return 0
        if state is ImportState.QUEUED:
            if priority:
                self._queue.remove(guid)
                self._queue.appendleft(guid)
            return 0
        if not force and not self.db.is_dirty(guid):
            self._states[guid] = ImportState.OK
            return 0
        if self._batch_total == 0:
            self.events.emit(ImportBatchStarted(1))
        self._batch_total += 1
        self._states[guid] = ImportState.QUEUED
        if priority:
            self._queue.appendleft(guid)
        else:
            self._queue.append(guid)
        return 1

    def _feed(self) -> None:
        """Отдать в ``JobSystem`` столько ассетов, сколько позволяет лимит."""
        while self._queue and len(self._running) < self.max_parallel:
            guid = self._queue.popleft()
            if guid not in self.db:
                self._complete(guid, ok=True, notify=False)
                continue
            try:
                self.db.reimport(guid)
            except ResourceError as exc:
                self._errors[guid] = str(exc)
                self._states[guid] = ImportState.FAILED
                self._failed_stamp[guid] = self._stamp(guid) or (0, 0)
                self._complete(guid, ok=False, notify=True)
                continue
            self._states[guid] = ImportState.RUNNING
            self._running.add(guid)
        self._finish_if_idle()

    def _on_imported(self, event: AssetImported) -> None:
        self._errors.pop(event.guid, None)
        self._failed_stamp.pop(event.guid, None)
        self._states[event.guid] = ImportState.OK
        self._finished(event.guid, ok=True)

    def _on_failed(self, event: AssetImportFailed) -> None:
        self._errors[event.guid] = event.error
        self._failed_stamp[event.guid] = self._stamp(event.guid) or (0, 0)
        self._states[event.guid] = ImportState.FAILED
        self._finished(event.guid, ok=False)

    def _finished(self, guid: str, *, ok: bool) -> None:
        if guid not in self._running:
            return  # импорт запущен не менеджером
        self._running.discard(guid)
        self._complete(guid, ok=ok, notify=True)
        if guid in self._again:
            self._again.discard(guid)
            self._enqueue(guid)
        self._feed()
        self._finish_if_idle()

    def _complete(self, guid: str, *, ok: bool, notify: bool) -> None:
        self._batch_done += 1
        if ok:
            self._batch_ok += 1
        else:
            self._batch_failed += 1
        if notify:
            self.events.emit(
                ImportBatchProgress(self._batch_done, self._batch_total, guid)
            )

    def _finish_if_idle(self) -> None:
        if self.busy:
            return
        if self._batch_total > 0:
            self.events.emit(
                ImportBatchFinished(
                    self._batch_ok, self._batch_failed, self._batch_total
                )
            )
        self._reset_batch()

    def _reset_batch(self) -> None:
        self._batch_total = 0
        self._batch_done = 0
        self._batch_ok = 0
        self._batch_failed = 0

    def __repr__(self) -> str:
        return (
            f"<ImportManager queued={len(self._queue)} running={len(self._running)} "
            f"assets={len(self.db)}>"
        )


def _is_hex(text: str) -> bool:
    return all(char in "0123456789abcdef" for char in text)
