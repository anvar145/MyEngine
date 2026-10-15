"""Мост событий плеера в Qt-сигналы с ограничением частоты."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

from editor.bridge.ipc_client import IpcClient

__all__ = ["EngineBridge"]

#: Период опроса очереди: 30 Гц — предел для логов.
TICK_MS = 33
#: Статистика уходит в интерфейс не чаще 5 Гц.
STATS_MS = 200


class EngineBridge(QObject):
    """Забирает сообщения из :class:`IpcClient` в GUI-потоке.

    Статистика схлопывается до последней записи не чаще 5 Гц, журнал
    отдаётся батчами 30 Гц.

    Signals:
        stats: Словарь статистики плеера.
        logs: Список записей журнала.
        selection: ``id`` выбранного в плеере объекта.
        load_progress: Прогресс загрузки сцены (0..1) и фаза.
        scene_loaded: Сцена загружена и активирована.
        reply: Ответ на команду с полем ``id``.
        crashed: Текст traceback падения плеера.
        exited: Плеер завершился или соединение потеряно.
    """

    stats = Signal(dict)
    logs = Signal(list)
    selection = Signal(int)
    reply = Signal(dict)
    crashed = Signal(str)
    exited = Signal()
    load_progress = Signal(float, str)
    scene_loaded = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._client: IpcClient | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self.pump)
        self._pending_stats: dict[str, Any] | None = None
        self._since_stats = STATS_MS

    def attach(self, client: IpcClient) -> None:
        """Начать читать сообщения клиента."""
        self._client = client
        self._pending_stats = None
        self._since_stats = STATS_MS
        self._timer.start()

    def detach(self) -> None:
        """Прекратить чтение (клиент закрывает вызывающий)."""
        self._timer.stop()
        self._client = None

    def pump(self) -> None:
        """Разобрать очередь сообщений (вызывается таймером)."""
        client = self._client
        if client is None:
            return
        records: list[dict[str, Any]] = []
        finished = False
        for message in client.poll():
            kind = message.get("event")
            if kind == "stats":
                self._pending_stats = message
            elif kind == "log":
                records.append(message)
            elif kind == "selection":
                self.selection.emit(int(message.get("entity", -1)))
            elif kind == "scene_load_progress":
                self.load_progress.emit(
                    float(message.get("progress", 0.0)), str(message.get("phase", ""))
                )
            elif kind == "scene_loaded":
                self.scene_loaded.emit()
            elif kind == "crash":
                self.crashed.emit(str(message.get("traceback", "")))
            elif kind in ("exit", "disconnected"):
                finished = True
            elif "id" in message:
                self.reply.emit(message)
        self._since_stats += TICK_MS
        if self._pending_stats is not None and self._since_stats >= STATS_MS:
            self._since_stats = 0
            stats, self._pending_stats = self._pending_stats, None
            self.stats.emit(stats)
        if records:
            self.logs.emit(records)
        if finished:
            self.exited.emit()
