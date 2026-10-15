"""Play Mode: запуск сцены в отдельном процессе плеера и управление им."""

from __future__ import annotations

import logging
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

from editor.bridge.engine_bridge import EngineBridge
from editor.bridge.ipc_client import IpcClient, IpcError
from engine.player.snapshot import SceneSnapshot

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from editor.context import EditorContext

__all__ = ["PlayModeService"]

_log = logging.getLogger("editor.play_mode")

EDITING = "editing"
PLAYING = "playing"
PAUSED = "paused"
STOPPING = "stopping"

#: Сколько ждать ``hello`` после старта процесса (мс).
START_TIMEOUT_MS = 10000
#: Сколько ждать штатного завершения после ``stop`` (мс).
STOP_TIMEOUT_MS = 2000

_PORT_RE = re.compile(r"RPC_PORT (\d+)")

#: Имя exe плеера рядом с exe редактора (собранная версия).
PLAYER_EXE_NAME = "MyEngine-Player.exe"


def _default_player_command() -> list[str]:
    """Команда запуска плеера: exe рядом с редактором или ``-m engine.player``."""
    if getattr(sys, "frozen", False):
        return [str(Path(sys.executable).with_name(PLAYER_EXE_NAME))]
    return [sys.executable, "-m", "engine.player"]


def _player_workdir() -> Path:
    """Рабочий каталог плеера: корень репозитория или папка с exe."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


class PlayModeService(QObject):
    """Жизненный цикл плеера: старт, пауза, шаг, остановка, события.

    Args:
        context: Контекст редактора (сцены, проект).
        parent: Родительский QObject.

    Signals:
        state_changed: ``editing`` / ``playing`` / ``paused`` / ``stopping``.
        stats_received: Статистика плеера (``fps``, ``dc``, ``tri``, ...).
        log_received: Запись журнала плеера.
        selection_received: ``id`` объекта, выбранного кликом в плеере.
        crashed: Текст traceback или причина аварийного завершения.
        load_progress: Прогресс загрузки сцены в плеере (0..1) и фаза.
        scene_loaded: Сцена в плеере загружена.
    """

    state_changed = Signal(str)
    stats_received = Signal(dict)
    log_received = Signal(dict)
    selection_received = Signal(int)
    crashed = Signal(str)
    load_progress = Signal(float, str)
    scene_loaded = Signal()

    def __init__(self, context: EditorContext, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.context = context
        self.state = EDITING
        self.pid = 0
        #: Вызывается перед запуском: сохранить грязные сцены; ``False`` — отмена.
        self.before_play: Callable[[], bool] = lambda: True
        #: Команда запуска плеера (подменяется в тестах).
        self.player_command: Callable[[], list[str]] = _default_player_command
        self.extra_args: list[str] = []
        self._process: QProcess | None = None
        self._client: IpcClient | None = None
        self._bridge = EngineBridge(self)
        self._bridge.stats.connect(self.stats_received)
        self._bridge.selection.connect(self.selection_received)
        self._bridge.load_progress.connect(self.load_progress)
        self._bridge.scene_loaded.connect(self.scene_loaded)
        self._bridge.crashed.connect(self._on_crash_report)
        self._bridge.exited.connect(self._on_player_gone)
        self._bridge.logs.connect(self._emit_logs)
        self._snapshot_path: Path | None = None
        self._stdout = ""
        self._crash_reported = False
        self._expected_stop = False
        self._start_timer = QTimer(self)
        self._start_timer.setSingleShot(True)
        self._start_timer.timeout.connect(self._on_start_timeout)
        self._stop_timer = QTimer(self)
        self._stop_timer.setSingleShot(True)
        self._stop_timer.timeout.connect(self._kill)

    # ------------------------------------------------------------------ #
    # Состояние
    # ------------------------------------------------------------------ #
    def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.state_changed.emit(state)

    @property
    def is_active(self) -> bool:
        """Плеер запущен (играет, на паузе или останавливается)."""
        return self.state != EDITING or self._process is not None

    # ------------------------------------------------------------------ #
    # Запуск
    # ------------------------------------------------------------------ #
    def enter_play(self) -> bool:
        """Сохранить сцену, заморозить снимок и запустить плеер.

        Returns:
            ``True``, если процесс запущен (``hello`` придёт асинхронно).
        """
        if self.is_active:
            return False
        if not self.before_play():
            return False
        document = self.context.active_document
        snapshot = SceneSnapshot.freeze(
            document.scene,
            self.context.asset_db,
            getattr(document, "sky", None),
            getattr(document, "post", None),
        )
        folder = self.context.project_root / "Library" / "PlayMode"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        self._snapshot_path = snapshot.save(folder / f"snapshot_{stamp}.json")
        command = self.player_command()
        args = [
            *command[1:],
            "--project",
            str(self.context.project_root),
            "--snapshot",
            str(self._snapshot_path),
            "--rpc-port",
            "0",
            "--windowed",
            "--use-project-saves",
            *self.extra_args,
        ]
        self._stdout = ""
        self._crash_reported = False
        self._expected_stop = False
        process = QProcess(self)
        process.setProgram(command[0])
        process.setArguments(args)
        process.setWorkingDirectory(str(_player_workdir()))
        process.readyReadStandardOutput.connect(self._on_stdout)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_process_error)
        self._process = process
        process.start()
        self._start_timer.start(START_TIMEOUT_MS)
        return True

    def _on_stdout(self) -> None:
        process = self._process
        if process is None:
            return
        self._stdout += bytes(process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        if self._client is not None:
            return
        found = _PORT_RE.search(self._stdout)
        if found:
            self._connect(int(found.group(1)))

    def _connect(self, port: int) -> None:
        client = IpcClient()
        try:
            hello = client.connect(port, timeout=5.0)
        except IpcError as error:
            self._fail(str(error))
            return
        self._start_timer.stop()
        self._client = client
        self.pid = int(hello.get("pid", 0))
        self._bridge.attach(client)
        self._set_state(PLAYING)

    def _on_start_timeout(self) -> None:
        if self._client is None:
            self._fail("Плеер не ответил за 10 секунд")

    def _on_process_error(self, _error: QProcess.ProcessError) -> None:
        process = self._process
        stopped = (
            process is not None and process.state() == QProcess.ProcessState.NotRunning
        )
        if stopped and self._client is None:
            self._fail(f"Плеер не запустился: {process.errorString()}")

    # ------------------------------------------------------------------ #
    # Команды
    # ------------------------------------------------------------------ #
    def _send(self, message: dict[str, Any]) -> bool:
        return self._client is not None and self._client.send(message)

    def pause(self) -> None:
        """Приостановить симуляцию."""
        if self.state == PLAYING and self._send({"cmd": "pause"}):
            self._set_state(PAUSED)

    def resume(self) -> None:
        """Продолжить симуляцию."""
        if self.state == PAUSED and self._send({"cmd": "resume"}):
            self._set_state(PLAYING)

    def toggle_pause(self) -> None:
        """Пауза/продолжить."""
        if self.state == PLAYING:
            self.pause()
        elif self.state == PAUSED:
            self.resume()

    def step(self, frames: int = 1) -> None:
        """Выполнить ``frames`` кадров и остаться на паузе."""
        if self.state in (PLAYING, PAUSED) and self._send(
            {"cmd": "step", "frames": int(frames)}
        ):
            self._set_state(PAUSED)

    def quick_save(self) -> None:
        """Быстрое сохранение в плеере (как F5 в его окне)."""
        if self.state in (PLAYING, PAUSED):
            self._send({"cmd": "quick_save"})

    def quick_load(self) -> None:
        """Быстрая загрузка в плеере (как F9 в его окне)."""
        if self.state in (PLAYING, PAUSED):
            self._send({"cmd": "quick_load"})

    def pick(self, x: int, y: int) -> None:
        """Выбрать объект в плеере по точке окна."""
        self._send({"cmd": "pick", "x": x, "y": y, "id": 1})

    def send_setting(self, path: str, value: Any) -> None:
        """Применить настройку «на лету», например ``sky.zenith_color``."""
        if isinstance(value, tuple):
            value = list(value)
        self._send({"cmd": "set_setting", "path": path, "value": value})

    # ------------------------------------------------------------------ #
    # Остановка
    # ------------------------------------------------------------------ #
    def stop(self) -> None:
        """Остановить плеер: ``stop``, ожидание 2 с, затем завершение."""
        if self._process is None or self.state == STOPPING:
            return
        self._expected_stop = True
        self._set_state(STOPPING)
        if not self._send({"cmd": "stop"}):
            self._kill()
            return
        self._stop_timer.start(STOP_TIMEOUT_MS)

    def shutdown(self) -> None:
        """Синхронно завершить плеер (закрытие редактора)."""
        process = self._process
        if process is None:
            return
        self._expected_stop = True
        self._send({"cmd": "stop"})
        if not process.waitForFinished(STOP_TIMEOUT_MS):
            process.kill()
            process.waitForFinished(1000)
        self._cleanup()

    def _kill(self) -> None:
        process = self._process
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            process.kill()
        elif process is not None:
            self._cleanup()

    def _emit_logs(self, records: list) -> None:
        for record in records:
            self.log_received.emit(record)

    def _on_crash_report(self, trace: str) -> None:
        self._crash_reported = True
        self.crashed.emit(trace)

    def _on_player_gone(self) -> None:
        """Плеер сообщил о выходе или оборвал соединение."""
        process = self._process
        if process is None:
            return
        if process.state() == QProcess.ProcessState.NotRunning:
            return
        if not self._expected_stop:
            # Соединение пропало, а процесс жив — даём шанс завершиться самому.
            self._stop_timer.start(STOP_TIMEOUT_MS)

    def _on_finished(self, code: int, _status: QProcess.ExitStatus) -> None:
        if not self._expected_stop and not self._crash_reported:
            tail = "\n".join(self._stdout.strip().splitlines()[-20:])
            self.crashed.emit(f"Плеер завершился неожиданно (код {code})\n{tail}")
        elif code != 0 and self._expected_stop and not self._crash_reported:
            self.crashed.emit(f"Плеер завершился с кодом {code}")
        self._cleanup()

    def _fail(self, reason: str) -> None:
        _log.error(reason)
        self._expected_stop = True
        self.crashed.emit(reason)
        self._kill()
        if self._process is None or self._process.state() == (
            QProcess.ProcessState.NotRunning
        ):
            self._cleanup()

    def _cleanup(self) -> None:
        self._start_timer.stop()
        self._stop_timer.stop()
        self._bridge.detach()
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._process is not None:
            self._process.deleteLater()
            self._process = None
        if self._snapshot_path is not None:
            self._snapshot_path.unlink(missing_ok=True)
            self._snapshot_path = None
        self.pid = 0
        self._set_state(EDITING)
