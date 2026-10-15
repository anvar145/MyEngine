"""Клиент IPC плеера: TCP на 127.0.0.1, JSON-строки.

Читает сокет в отдельном потоке и складывает сообщения в очередь; GUI-поток
забирает их через :meth:`IpcClient.poll`, поэтому интерфейс не блокируется.
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import socket
import threading
from typing import Any

__all__ = ["PROTOCOL_VERSION", "IpcClient", "IpcError"]

PROTOCOL_VERSION = 1
_log = logging.getLogger("editor.ipc")


class IpcError(RuntimeError):
    """Не удалось установить или поддерживать соединение с плеером."""


class IpcClient:
    """Соединение редактора с плеером."""

    def __init__(self) -> None:
        self._socket: socket.socket | None = None
        self._messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.hello: dict[str, Any] = {}
        #: ``True``, пока соединение живо (сбрасывается потоком чтения).
        self.connected = False

    def connect(self, port: int, timeout: float = 10.0) -> dict[str, Any]:
        """Подключиться, принять ``hello`` и ответить ``welcome``.

        Args:
            port: Порт плеера на 127.0.0.1.
            timeout: Сколько ждать ``hello`` (секунды).

        Returns:
            Сообщение ``hello`` плеера.

        Raises:
            IpcError: Нет соединения, нет ``hello`` или версия протокола иная.
        """
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        except OSError as error:
            raise IpcError(f"Плеер не принимает соединение: {error}") from error
        sock.settimeout(timeout)
        buffer = b""
        try:
            while b"\n" not in buffer:
                chunk = sock.recv(65536)
                if not chunk:
                    raise IpcError("Плеер закрыл соединение до hello")
                buffer += chunk
        except (TimeoutError, OSError) as error:
            sock.close()
            raise IpcError(f"Плеер не прислал hello: {error}") from error
        line, rest = buffer.split(b"\n", 1)
        hello = json.loads(line.decode("utf-8"))
        if hello.get("type") != "hello" or hello.get("protocol") != PROTOCOL_VERSION:
            sock.close()
            raise IpcError(f"Несовместимый протокол: {hello!r}")
        self._socket = sock
        self.hello = hello
        self.connected = True
        self.send({"type": "welcome", "protocol": PROTOCOL_VERSION})
        sock.settimeout(0.2)
        self._thread = threading.Thread(
            target=self._read_loop, args=(rest,), name="editor-ipc", daemon=True
        )
        self._thread.start()
        return hello

    def _read_loop(self, buffer: bytes) -> None:
        sock = self._socket
        assert sock is not None
        while True:
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if line.strip():
                    try:
                        self._messages.put(json.loads(line.decode("utf-8")))
                    except ValueError:
                        _log.warning("Непонятное сообщение плеера: %r", line[:80])
            if self._stop.is_set():
                break
            try:
                chunk = sock.recv(65536)
            except TimeoutError:
                continue
            except OSError:
                break
            if not chunk:
                break
            buffer += chunk
        self.connected = False
        self._messages.put({"event": "disconnected"})

    def send(self, message: dict[str, Any]) -> bool:
        """Отправить команду; ``False``, если соединение потеряно."""
        sock = self._socket
        if sock is None or not self.connected:
            return False
        data = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
        with self._lock:
            try:
                sock.sendall(data)
            except OSError:
                return False
        return True

    def poll(self) -> list[dict[str, Any]]:
        """Забрать все накопленные сообщения плеера."""
        found = []
        while True:
            try:
                found.append(self._messages.get_nowait())
            except queue.Empty:
                return found

    def close(self) -> None:
        """Закрыть соединение и остановить поток чтения."""
        self._stop.set()
        self.connected = False
        if self._socket is not None:
            with contextlib.suppress(OSError):
                self._socket.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._socket = None
