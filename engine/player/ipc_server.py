"""Сервер IPC плеера: JSON-строки поверх TCP на 127.0.0.1.

Сообщения — UTF-8 JSON, каждое оканчивается переводом строки. Рукопожатие:
плеер шлёт ``hello``, редактор отвечает ``welcome``. Команды приходят в
отдельном потоке и складываются в очередь, которую главный поток забирает
через :meth:`IpcServer.poll`; вызовы GL из потока сервера не делаются.
Исходящие сообщения кладутся в очередь и уходят из отдельного потока, поэтому
медленный клиент не тормозит кадр главного потока; при переполнении очереди
теряются самые старые сообщения.
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import socket
import threading
import time
from typing import Any

__all__ = ["PROTOCOL_VERSION", "IpcServer"]

PROTOCOL_VERSION = 1
#: Размер исходящей очереди (сообщений): дальше вытесняются самые старые.
OUTGOING_LIMIT = 2000
_log = logging.getLogger("engine.player.ipc")


class IpcServer:
    """Одноклиентный сервер команд плеера.

    Args:
        port: Порт на 127.0.0.1; 0 — выбрать свободный.
    """

    def __init__(self, port: int = 0) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.bind(("127.0.0.1", port))
        self._socket.listen(1)
        self.port: int = self._socket.getsockname()[1]
        self._client: socket.socket | None = None
        self._commands: queue.Queue[dict[str, Any]] = queue.Queue()
        self._lock = threading.Lock()
        self._outgoing: queue.Queue[bytes] = queue.Queue(maxsize=OUTGOING_LIMIT)
        self._writer: threading.Thread | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.connected = threading.Event()
        self.welcomed = threading.Event()
        self.hello_payload: dict[str, Any] = {}

    def start(self, hello: dict[str, Any] | None = None) -> None:
        """Запустить поток приёма соединения.

        Args:
            hello: Дополнительные поля сообщения ``hello`` (``pid``, ...).
        """
        self.hello_payload = {
            "type": "hello",
            "port": self.port,
            "protocol": PROTOCOL_VERSION,
            **(hello or {}),
        }
        self._thread = threading.Thread(
            target=self._serve, name="player-ipc", daemon=True
        )
        self._thread.start()
        self._writer = threading.Thread(
            target=self._write_loop, name="player-ipc-out", daemon=True
        )
        self._writer.start()

    def _write_loop(self) -> None:
        """Отправлять исходящие сообщения из очереди (отдельный поток)."""
        while not self._stop.is_set():
            try:
                data = self._outgoing.get(timeout=0.2)
            except queue.Empty:
                continue
            self._write(data)

    def flush(self, timeout: float = 0.5) -> None:
        """Дождаться отправки очереди (при падении и остановке)."""
        deadline = time.monotonic() + timeout
        while not self._outgoing.empty() and time.monotonic() < deadline:
            time.sleep(0.005)

    def _serve(self) -> None:
        self._socket.settimeout(0.2)
        while not self._stop.is_set():
            try:
                client, _ = self._socket.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            self._client = client
            self.connected.set()
            hello = json.dumps(self.hello_payload, ensure_ascii=False) + "\n"
            self._write(hello.encode("utf-8"))
            self._read_loop(client)
            self.connected.clear()
            self.welcomed.clear()
            self._client = None
            if self._stop.is_set():
                return
            self._commands.put({"cmd": "stop", "reason": "editor_disconnected"})

    def _read_loop(self, client: socket.socket) -> None:
        client.settimeout(0.2)
        buffer = b""
        while not self._stop.is_set():
            try:
                chunk = client.recv(65536)
            except TimeoutError:
                continue
            except OSError:
                return
            if not chunk:
                return
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if line.strip():
                    self._handle_line(line)

    def _handle_line(self, line: bytes) -> None:
        try:
            message = json.loads(line.decode("utf-8"))
        except ValueError:
            _log.warning("Непонятное сообщение IPC: %r", line[:80])
            return
        if message.get("type") == "welcome":
            self.welcomed.set()
        else:
            self._commands.put(message)

    def wait_welcome(self, timeout: float) -> bool:
        """Дождаться ``welcome`` от редактора."""
        return self.welcomed.wait(timeout)

    def poll(self) -> list[dict[str, Any]]:
        """Забрать накопленные команды (вызывать из главного потока)."""
        commands = []
        while True:
            try:
                commands.append(self._commands.get_nowait())
            except queue.Empty:
                return commands

    def send(self, message: dict[str, Any]) -> bool:
        """Поставить сообщение в очередь отправки; ``False``, если клиента нет."""
        if self._client is None:
            return False
        data = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
        while True:
            try:
                self._outgoing.put_nowait(data)
                return True
            except queue.Full:
                with contextlib.suppress(queue.Empty):
                    self._outgoing.get_nowait()

    def _write(self, data: bytes) -> bool:
        """Записать данные в сокет клиента (поток отправки и рукопожатие)."""
        client = self._client
        if client is None:
            return False
        with self._lock:
            try:
                client.sendall(data)
            except OSError:
                return False
        return True

    def event(self, name: str, **fields: Any) -> bool:
        """Отправить событие ``{"event": name, ...}``."""
        return self.send({"event": name, **fields})

    def reply(self, command: dict[str, Any], result: Any) -> bool:
        """Ответить на команду с полем ``id``."""
        return self.send({"id": command.get("id"), "result": result})

    def crash(self, trace: str) -> None:
        """Сообщить о падении плеера."""
        self.event("crash", traceback=trace)
        self.flush()

    def shutdown(self) -> None:
        """Остановить сервер и закрыть сокеты."""
        self.flush()
        self._stop.set()
        for sock in (self._client, self._socket):
            if sock is not None:
                with contextlib.suppress(OSError):
                    sock.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
