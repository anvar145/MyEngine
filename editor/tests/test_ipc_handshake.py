"""Тесты IPC плеера: рукопожатие, команды и события."""

from __future__ import annotations

import time

from editor.bridge.ipc_client import PROTOCOL_VERSION, IpcClient
from engine.player.ipc_server import IpcServer


def _wait(predicate, timeout: float = 3.0) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_ipc_handshake() -> None:
    """hello → welcome, затем команда доходит до плеера, а событие — до клиента."""
    server = IpcServer(0)
    server.start({"pid": 4242, "capabilities": {"stats": True}})
    client = IpcClient()
    try:
        hello = client.connect(server.port, timeout=3.0)
        assert hello["type"] == "hello"
        assert hello["protocol"] == PROTOCOL_VERSION == 1
        assert hello["pid"] == 4242
        assert hello["port"] == server.port
        assert server.wait_welcome(3.0)

        assert client.send({"cmd": "pause"})
        commands: list[dict] = []
        assert _wait(lambda: commands.extend(server.poll()) or commands)
        assert commands[0]["cmd"] == "pause"

        server.event("stats", fps=58, dc=12, tri=2450, script_ms=0.8)
        messages: list[dict] = []
        assert _wait(lambda: messages.extend(client.poll()) or messages)
        assert messages[0]["event"] == "stats"
        assert messages[0]["tri"] == 2450
    finally:
        client.close()
        server.shutdown()


def test_server_reports_editor_disconnect_as_stop() -> None:
    """Обрыв соединения превращается в команду stop (плеер не остаётся сиротой)."""
    server = IpcServer(0)
    server.start({"pid": 1})
    client = IpcClient()
    client.connect(server.port, timeout=3.0)
    client.close()
    commands: list[dict] = []
    try:
        assert _wait(lambda: commands.extend(server.poll()) or commands)
        assert commands[-1]["cmd"] == "stop"
    finally:
        server.shutdown()
