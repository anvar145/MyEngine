"""IPC плеера: исходящие сообщения не блокируют главный поток."""

from __future__ import annotations

import json
import socket
import time

from engine.player.ipc_server import IpcServer


def _connect(server: IpcServer) -> socket.socket:
    client = socket.create_connection(("127.0.0.1", server.port), timeout=2.0)
    assert server.connected.wait(2.0)
    return client


def test_send_does_not_block_when_client_does_not_read() -> None:
    server = IpcServer()
    server.start()
    client = _connect(server)
    try:
        started = time.perf_counter()
        for i in range(5000):
            server.event("log", text="x" * 2000, n=i)
        assert time.perf_counter() - started < 0.5
    finally:
        client.close()
        server.shutdown()


def test_messages_arrive_in_order_after_flush() -> None:
    server = IpcServer()
    server.start()
    client = _connect(server)
    try:
        for i in range(20):
            server.event("tick", n=i)
        server.flush()
        data = b""
        client.settimeout(2.0)
        while data.count(b"\n") < 21:
            data += client.recv(65536)
        lines = [json.loads(x) for x in data.decode().splitlines()]
        assert lines[0]["type"] == "hello"
        assert [m["n"] for m in lines[1:21]] == list(range(20))
    finally:
        client.close()
        server.shutdown()


def test_send_without_client_returns_false() -> None:
    server = IpcServer()
    try:
        assert server.event("x") is False
    finally:
        server.shutdown()
