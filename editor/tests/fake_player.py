"""Поддельный плеер для тестов ``PlayModeService`` (без GL и сцены).

Поведение задаётся переменной окружения ``FAKE_PLAYER_MODE``: ``normal`` —
слушает команды до ``stop``; ``crash`` — шлёт ``crash`` и выходит с кодом 1;
``hang`` — игнорирует ``stop`` (проверка принудительного завершения).
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from engine.player.ipc_server import IpcServer  # noqa: E402


def main() -> int:
    mode = os.environ.get("FAKE_PLAYER_MODE", "normal")
    server = IpcServer(0)
    server.start({"pid": os.getpid()})
    print(f"RPC_PORT {server.port}", flush=True)
    server.wait_welcome(10.0)
    if mode == "crash":
        server.crash("Traceback: fake crash")
        time.sleep(0.2)
        server.shutdown()
        return 1
    last_stats = 0.0
    while True:
        for command in server.poll():
            name = command.get("cmd")
            if name == "stop" and mode != "hang":
                server.event("exit", code=0)
                server.shutdown()
                return 0
            if name == "pick":
                server.reply(command, {"entity": 7, "name": "Cube"})
                server.event("selection", entity=7)
        now = time.perf_counter()
        if now - last_stats > 0.05:
            last_stats = now
            server.event("stats", fps=60.0, dc=3, tri=1934, script_ms=0.1)
            server.event("log", level="INFO", msg="tick", source="fake", line=1)
        time.sleep(0.01)


if __name__ == "__main__":
    sys.exit(main())
