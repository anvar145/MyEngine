"""Мост плеера: события прогресса загрузки сцены доходят сигналами."""

from __future__ import annotations

from typing import Any

from editor.bridge.engine_bridge import EngineBridge


class _FakeClient:
    """Клиент IPC с готовой очередью сообщений."""

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = messages

    def poll(self) -> list[dict[str, Any]]:
        messages, self._messages = self._messages, []
        return messages


def test_load_progress_and_loaded_signals(qapp) -> None:
    bridge = EngineBridge()
    progress: list[tuple[float, str]] = []
    loaded: list[bool] = []
    bridge.load_progress.connect(lambda value, phase: progress.append((value, phase)))
    bridge.scene_loaded.connect(lambda: loaded.append(True))
    bridge._client = _FakeClient(
        [
            {"event": "scene_load_progress", "progress": 0.45, "phase": "build"},
            {"event": "scene_load_progress", "progress": 1.0, "phase": "activate"},
            {"event": "scene_loaded", "path": "a.scene.json"},
        ]
    )
    bridge.pump()
    assert progress == [(0.45, "build"), (1.0, "activate")]
    assert loaded == [True]
