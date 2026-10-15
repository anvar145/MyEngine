"""Тесты PlayModeService на поддельном плеере."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from editor.context import EditorContext  # noqa: E402
from editor.services.play_mode_service import PlayModeService  # noqa: E402

FAKE = str(Path(__file__).with_name("fake_player.py"))


def _wait(qapp, predicate, timeout: float = 10.0) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def service(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_PLAYER_MODE", "normal")
    context = EditorContext(project_root=tmp_path, workspace_root=tmp_path)
    play = PlayModeService(context)
    play.player_command = lambda: [sys.executable, FAKE]
    yield play
    play.stop()
    _wait(qapp, lambda: not play.is_active, 5.0)


def test_play_mode_service_states(qapp, service, tmp_path) -> None:
    """editing → playing → paused → playing → stopping → editing."""
    states: list[str] = []
    service.state_changed.connect(states.append)
    stats: list[dict] = []
    service.stats_received.connect(stats.append)
    logs: list[dict] = []
    service.log_received.connect(logs.append)

    assert service.state == "editing"
    assert service.enter_play()
    assert not service.enter_play(), "повторный запуск не допускается"
    assert _wait(qapp, lambda: service.state == "playing")
    assert service.pid > 0

    assert _wait(qapp, lambda: stats and logs), "статистика и журнал доходят"
    assert stats[0]["tri"] == 1934

    service.pause()
    assert service.state == "paused"
    service.resume()
    assert service.state == "playing"
    service.step()
    assert service.state == "paused"
    service.resume()

    picked: list[int] = []
    service.selection_received.connect(picked.append)
    service.pick(10, 10)
    assert _wait(qapp, lambda: picked == [7])

    service.stop()
    assert service.state == "stopping"
    assert _wait(qapp, lambda: service.state == "editing")
    assert states[0] == "playing" and states[-1] == "editing"
    assert "stopping" in states
    assert not list((tmp_path / "Library" / "PlayMode").glob("snapshot_*.json"))


def test_player_crash_returns_to_editing(qapp, service, monkeypatch) -> None:
    """Падение плеера: сигнал crashed с traceback и возврат в editing."""
    monkeypatch.setenv("FAKE_PLAYER_MODE", "crash")
    crashes: list[str] = []
    service.crashed.connect(crashes.append)
    assert service.enter_play()
    assert _wait(qapp, lambda: crashes and service.state == "editing")
    assert "fake crash" in crashes[0]


def test_killed_player_is_detected(qapp, service) -> None:
    """Плеер убит снаружи: редактор не виснет и возвращается в editing."""
    crashes: list[str] = []
    service.crashed.connect(crashes.append)
    assert service.enter_play()
    assert _wait(qapp, lambda: service.state == "playing")
    pid = service.pid
    assert pid > 0
    os.kill(pid, 9)
    assert _wait(qapp, lambda: service.state == "editing", 8.0)
    assert crashes, "об аварийном завершении сообщено"


def test_hung_player_is_killed_after_timeout(qapp, service, monkeypatch) -> None:
    """Плеер игнорирует stop: через 2 с процесс завершается принудительно."""
    monkeypatch.setenv("FAKE_PLAYER_MODE", "hang")
    assert service.enter_play()
    assert _wait(qapp, lambda: service.state == "playing")
    service.stop()
    assert _wait(qapp, lambda: service.state == "editing", 6.0)
