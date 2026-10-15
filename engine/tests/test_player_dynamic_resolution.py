"""Динамическое разрешение в плеере (headless, HDR): интеграционный тест A10b."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "projects" / "PlayModeTest"


def _run(*extra: str) -> str:
    command = [
        sys.executable, "-m", "engine.player", "--project", str(PROJECT),
        "--headless", "--frames", "5", "--hdr", "on", *extra,
    ]  # fmt: skip
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    done = subprocess.run(
        command, env=env, cwd=ROOT, capture_output=True, text=True,
        errors="replace", timeout=180,
    )  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout + done.stderr


def test_medium_starts_with_dynamic_resolution() -> None:
    log = _run("--quality", "medium")
    assert "render_scale 0.75 (dynamic=True, 0.50..1.00)" in log


def test_no_dynamic_res_flag_disables_it() -> None:
    log = _run("--quality", "medium", "--no-dynamic-res")
    assert "dynamic=False" in log


def test_high_has_no_dynamic_resolution() -> None:
    log = _run("--quality", "high")
    assert "render_scale 1.00 (dynamic=False" in log
