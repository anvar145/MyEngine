"""Рисует демо физики плеером (headless) на кадрах 0, 60 и 600.

Результат — ``_work/physics_demo_<кадр>.png`` и
``_work/physics_demo_<кадр>_colliders.png`` (тот же кадр с каркасами
коллайдеров), плюс сводка ``_work/physics_demo_report.json`` (время прогона,
код возврата, наличие traceback).

Сцена должна быть создана ``scripts/seed_physics_demo.py``. Запуск:

    .venv\\Scripts\\python.exe scripts\\render_physics_demo.py [--frames 0 60 600]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "_work"
PROJECT = ROOT / "projects" / "PlayModeTest"
SCENE = "Assets/Scenes/physics_demo.scene.json"
#: В плеере кадр 0 — это один кадр после загрузки (рендер без шагов не бывает).
MIN_FRAMES = 1


def render(frames: int, colliders: bool) -> dict[str, object]:
    """Запустить плеер на ``frames`` кадров и сохранить скриншот."""
    suffix = "_colliders" if colliders else ""
    out = WORK / f"physics_demo_{frames}{suffix}.png"
    command = [
        sys.executable, "-m", "engine.player",
        "--project", str(PROJECT), "--scene", SCENE, "--headless",
        "--frames", str(max(frames, MIN_FRAMES)), "--screenshot", str(out),
    ]
    if colliders:
        command.append("--show-colliders")
    started = time.perf_counter()
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=False,
    )
    return {
        "frames": frames,
        "colliders": colliders,
        "png": str(out.relative_to(ROOT)),
        "exists": out.is_file(),
        "returncode": result.returncode,
        "seconds": round(time.perf_counter() - started, 2),
        "traceback": "Traceback" in (result.stdout + result.stderr),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 60, 600])
    args = parser.parse_args()
    WORK.mkdir(exist_ok=True)
    if not (PROJECT / SCENE).is_file():
        print("Нет physics_demo.scene.json: сначала scripts/seed_physics_demo.py")
        return 1
    report = [
        render(frames, colliders)
        for frames in args.frames
        for colliders in (False, True)
    ]
    (WORK / "physics_demo_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    failed = [r for r in report if r["returncode"] or r["traceback"] or not r["exists"]]
    for entry in report:
        print(entry)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
