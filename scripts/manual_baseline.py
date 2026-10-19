"""P2: автоматическая часть baseline сцен PlayModeTest → ``_work/manual_baseline.md``.

Каждую из пяти сцен плеер гоняет headless 120 кадров со скриншотом, с оверлеями
F3 и ``--stats-json`` / ``--dump-state``. Скрипт собирает код возврата, наличие
traceback/ERROR в журнале, CPU-время и скриншоты (``_work/baseline/``) в таблицу.

Ручные пункты (открыть в редакторе, правка в инспекторе, Play Mode в окне)
остаются чекбоксами в отчёте: их отмечает человек на Windows с живым окном.
Повторный запуск сохраняет отметки ``[x]`` из прежнего отчёта.

    .venv\\Scripts\\python.exe scripts\\manual_baseline.py [--frames 120]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "_work"
SHOTS = WORK / "baseline"
REPORT = WORK / "manual_baseline.md"
PROJECT = ROOT / "projects" / "PlayModeTest"
SCENES = ("main", "main_menu", "physics_demo", "mesh_level", "realistic")
MANUAL = (
    "открывается в редакторе без ошибок в консоли",
    "объект выбирается кликом во вьюпорте (в т. ч. вложенный)",
    "поле в инспекторе правится, Ctrl+Z/Ctrl+Y работают",
    "Play Mode (F5) запускается, Shift+F5 останавливает",
    "в Play Mode F2 (коллайдеры) и F3 (оверлеи отладки) переключаются",
    "скриншот Play Mode выглядит как в headless-baseline",
)


def run_scene(scene: str, frames: int) -> dict[str, object]:
    """Прогнать сцену и вернуть строку отчёта."""
    SHOTS.mkdir(parents=True, exist_ok=True)
    shot = SHOTS / f"{scene}.png"
    stats = SHOTS / f"{scene}.stats.json"
    state = SHOTS / f"{scene}.state.json"
    command = [
        sys.executable, "-m", "engine.player", "--project", str(PROJECT),
        "--scene", f"Assets/Scenes/{scene}.scene.json", "--headless", "--no-audio",
        "--frames", str(frames), "--debug-overlays", "--screenshot", str(shot),
        "--stats-json", str(stats), "--dump-state", str(state),
    ]
    started = time.perf_counter()
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=False,
    )
    log = result.stdout + result.stderr
    errors = [ln for ln in log.splitlines() if "ERROR" in ln.split(":")[0]]
    row: dict[str, object] = {
        "scene": scene,
        "returncode": result.returncode,
        "traceback": "Traceback" in log,
        "errors": len(errors),
        "first_error": errors[0][:160] if errors else "",
        "screenshot": shot.is_file(),
        "seconds": round(time.perf_counter() - started, 1),
    }
    if stats.is_file():
        data = json.loads(stats.read_text(encoding="utf-8"))
        row["cpu_ms_avg"] = round(float(data.get("cpu_ms_avg", 0.0)), 2)
        row["draw_calls"] = int(data.get("draw_calls", 0))
        row["awake_max"] = data.get("physics_awake_max")
        row["driver"] = str(data.get("driver", ""))
    return row


def _previous_marks() -> dict[str, bool]:
    """Отметки ``[x]`` из прежнего отчёта: ``"сцена|пункт" -> True``."""
    if not REPORT.is_file():
        return {}
    marks: dict[str, bool] = {}
    scene = ""
    for line in REPORT.read_text(encoding="utf-8").splitlines():
        heading = re.match(r"### (\w+)", line)
        if heading:
            scene = heading.group(1)
        item = re.match(r"- \[(.)\] (.+)", line)
        if item and scene:
            marks[f"{scene}|{item.group(2)}"] = item.group(1) == "x"
    return marks


def render(rows: list[dict[str, object]], frames: int) -> str:
    """Собрать markdown отчёта."""
    marks = _previous_marks()
    ok = lambda r: not r["returncode"] and not r["traceback"] and r["screenshot"]  # noqa: E731
    lines = [
        "# Manual baseline: сцены PlayModeTest",
        "",
        f"Автоматическая часть: `scripts/manual_baseline.py`, {frames} кадров headless "
        "с `--debug-overlays`. Скриншоты лежат в `_work/baseline/`.",
        "Ручная часть (чекбоксы ниже) заполняется человеком в Windows с окном.",
        "После каждого среза B прогон повторяется и сравнивается с этим файлом.",
        "",
        "## Автоматически",
        "",
        "| Сцена | Статус | Ошибок в логе | CPU мс (ср.) | Draw calls | Тел (макс.) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['scene']} | {'OK' if ok(r) else 'СБОЙ'} | {r['errors']} | "
            f"{r.get('cpu_ms_avg', '-')} | {r.get('draw_calls', '-')} | "
            f"{r.get('awake_max', '-')} |"
        )
    drivers = {str(r.get("driver", "")) for r in rows if r.get("driver")}
    lines += ["", f"Драйвер: {'; '.join(sorted(drivers)) or 'нет данных'}.",
              "Время на программном рендере (llvmpipe) не сравнимо с HD 520: "
              "цифры нужны для сравнения между срезами на одной машине.", ""]
    problems = [r for r in rows if r["first_error"] or not ok(r)]
    if problems:
        lines.append("Замечания:")
        for r in problems:
            lines.append(f"- {r['scene']}: код {r['returncode']}, {r['first_error']}")
        lines.append("")
    lines += ["## Вручную", ""]
    for scene in SCENES:
        lines += [f"### {scene}", ""]
        for item in MANUAL:
            mark = "x" if marks.get(f"{scene}|{item}") else " "
            lines.append(f"- [{mark}] {item}")
        lines.append("")
    lines += ["## Что не работает", "", "(заполняется по итогам ручного прохода)", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=120)
    args = parser.parse_args()
    WORK.mkdir(exist_ok=True)
    rows = [run_scene(scene, args.frames) for scene in SCENES]
    REPORT.write_text(render(rows, args.frames), encoding="utf-8")
    print(REPORT)
    return 1 if any(r["returncode"] or r["traceback"] for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
