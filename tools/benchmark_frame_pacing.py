"""Стенд frame pacing: замкнутый цикл плеера, эмуляция vsync, трасса кадров.

Плеер запускается в headless-режиме с ``--closed-loop`` (реальное dt кадра и
бюджет физики, как в окне) и ``--present-hz`` (кадр ждёт границы vsync). Персонаж
идёт сквозь узкие щели (``tools/_frame_pacing_scenes.py``). Для каждого сценария
считаются кадры, физика и скорость игрового времени (доля реального).

Запуск:
    python tools/benchmark_frame_pacing.py [--only gap_100,gap_slalom]
        [--frames 400] [--hz 60] [--runs 1] [--budget-ms 9] [--out ИМЯ]

Гигиена замеров: перед стартом проверяется загрузка CPU, прогоны идут строго по
одному, трассы лежат в ``_work/frame_pacing/``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
OUT = ROOT / "_work" / "frame_pacing"
WARMUP = 60
FIXED_DELTA_MS = 1000.0 / 60.0


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


def summarize(frames: list[dict[str, float]], hz: float) -> dict[str, float]:
    """Свести трассу кадров в метрики (без прогрева)."""
    rows = frames[WARMUP:]
    tick = [r["tick_ms"] for r in rows]
    wall = [r["delta_ms"] for r in rows]
    fixed = [r["fixed_ms"] for r in rows if r["fixed_steps"]]
    steps = [r["fixed_steps"] for r in rows]
    window = max(1, int(hz))
    worst = 1.0
    for i in range(0, max(1, len(rows) - window), window // 4 or 1):
        w = sum(wall[i : i + window])
        s = sum(steps[i : i + window]) * FIXED_DELTA_MS
        if w > 0:
            worst = min(worst, s / w)
    period = 1000.0 / hz
    return {
        "frames": len(rows),
        "wall_p50": statistics.median(wall),
        "wall_p95": _percentile(wall, 0.95),
        "wall_max": max(wall),
        "late_frames": sum(1 for v in wall if v > period * 1.5),
        "tick_p50": statistics.median(tick),
        "tick_p95": _percentile(tick, 0.95),
        "tick_max": max(tick),
        "step_p50": statistics.median(fixed) if fixed else 0.0,
        "step_p95": _percentile(fixed, 0.95) if fixed else 0.0,
        "step_max": max(fixed) if fixed else 0.0,
        "budget_stops": sum(int(r["budget_stop"]) for r in rows),
        "sim_rate": sum(steps) * FIXED_DELTA_MS / max(1e-9, sum(wall)),
        "sim_rate_worst_1s": worst,
        "overrun_frames": rows[-1]["overrun"] - rows[0]["overrun"],
    }


def run_scene(project: Path, scene: str, args: argparse.Namespace) -> dict[str, float]:
    """Один прогон сцены в дочернем процессе."""
    trace = OUT / "runs" / f"{scene}.json"
    trace.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        "-m",
        "engine.player",
        "--project",
        str(project),
        "--scene",
        f"{scene}.scene.json",
        "--headless",
        "--closed-loop",
        "--present-hz",
        str(args.hz),
        "--hold-keys",
        "w",
        "--frames",
        str(args.frames),
        "--frame-trace",
        str(trace),
    ]
    subprocess.run(cmd, cwd=ROOT, capture_output=True, check=False)
    data = json.loads(trace.read_text(encoding="utf-8"))
    return summarize(data["frames"], args.hz)


def _cpu_load() -> float:
    out = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Processor | "
            "Measure-Object LoadPercentage -Average).Average",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return float(out or 0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="")
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--out", default="pacing")
    parser.add_argument("--project", default=None, help="проект вместо стендового")
    parser.add_argument("--scenes", default="", help="сцены проекта через запятую")
    args = parser.parse_args()

    load = _cpu_load()
    print(f"загрузка CPU перед стартом: {load:.0f} %")
    if load > 25.0:
        print("ВНИМАНИЕ: машина занята, цифры будут завышены")

    if args.project:
        project = Path(args.project)
        names = [s for s in args.scenes.split(",") if s]
    else:
        from _frame_pacing_scenes import generate

        project = generate()
        names = sorted(
            p.name[: -len(".scene.json")]
            for p in (project / "Assets" / "Scenes").glob("*.scene.json")
        )
    only = {s for s in args.only.split(",") if s}
    results: dict[str, dict[str, float]] = {}
    head = ("сцена", "wall50", "wall95", "wallмакс", "поздних", "шаг50", "шаг95",
            "шагмакс", "стопов", "sim", "sim1с")
    print(" ".join(f"{h:>10}" for h in head))
    for name in names:
        if only and name not in only:
            continue
        runs = [run_scene(project, name, args) for _ in range(args.runs)]
        res = {k: statistics.median(r[k] for r in runs) for k in runs[0]}
        results[name] = res
        print(
            f"{name:>10} {res['wall_p50']:10.1f} {res['wall_p95']:10.1f} "
            f"{res['wall_max']:10.1f} {res['late_frames']:10.0f} "
            f"{res['step_p50']:10.1f} {res['step_p95']:10.1f} {res['step_max']:10.1f} "
            f"{res['budget_stops']:10.0f} {res['sim_rate']:10.2f} "
            f"{res['sim_rate_worst_1s']:10.2f}",
            flush=True,
        )
    target = OUT / f"{args.out}.json"
    target.write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
