"""Микробенчмарк ядра: проверка, что горячие пути действительно быстрые.

Запуск::

    .venv\\Scripts\\python.exe tools\\benchmark_core.py

Скрипт измеряет то, что выполняется каждый кадр: обновление иерархии
трансформов, обход компонентов, запросы и доставку событий.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from engine.core import (  # noqa: E402 - импорт после правки sys.path
    Component,
    Event,
    EventBus,
    Scene,
    Vector3,
)

OBJECTS = 5_000
FRAMES = 60


class Ticker(Component):
    """Компонент, который что-то делает каждый кадр (имитация логики)."""

    def __init__(self) -> None:
        super().__init__()
        self.ticks = 0

    def on_update(self, dt: float) -> None:
        self.ticks += 1


@dataclass(slots=True)
class Ping(Event):
    """Тестовое событие для замера шины."""

    value: int


def measure(label: str, func) -> tuple[str, float]:
    """Замерить время выполнения функции.

    Returns:
        Пару «подпись, секунды».
    """
    started = time.perf_counter()
    func()
    elapsed = time.perf_counter() - started
    print(f"{label:<52} {elapsed * 1000:>9.2f} мс")
    return label, elapsed


def main() -> int:
    """Прогнать набор замеров и напечатать итоговую таблицу."""
    print(f"Объектов: {OBJECTS}, кадров: {FRAMES}\n")

    scene = Scene("Benchmark", transform_capacity=OBJECTS + 16)

    def build() -> None:
        root = scene.create_game_object("Root")
        for index in range(OBJECTS):
            parent = root if index % 4 == 0 else None
            game_object = scene.create_game_object(
                f"Object{index}",
                parent=parent,
                position=Vector3(index % 64, index // 64, 0.0),
            )
            game_object.add_component(Ticker)

    measure(f"создание {OBJECTS} объектов с компонентами", build)

    first = scene.objects[1]
    hierarchy_time = 0.0
    update_time = 0.0

    def run_frames() -> None:
        nonlocal hierarchy_time, update_time
        for _ in range(FRAMES):
            transform = first.transform
            transform.position = Vector3(transform.position.x + 0.01, 0.0, 0.0)

            started = time.perf_counter()
            scene.transforms.update_world_matrices()
            hierarchy_time += time.perf_counter() - started

            started = time.perf_counter()
            scene.update(1.0 / 60.0)
            update_time += time.perf_counter() - started

    measure(
        f"{FRAMES} кадров scene.update() (логика {OBJECTS} компонентов)", run_frames
    )
    print(
        f"  из них: пересчёт иерархии {hierarchy_time * 1000:.2f} мс "
        f"({hierarchy_time / FRAMES * 1e6:.0f} мкс/кадр), "
        f"полный кадр {update_time / FRAMES * 1000:.2f} мс"
    )

    query_count = 200

    def run_queries() -> None:
        for _ in range(query_count):
            scene.query(Ticker)

    _label, elapsed = measure(
        f"{query_count} запросов scene.query(Ticker) по кэшу", run_queries
    )
    print(f"  то есть {elapsed / query_count * 1e6:.1f} мкс на запрос")

    bus = EventBus()
    counter = {"events": 0}

    def on_ping(event: Ping) -> None:
        counter["events"] += 1

    bus.subscribe(Ping, on_ping)
    emits = 100_000

    def run_events() -> None:
        for index in range(emits):
            bus.emit(Ping(index))

    measure(f"{emits} событий через шину (1 подписчик)", run_events)

    def run_destroy() -> None:
        for game_object in list(scene.objects)[:1000]:
            game_object.destroy()
        scene.update(1.0 / 60.0)

    measure("уничтожение 1000 объектов (отложенное)", run_destroy)

    stats = scene.stats
    print(
        f"\nОсталось объектов: {stats.game_objects}, "
        f"трансформов: {stats.transforms_alive}/{stats.transforms_capacity}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
