"""Демонстрация ядра MyEngine: сцена, компоненты, системы, события, ресурсы, цикл.

Запуск::

    .venv\\Scripts\\python.exe scripts\\run_demo.py

Скрипт не требует ни OpenGL, ни PySide6: он показывает, что игровая логика,
иерархия, ресурсы и цикл работают автономно (headless), как и задумано
архитектурой — рендер и редактор подключаются поверх.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # запуск без установки пакета
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):  # корректная кириллица в консоли Windows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from engine.core import (  # noqa: E402 - импорт после правки sys.path
    Component,
    Event,
    EventBus,
    GameLoop,
    ResourceManager,
    Scene,
    System,
    Vector3,
)


@dataclass(slots=True)
class PickupEvent(Event):
    """Событие подбора предмета: кто подобрал и сколько очков получил."""

    collector: str
    points: int


class PlayerController(Component):
    """Простое поведение игрока: движение по кругу и «подбор» предмета."""

    def __init__(self, speed: float = 90.0, pickup_at: float = 2.0) -> None:
        super().__init__()
        self.speed = speed
        self.pickup_at = pickup_at
        self.distance_travelled = 0.0
        self._picked_up = False

    def on_start(self) -> None:
        print(f"[{self.game_object.name}] старт: скорость {self.speed}°/с")

    def on_fixed_update(self, dt: float) -> None:
        turn = self.speed * dt
        self.transform.rotate(Vector3.up(), turn)
        self.distance_travelled += turn
        if not self._picked_up and self.distance_travelled >= self.pickup_at:
            self._picked_up = True
            scene = self.scene
            if scene is not None:
                scene.events.emit(PickupEvent(self.game_object.name, points=10))


class ScoreSystem(System):
    """Система-наблюдатель: начисляет очки за события подбора."""

    priority = 100

    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self.log = log
        self.score = 0
        self._subscription = None

    def on_added(self, scene: Scene) -> None:
        self._subscription = scene.events.subscribe(PickupEvent, self._on_pickup)

    def on_removed(self, scene: Scene) -> None:
        if self._subscription is not None:
            self._subscription.cancel()
            self._subscription = None

    def _on_pickup(self, event: PickupEvent) -> None:
        self.score += event.points
        self.log.append(f"{event.collector} подобрал предмет (+{event.points})")

    def on_update(self, scene: Scene, dt: float) -> None:
        """Проверяем инвариант иерархии: у всех объектов есть трансформ."""
        for game_object in scene.query(PlayerController):
            assert game_object.transform.is_valid, "трансформ должен быть жив"


def main() -> int:
    """Собрать маленький мир и прокрутить 180 кадров цикла."""
    print("=== 1. Ресурсы ===")
    resources = ResourceManager(roots=[ROOT / "assets"], name="Demo")
    mesh = resources.load("primitives/cube.obj")
    shader = resources.load("shaders/simple.frag")
    same_mesh = resources.load("primitives/cube.obj")  # повторно с диска не читается
    print(
        f"меш: {mesh.name}, вершин={mesh.vertex_count}, "
        f"треугольников={mesh.triangle_count}"
    )
    print(f"границы меша: {mesh.bounds[0]} .. {mesh.bounds[1]}")
    print(
        f"шейдер: {shader.name}, стадия={shader.stage.value}, "
        f"включений={len(shader.includes)}"
    )
    print(f"повторная загрузка вернула тот же объект: {mesh is same_mesh}")

    print("\n=== 2. Сцена ===")
    scene = Scene("Demo", event_bus=EventBus())
    log: list[str] = []
    score_system = scene.add_system(ScoreSystem(log))

    player = scene.create_game_object("Player", position=Vector3(0, 1, 0), tag="Hero")
    player.add_component(PlayerController, speed=180.0, pickup_at=90.0)
    child = scene.create_game_object(
        "Weapon", parent=player, position=Vector3(0, 0, 0.6)
    )
    assert child.transform.local_matrix.shape == (4, 4)  # прогрев кэша матрицы
    print(f"объектов: {len(scene)}, корней: {len(scene.root_objects)}")
    print(f"мировая позиция оружия: {child.transform.world_position}")

    print("\n=== 3. Игровой цикл ===")
    loop = GameLoop(scene, fixed_delta=1.0 / 60.0, target_fps=60, max_substeps=4)
    rendered: list[float] = []
    loop.add_render(lambda dt, alpha: rendered.append(alpha))
    frames = loop.run(max_frames=180)

    print(f"кадров: {frames}, фиксированных шагов: {loop.stats.fixed_steps}")
    print(f"игровое время: {loop.time.elapsed:.3f} с, FPS: {loop.stats.fps:.1f}")
    print(f"очки: {score_system.score}, события: {log}")
    print(f"позиция игрока после вращения: {player.transform.world_position}")
    print(f"позиция оружия: {child.transform.world_position}")

    print("\n=== 4. Статистика ядра ===")
    stats = scene.stats
    print(
        f"объекты={stats.game_objects}, компоненты={stats.components}, "
        f"активные={stats.active_components}, системы={stats.systems}"
    )
    resource_stats = resources.stats
    print(
        f"ресурсов в кэше={resource_stats.cached}, "
        f"попаданий в кэш={resource_stats.hits}, "
        f"промахов={resource_stats.misses}, "
        f"объём ~{resource_stats.bytes_estimated} байт"
    )

    print("\n=== 5. Завершение ===")
    player.destroy()
    scene.update(1.0 / 60.0)
    print(f"после уничтожения игрока объектов: {len(scene)}")
    resources.release(mesh)
    resources.release(shader)
    freed = resources.unload_unused()
    print(f"выгружено ресурсов: {freed}, в кэше: {len(resources)}")
    scene.shutdown()
    print("готово")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
