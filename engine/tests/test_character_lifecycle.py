"""Удаление, деактивация и «заморозка» ``CharacterControllerComponent``.

* Удаление компонента (``remove_component``) убирает kinematic-двойника из
  ``PhysicsWorld``: коллизии перестают срабатывать, объект остаётся в сцене.
* Деактивация объекта (``active_self=False``) делает то же; возврат активности
  создаёт двойника заново в текущей позе трансформа.
* Отключение компонента (``enabled=False``) - **заморозка**: двойник остаётся в
  мире, не двигается и не падает, методы движения - no-op. Подробно - в
  ``test_character_frozen.py``; здесь только то, что отличает её от удаления.

Подробности - ``docs/CORE.md`` §7.2.
"""

from __future__ import annotations

import pytest

from engine.core.math3d import Vector3
from engine.core.physics_components import (
    CharacterControllerComponent,
    ColliderComponent,
)
from engine.core.scene import Scene

DT = 1.0 / 60.0
REST_Y = 0.92


def _floor(scene: Scene):
    floor = scene.create_game_object(
        "Floor", position=Vector3(0, -0.5, 0), scale=Vector3(40, 1, 40)
    )
    floor.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    return floor


def _hero(scene: Scene, name: str = "Hero", x: float = 0.0, z: float = 0.0):
    go = scene.create_game_object(name, position=Vector3(x, REST_Y, z))
    comp = go.add_component(CharacterControllerComponent)
    return go, comp


def _run(scene: Scene, steps: int) -> None:
    for _ in range(steps):
        scene.fixed_update(DT)
        scene.update(DT)


def _world(scene: Scene):
    scene.physics.sync_structure()
    return scene.physics


def _blocked_at(scene: Scene, x: float, z: float) -> bool:
    """Есть ли в мире что-то, кроме пола, в шаре у точки ``(x, 0.92, z)``."""
    hits = _world(scene).overlap_sphere((x, REST_Y, z), 0.3)
    return any(go.name != "Floor" for go in hits)


# --- удаление ---------------------------------------------------------------

@pytest.mark.parametrize("by_type", [True, False], ids=["by_type", "by_instance"])
def test_remove_character_component_removes_twin(by_type):
    scene = Scene()
    _floor(scene)
    hero, comp = _hero(scene)
    _run(scene, 5)
    physics = _world(scene)
    handle = physics.handle_of(hero)
    assert handle is not None and _blocked_at(scene, 0.0, 0.0)

    assert hero.remove_component(
        CharacterControllerComponent if by_type else comp
    )
    _run(scene, 2)

    assert physics.handle_of(hero) is None
    assert handle not in physics.world
    assert physics.binding_count() == 1  # только пол
    assert not _blocked_at(scene, 0.0, 0.0)  # коллизии не срабатывают
    assert hero in scene  # объект остаётся в сцене
    assert hero.transform.world_position.y == pytest.approx(REST_Y, abs=1e-3)


def test_removed_character_does_not_block_other_character():
    scene = Scene()
    _floor(scene)
    hero, _ = _hero(scene, "Hero", x=0.0)
    walker, _ = _hero(scene, "Walker", x=-1.0)
    _run(scene, 5)

    def walk(steps: int) -> float:
        for _ in range(steps):
            scene.physics.update_character(walker, (3.0, 0.0, 0.0))
            scene.fixed_update(DT)
        return walker.transform.world_position.x

    assert walk(30) < 0.0  # упёрся в Hero (радиусы 0.4 + 0.4)

    hero.remove_component(CharacterControllerComponent)
    assert walk(60) > 1.0  # прошёл сквозь бывшего Hero


def test_removed_character_api_raises_clear_error():
    scene = Scene()
    _floor(scene)
    hero, _ = _hero(scene)
    _run(scene, 5)
    hero.remove_component(CharacterControllerComponent)

    with pytest.raises(RuntimeError, match="нет CharacterControllerComponent"):
        scene.physics.update_character(hero, (1.0, 0.0, 0.0))


def test_destroyed_character_object_leaves_world():
    scene = Scene()
    _floor(scene)
    hero, _ = _hero(scene)
    _run(scene, 5)
    handle = _world(scene).handle_of(hero)

    scene.destroy_game_object(hero)
    _run(scene, 2)

    assert handle not in scene.physics.world
    assert scene.physics.binding_count() == 1


# --- отключение и включение --------------------------------------------------

def test_disabled_character_component_is_frozen_in_the_world():
    scene = Scene()
    _floor(scene)
    hero, comp = _hero(scene)
    _run(scene, 5)
    handle = _world(scene).handle_of(hero)
    rest = hero.transform.world_position.to_tuple()

    comp.enabled = False
    _run(scene, 30)

    physics = _world(scene)
    assert physics.handle_of(hero) == handle  # тот же двойник, не пересоздан
    assert handle in physics.world
    assert _blocked_at(scene, 0.0, 0.0)  # заморозка - не удаление
    assert hero in scene
    assert hero.transform.world_position.to_tuple() == rest
    # метод движения - no-op, а не исключение (оно только у объекта без компонента)
    assert physics.update_character(hero, (1.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)
    assert hero.transform.world_position.to_tuple() == rest


def test_inactive_object_removes_twin_and_activation_restores_it():
    scene = Scene()
    _floor(scene)
    hero, _ = _hero(scene)
    _run(scene, 5)

    hero.active_self = False
    _run(scene, 5)
    assert _world(scene).handle_of(hero) is None
    assert not _blocked_at(scene, 0.0, 0.0)

    hero.active_self = True
    _run(scene, 5)
    assert _world(scene).handle_of(hero) is not None
    assert _blocked_at(scene, 0.0, 0.0)


def test_unfrozen_character_works_again():
    scene = Scene()
    _floor(scene)
    hero, comp = _hero(scene)
    _run(scene, 5)
    comp.enabled = False
    _run(scene, 10)

    comp.enabled = True
    _run(scene, 5)

    physics = _world(scene)
    assert physics.handle_of(hero) is not None
    assert physics.is_grounded(hero) is True
    assert _blocked_at(scene, 0.0, 0.0)
    for _ in range(30):
        physics.update_character(hero, (2.0, 0.0, 0.0))
        scene.fixed_update(DT)
    assert hero.transform.world_position.x == pytest.approx(1.0, abs=0.05)
    assert hero.transform.world_position.y == pytest.approx(REST_Y, abs=1e-3)


def test_unfrozen_character_is_at_the_transform_moved_while_frozen():
    scene = Scene()
    _floor(scene)
    hero, comp = _hero(scene)
    _run(scene, 5)
    comp.enabled = False
    _run(scene, 2)

    hero.transform.position = Vector3(6.0, REST_Y, 2.0)  # пока заморожен
    _run(scene, 2)
    assert _blocked_at(scene, 6.0, 2.0)  # замороженный двойник переехал сразу
    assert not _blocked_at(scene, 0.0, 0.0)
    comp.enabled = True
    _run(scene, 5)

    assert _blocked_at(scene, 6.0, 2.0)
    assert not _blocked_at(scene, 0.0, 0.0)
    assert scene.physics.is_grounded(hero) is True


def test_character_without_gravity_stays_in_the_air():
    scene = Scene()
    _floor(scene)
    hero = scene.create_game_object("Hero", position=Vector3(0, 5.0, 0))
    hero.add_component(CharacterControllerComponent, use_gravity=False)

    _run(scene, 60)

    assert _world(scene).handle_of(hero) is not None
    assert hero.transform.world_position.y == pytest.approx(5.0)
    assert any(
        go.name == "Hero" for go in _world(scene).overlap_sphere((0, 5.0, 0), 0.3)
    )
