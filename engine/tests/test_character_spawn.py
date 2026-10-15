"""Спавн персонажа: двойник стартует в позиции Transform, grounded честный.

Регрессия бага «персонаж висит над землёй» (``_work/character_elevation_bug.md``):
без вызова ``simple_move`` из скрипта персонаж, заспавненный в воздухе, не падал,
а ``is_grounded()`` у стоящего на полу был ``False``, пока его не двигали, -
двойника создавали раньше статической геометрии, и проба земли в конструкторе
не видела пола.
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
HEIGHT = 1.8
SKIN = 0.02
REST_Y = HEIGHT / 2.0 + SKIN


def _floor(scene: Scene):
    floor = scene.create_game_object(
        "Floor", position=Vector3(0, -0.5, 0), scale=Vector3(40, 1, 40)
    )
    floor.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    return floor


def _hero(scene: Scene, y: float, **cc):
    go = scene.create_game_object("Hero", position=Vector3(0, y, 0))
    go.add_component(CharacterControllerComponent, **cc)
    return go


def _twin(scene: Scene, go):
    """Двойник персонажа (kinematic-тело контроллера) в мире физики."""
    scene.physics.sync_structure()
    return scene.physics._bindings[go.id].character


def _run(scene: Scene, steps: int) -> None:
    for _ in range(steps):
        scene.fixed_update(DT)
        scene.update(DT)


def test_character_spawn_above_ground():
    scene = Scene()
    _floor(scene)
    hero = _hero(scene, 5.0)

    # на старте под ногами пусто: не grounded, двойник ровно там, где Transform
    twin = _twin(scene, hero)
    assert scene.physics.is_grounded(hero) is False
    assert twin.position[1] == pytest.approx(5.0)

    # первый кадр: персонаж уже падает, но не «телепортирован» к земле
    _run(scene, 1)
    assert 4.9 < hero.transform.world_position.y < 5.0
    assert scene.physics.is_grounded(hero) is False

    # через секунду от падения с 4 м (t ≈ 0.9 с) он стоит на земле
    _run(scene, 59)
    assert scene.physics.is_grounded(hero) is True
    assert hero.transform.world_position.y == pytest.approx(REST_Y, abs=0.01)


def test_character_spawn_grounded_on_floor():
    scene = Scene()
    _floor(scene)
    hero = _hero(scene, REST_Y)

    # grounded уже в момент создания двойника - не после первого движения
    _twin(scene, hero)
    assert scene.physics.is_grounded(hero) is True

    _run(scene, 60)
    assert scene.physics.is_grounded(hero) is True
    assert hero.transform.world_position.y == pytest.approx(REST_Y, abs=1e-3)


def test_character_spawn_grounded_does_not_depend_on_creation_order():
    # пол создан ПОСЛЕ персонажа: на grounded порядок объектов не влияет
    scene = Scene()
    hero = _hero(scene, REST_Y)
    _floor(scene)

    _twin(scene, hero)
    assert scene.physics.is_grounded(hero) is True


def test_character_twin_spawns_at_transform_world_position():
    scene = Scene()
    _floor(scene)
    parent = scene.create_game_object("Parent", position=Vector3(2.0, 4.0, -1.0))
    hero = scene.create_game_object("Hero", position=Vector3(0.0, 1.0, 0.0))
    hero.set_parent(parent, keep_world=False)
    hero.add_component(CharacterControllerComponent)

    twin = _twin(scene, hero)

    assert twin.position == pytest.approx([2.0, 5.0, -1.0])
    assert hero.transform.world_position.y == pytest.approx(5.0)
