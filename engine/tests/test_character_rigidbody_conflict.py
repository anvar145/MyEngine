"""CharacterController и RigidBody на одном объекте несовместимы.

Оба компонента управляют движением объекта (контроллер - кинематикой, тело -
решателем), поэтому вместе их работа непредсказуема. Такой объект не попадает
в ``PhysicsWorld``: в лог идёт ``ERROR`` с именем объекта и причиной, а вызовы
управления персонажем бросают ``RuntimeError``. Подробности - в
``_work/character_rigidbody_conflict.md`` и ``docs/CORE.md`` §7.2.
"""

from __future__ import annotations

import logging

import pytest

from engine.core.math3d import Vector3
from engine.core.physics_components import (
    CharacterControllerComponent,
    ColliderComponent,
    RigidBodyComponent,
)
from engine.core.scene import Scene
from engine.scripting.script_component import ScriptComponent

DT = 1.0 / 60.0
MESSAGE = "CharacterController и RigidBody несовместимы"
_NOOP = "def on_start(self):\n    pass\n"


def _floor(scene: Scene):
    floor = scene.create_game_object(
        "Floor", position=Vector3(0, -0.5, 0), scale=Vector3(40, 1, 40)
    )
    floor.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    return floor


def _conflict(scene: Scene, name: str = "Hero", y: float = 3.0):
    """Объект с контроллером, телом и коллайдером разом."""
    go = scene.create_game_object(name, position=Vector3(0, y, 0))
    go.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    body = go.add_component(RigidBodyComponent, body_type="dynamic", mass=5.0)
    controller = go.add_component(CharacterControllerComponent)
    return go, body, controller


def _errors(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]


def test_character_and_rigidbody_conflict(caplog):
    scene = Scene()
    _floor(scene)
    hero, _, _ = _conflict(scene)

    with caplog.at_level(logging.ERROR):
        scene.fixed_update(DT)

    errors = _errors(caplog)
    assert len(errors) == 1
    assert "Hero" in errors[0]
    assert MESSAGE in errors[0]
    # объект не зарегистрирован: ни двойника, ни тела, ни коллайдеров
    assert scene.physics.handle_of(hero) is None
    assert hero.id not in scene.physics._bindings
    assert scene.physics.binding_count() == 1  # только пол


def test_conflict_error_is_logged_once(caplog):
    scene = Scene()
    _floor(scene)
    _conflict(scene)

    with caplog.at_level(logging.ERROR):
        for _ in range(30):
            scene.fixed_update(DT)

    assert len(_errors(caplog)) == 1


def test_conflicting_object_does_not_move(caplog):
    scene = Scene()
    _floor(scene)
    hero, _, _ = _conflict(scene, y=3.0)

    with caplog.at_level(logging.ERROR):
        for _ in range(60):
            scene.fixed_update(DT)

    assert hero.transform.world_position.y == pytest.approx(3.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda p, go: p.move_character(go, (0.1, 0.0, 0.0)),
        lambda p, go: p.update_character(go, (1.0, 0.0, 0.0)),
        lambda p, go: p.move_input_character(go, (1.0, 0.0, 0.0)),
        lambda p, go: p.jump_character(go),
        lambda p, go: p.is_grounded(go),
    ],
    ids=["move", "simple_move", "move_input", "jump", "is_grounded"],
)
def test_conflict_character_api_raises(call, caplog):
    scene = Scene()
    _floor(scene)
    hero, _, _ = _conflict(scene)

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError) as info:
        call(scene.physics, hero)

    assert str(info.value) == MESSAGE


@pytest.mark.parametrize(
    "call",
    [
        lambda p, go: p.apply_impulse(go, (0.0, 1.0, 0.0)),
        lambda p, go: p.set_velocity(go, (0.0, 1.0, 0.0)),
        lambda p, go: p.get_velocity(go),
    ],
    ids=["apply_impulse", "set_velocity", "get_velocity"],
)
def test_conflict_rigidbody_api_names_the_cause(call, caplog):
    scene = Scene()
    _floor(scene)
    hero, _, _ = _conflict(scene)

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError) as info:
        call(scene.physics, hero)

    assert str(info.value) == MESSAGE


def test_conflict_script_facade_raises(caplog):
    scene = Scene()
    _floor(scene)
    hero, _, _ = _conflict(scene)
    script = hero.add_component(ScriptComponent, source=_NOOP)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match=MESSAGE):
            script.physics.simple_move((1.0, 0.0, 0.0))  # явная скорость
        with pytest.raises(RuntimeError, match=MESSAGE):
            script.physics.simple_move()  # по вводу, move_speed компонента
        with pytest.raises(RuntimeError, match=MESSAGE):
            script.physics.jump()
        with pytest.raises(RuntimeError, match=MESSAGE):
            script.physics.is_grounded()


def test_conflict_component_methods_raise(caplog):
    scene = Scene()
    _floor(scene)
    _, _, controller = _conflict(scene)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match=MESSAGE):
            controller.simple_move((1.0, 0.0, 0.0))
        with pytest.raises(RuntimeError, match=MESSAGE):
            _ = controller.is_grounded


def test_conflict_does_not_touch_other_characters(caplog):
    scene = Scene()
    _floor(scene)
    _conflict(scene, name="Bad", y=3.0)
    good = scene.create_game_object("Good", position=Vector3(5, 3.0, 0))
    good.add_component(CharacterControllerComponent)

    with caplog.at_level(logging.ERROR):
        for _ in range(90):
            scene.fixed_update(DT)

    assert scene.physics.is_grounded(good) is True
    assert good.transform.world_position.y == pytest.approx(0.92, abs=0.01)
    assert len(_errors(caplog)) == 1


def test_removing_rigidbody_resolves_conflict(caplog):
    scene = Scene()
    _floor(scene)
    hero, body, _ = _conflict(scene)
    with caplog.at_level(logging.ERROR):
        scene.fixed_update(DT)
    assert scene.physics.handle_of(hero) is None

    hero.remove_component(body)
    for _ in range(90):
        scene.fixed_update(DT)

    assert scene.physics.is_grounded(hero) is True
    assert scene.physics._bindings[hero.id].kind == "character"


def test_disabled_rigidbody_is_not_a_conflict(caplog):
    scene = Scene()
    _floor(scene)
    hero, body, _ = _conflict(scene)
    body.enabled = False

    with caplog.at_level(logging.ERROR):
        for _ in range(90):
            scene.fixed_update(DT)

    assert not _errors(caplog)
    assert scene.physics._bindings[hero.id].kind == "character"
    assert scene.physics.is_grounded(hero) is True


def test_disabled_character_leaves_a_plain_rigidbody(caplog):
    scene = Scene()
    _floor(scene)
    hero, _, controller = _conflict(scene)
    controller.enabled = False

    with caplog.at_level(logging.ERROR):
        for _ in range(30):
            scene.fixed_update(DT)

    assert not _errors(caplog)
    assert scene.physics._bindings[hero.id].kind == "dynamic"
    assert hero.transform.world_position.y < 3.0  # тело падает


def test_adding_rigidbody_to_registered_character_unregisters_it(caplog):
    scene = Scene()
    _floor(scene)
    hero = scene.create_game_object("Hero", position=Vector3(0, 0.92, 0))
    hero.add_component(CharacterControllerComponent)
    scene.fixed_update(DT)
    assert scene.physics.is_grounded(hero) is True

    hero.add_component(RigidBodyComponent, body_type="dynamic")
    with caplog.at_level(logging.ERROR):
        scene.fixed_update(DT)

    assert len(_errors(caplog)) == 1
    assert scene.physics.handle_of(hero) is None
    with pytest.raises(RuntimeError, match=MESSAGE):
        scene.physics.is_grounded(hero)


def test_conflict_error_is_logged_again_after_it_reappears(caplog):
    scene = Scene()
    _floor(scene)
    _, body, _ = _conflict(scene)
    with caplog.at_level(logging.ERROR):
        scene.fixed_update(DT)
        body.enabled = False
        scene.fixed_update(DT)
        body.enabled = True
        scene.fixed_update(DT)

    assert len(_errors(caplog)) == 2
