"""Режим «заморожен»: ``CharacterControllerComponent.enabled = False``.

Заморозка - третий уровень между «активен» и «удалён»:

* **активен** (``enabled=True``): двойник в мире, движение и гравитация;
* **заморожен** (``enabled=False``): двойник остаётся в мире (коллизии, лучи,
  overlap, солвер как у kinematic-тела), но не двигается и не падает; методы
  движения - no-op без исключений;
* **удалён** (``active_self=False`` объекта, ``remove_component``): двойника нет.

Переключение между «активен» и «заморожен» не пересоздаёт двойник (handle тот же).
Подробности - ``docs/CORE.md`` §7.2.
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
from engine.scene_io import scene_from_dict, scene_to_dict
from engine.scripting.script_component import ScriptComponent

DT = 1.0 / 60.0
REST_Y = 0.92
ZERO = (0.0, 0.0, 0.0)
_NOOP = "def on_start(self):\n    pass\n"


def _floor(scene: Scene):
    floor = scene.create_game_object(
        "Floor", position=Vector3(0, -0.5, 0), scale=Vector3(40, 1, 40)
    )
    floor.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    return floor


def _hero(scene: Scene, name: str = "Hero", x: float = 0.0, y: float = REST_Y,
          z: float = 0.0, **cc):
    go = scene.create_game_object(name, position=Vector3(x, y, z))
    comp = go.add_component(CharacterControllerComponent, **cc)
    return go, comp


def _run(scene: Scene, steps: int) -> None:
    for _ in range(steps):
        scene.fixed_update(DT)
        scene.update(DT)


def _world(scene: Scene):
    scene.physics.sync_structure()
    return scene.physics


def _blocked_at(scene: Scene, x: float, z: float, y: float = REST_Y) -> bool:
    """Есть ли в мире что-то, кроме пола, в шаре у точки ``(x, y, z)``."""
    hits = _world(scene).overlap_sphere((x, y, z), 0.3)
    return any(go.name != "Floor" for go in hits)


def _standing_frozen(scene: Scene | None = None):
    """Персонаж, постоявший на земле и затем замороженный."""
    scene = scene or Scene()
    _floor(scene)
    hero, comp = _hero(scene)
    _run(scene, 5)
    comp.enabled = False
    return scene, hero, comp


# --- двойник остаётся в мире ---------------------------------------------------

def test_toggle_keeps_the_same_twin():
    scene = Scene()
    _floor(scene)
    hero, comp = _hero(scene)
    _run(scene, 5)
    physics = _world(scene)
    handle = physics.handle_of(hero)
    count = physics.binding_count()

    comp.enabled = False
    _run(scene, 5)
    assert physics.handle_of(hero) == handle
    assert handle in physics.world
    assert physics.binding_count() == count

    comp.enabled = True
    _run(scene, 5)
    assert physics.handle_of(hero) == handle
    assert physics.binding_count() == count


def test_frozen_twin_is_found_by_overlap_and_raycast():
    scene, hero, _ = _standing_frozen()
    _run(scene, 5)

    assert _blocked_at(scene, 0.0, 0.0)
    hit = scene.physics.raycast((0.0, 5.0, 0.0), (0.0, -1.0, 0.0), 10.0)
    assert hit is not None and hit["object"] is hero
    assert hit["point"][1] == pytest.approx(REST_Y + 0.9, abs=0.05)


def test_frozen_twin_blocks_another_character():
    scene = Scene()
    _floor(scene)
    hero, comp = _hero(scene, "Hero", x=0.0)
    walker, _ = _hero(scene, "Walker", x=-1.0)
    _run(scene, 5)
    comp.enabled = False

    for _ in range(60):
        scene.physics.update_character(walker, (3.0, 0.0, 0.0))
        scene.fixed_update(DT)

    assert walker.transform.world_position.x < 0.0  # упёрся (радиусы 0.4 + 0.4)
    assert hero.transform.world_position.x == pytest.approx(0.0, abs=1e-6)


def test_dynamic_body_rests_on_frozen_character():
    scene, hero, _ = _standing_frozen()
    rest_y = hero.transform.world_position.y
    box = scene.create_game_object("Box", position=Vector3(0, 4.0, 0))
    box.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    box.add_component(RigidBodyComponent, body_type="dynamic", mass=1.0)

    _run(scene, 180)

    # голова персонажа на y = 0.92 + 0.9 = 1.82, коробка сидит сверху
    assert box.transform.world_position.y == pytest.approx(1.82 + 0.5, abs=0.15)
    assert hero.transform.world_position.y == pytest.approx(rest_y, abs=1e-9)


# --- не двигается ---------------------------------------------------------------

def test_character_created_frozen_stays_in_the_air():
    scene = Scene()
    _floor(scene)
    hero, comp = _hero(scene, y=5.0)
    comp.enabled = False

    _run(scene, 60)

    physics = _world(scene)
    assert physics.handle_of(hero) is not None
    assert hero.transform.world_position.y == pytest.approx(5.0, abs=1e-9)
    assert any(go.name == "Hero" for go in physics.overlap_sphere((0, 5.0, 0), 0.3))
    assert physics.is_grounded(hero) is False


def test_freezing_a_falling_character_stops_it_in_place():
    scene = Scene()
    _floor(scene)
    hero, comp = _hero(scene, y=5.0)
    _run(scene, 10)
    comp.enabled = False
    _run(scene, 1)
    y = hero.transform.world_position.y
    assert y < 5.0

    _run(scene, 60)

    assert hero.transform.world_position.y == pytest.approx(y, abs=1e-9)


def test_freezing_zeroes_velocities():
    scene = Scene()
    _floor(scene)
    hero, comp = _hero(scene, y=5.0)
    _run(scene, 20)
    ctrl = _world(scene)._bindings[hero.id].character
    assert ctrl.vertical_velocity < -1.0

    comp.enabled = False
    _world(scene)  # применить переключение

    assert ctrl.vertical_velocity == 0.0
    assert not ctrl.horizontal_velocity.any()


def test_frozen_motion_api_is_a_noop():
    scene, hero, comp = _standing_frozen()
    physics = scene.physics
    before = hero.transform.world_position.to_tuple()

    assert physics.update_character(hero, (3.0, 0.0, 0.0)) == ZERO
    assert physics.move_character(hero, (1.0, 0.0, 0.0)) == ZERO
    assert physics.move_input_character(hero, (1.0, 0.0, 0.0), sprint=True) == ZERO
    assert physics.jump_character(hero) is False
    assert comp.simple_move((2.0, 0.0, 0.0)) == ZERO
    assert comp.move((0.0, 1.0, 0.0)) == ZERO
    assert comp.move_input((0.0, 0.0, 1.0)) == ZERO
    assert comp.jump() is False
    _run(scene, 5)

    assert hero.transform.world_position.to_tuple() == before


def test_frozen_is_grounded_is_the_last_value():
    scene = Scene()
    floor = _floor(scene)
    hero, comp = _hero(scene)
    _run(scene, 5)
    assert scene.physics.is_grounded(hero) is True
    comp.enabled = False

    floor.active_self = False  # земля ушла из-под ног
    _run(scene, 3)

    assert scene.physics.is_grounded(hero) is True  # замороженный не пересчитывает
    comp.enabled = True
    # размораживание сразу пересчитывает землю (без физического шага)
    assert scene.physics.is_grounded(hero) is False


# --- возобновление ---------------------------------------------------------------

def test_unfreezing_resumes_from_the_current_pose():
    scene = Scene()
    _floor(scene)
    hero, comp = _hero(scene, y=5.0)
    comp.enabled = False
    _run(scene, 30)
    assert hero.transform.world_position.y == pytest.approx(5.0, abs=1e-9)

    comp.enabled = True
    _run(scene, 20)
    mid = hero.transform.world_position.y
    assert REST_Y < mid < 5.0  # падает с места заморозки, а не с нуля

    _run(scene, 120)
    assert hero.transform.world_position.y == pytest.approx(REST_Y, abs=0.01)
    assert _world(scene).is_grounded(hero) is True


def test_unfreezing_does_not_inherit_the_old_fall_speed():
    scene = Scene()
    _floor(scene)
    hero, comp = _hero(scene, y=5.0)
    _run(scene, 20)
    comp.enabled = False
    _run(scene, 30)
    comp.enabled = True
    y0 = hero.transform.world_position.y

    _run(scene, 1)

    # с нуля за один шаг: g * dt^2 ~ 0.0027 м; со старой скоростью было бы ~0.05 м
    assert y0 - hero.transform.world_position.y < 0.01


def test_frozen_walk_resumes_after_unfreezing():
    scene, hero, comp = _standing_frozen()
    comp.enabled = True
    for _ in range(30):
        scene.physics.update_character(hero, (2.0, 0.0, 0.0))
        scene.fixed_update(DT)

    assert hero.transform.world_position.x == pytest.approx(1.0, abs=0.05)
    assert hero.transform.world_position.y == pytest.approx(REST_Y, abs=1e-3)


def test_frozen_twin_follows_the_transform():
    scene, hero, _ = _standing_frozen()
    handle = _world(scene).handle_of(hero)

    hero.transform.position = Vector3(6.0, REST_Y, 2.0)  # перенос из инспектора
    _run(scene, 5)

    assert _blocked_at(scene, 6.0, 2.0)
    assert not _blocked_at(scene, 0.0, 0.0)
    assert _world(scene).handle_of(hero) == handle
    assert hero.transform.world_position.to_tuple() == pytest.approx(
        (6.0, REST_Y, 2.0), abs=1e-6
    )


def test_resize_while_frozen_keeps_it_frozen():
    scene = Scene()
    _floor(scene)
    hero, comp = _hero(scene, y=5.0)
    comp.enabled = False
    _run(scene, 2)

    hero.transform.scale = Vector3(1.0, 2.0, 1.0)
    _run(scene, 30)

    binding = _world(scene)._bindings[hero.id]
    assert binding.frozen is True
    assert binding.size[1] == pytest.approx(3.6)
    assert _world(scene).update_character(hero, (1.0, 0.0, 0.0)) == ZERO


# --- удаление и деактивация объекта ---------------------------------------------

def test_frozen_twin_leaves_world_when_object_is_deactivated():
    scene, hero, _ = _standing_frozen()
    handle = _world(scene).handle_of(hero)

    hero.active_self = False
    _run(scene, 3)
    assert _world(scene).handle_of(hero) is None
    assert handle not in scene.physics.world
    assert not _blocked_at(scene, 0.0, 0.0)

    hero.active_self = True  # компонент по-прежнему выключен: возвращается замороженным
    _run(scene, 3)
    assert _world(scene).handle_of(hero) is not None
    assert _blocked_at(scene, 0.0, 0.0)
    assert _world(scene).update_character(hero, (1.0, 0.0, 0.0)) == ZERO


def test_deactivated_object_with_enabled_component_keeps_raising():
    """``active_self=False`` - не заморозка: поведение API прежнее (``RuntimeError``)."""
    scene = Scene()
    _floor(scene)
    hero, _ = _hero(scene)
    _run(scene, 5)
    hero.active_self = False
    _run(scene, 3)

    with pytest.raises(RuntimeError, match="нет CharacterControllerComponent"):
        scene.physics.update_character(hero, (1.0, 0.0, 0.0))
    with pytest.raises(RuntimeError, match="нет CharacterControllerComponent"):
        scene.physics.jump_character(hero)


@pytest.mark.parametrize("by_type", [True, False], ids=["by_type", "by_instance"])
def test_removing_a_frozen_component_removes_the_twin(by_type):
    scene, hero, comp = _standing_frozen()
    handle = _world(scene).handle_of(hero)
    assert handle is not None

    assert hero.remove_component(CharacterControllerComponent if by_type else comp)
    _run(scene, 3)

    assert _world(scene).handle_of(hero) is None
    assert handle not in scene.physics.world
    assert not _blocked_at(scene, 0.0, 0.0)
    with pytest.raises(RuntimeError, match="нет CharacterControllerComponent"):
        scene.physics.update_character(hero, (1.0, 0.0, 0.0))


# --- CharacterController и RigidBody ---------------------------------------------

def _body_hero(scene: Scene, y: float = 3.0, *, body_enabled: bool = True,
               character_enabled: bool = True):
    go = scene.create_game_object("Hero", position=Vector3(0, y, 0))
    go.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    body = go.add_component(RigidBodyComponent, body_type="dynamic", mass=5.0)
    controller = go.add_component(CharacterControllerComponent)
    body.enabled = body_enabled
    controller.enabled = character_enabled
    return go, body, controller


def test_disabled_character_with_active_rigidbody_has_no_twin():
    scene = Scene()
    _floor(scene)
    hero, _, controller = _body_hero(scene, character_enabled=False)

    _run(scene, 30)

    physics = _world(scene)
    assert physics._bindings[hero.id].kind == "dynamic"
    assert physics.character_handles() == frozenset()
    assert hero.transform.world_position.y < 3.0  # тело падает
    assert physics.update_character(hero, (1.0, 0.0, 0.0)) == ZERO
    assert controller.jump() is False


def test_both_disabled_object_is_not_in_physics():
    scene = Scene()
    _floor(scene)
    hero, _, _ = _body_hero(scene, body_enabled=False, character_enabled=False)
    for comp in hero.get_components(ColliderComponent):
        comp.enabled = False

    _run(scene, 30)

    physics = _world(scene)
    assert physics.handle_of(hero) is None
    assert hero.transform.world_position.y == pytest.approx(3.0, abs=1e-9)


def test_both_active_still_conflict(caplog):
    scene = Scene()
    _floor(scene)
    hero, _, _ = _body_hero(scene)

    with caplog.at_level(logging.ERROR):
        _run(scene, 3)

    assert _world(scene).handle_of(hero) is None
    assert sum(r.levelno >= logging.ERROR for r in caplog.records) == 1


def test_active_character_with_disabled_rigidbody_works():
    scene = Scene()
    _floor(scene)
    hero, _, _ = _body_hero(scene, y=0.92, body_enabled=False)

    _run(scene, 30)

    assert _world(scene)._bindings[hero.id].kind == "character"
    assert _world(scene).is_grounded(hero) is True


def test_frozen_twin_is_replaced_when_an_active_rigidbody_appears():
    scene, hero, _ = _standing_frozen()
    assert _world(scene)._bindings[hero.id].kind == "character"

    hero.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    hero.add_component(RigidBodyComponent, body_type="dynamic", mass=1.0)
    _run(scene, 3)

    binding = _world(scene)._bindings[hero.id]
    assert binding.kind == "dynamic"
    assert _world(scene).character_handles() == frozenset()


def test_enabling_character_replaces_the_body_twin():
    scene = Scene()
    _floor(scene)
    hero, body, controller = _body_hero(scene, y=0.92, character_enabled=False)
    _run(scene, 3)
    assert _world(scene)._bindings[hero.id].kind == "dynamic"

    body.enabled = False  # оба выключены (и коллайдер): объекта нет
    for collider in hero.get_components(ColliderComponent):
        collider.enabled = False
    _run(scene, 2)
    assert _world(scene).handle_of(hero) is None

    controller.enabled = True
    _run(scene, 30)
    assert _world(scene)._bindings[hero.id].kind == "character"
    assert _world(scene).is_grounded(hero) is True


# --- перепривязка двойника ---------------------------------------------------------

def test_character_added_to_an_object_with_a_collider_replaces_the_static_twin():
    scene = Scene()
    _floor(scene)
    hero = scene.create_game_object("Hero", position=Vector3(0, REST_Y, 0))
    hero.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    _run(scene, 3)
    assert _world(scene)._bindings[hero.id].kind == "static"

    hero.add_component(CharacterControllerComponent)
    _run(scene, 3)

    assert _world(scene)._bindings[hero.id].kind == "character"
    assert _world(scene).character_handles() == frozenset({_world(scene).handle_of(hero)})


def test_disabling_the_bound_component_rebinds_to_the_other_active_one():
    scene = Scene()
    _floor(scene)
    hero = scene.create_game_object("Hero", position=Vector3(0, 1.0, 0))
    big = hero.add_component(CharacterControllerComponent, radius=0.4, height=1.8)
    small = hero.add_component(CharacterControllerComponent, radius=0.3, height=1.2)
    _run(scene, 3)
    assert _world(scene)._bindings[hero.id].component is big

    big.enabled = False
    _run(scene, 3)

    binding = _world(scene)._bindings[hero.id]
    assert binding.component is small
    assert binding.frozen is False


# --- скрипты ------------------------------------------------------------------------

_TICK = (
    "def on_fixed_update(self, dt):\n"
    "    self.physics.simple_move((2, 0, 0))\n"
    "    self.game_object.tag = 'ticked'\n"
)


def _scripted_frozen(source: str = _NOOP):
    scene = Scene()
    _floor(scene)
    hero, comp = _hero(scene)
    hero.add_component(ScriptComponent, source=source)
    _run(scene, 5)
    comp.enabled = False
    return scene, hero, comp, hero.get_component(ScriptComponent)


def test_scripts_of_a_frozen_character_keep_ticking_without_errors(caplog):
    scene, hero, _, _ = _scripted_frozen(_TICK)
    hero.tag = "idle"  # метка от шагов до заморозки не считается
    x = hero.transform.world_position.x

    with caplog.at_level(logging.ERROR):
        _run(scene, 30)

    assert hero.tag == "ticked"  # скрипт отработал
    assert hero.transform.world_position.x == pytest.approx(x, abs=1e-9)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_script_physics_api_of_a_frozen_character():
    _, _, comp, script = _scripted_frozen()
    api = script.physics

    assert api.simple_move((1.0, 0.0, 0.0)) == ZERO
    assert api.simple_move() == ZERO  # WASD
    assert api.move_character((1.0, 0.0, 0.0)) == ZERO
    assert api.jump() is False
    assert api.jump(3.0) is False
    assert isinstance(api.is_grounded(), bool)
    api.sprint(True)
    api.set_air_control(0.5)
    comp.enabled = True
    assert comp.is_sprinting is True
    assert comp.air_control_override == 0.5


def test_script_physics_api_without_twin_does_not_raise():
    scene = Scene()
    _floor(scene)
    hero, _, _ = _body_hero(scene, character_enabled=False)
    hero.add_component(ScriptComponent, source=_NOOP)
    api = hero.get_component(ScriptComponent).physics

    assert api.simple_move((1.0, 0.0, 0.0)) == ZERO
    assert api.jump() is False
    assert api.is_grounded() is False


# --- сериализация -------------------------------------------------------------------

def test_disabled_flag_survives_scene_roundtrip():
    scene = Scene()
    go = scene.create_game_object("Hero")
    go.add_component(ColliderComponent, shape="box", size=(1, 1, 1))
    go.add_component(RigidBodyComponent, body_type="dynamic").enabled = False
    go.add_component(CharacterControllerComponent).enabled = False

    data = scene_to_dict(scene)
    payload = {c["type"]: c for c in data["objects"][0]["components"]}
    assert payload["CharacterControllerComponent"]["enabled"] is False
    assert payload["RigidBodyComponent"]["enabled"] is False
    assert "enabled" not in payload["ColliderComponent"]  # включённый - без поля

    loaded = scene_from_dict(data)
    hero = loaded.objects[0]
    assert hero.get_component(CharacterControllerComponent).enabled is False
    assert hero.get_component(RigidBodyComponent).enabled is False
    assert hero.get_component(ColliderComponent).enabled is True
