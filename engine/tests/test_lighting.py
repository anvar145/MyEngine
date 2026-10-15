"""Тесты системы освещения: источники, компонент, менеджер и упаковка данных.

Модуль проверяет чистую логику без OpenGL: параметры источников, привязку к
трансформу объекта, отсечение по радиусу влияния, кэширование наборов света и
раскладку данных по uniform-массивам.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.core.game_object import GameObject
from engine.core.math3d import Quaternion, Vector3
from engine.core.scene import Scene
from engine.rendering.lighting import (
    ALL_LAYERS_MASK,
    AMBIENT_TYPE,
    DIRECTIONAL_TYPE,
    LIGHT_TYPES,
    POINT_TYPE,
    SPOT_TYPE,
    AmbientLight,
    DirectionalLight,
    Light,
    LightBuffer,
    LightComponent,
    LightManager,
    LightSample,
    PointLight,
    SpotLight,
    create_light,
    light_from_dict,
)


class TestLightTypes:
    def test_all_requirements_types_exist(self) -> None:
        for kind in LIGHT_TYPES:
            light = create_light(kind)
            assert isinstance(light, Light)
            assert light.type_name == kind

    def test_type_ids_are_distinct(self) -> None:
        ids = {
            DirectionalLight().type_id,
            PointLight().type_id,
            SpotLight().type_id,
            AmbientLight().type_id,
        }
        assert ids == {DIRECTIONAL_TYPE, POINT_TYPE, SPOT_TYPE, AMBIENT_TYPE}

    def test_directional_defaults(self) -> None:
        sun = DirectionalLight()
        assert sun.cast_shadow is False
        assert sun.shadow_map_size == 512  # значение для слабых устройств
        assert sun.unit_direction.length == pytest.approx(1.0)
        assert sun.reach() == math.inf

    def test_directional_vector_to_light_is_opposite(self) -> None:
        sun = DirectionalLight(direction=Vector3(0, -1, 0))
        assert sun.vector_to_light.almost_equals(Vector3(0, 1, 0), 1e-6)

    def test_point_light_defaults(self) -> None:
        lamp = PointLight()
        assert lamp.range == pytest.approx(8.0)
        assert lamp.falloff == pytest.approx(2.0)
        assert lamp.reach() == pytest.approx(8.0)

    def test_point_light_range_is_positive(self) -> None:
        assert PointLight(range=-5.0).range > 0.0

    def test_spot_light_angles_are_ordered(self) -> None:
        spot = SpotLight(inner_angle=30.0, outer_angle=10.0)
        # Внешний угол обязан быть больше внутреннего, иначе конус вырожден.
        assert spot.outer_angle > spot.inner_angle

    def test_spot_light_cosines_match_angles(self) -> None:
        spot = SpotLight(inner_angle=20.0, outer_angle=30.0)
        assert spot.inner_cosine == pytest.approx(math.cos(math.radians(20.0)))
        assert spot.outer_cosine == pytest.approx(math.cos(math.radians(30.0)))
        assert spot.inner_cosine > spot.outer_cosine

    def test_spot_light_angle_is_clamped(self) -> None:
        assert SpotLight(outer_angle=179.0).outer_angle < 90.0

    def test_ambient_light_has_no_direction(self) -> None:
        ambient = AmbientLight()
        assert ambient.is_ambient
        assert ambient.world_direction().is_zero
        assert ambient.reach() == math.inf

    def test_intensity_scales_color(self) -> None:
        light = PointLight(color=Vector3(0.5, 0.5, 0.5), intensity=2.0)
        level = light.rgb
        assert level.x == pytest.approx(1.0)
        assert level.y == pytest.approx(1.0)

    def test_negative_intensity_is_clamped(self) -> None:
        light = PointLight(color=Vector3(1.0, 1.0, 1.0), intensity=-3.0)
        assert light.rgb.x == pytest.approx(0.0)

    def test_layers_mask_defaults_to_all(self) -> None:
        light = PointLight()
        assert light.layers == ALL_LAYERS_MASK
        for layer in range(32):
            assert light.affects_layer(layer)

    def test_layers_mask_filters(self) -> None:
        light = PointLight(layers=0b1010)
        assert light.affects_layer(1)
        assert light.affects_layer(3)
        assert not light.affects_layer(0)

    def test_unknown_layer_is_visible(self) -> None:
        assert PointLight(layers=0b1).affects_layer(99)

    def test_create_light_rejects_unknown_kind(self) -> None:
        with pytest.raises(ValueError, match="Неизвестный тип"):
            create_light("laser")

    def test_create_light_drops_irrelevant_arguments(self) -> None:
        # У фонового света нет направления: лишний аргумент не должен падать.
        light = create_light("ambient", direction=Vector3(1, 0, 0), intensity=0.3)
        assert isinstance(light, AmbientLight)
        assert light.intensity == pytest.approx(0.3)

    def test_repr_contains_state(self) -> None:
        assert "PointLight" in repr(PointLight(name="Lamp"))
        assert "on" in repr(PointLight())


class TestLightSerialization:
    def test_round_trip_directional(self) -> None:
        sun = DirectionalLight(
            direction=Vector3(1.0, -2.0, 0.5),
            color=Vector3(0.9, 0.8, 0.7),
            intensity=1.5,
            cast_shadow=True,
            shadow_map_size=256,
            name="Sun",
        )
        restored = light_from_dict(sun.to_dict())
        assert isinstance(restored, DirectionalLight)
        assert restored.cast_shadow is True
        assert restored.shadow_map_size == 256
        assert restored.unit_direction.almost_equals(sun.unit_direction, 1e-5)

    def test_round_trip_point(self) -> None:
        lamp = PointLight(
            position=Vector3(1.0, 2.0, 3.0),
            range=12.0,
            falloff=1.5,
            color=Vector3(1.0, 0.5, 0.25),
        )
        restored = light_from_dict(lamp.to_dict())
        assert isinstance(restored, PointLight)
        assert restored.position.almost_equals(Vector3(1, 2, 3), 1e-6)
        assert restored.range == pytest.approx(12.0)
        assert restored.falloff == pytest.approx(1.5)

    def test_round_trip_spot(self) -> None:
        spot = SpotLight(
            position=Vector3(0.0, 5.0, 0.0),
            direction=Vector3(0.0, -1.0, -0.2),
            inner_angle=15.0,
            outer_angle=25.0,
            range=20.0,
        )
        restored = light_from_dict(spot.to_dict())
        assert isinstance(restored, SpotLight)
        assert restored.inner_angle == pytest.approx(15.0)
        assert restored.outer_angle == pytest.approx(25.0)
        assert restored.range == pytest.approx(20.0)

    def test_round_trip_ambient(self) -> None:
        ambient = AmbientLight(color=Vector3(0.2, 0.3, 0.4), intensity=0.5)
        restored = light_from_dict(ambient.to_dict())
        assert isinstance(restored, AmbientLight)
        assert restored.intensity == pytest.approx(0.5)

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Неизвестный тип"):
            light_from_dict({"type": "portal"})

    def test_copy_is_independent(self) -> None:
        source = PointLight(range=5.0, color=Vector3(1.0, 0.0, 0.0))
        clone = source.copy()
        clone.range = 99.0
        assert source.range == pytest.approx(5.0)
        assert isinstance(clone, PointLight)


class TestLightComponent:
    def test_component_follows_owner_position(self) -> None:
        scene = Scene("Scene")
        game_object = scene.create_game_object("Lamp", position=Vector3(2, 3, 4))
        component = game_object.add_component(LightComponent, light=PointLight())
        component.sync_to_light()
        assert component.light.world_position().almost_equals(Vector3(2, 3, 4), 1e-5)

    def test_local_offset_is_applied(self) -> None:
        scene = Scene("Scene")
        game_object = scene.create_game_object("Lamp")
        component = game_object.add_component(
            LightComponent,
            light=PointLight(),
            local_offset=Vector3(0.0, 2.0, 0.0),
        )
        component.sync_to_light()
        assert component.light.world_position().almost_equals(
            Vector3(0, 2, 0), 1e-5
        )

    def test_spot_direction_follows_rotation(self) -> None:
        scene = Scene("Scene")
        game_object = scene.create_game_object("Spot")
        # Объект повёрнут на 90° вокруг Y: локальный +Z смотрит в мировой +X.
        game_object.transform.rotation = Quaternion.from_euler(0, 90, 0)
        component = game_object.add_component(
            LightComponent, light=SpotLight()
        )
        component.sync_to_light()
        assert component.light.world_direction().almost_equals(
            Vector3(1, 0, 0), 1e-4
        )

    def test_directional_uses_owner_direction(self) -> None:
        scene = Scene("Scene")
        game_object = scene.create_game_object("Sun")
        component = game_object.add_component(
            LightComponent, light=DirectionalLight()
        )
        component.sync_to_light()
        # Без поворота локальный +Z объекта — это мировой +Z.
        assert component.light.world_direction().almost_equals(
            Vector3(0, 0, 1), 1e-4
        )

    def test_ambient_is_not_moved(self) -> None:
        scene = Scene("Scene")
        game_object = scene.create_game_object("Ambient", position=Vector3(9, 9, 9))
        component = game_object.add_component(
            LightComponent, light=AmbientLight()
        )
        component.sync_to_light()
        assert component.light.world_position().is_zero

    def test_follow_owner_can_be_disabled(self) -> None:
        scene = Scene("Scene")
        game_object = scene.create_game_object("Lamp", position=Vector3(1, 0, 0))
        component = game_object.add_component(LightComponent, light=PointLight())
        component.follow_owner = False
        game_object.transform.position = Vector3(7, 7, 7)
        component.sync_to_light()
        assert component.light.world_position().almost_equals(Vector3(1, 0, 0), 1e-5)

    def test_display_name_falls_back_to_owner(self) -> None:
        scene = Scene("Scene")
        game_object = scene.create_game_object("Torch")
        component = game_object.add_component(LightComponent, light=PointLight())
        assert component.display_name == "Torch"
        assert component.kind == "point"

    def test_orphan_component_keeps_position(self) -> None:
        game_object = GameObject("Loose")
        component = game_object.add_component(LightComponent, light=PointLight())
        component.sync_to_light()
        assert component.light.world_position().almost_equals(Vector3.zero())

    def test_component_serialization(self) -> None:
        component = LightComponent(
            light=SpotLight(range=9.0),
            local_offset=Vector3(0.0, 1.0, 0.0),
            follow_owner=False,
        )
        restored = LightComponent.from_dict(component.to_dict())
        assert isinstance(restored.light, SpotLight)
        assert restored.light.range == pytest.approx(9.0)
        assert restored.follow_owner is False
        assert restored.local_offset.almost_equals(Vector3(0, 1, 0), 1e-6)

    def test_component_from_empty_dict(self) -> None:
        restored = LightComponent.from_dict({})
        assert isinstance(restored.light, DirectionalLight)


class TestLightManagerCollection:
    def _scene_with_lights(self) -> tuple[Scene, list[GameObject]]:
        """Сцена с четырьмя источниками разных типов."""
        scene = Scene("Scene")
        objects = []
        for name, light in (
            ("Sun", DirectionalLight(cast_shadow=True)),
            ("Lamp", PointLight(position=Vector3(3, 2, 0))),
            ("Spot", SpotLight(position=Vector3(-3, 2, 0))),
            ("Fill", AmbientLight(intensity=0.3)),
        ):
            game_object = scene.create_game_object(name)
            game_object.add_component(LightComponent, light=light)
            objects.append(game_object)
        return scene, objects

    def test_collect_gathers_all_lights(self) -> None:
        scene, _objects = self._scene_with_lights()
        manager = LightManager()
        assert manager.collect(scene.objects) == 4
        assert len(manager.lights) == 4

    def test_priority_puts_global_lights_first(self) -> None:
        scene, _objects = self._scene_with_lights()
        manager = LightManager()
        manager.collect(scene.objects)
        kinds = [light.type_name for light in manager.lights]
        # Солнце и фон влияют на всю сцену — они обязаны идти первыми.
        assert kinds[0] == "directional"
        assert kinds[1] == "ambient"

    def test_disabled_light_is_skipped(self) -> None:
        scene, objects = self._scene_with_lights()
        objects[1].get_component(LightComponent).light.enabled = False
        manager = LightManager()
        assert manager.collect(scene.objects) == 3

    def test_inactive_object_is_skipped(self) -> None:
        scene, objects = self._scene_with_lights()
        objects[0].active_self = False
        manager = LightManager()
        assert manager.collect(scene.objects) == 3

    def test_disabled_component_is_skipped(self) -> None:
        scene, objects = self._scene_with_lights()
        objects[2].get_component(LightComponent).enabled = False
        manager = LightManager()
        assert manager.collect(scene.objects) == 3

    def test_ambient_color_sums_ambient_lights(self) -> None:
        manager = LightManager()
        manager.set_lights(
            [
                AmbientLight(color=Vector3(1, 1, 1), intensity=0.2),
                AmbientLight(color=Vector3(1, 1, 1), intensity=0.3),
            ]
        )
        assert manager.ambient_color.x == pytest.approx(0.5)

    def test_ambient_color_falls_back_to_directional(self) -> None:
        manager = LightManager()
        manager.set_lights([DirectionalLight(ambient=0.4)])
        assert manager.ambient_color.x == pytest.approx(0.4)

    def test_shadow_caster_requires_flag(self) -> None:
        manager = LightManager()
        manager.set_lights([DirectionalLight(cast_shadow=False)])
        assert manager.shadow_caster is None
        manager.set_lights([DirectionalLight(cast_shadow=True)])
        assert manager.shadow_caster is not None

    def test_shadow_caster_disabled_when_shadows_off(self) -> None:
        manager = LightManager(cast_shadows=False)
        manager.set_lights([DirectionalLight(cast_shadow=True)])
        assert manager.shadow_caster is None

    def test_directional_property(self) -> None:
        manager = LightManager()
        assert manager.directional is None
        sun = DirectionalLight()
        manager.set_lights([PointLight(), sun])
        assert manager.directional is sun

    def test_add_and_clear(self) -> None:
        manager = LightManager()
        manager.add(PointLight())
        manager.add(DirectionalLight())
        assert len(manager.lights) == 2
        manager.clear()
        assert len(manager.lights) == 0
        assert manager.ambient_color.x >= 0.0

    def test_describe_mentions_limits(self) -> None:
        manager = LightManager(max_lights_per_object=4)
        manager.set_lights([PointLight()])
        text = manager.describe()
        assert "4" in text


class TestLightCulling:
    def test_directional_influences_everything(self) -> None:
        manager = LightManager()
        sun = DirectionalLight()
        assert manager.influences(sun, Vector3(1000, 1000, 1000), 1.0)

    def test_ambient_influences_everything(self) -> None:
        manager = LightManager()
        ambient = AmbientLight()
        assert manager.influences(ambient, Vector3(1000, 0, 0), 1.0)

    def test_point_light_culls_by_range(self) -> None:
        manager = LightManager()
        lamp = PointLight(position=Vector3.zero(), range=5.0)
        assert manager.influences(lamp, Vector3(3, 0, 0), 0.5)
        assert not manager.influences(lamp, Vector3(50, 0, 0), 0.5)

    def test_point_light_range_accounts_for_object_size(self) -> None:
        manager = LightManager()
        lamp = PointLight(position=Vector3.zero(), range=5.0)
        # Большой объект у границы радиуса всё ещё задет светом.
        assert manager.influences(lamp, Vector3(6, 0, 0), 2.0)
        assert not manager.influences(lamp, Vector3(20, 0, 0), 2.0)

    def test_point_light_boundary_is_inclusive(self) -> None:
        manager = LightManager()
        lamp = PointLight(position=Vector3.zero(), range=5.0)
        assert manager.influences(lamp, Vector3(5, 0, 0), 0.0)

    def test_spot_light_uses_cone_extent(self) -> None:
        manager = LightManager()
        spot = SpotLight(
            position=Vector3.zero(),
            direction=Vector3(0, -1, 0),
            range=10.0,
            outer_angle=30.0,
        )
        # Конус расширяется: на дальнем конце радиус больше range.
        spread = 10.0 * math.tan(math.radians(30.0))
        assert manager.influences(spot, Vector3(spread * 0.8, 0, 0), 0.0)

    def test_layer_mask_culls_light(self) -> None:
        manager = LightManager()
        lamp = PointLight(position=Vector3.zero(), range=100.0, layers=0b10)
        assert manager.influences(lamp, Vector3.zero(), 0.5, layer=1)
        assert not manager.influences(lamp, Vector3.zero(), 0.5, layer=2)

    def test_select_respects_cap(self) -> None:
        manager = LightManager(max_lights_per_object=2)
        manager.set_lights(
            [DirectionalLight()]
            + [PointLight(position=Vector3.zero(), range=100.0) for _ in range(5)]
        )
        selected = manager.select(Vector3.zero(), 0.5)
        assert len(selected) == 2

    def test_select_keeps_global_lights_when_capped(self) -> None:
        manager = LightManager(max_lights_per_object=2)
        manager.set_lights(
            [
                DirectionalLight(),
                AmbientLight(intensity=0.2),
                *[PointLight(position=Vector3.zero(), range=100.0) for _ in range(5)],
            ]
        )
        kinds = {light.type_name for light in manager.select(Vector3.zero(), 0.5)}
        assert kinds == {"directional", "ambient"}

    def test_select_caches_by_bounds(self) -> None:
        manager = LightManager()
        manager.set_lights([PointLight(position=Vector3.zero(), range=100.0)])
        manager.select(Vector3.zero(), 0.5)
        selections = manager.stats.selections
        manager.select(Vector3.zero(), 0.5)
        assert manager.stats.selection_hits == 1
        assert manager.stats.selections == selections + 1

    def test_cache_invalidated_when_lights_change(self) -> None:
        manager = LightManager()
        manager.set_lights([PointLight(position=Vector3.zero(), range=100.0)])
        manager.select(Vector3.zero(), 0.5)
        manager.set_lights([])
        assert manager.select(Vector3.zero(), 0.5) == ()

    def test_culled_counter_grows(self) -> None:
        manager = LightManager()
        manager.set_lights([PointLight(position=Vector3(50, 0, 0), range=1.0)])
        manager.select(Vector3.zero(), 0.1)
        assert manager.stats.culled >= 1

    def test_baked_mode_selects_only_ambient(self) -> None:
        manager = LightManager(dynamic=False)
        manager.set_lights([DirectionalLight(), AmbientLight(intensity=0.2)])
        selected = manager.select(Vector3.zero(), 0.5)
        assert [light.type_name for light in selected] == ["ambient"]


class TestLightBuffer:
    def test_fill_places_lights_in_order(self) -> None:
        buffer = LightBuffer(4)
        count = buffer.fill(
            [
                DirectionalLight(direction=Vector3(0, -1, 0)),
                PointLight(position=Vector3(1, 2, 3), range=7.0),
            ]
        )
        assert count == 2
        assert buffer.type_ids[0] == DIRECTIONAL_TYPE
        assert buffer.type_ids[1] == POINT_TYPE
        assert buffer.position[1] == pytest.approx((1.0, 2.0, 3.0))
        assert buffer.params[1, 0] == pytest.approx(7.0)

    def test_fill_respects_capacity(self) -> None:
        buffer = LightBuffer(2)
        count = buffer.fill([PointLight() for _ in range(10)])
        assert count == 2
        assert len(buffer) == 2

    def test_directional_direction_is_travel_direction(self) -> None:
        buffer = LightBuffer(1)
        buffer.fill([DirectionalLight(direction=Vector3(0, -1, 0))])
        # В массиве лежит направление, КУДА светит источник.
        assert buffer.direction[0] == pytest.approx((0.0, -1.0, 0.0))

    def test_spot_params_hold_cosines(self) -> None:
        buffer = LightBuffer(1)
        spot = SpotLight(inner_angle=15.0, outer_angle=30.0, range=9.0, falloff=3.0)
        buffer.fill([spot])
        assert buffer.params[0, 0] == pytest.approx(9.0)
        assert buffer.params[0, 1] == pytest.approx(spot.inner_cosine)
        assert buffer.params[0, 2] == pytest.approx(spot.outer_cosine)
        assert buffer.params[0, 3] == pytest.approx(3.0)

    def test_ambient_has_zero_position(self) -> None:
        buffer = LightBuffer(1)
        buffer.fill([AmbientLight()])
        assert buffer.position[0] == pytest.approx((0.0, 0.0, 0.0))

    def test_key_changes_with_parameters(self) -> None:
        first = LightBuffer(4)
        first.fill([PointLight(position=Vector3(1, 0, 0))])
        second = LightBuffer(4)
        second.fill([PointLight(position=Vector3(2, 0, 0))])
        assert first.key() != second.key()

    def test_key_stable_for_equal_parameters(self) -> None:
        first = LightBuffer(4)
        first.fill([PointLight(position=Vector3(1, 2, 3), range=4.0)])
        second = LightBuffer(4)
        second.fill([PointLight(position=Vector3(1, 2, 3), range=4.0)])
        assert first.key() == second.key()

    def test_arrays_are_contiguous_float32(self) -> None:
        buffer = LightBuffer(4)
        buffer.fill([PointLight()])
        for array in (buffer.position, buffer.direction, buffer.color, buffer.params):
            assert array.dtype == np.float32
            assert array.flags["C_CONTIGUOUS"]

    def test_describe_lists_types(self) -> None:
        buffer = LightBuffer(4)
        buffer.fill([DirectionalLight(), PointLight()])
        text = buffer.describe()
        assert "directional" in text
        assert "point" in text

    def test_empty_buffer_key(self) -> None:
        buffer = LightBuffer(4)
        buffer.fill([])
        assert buffer.key() == (0,)
        assert len(buffer) == 0


class TestLightSamples:
    def test_sample_from_point_light(self) -> None:
        sample = LightSample.from_light(
            PointLight(position=Vector3(1, 2, 3), range=5.0, intensity=2.0)
        )
        assert sample.type_id == POINT_TYPE
        assert sample.position == pytest.approx((1.0, 2.0, 3.0))
        assert sample.range == pytest.approx(5.0)
        assert sample.color[0] == pytest.approx(2.0)

    def test_sample_from_directional(self) -> None:
        sample = LightSample.from_light(
            DirectionalLight(direction=Vector3(0, -2, 0), intensity=1.0)
        )
        assert sample.type_id == DIRECTIONAL_TYPE
        assert sample.direction == pytest.approx((0.0, -1.0, 0.0))
        assert sample.range == pytest.approx(0.0)

    def test_type_name_lookup(self) -> None:
        assert LightSample.type_name(DIRECTIONAL_TYPE) == "directional"
        assert LightSample.type_name(SPOT_TYPE) == "spot"
        assert LightSample.type_name(999) == "?"


class TestManagerBinding:
    """Проверка загрузки света в шейдер на заглушке программы."""

    class _FakeProgram:
        """Минимальная программа: запоминает установленные uniform-массивы."""

        def __init__(self) -> None:
            self.arrays: dict[str, object] = {}
            self.uniforms: dict[str, object] = {}

        def set_uniform(self, name: str, value: object) -> bool:
            self.uniforms[name] = value
            return True

        def set_uniform_array(self, name: str, value: object) -> bool:
            self.arrays[name] = value
            return True

    def test_bind_uploads_all_arrays(self) -> None:
        manager = LightManager()
        manager.set_lights([DirectionalLight(), PointLight()])
        program = self._FakeProgram()
        count = manager.bind(program)
        assert count == 2
        for name in (
            "u_light_count",
            "u_light_position",
            "u_light_direction",
            "u_light_color",
            "u_light_params",
            "u_light_type",
            "u_max_lights",
        ):
            assert name in program.arrays or name in program.uniforms

    def test_bind_set_reuploads_when_sets_alternate(self) -> None:
        """A, B, A на одной программе: третий объект получает свой набор."""
        manager = LightManager()
        manager.set_lights(
            [
                PointLight(position=Vector3(0, 0, 0), range=2.0),
                PointLight(position=Vector3(50, 0, 0), range=2.0),
            ]
        )
        program = self._FakeProgram()
        near_a = manager.prepare(Vector3(0, 0, 0), 0.5)
        near_b = manager.prepare(Vector3(50, 0, 0), 0.5)
        manager.bind_set(program, near_a)
        position_a = program.arrays["u_light_position"].copy()
        manager.bind_set(program, near_b)
        # Третий объект снова в зоне A: набор берётся из кэша выбора.
        again = manager.prepare(Vector3(0, 0, 0), 0.5)
        manager.bind_set(program, again)
        assert (program.arrays["u_light_position"] == position_a).all()

    def test_bind_skips_repeated_set(self) -> None:
        manager = LightManager()
        manager.set_lights([PointLight()])
        program = self._FakeProgram()
        manager.bind(program)
        uploads = manager.stats.uploads
        manager.bind(program)
        assert manager.stats.upload_hits == 1
        assert manager.stats.uploads == uploads + 1

    def test_bind_reuploads_for_other_program(self) -> None:
        manager = LightManager()
        manager.set_lights([PointLight()])
        first = self._FakeProgram()
        second = self._FakeProgram()
        manager.bind(first)
        manager.bind(second)
        assert manager.stats.upload_hits == 0

    def test_force_reuploads(self) -> None:
        manager = LightManager()
        manager.set_lights([PointLight()])
        program = self._FakeProgram()
        manager.bind(program)
        manager.bind(program, force=True)
        assert manager.stats.upload_hits == 0

    def test_baked_mode_uploads_zero_lights(self) -> None:
        manager = LightManager(dynamic=False)
        manager.set_lights([DirectionalLight(), PointLight()])
        program = self._FakeProgram()
        count = manager.bind(program)
        assert count == 0
        counts = program.arrays["u_light_count"]
        assert int(np.asarray(counts)[0]) == 0

    def test_bind_set_uses_prepared_set(self) -> None:
        manager = LightManager()
        manager.set_lights([PointLight(position=Vector3(1, 0, 0), range=10.0)])
        light_set = manager.prepare(Vector3.zero(), 0.5)
        program = self._FakeProgram()
        assert manager.bind_set(program, light_set) == light_set.count


class TestBaking:
    def _mesh(self):
        """Простой меш: две вершины с нормалями вверх."""
        from engine.core.resources import MeshAsset

        return MeshAsset(
            path=__import__("pathlib").Path("<test>/bake"),
            positions=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
            indices=None,
            normals=np.array([[0, 1, 0], [0, 1, 0]], dtype=np.float32),
            uvs=None,
            name="bake",
        )

    def test_bake_uses_ambient_only_by_default(self) -> None:
        manager = LightManager()
        manager.set_lights(
            [DirectionalLight(direction=Vector3(0, -1, 0)), AmbientLight(intensity=0.5)]
        )
        colors = manager.bake_to_vertex_colors(self._mesh())
        assert colors.shape == (2, 4)
        assert colors.dtype == np.float32
        assert colors[0, 0] == pytest.approx(0.5, abs=1e-5)
        assert colors[0, 3] == pytest.approx(1.0)

    def test_bake_with_lights_brightens_lit_surface(self) -> None:
        manager = LightManager()
        manager.set_lights(
            [
                DirectionalLight(
                    direction=Vector3(0, -1, 0), color=Vector3(1, 1, 1), intensity=1.0
                ),
                AmbientLight(intensity=0.1),
            ]
        )
        lit = manager.bake_to_vertex_colors(self._mesh(), ambient_only=False)
        ambient = manager.bake_to_vertex_colors(self._mesh(), ambient_only=True)
        assert lit[0, 0] > ambient[0, 0]

    def test_bake_clamps_to_unit_range(self) -> None:
        manager = LightManager()
        manager.set_lights(
            [
                DirectionalLight(
                    direction=Vector3(0, -1, 0), color=Vector3(1, 1, 1), intensity=5.0
                ),
                AmbientLight(intensity=0.5),
            ]
        )
        colors = manager.bake_to_vertex_colors(self._mesh(), ambient_only=False)
        assert float(colors[:, :3].max()) <= 1.0
        assert float(colors[:, :3].min()) >= 0.0

    def test_bake_requires_normals(self) -> None:
        from pathlib import Path as _Path

        from engine.core.resources import MeshAsset

        mesh = MeshAsset(
            path=_Path("<test>/flat"),
            positions=np.zeros((3, 3), dtype=np.float32),
            indices=None,
            normals=None,
            uvs=None,
            name="flat",
        )
        manager = LightManager()
        with pytest.raises(ValueError, match="нормали"):
            manager.bake_to_vertex_colors(mesh)


class TestMaxLightsSetting:
    def test_default_is_eight(self) -> None:
        manager = LightManager()
        assert manager.max_lights_per_object == 8
        assert manager.max_lights == 8

    def test_cap_can_be_lowered(self) -> None:
        manager = LightManager(max_lights_per_object=4)
        manager.set_lights(
            [PointLight(position=Vector3.zero(), range=100.0) for _ in range(10)]
        )
        assert len(manager.select(Vector3.zero(), 0.5)) == 4

    def test_max_lights_setter_resizes_buffer(self) -> None:
        manager = LightManager()
        manager.max_lights = 3
        assert manager.max_lights == 3
        assert manager.buffer.capacity == 3

    def test_cap_must_be_positive(self) -> None:
        manager = LightManager(max_lights_per_object=0)
        assert manager.max_lights_per_object >= 1


class TestShaderSources:
    """Согласованность GLSL: канонический исходник и копия в ассетах.

    Освещение живёт в двух местах по необходимости: канонический файл рядом с
    движком и копия в ассетах, которую читает конвейер ресурсов. Если они
    разъедутся, редактор и тесты будут компилировать разный код — этот тест
    ловит расхождение сразу.
    """

    ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]

    def test_asset_copy_matches_engine_source(self) -> None:
        engine_copy = (
            self.ROOT / "engine" / "rendering" / "shaders" / "lighting.frag"
        )
        asset_copy = self.ROOT / "assets" / "shaders" / "lighting.frag"
        assert engine_copy.is_file(), "нет канонического исходника освещения"
        assert asset_copy.is_file(), "нет копии шейдера в ассетах"
        assert engine_copy.read_text(encoding="utf-8") == asset_copy.read_text(
            encoding="utf-8"
        ), "копии разошлись: запустите tools/sync_shaders.py"

    def test_library_has_no_version_directive(self) -> None:
        """Включаемый файл не должен содержать директиву #version.

        GLSL требует, чтобы версия была первым токеном: включаемый файл
        подставляется после неё, поэтому своя директива ломает компиляцию.
        Проверяется именно директива (строка, начинающаяся с ``#version``), а
        не упоминание в комментарии.
        """
        source = (
            self.ROOT / "engine" / "rendering" / "shaders" / "lighting.frag"
        ).read_text(encoding="utf-8")
        directives = [
            line.strip()
            for line in source.splitlines()
            if line.strip().startswith("#version")
        ]
        assert not directives, f"в библиотеке есть директива версии: {directives}"

    def test_library_has_no_main(self) -> None:
        """Библиотека не содержит main: его объявляет шейдер-хозяин."""
        source = (
            self.ROOT / "engine" / "rendering" / "shaders" / "lighting.frag"
        ).read_text(encoding="utf-8")
        assert "void main()" not in source

    def test_declares_expected_uniforms(self) -> None:
        """Имена uniform-переменных совпадают с тем, что грузит менеджер."""
        source = (
            self.ROOT / "engine" / "rendering" / "shaders" / "lighting.frag"
        ).read_text(encoding="utf-8")
        for name in (
            "u_light_position",
            "u_light_direction",
            "u_light_color",
            "u_light_params",
            "u_light_type",
            "u_light_count",
            "u_ambient",
            "u_shadow_map",
            "u_light_space_matrix",
            "u_shadow_texel",
            "u_shadow_bias",
            "u_shadow_normal_bias",
            "u_shadow_strength",
            "u_shadow_enabled",
        ):
            assert name in source, f"шейдер не объявляет {name}"

    def test_max_lights_matches_python_default(self) -> None:
        """Потолок источников в GLSL совпадает со значением движка."""
        from engine.rendering.lighting import DEFAULT_MAX_LIGHTS

        source = (
            self.ROOT / "engine" / "rendering" / "shaders" / "lighting.frag"
        ).read_text(encoding="utf-8")
        assert f"#define MAX_LIGHTS {DEFAULT_MAX_LIGHTS}" in source

    def test_shadow_shaders_exist(self) -> None:
        """Проходу теней нужны обе стадии."""
        shaders = self.ROOT / "assets" / "shaders"
        assert (shaders / "shadow.vert").is_file()
        assert (shaders / "shadow.frag").is_file()

    def test_shadow_vertex_declares_instanced_attributes(self) -> None:
        """Инстансные пачки обязаны отбрасывать тень одним вызовом."""
        source = (self.ROOT / "assets" / "shaders" / "shadow.vert").read_text(
            encoding="utf-8"
        )
        assert "INSTANCED" in source
        assert "in_instance_row0" in source


class TestShadowCulling:
    """Объём отсечения прохода теней совпадает с матрицей источника."""

    @staticmethod
    def _shadow_map():
        from engine.rendering.shadow import ShadowMap, directional_light_matrix

        shadow = ShadowMap(size=64)
        shadow.light_space_matrix = directional_light_matrix(
            Vector3(0, -1, 0), Vector3.zero(), 10.0
        )
        shadow.prepare_culling()
        return shadow

    def test_objects_above_area_center_cast_shadows(self) -> None:
        shadow = self._shadow_map()
        for height in (1.0, 5.0, 9.0, 15.0):
            assert shadow.is_visible(Vector3(0, height, 0), 0.5), height

    def test_objects_below_area_center_cast_shadows(self) -> None:
        shadow = self._shadow_map()
        assert shadow.is_visible(Vector3(0, -5.0, 0), 0.5)

    def test_far_objects_are_culled(self) -> None:
        shadow = self._shadow_map()
        assert not shadow.is_visible(Vector3(0, -200.0, 0), 0.5)
        assert not shadow.is_visible(Vector3(200.0, 0, 0), 0.5)


class TestShadowFit:
    def test_snap_moves_offset_to_texel_multiple(self) -> None:
        from engine.rendering.shadow import snap_to_texels

        matrix = np.identity(4)
        matrix[0, 3] = 0.01234
        matrix[1, 3] = -0.4321
        snapped = snap_to_texels(matrix, 512)
        step = 2.0 / 512
        assert snapped[0, 3] / step == pytest.approx(round(snapped[0, 3] / step))
        assert snapped[1, 3] / step == pytest.approx(round(snapped[1, 3] / step))
        assert abs(snapped[0, 3] - 0.01234) <= step / 2 + 1e-12

    def test_fit_grows_with_shadow_distance(self) -> None:
        from engine.rendering import Camera
        from engine.rendering.shadow import _shadow_fit

        camera = Camera(far=500.0)
        near = DirectionalLight(shadow_distance=20.0)
        far = DirectionalLight(shadow_distance=80.0)
        assert _shadow_fit(far, camera)[1] > _shadow_fit(near, camera)[1]

    def test_fit_supports_orthographic_camera(self) -> None:
        from engine.rendering import Camera
        from engine.rendering.shadow import _shadow_fit

        camera = Camera(far=500.0, mode="orthographic", ortho_height=10.0)
        _, radius = _shadow_fit(DirectionalLight(shadow_distance=20.0), camera)
        assert radius > 2.0


def test_zero_ambient_light_gives_black_ambient() -> None:
    manager = LightManager()
    manager.set_lights([DirectionalLight(), AmbientLight(intensity=0.0)])
    ambient = manager.ambient_color
    assert (ambient.x, ambient.y, ambient.z) == (0.0, 0.0, 0.0)
