"""Сборка кадра плеера из ECS-сцены без Qt.

В отличие от вьюпорта редактора, плеер умеет всё, что есть в сцене: меши по
GUID из ``Library/`` (cooked ``mesh.bin``), PBR-материалы, спрайты атласов,
камеры, свет и небо.
"""

from __future__ import annotations

import copy
import itertools
import logging
import math
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from engine.core.components import MeshRenderer, SpriteRenderer
from engine.core.input import MouseState
from engine.core.math3d import Vector3
from engine.core.resources import (
    cube_mesh,
    cylinder_mesh,
    plane_mesh,
    uv_sphere_mesh,
)
from engine.core.resources.sprite_atlas import load_sprite_atlas
from engine.core.ui_system import UISystem
from engine.rendering import GpuMesh
from engine.rendering.camera import Camera, CameraComponent
from engine.rendering.cooked_model import load_cooked_parts
from engine.rendering.debug_pass import DebugPass
from engine.rendering.lighting import LightComponent
from engine.rendering.material_loader import load_pbr_material
from engine.rendering.physics_debug import PhysicsDebugPass
from engine.rendering.sprite_batch import SpriteBatch, SpritePass
from engine.rendering.text import TextBatch, TextPass, fill_text_batch
from engine.rendering.text.atlas import upload_cached_atlases
from engine.rendering.ui import UIFiller, UIPass, UIScreen

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.game_object import GameObject
    from engine.core.resources import AssetDatabase
    from engine.core.scene import Scene
    from engine.rendering.render_graph import RenderTarget
    from engine.rendering.renderer import Renderer, RenderStats

__all__ = ["SceneRenderer"]

_log = logging.getLogger("engine.player.scene_renderer")

#: Минимум одинаковых (меш, материал) для отрисовки инстансингом.
INSTANCE_MIN = 3

_BUILTIN = {
    "cube": lambda: cube_mesh(1.0),
    "sphere": lambda: uv_sphere_mesh(0.5, segments=40, rings=24),
    "plane": lambda: plane_mesh(2.0),
    "cylinder": lambda: cylinder_mesh(0.5, 1.0),
}


class _UiLayer:
    """UI дополнительной сцены: свой раскладчик, ввод и пакеты (рисуется поверх)."""

    def __init__(
        self, renderer: Renderer, scene: Scene, screen: UIScreen, mouse: MouseState
    ) -> None:
        self.scene = scene
        self.renderer = renderer
        self.filler = UIFiller()
        self.system = UISystem(
            scene,
            screen,
            mouse,
            layout=self.filler.layout,
            measure=self.filler.measure,
        )
        self.ui_pass = UIPass(screen, text_batch=TextBatch())
        renderer.add_pass(self.ui_pass)

    def dispose(self) -> None:
        self.renderer.graph.remove_pass(self.ui_pass)
        self.ui_pass.batch.dispose()
        self.ui_pass.text_batch.dispose()
        self.filler.dispose()


class SceneRenderer:
    """Рисует сцену через :class:`~engine.rendering.renderer.Renderer`.

    Args:
        renderer: Инициализированный рендерер.
        scene: Сцена для отрисовки.
        assets: База ассетов проекта (``None`` — только встроенные меши).
        project_root: Корень проекта: путь к файлам мешей без GUID.
    """

    def __init__(
        self,
        renderer: Renderer,
        scene: Scene,
        assets: AssetDatabase | None = None,
        project_root: Path | None = None,
        mouse: MouseState | None = None,
    ) -> None:
        self.renderer = renderer
        self.scene = scene
        self.mouse = MouseState() if mouse is None else mouse
        self.ui_screen = UIScreen()
        self._ui_filler = UIFiller()
        self._ui_pass: UIPass | None = None
        self._physics_pass: PhysicsDebugPass | None = None
        #: Отладочные оверлеи ``engine.debug`` (F3); создаются при первом показе.
        self._debug_pass: DebugPass | None = None
        self.ui_system = self._make_ui_system()
        self.assets = assets
        self.project_root = project_root
        self._meshes: dict[str, list[tuple[GpuMesh, Any]]] = {}
        self._materials: dict[tuple, Any] = {}
        self._textures: dict[str, Any] = {}
        self._batches: dict[str, SpriteBatch] = {}
        self._sprite_pass: SpritePass | None = None
        self._text_batch: TextBatch | None = None
        self._failed: set[str] = set()
        self.camera: Camera | None = None
        self._fallback: Camera | None = None
        #: Пропускать кадр через пост-обработку (нужен ``Renderer.hdr_enabled``).
        self.post = False
        #: Дополнительные (аддитивные) сцены: рисуются вместе с основной.
        self.extra_scenes: list[Scene] = []
        self._ui_layers: dict[int, _UiLayer] = {}

    def _make_ui_system(self) -> UISystem:
        return UISystem(
            self.scene,
            self.ui_screen,
            self.mouse,
            layout=self._ui_filler.layout,
            measure=self._ui_filler.measure,
        )

    def set_scene(self, scene: Scene) -> None:
        """Сменить сцену: раскладка и ввод UI начинаются заново."""
        self.scene = scene
        self.extra_scenes = []
        self._sync_ui_layers()
        self._ui_filler.dispose()
        self._ui_filler = UIFiller()
        self.ui_system = self._make_ui_system()

    def set_extra_scenes(self, scenes: list[Scene]) -> None:
        """Задать дополнительные сцены (их UI - слоями поверх основной)."""
        self.extra_scenes = list(scenes)
        self._sync_ui_layers()

    def _sync_ui_layers(self) -> None:
        wanted = {id(scene): scene for scene in self.extra_scenes}
        for key in [k for k in self._ui_layers if k not in wanted]:
            self._ui_layers.pop(key).dispose()
        for key, scene in wanted.items():
            if key not in self._ui_layers:
                self._ui_layers[key] = _UiLayer(
                    self.renderer, scene, self.ui_screen, self.mouse
                )

    def _objects(self) -> Iterator[GameObject]:
        """Объекты основной и дополнительных сцен."""
        return itertools.chain(
            self.scene.objects, *(s.objects for s in self.extra_scenes)
        )

    def set_physics_debug(self, enabled: bool) -> None:
        """Показать/скрыть каркасы коллайдеров физики (F2 в плеере)."""
        if self._physics_pass is None:
            if not enabled:
                return
            self._physics_pass = PhysicsDebugPass(
                lambda: self.scene.physics_if_exists
            )
            names = [p.name for p in self.renderer.graph.passes]
            index = names.index("transparent") + 1 if "transparent" in names else None
            self.renderer.add_pass(self._physics_pass, index=index)
        self._physics_pass.enabled = bool(enabled)

    def set_debug_overlays(self, visible: bool) -> None:
        """Показать/скрыть оверлеи ``engine.debug`` (F3 в плеере)."""
        if self._debug_pass is None:
            if not visible:
                return
            self._debug_pass = DebugPass(context=lambda: self.scene)
        self._debug_pass.registry.set_visible(visible)
        self._keep_debug_last()

    def _keep_debug_last(self) -> None:
        """Держать ``DebugPass`` последним проходом основной фазы.

        Проходы текста и спрайтов добавляются лениво (по первому объекту) и
        иначе оказались бы после него — поверх отладочных линий.
        """
        debug = self._debug_pass
        if debug is None:
            return
        graph = self.renderer.graph
        passes = [p for p in graph.passes if not getattr(p, "after_post", False)]
        if passes and passes[-1] is debug:
            return
        graph.remove_pass(debug)
        rest = list(graph.passes)
        mains = [i for i, p in enumerate(rest) if not getattr(p, "after_post", False)]
        self.renderer.add_pass(debug, index=mains[-1] + 1 if mains else 0)

    def update_ui(self, viewport: tuple[int, int]) -> None:
        """Обработать мышь для UI (до обновления скриптов кадра)."""
        self.ui_screen.set_viewport(*viewport)
        for layer in list(self._ui_layers.values()):
            layer.system.update()
        self.ui_system.update()

    # ------------------------------------------------------------------ #
    # Ресурсы
    # ------------------------------------------------------------------ #
    def _warn_once(self, key: str, message: str, *args: object) -> None:
        if key not in self._failed:
            self._failed.add(key)
            _log.warning(message, *args)

    def _pbr_material(self, guid: str, artifact: str | None = None) -> Any | None:
        key = ("pbr", guid, artifact)
        if key in self._materials:
            return self._materials[key]
        material = None
        if self.assets is not None:
            try:
                material = load_pbr_material(
                    self.renderer,
                    self.assets,
                    guid,
                    artifact_name=artifact,
                    textures=self._textures,
                )
            except Exception as error:  # noqa: BLE001 - кадр не должен падать
                self._warn_once(
                    f"mat:{guid}:{artifact}", "Материал %s: %s", guid, error
                )
        self._materials[key] = material
        return material

    def _with_layer(self, material: Any, layer: int) -> Any:
        """Копия материала с другим слоем сортировки (кэшируется)."""
        key = ("layer", id(material), layer)
        clone = self._materials.get(key)
        if clone is None:
            clone = copy.copy(material)
            clone.sorting_layer = layer
            self._materials[key] = clone
        return clone

    def _plain_material(self, component: MeshRenderer) -> Any:
        color = component.color
        key = (
            "plain",
            color.x,
            color.y,
            color.z,
            component.roughness,
            component.metallic,
        )
        material = self._materials.get(key)
        if material is None:
            material = self.renderer.create_material(
                f"object_{len(self._materials)}",
                base_color=color,
                roughness=component.roughness,
                metallic=component.metallic,
            )
            self._materials[key] = material
        return material

    def _meshes_for(self, component: MeshRenderer) -> list[tuple[GpuMesh, Any]]:
        """Список ``(меш, PBR-материал или None)`` для компонента."""
        key = component.mesh_guid or component.mesh
        cached = self._meshes.get(key)
        if cached is not None:
            return cached
        result: list[tuple[GpuMesh, Any]] = []
        try:
            if component.mesh_guid and self.assets is not None:
                result = self._load_cooked(component.mesh_guid)
            elif component.mesh in _BUILTIN:
                result = [(GpuMesh.from_mesh_asset(_BUILTIN[component.mesh]()), None)]
            elif self.project_root is not None:
                result = self._load_by_path(component.mesh)
        except Exception as error:  # noqa: BLE001 - отсутствующий ассет не фатален
            self._warn_once(f"mesh:{key}", "Меш '%s' не загружен: %s", key, error)
        self._meshes[key] = result
        return result

    def _load_by_path(self, path: str) -> list[tuple[GpuMesh, Any]]:
        if self.assets is not None:
            guid = self.assets.guid_of(self.project_root / path)  # type: ignore[operator]
            if guid:
                return self._load_cooked(guid)
        raise FileNotFoundError(path)

    def _load_cooked(self, guid: str) -> list[tuple[GpuMesh, Any]]:
        return load_cooked_parts(self.assets, guid, self._pbr_material)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    # Камера, свет, спрайты
    # ------------------------------------------------------------------ #
    def sync(self) -> None:
        """Перенести трансформы в камеры и свет, собрать источники."""
        self.camera = None
        for game_object in self._objects():
            if not game_object.active_in_hierarchy:
                continue
            camera = game_object.get_component(CameraComponent)
            if camera is not None:
                camera.sync_to_camera()
                if self.camera is None and camera.enabled and camera.camera.enabled:
                    self.camera = camera.camera
            light = game_object.get_component(LightComponent)
            if light is not None:
                light.sync_to_light()
        if self.camera is None:
            self._warn_once("no-camera", "В сцене нет активной камеры: запасная")
            if self._fallback is None:
                self._fallback = Camera(position=Vector3(5.0, 4.0, 7.0))
                self._fallback.look_at(Vector3(0.0, 0.5, 0.0))
            self.camera = self._fallback
        self.renderer.lights.collect(self._objects())

    def _fill_sprites(self) -> None:
        for batch in self._batches.values():
            batch.clear()
        if self.assets is None:
            return
        for game_object in self._objects():
            if not game_object.active_in_hierarchy:
                continue
            sprite = game_object.get_component(SpriteRenderer)
            if sprite is None or not sprite.visible or not sprite.atlas_guid:
                continue
            batch = self._batch_for(sprite.atlas_guid)
            if batch is None:
                continue
            transform = game_object.transform
            position = transform.world_position
            scale = transform.scale
            q = transform.rotation
            angle = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )
            try:
                batch.add(
                    sprite.sprite,
                    (position.x, position.y, position.z),
                    scale=(scale.x, scale.y),
                    rotation=angle,
                    color=sprite.color,
                    flip_x=sprite.flip_x,
                    flip_y=sprite.flip_y,
                )
            except KeyError as error:
                self._warn_once(f"sprite:{sprite.sprite}", "%s", error)

    def _fill_text(self) -> None:
        if self._text_batch is None:
            self._text_batch = TextBatch()
            self.renderer.add_pass(TextPass(self._text_batch))
        roots = (self.project_root,) if self.project_root is not None else ()
        if self.extra_scenes:  # токен кэша описывает одну сцену
            fill_text_batch(self._text_batch, self._objects(), roots)
        else:
            fill_text_batch(
                self._text_batch, self.scene.objects, roots, self.scene
            )

    def _fill_ui(self, viewport: tuple[int, int]) -> None:
        if self._ui_pass is None:
            self._ui_pass = UIPass(self.ui_screen, text_batch=TextBatch())
            self.renderer.add_pass(self._ui_pass)
        self.ui_screen.set_viewport(*viewport)
        roots = (self.project_root,) if self.project_root is not None else ()
        self._ui_filler.fill(
            self.scene,
            self.ui_screen,
            self._ui_pass.batch,
            self._ui_pass.text_batch,
            roots,
        )
        self._sync_ui_layers()
        for layer in self._ui_layers.values():
            layer.filler.fill(
                layer.scene,
                self.ui_screen,
                layer.ui_pass.batch,
                layer.ui_pass.text_batch,
                roots,
            )

    def _batch_for(self, guid: str) -> SpriteBatch | None:
        batch = self._batches.get(guid)
        if batch is not None:
            return batch
        try:
            batch = SpriteBatch(load_sprite_atlas(self.assets, guid))  # type: ignore[arg-type]
        except Exception as error:  # noqa: BLE001
            self._warn_once(f"atlas:{guid}", "Атлас %s: %s", guid, error)
            return None
        self._batches[guid] = batch
        if self._sprite_pass is None:
            self._sprite_pass = SpritePass()
            self.renderer.add_pass(self._sprite_pass)
        self._sprite_pass.batches.append(batch)
        return batch

    def preload(self, viewport: tuple[int, int]) -> float:
        """Загрузить в GPU всё, что нужно сцене, до первого кадра.

        Без прогрева меши, текстуры и атласы текста заливаются лениво, и первый
        кадр растягивается на сотни миллисекунд.

        Args:
            viewport: Размер области вывода (раскладка UI зависит от него).

        Returns:
            Затраченное время, мс.
        """
        started = time.perf_counter()
        for game_object in self._objects():
            component = game_object.get_component(MeshRenderer)
            if component is None:
                continue
            if component.material_guid:
                self._pbr_material(component.material_guid)
            for mesh, _material in self._meshes_for(component):
                if not mesh.is_uploaded:
                    mesh.upload()
        for texture in self._textures.values():
            if texture is not None and not texture.is_uploaded:
                texture.upload()
        self._fill_sprites()
        for batch in self._batches.values():
            batch.preload()
        self._fill_text()
        self._fill_ui(viewport)
        upload_cached_atlases()
        elapsed = (time.perf_counter() - started) * 1000.0
        _log.info("Предзагрузка сцены: %.0f мс", elapsed)
        return elapsed

    # ------------------------------------------------------------------ #
    # Кадр
    # ------------------------------------------------------------------ #
    def render(
        self, viewport: tuple[int, int], target: RenderTarget | None = None
    ) -> RenderStats | None:
        """Нарисовать кадр.

        Args:
            viewport: Размер области вывода.
            target: Цель вывода (``None`` — экран).

        Returns:
            Статистику кадра или ``None``, если в сцене нет активной камеры.
        """
        self.sync()
        if self.camera is None:
            return None
        self._fill_sprites()
        self._fill_text()
        self._fill_ui(viewport)
        self._keep_debug_last()
        self.renderer.begin_frame(
            self.camera, viewport, target=target, post=self.post
        )
        groups: dict[tuple, tuple[Any, Any, list]] = {}
        for game_object in self._objects():
            if not game_object.active_in_hierarchy:
                continue
            component = game_object.get_component(MeshRenderer)
            if component is None or not component.visible:
                continue
            custom = (
                self._pbr_material(component.material_guid)
                if component.material_guid
                else None
            )
            matrix = game_object.transform.world_matrix
            for mesh, material in self._meshes_for(component):
                chosen = custom or material or self._plain_material(component)
                if component.sorting_layer:
                    chosen = self._with_layer(chosen, component.sorting_layer)
                key = (
                    id(mesh),
                    id(chosen),
                    component.cast_shadow,
                    component.receive_shadow,
                )
                entry = groups.get(key)
                if entry is None:
                    groups[key] = (mesh, chosen, [matrix])
                else:
                    entry[2].append(matrix)
        self._flush_groups(groups)
        return self.renderer.end_frame(target=target)

    def _flush_groups(self, groups: dict[tuple, tuple[Any, Any, list]]) -> None:
        """Отправить накопленные группы: повторяющиеся меши — инстансингом.

        Одна геометрия с одним материалом (враги, укрытия, пропсы) рисуется
        одним draw call вместо N: вызовы PyOpenGL на HD 520 дороже самого GPU.
        Непрозрачные группы от ``INSTANCE_MIN`` экземпляров идут пачкой (отсечение
        по группе целиком); остальное — обычными объектами очереди.
        """
        renderer = self.renderer
        for (_, _, cast, receive), (mesh, material, matrices) in groups.items():
            if len(matrices) >= INSTANCE_MIN and not material.transparent:
                renderer.draw_instanced(
                    mesh,
                    material,
                    np.stack(matrices),
                    cast_shadow=cast,
                    receive_shadow=receive,
                )
                continue
            for matrix in matrices:
                renderer.submit(
                    mesh,
                    material,
                    matrix,
                    cast_shadow=cast,
                    receive_shadow=receive,
                )

    def pick_candidates(self) -> list[tuple[str, int, Vector3]]:
        """Объекты с мешами для выбора кликом: ``(имя, id, позиция)``."""
        found = []
        for game_object in self._objects():
            if game_object.get_component(MeshRenderer) is not None:
                position = game_object.transform.world_position
                found.append((game_object.name, game_object.id, position))
        return found

    def dispose(self) -> None:
        """Освободить GPU-ресурсы, созданные сборщиком."""
        for entries in self._meshes.values():
            for mesh, _ in entries:
                mesh.dispose()
        for batch in self._batches.values():
            batch.dispose()
        if self._text_batch is not None:
            self._text_batch.dispose()
            self._text_batch = None
        if self._ui_pass is not None:
            self._ui_pass.batch.dispose()
            self._ui_pass.text_batch.dispose()
            self._ui_pass = None
        for layer in self._ui_layers.values():
            layer.dispose()
        self._ui_layers.clear()
        if self._physics_pass is not None:
            self._physics_pass.dispose()
            self.renderer.graph.remove_pass(self._physics_pass)
            self._physics_pass = None
        if self._debug_pass is not None:
            self._debug_pass.dispose()
            self.renderer.graph.remove_pass(self._debug_pass)
            self._debug_pass = None
        self._ui_filler.dispose()
        self._meshes.clear()
        self._batches.clear()
