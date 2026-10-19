"""Вьюпорт редактора: движковый ``QOpenGLWidget`` плюс связь со сценой.

Панель решает четыре задачи:

* встраивает рендерер из :mod:`engine.rendering` в окно редактора
  (центральная область ``QMainWindow``);
* переводит ECS-сцену в список объектов отрисовки — по компонентам
  :class:`~engine.core.components.MeshRenderer`, с кэшами мешей и материалов;
* синхронизирует игровые камеры (:class:`~engine.rendering.CameraComponent`)
  с их объектами и рисует гизмо камер — иконку и усечённый объём;
* показывает рамку выделения, умеет наводить камеру на выбранный объект и
  выбирать камеру кликом по её гизмо.

Камера управляется мышью: ЛКМ (или Alt+ЛКМ) — орбита, ПКМ/СКМ — панорама,
колесо — зум, ``F`` — фокус на выделенном.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject, QTimer, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from editor.context import EditorContext
from editor.gizmos import CameraGizmo, CameraGizmoStyle, LightGizmo, LightGizmoStyle
from editor.gizmos.transform_gizmo import (
    GizmoMode,
    TransformGizmo,
    arrow_mesh,
    ring_mesh,
    scale_handle_mesh,
)
from editor.mime import PREFAB_MIME
from editor.prefab_actions import instantiate_prefab_file
from editor.theme import SELECTION_COLOR
from editor.viewport.camera_overlay import OVERLAY_SIZE, CameraPreviewOverlay
from engine.core.commands import SetFieldCommand
from engine.core.commands.object_commands import clone_value
from engine.core.components import MeshRenderer
from engine.core.game_object import GameObject
from engine.core.math3d import MATRIX_DTYPE, Vector3, compose_matrix
from engine.core.physics_components import (
    CharacterControllerComponent,
    ColliderComponent,
)
from engine.core.resources import (
    MeshAsset,
    ResourceManager,
    cube_mesh,
    cylinder_mesh,
    line_grid_mesh,
    plane_mesh,
    uv_sphere_mesh,
    wire_box_mesh,
)
from engine.prefab import PrefabError
from engine.rendering import (
    Camera,
    CameraComponent,
    CameraPreview,
    GpuMesh,
    Light,
    LightComponent,
    Material,
    QualityTier,
    Renderer,
    RenderSettings,
    RenderStats,
)
from engine.rendering.camera import quaternion_from_forward
from engine.rendering.cooked_model import load_cooked_parts
from engine.rendering.dynamic_resolution import read_hdr_hint, read_render_hints
from engine.rendering.material_loader import load_pbr_material
from engine.rendering.physics_debug import PhysicsDebugPass
from engine.rendering.post import PostSettings
from engine.rendering.qt_viewport import ViewportWidget
from engine.rendering.sky import SkyPass
from engine.rendering.text import TextBatch, TextPass, fill_text_batch

__all__ = [
    "BUILTIN_MESH_FACTORIES",
    "COLLIDER_MODES",
    "EditorViewport",
]

COLLIDER_MODE_OFF = "off"
COLLIDER_MODE_SELECTED = "selected"
COLLIDER_MODE_ALL = "all"
#: Режимы каркасов коллайдеров в редакторе.
COLLIDER_MODES = (COLLIDER_MODE_OFF, COLLIDER_MODE_SELECTED, COLLIDER_MODE_ALL)

_log = logging.getLogger("editor.viewport")

#: Разрешение сетки редактора (мировые единицы).
GRID_SIZE = 20.0
GRID_DIVISIONS = 20

#: Фабрики встроенных процедурных мешей: имя из ``BUILTIN_MESHES`` → геометрия.
#: Словарь обязан покрывать весь :data:`engine.core.components.BUILTIN_MESHES`
#: (проверяется тестом): иначе инспектор предложит меш, который не загрузится.
BUILTIN_MESH_FACTORIES: dict[str, Callable[[], MeshAsset]] = {
    "cube": lambda: cube_mesh(1.0),
    "sphere": lambda: uv_sphere_mesh(0.5, segments=40, rings=24),
    "plane": lambda: plane_mesh(2.0),
    "cylinder": lambda: cylinder_mesh(0.5, 1.0),
}

#: Период обновления предпросмотра камеры (мс): ~15 кадров в секунду.
PREVIEW_INTERVAL_MS = 66

#: Цвет гизмо невыделенной камеры (совпадает с ``CameraGizmoStyle``).
GIZMO_COLOR = (0.62, 0.68, 0.78)


#: Расстояние от камеры для точки сброса, если луч ушёл в небо.
DROP_FALLBACK_DISTANCE = 10.0


class EditorViewport(QWidget):
    """Центральный 3D-вьюпорт редактора.

    Args:
        context: Контекст редактора (сцена, выделение, переводчик).
        parent: Родительский виджет.

    Signals:
        stats_updated: Обновлена статистика кадра (FPS и показатели рендера).
        ready: Вьюпорт готов к работе (контекст OpenGL создан).
    """

    stats_updated = Signal(float, object)
    ready = Signal()
    #: Объект перемещён/повёрнут/масштабирован гизмо (идёт перетаскивание).
    object_transformed = Signal(object)
    #: Перетаскивание гизмо закончено (аргумент — объект).
    transform_committed = Signal(object)
    #: Сменился режим гизмо (строка ``GizmoMode``).
    gizmo_mode_changed = Signal(str)

    def __init__(self, context: EditorContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.context = context
        self._meshes: dict[str, GpuMesh] = {}
        self._materials: dict[tuple, Material] = {}
        # Импортированные модели (glTF/FBX): GUID -> [(меш, PBR-материал)].
        self._models: dict[str, list[tuple[GpuMesh, object]]] = {}
        self._model_failed: set[str] = set()
        self._pbr: dict[tuple, object] = {}
        self._pbr_textures: dict[str, object] = {}
        self._grid: GpuMesh | None = None
        self._grid_material: Material | None = None
        self._selection_box: GpuMesh | None = None
        self._selection_material: Material | None = None
        self._renderer: Renderer | None = None
        self._sky_pass: SkyPass | None = None
        self._text_batch: TextBatch | None = None
        self._resources: ResourceManager | None = None

        # Гизмо камер: материалы переиспользуются, гизмо — по одной на камеру.
        self._gizmo_style = CameraGizmoStyle()
        self._gizmo_materials: dict[tuple, Material] = {}
        self._gizmos: dict[int, CameraGizmo] = {}
        self.show_camera_gizmos = True
        self.show_camera_frustum = False
        self._gizmo_provider: Callable[[], Sequence[tuple]] | None = None
        self._preview: CameraPreview | None = None
        self._preview_dirty = True
        # Гизмо источников света и кэш ассетов мешей (для запекания света).
        self._light_gizmo_style = LightGizmoStyle()
        self._light_gizmos: dict[int, LightGizmo] = {}
        self._mesh_assets: dict[str, MeshAsset] = {}
        self.show_light_gizmos = True
        self.show_light_ranges = False
        #: Режим каркасов коллайдеров: ``off`` / ``selected`` / ``all``.
        self.collider_mode = COLLIDER_MODE_SELECTED
        self._physics_pass: PhysicsDebugPass | None = None
        # Гизмо трансформа выделенного объекта (оси перемещения/поворота/масштаба).
        self.transform_gizmo = TransformGizmo()
        #: Трансформ цели в начале перетаскивания гизмо (для записи в историю).
        self._gizmo_before: tuple[str, Vector3, object, Vector3] | None = None
        self.show_transform_gizmo = True
        self._gizmo_meshes: dict[GizmoMode, GpuMesh] = {}
        self._overlay_materials: dict[tuple, Material] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.settings = RenderSettings.for_quality(
            QualityTier.MEDIUM,
            samples=0,
            hdr=read_render_hints(context.project_root)[2],
        )
        self.viewport = ViewportWidget(
            settings=self.settings,
            renderer=Renderer(
                settings=self.settings,
                shader_root=_shader_root(),
                program_cache_dir=context.shader_cache_dir,
            ),
            shader_root=_shader_root(),
            on_ready=self._on_renderer_ready,
            parent=self,
        )
        layout.addWidget(self.viewport)
        self.viewport.frame_rendered.connect(self._on_frame_rendered)
        self.viewport.set_pick_provider(self._pick_gizmo)
        self.viewport.set_drag_handler(_GizmoDragHandler(self))
        self.viewport.set_click_provider(self._on_viewport_click)

        # Предпросмотр выбранной камеры — маленькое окно в углу сцены.
        self.camera_overlay = CameraPreviewOverlay(self.viewport)
        self.viewport.setAcceptDrops(True)  # префабы из панели ассетов
        self.viewport.installEventFilter(self)
        self._overlay_timer = QTimer(self)
        self._overlay_timer.timeout.connect(self.update_camera_overlay)
        self._overlay_timer.start(PREVIEW_INTERVAL_MS)

        context.selection.changed.connect(lambda _objects: self.refresh())
        context.scene_changed.connect(self._on_scene_changed)
        context.scene_modified.connect(self.refresh)
        context.imports_changed.connect(self._on_imports_changed)

    # ------------------------------------------------------------------ #
    # Свойства
    # ------------------------------------------------------------------ #
    @property
    def renderer(self) -> Renderer | None:
        """Активный рендерер (``None`` до создания контекста)."""
        return self._renderer

    @property
    def camera(self):
        """Камера вьюпорта."""
        return self.viewport.camera

    @property
    def is_ready(self) -> bool:
        """``True``, если OpenGL-контекст создан и рендерер работает."""
        return self._renderer is not None

    def light_manager(self):
        """Менеджер света рендерера (``None``, пока контекст не создан).

        Через него панель света включает и выключает динамическое освещение и
        запекает свет: это единственная точка, где редактор обращается к свету.
        """
        if self._renderer is None:
            return None
        return self._renderer.lights

    def mesh_asset(self, key: str) -> MeshAsset | None:
        """Собрать (или взять из кэша) ассет меша по ключу.

        Нужен для запекания света: оно работает с массивами вершин, а не с
        GPU-буферами.

        Args:
            key: Имя встроенного меша или путь к файлу.

        Returns:
            Ассет меша или ``None``, если ключ неизвестен.
        """
        cached = self._mesh_assets.get(key)
        if cached is not None:
            return cached
        factory = BUILTIN_MESH_FACTORIES.get(key)
        if factory is not None:
            asset = factory()
            self._mesh_assets[key] = asset
            return asset
        try:
            asset = self._resources.load(key) if self._resources else None
        except Exception as exc:  # noqa: BLE001 - неизвестный меш не критичен
            _log.debug("Меш '%s' недоступен для запекания: %s", key, exc)
            return None
        if isinstance(asset, MeshAsset):
            self._mesh_assets[key] = asset
            return asset
        return None

    # ------------------------------------------------------------------ #
    # Инициализация рендера
    # ------------------------------------------------------------------ #
    def _on_renderer_ready(self, renderer: Renderer) -> None:
        """Подготовить ресурсы, когда Qt создал GL-контекст."""
        self._renderer = renderer
        self._sky_pass = SkyPass(self.context.active_document.sky)
        renderer.set_sky(self._sky_pass)
        self._project_hdr = bool(renderer.hdr_enabled)
        self._apply_post()
        self._resources = ResourceManager(
            roots=[_assets_root(), self.context.project_root], name="EditorAssets"
        )
        self._grid = GpuMesh.from_mesh_asset(
            line_grid_mesh(GRID_SIZE, divisions=GRID_DIVISIONS)
        )
        self._grid_material = renderer.create_material(
            "editor_grid", program="unlit", base_color=Vector3(0.30, 0.29, 0.27)
        )
        self._selection_box = GpuMesh.from_mesh_asset(wire_box_mesh(1.0))
        selection = _color_from_hex(SELECTION_COLOR)
        self._selection_material = renderer.create_material(
            "editor_selection", program="unlit", base_color=selection
        )
        self._text_batch = TextBatch()
        renderer.add_pass(TextPass(self._text_batch))
        self._physics_pass = PhysicsDebugPass(self._physics_source)
        self._physics_pass.enabled = self.show_colliders
        self._physics_pass.handle_filter = self._collider_filter
        renderer.add_pass(self._physics_pass)
        self.viewport.set_draw_provider(self._provide_draw_items)
        _log.info("Вьюпорт редактора готов: %s", renderer.info.describe())
        self.ready.emit()
        self.refresh()

    def refresh(self) -> None:
        """Запросить перерисовку кадра (и обновление предпросмотра камеры)."""
        self._preview_dirty = True
        self._apply_post()
        if self.viewport is not None:
            self.viewport.update()

    def _apply_post(self) -> None:
        """Передать рендереру пост-настройки активной сцены (``post`` документа)."""
        renderer = self._renderer
        if renderer is None:
            return
        post = self.context.active_document.post
        # Сцена с секцией ``post`` включает HDR сама (как в плеере), иначе
        # правки пост-обработки молча не давали бы эффекта.
        # Явный ``hdr: false`` в проекте уважается (баннер в инспекторе).
        hdr = getattr(self, "_project_hdr", False) or (
            post is not None and read_hdr_hint(self.context.project_root) is None
        )
        renderer.hdr_enabled = hdr
        if self.viewport is not None:
            self.viewport.post_enabled = bool(
                hdr and renderer.capabilities.supports_hdr_target
            )
        renderer.post_settings = post or PostSettings.for_quality(QualityTier.MEDIUM)
        renderer.post.assets = self.context.asset_db

    def _on_scene_changed(self, _scene: object) -> None:
        """Сбросить кэши при открытии другой сцены."""
        if self._sky_pass is not None:
            self._sky_pass.settings = self.context.active_document.sky
        self._materials.clear()
        self._meshes.clear()
        self._clear_models()
        self._gizmos.clear()
        self._light_gizmos.clear()
        self.refresh()

    # ------------------------------------------------------------------ #
    # Камеры сцены
    # ------------------------------------------------------------------ #
    def camera_components(self) -> list[CameraComponent]:
        """Все камеры-компоненты сцены в порядке объектов."""
        found: list[CameraComponent] = []
        for game_object in self.context.scene.objects:
            component = game_object.get_component(CameraComponent)
            if component is not None:
                found.append(component)
        return found

    def camera_objects(self) -> list[GameObject]:
        """Объекты, у которых есть компонент камеры."""
        return [
            component.game_object
            for component in self.camera_components()
            if component.game_object is not None
        ]

    def sync_cameras(self) -> None:
        """Перенести трансформы объектов в их камеры.

        Вызывается перед каждым кадром: камера обязана следовать за объектом,
        даже если логика компонентов в редакторе не выполняется.
        """
        for component in self.camera_components():
            component.sync_to_camera()

    def sync_lights(self) -> None:
        """Перенести трансформы объектов в их источники света.

        Как и камеры, свет обязан следовать за объектом: редактор не крутит
        игровой цикл, поэтому синхронизация идёт при сборке кадра.
        """
        for component in self.light_components():
            component.sync_to_light()
        if self._renderer is not None:
            # Менеджер света собирает источники сцены: по ним рендерер решает,
            # что и как освещать.
            self._renderer.lights.collect(self.context.scene.objects)

    def game_camera(self) -> Camera | None:
        """Первая активная игровая камера сцены (``None``, если их нет)."""
        for component in self.camera_components():
            if component.enabled and component.camera.enabled:
                return component.camera
        return None

    def add_camera_object(self, name: str = "Camera") -> GameObject:
        """Создать объект с камерой в точке интереса текущего вида.

        Args:
            name: Имя создаваемого объекта.

        Returns:
            Созданный объект с компонентом :class:`CameraComponent`.
        """
        # Начальная поза берётся из текущего вида, дальше камера объекта живёт
        # сама по себе: её не двигает ни орбита, ни зум камеры редактора.
        target = self.viewport.controller.target
        position = self.camera.position
        context = self.context
        component = CameraComponent()
        component.camera.name = name
        with context.transaction(context.tr("undo.create", name=name)):
            game_object = context.create_object(
                name, position=Vector3(position.x, position.y, position.z)
            )
            context.set_field(
                game_object,
                game_object.transform,
                "rotation",
                quaternion_from_forward(
                    Vector3(
                        target.x - position.x,
                        target.y - position.y,
                        target.z - position.z,
                    )
                ),
            )
            context.add_component(
                game_object, component, setup=lambda c: c.sync_to_camera()
            )
        return game_object

    # ------------------------------------------------------------------ #
    # Настройка гизмо
    # ------------------------------------------------------------------ #
    def set_gizmo_provider(
        self, provider: Callable[[], Sequence[tuple]] | None
    ) -> None:
        """Задать источник дополнительных элементов кадра (гизмо других типов).

        Args:
            provider: Функция без аргументов, возвращающая элементы очереди
                ``(меш, материал, матрица, примитив)``.
        """
        self._gizmo_provider = provider
        self.refresh()

    def gizmos(self) -> list[CameraGizmo]:
        """Гизмо всех камер сцены (кэшируются по объекту камеры)."""
        selected_objects = set(self.context.selection.objects)
        result: list[CameraGizmo] = []
        seen: set[int] = set()
        for component in self.camera_components():
            camera = component.camera
            seen.add(id(camera))
            gizmo = self._gizmos.get(id(camera))
            if gizmo is None:
                gizmo = CameraGizmo(
                    camera,
                    name=component.display_name,
                    style=self._gizmo_style,
                )
                self._gizmos[id(camera)] = gizmo
            gizmo.name = component.display_name
            gizmo.selected = component.game_object in selected_objects
            result.append(gizmo)
        # Подчистить гизмо удалённых камер, чтобы словарь не рос.
        for key in [key for key in self._gizmos if key not in seen]:
            self._gizmos.pop(key, None)
        return result

    def _gizmo_material(self, color: Vector3, alpha: float) -> Material | None:
        """Материал линий гизмо (кэшируется по цвету и прозрачности)."""
        if self._renderer is None:
            return None
        key = (round(color.x, 3), round(color.y, 3), round(color.z, 3), round(alpha, 2))
        material = self._gizmo_materials.get(key)
        if material is None:
            material = self._renderer.create_material(
                f"gizmo_{len(self._gizmo_materials)}",
                program="unlit",
                base_color=color,
                transparent=alpha < 1.0,
                alpha=alpha,
            )
            self._gizmo_materials[key] = material
        return material

    def _camera_gizmo_items(self) -> list[tuple]:
        """Собрать элементы отрисовки гизмо камер."""
        if not self.show_camera_gizmos or self._renderer is None:
            return []
        viewer = self.viewport.camera.position
        items: list[tuple] = []
        for gizmo in self.gizmos():
            icon_material = self._gizmo_material(gizmo.color, 1.0)
            frustum_color = (
                gizmo.color if gizmo.selected else gizmo.frustum_color
            )
            frustum_material = self._gizmo_material(
                frustum_color, gizmo.style.frustum_alpha
            )
            items.extend(
                gizmo.draw_items(
                    icon_material,
                    frustum_material,
                    viewer_position=viewer,
                    show_frustum=gizmo.selected or self.show_camera_frustum,
                )
            )
        return items

    # ------------------------------------------------------------------ #
    # Источники света
    # ------------------------------------------------------------------ #
    def light_components(self) -> list[LightComponent]:
        """Все компоненты света сцены в порядке объектов."""
        found: list[LightComponent] = []
        for game_object in self.context.scene.objects:
            component = game_object.get_component(LightComponent)
            if component is not None:
                found.append(component)
        return found

    def light_gizmos(self) -> list[LightGizmo]:
        """Гизмо всех источников сцены (кэшируются по объекту света)."""
        selected_objects = set(self.context.selection.objects)
        result: list[LightGizmo] = []
        seen: set[int] = set()
        for component in self.light_components():
            light = component.light
            seen.add(id(light))
            gizmo = self._light_gizmos.get(id(light))
            owner = component.game_object
            if gizmo is None:
                gizmo = LightGizmo(
                    light,
                    name=component.display_name,
                    style=self._light_gizmo_style,
                )
                self._light_gizmos[id(light)] = gizmo
            gizmo.name = component.display_name
            gizmo.selected = owner in selected_objects
            result.append(gizmo)
        for key in [key for key in self._light_gizmos if key not in seen]:
            self._light_gizmos.pop(key, None)
        return result

    def _light_gizmo_items(self) -> list[tuple]:
        """Собрать элементы отрисовки гизмо источников света."""
        if not self.show_light_gizmos or self._renderer is None:
            return []
        viewer = self.viewport.camera.position
        items: list[tuple] = []
        for gizmo in self.light_gizmos():
            icon_material = self._gizmo_material(gizmo.color, 1.0)
            wire_material = self._gizmo_material(gizmo.range_color, 0.55)
            items.extend(
                gizmo.draw_items(
                    icon_material,
                    wire_material,
                    viewer_position=viewer,
                    # Зона влияния нужна у выделенного света или по флагу меню:
                    # иначе десяток конусов закроет сцену.
                    show_range=gizmo.selected or self.show_light_ranges,
                )
            )
        return items

    def light_gizmo_at(self, x: float, y: float) -> Light | None:
        """Найти источник, чьё гизмо попадает под точку экрана.

        Args:
            x: Координата курсора по горизонтали.
            y: Координата курсора по вертикали.

        Returns:
            Ближайший к наблюдателю источник или ``None``.
        """
        width = max(1, int(self.viewport.width()))
        height = max(1, int(self.viewport.height()))
        origin, direction = self.viewport.camera.ray_from_screen(x, y, width, height)
        viewer = self.viewport.camera.position
        hits: list[tuple[float, Light]] = []
        for gizmo in self.light_gizmos():
            if gizmo.pick(origin, direction, viewer):
                hits.append(((gizmo.position - viewer).length, gizmo.light))
        if not hits:
            return None
        hits.sort(key=lambda entry: entry[0])
        return hits[0][1]

    def light_object_of(self, light: Light) -> GameObject | None:
        """Объект-владелец источника света."""
        for component in self.light_components():
            if component.light is light:
                return component.game_object
        return None

    def select_light_at(self, x: float, y: float) -> GameObject | None:
        """Выбрать источник кликом по его гизмо."""
        light = self.light_gizmo_at(x, y)
        if light is None:
            return None
        game_object = self.light_object_of(light)
        if game_object is not None:
            self.context.selection.select(game_object)
        return game_object

    def set_light_gizmos_visible(self, visible: bool) -> None:
        """Показать или скрыть гизмо источников света."""
        self.show_light_gizmos = bool(visible)
        self.refresh()

    def set_light_ranges_visible(self, visible: bool) -> None:
        """Показывать ли зоны влияния у всех источников, а не только у выбранного."""
        self.show_light_ranges = bool(visible)
        self.refresh()

    @property
    def show_colliders(self) -> bool:
        """Рисуются ли каркасы коллайдеров (режим не ``off``)."""
        return self.collider_mode != COLLIDER_MODE_OFF

    def set_collider_mode(self, mode: str) -> None:
        """Выбрать режим каркасов: ``off``, ``selected`` или ``all``."""
        if mode not in COLLIDER_MODES:
            raise ValueError(f"Неизвестный режим коллайдеров: {mode!r}")
        self.collider_mode = mode
        if self._physics_pass is not None:
            self._physics_pass.enabled = self.show_colliders
        self.refresh()

    def set_colliders_visible(self, visible: bool) -> None:
        """Совместимость: включить режим «все» или выключить каркасы."""
        self.set_collider_mode(COLLIDER_MODE_ALL if visible else COLLIDER_MODE_OFF)

    def _has_physics_components(self, objects) -> bool:
        return any(
            go.has_component(ColliderComponent)
            or go.has_component(CharacterControllerComponent)
            for go in objects
        )

    def _physics_source(self):
        """Адаптер физики текущей сцены с актуальными позами (без шага).

        В режиме редактирования мир не симулируется: каркасы лишь следуют за
        трансформами объектов. Мир создаётся, только если есть что рисовать.
        """
        if not self.show_colliders:
            return None
        scene = self.context.scene
        physics = scene.physics_if_exists
        if physics is None:
            if self.collider_mode == COLLIDER_MODE_SELECTED:
                candidates = self.context.selection.objects
            else:
                candidates = scene.objects
            if not self._has_physics_components(candidates):
                return None
            physics = scene.physics
        physics.sync_in()
        return physics

    def _collider_filter(self):
        """Handle'ы для отрисовки: выбранные объекты или ``None`` (все)."""
        if self.collider_mode != COLLIDER_MODE_SELECTED:
            return None
        physics = self.context.scene.physics_if_exists
        if physics is None:
            return frozenset()
        return physics.handles_for(self.context.selection.objects)

    def camera_gizmo_at(self, x: float, y: float) -> Camera | None:
        """Найти камеру, чьё гизмо попадает под точку экрана.

        Args:
            x: Координата курсора по горизонтали (пиксели).
            y: Координата курсора по вертикали.

        Returns:
            Ближайшая к наблюдателю камера или ``None``.
        """
        width = max(1, int(self.viewport.width()))
        height = max(1, int(self.viewport.height()))
        origin, direction = self.viewport.camera.ray_from_screen(x, y, width, height)
        viewer = self.viewport.camera.position
        hits: list[tuple[float, Camera]] = []
        for gizmo in self.gizmos():
            if gizmo.pick(origin, direction, viewer):
                hits.append(((gizmo.position - viewer).length, gizmo.camera))
        if not hits:
            return None
        hits.sort(key=lambda entry: entry[0])
        return hits[0][1]

    def camera_object_of(self, camera: Camera) -> GameObject | None:
        """Объект-владелец камеры (``None`` для камеры редактора)."""
        for component in self.camera_components():
            if component.camera is camera:
                return component.game_object
        return None

    def select_camera_at(self, x: float, y: float) -> GameObject | None:
        """Выбрать камеру кликом по её гизмо.

        Args:
            x: Координата курсора по горизонтали.
            y: Координата курсора по вертикали.

        Returns:
            Выбранный объект или ``None`, если под курсором гизмо нет.
        """
        camera = self.camera_gizmo_at(x, y)
        if camera is None:
            return None
        game_object = self.camera_object_of(camera)
        if game_object is not None:
            self.context.selection.select(game_object)
        return game_object

    def _pick_gizmo(self, x: float, y: float) -> bool:
        """Обработчик клика вьюпорта: выбирает камеру или источник света.

        Гизмо получает клик раньше орбиты — иначе выбор объекта «уводил» бы
        камеру. Камеры проверяются первыми: их иконки крупнее.

        Returns:
            ``True``, если клик поглощён выбором гизмо.
        """
        if self._renderer is None:
            return False
        if self.show_camera_gizmos and self.select_camera_at(x, y) is not None:
            return True
        return bool(self.show_light_gizmos and self.select_light_at(x, y) is not None)

    # ------------------------------------------------------------------ #
    # Гизмо трансформа и выбор объектов кликом
    # ------------------------------------------------------------------ #
    def set_gizmo_mode(self, mode: GizmoMode) -> None:
        """Сменить режим гизмо трансформа (перемещение, поворот, масштаб)."""
        if mode is self.transform_gizmo.mode:
            return
        self.transform_gizmo.set_mode(mode)
        self.gizmo_mode_changed.emit(mode.value)
        self.refresh()

    def gizmo_target(self) -> GameObject | None:
        """Объект, у которого рисуется гизмо (выделенный и живой)."""
        game_object = self.context.selection.primary
        if game_object is None or game_object.is_destroyed:
            return None
        return game_object

    def _viewport_size(self) -> tuple[int, int]:
        """Размер области вывода в пикселях."""
        return max(1, int(self.viewport.width())), max(1, int(self.viewport.height()))

    def _overlay_material(self, color: Vector3) -> Material | None:
        """Материал «поверх сцены» для гизмо трансформа (без проверки глубины)."""
        if self._renderer is None:
            return None
        key = (round(color.x, 3), round(color.y, 3), round(color.z, 3))
        material = self._overlay_materials.get(key)
        if material is None:
            material = self._renderer.create_material(
                f"transform_gizmo_{len(self._overlay_materials)}",
                program="unlit",
                base_color=color,
                transparent=True,
                double_sided=True,
                depth_test=False,
            )
            self._overlay_materials[key] = material
        return material

    def _gizmo_mesh(self, mode: GizmoMode) -> GpuMesh:
        """GPU-меш осей для режима (создаётся один раз)."""
        mesh = self._gizmo_meshes.get(mode)
        if mesh is None:
            factory = {
                GizmoMode.MOVE: arrow_mesh,
                GizmoMode.SCALE: scale_handle_mesh,
                GizmoMode.ROTATE: ring_mesh,
            }[mode]
            mesh = GpuMesh.from_mesh_asset(factory())
            self._gizmo_meshes[mode] = mesh
        return mesh

    def _transform_gizmo_items(self) -> list[tuple]:
        """Элементы отрисовки осей трансформа у выделенного объекта."""
        target = self.gizmo_target()
        if not self.show_transform_gizmo or target is None or self._renderer is None:
            return []
        gizmo = self.transform_gizmo
        mesh = self._gizmo_mesh(gizmo.mode)
        items: list[tuple] = []
        for index, matrix in gizmo.axis_matrices(target, self.viewport.camera):
            material = self._overlay_material(gizmo.color_of(index))
            if material is not None:
                items.append((mesh, material, matrix, None))
        return items

    def gizmo_press(self, x: float, y: float) -> bool:
        """Начать перетаскивание оси под курсором."""
        target = self.gizmo_target()
        if not self.show_transform_gizmo or target is None or self._renderer is None:
            return False
        width, height = self._viewport_size()
        started = self.transform_gizmo.begin(
            target, self.viewport.camera, x, y, width, height
        )
        if started:
            transform = target.transform
            self._gizmo_before = (
                target.uid,
                clone_value(transform.position),
                clone_value(transform.rotation),
                clone_value(transform.scale),
            )
        return started

    def gizmo_move(self, x: float, y: float) -> None:
        """Продолжить перетаскивание: обновить трансформ объекта."""
        target = self.gizmo_target()
        if target is None:
            return
        width, height = self._viewport_size()
        if self.transform_gizmo.drag(target, self.viewport.camera, x, y, width, height):
            self.object_transformed.emit(target)
            self.refresh()

    def gizmo_release(self) -> None:
        """Закончить перетаскивание и сообщить редактору об изменении сцены."""
        target = self.gizmo_target()
        before, self._gizmo_before = self._gizmo_before, None
        if self.transform_gizmo.end() and target is not None:
            if before is not None and before[0] == target.uid:
                self._record_gizmo_edit(target, before)
            self.transform_committed.emit(target)
        self.refresh()

    def _record_gizmo_edit(
        self, target: GameObject, before: tuple[str, Vector3, object, Vector3]
    ) -> None:
        """Записать перетаскивание гизмо одной записью истории."""
        transform = target.transform
        pairs = (
            ("position", before[1], transform.position),
            ("rotation", before[2], transform.rotation),
            ("scale", before[3], transform.scale),
        )
        label = self.context.tr("undo.edit", name=target.name)
        with self.context.transaction(label):
            for field, old, new in pairs:
                if old != new:
                    self.context.push(
                        SetFieldCommand(target, transform, field, old, new, label)
                    )

    def pick_object_at(self, x: float, y: float) -> GameObject | None:
        """Найти ближайший объект с мешем под точкой экрана (луч против AABB)."""
        if self._renderer is None:
            return None
        width, height = self._viewport_size()
        origin, direction = self.viewport.camera.ray_from_screen(x, y, width, height)
        hits: list[_PickHit] = []
        for game_object in self.context.scene.objects:
            if not game_object.active_in_hierarchy:
                continue
            component = game_object.get_component(MeshRenderer)
            if not isinstance(component, MeshRenderer) or not component.visible:
                continue
            bounds = self._bounds_for(component)
            if bounds is None:
                continue
            matrix = game_object.transform.world_matrix
            span = _ray_box_span(origin, direction, matrix, bounds)
            if span is not None:
                hits.append(
                    _PickHit(game_object, span[0], span[1], _box_volume(matrix, bounds))
                )
        return choose_pick(hits)

    def _on_viewport_click(self, x: float, y: float) -> None:
        """Клик без перетаскивания: выбрать объект под курсором (или снять выбор)."""
        picked = self.pick_object_at(x, y)
        if picked is None:
            self.context.selection.clear()
        else:
            self.context.selection.select(picked)

    # ------------------------------------------------------------------ #
    # Предпросмотр камеры
    # ------------------------------------------------------------------ #
    def preview(
        self, *, width: int = OVERLAY_SIZE[0], height: int = OVERLAY_SIZE[1]
    ) -> CameraPreview:
        """Вернуть (или создать) предпросмотр камеры.

        Предпросмотр рендерит сцену в собственную текстуру, поэтому не зависит
        ни от размера, ни от камеры вьюпорта редактора.

        Args:
            width: Ширина цели предпросмотра.
            height: Высота цели предпросмотра.

        Returns:
            Объект предпросмотра.
        """
        if self._preview is None:
            self._preview = CameraPreview(
                self.game_camera() or Camera(name="Preview"),
                width=width,
                height=height,
            )
        return self._preview

    def selected_camera(self) -> Camera | None:
        """Камера выделенного объекта (``None``, если выбрана не камера)."""
        game_object = self.context.selection.primary
        if game_object is None or game_object.is_destroyed:
            return None
        component = game_object.get_component(CameraComponent)
        return component.camera if component is not None else None

    def render_preview(self, camera: Camera | None = None) -> np.ndarray | None:
        """Отрисовать предпросмотр камеры и вернуть пиксели.

        Args:
            camera: Камера предпросмотра; ``None`` — камера выбранного объекта.

        Returns:
            Массив ``(H, W, 4)`` uint8 в GL-порядке (снизу вверх) или ``None``,
            если GL-контекст не готов или камеры нет.
        """
        if self._renderer is None:
            return None
        camera = camera or self.selected_camera()
        if camera is None:
            return None
        preview = self.preview()
        preview.set_camera(camera)
        # Без рамки выделения и подсветки: предпросмотр показывает кадр игрока.
        items = self._build_scene_items(highlight=False)
        self.viewport.makeCurrent()
        try:
            preview.render(self._renderer, items)
            return preview.read_pixels()
        except Exception as exc:  # noqa: BLE001 - превью не должно ронять редактор
            _log.error("Предпросмотр камеры не отрисован: %s", exc)
            return None
        finally:
            self.viewport.doneCurrent()

    def update_camera_overlay(self, *, force: bool = False) -> bool:
        """Обновить окно предпросмотра в углу сцены.

        Окно видно, только пока выбран объект-камера. Кадр перерисовывается,
        когда сцена менялась (``refresh``), а не по каждому тику таймера.

        Args:
            force: Перерисовать, даже если сцена не менялась.

        Returns:
            ``True``, если окно показано.
        """
        overlay = self.camera_overlay
        camera = self.selected_camera()
        if camera is None or self._renderer is None:
            overlay.hide()
            return False
        if not overlay.isVisible():
            force = True
        if not (force or self._preview_dirty):
            return True
        self._preview_dirty = False
        pixels = self.render_preview(camera)
        if pixels is None:
            overlay.hide()
            return False
        owner = self.camera_object_of(camera)
        overlay.set_title(owner.name if owner is not None else camera.name)
        overlay.set_pixels(pixels)
        overlay.show()
        overlay.reposition()
        return True

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - API Qt
        """Держать оверлей в углу при изменении размера вьюпорта."""
        if watched is self.viewport:
            kind = event.type()
            if kind == QEvent.Type.Resize:
                self.camera_overlay.reposition()
            elif kind in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
                if event.mimeData().hasFormat(PREFAB_MIME):
                    event.acceptProposedAction()
                    return True
            elif kind == QEvent.Type.Drop and event.mimeData().hasFormat(PREFAB_MIME):
                path = bytes(event.mimeData().data(PREFAB_MIME)).decode("utf-8")
                point = event.position()
                event.acceptProposedAction()
                self.drop_prefab(path, point.x(), point.y())
                return True
        return False

    def ground_point_at(self, x: float, y: float) -> Vector3:
        """Точка в сцене под курсором: попадание луча в меш или плоскость y=0.

        Если луч не попал ни в меш, ни в плоскость (смотрит вверх), точка
        берётся на расстоянии :data:`DROP_FALLBACK_DISTANCE` от камеры.
        """
        width, height = self._viewport_size()
        origin, direction = self.viewport.camera.ray_from_screen(x, y, width, height)
        best: float | None = None
        for game_object in self.context.scene.objects:
            component = game_object.get_component(MeshRenderer)
            if (
                not game_object.active_in_hierarchy
                or not isinstance(component, MeshRenderer)
                or not component.visible
            ):
                continue
            bounds = self._bounds_for(component)
            if bounds is None:
                continue
            distance = _ray_box_distance(
                origin, direction, game_object.transform.world_matrix, bounds
            )
            if distance is not None and (best is None or distance < best):
                best = distance
        if best is None and direction.y < -1e-9:
            best = -origin.y / direction.y
        if best is None or best < 0:
            best = DROP_FALLBACK_DISTANCE
        return Vector3(
            origin.x + direction.x * best,
            origin.y + direction.y * best,
            origin.z + direction.z * best,
        )

    def drop_prefab(self, path: str, x: float, y: float) -> GameObject | None:
        """Создать экземпляр префаба в точке под курсором вьюпорта.

        Низ габарита экземпляра ставится на точку попадания (для объекта со
        мешами это «стоит на поверхности»), без мешей - центр в точке.

        Returns:
            Корень экземпляра или ``None`` (ошибка показана в логе).
        """
        point = self.ground_point_at(x, y)
        context = self.context
        label = context.tr("undo.create", name=Path(path).name)
        try:
            with context.transaction(label):
                root = instantiate_prefab_file(context, path, position=point)
                low = self._world_min_y(root)
                if low is not None:
                    position = root.transform.position
                    context.set_field(
                        root,
                        root.transform,
                        "position",
                        Vector3(position.x, position.y + point.y - low, position.z),
                    )
        except (PrefabError, OSError) as exc:
            _log.error("Префаб '%s' не создан: %s", path, exc)
            return None
        return root

    def _world_min_y(self, root: GameObject) -> float | None:
        """Нижняя мировая координата Y мешей поддерева или ``None``."""
        lows: list[float] = []
        stack = [root]
        while stack:
            node = stack.pop()
            stack.extend(node.children)
            component = node.get_component(MeshRenderer)
            if not isinstance(component, MeshRenderer):
                continue
            bounds = self._bounds_for(component)
            if bounds is None:
                continue
            matrix = np.asarray(node.transform.world_matrix, dtype=float).reshape(4, 4)
            for cx in (bounds[0].x, bounds[1].x):
                for cy in (bounds[0].y, bounds[1].y):
                    for cz in (bounds[0].z, bounds[1].z):
                        lows.append(float((matrix @ np.array([cx, cy, cz, 1.0]))[1]))
        return min(lows) if lows else None

    def dispose_preview(self) -> None:
        """Освободить ресурсы предпросмотра (при закрытии редактора)."""
        self._overlay_timer.stop()
        self.camera_overlay.hide()
        if self._preview is None:
            return
        self.viewport.makeCurrent()
        try:
            self._preview.dispose()
        finally:
            self.viewport.doneCurrent()
        self._preview = None

    def set_camera_gizmos_visible(self, visible: bool) -> None:
        """Показать или скрыть гизмо камер в сцене."""
        self.show_camera_gizmos = bool(visible)
        self.refresh()

    def set_camera_frustum_visible(self, visible: bool) -> None:
        """Показывать ли фрустум у всех камер, а не только у выделенной."""
        self.show_camera_frustum = bool(visible)
        self.refresh()

    # ------------------------------------------------------------------ #
    # Построение списка отрисовки
    # ------------------------------------------------------------------ #
    def _provide_draw_items(self) -> list[tuple]:
        """Собрать объекты кадра из ECS-сцены.

        Returns:
            Список кортежей ``(меш, материал, матрица, примитив)``.
        """
        items: list[tuple] = []
        if self._renderer is None:
            return items

        if self._grid is not None and self._grid_material is not None:
            from OpenGL import GL

            items.append(
                (
                    self._grid,
                    self._grid_material,
                    compose_matrix(),
                    GL.GL_LINES,
                )
            )

        items.extend(self._build_scene_items())
        items.extend(self._camera_gizmo_items())
        items.extend(self._light_gizmo_items())
        items.extend(self._transform_gizmo_items())
        if self._gizmo_provider is not None:
            items.extend(self._gizmo_provider())
        return items

    def _build_scene_items(self, *, highlight: bool = True) -> list[tuple]:
        """Объекты сцены без служебной геометрии редактора.

        Этот список используется и вьюпортом, и предпросмотром камеры: так
        предпросмотр показывает ровно то, что увидит игровая камера.

        Args:
            highlight: Подсвечивать ли выделенные объекты (в предпросмотре
                камеры подсветка выключена).

        Returns:
            Список ``(меш, материал, матрица, примитив)``.
        """
        items: list[tuple] = []
        if self._renderer is None:
            return items

        self.sync_cameras()
        self.sync_lights()
        if self._text_batch is not None:
            fill_text_batch(
                self._text_batch,
                self.context.scene.objects,
                (self.context.project_root,),
                self.context.scene,
            )
        selected = set(self.context.selection.objects) if highlight else set()
        for game_object in self.context.scene.objects:
            if not game_object.active_in_hierarchy:
                continue
            component = game_object.get_component(MeshRenderer)
            if not isinstance(component, MeshRenderer) or not component.visible:
                continue
            parts = self._parts_for(component)
            if not parts:
                continue
            highlighted = game_object in selected
            matrix = game_object.transform.world_matrix
            flags = (component.cast_shadow, component.receive_shadow)
            for mesh, pbr in parts:
                material = pbr or self._material_for(component, highlighted)
                items.append((mesh, material, matrix, None, flags))
            if highlighted and self._selection_box is not None:
                bounds = self._bounds_for(component)
                if bounds is not None:
                    items.extend(self._selection_frame(game_object, bounds))
        return items

    def _selection_frame(
        self, game_object: GameObject, bounds: tuple[Vector3, Vector3]
    ) -> list[tuple]:
        """Собрать рамку выделения вокруг объекта.

        Args:
            game_object: Выделенный объект.
            bounds: Локальные габариты его мешей.

        Returns:
            Список из одного элемента очереди отрисовки (линии).
        """
        from OpenGL import GL

        if self._selection_box is None or self._selection_material is None:
            return []
        minimum, maximum = bounds
        size = maximum - minimum
        center = (minimum + maximum) * 0.5
        scale = Vector3(
            max(size.x, 1e-3) * 1.06, max(size.y, 1e-3) * 1.06, max(size.z, 1e-3) * 1.06
        )
        matrix = game_object.transform.world_matrix.astype(MATRIX_DTYPE)
        offset = compose_matrix(position=center, scale=scale)
        return [
            (
                self._selection_box,
                self._selection_material,
                matrix @ offset,
                GL.GL_LINES,
            )
        ]

    def _clear_models(self) -> None:
        """Сбросить кэш моделей из ``Library/`` (GPU-ресурсы освободит GC)."""
        self._models.clear()
        self._model_failed.clear()
        self._pbr.clear()
        self._pbr_textures.clear()

    def _on_imports_changed(self) -> None:
        """Импорт изменился: недогруженные модели пробуем снова."""
        if self._model_failed or self._models:
            self._clear_models()
        self.refresh()

    def _pbr_material(self, guid: str, artifact: str):
        """PBR-материал артефакта модели (кэшируется, ``None`` при ошибке)."""
        key = (guid, artifact)
        if key not in self._pbr:
            material = None
            assets = self.context.asset_db
            if assets is not None and self._renderer is not None:
                try:
                    material = load_pbr_material(
                        self._renderer,
                        assets,
                        guid,
                        artifact_name=artifact,
                        textures=self._pbr_textures,
                    )
                except Exception as exc:  # noqa: BLE001 - кадр не должен падать
                    _log.error("Материал %s/%s не загружен: %s", guid, artifact, exc)
            self._pbr[key] = material
        return self._pbr[key]

    def _model_guid(self, component: MeshRenderer) -> str:
        """GUID импортированной модели компонента (``""`` — это не модель).

        Встроенные меши и ``.obj`` по пути грузятся напрямую (:meth:`_mesh_for`);
        остальные форматы (glTF, FBX) приходят из ``Library/``.
        """
        if component.mesh_guid:
            return component.mesh_guid
        key = component.mesh
        if key in BUILTIN_MESH_FACTORIES or Path(key).suffix.lower() in ("", ".obj"):
            return ""
        assets = self.context.asset_db
        if assets is None:
            return ""
        path = Path(key)
        if not path.is_absolute():
            path = self.context.project_root / path
        return assets.guid_of(path) or ""

    def _parts_for(self, component: MeshRenderer) -> list[tuple[GpuMesh, object]]:
        """Меши компонента: ``[(меш, материал)]``; материал ``None`` — простой.

        Пусто, если меш не загрузился (ассет ещё импортируется или повреждён).
        """
        guid = self._model_guid(component)
        if not guid:
            mesh = self._mesh_for(component.mesh)
            return [] if mesh is None else [(mesh, None)]
        parts = self._models.get(guid)
        if parts is not None:
            return parts
        assets = self.context.asset_db
        if assets is None or self._renderer is None or guid in self._model_failed:
            return []
        try:
            parts = load_cooked_parts(assets, guid, self._pbr_material)
        except Exception as exc:  # noqa: BLE001 - редактор не должен падать
            self._model_failed.add(guid)
            _log.error("Модель %s не загружена: %s", guid, exc)
            return []
        self._models[guid] = parts
        return parts

    def _bounds_for(self, component: MeshRenderer) -> tuple[Vector3, Vector3] | None:
        """Локальные габариты всех частей компонента или ``None``."""
        parts = self._parts_for(component)
        if not parts:
            return None
        lows = [mesh.bounds[0] for mesh, _ in parts]
        highs = [mesh.bounds[1] for mesh, _ in parts]
        return (
            Vector3(
                min(v.x for v in lows), min(v.y for v in lows), min(v.z for v in lows)
            ),
            Vector3(
                max(v.x for v in highs),
                max(v.y for v in highs),
                max(v.z for v in highs),
            ),
        )

    def _mesh_for(self, key: str) -> GpuMesh | None:
        """Получить GPU-меш по ключу (встроенный или файл ассета).

        Args:
            key: ``cube``, ``sphere``, ``plane`` или путь к файлу.

        Returns:
            Меш или ``None``, если ассет не удалось загрузить.
        """
        cached = self._meshes.get(key)
        if cached is not None:
            return cached
        try:
            factory = BUILTIN_MESH_FACTORIES.get(key)
            if factory is not None:
                mesh = GpuMesh.from_mesh_asset(factory())
            else:
                assert self._resources is not None
                asset = self._resources.load(key)
                mesh = GpuMesh.from_mesh_asset(asset)
        except Exception as exc:  # noqa: BLE001 - редактор не должен падать
            _log.error("Не удалось загрузить меш '%s': %s", key, exc)
            return None
        self._meshes[key] = mesh
        return mesh

    def _material_for(self, component: MeshRenderer, highlighted: bool) -> Material:
        """Получить материал под параметры компонента (с кэшем).

        Args:
            component: Компонент отрисовки.
            highlighted: Выделен ли объект (добавляется подсветка).

        Returns:
            Материал рендерера.
        """
        color = component.color
        if highlighted:
            # Подсветка выделения: цвет смещается к акцентному.
            accent = _color_from_hex(SELECTION_COLOR)
            color = Vector3(
                min(1.0, color.x * 0.55 + accent.x * 0.45),
                min(1.0, color.y * 0.55 + accent.y * 0.45),
                min(1.0, color.z * 0.55 + accent.z * 0.45),
            )
        key = (
            color.x,
            color.y,
            color.z,
            component.roughness,
            component.metallic,
            highlighted,
        )
        material = self._materials.get(key)
        if material is None:
            assert self._renderer is not None
            material = self._renderer.create_material(
                f"object_{len(self._materials)}",
                base_color=color,
                roughness=component.roughness,
                metallic=component.metallic,
            )
            self._materials[key] = material
        return material

    # ------------------------------------------------------------------ #
    # Камера и статистика
    # ------------------------------------------------------------------ #
    def frame_selected(self) -> bool:
        """Навести камеру на выделенный объект.

        Returns:
            ``True``, если объект найден и камера перемещена.
        """
        game_object = self.context.selection.primary
        if game_object is None:
            return False
        component = game_object.get_component(MeshRenderer)
        bounds = self._bounds_for(component) if component is not None else None
        radius = 1.0
        if bounds is not None:
            minimum, maximum = bounds
            radius = max((maximum - minimum).length * 0.5, 0.2)
        self.viewport.focus_on(game_object.transform.world_position, radius)
        return True

    def reset_camera(self) -> None:
        """Вернуть камеру в исходное положение."""
        self.viewport.reset_view()

    def camera_distance(self) -> float:
        """Дистанция от камеры до точки интереса."""
        controller = self.viewport.controller
        return (self.viewport.camera.position - controller.target).length

    def _on_frame_rendered(self, fps: float) -> None:
        """Передать статистику кадра подписчикам."""
        stats: RenderStats | None = self.viewport.stats
        self.stats_updated.emit(fps, stats)

    def render_stats(self) -> RenderStats | None:
        """Статистика последнего кадра."""
        return self.viewport.stats

    def current_fps(self) -> float:
        """Сглаженный FPS вьюпорта."""
        return float(self.viewport.fps)

    def driver_description(self) -> str:
        """Строка с описанием драйвера (или пустая строка)."""
        if self._renderer is None:
            return ""
        return self._renderer.info.describe()


def _assets_root() -> Path:
    """Каталог ресурсов движка (``assets``)."""
    return Path(__file__).resolve().parents[2] / "assets"


def _shader_root() -> Path:
    """Каталог шейдеров по умолчанию."""
    return _assets_root()


def _color_from_hex(value: str) -> Vector3:
    """Преобразовать ``#rrggbb`` в вектор цвета ``0..1``."""
    text = value.lstrip("#")
    return Vector3(
        int(text[0:2], 16) / 255.0,
        int(text[2:4], 16) / 255.0,
        int(text[4:6], 16) / 255.0,
    )


class _GizmoDragHandler:
    """Мост между мышью ``ViewportWidget`` и гизмо трансформа вьюпорта."""

    def __init__(self, viewport: EditorViewport) -> None:
        self._viewport = viewport

    def press(self, x: float, y: float) -> bool:
        return self._viewport.gizmo_press(x, y)

    def move(self, x: float, y: float) -> None:
        self._viewport.gizmo_move(x, y)

    def release(self) -> None:
        self._viewport.gizmo_release()


@dataclass(frozen=True)
class _PickHit:
    """Попадание луча в AABB объекта: вход, выход и объём коробки."""

    game_object: GameObject
    enter: float
    exit: float
    volume: float


#: Вложенный объект выбирается вместо внешнего, если его коробка меньше во
#: столько раз (иначе соседние перекрывающиеся объекты не «дрожат» при клике).
_NESTED_PICK_RATIO = 0.5


def choose_pick(hits: list[_PickHit]) -> GameObject | None:
    """Выбрать объект среди попаданий луча в AABB.

    Базово побеждает ближайший по входу луча. Но огромная коробка (уровень),
    внутри которой стоят мелкие объекты, всегда перехватывала бы клик: луч
    входит в неё раньше. Поэтому, если луч входит в меньшую коробку, пока он
    ещё внутри текущей победившей (и та меньше хотя бы вдвое), выбирается
    вложенная.

    Args:
        hits: Попадания (порядок любой).

    Returns:
        Выбранный объект или ``None``.
    """
    if not hits:
        return None
    ordered = sorted(hits, key=lambda hit: hit.enter)
    best = ordered[0]
    for hit in ordered[1:]:
        if hit.enter > best.exit:
            break
        if hit.volume < best.volume * _NESTED_PICK_RATIO:
            best = hit
    return best.game_object


def _box_volume(world_matrix: np.ndarray, bounds: tuple[Vector3, Vector3]) -> float:
    """Объём AABB меша в мировых единицах (с учётом масштаба объекта)."""
    matrix = np.asarray(world_matrix, dtype=float).reshape(4, 4)
    size = np.array(bounds[1].to_tuple(), dtype=float) - np.array(
        bounds[0].to_tuple(), dtype=float
    )
    return float(abs(np.linalg.det(matrix[:3, :3])) * np.prod(np.abs(size)))


def _ray_box_distance(
    origin: Vector3,
    direction: Vector3,
    world_matrix: np.ndarray,
    bounds: tuple[Vector3, Vector3],
) -> float | None:
    """Расстояние до входа луча в AABB меша (см. :func:`_ray_box_span`)."""
    span = _ray_box_span(origin, direction, world_matrix, bounds)
    return None if span is None else span[0]


def _ray_box_span(
    origin: Vector3,
    direction: Vector3,
    world_matrix: np.ndarray,
    bounds: tuple[Vector3, Vector3],
) -> tuple[float, float] | None:
    """Интервал луча внутри AABB меша в локальных осях объекта.

    Луч переводится в локальное пространство объекта и проверяется «слэбами»:
    так масштаб и поворот объекта учитываются без построения OBB.

    Returns:
        ``(вход, выход)`` вдоль луча в мировых единицах или ``None``, если промах.
    """
    matrix = np.asarray(world_matrix, dtype=float).reshape(4, 4)
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None
    o = inverse @ np.array([origin.x, origin.y, origin.z, 1.0])
    d = inverse @ np.array([direction.x, direction.y, direction.z, 0.0])
    low = np.array(bounds[0].to_tuple(), dtype=float)
    high = np.array(bounds[1].to_tuple(), dtype=float)
    t_min, t_max = 0.0, float("inf")
    for axis in range(3):
        if abs(d[axis]) < 1e-12:
            if o[axis] < low[axis] or o[axis] > high[axis]:
                return None
            continue
        t1 = (low[axis] - o[axis]) / d[axis]
        t2 = (high[axis] - o[axis]) / d[axis]
        t_min = max(t_min, min(t1, t2))
        t_max = min(t_max, max(t1, t2))
        if t_min > t_max:
            return None
    # Параметр в локальных единицах совпадает с мировым: направление луча
    # преобразовано той же матрицей без нормализации.
    return t_min, t_max
