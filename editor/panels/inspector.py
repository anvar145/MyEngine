"""Панель инспектора: свойства выделенного объекта.

Инспектор собирается динамически: секции «Объект», «Трансформ» и «Материал»
строятся заново при смене выделения или языка. Значения применяются к ECS
сразу при вводе, поэтому изменения видны во вьюпорте без подтверждения.

Example:
    >>> panel = InspectorPanel(context)
    >>> context.selection.select(cube)     # инспектор сам перестроится
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from editor.panels.auto_section import (
    AutoSection,
    ComponentSection,
    has_auto_section,
)
from editor.panels.base import DockPanel
from editor.panels.camera_section import CameraSection
from editor.panels.import_settings import ImportSettingsSection
from editor.panels.light_section import LightSection
from editor.panels.physics_sections import (
    CharacterControllerSection,
    ColliderSection,
    RigidBodySection,
)
from editor.panels.post_section import PostSection
from editor.panels.prefab_section import PrefabInstanceSection
from editor.panels.script_fields import (
    ReferenceField,
    ScriptFieldsForm,
    make_asset_ref,
)
from editor.panels.script_picker import ScriptPicker
from editor.panels.sky_section import SkySection
from editor.panels.text_section import TextSection
from editor.panels.ui_sections import UISection
from editor.prefab_actions import apply_override_to_prefab, revert_instance_override
from editor.project import scripts_root_for
from editor.theme import icon
from editor.widgets import (
    ColorButton,
    FocusComboBox,
    FocusDoubleSpinBox,
    FocusSpinBox,
    SectionBox,
    VectorEditor,
    fit_widths,
)
from engine.audio.components import AudioListenerComponent, AudioSourceComponent
from engine.core.commands import SetScriptFieldCommand
from engine.core.components import BUILTIN_MESHES, MeshRenderer, TextRenderer
from engine.core.game_object import GameObject
from engine.core.math3d import Quaternion, Vector3, quaternion_to_euler
from engine.core.physics_components import (
    CharacterControllerComponent,
    ColliderComponent,
    MeshColliderComponent,
    RigidBodyComponent,
)
from engine.core.prefab_components import PrefabInstance
from engine.core.resources import AssetRef
from engine.core.save.components import AutoSavePolicy, CheckpointComponent, SafeZone
from engine.core.ui_components import UIButton, UIElement, UIImage, UIPanel, UIText
from engine.prefab import PrefabError, find_instance, get_overrides
from engine.rendering.camera import CameraComponent
from engine.rendering.lighting import LightComponent, create_light
from engine.scripting import ScriptComponent

__all__ = ["COMPONENT_GROUPS", "ComponentEntry", "ComponentGroup", "InspectorPanel"]


#: Компоненты с ручной секцией: авто-секция по дескрипторам для них не строится.
_MANUAL_SECTION_TYPES: frozenset[type] = frozenset(
    {RigidBodyComponent, ColliderComponent, CharacterControllerComponent}
)

#: Взаимоисключающие физические компоненты (CharacterController ↔ RigidBody).
_PHYSICS_CONFLICTS: dict[object, type] = {
    RigidBodyComponent: CharacterControllerComponent,
    CharacterControllerComponent: RigidBodyComponent,
}


@dataclass(frozen=True)
class ComponentEntry:
    """Пункт меню «Добавить компонент».

    Attributes:
        kind: Семейство: ``camera``, ``text``, ``ui``, ``light`` или ``physics``.
        label_key: Ключ перевода подписи.
        icon: Имя иконки из ``assets/icons``.
        arg: Параметр обработчика (класс компонента или тип света).
    """

    kind: str
    label_key: str
    icon: str
    arg: object = None


@dataclass(frozen=True)
class ComponentGroup:
    """Подменю («сноска») меню «Добавить компонент».

    Attributes:
        key: Суффикс ключа перевода ``inspector.group.<key>``.
        icon: Иконка заголовка группы.
        entries: Пункты группы.
    """

    key: str
    icon: str
    entries: tuple[ComponentEntry, ...]


#: Группы меню «Добавить компонент»: Рендеринг, UI, Освещение, Физика.
COMPONENT_GROUPS: tuple[ComponentGroup, ...] = (
    ComponentGroup(
        "rendering",
        "layers",
        (
            ComponentEntry("camera", "inspector.component.camera", "camera"),
            ComponentEntry("text", "inspector.component.text", "text"),
        ),
    ),
    ComponentGroup(
        "ui",
        "panel-top",
        (
            ComponentEntry("ui", "inspector.component.ui_panel", "panel-top", UIPanel),
            ComponentEntry("ui", "inspector.component.ui_image", "image", UIImage),
            ComponentEntry(
                "ui", "inspector.component.ui_button", "mouse-pointer-click", UIButton
            ),
            ComponentEntry("ui", "inspector.component.ui_text", "text", UIText),
        ),
    ),
    ComponentGroup(
        "lighting",
        "lightbulb",
        (
            ComponentEntry("light", "light.kind.directional", "sun", "directional"),
            ComponentEntry("light", "light.kind.point", "lightbulb", "point"),
            ComponentEntry("light", "light.kind.spot", "flashlight", "spot"),
            ComponentEntry("light", "light.kind.ambient", "sun-dim", "ambient"),
        ),
    ),
    ComponentGroup(
        "physics",
        "atom",
        (
            ComponentEntry(
                "physics",
                "inspector.physics.add.rigidbody",
                "atom",
                RigidBodyComponent,
            ),
            ComponentEntry(
                "physics",
                "inspector.physics.add.collider",
                "box-select",
                ColliderComponent,
            ),
            ComponentEntry(
                "physics",
                "inspector.physics.add.character",
                "person-standing",
                CharacterControllerComponent,
            ),
            ComponentEntry(
                "physics",
                "inspector.physics.add.mesh_collider",
                "box",
                MeshColliderComponent,
            ),
        ),
    ),
    ComponentGroup(
        "saves",
        "save",
        (
            ComponentEntry(
                "save", "inspector.saves.add.checkpoint", "save", CheckpointComponent
            ),
            ComponentEntry(
                "save", "inspector.saves.add.safe_zone", "box-select", SafeZone
            ),
            ComponentEntry(
                "save", "inspector.saves.add.autosave", "clock", AutoSavePolicy
            ),
        ),
    ),
    ComponentGroup(
        "audio",
        "volume-2",
        (
            ComponentEntry(
                "audio",
                "inspector.audio.add.source",
                "volume-2",
                AudioSourceComponent,
            ),
            ComponentEntry(
                "audio",
                "inspector.audio.add.listener",
                "ear",
                AudioListenerComponent,
            ),
        ),
    ),
)


@dataclass(frozen=True)
class _InstanceInfo:
    """Экземпляр префаба, которому принадлежит выбранный объект.

    Attributes:
        root: Корень экземпляра.
        path: Путь выбранного объекта внутри префаба (``local_id``).
        all: Все живые overrides экземпляра.
        own: Overrides выбранного объекта.
    """

    root: GameObject
    path: str
    all: list[dict]
    own: list[dict]


def _mark_bold(widget: QWidget, tooltip: str) -> None:
    """Выделить жирным подпись поля, заданного override."""
    font = widget.font()
    font.setBold(True)
    widget.setFont(font)
    widget.setToolTip(tooltip)

_MODEL_SUFFIXES = frozenset({".obj", ".gltf", ".glb", ".fbx"})


class InspectorPanel(DockPanel):
    """Редактор свойств выделенного объекта.

    Signals:
        material_changed: Изменён материал — вьюпорт должен перерисоваться.
        transform_changed: Изменён трансформ объекта.
    """

    material_changed = Signal()
    transform_changed = Signal()

    def __init__(self, context, parent: QWidget | None = None) -> None:
        super().__init__("panel.inspector", context, parent)
        self._game_object: GameObject | None = None
        self._asset_path: Path | None = None
        self._asset_section: ImportSettingsSection | None = None
        self._viewport = None
        self.rigidbody_section: RigidBodySection | None = None
        self.collider_section: ColliderSection | None = None
        self.character_section: CharacterControllerSection | None = None
        self.prefab_section: PrefabInstanceSection | None = None
        self.auto_sections: list[AutoSection] = []
        # (компонент, поле) -> виджет подписи; подсвечиваются при override
        self._marks: dict[tuple[str, str], QWidget | None] = {}
        self._section_marks: dict[str, QWidget] = {}
        self._instance: _InstanceInfo | None = None

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._content = QWidget(self._scroll)
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(10, 10, 10, 10)
        self._content_layout.setSpacing(10)
        self._content_layout.setAlignment(Qt.AlignTop)
        self._scroll.setWidget(self._content)
        self.setWidget(self._scroll)

        self._placeholder = self._make_placeholder()
        self._game_object: GameObject | None = None
        self._placeholder.setParent(None)

        self.context.selection.changed.connect(self._on_selection_changed)
        self.context.imports_changed.connect(self._on_imports_changed)
        self.context.imports_restarted.connect(self._on_imports_restarted)
        self.refresh()

    # ------------------------------------------------------------------ #
    # Состояние
    # ------------------------------------------------------------------ #
    @property
    def game_object(self) -> GameObject | None:
        """Объект, свойства которого показывает инспектор."""
        return self._game_object

    def connect_viewport(self, viewport) -> None:
        """Подключить вьюпорт (нужен секциям камеры и света)."""
        self._viewport = viewport

    def retranslate(self, *_args: object) -> None:
        """Перевести панель и перестроить поля."""
        super().retranslate()
        self.refresh()

    def sync_transform(self) -> None:
        """Обновить поля трансформа на месте (без перестройки панели).

        Нужно, когда объект двигают вне инспектора (гизмо во вьюпорте): полная
        перестройка на каждом кадре перетаскивания сбивала бы фокус полей.
        """
        game_object = self._game_object
        if game_object is None or game_object.is_destroyed:
            return
        editors = (
            getattr(self, "position_editor", None),
            getattr(self, "rotation_editor", None),
            getattr(self, "scale_editor", None),
        )
        if any(editor is None for editor in editors):
            return
        transform = game_object.transform
        position, rotation, scale = editors
        position.set_value(transform.position)
        rotation.set_value(quaternion_to_euler(transform.rotation))
        scale.set_value(transform.scale)

    def _on_selection_changed(self, objects: list[GameObject]) -> None:
        """Обновить панель при смене выделения."""
        self._game_object = objects[0] if objects else None
        if self._game_object is not None:
            self._asset_path = None
        self.refresh()

    def show_asset(self, path: Path | str | None) -> None:
        """Показать настройки импорта файла (``None`` — вернуться к объекту).

        Args:
            path: Исходный файл ассета, выбранный в панели ассетов.
        """
        self._asset_path = None if path is None else Path(path)
        self.refresh()

    @property
    def asset_section(self) -> ImportSettingsSection | None:
        """Секция настроек импорта, если инспектор показывает ассет."""
        return self._asset_section

    def _on_imports_changed(self) -> None:
        """Обновить строку состояния импорта открытого ассета."""
        if self._asset_section is not None:
            self._asset_section.refresh_state()

    def _on_imports_restarted(self) -> None:
        """Менеджер импортов пересоздан: форма ссылалась на старый."""
        if self._asset_path is not None:
            self.refresh()

    def refresh(self) -> None:
        """Перестроить содержимое инспектора."""
        self._populate()
        fit_widths(self._content)

    def _populate(self) -> None:
        """Собрать секции для текущего выделения."""
        self._clear()
        self._asset_section = None
        if (
            self._asset_path is not None
            and self._game_object is None
            and self._asset_path.is_file()
        ):
            title = QLabel(self._asset_path.name, self._content)
            title.setProperty("role", "section")
            self._asset_section = ImportSettingsSection(
                self.context, self._asset_path, self._content
            )
            self._content_layout.addWidget(title)
            self._content_layout.addWidget(self._asset_section)
            return
        if self._game_object is None or self._game_object.is_destroyed:
            self._content_layout.addWidget(self._placeholder)
            self._placeholder.setText(self.tr("inspector.no_selection"))
            return
        self._marks.clear()
        self._section_marks.clear()
        self._instance = self._resolve_instance()
        self._build_object_section()
        self._build_transform_section()
        self._build_material_section()
        self._build_camera_section()
        self._build_light_section()
        self._build_text_section()
        self._build_ui_section()
        self._build_physics_sections()
        self._build_auto_sections()
        self._build_prefab_section()
        self._build_sky_section()
        self._build_post_section()
        self._build_scripts_section()
        self._build_add_component_button()
        self._apply_override_marks()

    # ------------------------------------------------------------------ #
    # Построение секций
    # ------------------------------------------------------------------ #
    def _build_object_section(self) -> None:
        """Секция «Объект»: имя, тег, активность."""
        assert self._game_object is not None
        section = SectionBox(self.tr("inspector.section.object"), self._content)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setHorizontalSpacing(12)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        name_edit = QLineEdit(self._game_object.name, section)
        name_edit.editingFinished.connect(
            lambda: self._apply_name(name_edit.text())
        )
        form.addRow(self.tr("inspector.name"), name_edit)

        tag_edit = QLineEdit(self._game_object.tag, section)
        tag_edit.editingFinished.connect(
            lambda: self._apply_tag(tag_edit.text())
        )
        form.addRow(self.tr("inspector.tag"), tag_edit)

        active = QCheckBox(self.tr("inspector.active"), section)
        active.setChecked(self._game_object.active_self)
        active.toggled.connect(self._apply_active)
        form.addRow("", active)
        self._marks[("Root", "name")] = form.labelForField(name_edit)
        self._marks[("Root", "tag")] = form.labelForField(tag_edit)
        self._marks[("Root", "active")] = active

        section.add_layout(form)
        self._content_layout.addWidget(section)
        self._name_edit = name_edit
        self._tag_edit = tag_edit

    def _build_transform_section(self) -> None:
        """Секция «Трансформ»: позиция, поворот (Эйлер), масштаб."""
        assert self._game_object is not None
        transform = self._game_object.transform
        section = SectionBox(self.tr("inspector.section.transform"), self._content)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setHorizontalSpacing(12)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.position_editor = VectorEditor(step=0.1, parent=section)
        self.position_editor.set_value(transform.position)
        self.position_editor.value_changed.connect(self._apply_position)
        form.addRow(self.tr("inspector.position"), self.position_editor)

        self.rotation_editor = VectorEditor(
            minimum=-360.0, maximum=360.0, step=5.0, decimals=2, parent=section
        )
        self.rotation_editor.set_value(quaternion_to_euler(transform.rotation))
        self.rotation_editor.value_changed.connect(self._apply_rotation)
        form.addRow(self.tr("inspector.rotation"), self.rotation_editor)

        self.scale_editor = VectorEditor(
            minimum=0.001, maximum=1000.0, step=0.1, parent=section
        )
        self.scale_editor.set_value(transform.scale)
        self.scale_editor.value_changed.connect(self._apply_scale)
        form.addRow(self.tr("inspector.scale"), self.scale_editor)
        for name, editor in (
            ("position", self.position_editor),
            ("rotation", self.rotation_editor),
            ("scale", self.scale_editor),
        ):
            self._marks[("Transform", name)] = form.labelForField(editor)

        section.add_layout(form)
        self._content_layout.addWidget(section)

    def _build_material_section(self) -> None:
        """Секция «Материал»: параметры :class:`MeshRenderer`."""
        assert self._game_object is not None
        renderer = self._game_object.get_component(MeshRenderer)
        if renderer is None:
            button = QPushButton(self.tr("inspector.add_mesh_renderer"), self._content)
            button.setProperty("accent", True)
            button.clicked.connect(self._add_mesh_renderer)
            self._content_layout.addWidget(button)
            return

        section = SectionBox(self.tr("inspector.section.material"), self._content)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setHorizontalSpacing(12)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        mesh_combo = ReferenceField("asset", section)
        mesh_combo.set_asset(renderer.mesh_ref, self._ref_resolver())
        mesh_combo.dropped.connect(self._drop_mesh)
        mesh_combo.cleared.connect(lambda: self._apply_mesh("cube"))
        menu = QMenu(mesh_combo)
        for name in BUILTIN_MESHES:
            menu.addAction(name, lambda n=name: self._apply_mesh(n))
        pick = QToolButton(mesh_combo)
        pick.setText("▾")
        pick.setToolTip(self.tr("inspector.mesh"))
        pick.setMenu(menu)
        pick.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        mesh_combo.layout().addWidget(pick)
        form.addRow(self.tr("inspector.mesh"), mesh_combo)

        color_button = ColorButton(renderer.color, section)
        color_button.set_label(self.tr("inspector.color"))
        color_button.color_changed.connect(self._apply_color)
        form.addRow(self.tr("inspector.color"), color_button)

        roughness = FocusDoubleSpinBox(section)
        roughness.setRange(0.0, 1.0)
        roughness.setSingleStep(0.05)
        roughness.setDecimals(2)
        roughness.setValue(renderer.roughness)
        roughness.valueChanged.connect(self._apply_roughness)
        form.addRow(self.tr("inspector.roughness"), roughness)

        metallic = FocusDoubleSpinBox(section)
        metallic.setRange(0.0, 1.0)
        metallic.setSingleStep(0.05)
        metallic.setDecimals(2)
        metallic.setValue(renderer.metallic)
        metallic.valueChanged.connect(self._apply_metallic)
        form.addRow(self.tr("inspector.metallic"), metallic)

        visible = QCheckBox(self.tr("inspector.visible"), section)
        visible.setChecked(renderer.visible)
        visible.toggled.connect(self._apply_visible)
        form.addRow("", visible)

        cast_shadow = QCheckBox(self.tr("inspector.cast_shadow"), section)
        cast_shadow.setChecked(renderer.cast_shadow)
        cast_shadow.toggled.connect(self._apply_cast_shadow)
        form.addRow("", cast_shadow)

        receive_shadow = QCheckBox(self.tr("inspector.receive_shadow"), section)
        receive_shadow.setChecked(renderer.receive_shadow)
        receive_shadow.toggled.connect(self._apply_receive_shadow)
        form.addRow("", receive_shadow)

        sorting_layer = FocusSpinBox(section)
        sorting_layer.setRange(-100, 100)
        sorting_layer.setValue(renderer.sorting_layer)
        sorting_layer.setToolTip(self.tr("inspector.sorting_layer.tip"))
        sorting_layer.valueChanged.connect(self._apply_sorting_layer)
        form.addRow(self.tr("inspector.sorting_layer"), sorting_layer)
        for name, widget in (
            ("mesh", mesh_combo),
            ("color", color_button),
            ("roughness", roughness),
            ("metallic", metallic),
            ("sorting_layer", sorting_layer),
        ):
            self._marks[("MeshRenderer", name)] = form.labelForField(widget)
        for name, check in (
            ("visible", visible),
            ("cast_shadow", cast_shadow),
            ("receive_shadow", receive_shadow),
        ):
            self._marks[("MeshRenderer", name)] = check
        self._section_marks["MeshRenderer"] = section

        section.add_layout(form)
        self._content_layout.addWidget(section)
        self._color_button = color_button

    # ------------------------------------------------------------------ #
    # Камера и свет
    # ------------------------------------------------------------------ #
    def _build_camera_section(self) -> None:
        """Секции камеры, если у объекта есть :class:`CameraComponent`."""
        assert self._game_object is not None
        component = self._game_object.get_component(CameraComponent)
        if component is None:
            return
        section = CameraSection(
            self.context,
            component.camera,
            self._game_object,
            self._viewport,
            self._content,
        )
        section.changed.connect(self._on_component_changed)
        self._content_layout.addWidget(section)
        self.camera_section = section

    def _build_light_section(self) -> None:
        """Секции источника света, если у объекта есть :class:`LightComponent`."""
        assert self._game_object is not None
        component = self._game_object.get_component(LightComponent)
        if component is None:
            return
        section = LightSection(
            self.context,
            component.light,
            self._game_object,
            self._viewport,
            self._content,
        )
        section.changed.connect(self._on_component_changed)
        self._content_layout.addWidget(section)
        self.light_section = section

    def _build_text_section(self) -> None:
        """Секция «Текст», если у объекта есть :class:`TextRenderer`."""
        assert self._game_object is not None
        component = self._game_object.get_component(TextRenderer)
        if component is None:
            return
        section = TextSection(self.context, component, self._content)
        section.changed.connect(self._on_component_changed)
        self._content_layout.addWidget(section)
        self.text_section = section

    def _build_ui_section(self) -> None:
        """Секция UI-виджета (и текста у ``UIText``), если он есть у объекта."""
        assert self._game_object is not None
        component = next(
            (c for c in self._game_object.components if isinstance(c, UIElement)),
            None,
        )
        if component is None:
            return
        section = UISection(self.context, component, self._content)
        section.changed.connect(self._on_component_changed)
        self._content_layout.addWidget(section)
        self.ui_section = section
        if isinstance(component, UIText):
            text = TextSection(self.context, component.text, self._content)
            text.changed.connect(self._on_component_changed)
            self._content_layout.addWidget(text)
            self.ui_text_section = text

    def _build_physics_sections(self) -> None:
        """Секции физики: тело, коллайдер, контроллер персонажа."""
        assert self._game_object is not None
        sections = (
            (RigidBodyComponent, RigidBodySection, "rigidbody_section"),
            (ColliderComponent, ColliderSection, "collider_section"),
            (
                CharacterControllerComponent,
                CharacterControllerSection,
                "character_section",
            ),
        )
        for _, _, attribute in sections:
            setattr(self, attribute, None)  # прежние секции уже удалены
        for component_type, section_type, attribute in sections:
            component = self._game_object.get_component(component_type)
            if component is None:
                continue
            section = section_type(self.context, component, self._content)
            section.changed.connect(self._on_component_changed)
            section.changed.connect(self._refresh_physics_conflicts)
            section.remove_requested.connect(
                lambda component=component: self._remove_physics_component(component)
            )
            self._content_layout.addWidget(section)
            setattr(self, attribute, section)

    def _build_auto_sections(self) -> None:
        """Секции по дескрипторам полей для компонентов без ручной секции."""
        assert self._game_object is not None
        self.auto_sections = []
        for component in self._game_object.components:
            component_type = type(component)
            if component_type in _MANUAL_SECTION_TYPES:
                continue
            if not (
                has_auto_section(component_type)
                or component_type is AudioListenerComponent
            ):
                continue
            section = AutoSection(self.context, component, self._content)
            section.changed.connect(self._on_component_changed)
            section.remove_requested.connect(
                lambda component=component: self._remove_physics_component(component)
            )
            self._content_layout.addWidget(section)
            self.auto_sections.append(section)

    def _resolve_instance(self) -> _InstanceInfo | None:
        """Экземпляр префаба, которому принадлежит выбранный объект."""
        game_object = self._game_object
        if game_object is None:
            return None
        assets = self.context.asset_db
        found = find_instance(game_object, assets=assets)
        if found is None:
            return None
        root, path = found
        try:
            everything = get_overrides(root, assets=assets)
        except PrefabError:
            everything = []
        own = [r for r in everything if r.get("target") == path]
        return _InstanceInfo(root, path, everything, own)

    def _build_prefab_section(self) -> None:
        """Секция «Префаб»: объект - корень экземпляра или его член."""
        assert self._game_object is not None
        self.prefab_section = None
        info = self._instance
        if info is None:
            return
        marker = info.root.get_component(PrefabInstance)
        if marker is None:
            return
        shown = info.all if info.root is self._game_object else info.own
        self.prefab_section = PrefabInstanceSection(
            self.context,
            marker,
            self._content,
            overrides=shown,
            on_revert=self._revert_override,
            on_apply=self._apply_override_to_prefab,
        )
        self._content_layout.addWidget(self.prefab_section)

    def _apply_override_marks(self) -> None:
        """Выделить жирным поля и секции, значения которых заданы override."""
        info = self._instance
        if info is None:
            return
        tip = self.tr("inspector.prefab.overridden")
        for rec in info.own:
            kind = str(rec.get("component"))
            field = str(rec.get("field") or "").split(".")[0]
            widget = self._marks.get((kind, field))
            if widget is not None:
                _mark_bold(widget, tip)
        by_type: dict[str, set[str]] = {}
        for rec in info.own:
            by_type.setdefault(str(rec.get("component")), set()).add(
                str(rec.get("field") or "").split(".")[0]
            )
        for kind, section in self._section_marks.items():
            if kind in by_type and isinstance(section, SectionBox):
                _mark_bold(section.title, tip)
        for section in self._content.findChildren(ComponentSection):
            names = by_type.get(type(section.component).__name__)
            if not names:
                continue
            _mark_bold(section.title, tip)
            for field, label in getattr(section, "labels", {}).items():
                if field in names:
                    _mark_bold(label, tip)

    def _revert_override(self, record: dict) -> None:
        """«Вернуть»: значение из префаба."""
        info = self._instance
        if info is None:
            return
        try:
            revert_instance_override(self.context, info.root, record)
        except PrefabError as exc:
            self.status(str(exc))
            return
        self.refresh()

    def _apply_override_to_prefab(self, record: dict) -> None:
        """«Применить к префабу»: записать файл и пересобрать экземпляры."""
        info = self._instance
        if info is None:
            return
        try:
            count = apply_override_to_prefab(
                self.context,
                info.root,
                record,
                progress=lambda done, total: self.status(
                    self.tr("status.prefab_rebuild", done=done, total=total)
                ),
                warn=self.status,
            )
        except (PrefabError, OSError) as exc:
            self.status(str(exc))
            return
        self.status(self.tr("status.prefab_applied", count=count))
        self.refresh()

    def _remove_physics_component(self, component: object) -> None:
        """Удалить физический компонент после подтверждения.

        Удаление откатывается через undo, но без явного «Да» ничего не удаляется
        (ответ по умолчанию - «Нет»).
        """
        if self._game_object is None:
            return
        answer = QMessageBox.question(
            self,
            self.tr("inspector.physics.remove_title"),
            self.tr("inspector.physics.remove_confirm"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        name = type(component).__name__
        self.context.remove_component(self._game_object, component)  # type: ignore[arg-type]
        self.refresh()
        self.material_changed.emit()
        self.status(
            self.tr(
                "inspector.physics.removed", name=name, target=self._game_object.name
            )
        )

    def _refresh_physics_conflicts(self) -> None:
        """Пересчитать подписи конфликта CC + RB у всех физических секций.

        Правка одной секции (например, включение компонента) меняет состояние
        соседней, а пересобирать инспектор из-за этого не нужно.
        """
        for attribute in ("rigidbody_section", "character_section"):
            section = getattr(self, attribute, None)
            if section is not None:
                section.refresh_conflict()

    def _build_sky_section(self) -> None:
        """Секция «Небо» у объекта с направленным светом (солнца сцены)."""
        assert self._game_object is not None
        component = self._game_object.get_component(LightComponent)
        if component is None or component.light.type_name != "directional":
            return
        section = SkySection(self.context, self._content)
        section.changed.connect(self._on_component_changed)
        self._content_layout.addWidget(section)
        self.sky_section = section

    def _build_post_section(self) -> None:
        """Секция «Пост-процесс» у объекта с камерой (настройки всей сцены)."""
        assert self._game_object is not None
        if self._game_object.get_component(CameraComponent) is None:
            return
        section = PostSection(self.context, self._content)
        section.changed.connect(self._on_component_changed)
        self._content_layout.addWidget(section)
        self.post_section = section

    def _on_component_changed(self) -> None:
        """Свойство камеры или света изменено: перерисовать вьюпорт."""
        self.material_changed.emit()

    def _build_add_component_button(self) -> None:
        """Кнопка «Добавить компонент»: меню с группами-подменю.

        Уже имеющиеся на объекте компоненты не прячутся, а показываются
        серыми с подсказкой, почему добавить их нельзя.
        """
        assert self._game_object is not None
        button = QPushButton(self.tr("inspector.add_component"), self._content)
        button.setIcon(icon("plus"))
        menu = QMenu(button)
        menu.setToolTipsVisible(True)
        for group in COMPONENT_GROUPS:
            submenu = menu.addMenu(
                icon(group.icon), self.tr(f"inspector.group.{group.key}")
            )
            submenu.setToolTipsVisible(True)
            for entry in group.entries:
                action = submenu.addAction(
                    icon(entry.icon), self.tr(entry.label_key)
                )
                reason = self._component_block_reason(entry)
                if reason is None:
                    action.triggered.connect(
                        lambda _checked=False, value=entry: self._add_component(value)
                    )
                else:
                    action.setEnabled(False)
                    action.setToolTip(self.tr(reason))
        button.setMenu(menu)
        self.add_component_menu = menu
        self._content_layout.addWidget(button)

    def _component_block_reason(self, entry: ComponentEntry) -> str | None:
        """Почему компонент нельзя добавить объекту.

        Args:
            entry: Пункт меню.

        Returns:
            Ключ перевода причины или ``None``, если добавить можно.
        """
        obj = self._game_object
        if obj is None:
            return "inspector.component.already_added"
        if entry.kind == "camera":
            present = obj.get_component(CameraComponent) is not None
        elif entry.kind == "text":
            present = obj.get_component(TextRenderer) is not None
        elif entry.kind == "ui":
            return (
                "inspector.component.ui_present"
                if any(isinstance(c, UIElement) for c in obj.components)
                else None
            )
        elif entry.kind == "light":
            return (
                "inspector.component.light_present"
                if obj.get_component(LightComponent) is not None
                else None
            )
        else:
            present = obj.get_component(entry.arg) is not None  # type: ignore[arg-type]
            opposite = _PHYSICS_CONFLICTS.get(entry.arg)
            if (
                not present
                and opposite is not None
                and obj.get_component(opposite) is not None
            ):
                return "inspector.component.physics_conflict"
        return "inspector.component.already_added" if present else None

    def _add_component(self, entry: ComponentEntry) -> None:
        """Добавить компонент из пункта меню."""
        if entry.kind == "camera":
            self._add_camera_component()
        elif entry.kind == "text":
            self._add_text_component()
        elif entry.kind == "ui":
            self._add_ui_component(entry.arg)  # type: ignore[arg-type]
        elif entry.kind == "light":
            self._add_light_component(entry.arg)  # type: ignore[arg-type]
        else:
            self._add_physics_component(entry.arg)  # type: ignore[arg-type]

    def _add_camera_component(self) -> None:
        """Добавить объекту камеру."""
        if self._game_object is None:
            return
        self.context.add_component(self._game_object, CameraComponent())
        self.refresh()

    def _add_text_component(self) -> None:
        """Добавить объекту текст."""
        if self._game_object is None:
            return
        self.context.add_component(self._game_object, TextRenderer())
        self.refresh()

    def _add_ui_component(self, kind: type[UIElement]) -> None:
        """Добавить объекту UI-виджет."""
        if self._game_object is None:
            return
        self.context.add_component(self._game_object, kind)
        self.refresh()

    def _add_physics_component(self, kind: type) -> None:
        """Добавить объекту физический компонент."""
        if self._game_object is None:
            return
        self.context.add_component(self._game_object, kind)
        self.refresh()

    def _add_light_component(self, kind: str) -> None:
        """Добавить объекту источник света заданного типа."""
        if self._game_object is None:
            return
        self.context.add_component(
            self._game_object,
            LightComponent,
            setup=lambda component: component.sync_to_light(),  # type: ignore[attr-defined]
            light=create_light(kind),
        )
        self.refresh()

    # ------------------------------------------------------------------ #
    # Скрипты объекта
    # ------------------------------------------------------------------ #
    def _build_scripts_section(self) -> None:
        """Секция «Скрипты»: прикреплённые скрипты и запуск на старте."""
        assert self._game_object is not None
        section = SectionBox(self.tr("inspector.section.scripts"), self._content)
        components = self._scripts()
        if not components:
            hint = QLabel(self.tr("inspector.scripts.empty"), section)
            hint.setProperty("role", "hint")
            hint.setWordWrap(True)
            section.add_widget(hint)
        for component in components:
            section.add_widget(self._script_row(component, section))
            missing = component.missing_requirements()
            if missing:
                warning = QLabel(
                    self.tr(
                        "inspector.scripts.requires",
                        components=", ".join(t.__name__ for t in missing),
                    ),
                    section,
                )
                warning.setObjectName("script_requires_label")
                warning.setProperty("role", "error")
                warning.setWordWrap(True)
                section.add_widget(warning)
            fields_form = ScriptFieldsForm(
                component,
                self.tr("inspector.scripts.fields", name=component.name),
                self.tr,
                self.context.notify_modified,
                section,
                asset_db=self.context.asset_db,
                on_changed=lambda name, before, after, c=component: (
                    self._record_script_field(c, name, before, after)
                ),
            )
            if fields_form.field_names:
                fields_form.reset_requested.connect(
                    lambda: QTimer.singleShot(0, self.refresh)
                )
                section.add_widget(fields_form)
            else:
                fields_form.deleteLater()

        add_button = QPushButton(self.tr("inspector.scripts.add"), section)
        add_button.setProperty("accent", True)
        add_button.clicked.connect(self._add_script)
        section.add_widget(add_button)

        self._content_layout.addWidget(section)

    def _record_script_field(
        self,
        component: ScriptComponent,
        name: str,
        before: object,
        after: object,
    ) -> None:
        """Записать уже применённую правку поля скрипта в историю."""
        if self._game_object is None:
            return
        missing = SetScriptFieldCommand.MISSING
        self.context.push(
            SetScriptFieldCommand(
                self._game_object,
                component,
                name,
                missing if before is None else before,
                missing if after is None else after,
                self.tr("undo.edit", name=self._game_object.name),
            )
        )

    def _script_row(self, component: ScriptComponent, parent: QWidget) -> QWidget:
        """Собрать строку скрипта: имя, галочка автостарта и кнопка «убрать».

        Args:
            component: Компонент-скрипт объекта.
            parent: Родительский виджет секции.

        Returns:
            Готовая строка.
        """
        row = QWidget(parent)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title = QLabel(component.name, row)
        # Длинное имя не должно распирать панель: подпись сжимается.
        title.setMinimumWidth(0)
        title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        title.setToolTip(self._script_tooltip(component))
        if not component.is_loaded:
            # Ошибку загрузки видно сразу: имя подсвечено цветом ошибки.
            title.setProperty("role", "error")

        auto_start = QCheckBox(self.tr("inspector.scripts.auto_start"), row)
        auto_start.setChecked(component.auto_start)
        auto_start.setToolTip(self.tr("inspector.scripts.auto_start_hint"))
        auto_start.toggled.connect(
            lambda value, component=component: self._apply_auto_start(component, value)
        )

        remove = QPushButton(self.tr("inspector.scripts.remove"), row)
        remove.clicked.connect(
            lambda _checked=False, component=component: self._remove_script(component)
        )

        layout.addWidget(title, 1)
        classes = component.behaviour_class_names()
        if len(classes) > 1:
            picker = FocusComboBox(row)
            picker.setToolTip(self.tr("inspector.scripts.class_hint"))
            for name in classes:
                picker.addItem(name, name)
            current = type(component.behaviour).__name__
            picker.setCurrentIndex(max(0, picker.findData(current)))
            picker.currentIndexChanged.connect(
                lambda _i, c=component, p=picker: self._apply_script_class(
                    c, str(p.currentData())
                )
            )
            picker.setObjectName("script_class_picker")
            layout.addWidget(picker)
        layout.addWidget(auto_start)
        layout.addWidget(remove)
        return row

    def _apply_script_class(self, component: ScriptComponent, name: str) -> None:
        """Выбрать класс ``Behaviour``: записать ``script_class`` и перечитать."""
        if component.script_class == name:
            return
        if self._game_object is not None:
            self.context.set_field(
                self._game_object, component, "script_class", name
            )
        component.reload()
        QTimer.singleShot(0, self.refresh)

    def _script_tooltip(self, component: ScriptComponent) -> str:
        """Подсказка строки скрипта: путь, состояние и ошибка загрузки."""
        path = str(component.path) if component.path is not None else "<inline>"
        if component.is_loaded:
            state = self.tr("inspector.scripts.loaded")
        else:
            state = self.tr(
                "inspector.scripts.failed", message=component.last_error or "?"
            )
        return f"{path}\n{state}"

    def _scripts(self) -> list[ScriptComponent]:
        """Скрипты выделенного объекта."""
        if self._game_object is None:
            return []
        return [
            component
            for component in self._game_object.get_components(ScriptComponent)
            if isinstance(component, ScriptComponent)
        ]

    def _pick_script_path(self) -> Path | None:
        """Показать встроенный выбор скрипта (только папка ``scripts``)."""
        dialog = ScriptPicker(
            scripts_root_for(self.context.project_root),
            self.tr("inspector.scripts.add"),
            self,
        )
        if dialog.exec() != ScriptPicker.DialogCode.Accepted:
            return None
        return dialog.selected_path()

    def _add_script(self) -> None:
        """Прикрепить к объекту выбранный на диске скрипт."""
        if self._game_object is None:
            return
        path = self._pick_script_path()
        if path is None:
            return
        component = self.context.add_component(
            self._game_object,
            ScriptComponent(path, runtime=self.context.script_runtime),
        )
        self.refresh()
        self.material_changed.emit()
        self.status(
            self.tr(
                "script.status.attached",
                name=component.name,
                target=self._game_object.name,
            )
        )

    def _apply_auto_start(self, component: ScriptComponent, value: bool) -> None:
        """Включить или выключить запуск скрипта на старте."""
        if self._game_object is not None:
            self.context.set_field(
                self._game_object, component, "auto_start", bool(value)
            )
        self.status(
            self.tr(
                "inspector.scripts.auto_start_changed",
                name=component.name,
                state=self.tr("common.yes") if value else self.tr("common.no"),
            )
        )

    def _remove_script(self, component: ScriptComponent) -> None:
        """Открепить скрипт от объекта и выгрузить его."""
        if self._game_object is None:
            return
        name = component.name
        self.context.remove_component(self._game_object, component)
        self.refresh()
        self.material_changed.emit()
        self.status(self.tr("script.status.detached", name=name))

    # ------------------------------------------------------------------ #
    # Применение изменений
    # ------------------------------------------------------------------ #
    def _apply_name(self, value: str) -> None:
        """Применить новое имя объекта."""
        if self._game_object is None or not value.strip():
            return
        self.context.rename_object(self._game_object, value.strip())

    def _apply_tag(self, value: str) -> None:
        """Применить новый тег."""
        if self._game_object is None:
            return
        self.context.set_object_tag(self._game_object, value.strip() or "Untagged")

    def _apply_active(self, value: bool) -> None:
        """Включить или выключить объект."""
        if self._game_object is None:
            return
        self.context.set_object_active(self._game_object, value)
        self.transform_changed.emit()

    def _apply_position(self, value: Vector3) -> None:
        """Применить позицию."""
        if self._game_object is None:
            return
        self.context.set_field(
            self._game_object, self._game_object.transform, "position", value
        )
        self.transform_changed.emit()

    def _apply_rotation(self, value: Vector3) -> None:
        """Применить поворот (углы Эйлера переводятся в кватернион)."""
        if self._game_object is None:
            return
        self.context.set_field(
            self._game_object,
            self._game_object.transform,
            "rotation",
            Quaternion.from_euler(value.x, value.y, value.z),
        )
        self.transform_changed.emit()

    def _apply_scale(self, value: Vector3) -> None:
        """Применить масштаб."""
        if self._game_object is None:
            return
        self.context.set_field(
            self._game_object, self._game_object.transform, "scale", value
        )
        self.transform_changed.emit()

    def _ref_resolver(self):
        db = self.context.asset_db
        return (lambda ref: ref.resolve(db)) if db is not None else None

    def _drop_mesh(self, path: str) -> None:
        """Файл модели брошен в поле меша."""
        if Path(path).suffix.lower() not in _MODEL_SUFFIXES:
            return
        self._apply_mesh(make_asset_ref(self.context.asset_db, path))

    def _apply_mesh(self, mesh: str | AssetRef) -> None:
        """Сменить меш объекта (имя примитива или ссылка на модель)."""
        renderer = self._renderer()
        if renderer is None:
            return
        # GUID пересчитается при сохранении сцены.
        new = mesh if isinstance(mesh, AssetRef) else AssetRef(path=mesh)
        self._set_renderer_field(renderer, "mesh_ref", new)
        self.material_changed.emit()
        self.refresh()

    def _set_renderer_field(
        self, renderer: MeshRenderer, field: str, value: object
    ) -> None:
        """Записать поле ``MeshRenderer`` командой с undo."""
        if self._game_object is not None:
            self.context.set_field(self._game_object, renderer, field, value)

    def _apply_color(self, color: Vector3) -> None:
        """Сменить цвет материала."""
        renderer = self._renderer()
        if renderer is None:
            return
        self._set_renderer_field(renderer, "color", color)
        self.material_changed.emit()

    def _apply_roughness(self, value: float) -> None:
        """Сменить шероховатость."""
        renderer = self._renderer()
        if renderer is None:
            return
        self._set_renderer_field(renderer, "roughness", float(value))
        self.material_changed.emit()

    def _apply_metallic(self, value: float) -> None:
        """Сменить металличность."""
        renderer = self._renderer()
        if renderer is None:
            return
        self._set_renderer_field(renderer, "metallic", float(value))
        self.material_changed.emit()

    def _apply_visible(self, value: bool) -> None:
        """Показать или скрыть объект."""
        renderer = self._renderer()
        if renderer is None:
            return
        self._set_renderer_field(renderer, "visible", bool(value))
        self.material_changed.emit()

    def _apply_cast_shadow(self, value: bool) -> None:
        """Включить или выключить отбрасывание тени объектом."""
        renderer = self._renderer()
        if renderer is None:
            return
        self._set_renderer_field(renderer, "cast_shadow", bool(value))
        self.material_changed.emit()

    def _apply_receive_shadow(self, value: bool) -> None:
        """Включить или выключить приём чужих теней объектом."""
        renderer = self._renderer()
        if renderer is None:
            return
        self._set_renderer_field(renderer, "receive_shadow", bool(value))
        self.material_changed.emit()

    def _apply_sorting_layer(self, value: int) -> None:
        """Задать слой сортировки прозрачных (действует в плеере)."""
        renderer = self._renderer()
        if renderer is None:
            return
        self._set_renderer_field(renderer, "sorting_layer", int(value))
        self.material_changed.emit()

    def _add_mesh_renderer(self) -> None:
        """Добавить объекту компонент :class:`MeshRenderer`."""
        if self._game_object is None:
            return
        self.context.add_component(self._game_object, MeshRenderer)
        self.refresh()

    def _renderer(self) -> MeshRenderer | None:
        """Компонент отрисовки выделенного объекта."""
        if self._game_object is None:
            return None
        component = self._game_object.get_component(MeshRenderer)
        return component if isinstance(component, MeshRenderer) else None

    def _make_placeholder(self) -> QLabel:
        """Создать подсказку «ничего не выбрано».

        Виджет создаётся без родителя: он станет частью раскладки только при
        добавлении, иначе Qt показал бы его поверх остальных полей.
        """
        label = QLabel(self.tr("inspector.no_selection"))
        label.setWordWrap(True)
        label.setProperty("role", "hint")
        label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        return label

    def _clear(self) -> None:
        """Очистить содержимое инспектора."""
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # Скрываем и удаляем отложенно: setParent(None) сделал бы виджет
                # отдельным окном, которое мелькает поверх редактора.
                widget.hide()
                widget.deleteLater()
        self._placeholder = self._make_placeholder()
