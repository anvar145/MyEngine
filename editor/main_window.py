"""Главное окно редактора: меню, панели инструментов, доки и вьюпорт.

Окно связывает всё воедино: центральный 3D-вьюпорт, док-панели (иерархия,
инспектор, ассеты, консоль, статистика), меню с командами и переключением
языка, а также сохранение раскладки между запусками через ``QSettings``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from PySide6.QtCore import QByteArray, QSettings, Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QCloseEvent,
    QGuiApplication,
    QKeySequence,
    QShowEvent,
)
from PySide6.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from editor import scene_io
from editor.context import EditorContext
from editor.gizmos.transform_gizmo import GizmoMode
from editor.panels import (
    AssetsPanel,
    ConsolePanel,
    HierarchyPanel,
    ImportPanel,
    InputSettingsPanel,
    InspectorPanel,
    ScriptEditorPanel,
    StatsPanel,
)
from editor.prefab_actions import instantiate_prefab_file
from editor.project import (
    assets_root_for,
    create_project,
    ensure_projects_dir,
    projects_root,
    scenes_root_for,
)
from editor.services.play_mode_service import PlayModeService
from editor.theme import icon
from editor.viewport import EditorViewport
from editor.viewport.editor_viewport import COLLIDER_MODES
from editor.viewport.scene_tabs import SceneTabs
from editor.widgets import SideRail
from engine.core.components import TextRenderer
from engine.core.game_object import GameObject
from engine.core.math3d import Vector3
from engine.core.ui_components import UIButton, UIImage, UIPanel, UIText
from engine.prefab import PrefabError
from engine.rendering.camera import EditorCameraSession, quaternion_from_forward
from engine.rendering.lighting import (
    LIGHT_TYPES,
    DirectionalLight,
    LightComponent,
    SpotLight,
    create_light,
)

__all__ = ["MainWindow"]

#: Иконки режимов гизмо (``assets/icons``).
_GIZMO_ICONS = {
    GizmoMode.MOVE: "move",
    GizmoMode.ROTATE: "rotate",
    GizmoMode.SCALE: "scale",
}

_log = logging.getLogger("editor.window")

#: Интервал обновления панели статистики (мс).
STATS_INTERVAL_MS = 400

#: Ключ, под которым сохраняется вид камеры редактора.
EDITOR_CAMERA_KEY = EditorCameraSession.DEFAULT_KEY

#: Позиция нового источника света относительно точки интереса редактора.
NEW_LIGHT_OFFSET = Vector3(0.0, 3.0, 0.0)

#: Имена создаваемых источников света: всегда английские, не зависят от языка UI.
_LIGHT_OBJECT_NAMES = {
    "directional": "Directional Light",
    "point": "Point Light",
    "spot": "Spot Light",
    "ambient": "Ambient Light",
}

#: Размер окна по умолчанию, пока нет сохранённой геометрии.
DEFAULT_WINDOW_SIZE = (1480, 900)

#: Организация и приложение для QSettings.
SETTINGS_ORG = "MyEngine"
SETTINGS_APP = "Editor"


class MainWindow(QMainWindow):
    """Главное окно редактора.

    Args:
        context: Контекст редактора со сценой и сервисами.
        parent: Родительский виджет.

    Signals:
        (нет: окно управляет панелями напрямую, а панели — контекстом)
    """

    def __init__(self, context: EditorContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.context = context
        self.setObjectName("myengine_main_window")
        self.setDockNestingEnabled(True)
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)

        self.viewport = EditorViewport(context, self)
        self.scene_tabs = SceneTabs(context, self)
        central = QWidget(self)
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.scene_tabs)
        central_layout.addWidget(self.viewport, 1)
        self.setCentralWidget(central)

        self.hierarchy = HierarchyPanel(context, self)
        self.inspector = InspectorPanel(context, self)
        self.assets = AssetsPanel(context, self)
        self.console = ConsolePanel(context, self)
        self.stats_panel = StatsPanel(context, self)
        self.import_panel = ImportPanel(context, self)
        self.input_panel = InputSettingsPanel(context, self)
        self.script_editor = ScriptEditorPanel(context, self)
        self.inspector.connect_viewport(self.viewport)
        # Вид камеры редактора переживает перезапуск: состояние лежит в файле
        # проекта и привязывается к имени камеры.
        self.camera_session = EditorCameraSession(
            EditorCameraSession.default_path(self.context.project_root)
        )
        self.camera_session.load()
        self._restore_editor_camera()

        self._last_fps: float = 0.0
        self._last_render_stats: object | None = None
        self._player_stats: dict | None = None
        self._player_loading = False

        self.play_mode = PlayModeService(context, self)
        context.play_mode = self.play_mode  # живая правка неба из инспектора
        self.play_mode.before_play = self._save_dirty_documents

        self._create_actions()
        self._create_menus()
        self._create_toolbars()
        self._create_docks()
        self._create_statusbar()
        self._connect_signals()

        self._fitted_on_show = False
        self.settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        self._stats_timer = QTimer(self)
        self._stats_timer.timeout.connect(self._refresh_stats)
        self._stats_timer.start(STATS_INTERVAL_MS)

        self.new_scene()
        self.resize(*DEFAULT_WINDOW_SIZE)
        self._restore_layout()
        self.retranslate()
        self.fit_to_available_screen()

    # ------------------------------------------------------------------ #
    # Действия
    # ------------------------------------------------------------------ #
    def _create_actions(self) -> None:
        """Создать все действия (команды) редактора."""
        self.action_new = QAction(self)
        self.action_new.setShortcut(QKeySequence.New)
        self.action_new.triggered.connect(self.new_scene)

        self.action_open = QAction(self)
        self.action_open.setShortcut(QKeySequence.Open)
        self.action_open.triggered.connect(self.open_scene)

        self.action_save = QAction("", self, icon=icon("save"))
        self.action_save.setShortcut(QKeySequence.Save)
        self.action_save.triggered.connect(self.save_scene)

        self.action_save_as = QAction(self)
        self.action_save_as.setShortcut(QKeySequence.SaveAs)
        self.action_save_as.triggered.connect(self.save_scene_as)

        self.action_exit = QAction(self)
        self.action_exit.setShortcut(QKeySequence.Quit)
        self.action_exit.triggered.connect(self.close)

        self.action_undo = QAction(self)
        self.action_undo.setShortcut(QKeySequence("Ctrl+Z"))
        self.action_undo.triggered.connect(self.undo)

        self.action_redo = QAction(self)
        self.action_redo.setShortcuts(
            [QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")]
        )
        self.action_redo.triggered.connect(self.redo)

        self.action_delete = QAction("", self, icon=icon("delete"))
        self.action_delete.setShortcut(QKeySequence.Delete)
        self.action_delete.triggered.connect(self.delete_selected)

        self.action_duplicate = QAction("", self, icon=icon("duplicate"))
        self.action_duplicate.setShortcut(QKeySequence("Ctrl+D"))
        self.action_duplicate.triggered.connect(self.duplicate_selected)

        self.action_deselect = QAction(self)
        self.action_deselect.setShortcut(QKeySequence("Ctrl+Shift+A"))
        self.action_deselect.triggered.connect(self.context.selection.clear)

        self.action_create_cube = QAction("", self, icon=icon("cube"))
        self.action_create_cube.triggered.connect(lambda: self.create_primitive("cube"))
        self.action_create_sphere = QAction("", self, icon=icon("sphere"))
        self.action_create_sphere.triggered.connect(
            lambda: self.create_primitive("sphere")
        )
        self.action_create_plane = QAction("", self, icon=icon("plane"))
        self.action_create_plane.triggered.connect(
            lambda: self.create_primitive("plane")
        )
        self.action_create_empty = QAction("", self, icon=icon("empty"))
        self.action_create_empty.triggered.connect(self.create_empty)

        self.action_create_text = QAction("", self, icon=icon("text"))
        self.action_create_text.triggered.connect(self.create_text)
        self.action_create_prefab = QAction("", self, icon=icon("box"))
        self.action_create_prefab.triggered.connect(lambda: self.create_prefab())
        self.ui_actions: dict[str, QAction] = {}
        for key, kind, icon_name in (
            ("ui_panel", UIPanel, "layers"),
            ("ui_button", UIButton, "command"),
            ("ui_text", UIText, "text"),
            ("ui_image", UIImage, "image"),
        ):
            action = QAction("", self, icon=icon(icon_name))
            action.triggered.connect(
                lambda _checked=False, value=kind: self.create_ui(value)
            )
            self.ui_actions[key] = action
        self.action_create_camera = QAction("", self, icon=icon("video"))
        self.action_create_camera.triggered.connect(self.create_camera)

        # Создание источников света: по действию на тип.
        self.light_actions: dict[str, QAction] = {}
        for kind in LIGHT_TYPES:
            action = QAction("", self, icon=icon("light"))
            action.triggered.connect(
                lambda _checked=False, value=kind: self.create_light(value)
            )
            self.light_actions[kind] = action

        # Режимы гизмо трансформа: W — перемещение, E — поворот, R — масштаб.
        self.gizmo_mode_group = QActionGroup(self)
        self.gizmo_mode_actions: dict[GizmoMode, QAction] = {}
        for mode, key in (
            (GizmoMode.MOVE, "W"),
            (GizmoMode.ROTATE, "E"),
            (GizmoMode.SCALE, "R"),
        ):
            action = QAction(icon(_GIZMO_ICONS[mode]), "", self)
            action.setCheckable(True)
            action.setShortcut(QKeySequence(key))
            action.setChecked(mode is GizmoMode.MOVE)
            action.triggered.connect(
                lambda _checked=False, value=mode: self.viewport.set_gizmo_mode(value)
            )
            self.gizmo_mode_group.addAction(action)
            self.gizmo_mode_actions[mode] = action
        self.viewport.gizmo_mode_changed.connect(self._on_gizmo_mode_changed)

        self.action_reset_camera = QAction("", self, icon=icon("reset_camera"))
        self.action_reset_camera.setShortcut(QKeySequence("Home"))
        self.action_reset_camera.triggered.connect(self.viewport.reset_camera)

        self.action_focus = QAction("", self, icon=icon("focus"))
        self.action_focus.setShortcut(QKeySequence("F"))
        self.action_focus.triggered.connect(self.focus_selected)

        self.action_camera_gizmos = QAction(self)
        self.action_camera_gizmos.setCheckable(True)
        self.action_camera_gizmos.setChecked(self.viewport.show_camera_gizmos)
        self.action_camera_gizmos.toggled.connect(
            self.viewport.set_camera_gizmos_visible
        )

        self.action_camera_frustum = QAction(self)
        self.action_camera_frustum.setCheckable(True)
        self.action_camera_frustum.setChecked(self.viewport.show_camera_frustum)
        self.action_camera_frustum.toggled.connect(
            self.viewport.set_camera_frustum_visible
        )

        self.action_light_gizmos = QAction(self)
        self.action_light_gizmos.setCheckable(True)
        self.action_light_gizmos.setChecked(self.viewport.show_light_gizmos)
        self.action_light_gizmos.toggled.connect(
            self.viewport.set_light_gizmos_visible
        )

        self.action_light_ranges = QAction(self)
        self.action_light_ranges.setCheckable(True)
        self.action_light_ranges.setChecked(self.viewport.show_light_ranges)
        self.action_light_ranges.toggled.connect(
            self.viewport.set_light_ranges_visible
        )

        self.collider_actions: dict[str, QAction] = {}
        collider_group = QActionGroup(self)
        collider_group.setExclusive(True)
        for mode in COLLIDER_MODES:
            action = QAction(self)
            action.setCheckable(True)
            action.setChecked(mode == self.viewport.collider_mode)
            action.triggered.connect(
                lambda _checked=False, m=mode: self.set_collider_mode(m)
            )
            collider_group.addAction(action)
            self.collider_actions[mode] = action

        self.action_debug_overlays = QAction(self)
        self.action_debug_overlays.setCheckable(True)
        self.action_debug_overlays.setShortcut(QKeySequence("F3"))
        self.action_debug_overlays.setChecked(self.viewport.debug_overlays.visible)
        self.action_debug_overlays.toggled.connect(
            self.viewport.set_debug_overlays_visible
        )
        #: Пункты оверлеев engine.debug (коллайдеры — в своём подменю).
        self.overlay_actions: dict[str, QAction] = {}
        for overlay in self.viewport.debug_overlays:
            if overlay.external:
                continue
            action = QAction(self)
            action.setCheckable(True)
            action.setChecked(overlay.enabled)
            action.toggled.connect(
                lambda checked, n=overlay.name: (
                    self.viewport.set_debug_overlay_enabled(n, checked)
                )
            )
            self.overlay_actions[overlay.name] = action

        self.action_reset_layout = QAction(self)
        self.action_reset_layout.triggered.connect(self.reset_layout)

        self.action_new_project = QAction(self)
        self.action_new_project.setShortcut(QKeySequence("Ctrl+Shift+N"))
        self.action_new_project.triggered.connect(self.new_project)

        self.action_open_project = QAction(self)
        self.action_open_project.triggered.connect(self.open_project)

        self.action_play = QAction(self)
        self.action_play.setShortcut(QKeySequence("F5"))
        self.action_play.triggered.connect(self.play_mode.enter_play)
        self.action_play_stop = QAction(self)
        self.action_play_stop.setShortcut(QKeySequence("Shift+F5"))
        self.action_play_stop.triggered.connect(self.play_mode.stop)
        self.action_play_pause = QAction(self)
        self.action_play_pause.setShortcut(QKeySequence("F6"))
        self.action_play_pause.triggered.connect(self.play_mode.toggle_pause)
        self.action_play_step = QAction(self)
        self.action_play_step.setShortcut(QKeySequence("F7"))
        self.action_play_step.triggered.connect(lambda: self.play_mode.step(1))
        self.action_quick_save = QAction(self)
        self.action_quick_save.setShortcut(QKeySequence("Ctrl+F5"))
        self.action_quick_save.triggered.connect(self.play_mode.quick_save)
        self.action_quick_load = QAction(self)
        self.action_quick_load.setShortcut(QKeySequence("Ctrl+F9"))
        self.action_quick_load.triggered.connect(self.play_mode.quick_load)

        self.action_script_panel = QAction(self)
        self.action_script_panel.setCheckable(True)
        self.action_script_panel.setShortcut(QKeySequence("Ctrl+Shift+E"))
        self.action_script_panel.toggled.connect(self.set_script_panel_visible)

        self.action_about = QAction(self)
        self.action_about.triggered.connect(self.show_about)

        self.language_group = QActionGroup(self)
        self.language_group.setExclusive(True)
        self.language_actions: dict[str, QAction] = {}
        for code, label in self.context.translator.available_languages:
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(code == self.context.translator.language)
            action.triggered.connect(
                lambda _checked, code=code: self.set_language(code)
            )
            self.language_group.addAction(action)
            self.language_actions[code] = action

        self.panel_actions: dict[str, QAction] = {}

    def _create_menus(self) -> None:
        """Собрать строку меню."""
        self.menu_file = self.menuBar().addMenu("")
        self.menu_file.addAction(self.action_new)
        self.menu_file.addAction(self.action_open)
        self.menu_file.addAction(self.action_save)
        self.menu_file.addAction(self.action_save_as)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.action_new_project)
        self.menu_file.addAction(self.action_open_project)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.action_exit)

        self.menu_edit = self.menuBar().addMenu("")
        self.menu_edit.addAction(self.action_undo)
        self.menu_edit.addAction(self.action_redo)
        self.menu_edit.addSeparator()
        self.menu_edit.addAction(self.action_duplicate)
        self.menu_edit.addAction(self.action_delete)
        self.menu_edit.addSeparator()
        self.menu_edit.addAction(self.action_deselect)

        self.menu_create = self.menuBar().addMenu("")
        self.menu_create.addAction(self.action_create_cube)
        self.menu_create.addAction(self.action_create_sphere)
        self.menu_create.addAction(self.action_create_plane)
        self.menu_create.addSeparator()
        self.menu_create.addAction(self.action_create_empty)
        self.menu_create.addAction(self.action_create_text)
        self.menu_create.addAction(self.action_create_prefab)
        self.menu_create.addAction(self.action_create_camera)
        self.menu_create_ui = self.menu_create.addMenu("")
        for action in self.ui_actions.values():
            self.menu_create_ui.addAction(action)
        self.menu_create_light = self.menu_create.addMenu("")
        for action in self.light_actions.values():
            self.menu_create_light.addAction(action)

        self.menu_play = self.menuBar().addMenu("")
        self.menu_play.addAction(self.action_play)
        self.menu_play.addAction(self.action_play_stop)
        self.menu_play.addAction(self.action_play_pause)
        self.menu_play.addAction(self.action_play_step)
        self.menu_play.addSeparator()
        self.menu_play.addAction(self.action_quick_save)
        self.menu_play.addAction(self.action_quick_load)

        self.menu_script = self.menuBar().addMenu("")
        self.menu_script.addAction(self.action_script_panel)
        self.menu_script.addSeparator()
        self.menu_script.addAction(self.script_editor.action_new)
        self.menu_script.addAction(self.script_editor.action_open)
        self.menu_script.addAction(self.script_editor.action_save)
        self.menu_script.addSeparator()
        self.menu_script.addAction(self.script_editor.action_run)
        self.menu_script.addAction(self.script_editor.action_stop)
        self.menu_script.addAction(self.script_editor.action_reload)

        self.menu_view = self.menuBar().addMenu("")
        self.menu_view.addAction(self.action_focus)
        self.menu_view.addAction(self.action_reset_camera)
        self.menu_view.addSeparator()
        self.menu_view.addAction(self.action_camera_gizmos)
        self.menu_view.addAction(self.action_camera_frustum)
        self.menu_view.addAction(self.action_light_gizmos)
        self.menu_view.addAction(self.action_light_ranges)
        self.menu_colliders = self.menu_view.addMenu("")
        for action in self.collider_actions.values():
            self.menu_colliders.addAction(action)
        self.menu_overlays = self.menu_view.addMenu("")
        self.menu_overlays.addAction(self.action_debug_overlays)
        self.menu_overlays.addSeparator()
        for action in self.overlay_actions.values():
            self.menu_overlays.addAction(action)
        self.view_menu_separator = self.menu_view.addSeparator()
        self.menu_view.addAction(self.action_reset_layout)

        self.menu_language = self.menuBar().addMenu("")
        for action in self.language_actions.values():
            self.menu_language.addAction(action)

        self.menu_help = self.menuBar().addMenu("")
        self.menu_help.addAction(self.action_about)

    def _create_toolbars(self) -> None:
        """Создать панели инструментов создания и вида."""
        self.toolbar = QToolBar(self)
        self.toolbar.setObjectName("myengine_toolbar_main")
        self.toolbar.setMovable(True)
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)

        self.toolbar.addAction(self.action_create_cube)
        self.toolbar.addAction(self.action_create_sphere)
        self.toolbar.addAction(self.action_create_plane)
        self.toolbar.addAction(self.action_create_empty)
        self.toolbar.addAction(self.action_create_camera)
        # Свет создаётся из той же панели: кнопка с меню типов источников.
        self.light_button = QToolButton(self.toolbar)
        self.light_button.setIcon(icon("light"))
        self.light_button.setPopupMode(QToolButton.InstantPopup)
        self.light_button.setMenu(self.menu_create_light)
        self.toolbar.addWidget(self.light_button)
        self.toolbar.addSeparator()
        for action in self.gizmo_mode_actions.values():
            self.toolbar.addAction(action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.action_focus)
        self.toolbar.addAction(self.action_reset_camera)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.action_duplicate)
        self.toolbar.addAction(self.action_delete)

    def _create_docks(self) -> None:
        """Разместить док-панели в окне."""
        self.addDockWidget(Qt.LeftDockWidgetArea, self.hierarchy)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.assets)
        self.splitDockWidget(self.hierarchy, self.assets, Qt.Vertical)
        self.addDockWidget(Qt.RightDockWidgetArea, self.inspector)
        # Статистика — вкладка рядом с ассетами, чтобы инспектору хватало высоты.
        self.addDockWidget(Qt.LeftDockWidgetArea, self.stats_panel)
        self.tabifyDockWidget(self.assets, self.stats_panel)
        self.assets.raise_()
        self.addDockWidget(Qt.BottomDockWidgetArea, self.console)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.import_panel)
        self.tabifyDockWidget(self.console, self.import_panel)
        self.console.raise_()
        # Настройки ввода: док справа, по умолчанию скрыт (меню «Вид»).
        self.addDockWidget(Qt.RightDockWidgetArea, self.input_panel)
        self.input_panel.hide()

        self.resizeDocks([self.hierarchy, self.inspector], [300, 360], Qt.Horizontal)
        self.resizeDocks([self.console], [200], Qt.Vertical)

        for key, panel in (
            ("panel.hierarchy", self.hierarchy),
            ("panel.inspector", self.inspector),
            ("panel.assets", self.assets),
            ("panel.console", self.console),
            ("panel.stats", self.stats_panel),
            ("panel.imports", self.import_panel),
            ("panel.input", self.input_panel),
        ):
            action = panel.toggleViewAction()
            action.setChecked(panel.isVisible())
            self.panel_actions[key] = action
            self.menu_view.addAction(action)

        self.side_rail = SideRail(self)
        self.addToolBar(Qt.LeftToolBarArea, self.side_rail)
        for key, panel, icon_name in (
            ("panel.hierarchy", self.hierarchy, "list-tree"),
            ("panel.assets", self.assets, "folder"),
            ("panel.inspector", self.inspector, "inspector"),
            ("panel.console", self.console, "terminal"),
            ("panel.imports", self.import_panel, "import"),
            ("panel.stats", self.stats_panel, "stats"),
        ):
            self.side_rail.add_dock(key, panel, icon_name)

    def _create_statusbar(self) -> None:
        """Собрать статусную строку."""
        self.status_label = QLabel("", self)
        self.hint_label = QLabel("", self)
        self.hint_label.setProperty("role", "hint")
        self.language_label = QLabel("", self)
        self.language_label.setProperty("role", "hint")
        self.statusBar().addWidget(self.status_label, 1)
        self.import_progress = QProgressBar(self)
        self.import_progress.setMaximumWidth(160)
        self.import_progress.setMaximumHeight(14)
        self.import_progress.hide()
        self.statusBar().addPermanentWidget(self.import_progress)
        self.play_label = QLabel("", self)
        self.statusBar().addPermanentWidget(self.play_label)
        self.statusBar().addPermanentWidget(self.hint_label)
        self.statusBar().addPermanentWidget(self.language_label)

    def _connect_signals(self) -> None:
        """Подписать окно на сигналы панелей и контекста."""
        self.hierarchy.status_message.connect(self.show_status)
        self.assets.status_message.connect(self.show_status)
        self.assets.model_activated.connect(self.add_model_from_asset)
        self.assets.scene_activated.connect(self.load_scene_file)
        self.assets.script_activated.connect(self.open_script_from_asset)
        self.assets.asset_selected.connect(self._on_asset_selected)
        self.import_panel.status_message.connect(self.show_status)
        self.input_panel.status_message.connect(self.show_status)
        self.context.imports_changed.connect(self._update_import_progress)
        self.hierarchy.focus_requested.connect(lambda _obj: self.focus_selected())
        self.inspector.material_changed.connect(self.viewport.refresh)
        self.inspector.transform_changed.connect(self.viewport.refresh)
        self.viewport.object_transformed.connect(self._on_object_transformed)
        self.viewport.transform_committed.connect(self._on_transform_committed)
        self.viewport.stats_updated.connect(self._on_viewport_stats)
        self.context.selection.changed.connect(self._on_selection_changed)
        self.context.scene_changed.connect(self._on_scene_replaced)
        self.play_mode.state_changed.connect(self._on_play_state)
        self.play_mode.stats_received.connect(self._on_player_stats)
        self.play_mode.load_progress.connect(self._on_player_load_progress)
        self.play_mode.scene_loaded.connect(self._on_player_scene_loaded)
        self.play_mode.log_received.connect(self._on_player_log)
        self.play_mode.crashed.connect(self._on_player_crashed)
        self.context.document_deactivating.connect(self._remember_document_camera)
        self.context.documents_changed.connect(self._update_title)
        self.scene_tabs.close_requested.connect(self.close_scene_tab)
        self.scene_tabs.new_requested.connect(self.new_scene)
        self.context.project_changed.connect(self._on_project_changed)
        # Меню, тулбар и статусную строку переводит само окно: панели делают
        # это самостоятельно, подписавшись на тот же сигнал.
        self.context.translator.language_changed.connect(self.retranslate)
        self.context.history_changed.connect(self._update_history_actions)
        self.context.documents_changed.connect(self._update_history_actions)
        self.play_mode.state_changed.connect(self._update_history_actions)

    def undo(self) -> None:
        """Отменить последнюю правку сцены."""
        if self.context.undo():
            self._after_history_step()

    def redo(self) -> None:
        """Повторить отменённую правку сцены."""
        if self.context.redo():
            self._after_history_step()

    def _after_history_step(self) -> None:
        """Обновить панели после undo/redo (сцена заменена снимком)."""
        self.hierarchy.refresh()
        self.inspector.refresh()
        self.viewport.refresh()
        self._update_title()

    def _update_history_actions(self, *_args: object) -> None:
        """Включить пункты undo/redo и подписать их последним шагом."""
        context = self.context
        tr = context.tr
        can_undo = context.can_undo
        can_redo = context.can_redo
        undo_text = context.undo_text()
        redo_text = context.redo_text()
        self.action_undo.setEnabled(can_undo)
        self.action_redo.setEnabled(can_redo)
        self.action_undo.setText(
            tr("menu.edit.undo_label", label=undo_text)
            if can_undo and undo_text
            else tr("menu.edit.undo")
        )
        self.action_redo.setText(
            tr("menu.edit.redo_label", label=redo_text)
            if can_redo and redo_text
            else tr("menu.edit.redo")
        )

    def _on_object_transformed(self, _game_object: object) -> None:
        """Объект двигают гизмо во вьюпорте: поля трансформа следуют за ним."""
        self.inspector.sync_transform()

    def _on_transform_committed(self, _game_object: object) -> None:
        """Перетаскивание гизмо закончено: сцена изменена."""
        self.inspector.sync_transform()

    def _on_gizmo_mode_changed(self, value: str) -> None:
        """Отметить в тулбаре режим гизмо, выбранный иначе (клавиатура, код)."""
        action = self.gizmo_mode_actions.get(GizmoMode(value))
        if action is not None:
            action.setChecked(True)

    def _on_scene_replaced(self, scene: object) -> None:
        """Переключить панели на новую сцену (вкладку)."""
        # Скрипты прежней сцены продолжать нельзя: их объекты уже не на экране.
        if self.script_editor.runner.running:
            self.script_editor.runner.stop()
        self.play_mode.stop()
        self.hierarchy.set_scene(scene)
        state = self.context.active_document.camera_state
        if state is not None:
            self.viewport.viewport.controller.restore(state, self.viewport.camera)
        self.inspector.refresh()
        self.viewport.refresh()
        self._update_title()

    def _remember_document_camera(self, document: object) -> None:
        """Запомнить вид камеры редактора в уходящей вкладке."""
        controller = self.viewport.viewport.controller
        document.camera_state = controller.capture(self.viewport.camera)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ #
    # Локализация
    # ------------------------------------------------------------------ #
    def retranslate(self, *_args: object) -> None:
        """Перевести все элементы интерфейса."""
        tr = self.context.tr
        self._update_title()

        self.menu_file.setTitle(tr("menu.file"))
        self.menu_edit.setTitle(tr("menu.edit"))
        self.menu_create.setTitle(tr("menu.create"))
        self.menu_script.setTitle(tr("menu.script"))
        self.menu_play.setTitle(tr("menu.play"))
        self.menu_view.setTitle(tr("menu.view"))
        self.menu_colliders.setTitle(tr("menu.view.colliders"))
        self.menu_overlays.setTitle(tr("menu.view.overlays"))
        self.action_debug_overlays.setText(tr("menu.view.overlays_show"))
        for name, action in self.overlay_actions.items():
            action.setText(tr(f"menu.view.overlay.{name}"))
        self._update_play_label()
        self.menu_language.setTitle(tr("menu.language"))
        self.menu_help.setTitle(tr("menu.help"))

        for action, key in (
            (self.action_new, "menu.file.new"),
            (self.action_open, "menu.file.open"),
            (self.action_save, "menu.file.save"),
            (self.action_save_as, "menu.file.save_as"),
            (self.action_exit, "menu.file.exit"),
            (self.action_new_project, "menu.project.new"),
            (self.action_open_project, "menu.project.open"),
            (self.action_duplicate, "menu.edit.duplicate"),
            (self.action_delete, "menu.edit.delete"),
            (self.action_deselect, "menu.edit.deselect"),
            (self.action_create_cube, "menu.create.cube"),
            (self.action_create_sphere, "menu.create.sphere"),
            (self.action_create_plane, "menu.create.plane"),
            (self.action_create_empty, "menu.create.empty"),
            (self.action_create_text, "menu.create.text"),
            (self.action_create_prefab, "menu.create.prefab"),
            (self.action_create_camera, "menu.create.camera"),
            (self.action_focus, "menu.view.frame_selected"),
            (self.action_reset_camera, "menu.view.reset_camera"),
            (self.action_camera_gizmos, "menu.view.camera_gizmos"),
            (self.action_camera_frustum, "menu.view.camera_frustum"),
            (self.action_light_gizmos, "menu.view.light_gizmos"),
            (self.action_light_ranges, "menu.view.light_ranges"),
            (self.collider_actions["off"], "menu.view.colliders_off"),
            (self.collider_actions["selected"], "menu.view.colliders_selected"),
            (self.collider_actions["all"], "menu.view.colliders_all"),
            (self.action_reset_layout, "menu.view.reset_layout"),
            (self.action_script_panel, "menu.script.panel"),
            (self.action_play, "menu.play.play"),
            (self.action_play_stop, "menu.play.stop"),
            (self.action_play_pause, "menu.play.pause"),
            (self.action_play_step, "menu.play.step"),
            (self.action_quick_save, "menu.play.quick_save"),
            (self.action_quick_load, "menu.play.quick_load"),
            (self.action_about, "menu.help.about"),
        ):
            action.setText(tr(key))

        self._update_history_actions()

        for mode, action in self.gizmo_mode_actions.items():
            text = tr(f"toolbar.gizmo.{mode.value}")
            action.setText(text)
            action.setToolTip(text)
            action.setStatusTip(text)
        self.light_button.setToolTip(tr("menu.create.light"))

        # Подменю источников света: заголовок и названия типов.
        self.menu_create_light.setTitle(tr("menu.create.light"))
        for kind, action in self.light_actions.items():
            action.setText(tr(f"light.kind.{kind}"))
        self.menu_create_ui.setTitle(tr("menu.create.ui"))
        for key, action in self.ui_actions.items():
            action.setText(tr(f"menu.create.{key}"))

        for key, action in self.panel_actions.items():
            action.setText(tr(key))
        for key, action in self.side_rail.buttons.items():
            action.setToolTip(tr(key))
            action.setText(tr(key))

        self.toolbar.setWindowTitle(tr("toolbar.create"))
        for action, key in (
            (self.action_create_cube, "toolbar.cube"),
            (self.action_create_sphere, "toolbar.sphere"),
            (self.action_create_plane, "toolbar.plane"),
            (self.action_delete, "toolbar.delete"),
            (self.action_focus, "toolbar.focus"),
            (self.action_reset_camera, "toolbar.reset_camera"),
        ):
            action.setToolTip(tr(key))
            action.setStatusTip(tr(key))

        for code, action in self.language_actions.items():
            action.setText(self.context.tr(f"menu.language.{code}"))
            action.setChecked(code == self.context.translator.language)

        self.hint_label.setText(tr("viewport.hint"))
        self.language_label.setText(self.context.translator.language_name)
        self.show_status(tr("status.ready"))

    def set_language(self, code: str) -> None:
        """Переключить язык интерфейса.

        Args:
            code: Код языка (``"ru"`` или ``"en"``).
        """
        if not self.context.set_language(code):
            return
        self.settings.setValue("language", code)
        self.show_status(
            self.context.tr(
                "status.language",
                language=self.context.translator.language_name,
            )
        )

    # ------------------------------------------------------------------ #
    # Команды сцены
    # ------------------------------------------------------------------ #
    def new_scene(self) -> None:
        """Создать пустую сцену с базовым набором объектов."""
        from engine.core.scene import Scene

        scene = Scene("Scene")
        documents = self.context.documents
        pristine = (
            len(documents) == 1
            and not documents[0].dirty
            and documents[0].path is None
            and len(documents[0].scene) == 0
        )
        if pristine:
            self.context.set_scene(scene)
        else:
            self.context.open_document(scene)
        self.context.create_object("Ground", mesh="plane", position=Vector3(0, -0.5, 0))
        cube = self.context.create_object(
            "Cube", mesh="cube", position=Vector3(0, 0.5, 0)
        )
        # Без источника света сцена рисуется плоским фоновым цветом: кладём
        # солнце с тенями, чтобы сразу было видно объём.
        sun = self._add_light_object("directional")
        if sun is not None:
            sun.get_component(LightComponent).light.cast_shadow = True
        # Сразу выделяем куб: пользователь видит заполненный инспектор.
        self.context.selection.select(cube)
        self.hierarchy.refresh()
        self.inspector.refresh()
        # Заготовка из стандартных объектов не считается правкой пользователя.
        self.context.mark_clean()
        self._update_title()

    # ------------------------------------------------------------------ #
    # Проекты
    # ------------------------------------------------------------------ #
    def new_project(self) -> Path | None:
        """Спросить имя и создать проект со стандартной структурой.

        Returns:
            Путь к созданному проекту или ``None``, если пользователь отказался.
        """
        name, accepted = QInputDialog.getText(
            self,
            self.context.tr("dialog.new_project.title"),
            self.context.tr("dialog.new_project.label"),
            text=self.context.tr("dialog.new_project.default"),
        )
        if not accepted or not name.strip():
            return None
        try:
            project = create_project(self.context.workspace_root, name)
        except (OSError, ValueError) as exc:
            self._show_error(str(exc))
            return None
        self.switch_project(project)
        self.show_status(
            self.context.tr("status.project_created", name=project.name)
        )
        return project

    def open_project(self) -> Path | None:
        """Выбрать каталог проекта и переключиться на него.

        Returns:
            Путь к выбранному проекту или ``None``, если выбор отменён.
        """
        ensure_projects_dir(self.context.workspace_root)
        selected = QFileDialog.getExistingDirectory(
            self,
            self.context.tr("dialog.open_project"),
            str(projects_root(self.context.workspace_root)),
        )
        if not selected:
            return None
        return self.switch_project(Path(selected))

    def switch_project(self, path: Path | str) -> Path:
        """Переключить редактор на другой корень проекта.

        Args:
            path: Каталог проекта.

        Returns:
            Путь, на который переключился редактор.
        """
        root = Path(path)
        self.context.set_project_root(root)
        self.show_status(self.context.tr("status.project_opened", name=root.name))
        return root

    def open_scene(self) -> None:
        """Выбрать файл сцены в диалоге и открыть его во вкладке."""
        path, _filter = QFileDialog.getOpenFileName(
            self,
            self.context.tr("dialog.open_scene"),
            str(scenes_root_for(self.context.project_root)),
            self.context.tr("dialog.scene_filter"),
        )
        if path:
            self.load_scene_file(path)

    def load_scene_file(self, path: Path | str) -> bool:
        """Открыть файл сцены в новой вкладке (или перейти к уже открытой).

        Args:
            path: Путь к ``*.scene.json``.

        Returns:
            ``True``, если сцена открыта.
        """
        target = Path(path)
        existing = self.context.find_document(target)
        if existing is not None:
            self.context.activate_document(existing)
            return True
        try:
            scene, sky, post = scene_io.load_scene_full(target, self.context.asset_db)
        except (OSError, ValueError) as exc:
            self._show_error(str(exc))
            return False
        document = self.context.open_document(scene, target)
        document.sky = sky
        document.post = post
        self.show_status(self.context.tr("status.loaded", path=str(target)))
        self._update_title()
        return True

    def save_scene(self) -> bool:
        """Сохранить активную сцену: без диалога, если путь уже известен."""
        return self._save_document(self.context.active_document, ask_path=False)

    def save_scene_as(self) -> bool:
        """Сохранить активную сцену под новым именем."""
        return self._save_document(self.context.active_document, ask_path=True)

    def _save_document(self, document: object, *, ask_path: bool) -> bool:
        """Записать сцену документа в файл.

        Args:
            document: Документ (вкладка) для сохранения.
            ask_path: Спрашивать путь, даже если он известен.

        Returns:
            ``True``, если файл записан.
        """
        path = document.path  # type: ignore[attr-defined]
        if ask_path or path is None:
            suggested = str(
                path
                or scenes_root_for(self.context.project_root)
                / f"{document.scene.name}.scene.json"  # type: ignore[attr-defined]
            )
            chosen, _filter = QFileDialog.getSaveFileName(
                self,
                self.context.tr("dialog.save_scene"),
                suggested,
                self.context.tr("dialog.scene_filter"),
            )
            if not chosen:
                return False
            path = chosen
        try:
            saved = scene_io.save_scene(
                document.scene,  # type: ignore[attr-defined]
                path,
                self.context.asset_db,
                document.sky,  # type: ignore[attr-defined]
                document.post,  # type: ignore[attr-defined]
            )
        except OSError as exc:
            self._show_error(str(exc))
            return False
        self.context.mark_saved(document, saved)  # type: ignore[arg-type]
        self.show_status(self.context.tr("status.saved", path=str(saved)))
        self._update_title()
        return True

    def _confirm_discard(self, document: object) -> bool:
        """Спросить про несохранённую сцену.

        Returns:
            ``True``, если можно закрывать (сохранили или отказались от правок).
        """
        if not document.dirty:  # type: ignore[attr-defined]
            return True
        name = document.display_name(  # type: ignore[attr-defined]
            self.context.tr("app.untitled_scene")
        )
        answer = QMessageBox.question(
            self,
            self.context.tr("dialog.unsaved_scene.title"),
            self.context.tr("dialog.unsaved_scene", name=name),
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            return self._save_document(document, ask_path=False)
        return True

    def close_scene_tab(self, index: int) -> bool:
        """Закрыть вкладку сцены, спросив про несохранённые изменения.

        Returns:
            ``True``, если вкладка закрыта.
        """
        if not 0 <= index < len(self.context.documents):
            return False
        if not self._confirm_discard(self.context.documents[index]):
            return False
        self.context.close_document(index)
        return True

    def create_primitive(self, mesh: str) -> GameObject:
        """Создать объект с примитивом в центре внимания камеры.

        Args:
            mesh: Ключ меша (``cube``, ``sphere``, ``plane``).

        Returns:
            Созданный объект.
        """
        names = {"cube": "Cube", "sphere": "Sphere", "plane": "Plane"}
        target = self.viewport.viewport.controller.target
        position = Vector3(target.x, max(target.y, 0.0) + 0.5, target.z)
        game_object = self.context.create_object(
            names.get(mesh, "Object"), mesh=mesh, position=position
        )
        self.context.selection.select(game_object)
        self.hierarchy.refresh()
        self.inspector.refresh()
        self.viewport.refresh()
        self.show_status(self.context.tr("status.created", name=game_object.name))
        return game_object

    def create_empty(self) -> GameObject:
        """Создать пустой объект без отрисовки."""
        game_object = self.context.create_object("EmptyObject")
        self.context.selection.select(game_object)
        self.hierarchy.refresh()
        self.inspector.refresh()
        self.show_status(self.context.tr("status.created", name=game_object.name))
        return game_object

    def create_text(self) -> GameObject:
        """Создать объект с текстом (``Text``, 32 px, billboard)."""
        with self.context.transaction(self.context.tr("undo.create", name="Text")):
            game_object = self.context.create_object("Text")
            self.context.add_component(
                game_object, TextRenderer(text="Text", size_px=32)
            )
        self.context.selection.select(game_object)
        self.hierarchy.refresh()
        self.inspector.refresh()
        self.show_status(self.context.tr("status.created", name=game_object.name))
        return game_object

    def create_prefab(self, path: str | None = None) -> GameObject | None:
        """Создать экземпляр префаба из файла ``*.prefab.json``.

        Args:
            path: Путь к префабу; ``None`` - показать диалог выбора.

        Returns:
            Корень экземпляра или ``None`` (отмена / ошибка).
        """
        context = self.context
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                context.tr("dialog.open_prefab"),
                str(assets_root_for(context.project_root)),
                context.tr("dialog.prefab_filter"),
            )
            if not path:
                return None
        try:
            root = instantiate_prefab_file(context, path)
        except (PrefabError, OSError) as exc:
            self._show_error(str(exc))
            return None
        context.selection.select(root)
        self.hierarchy.refresh()
        self.inspector.refresh()
        self.show_status(context.tr("status.created", name=root.name))
        return root

    _UI_NAMES: ClassVar[dict[type, str]] = {
        UIPanel: "UI Panel",
        UIButton: "UI Button",
        UIText: "UI Text",
        UIImage: "UI Image",
    }

    def create_ui(self, kind: type) -> GameObject:
        """Создать объект с UI-виджетом (``UIPanel``/``UIButton``/...)."""
        extra = {"text": "Text"} if kind is UIText else {}
        name = self._UI_NAMES[kind]
        with self.context.transaction(self.context.tr("undo.create", name=name)):
            game_object = self.context.create_object(name)
            self.context.add_component(game_object, kind, **extra)
        self.context.selection.select(game_object)
        self.hierarchy.refresh()
        self.inspector.refresh()
        self.show_status(self.context.tr("status.created", name=game_object.name))
        return game_object

    # ------------------------------------------------------------------ #
    # Камеры
    # ------------------------------------------------------------------ #
    def create_camera(self) -> GameObject:
        """Создать объект с камерой в точке интереса текущего вида.

        Returns:
            Созданный объект с компонентом :class:`CameraComponent`.
        """
        # Имена создаваемых объектов не переводятся: всегда английские.
        name = self.context.unique_object_name("Camera")
        game_object = self.viewport.add_camera_object(name)
        self.context.selection.select(game_object)
        self.hierarchy.refresh()
        self.inspector.refresh()
        self.viewport.refresh()
        self.show_status(self.context.tr("camera.status.created", name=name))
        return game_object

    # ------------------------------------------------------------------ #
    # Свет
    # ------------------------------------------------------------------ #
    def create_light(self, kind: str = "point") -> GameObject | None:
        """Создать объект с источником света.

        Args:
            kind: Тип источника (``"directional"``, ``"point"``, ``"spot"``,
                ``"ambient"``).

        Returns:
            Созданный объект или ``None``, если тип неизвестен.
        """
        game_object = self._add_light_object(kind)
        if game_object is None:
            return None
        self.hierarchy.refresh()
        self.inspector.refresh()
        self.viewport.refresh()
        self.show_status(
            self.context.tr(
                "light.status.created",
                kind=self.context.tr(f"light.kind.{kind}"),
            )
        )
        return game_object

    def _add_light_object(self, kind: str) -> GameObject | None:
        """Создать объект с источником света в точке интереса редактора.

        Источник появляется выше точки интереса: так он сразу виден в кадре, а
        не оказывается внутри пола. Направленный и прожекторный свет сразу
        смотрят вниз (направление света — это поворот объекта).

        Args:
            kind: Тип источника.

        Returns:
            Созданный объект или ``None``, если тип неизвестен.
        """
        if kind not in LIGHT_TYPES:
            _log.warning("Неизвестный тип света: %s", kind)
            return None
        light = create_light(kind)
        target = self.viewport.viewport.controller.target
        position = Vector3(target.x, target.y, target.z) + NEW_LIGHT_OFFSET

        name = self.context.unique_object_name(_LIGHT_OBJECT_NAMES[kind])
        context = self.context
        with context.transaction(context.tr("undo.create", name=name)):
            game_object = context.create_object(name, position=position)
            if isinstance(light, (DirectionalLight, SpotLight)):
                context.set_field(
                    game_object,
                    game_object.transform,
                    "rotation",
                    quaternion_from_forward(light.unit_direction),
                )
            context.add_component(
                game_object,
                LightComponent,
                light=light,
                setup=lambda c: c.sync_to_light(),
            )
        self.context.selection.select(game_object)
        return game_object

    def _restore_editor_camera(self) -> None:
        """Вернуть вид камеры редактора, сохранённый в прошлой сессии."""
        state = self.camera_session.get(EDITOR_CAMERA_KEY)
        if state is None:
            return
        controller = self.viewport.viewport.controller
        controller.restore(state, self.viewport.camera)

    def save_editor_camera(self) -> bool:
        """Запомнить текущий вид камеры редактора.

        Returns:
            ``True``, если состояние записано в файл.
        """
        controller = self.viewport.viewport.controller
        state = controller.capture(self.viewport.camera)
        self.camera_session.set(state, EDITOR_CAMERA_KEY)
        return self.camera_session.save()

    def _on_project_changed(self, root: object) -> None:
        """Перенести файл состояния камеры в новый проект."""
        self.save_editor_camera()
        self.camera_session.path = EditorCameraSession.default_path(root)  # type: ignore[arg-type]
        self.camera_session.load()
        self._restore_editor_camera()
        self._update_title()

    def add_model_from_asset(self, path: str) -> GameObject:
        """Добавить в сцену объект из файла модели.

        Args:
            path: Путь к файлу (``.obj``, ``.glb``, ...).

        Returns:
            Созданный объект.
        """
        name = Path(path).stem
        game_object = self.context.create_object(f"{name}", mesh=path)
        self.context.selection.select(game_object)
        self.hierarchy.refresh()
        self.inspector.refresh()
        self.viewport.refresh()
        self.show_status(self.context.tr("status.assets_hint", name=name))
        return game_object

    def delete_selected(self) -> None:
        """Удалить выделенные объекты."""
        count = self.context.delete_objects(list(self.context.selection.objects))
        if count:
            self.show_status(self.context.tr("status.deleted", count=count))
        self.hierarchy.refresh()
        self.inspector.refresh()
        self.viewport.refresh()

    def open_script_from_asset(self, path: str) -> None:
        """Открыть скрипт из панели ассетов в редакторе скриптов."""
        self.set_script_panel_visible(True)
        self.script_editor.open_script(path)

    def set_script_panel_visible(self, visible: bool) -> None:
        """Открыть или скрыть окно редактора скриптов.

        Окно самостоятельное, поэтому оно не пристыковывается к главному:
        при первом показе оно встаёт со сдвигом рядом, дальше перемещается
        пользователем как обычное окно.

        Args:
            visible: ``True`` — открыть окно.
        """
        if visible:
            if not self.script_editor.isVisible():
                self.script_editor.place_next_to(self)
            self.script_editor.show()
            self.script_editor.raise_()
            self.script_editor.activateWindow()
        else:
            self.script_editor.hide()
        self.action_script_panel.setChecked(visible)

    def duplicate_selected(self) -> None:
        """Дублировать выделенные объекты."""
        created: list[GameObject] = []
        for game_object in list(self.context.selection.objects):
            clone = self.context.duplicate_object(game_object)
            if clone is not None:
                created.append(clone)
        if created:
            self.context.selection.select_many(created)
            self.hierarchy.refresh()
            self.viewport.refresh()
            self.show_status(
                self.context.tr("status.duplicated", name=created[0].name)
            )

    def set_collider_mode(self, mode: str) -> None:
        """Выбрать режим каркасов коллайдеров и отметить пункт меню."""
        self.viewport.set_collider_mode(mode)
        self.collider_actions[mode].setChecked(True)

    def focus_selected(self) -> None:
        """Навести камеру на выделенный объект."""
        if not self.viewport.frame_selected():
            self.show_status(self.context.tr("status.no_selection"))

    # ------------------------------------------------------------------ #
    # Раскладка и вспомогательное
    # ------------------------------------------------------------------ #
    def _restore_layout(self) -> None:
        """Восстановить раскладку панелей и язык из настроек."""
        geometry = self.settings.value("geometry")
        state = self.settings.value("window_state")
        if isinstance(geometry, QByteArray) and not geometry.isEmpty():
            self.restoreGeometry(geometry)
        if isinstance(state, QByteArray) and not state.isEmpty():
            self.restoreState(state)
        # Раскладка восстанавливает видимость дока, поэтому синхронизируем галочку.
        self.action_script_panel.setChecked(self.script_editor.isVisible())
        mode = str(self.settings.value("collider_mode", self.viewport.collider_mode))
        if mode in COLLIDER_MODES:
            self.set_collider_mode(mode)
        script_geometry = self.settings.value("script_geometry")
        if isinstance(script_geometry, QByteArray) and not script_geometry.isEmpty():
            self.script_editor.restoreGeometry(script_geometry)
        language = str(
            self.settings.value("language", self.context.translator.language)
        )
        if language != self.context.translator.language:
            self.context.set_language(language)
            for code, action in self.language_actions.items():
                action.setChecked(code == language)

    def fit_to_available_screen(self) -> None:
        """Вписать окно в рабочую область экрана (без панели задач).

        Сохранённая геометрия может оказаться выше или шире текущего экрана
        (другой монитор, масштаб): тогда нижние панели уезжают за край. Размер
        ограничивается рабочей областью, а позиция сдвигается внутрь неё.
        """
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        frame = self.frameGeometry()
        extra_w = frame.width() - self.width()
        extra_h = frame.height() - self.height()
        width = min(self.width(), max(area.width() - extra_w, 1))
        height = min(self.height(), max(area.height() - extra_h, 1))
        if (width, height) != (self.width(), self.height()):
            self.resize(width, height)
        frame = self.frameGeometry()
        left = min(max(frame.left(), area.left()), area.right() - frame.width() + 1)
        top = min(max(frame.top(), area.top()), area.bottom() - frame.height() + 1)
        if (left, top) != (frame.left(), frame.top()):
            self.move(left, top)

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 - API Qt
        """При первом показе повторно вписать окно: экран уже известен."""
        super().showEvent(event)
        if not self._fitted_on_show:
            self._fitted_on_show = True
            self.fit_to_available_screen()

    def reset_layout(self) -> None:
        """Вернуть панели на исходные места."""
        for panel in (
            self.hierarchy,
            self.inspector,
            self.assets,
            self.console,
            self.stats_panel,
            self.import_panel,
        ):
            panel.setFloating(False)
            panel.show()
        self.addDockWidget(Qt.LeftDockWidgetArea, self.hierarchy)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.assets)
        self.splitDockWidget(self.hierarchy, self.assets, Qt.Vertical)
        self.addDockWidget(Qt.RightDockWidgetArea, self.inspector)
        # Статистика — вкладка рядом с ассетами, чтобы инспектору хватало высоты.
        self.addDockWidget(Qt.LeftDockWidgetArea, self.stats_panel)
        self.tabifyDockWidget(self.assets, self.stats_panel)
        self.assets.raise_()
        self.addDockWidget(Qt.BottomDockWidgetArea, self.console)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.import_panel)
        self.tabifyDockWidget(self.console, self.import_panel)
        self.console.raise_()
        self.addDockWidget(Qt.RightDockWidgetArea, self.input_panel)
        self.input_panel.hide()
        # Окно скриптов не пристыковано: его просто закрываем.
        self.script_editor.hide()
        self.set_script_panel_visible(False)
        self.viewport.refresh()

    def show_about(self) -> None:
        """Показать окно «О программе»."""
        QMessageBox.about(
            self,
            self.context.tr("dialog.about.title"),
            self.context.tr("dialog.about.text"),
        )

    def show_status(self, message: str) -> None:
        """Показать сообщение в статусной строке."""
        self.status_label.setText(message)

    def _show_error(self, message: str) -> None:
        """Показать сообщение об ошибке."""
        _log.error("%s", message)
        self.show_status(self.context.tr("status.error", message=message))

    def _scene_name(self) -> str:
        """Имя открытой сцены (или «без имени»)."""
        if self.context.scene_path is not None:
            return self.context.scene_path.name
        return self.context.tr("app.untitled_scene")

    def _update_title(self) -> None:
        """Показать в заголовке окна приложение, проект и сцену."""
        mark = "*" if self.context.active_document.dirty else ""
        self.setWindowTitle(
            f"{self.context.tr('app.title')} — "
            f"{self.context.project_root.name} / {self._scene_name()}{mark}"
        )

    def _on_selection_changed(self, objects: list[GameObject]) -> None:
        """Обновить статус и вьюпорт при смене выделения."""
        self.viewport.refresh()
        if objects:
            self.show_status(self.context.tr("status.selected", name=objects[0].name))
        else:
            self.show_status(self.context.tr("status.selection_cleared"))

    def _on_asset_selected(self, path: str) -> None:
        """Показать выбранный ассет в статусной строке и его настройки импорта."""
        self.show_status(Path(path).name)
        self.context.selection.clear()
        self.inspector.show_asset(path)

    def _update_import_progress(self) -> None:
        """Показать прогресс фонового импорта в статусной строке."""
        manager = self.context.imports
        done, total = (0, 0) if manager is None else manager.progress()
        busy = manager is not None and manager.busy and total > 0
        self.import_progress.setVisible(busy)
        if busy:
            self.import_progress.setRange(0, total)
            self.import_progress.setValue(done)
            self.import_progress.setFormat(
                self.context.tr("import.progress", done=done, total=total)
            )

    # ------------------------------------------------------------------ #
    # Play Mode
    # ------------------------------------------------------------------ #
    def _save_dirty_documents(self) -> bool:
        """Сохранить изменённые сцены перед запуском плеера."""
        for document in self.context.documents:
            if document.dirty and not self._save_document(document, ask_path=False):
                return False
        return True

    def _update_play_label(self) -> None:
        """Показать состояние плеера в статусной строке."""
        tr = self.context.tr
        state = self.play_mode.state
        if state == "playing":
            text = tr("play.state.playing", pid=self.play_mode.pid)
        elif state == "paused":
            text = tr("play.state.paused")
        else:
            text = tr("play.state.stopped")
        self.play_label.setText(text)

    def _on_play_state(self, state: str) -> None:
        self._update_play_label()
        if state == "editing":
            self._player_stats = None
            self._clear_player_loading()

    def _on_player_load_progress(self, progress: float, _phase: str) -> None:
        self._player_loading = True
        self.show_status(
            self.context.tr("play.loading", percent=round(progress * 100))
        )

    def _on_player_scene_loaded(self) -> None:
        self._clear_player_loading()

    def _clear_player_loading(self) -> None:
        if self._player_loading:
            self._player_loading = False
            self.show_status(self.context.tr("status.ready"))

    def _on_player_stats(self, stats: dict) -> None:
        self._player_stats = stats
        self._clear_player_loading()

    def _on_player_log(self, record: dict) -> None:
        level = getattr(logging, str(record.get("level", "INFO")), logging.INFO)
        self.console.append(f"[player] {record.get('msg', '')}", level)

    def _on_player_crashed(self, trace: str) -> None:
        """Вывести падение плеера в консоль и сообщить пользователю."""
        self.console.append(trace, logging.ERROR)
        self.show_status(self.context.tr("play.crash.title"))

    def _on_viewport_stats(self, fps: float, stats: object) -> None:
        """Сохранить статистику кадра для панели."""
        self._last_fps = fps
        self._last_render_stats = stats

    def _refresh_stats(self) -> None:
        """Обновить панель статистики (вызывается таймером)."""
        position = self.viewport.camera.position
        self.stats_panel.set_player_stats(self._player_stats)
        self.stats_panel.update_stats(
            self.context.scene.stats,
            self._last_render_stats,
            fps=self._last_fps,
            camera_position=f"{position.x:.2f}, {position.y:.2f}, {position.z:.2f}",
            camera_distance=f"{self.viewport.camera_distance():.2f}",
            driver=self.viewport.driver_description(),
        )

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - API Qt
        """Сохранить раскладку, вид камеры и завершить работу."""
        for document in list(self.context.documents):
            if not self._confirm_discard(document):
                event.ignore()
                return
        if not self.script_editor.confirm_close():
            event.ignore()
            return
        self.play_mode.shutdown()
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("window_state", self.saveState())
        self.settings.setValue("script_geometry", self.script_editor.saveGeometry())
        self.settings.setValue("language", self.context.translator.language)
        self.settings.setValue("collider_mode", self.viewport.collider_mode)
        self.settings.sync()
        # Вид камеры редактора сохраняется отдельно от раскладки: он привязан к
        # проекту, а не к пользовательским настройкам окна.
        self.save_editor_camera()
        self.viewport.dispose_preview()
        # Окно скриптов самостоятельное: закрываем его вместе с редактором.
        self.script_editor.close()
        self.context.stop_imports()
        _log.info("Редактор закрыт")
        super().closeEvent(event)

    def __repr__(self) -> str:
        return (
            f"<MainWindow scene='{self.context.scene.name}' "
            f"objects={len(self.context.scene)} "
            f"language='{self.context.translator.language}'>"
        )


def create_window(context: EditorContext | None = None, **kwargs: object) -> MainWindow:
    """Удобная фабрика главного окна.

    Args:
        context: Готовый контекст; ``None`` — создать со значениями по умолчанию.
        **kwargs: Аргументы :class:`EditorContext`, если контекст не передан.

    Returns:
        Настроенное окно редактора.
    """
    return MainWindow(context or EditorContext(**kwargs))  # type: ignore[arg-type]
