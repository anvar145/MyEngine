"""Общее состояние редактора: сцена, выделение, переводчик и сервисы.

:class:`EditorContext` — единственная точка, через которую панели получают
доступ к сцене и друг к другу. Такой «контекст вместо глобальных переменных»
позволяет держать панели независимыми (они знают только про контекст) и
подменять сцену целиком — например, при открытии другого файла.

Example:
    >>> context = EditorContext(project_root=Path("."))
    >>> cube = context.create_object("Cube", mesh="cube")
    >>> context.selection.select(cube)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

from editor.commands import SetDocumentSettingCommand
from editor.i18n import Translator
from editor.project import LIBRARY_DIRNAME, assets_root_for, ensure_projects_dir
from editor.scene_document import SceneDocument
from engine.core.commands import (
    AddComponentCommand,
    Command,
    CreateObjectCommand,
    DeleteObjectCommand,
    RemoveComponentCommand,
    RenameObjectCommand,
    ReparentCommand,
    SetActiveCommand,
    SetFieldCommand,
    SetLayerCommand,
    SetTagCommand,
    Transaction,
)
from engine.core.component import Component
from engine.core.components import MeshRenderer, TextRenderer
from engine.core.event_bus import EventBus
from engine.core.game_object import GameObject
from engine.core.math3d import Vector3
from engine.core.physics_components import (
    CharacterControllerComponent,
    ColliderComponent,
    MeshColliderComponent,
    RigidBodyComponent,
)
from engine.core.resources import (
    AssetDatabase,
    AssetImported,
    AssetImportFailed,
    AssetMoved,
    AssetRemoved,
    ImportBatchFinished,
    ImportBatchProgress,
    ImportBatchStarted,
    ImportManager,
)
from engine.core.scene import Scene
from engine.core.ui_components import UIElement
from engine.rendering.camera import CameraComponent
from engine.rendering.lighting import LightComponent
from engine.rendering.post import PostSettings
from engine.rendering.settings import QualityTier
from engine.scripting.runtime import ScriptRuntime

__all__ = ["EditorContext", "SelectionModel"]

_log = logging.getLogger("editor.context")

#: Период опроса диска и применения результатов импорта (мс).
IMPORT_POLL_MS = 500

#: События менеджера импортов, после которых панели обновляют состояния.
_IMPORT_EVENTS = (
    AssetImported,
    AssetImportFailed,
    AssetMoved,
    AssetRemoved,
    ImportBatchStarted,
    ImportBatchProgress,
    ImportBatchFinished,
)


class SelectionModel(QObject):
    """Набор выделенных объектов с сигналами об изменении.

    Signals:
        changed: Список выделенных объектов изменился.
        cleared: Выделение снято.
    """

    changed = Signal(list)
    cleared = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._objects: list[GameObject] = []

    # ------------------------------------------------------------------ #
    # Состояние
    # ------------------------------------------------------------------ #
    @property
    def objects(self) -> tuple[GameObject, ...]:
        """Выделенные объекты (в порядке добавления)."""
        return tuple(self._objects)

    @property
    def primary(self) -> GameObject | None:
        """Первый выделенный объект — на него смотрит инспектор."""
        return self._objects[0] if self._objects else None

    @property
    def is_empty(self) -> bool:
        """``True``, если ничего не выделено."""
        return not self._objects

    def __len__(self) -> int:
        return len(self._objects)

    def __iter__(self) -> Iterator[GameObject]:
        return iter(self._objects)

    def __contains__(self, game_object: object) -> bool:
        return game_object in self._objects

    # ------------------------------------------------------------------ #
    # Изменение
    # ------------------------------------------------------------------ #
    def select(self, game_object: GameObject | None, *, additive: bool = False) -> None:
        """Выделить объект.

        Args:
            game_object: Объект или ``None`` (снять выделение).
            additive: Добавить к текущему выделению (Ctrl+клик).
        """
        if game_object is None:
            self.clear()
            return
        if game_object.is_destroyed or not game_object.is_valid:
            self.forget(game_object)
            return
        if additive:
            if game_object in self._objects:
                self._objects.remove(game_object)
            else:
                self._objects.append(game_object)
        else:
            self._objects = [game_object]
        self._emit()

    def select_many(self, objects: Iterable[GameObject]) -> None:
        """Выделить несколько объектов сразу."""
        self._objects = [
            item for item in objects if not item.is_destroyed and item.is_valid
        ]
        self._emit()

    def clear(self) -> None:
        """Снять выделение."""
        if not self._objects:
            return
        self._objects.clear()
        self.cleared.emit()
        self.changed.emit([])

    def forget(self, game_object: GameObject) -> None:
        """Убрать объект из выделения (например, после удаления)."""
        if game_object in self._objects:
            self._objects.remove(game_object)
            self._emit()

    def _emit(self) -> None:
        """Сообщить подписчикам о новом выделении."""
        self.changed.emit(list(self._objects))


class EditorContext(QObject):
    """Сервисы редактора: сцена, выделение, переводчик, ресурсы проекта.

    Args:
        scene: Открытая сцена; по умолчанию создаётся пустая.
        project_root: Корень активного проекта (без ``workspace_root`` он же
            рабочий корень).
        workspace_root: Рабочий корень, внутри которого лежит ``projects/``;
            по умолчанию совпадает с ``project_root``.
        language: Язык интерфейса.
        parent: Родительский QObject.

    Attributes:
        workspace_root: Базовый корень, рядом с которым лежит ``projects/``.
        project_root: Корень активного проекта.

    Signals:
        scene_changed: Сцена заменена целиком (открыт другой файл или вкладка).
        documents_changed: Изменился список открытых сцен или флаг «изменена».
        document_deactivating: Вкладка вот-вот сменится (аргумент — уходящий
            документ): UI запоминает в нём вид камеры.
        scene_modified: Сцена изменилась (создан, удалён или изменён объект).
        project_changed: Сменился корень проекта (открыт или создан проект).
        status_message: Текст для статусной строки.
        imports_changed: Состояние импорта изменилось (ассет импортирован,
            упал, удалён, пачка началась или закончилась).
        imports_restarted: Менеджер импортов создан заново или остановлен
            (смена проекта).
        history_changed: Изменились возможности undo/redo или их подписи.
    """

    history_changed = Signal()
    scene_changed = Signal(object)
    scene_modified = Signal()
    documents_changed = Signal()
    document_deactivating = Signal(object)
    project_changed = Signal(object)
    status_message = Signal(str)
    imports_changed = Signal()
    imports_restarted = Signal()

    def __init__(
        self,
        scene: Scene | None = None,
        *,
        project_root: Path | None = None,
        workspace_root: Path | None = None,
        language: str = "ru",
        shader_cache: bool = False,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        #: Хранить бинарники шейдерных программ в ``Library/ShaderCache``.
        #: Выключено по умолчанию, чтобы тесты не писали в каталог проекта.
        self.shader_cache = shader_cache
        # Базовый корень (рабочий каталог редактора) не меняется: рядом с ним
        # живёт папка projects/. Активный проект переключается отдельно.
        self.project_root = Path(project_root or workspace_root or Path.cwd())
        self.workspace_root = Path(workspace_root or self.project_root)
        self.translator = Translator(language, parent=self)
        self.selection = SelectionModel(self)
        self.scene = scene or Scene("Scene", event_bus=EventBus())
        self.documents: list[SceneDocument] = [SceneDocument(self.scene)]
        self.active_index = 0
        self._tracking_dirty = True
        self._stepping = False
        self.scene_modified.connect(self._on_scene_modified)
        self._bind_stack(self.active_document)
        self._script_runtime: ScriptRuntime | None = None
        self.imports: ImportManager | None = None
        self._import_timer: QTimer | None = None
        self._import_subscriptions: list = []
        _ensure_projects_directory(self.workspace_root)

    # ------------------------------------------------------------------ #
    # Перевод
    # ------------------------------------------------------------------ #
    def tr(self, key: str, **kwargs: object) -> str:
        """Перевести строку интерфейса (сокращение для ``context.translator``)."""
        return self.translator.tr(key, **kwargs)

    def set_language(self, language: str) -> bool:
        """Переключить язык интерфейса.

        Returns:
            ``True``, если язык изменился.
        """
        return self.translator.set_language(language)

    # ------------------------------------------------------------------ #
    # Работа со сценой
    # ------------------------------------------------------------------ #
    @property
    def active_document(self) -> SceneDocument:
        """Документ активной вкладки."""
        return self.documents[self.active_index]

    @property
    def scene_path(self) -> Path | None:
        """Файл активной сцены (``None`` для несохранённой)."""
        return self.active_document.path

    @scene_path.setter
    def scene_path(self, path: Path | None) -> None:
        self.active_document.path = path
        self.documents_changed.emit()

    def _on_scene_modified(self) -> None:
        """Пометить активную сцену изменённой (если отслеживание включено)."""
        document = self.active_document
        if self._tracking_dirty and not document.dirty:
            document.dirty = True
            self.documents_changed.emit()

    def _switch_to(self, index: int) -> None:
        """Сделать документ активным и сообщить панелям о новой сцене."""
        self.document_deactivating.emit(self.active_document)
        self.selection.clear()
        self.active_index = index
        self.scene = self.active_document.scene
        self._bind_stack(self.active_document)
        if self.scene.asset_source is None:
            self.scene.asset_source = self.asset_db  # геометрия mesh-коллайдеров
        self._tracking_dirty = False
        try:
            self.scene_changed.emit(self.scene)
            self.scene_modified.emit()
        finally:
            self._tracking_dirty = True
        self.documents_changed.emit()
        self.history_changed.emit()

    def set_scene(self, scene: Scene, path: Path | None = None) -> None:
        """Заменить сцену активной вкладки.

        Args:
            scene: Новая сцена.
            path: Путь к файлу сцены (``None`` для новой сцены).
        """
        self.documents[self.active_index] = SceneDocument(scene, path)
        self._switch_to(self.active_index)

    def find_document(self, path: Path | str) -> int | None:
        """Номер вкладки, в которой уже открыт файл сцены."""
        target = Path(path)
        for index, document in enumerate(self.documents):
            if document.path is not None and document.path == target:
                return index
        return None

    def open_document(self, scene: Scene, path: Path | None = None) -> SceneDocument:
        """Открыть сцену в новой вкладке (или перейти к уже открытой по пути).

        Args:
            scene: Сцена.
            path: Файл сцены.

        Returns:
            Документ новой (или найденной) вкладки.
        """
        existing = None if path is None else self.find_document(path)
        if existing is not None:
            self.activate_document(existing)
            return self.documents[existing]
        self.documents.append(SceneDocument(scene, path))
        self._switch_to(len(self.documents) - 1)
        return self.active_document

    def activate_document(self, index: int) -> bool:
        """Переключиться на вкладку.

        Returns:
            ``True``, если вкладка сменилась.
        """
        if not 0 <= index < len(self.documents) or index == self.active_index:
            return False
        self._switch_to(index)
        return True

    def close_document(self, index: int) -> None:
        """Закрыть вкладку без вопросов (вопрос о сохранении задаёт UI).

        Закрытие последней вкладки оставляет пустую сцену: у редактора
        всегда есть открытая сцена.
        """
        if not 0 <= index < len(self.documents):
            return
        closing_active = index == self.active_index
        if len(self.documents) == 1:
            self.documents[0] = SceneDocument(Scene("Scene", event_bus=EventBus()))
            self._switch_to(0)
            return
        if closing_active:
            self.document_deactivating.emit(self.active_document)
        del self.documents[index]
        if closing_active:
            self.active_index = min(index, len(self.documents) - 1)
            self.selection.clear()
            self.scene = self.active_document.scene
            self._tracking_dirty = False
            try:
                self.scene_changed.emit(self.scene)
                self.scene_modified.emit()
            finally:
                self._tracking_dirty = True
        elif index < self.active_index:
            self.active_index -= 1
        self.documents_changed.emit()

    def mark_saved(self, document: SceneDocument, path: Path) -> None:
        """Запомнить путь файла и снять флаг «изменена»."""
        document.commands.mark_clean()
        document.path = path
        document.dirty = False
        self.documents_changed.emit()

    def mark_clean(self) -> None:
        """Снять флаг «изменена» у активной сцены (свежесозданная заготовка).

        Текущее состояние становится началом истории: заготовку нельзя отменить.
        """
        document = self.active_document
        document.commands.clear()
        document.commands.mark_clean()
        self.history_changed.emit()
        if document.dirty:
            document.dirty = False
            self.documents_changed.emit()

    # ------------------------------------------------------------------ #
    # Команды, undo / redo
    # ------------------------------------------------------------------ #
    def _bind_stack(self, document: SceneDocument) -> None:
        """Подписаться на стек команд документа (один раз на документ)."""
        if document.bound:
            return
        document.bound = True

        def listener(_kind: str) -> None:
            stack = document.commands
            if self._stepping or stack.in_transaction:
                return
            if document is self.active_document:
                self._after_stack_change(document)

        document.commands.subscribe(listener)

    def _after_stack_change(self, document: SceneDocument) -> None:
        """Стек изменил сцену: оповестить панели, пересчитать флаг «изменена»."""
        self.scene_modified.emit()
        dirty = not document.commands.is_clean
        if document.dirty != dirty:
            document.dirty = dirty
            self.documents_changed.emit()
        self.history_changed.emit()

    def _selection_uids(self) -> tuple[str, ...]:
        """``uid`` выделенных объектов активной сцены."""
        return tuple(
            obj.uid
            for obj in self.selection.objects
            if not obj.is_destroyed and obj.scene is self.scene
        )

    def _select_uids(self, uids: Iterable[str]) -> None:
        """Выделить объекты по ``uid`` (исчезнувшие пропускаются)."""
        found = (self.scene.find_by_uid(uid) for uid in uids)
        self.selection.select_many(obj for obj in found if obj is not None)

    @property
    def history_enabled(self) -> bool:
        """Правки и undo/redo разрешены (не идёт Play Mode)."""
        play_mode = getattr(self, "play_mode", None)
        return not (play_mode is not None and play_mode.is_active)

    def execute(self, command: Command) -> None:
        """Выполнить команду над сценой и записать её в историю.

        В Play Mode история не ведётся: команда только применяется.
        """
        if not self.history_enabled:
            command.do(self.scene)
            self.notify_modified()
            return
        stack = self.active_document.commands
        before = self._selection_uids()
        stack.execute(command)
        if stack.top is command:
            command.selection_before = before  # type: ignore[attr-defined]

    def push(self, command: Command) -> None:
        """Записать в историю уже выполненное действие (см. ``CommandStack.push``)."""
        if not self.history_enabled:
            self.notify_modified()
            return
        stack = self.active_document.commands
        before = self._selection_uids()
        stack.push(command)
        if stack.top is command:
            command.selection_before = before  # type: ignore[attr-defined]

    @contextmanager
    def transaction(self, name: str) -> Iterator[Transaction]:
        """Несколько команд одной записью истории (один undo).

        Args:
            name: Подпись записи для пункта меню «Отменить».
        """
        document = self.active_document
        stack = document.commands
        outer = not stack.in_transaction
        before = self._selection_uids()
        tx = stack.begin_transaction(name)
        with tx:
            yield tx
        if not outer or not tx.commands:
            return
        top = stack.top
        if getattr(top, "commands", None) is tx.commands:
            top.selection_before = before  # type: ignore[union-attr]
        self._after_stack_change(document)

    def set_field(
        self,
        game_object: GameObject,
        component: Component | object,
        field: str,
        value: Any,
        *,
        name: str | None = None,
        via: str | None = None,
    ) -> bool:
        """Изменить поле компонента (или ``transform``) командой с undo.

        Args:
            game_object: Владелец компонента.
            component: Компонент или ``game_object.transform``.
            field: Имя атрибута.
            value: Новое значение.
            name: Подпись записи (по умолчанию «изменение <объект>»).
            via: Атрибут компонента с вложенным объектом (``"camera"``, ``"light"``).

        Returns:
            ``False``, если значение не изменилось и команды нет.
        """
        owner = component if via is None else getattr(component, via)
        old = getattr(owner, field)
        if _same_value(old, value):
            return False
        label = name or self.tr("undo.edit", name=game_object.name)
        self.execute(
            SetFieldCommand(game_object, component, field, old, value, label, via=via)
        )
        return True

    def set_attr(
        self,
        target: object,
        field: str,
        value: Any,
        *,
        game_object: GameObject | None = None,
    ) -> bool:
        """Изменить поле компонента, камеры или света, найдя владельца.

        Args:
            target: Компонент, ``Camera`` или ``Light`` (из компонента камеры
                и света соответственно).
            field: Имя атрибута.
            value: Новое значение.
            game_object: Владелец (если известен; иначе ищется по сцене).

        Returns:
            ``False``, если значение не изменилось или владельца нет
            (тогда значение просто присваивается без записи в историю).
        """
        if isinstance(target, Component):
            owner = target.game_object
            if owner is not None:
                return self.set_field(owner, target, field, value)
        else:
            for obj in self.scene.objects if game_object is None else (game_object,):
                pairs = ((CameraComponent, "camera"), (LightComponent, "light"))
                for kind, via in pairs:
                    for component in obj.get_components(kind):
                        if getattr(component, via) is target:
                            return self.set_field(obj, component, field, value, via=via)
        setattr(target, field, value)
        self.notify_modified()
        return False

    def set_document_setting(
        self, section: str, key: str, value: Any, *, name: str | None = None
    ) -> bool:
        """Изменить параметр неба (``sky``) или пост-обработки (``post``).

        Returns:
            ``False``, если значение не изменилось.
        """
        document = self.active_document
        created = False
        if section == "sky":
            settings: Any = document.sky
        elif document.post is None:
            created = True
            settings = PostSettings.for_quality(QualityTier.MEDIUM)
        else:
            settings = document.post
        old = getattr(settings, key)
        if _same_value(old, value):
            return False
        label = name or self.tr("undo.settings", name=section)
        self.execute(
            SetDocumentSettingCommand(
                document, section, key, old, value, label, created=created
            )
        )
        return True

    def add_component(
        self,
        game_object: GameObject,
        component: Any,
        *,
        setup: Callable[[Component], None] | None = None,
        **kwargs: Any,
    ) -> Component:
        """Добавить компонент к объекту (тип или экземпляр) с записью в историю.

        Args:
            game_object: Владелец.
            component: Тип или экземпляр компонента.
            setup: Доводка компонента до записи в историю (состояние
                снимается уже после неё).
            **kwargs: Аргументы конструктора для типа.
        """
        added = game_object.add_component(component, **kwargs)
        if setup is not None:
            setup(added)
        self.push(
            AddComponentCommand.capture(
                game_object,
                added,
                self.tr("undo.component_add", name=game_object.name),
            )
        )
        return added

    def remove_component(self, game_object: GameObject, component: Component) -> None:
        """Удалить компонент командой с undo."""
        self.execute(
            RemoveComponentCommand(
                game_object,
                component,
                self.tr("undo.component_remove", name=game_object.name),
            )
        )

    def rename_object(self, game_object: GameObject, name: str) -> bool:
        """Переименовать объект (правки в окне слияния - одной записью)."""
        if game_object.name == name:
            return False
        command = RenameObjectCommand(game_object, name)
        command.name = self.tr("undo.rename", name=game_object.name)
        self.execute(command)
        return True

    def set_object_active(self, game_object: GameObject, active: bool) -> bool:
        """Включить или выключить объект (``active_self``)."""
        if game_object.active_self == active:
            return False
        command = SetActiveCommand(game_object, active)
        command.name = self.tr("undo.active", name=game_object.name)
        self.execute(command)
        return True

    def set_object_tag(self, game_object: GameObject, tag: str) -> bool:
        """Сменить тег объекта."""
        if game_object.tag == tag:
            return False
        command = SetTagCommand(game_object, tag)
        command.name = self.tr("undo.edit", name=game_object.name)
        self.execute(command)
        return True

    def set_object_layer(self, game_object: GameObject, layer: int) -> bool:
        """Сменить слой объекта."""
        if game_object.layer == layer:
            return False
        command = SetLayerCommand(game_object, layer)
        command.name = self.tr("undo.edit", name=game_object.name)
        self.execute(command)
        return True

    @property
    def can_undo(self) -> bool:
        """Есть что отменять."""
        return self.history_enabled and self.active_document.commands.can_undo

    @property
    def can_redo(self) -> bool:
        """Есть что повторять."""
        return self.history_enabled and self.active_document.commands.can_redo

    def undo_text(self) -> str:
        """Подпись для пункта «Отменить» (пустая, если отменять нечего)."""
        return self.active_document.commands.undo_name or ""

    def redo_text(self) -> str:
        """Подпись для пункта «Повторить» (пустая, если повторять нечего)."""
        return self.active_document.commands.redo_name or ""

    def undo(self) -> bool:
        """Отменить последнюю правку сцены.

        Returns:
            ``True``, если сцена изменилась.
        """
        return self._step_history(forward=False)

    def redo(self) -> bool:
        """Повторить отменённую правку.

        Returns:
            ``True``, если сцена изменилась.
        """
        return self._step_history(forward=True)

    def _step_history(self, *, forward: bool) -> bool:
        """Шаг по стеку команд с восстановлением выделения (по ``uid``).

        Выделение снимается до шага: откат создания и удаления пересоздаёт
        объекты, и панели не должны держать устаревшие ссылки.
        """
        if not self.history_enabled:
            return False
        document = self.active_document
        stack = document.commands
        command = stack.redo_top if forward else stack.top
        if command is None:
            return False
        current = self._selection_uids()
        self.selection.clear()
        self._stepping = True
        try:
            if forward:
                stack.redo()
            else:
                stack.undo()
        finally:
            self._stepping = False
        if forward:
            target = getattr(command, "selection_after", None)
        else:
            command.selection_after = current  # type: ignore[attr-defined]
            target = getattr(command, "selection_before", None)
        self._select_uids(current if target is None else target)
        self._after_stack_change(document)
        return True

    def set_project_root(self, path: Path | str) -> bool:
        """Переключить редактор на другой проект.

        Панель ассетов, дерево скриптов и пути сцен берут корень отсюда.

        Args:
            path: Корень проекта.

        Returns:
            ``True``, если корень действительно изменился.
        """
        root = Path(path)
        if root == self.project_root:
            return False
        self.project_root = root
        self.active_document.path = None
        if self._script_runtime is not None:
            # Скрипты проекта ищут соседние файлы относительно нового корня.
            self._script_runtime.project_root = root
        if self.imports is not None:
            self.start_imports()
        self.project_changed.emit(root)
        return True

    # ------------------------------------------------------------------ #
    # Импорт ассетов
    # ------------------------------------------------------------------ #
    @property
    def assets_root(self) -> Path:
        """Каталог исходных ассетов активного проекта."""
        return assets_root_for(self.project_root)

    @property
    def library_root(self) -> Path:
        """Каталог кэша импорта (``Library``) активного проекта."""
        return self.project_root / LIBRARY_DIRNAME

    @property
    def asset_db(self) -> AssetDatabase | None:
        """База ассетов запущенного менеджера импортов (или ``None``)."""
        return self.imports.db if self.imports is not None else None

    @property
    def shader_cache_dir(self) -> Path | None:
        """Каталог кэша шейдерных программ или ``None``, если он выключен."""
        if not self.shader_cache:
            return None
        return self.library_root / "ShaderCache"

    def start_imports(self, *, interval_ms: int = IMPORT_POLL_MS) -> ImportManager:
        """Запустить слежение за ассетами активного проекта.

        Прежний менеджер (если был) останавливается. Менеджер создаётся только
        по явному вызову: сканирование пишет ``.meta`` рядом с исходниками, и
        тестам или скриптам этого не нужно.

        Args:
            interval_ms: Период таймера опроса диска.

        Returns:
            Новый менеджер импортов.
        """
        self.stop_imports(notify=False)
        database = AssetDatabase(
            self.assets_root,
            library=self.library_root,
            event_bus=EventBus(),
            read_only=_is_builtin_assets(self.assets_root),
        )
        manager = ImportManager(database)
        self._import_subscriptions = [
            manager.events.subscribe(event_type, self._on_import_event)
            for event_type in _IMPORT_EVENTS
        ]
        self.imports = manager
        for document in self.documents:
            # геометрия mesh-коллайдеров: и у стартовой сцены, и у открытых вкладок
            document.scene.asset_source = database
        manager.open()
        if self._import_timer is None:
            self._import_timer = QTimer(self)
            self._import_timer.timeout.connect(self._tick_imports)
        self._import_timer.start(interval_ms)
        self.imports_restarted.emit()
        self.imports_changed.emit()
        return manager

    def stop_imports(self, *, notify: bool = True) -> None:
        """Остановить слежение и дождаться идущих импортов."""
        if self._import_timer is not None:
            self._import_timer.stop()
        manager, self.imports = self.imports, None
        for subscription in self._import_subscriptions:
            subscription.cancel()
        self._import_subscriptions = []
        if manager is not None:
            manager.close()
            manager.db.close()
            if notify:
                self.imports_restarted.emit()

    def _tick_imports(self) -> None:
        """Тик таймера: заметить изменения на диске и применить результаты."""
        if self.imports is None:
            return
        try:
            self.imports.poll()
        except Exception:  # noqa: BLE001 - тик не должен ронять редактор
            _log.exception("Ошибка опроса ассетов")

    def _on_import_event(self, event: object) -> None:
        """Переслать событие менеджера панелям Qt-сигналом."""
        if isinstance(event, ImportBatchFinished):
            self.status_message.emit(
                self.tr("import.status.finished", ok=event.ok, failed=event.failed)
            )
        self.imports_changed.emit()

    @property
    def script_runtime(self) -> ScriptRuntime:
        """Общий реестр скриптов редактора.

        Один и тот же реестр используют окно редактора скриптов (оно выполняет
        компоненты) и инспектор (он прикрепляет скрипты к объектам): иначе
        прикреплённый скрипт не попал бы в общий обход кадра.
        """
        if self._script_runtime is None:
            self._script_runtime = ScriptRuntime(project_root=self.project_root)
        return self._script_runtime

    def unique_object_name(self, base: str) -> str:
        """Подобрать имя, не совпадающее с именами объектов сцены.

        Args:
            base: Желаемое имя.

        Returns:
            ``base`` или ``base 2``, ``base 3`` ... — первое свободное.
        """
        taken = {game_object.name for game_object in self.scene.objects}
        if base not in taken:
            return base
        index = 2
        while f"{base} {index}" in taken:
            index += 1
        return f"{base} {index}"

    def create_object(
        self,
        name: str,
        *,
        mesh: str | None = None,
        color: Vector3 | None = None,
        parent: GameObject | None = None,
        position: Vector3 | None = None,
    ) -> GameObject:
        """Создать объект в сцене (с компонентом :class:`MeshRenderer`).

        Args:
            name: Имя объекта.
            mesh: Ключ меша; ``None`` — объект без отрисовки.
            color: Цвет материала.
            parent: Родительский объект.
            position: Начальная позиция.

        Returns:
            Созданный объект.
        """
        # Индекс берём до создания объекта, чтобы палитра шла по порядку.
        index = len(self.scene)
        game_object = self.scene.create_game_object(
            name, parent=parent, position=position or Vector3.zero()
        )
        if mesh is not None:
            game_object.add_component(
                MeshRenderer, mesh=mesh, color=color or _default_color(index)
            )
        self.push(
            CreateObjectCommand.capture(
                game_object, self.tr("undo.create", name=game_object.name)
            )
        )
        return game_object

    def delete_objects(self, objects: Iterable[GameObject]) -> int:
        """Удалить объекты сцены командой (уничтожение немедленное).

        Во время сигнала ``scene_modified`` выделение ещё может содержать
        уничтоженные объекты: подписчики проверяют ``obj.is_destroyed``.

        Args:
            objects: Объекты для удаления.

        Returns:
            Число удалённых объектов.
        """
        targets = [
            obj
            for obj in dict.fromkeys(objects)
            if obj.scene is self.scene and not obj.is_destroyed
        ]
        # Потомок удаляемого предка уходит вместе с ним: команда на корень.
        roots = [
            obj
            for obj in targets
            if not any(o is not obj and o._is_ancestor_of(obj) for o in targets)  # noqa: SLF001
        ]
        if not roots:
            return 0
        name = roots[0].name if len(roots) == 1 else ""
        # Команда уничтожает объекты сразу: редактор не крутит игровой цикл,
        # а отложенное удаление оставило бы объект в дереве и во вьюпорте.
        with self.transaction(self.tr("undo.delete", name=name or "…")):
            for root in roots:
                self.execute(DeleteObjectCommand(root))
        # Подписчики `scene_modified` внутри транзакции ещё видят уничтоженные
        # объекты в выделении (команда уничтожает сразу), поэтому они обязаны
        # пропускать `is_destroyed`; снять выделение раньше нельзя: undo берёт
        # его из `selection_before`.
        # Выделение снимаем после записи: undo вернёт его по `selection_before`.
        for obj in targets:
            self.selection.forget(obj)
        return len(targets)

    def duplicate_object(self, game_object: GameObject) -> GameObject | None:
        """Создать копию объекта со всеми компонентами рядом с оригиналом.

        Args:
            game_object: Объект-образец.

        Returns:
            Копия или ``None``, если объект принадлежит другой сцене.
        """
        if game_object.scene is not self.scene:
            return None
        clone = self.scene.create_game_object(
            f"{game_object.name} (copy)",
            parent=game_object.parent,
            position=game_object.transform.position,
            rotation=game_object.transform.rotation,
            scale=game_object.transform.scale,
        )
        clone.tag = game_object.tag
        for component in game_object.components:
            if isinstance(component, MeshRenderer):
                clone.add_component(
                    MeshRenderer,
                    mesh_ref=component.mesh_ref,
                    color=component.color,
                    roughness=component.roughness,
                    metallic=component.metallic,
                    visible=component.visible,
                    cast_shadow=component.cast_shadow,
                    receive_shadow=component.receive_shadow,
                    sorting_layer=component.sorting_layer,
                    material_guid=component.material_guid,
                )
            elif isinstance(component, TextRenderer):
                clone.add_component(TextRenderer.from_dict(component.to_dict()))
            elif isinstance(component, UIElement):
                clone.add_component(type(component).from_dict(component.to_dict()))
            elif isinstance(
                component,
                (
                    CameraComponent,
                    LightComponent,
                    RigidBodyComponent,
                    ColliderComponent,
                    MeshColliderComponent,
                    CharacterControllerComponent,
                ),
            ):
                # Камеры и свет копируются через сериализацию: так клон получает
                # собственные независимые объекты Camera/Light.
                clone.add_component(type(component).from_dict(component.to_dict()))
        clone.transform.translate(Vector3(1.0, 0.0, 0.0))
        self.push(
            CreateObjectCommand.capture(clone, self.tr("undo.create", name=clone.name))
        )
        return clone

    def reparent_objects(
        self,
        objects: list[GameObject],
        parent: GameObject | None,
        row: int | None = None,
    ) -> bool:
        """Перенести объекты под ``parent`` (``None`` - в корень сцены).

        Args:
            objects: Переносимые объекты (в порядке вставки).
            parent: Новый родитель.
            row: Позиция среди текущих детей ``parent`` (до изъятия переносимых);
                ``None`` - в конец.

        Returns:
            ``True``, если сцена изменилась. Попытка сделать объект потомком
            самого себя или своего потомка отклоняется: сообщение уходит в
            статусную строку, сцена не меняется.
        """
        if not self.history_enabled:
            return False
        moved = [obj for obj in objects if not obj.is_destroyed]
        # Потомка выделенного предка отдельно не переносим: он едет с предком.
        moved = [
            obj
            for obj in moved
            if not any(o is not obj and o._is_ancestor_of(obj) for o in moved)  # noqa: SLF001
        ]
        if parent is not None and any(o._is_ancestor_of(parent) for o in moved):  # noqa: SLF001
            self.status_message.emit(self.tr("status.reparent_cycle"))
            return False
        siblings = list(self.scene.root_objects if parent is None else parent.children)
        if row is None:
            row = len(siblings)
        # Якорь - первый непереносимый брат с позиции ``row``: объекты встают
        # перед ним по очереди, их взаимный порядок сохраняется.
        anchor = next((s for s in siblings[max(0, row) :] if s not in moved), None)
        label = self.tr(
            "undo.reparent", name=moved[0].name if len(moved) == 1 else "…"
        )
        changed = False
        with self.transaction(label):
            for obj in moved:
                current = self.scene.root_objects if parent is None else parent.children
                rest = [s for s in current if s is not obj]
                target = len(rest) if anchor is None else rest.index(anchor)
                if obj.parent is parent and obj.sibling_index() == target:
                    continue
                self.execute(ReparentCommand(obj, parent, target, name=label))
                changed = True
        return changed

    def notify_modified(self) -> None:
        """Сообщить панелям, что сцена изменилась (правка без записи в историю)."""
        self.scene_modified.emit()

    def process_pending(self) -> None:
        """Применить отложенные удаления (вызывается перед обновлением мира)."""
        if self.scene.process_pending():
            self.scene_modified.emit()

    def __repr__(self) -> str:
        return (
            f"<EditorContext scene='{self.scene.name}' objects={len(self.scene)} "
            f"selected={len(self.selection)} language='{self.translator.language}'>"
        )


def _ensure_projects_directory(root: Path) -> None:
    """Создать папку проектов рядом с корнем проекта.

    Ошибки файловой системы не должны мешать запуску редактора: они попадают
    в лог, а интерфейс продолжает работать.

    Args:
        root: Корень, внутри которого создаётся ``projects/``.
    """
    try:
        ensure_projects_dir(root)
    except OSError as exc:  # pragma: no cover - зависит от файловой системы
        _log.error("Не удалось создать папку проектов в '%s': %s", root, exc)


def _same_value(old: object, new: object) -> bool:
    """Значения поля равны (сравнение массивов numpy и прочих не должно падать)."""
    try:
        return bool(old == new)
    except (TypeError, ValueError):
        return False


def _default_color(index: int) -> Vector3:
    """Подобрать приятный цвет для нового объекта (циклическая палитра)."""
    palette = (
        Vector3(0.87, 0.58, 0.24),
        Vector3(0.42, 0.66, 0.86),
        Vector3(0.55, 0.78, 0.48),
        Vector3(0.82, 0.45, 0.52),
        Vector3(0.72, 0.62, 0.86),
    )
    return palette[index % len(palette)]


def _is_builtin_assets(path: Path) -> bool:
    """``True``, если ``path`` — встроенный каталог ``assets/`` самого движка.

    Редактор только загружает такие ассеты и ничего не пишет рядом с ними
    (``.meta``).
    """
    builtin = Path(__file__).resolve().parents[1] / "assets"
    try:
        return path.resolve() == builtin
    except OSError:
        return False
