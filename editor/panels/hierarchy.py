"""Панель иерархии сцены: дерево объектов с поиском и контекстным меню."""

from __future__ import annotations

from PySide6.QtCore import (
    QItemSelectionModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    QUrl,
    Signal,
)
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QInputDialog,
    QLineEdit,
    QMenu,
    QMessageBox,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from editor.mime import OBJECT_UID_MIME, PREFAB_MIME
from editor.models.scene_tree_model import SceneTreeModel
from editor.panels.base import DockPanel
from editor.prefab_actions import (
    apply_all_overrides,
    instantiate_prefab_file,
    prefab_file,
    revert_all_overrides,
    save_object_as_prefab,
    unpack_object,
)
from editor.project import prefabs_root_for
from engine.core.game_object import GameObject
from engine.core.prefab_components import PrefabInstance
from engine.prefab import PrefabError, get_overrides, make_prefab_loader
from engine.prefab_save import find_instance_root

__all__ = ["HierarchyPanel"]


class _HierarchyFilter(QSortFilterProxyModel):
    """Фильтр дерева по подстроке имени (с учётом вложенности)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._needle = ""
        self.setRecursiveFilteringEnabled(True)

    def set_needle(self, text: str) -> None:
        """Задать подстроку поиска и пересчитать фильтр."""
        # Новый API Qt (6.10+): beginFilterChange/endFilterChange.
        self.beginFilterChange()
        self._needle = text.strip().lower()
        self.endFilterChange(QSortFilterProxyModel.Direction.Rows)

    def filterAcceptsRow(  # noqa: N802 - API Qt
        self, source_row: int, source_parent: QModelIndex
    ) -> bool:
        """Пропустить узел, если он сам или его потомок подходит под поиск."""
        if not self._needle:
            return True
        model = self.sourceModel()
        if model is None:
            return True
        index = model.index(source_row, 0, source_parent)
        name = str(model.data(index, Qt.DisplayRole) or "").lower()
        if self._needle in name:
            return True
        return self._has_matching_child(index)

    def _has_matching_child(self, parent: QModelIndex) -> bool:
        """Рекурсивно проверить детей узла."""
        model = self.sourceModel()
        if model is None:
            return False
        for row in range(model.rowCount(parent)):
            child = model.index(row, 0, parent)
            name = str(model.data(child, Qt.DisplayRole) or "").lower()
            if self._needle in name or self._has_matching_child(child):
                return True
        return False


class _HierarchyTree(QTreeView):
    """Дерево: перестановка объектов и приём префабов из панели ассетов.

    Signals:
        prefab_dropped: Брошен префаб (путь файла, индекс под курсором).
    """

    prefab_dropped = Signal(str, QModelIndex)

    @staticmethod
    def _prefab_path(event) -> str | None:
        mime = event.mimeData()
        if not mime.hasFormat(PREFAB_MIME):
            return None
        return bytes(mime.data(PREFAB_MIME)).decode("utf-8") or None

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - API Qt
        """Принять префаб; перестановку объектов ведёт стандартная логика Qt."""
        if self._prefab_path(event) is not None:
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # noqa: N802 - API Qt
        """То же, пока курсор двигается."""
        if self._prefab_path(event) is not None:
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802 - API Qt
        """Брошенный префаб - сигнал; брошенные объекты - перестановка."""
        path = self._prefab_path(event)
        if path is not None:
            event.acceptProposedAction()
            self.prefab_dropped.emit(path, self.indexAt(event.position().toPoint()))
            return
        if not event.mimeData().hasFormat(OBJECT_UID_MIME):
            super().dropEvent(event)
            return
        parent, row = self._drop_target(event.position().toPoint())
        self.model().dropMimeData(
            event.mimeData(), Qt.MoveAction, row, 0, parent
        )
        # IgnoreAction: иначе источник drag удалит «перенесённые» строки сам.
        event.setDropAction(Qt.IgnoreAction)
        event.accept()
        self.viewport().update()

    def _drop_target(self, point) -> tuple[QModelIndex, int]:
        """Родитель и строка вставки по позиции курсора (как у Qt)."""
        index = self.indexAt(point)
        if not index.isValid():
            return QModelIndex(), -1
        position = self.dropIndicatorPosition()
        if position == QAbstractItemView.DropIndicatorPosition.AboveItem:
            return index.parent(), index.row()
        if position == QAbstractItemView.DropIndicatorPosition.BelowItem:
            return index.parent(), index.row() + 1
        return index, -1


class HierarchyPanel(DockPanel):
    """Дерево объектов сцены.

    Панель двусторонне синхронизирована с :class:`~editor.context.SelectionModel`:
    выбор в дереве меняет выделение редактора, а выделение из других панелей
    подсвечивается в дереве.

    Signals:
        focus_requested: Запрошено приближение камеры к объекту (клавиша ``F``).
    """

    focus_requested = Signal(object)

    def __init__(self, context, parent: QWidget | None = None) -> None:
        super().__init__("panel.hierarchy", context, parent)
        self.model = SceneTreeModel(context.scene, self)
        self.proxy = _HierarchyFilter(self)
        self.proxy.setSourceModel(self.model)
        self.model.set_instance_state_provider(self._instance_modified)

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.search = QLineEdit(container)
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.proxy.set_needle)
        layout.addWidget(self.search)

        self.tree = _HierarchyTree(container)
        self.tree.setModel(self.proxy)
        self.tree.setHeaderHidden(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # объект тащится в поле-ссылку скрипта и внутри дерева (reparent/порядок);
        # из панели ассетов принимаются префабы
        self.tree.setDragEnabled(True)
        self.tree.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.tree.setDefaultDropAction(Qt.MoveAction)
        self.tree.setDropIndicatorShown(True)
        self.tree.setAcceptDrops(True)
        self.model.set_move_handler(self.context.reparent_objects)
        self.tree.prefab_dropped.connect(self._on_prefab_dropped)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        self.tree.doubleClicked.connect(self._rename_from_index)
        self.tree.selectionModel().selectionChanged.connect(self._on_tree_selection)
        layout.addWidget(self.tree, 1)
        self.setWidget(container)

        self.context.selection.changed.connect(self._sync_from_selection)
        self.context.scene_modified.connect(self.refresh)
        self.context.translator.language_changed.connect(self.retranslate)
        self.retranslate()
        self.refresh()

    # ------------------------------------------------------------------ #
    # Локализация
    # ------------------------------------------------------------------ #
    def retranslate(self, *_args: object) -> None:
        """Перевести подписи панели."""
        super().retranslate()
        self.search.setPlaceholderText(self.tr("hierarchy.search"))
        self.tree.setToolTip(self.tr("hierarchy.tooltip"))

    def set_scene(self, scene: object) -> None:
        """Переключить панель на другую сцену.

        Args:
            scene: Новая сцена движка.
        """
        self.model.set_scene(scene)  # type: ignore[arg-type]
        self.refresh()

    # ------------------------------------------------------------------ #
    # Обновление
    # ------------------------------------------------------------------ #
    def _instance_modified(self, root: GameObject) -> bool:
        """Есть ли у экземпляра отличия от префаба (цвет иконки)."""
        return bool(get_overrides(root, assets=self.context.asset_db))

    def refresh(self) -> None:
        """Перечитать дерево после изменения сцены."""
        if not self.model.is_stale():
            self.model.invalidate_instances()
            return
        expanded = self._expanded_objects()
        self.model.refresh()
        self._restore_expanded(expanded)
        self._sync_from_selection(list(self.context.selection.objects))

    def _expanded_objects(self) -> set[int]:
        """Собрать идентификаторы раскрытых узлов (только у объектов с детьми)."""
        expanded: set[int] = set()
        for game_object in self.model.scene.objects:
            if not game_object.children:
                continue
            index = self.proxy.mapFromSource(self.model.index_for(game_object))
            if index.isValid() and self.tree.isExpanded(index):
                expanded.add(game_object.id)
        return expanded

    def _restore_expanded(self, expanded: set[int]) -> None:
        """Раскрыть узлы, которые были раскрыты до обновления."""
        if not expanded:
            return
        for game_object in self.model.scene.objects:
            if game_object.id not in expanded:
                continue
            index = self.proxy.mapFromSource(self.model.index_for(game_object))
            if index.isValid():
                self.tree.expand(index)

    # ------------------------------------------------------------------ #
    # Синхронизация выделения
    # ------------------------------------------------------------------ #
    def _on_tree_selection(self, *_args: object) -> None:
        """Перенести выбор дерева в модель выделения редактора."""
        objects: list[GameObject] = []
        for index in self.tree.selectionModel().selectedIndexes():
            source_index = self.proxy.mapToSource(index)
            game_object = self.model.object_for(source_index)
            if game_object is not None:
                objects.append(game_object)
        if objects:
            self.context.selection.select_many(objects)
        else:
            self.context.selection.clear()

    def _sync_from_selection(self, objects: list[GameObject]) -> None:
        """Подсветить в дереве объекты, выделенные в редакторе."""
        selection_model = self.tree.selectionModel()
        selection_model.blockSignals(True)
        self.tree.clearSelection()
        for game_object in objects:
            if game_object.is_destroyed:  # удаление командой немедленное
                continue
            source_index = self.model.index_for(game_object)
            if not source_index.isValid():
                continue
            index = self.proxy.mapFromSource(source_index)
            if index.isValid():
                self.tree.setCurrentIndex(index)
                self.tree.selectionModel().select(
                    index,
                    QItemSelectionModel.SelectionFlag.Select
                    | QItemSelectionModel.SelectionFlag.Rows,
                )
                self.tree.scrollTo(index)
        selection_model.blockSignals(False)

    # ------------------------------------------------------------------ #
    # Действия
    # ------------------------------------------------------------------ #
    def _show_context_menu(self, position: object) -> None:
        """Показать контекстное меню дерева."""
        menu = QMenu(self)
        objects = list(self.context.selection.objects)
        rename = QAction(self.tr("inspector.name"), menu)
        duplicate = QAction(self.tr("menu.edit.duplicate"), menu)
        delete = QAction(self.tr("menu.edit.delete"), menu)
        deselect = QAction(self.tr("menu.edit.deselect"), menu)
        rename.setEnabled(len(objects) == 1)
        duplicate.setEnabled(bool(objects))
        delete.setEnabled(bool(objects))
        deselect.setEnabled(bool(objects))
        rename.triggered.connect(self._rename_selected)
        duplicate.triggered.connect(self._duplicate_selected)
        delete.triggered.connect(self._delete_selected)
        deselect.triggered.connect(self.context.selection.clear)
        save_prefab = QAction(self.tr("menu.hierarchy.save_prefab"), menu)
        save_prefab.setEnabled(bool(objects))
        save_prefab.triggered.connect(lambda: self.save_selected_as_prefab())
        menu.addAction(rename)
        menu.addAction(duplicate)
        menu.addSeparator()
        menu.addAction(save_prefab)
        instance_root = (
            find_instance_root(objects[0]) if len(objects) == 1 else None
        )
        if instance_root is not None:
            menu.addSeparator()
            for key, handler in (
                ("menu.hierarchy.open_prefab", self.open_prefab_file),
                ("menu.hierarchy.apply_all", self.apply_all_to_prefab),
                ("menu.hierarchy.revert_all", self.revert_all_overrides),
                ("menu.hierarchy.unpack", self.unpack_selected),
            ):
                action = QAction(self.tr(key), menu)
                action.triggered.connect(lambda _=False, fn=handler: fn())
                menu.addAction(action)
        menu.addSeparator()
        menu.addAction(delete)
        menu.addAction(deselect)
        menu.exec(self.tree.viewport().mapToGlobal(position))  # type: ignore[arg-type]

    def _on_prefab_dropped(self, path: str, index: QModelIndex) -> None:
        """Создать экземпляр префаба: ребёнком объекта под курсором или в корне."""
        parent = self.model.object_for(self.proxy.mapToSource(index))
        self.drop_prefab(path, parent)

    def drop_prefab(
        self, path: str, parent: GameObject | None = None
    ) -> GameObject | None:
        """Создать экземпляр префаба из файла (ребёнком ``parent``).

        Returns:
            Корень экземпляра или ``None`` (ошибка показана пользователю).
        """
        try:
            instance = instantiate_prefab_file(self.context, path, parent=parent)
        except (PrefabError, OSError) as exc:
            self._warn(str(exc))
            return None
        self.status(self.tr("status.created", name=instance.name))
        return instance

    def save_selected_as_prefab(self, path: str | None = None) -> GameObject | None:
        """Сохранить выделенный объект как префаб и заменить его экземпляром.

        Args:
            path: Файл ``*.prefab.json``; ``None`` - показать диалог (по умолчанию
                ``Assets/Prefabs/<имя объекта>.prefab.json``).

        Returns:
            Корень нового экземпляра или ``None`` (отмена / ошибка).
        """
        objects = list(self.context.selection.objects)
        if len(objects) != 1:
            self._warn(self.tr("dialog.save_prefab_one"))
            return None
        game_object = objects[0]
        if path is None:
            folder = prefabs_root_for(self.context.project_root)
            path, _ = QFileDialog.getSaveFileName(
                self,
                self.tr("dialog.save_prefab"),
                str(folder / f"{game_object.name}.prefab.json"),
                self.tr("dialog.prefab_filter"),
            )
            if not path:
                return None
            if not path.endswith(".prefab.json"):
                path = path.removesuffix(".json") + ".prefab.json"
        try:
            instance = save_object_as_prefab(self.context, game_object, path)
        except (PrefabError, OSError) as exc:
            self._warn(f"{self.tr('dialog.save_prefab_error')}: {exc}")
            return None
        self.status(self.tr("status.prefab_saved", name=instance.name))
        return instance

    def _selected_instance_root(self) -> GameObject | None:
        """Корень экземпляра для единственного выделенного объекта."""
        objects = list(self.context.selection.objects)
        return find_instance_root(objects[0]) if len(objects) == 1 else None

    def unpack_selected(self, confirm: bool = True) -> GameObject | None:
        """Разъединить экземпляр выделенного объекта (Unpack).

        Args:
            confirm: Спросить подтверждение (правки префаба больше не
                применятся к этим объектам).

        Returns:
            Бывший корень экземпляра или ``None`` (отмена / не экземпляр).
        """
        root = self._selected_instance_root()
        if root is None:
            return None
        if confirm:
            answer = QMessageBox.question(
                self,
                self.tr("menu.hierarchy.unpack"),
                self.tr("dialog.unpack_confirm", name=root.name),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return None
        try:
            result = unpack_object(self.context, root)
        except PrefabError as exc:
            self._warn(str(exc))
            return None
        self.status(self.tr("status.unpacked", name=result.name))
        return result

    def apply_all_to_prefab(self) -> int:
        """Записать все overrides выделенного экземпляра в файл префаба."""
        root = self._selected_instance_root()
        if root is None:
            return 0
        try:
            return apply_all_overrides(
                self.context, root, warn=self.status
            )
        except (PrefabError, OSError) as exc:
            self._warn(str(exc))
            return 0

    def revert_all_overrides(self) -> int:
        """Вернуть все значения выделенного экземпляра к префабу."""
        root = self._selected_instance_root()
        if root is None:
            return 0
        try:
            return revert_all_overrides(self.context, root)
        except (PrefabError, OSError) as exc:
            self._warn(str(exc))
            return 0

    def open_prefab_file(self) -> bool:
        """Открыть файл префаба во внешнем редакторе JSON.

        Префаб другого формата, чем сцена, поэтому вкладкой сцены он не
        открывается (см. docs/EDITOR.md).
        """
        root = self._selected_instance_root()
        marker = root.get_component(PrefabInstance) if root is not None else None
        if marker is None:
            return False
        try:
            loader = make_prefab_loader(
                self.context.asset_db, self.context.project_root
            )
            path = prefab_file(
                self.context, loader(marker.prefab_guid, marker.prefab_path)
            )
        except (PrefabError, OSError) as exc:
            self._warn(str(exc))
            return False
        return bool(QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))))

    def _warn(self, text: str) -> None:
        """Показать предупреждение пользователю."""
        QMessageBox.warning(self, self.tr("dialog.save_prefab"), text)

    def _rename_from_index(self, index: QModelIndex) -> None:
        """Переименовать объект двойным кликом."""
        game_object = self.model.object_for(self.proxy.mapToSource(index))
        if game_object is None:
            return
        self.context.selection.select(game_object)
        self._rename_selected()

    def _rename_selected(self) -> None:
        """Запросить новое имя у пользователя."""
        game_object = self.context.selection.primary
        if game_object is None:
            return
        name, accepted = QInputDialog.getText(
            self,
            self.tr("inspector.name"),
            self.tr("inspector.name"),
            text=game_object.name,
        )
        if accepted and name.strip():
            self.context.rename_object(game_object, name.strip())
            self.status(self.tr("status.selected", name=game_object.name))

    def _duplicate_selected(self) -> None:
        """Дублировать выделенные объекты."""
        created: list[GameObject] = []
        for game_object in list(self.context.selection.objects):
            clone = self.context.duplicate_object(game_object)
            if clone is not None:
                created.append(clone)
        if created:
            self.context.selection.select_many(created)

    def _delete_selected(self) -> None:
        """Удалить выделенные объекты."""
        count = self.context.delete_objects(list(self.context.selection.objects))
        if count:
            self.status(self.tr("status.deleted", count=count))

    def keyPressEvent(self, event) -> None:  # noqa: N802 - API Qt
        """Обработать ``Delete`` и ``F``."""
        if event.matches(QKeySequence.Delete):
            self._delete_selected()
            return
        if event.key() == Qt.Key_F:
            game_object = self.context.selection.primary
            if game_object is not None:
                self.focus_requested.emit(game_object)
            return
        super().keyPressEvent(event)
