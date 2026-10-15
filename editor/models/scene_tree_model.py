"""Qt-модели редактора: дерево сцены и фильтр ассетов.

Модель дерева читает иерархию напрямую из ECS-сцены (``GameObject.parent`` и
``GameObject.children``), поэтому редактор всегда показывает реальное состояние
движка, а не собственную копию.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QAbstractItemModel, QMimeData, QModelIndex, Qt

from editor.mime import OBJECT_MIME, OBJECT_UID_MIME
from engine.core.game_object import GameObject
from engine.core.prefab_components import PrefabInstance
from engine.core.scene import Scene

__all__ = ["SceneTreeModel", "prefab_display_name"]


#: Обработчик перестановки: (объекты, новый родитель, строка среди его детей).
MoveHandler = Callable[[list[GameObject], "GameObject | None", "int | None"], bool]

#: Флаги узла дерева (собираются один раз: ``|`` по enum Qt медленный).
_ITEM_FLAGS = Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled


class SceneTreeModel(QAbstractItemModel):
    """Дерево объектов сцены для ``QTreeView``.

    Модель ленивая: дети запрашиваются у объектов сцены по требованию, а
    обновление выполняется методом :meth:`refresh` (вызывается редактором
    после создания, удаления или переименования объектов).

    Args:
        scene: Сцена, которую отображает модель.
        parent: Родительский QObject.
    """

    def __init__(self, scene: Scene, parent: object = None) -> None:
        super().__init__(parent)
        self._scene = scene
        # Кэши порядка детей: без них отрисовка дерева квадратична по числу узлов.
        self._children_cache: dict[int | None, tuple[GameObject, ...]] = {}
        self._row_cache: dict[int, int] = {}
        self._last_signature: tuple | None = None
        self._state_provider: Callable[[GameObject], bool] | None = None
        self._state_cache: dict[str, bool] = {}
        self._move_handler: MoveHandler | None = None

    # ------------------------------------------------------------------ #
    # Сцена
    # ------------------------------------------------------------------ #
    @property
    def scene(self) -> Scene:
        """Отображаемая сцена."""
        return self._scene

    def set_scene(self, scene: Scene) -> None:
        """Переключить модель на другую сцену."""
        self.beginResetModel()
        self._scene = scene
        self._last_signature = None
        self._reset_caches()
        self.endResetModel()

    def _signature(self) -> tuple:
        """Всё, что видно в дереве: порядок, имена, родители, активность."""
        # Порядок братьев входит в подпись списком id детей (линейно по сцене:
        # ``sibling_index`` на каждом объекте дал бы квадрат).
        objects = tuple(
            (
                obj.id,
                obj.name,
                obj.parent.id if obj.parent is not None else 0,
                obj.active_in_hierarchy,
                tuple(child.id for child in obj.children),
            )
            for obj in self._scene.objects
        )
        return (objects, tuple(root.id for root in self._scene.root_objects))

    def is_stale(self) -> bool:
        """``True``, если вид дерева отличается от последнего обновления.

        Правка трансформа или материала дерево не меняет, и пересобирать его
        на каждый такой шаг — главный источник лагов на больших сценах.
        """
        return self._signature() != self._last_signature

    def refresh(self) -> None:
        """Перечитать структуру сцены (раскрытие узлов хранит панель)."""
        self._last_signature = self._signature()
        self.beginResetModel()
        self._reset_caches()
        self.endResetModel()

    def _reset_caches(self) -> None:
        """Сбросить кэши порядка детей."""
        self._children_cache.clear()
        self._row_cache.clear()
        self._state_cache.clear()

    # ------------------------------------------------------------------ #
    # Иерархия QAbstractItemModel
    # ------------------------------------------------------------------ #
    def index(
        self, row: int, column: int, parent: QModelIndex = QModelIndex()
    ) -> QModelIndex:
        """Создать индекс для узла дерева."""
        if column != 0 or row < 0:
            return QModelIndex()
        parent_object = self.object_for(parent)
        children = self._children_of(parent_object)
        if row >= len(children):
            return QModelIndex()
        return self.createIndex(row, column, children[row])

    def parent(self, index: QModelIndex) -> QModelIndex:  # type: ignore[override]
        """Индекс родительского узла."""
        if not index.isValid():
            return QModelIndex()
        game_object = self.object_for(index)
        if game_object is None:
            return QModelIndex()
        parent_object = game_object.parent
        if parent_object is None:
            return QModelIndex()
        return self.createIndex(
            self._row_of(parent_object), 0, parent_object
        )

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802 - API Qt
        """Число дочерних узлов."""
        if parent.column() > 0:
            return 0
        return len(self._children_of(self.object_for(parent)))

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        """Число колонок (в дереве сцены одна — имя)."""
        return 1

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> object:
        """Данные узла: имя, подсказка, цвет неактивных объектов."""
        if not index.isValid():
            return None
        game_object = self.object_for(index)
        if game_object is None:
            return None
        if role == Qt.DisplayRole:
            return game_object.name
        if role == Qt.ToolTipRole:
            text = (
                f"{game_object.name}\n"
                f"id: {game_object.id}\n"
                f"tag: {game_object.tag}\n"
                f"components: {game_object.num_components()}"
            )
            marker = game_object.get_component(PrefabInstance)
            if marker is not None and marker.instance_uid:
                text += f"\nprefab: {prefab_display_name(marker)}"
            return text
        if role == Qt.DecorationRole:
            return self._instance_icon(game_object)
        if role == Qt.ForegroundRole and not game_object.active_in_hierarchy:
            from PySide6.QtGui import QColor

            from editor.theme import PALETTE

            return QColor(PALETTE["text_muted"])
        return None

    def set_instance_state_provider(
        self, provider: Callable[[GameObject], bool] | None
    ) -> None:
        """Источник состояния экземпляров: ``True`` - есть изменения (overrides)."""
        self._state_provider = provider
        self._state_cache.clear()

    def invalidate_instances(self) -> None:
        """Сбросить кэш состояния корней экземпляров и перерисовать иконки."""
        self._state_cache.clear()
        roots = [o for o in self._scene.objects if o.has_component(PrefabInstance)]
        for game_object in roots:
            index = self.index_for(game_object)
            if index.isValid():
                self.dataChanged.emit(index, index, [Qt.DecorationRole])

    def _instance_icon(self, game_object: GameObject) -> object:
        """Иконка корня экземпляра префаба; цвет зависит от наличия overrides."""
        marker = game_object.get_component(PrefabInstance)
        if marker is None or not marker.instance_uid:
            return None
        modified = self._state_cache.get(game_object.uid)
        if modified is None:
            try:
                modified = bool(
                    self._state_provider(game_object) if self._state_provider else False
                )
            except Exception:  # noqa: BLE001 - иконка не должна ронять дерево
                modified = False
            self._state_cache[game_object.uid] = modified
        from editor.theme import icon

        return icon("package", "accent" if modified else "text_muted")

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        """Разрешить выбор, перетаскивание узлов и сброс на узел/в корень."""
        if not index.isValid():
            return Qt.ItemIsDropEnabled
        return _ITEM_FLAGS

    def supportedDropActions(self) -> Qt.DropAction:  # noqa: N802 - API Qt
        """Перестановка в иерархии - перемещение."""
        return Qt.MoveAction

    def mimeTypes(self) -> list[str]:  # noqa: N802 - API Qt
        """Типы данных перетаскивания: id объекта (поля скриптов) и uid-список."""
        return [OBJECT_MIME, OBJECT_UID_MIME]

    def mimeData(self, indexes: list[QModelIndex]) -> QMimeData:  # noqa: N802 - API Qt
        """Данные перетаскивания.

        ``OBJECT_MIME`` - id первого объекта (для полей скриптов);
        ``OBJECT_UID_MIME`` - ``uid`` всех перетаскиваемых объектов.
        """
        data = QMimeData()
        uids: list[str] = []
        for index in indexes:
            game_object = self.object_for(index)
            if game_object is None or game_object.uid in uids:
                continue
            if not uids:
                data.setData(OBJECT_MIME, str(game_object.id).encode("ascii"))
            uids.append(game_object.uid)
        if uids:
            data.setData(OBJECT_UID_MIME, "\n".join(uids).encode("ascii"))
        return data

    # -- перестановка (drag-and-drop внутри дерева) ----------------------- #
    def set_move_handler(self, handler: MoveHandler | None) -> None:
        """Обработчик перестановки: ``handler(объекты, новый_родитель, строка)``.

        ``строка`` - позиция среди текущих детей родителя до изъятия
        переносимых объектов (``None`` - в конец). Модель сама сцену не меняет:
        правки идут через команды редактора.
        """
        self._move_handler = handler

    def _dragged_objects(self, data: QMimeData) -> list[GameObject]:
        if not data.hasFormat(OBJECT_UID_MIME):
            return []
        raw = bytes(data.data(OBJECT_UID_MIME)).decode("ascii")
        found = (self._scene.find_by_uid(uid) for uid in raw.split("\n") if uid)
        return [obj for obj in found if obj is not None]

    def canDropMimeData(  # noqa: N802 - API Qt
        self,
        data: QMimeData,
        action: Qt.DropAction,  # noqa: ARG002
        row: int,  # noqa: ARG002
        column: int,  # noqa: ARG002
        parent: QModelIndex,  # noqa: ARG002
    ) -> bool:
        """Принимать перестановку объектов этой сцены."""
        return self._move_handler is not None and bool(self._dragged_objects(data))

    def dropMimeData(  # noqa: N802 - API Qt
        self,
        data: QMimeData,
        action: Qt.DropAction,
        row: int,
        column: int,  # noqa: ARG002
        parent: QModelIndex,
    ) -> bool:
        """Перенести брошенные объекты под ``parent`` на позицию ``row``.

        ``row == -1`` - бросок на сам узел (в конец его детей).
        """
        if action == Qt.IgnoreAction:
            return True
        if not self.canDropMimeData(data, action, row, column, parent):
            return False
        objects = self._dragged_objects(data)
        assert self._move_handler is not None
        target = self.object_for(parent)
        return bool(self._move_handler(objects, target, None if row < 0 else row))

    def headerData(  # noqa: N802 - API Qt
        self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole
    ) -> object:
        """Заголовок колонки."""
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return "Object"
        return None

    # ------------------------------------------------------------------ #
    # Помощники
    # ------------------------------------------------------------------ #
    def object_for(self, index: QModelIndex) -> GameObject | None:
        """Объект, стоящий за индексом модели.

        Args:
            index: Индекс модели.

        Returns:
            Игровой объект или ``None`` для корневого индекса.
        """
        if not index.isValid():
            return None
        internal = index.internalPointer()
        return internal if isinstance(internal, GameObject) else None

    def index_for(self, game_object: GameObject) -> QModelIndex:
        """Индекс модели для объекта.

        Args:
            game_object: Искомый объект.

        Returns:
            Индекс или недействительный индекс, если объект не из этой сцены.
        """
        if game_object.scene is not self._scene:
            return QModelIndex()
        return self.createIndex(self._row_of(game_object), 0, game_object)

    def _children_of(self, game_object: GameObject | None) -> tuple[GameObject, ...]:
        """Дети узла: корни сцены для ``None``, иначе дети объекта."""
        key = None if game_object is None else game_object.id
        children = self._children_cache.get(key)
        if children is None:
            children = (
                self._scene.root_objects
                if game_object is None
                else game_object.children
            )
            self._children_cache[key] = children
            for row, child in enumerate(children):
                self._row_cache[child.id] = row
        return children

    def _row_of(self, game_object: GameObject) -> int:
        """Позиция объекта среди соседей."""
        self._children_of(game_object.parent)
        return self._row_cache.get(game_object.id, 0)


def prefab_display_name(marker: PrefabInstance) -> str:
    """Имя префаба для подсказки: имя файла без ``.prefab.json``."""
    path = marker.prefab_path.replace("\\", "/")
    name = Path(path).name if path else ""
    for suffix in (".prefab.json", ".json"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name or marker.prefab_guid or "?"
