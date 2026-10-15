"""Стек команд отмены: ``Command``, ``CommandStack``, ``Transaction``.

Команда мутирует :class:`~engine.core.scene.Scene` через публичные методы и
умеет откатить своё действие. В отличие от снимков сцены, цена записи
пропорциональна размеру правки, а не размеру сцены.

Идентичность объектов. Ссылки на ``GameObject`` извне редактора **инвалидируются**
после undo/redo создания и удаления: откат пересоздаёт объект (тот же ``uid``,
новый runtime-``id`` и новый экземпляр). Долгоживущие ссылки (выделение,
панели) нужно хранить как ``uid`` и каждый раз искать через
``Scene.find_by_uid``.

Память. Команда хранит только то, что нужно для отката: старое и новое значение
поля. Крупные значения (длинные списки) копируются целиком, поэтому правки
одного поля в окне :data:`MERGE_WINDOW` сливаются в одну запись, а стек
ограничен :data:`DEFAULT_LIMIT` записями.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from engine.core.scene import Scene

DEFAULT_LIMIT = 200
MERGE_WINDOW = 1.0


@runtime_checkable
class Command(Protocol):
    """Обратимое действие над сценой."""

    name: str

    def do(self, scene: Scene) -> None:
        """Выполнить (или повторить после отката) действие."""

    def undo(self, scene: Scene) -> None:
        """Откатить действие; после ``undo`` сцена равна состоянию до ``do``."""

    def merge_with(self, other: Command) -> bool:
        """Поглотить следующую команду ``other`` (уже выполненную).

        Returns:
            ``True``, если ``other`` слита с этой командой и отдельной записи
            в стеке не нужно.
        """


class CompositeCommand:
    """Несколько команд как одна запись стека (результат транзакции)."""

    def __init__(self, name: str, commands: list[Command] | None = None) -> None:
        self.name = name
        self.commands: list[Command] = commands if commands is not None else []

    def do(self, scene: Scene) -> None:
        for command in self.commands:
            command.do(scene)

    def undo(self, scene: Scene) -> None:
        for command in reversed(self.commands):
            command.undo(scene)

    def merge_with(self, other: Command) -> bool:  # noqa: ARG002
        return False


class Transaction:
    """Группа команд, откатываемая одним undo (контекстный менеджер).

    Команды внутри блока выполняются сразу, но в стек попадает одна запись.
    Исключение внутри блока откатывает уже выполненные команды и пробрасывается.
    """

    def __init__(self, stack: CommandStack, name: str) -> None:
        self._stack = stack
        self.name = name
        self.commands: list[Command] = []

    def __len__(self) -> int:
        return len(self.commands)

    def __enter__(self) -> Transaction:
        self._stack._open.append(self)  # noqa: SLF001
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        stack = self._stack
        stack._open.pop()  # noqa: SLF001
        if exc_type is not None:
            for command in reversed(self.commands):
                command.undo(stack.scene)
            return
        if not self.commands:
            return
        if stack._open:  # noqa: SLF001
            stack._open[-1].commands.append(  # noqa: SLF001
                CompositeCommand(self.name, self.commands)
            )
            return
        stack._record(  # noqa: SLF001
            CompositeCommand(self.name, self.commands), merge=False
        )


class CommandStack:
    """Стеки undo/redo для одной сцены.

    Args:
        scene: Сцена, над которой работают команды.
        limit: Максимум записей в undo-стеке (старые вытесняются).
    """

    def __init__(
        self,
        scene: Scene,
        limit: int = DEFAULT_LIMIT,
    ) -> None:
        if limit < 1:
            raise ValueError("limit должен быть >= 1")
        self.scene = scene
        self.limit = limit
        self._undo: deque[Command] = deque()
        self._redo: list[Command] = []
        self._open: list[Transaction] = []
        self._listeners: list[Callable[[str], None]] = []
        # Идентификаторы состояний: у каждого шага - состояние *после* него.
        # По ним находится «чистая» точка (сохранённое на диск состояние), даже
        # если правки слились или старые записи вытеснены лимитом.
        self._counter = 0
        self._base_id = 0
        self._undo_ids: deque[int] = deque()
        self._redo_ids: list[int] = []
        self._clean_id: int | None = 0

    # ------------------------------------------------------------------ #
    # Подписка
    # ------------------------------------------------------------------ #
    def subscribe(self, callback: Callable[[str], None]) -> Callable[[], None]:
        """Подписаться на изменения сцены командами.

        Args:
            callback: Вызывается после ``execute``/``push``/``undo``/``redo``
                с названием действия (``"do"``, ``"undo"``, ``"redo"``).

        Returns:
            Функция отписки.
        """
        self._listeners.append(callback)

        def unsubscribe() -> None:
            if callback in self._listeners:
                self._listeners.remove(callback)

        return unsubscribe

    def _notify(self, kind: str) -> None:
        for callback in list(self._listeners):
            callback(kind)

    # ------------------------------------------------------------------ #
    # Состояние
    # ------------------------------------------------------------------ #
    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_name(self) -> str | None:
        """Название команды, которую отменит ``undo``."""
        return self._undo[-1].name if self._undo else None

    @property
    def redo_name(self) -> str | None:
        """Название команды, которую повторит ``redo``."""
        return self._redo[-1].name if self._redo else None

    @property
    def undo_count(self) -> int:
        return len(self._undo)

    @property
    def redo_count(self) -> int:
        return len(self._redo)

    @property
    def in_transaction(self) -> bool:
        return bool(self._open)

    @property
    def top(self) -> Command | None:
        """Последняя запись undo-стека (или ``None``)."""
        return self._undo[-1] if self._undo else None

    @property
    def redo_top(self) -> Command | None:
        """Запись, которую повторит ``redo`` (или ``None``)."""
        return self._redo[-1] if self._redo else None

    @property
    def state_id(self) -> int:
        """Идентификатор текущего состояния сцены относительно истории."""
        return self._undo_ids[-1] if self._undo_ids else self._base_id

    @property
    def is_clean(self) -> bool:
        """Текущее состояние совпадает с отмеченным :meth:`mark_clean`."""
        return self._clean_id == self.state_id

    def mark_clean(self) -> None:
        """Запомнить текущее состояние как сохранённое."""
        self._clean_id = self.state_id

    def _new_id(self) -> int:
        self._counter += 1
        return self._counter

    def clear(self) -> None:
        """Забыть всю историю (загрузка другой сцены)."""
        self._undo.clear()
        self._redo.clear()
        self._undo_ids.clear()
        self._redo_ids.clear()
        self._base_id = self._new_id()
        self._clean_id = None

    # ------------------------------------------------------------------ #
    # Операции
    # ------------------------------------------------------------------ #
    def execute(self, command: Command) -> None:
        """Выполнить команду и записать её в историю."""
        command.do(self.scene)
        self._after_done(command)

    def push(self, command: Command) -> None:
        """Записать уже выполненное действие (объект создан обычным кодом).

        ``do`` не вызывается: команда должна быть собрана по готовому
        результату (``CreateObjectCommand.capture`` и подобные).
        """
        self._after_done(command)

    def _after_done(self, command: Command) -> None:
        if self._open:
            self._open[-1].commands.append(command)
        else:
            self._record(command, merge=True)
        self._notify("do")

    def _record(self, command: Command, *, merge: bool) -> None:
        self._redo.clear()
        self._redo_ids.clear()
        if merge and self._undo and self._undo[-1].merge_with(command):
            self._undo_ids[-1] = self._new_id()
            return
        self._undo.append(command)
        self._undo_ids.append(self._new_id())
        while len(self._undo) > self.limit:
            self._undo.popleft()
            self._base_id = self._undo_ids.popleft()

    def undo(self) -> str | None:
        """Отменить последнюю запись.

        Returns:
            Название отменённой команды или ``None``, если отменять нечего.
        """
        if self._open:
            raise RuntimeError("undo внутри открытой транзакции")
        if not self._undo:
            return None
        command = self._undo.pop()
        command.undo(self.scene)
        self._redo.append(command)
        self._redo_ids.append(self._undo_ids.pop())
        self._notify("undo")
        return command.name

    def redo(self) -> str | None:
        """Повторить последнюю отменённую запись."""
        if self._open:
            raise RuntimeError("redo внутри открытой транзакции")
        if not self._redo:
            return None
        command = self._redo.pop()
        command.do(self.scene)
        self._undo.append(command)
        self._undo_ids.append(self._redo_ids.pop())
        self._notify("redo")
        return command.name

    def begin_transaction(self, name: str) -> Transaction:
        """Открыть транзакцию (использовать как ``with``).

        Вложенная транзакция входит в внешнюю одной записью.
        """
        return Transaction(self, name)

