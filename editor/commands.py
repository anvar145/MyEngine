"""Команды редактора над данными документа вне сцены (небо, пост-обработка).

``Command.do/undo`` получают сцену, но эти настройки живут в
:class:`~editor.scene_document.SceneDocument`, поэтому команда держит ссылку на
документ.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from engine.core.commands import MERGE_WINDOW, Command
from engine.core.commands.object_commands import clone_value
from engine.rendering.post import PostSettings
from engine.rendering.settings import QualityTier

if TYPE_CHECKING:
    from editor.scene_document import SceneDocument
    from engine.core.scene import Scene


class SetDocumentSettingCommand:
    """Правка одного параметра неба (``sky``) или пост-обработки (``post``).

    Если секции ``post`` в документе ещё не было, первая правка создаёт её, а
    откат этой правки возвращает ``None`` (файл сцены снова не получает секцию).

    Args:
        document: Документ, чьи настройки правятся.
        section: ``"sky"`` или ``"post"``.
        key: Имя параметра.
        old: Значение до правки.
        new: Значение после правки.
        name: Название записи.
        created: Секция создана этой правкой (только для ``post``).
        stamp: Момент правки; для тестов.
    """

    def __init__(
        self,
        document: SceneDocument,
        section: str,
        key: str,
        old: Any,
        new: Any,
        name: str,
        *,
        created: bool = False,
        stamp: float | None = None,
    ) -> None:
        self.document = document
        self.section = section
        self.key = key
        self.old = clone_value(old)
        self.new = clone_value(new)
        self.name = name
        self.created = created
        self.stamp = time.monotonic() if stamp is None else stamp

    def _settings(self) -> Any:
        if self.section == "sky":
            return self.document.sky
        if self.document.post is None:
            self.document.post = PostSettings.for_quality(QualityTier.MEDIUM)
        return self.document.post

    def do(self, scene: Scene) -> None:  # noqa: ARG002
        self._settings().set_value(self.key, clone_value(self.new))

    def undo(self, scene: Scene) -> None:  # noqa: ARG002
        self._settings().set_value(self.key, clone_value(self.old))
        if self.created:
            self.document.post = None

    def merge_with(self, other: Command) -> bool:
        if (
            not isinstance(other, SetDocumentSettingCommand)
            or other.document is not self.document
            or other.section != self.section
            or other.key != self.key
            or other.stamp - self.stamp > MERGE_WINDOW
        ):
            return False
        self.new = other.new
        self.stamp = other.stamp
        return True
