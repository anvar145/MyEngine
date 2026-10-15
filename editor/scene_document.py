"""Открытая сцена редактора: то, что живёт во вкладке над окном сцены."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from engine.core.commands import CommandStack
from engine.core.scene import Scene
from engine.rendering.post import PostSettings
from engine.rendering.sky import SkySettings

#: Суффикс файлов сцен; нужен, чтобы получить короткое имя вкладки.
SCENE_SUFFIX = ".scene.json"


@dataclass
class SceneDocument:
    """Сцена вместе с путём файла и состоянием вкладки.

    Attributes:
        scene: Сама сцена.
        path: Файл сцены или ``None`` для ещё не сохранённой.
        dirty: Есть несохранённые изменения.
        camera_state: Вид камеры редактора, запомненный при уходе с вкладки.
        sky: Настройки неба сцены (секция ``sky`` файла).
        post: Настройки пост-обработки (секция ``post``; ``None`` — секции нет,
            действуют умолчания профиля, и файл её не получит).
        commands: Стек команд undo/redo, привязанный к ``scene``.
        bound: Контекст редактора уже подписался на стек.
    """

    scene: Scene
    path: Path | None = None
    dirty: bool = False
    camera_state: object | None = None
    sky: SkySettings = field(default_factory=SkySettings)
    post: PostSettings | None = None
    commands: CommandStack = field(init=False, repr=False)
    bound: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.commands = CommandStack(self.scene)
        self.commands.mark_clean()

    def display_name(self, untitled: str) -> str:
        """Короткое имя для вкладки без служебного суффикса.

        Args:
            untitled: Имя для несохранённой сцены.

        Returns:
            ``level1`` для ``level1.scene.json``; иначе ``untitled``.
        """
        if self.path is None:
            return untitled
        name = self.path.name
        if name.lower().endswith(SCENE_SUFFIX):
            return name[: -len(SCENE_SUFFIX)]
        return self.path.stem
