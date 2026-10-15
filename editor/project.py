"""Система проектов: каталог ``projects/`` и структура нового проекта.

Проект — это обычная папка с предсказуемым набором подкаталогов. Всё, что
создаёт пользователь (скрипты, модели, спрайты, звуки), лежит внутри проекта,
поэтому редактору достаточно одного пути — корня проекта.

Example:
    >>> project = create_project(Path("."), "MyGame")
    >>> (project / "Assets" / "Scripts").is_dir()
    True
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

__all__ = [
    "ASSETS_DIRNAME",
    "LIBRARY_DIRNAME",
    "PROJECTS_DIRNAME",
    "PROJECT_FOLDERS",
    "PREFABS_DIRNAME",
    "PROJECT_TEMPLATES",
    "SCENES_DIRNAME",
    "SCRIPTS_DIRNAME",
    "SETTINGS_DIRNAME",
    "TEMPLATES_DIR",
    "assets_root_for",
    "create_project",
    "ensure_projects_dir",
    "list_projects",
    "prefabs_root_for",
    "projects_root",
    "scenes_root_for",
    "scripts_root_for",
    "validate_project_name",
]

_log = logging.getLogger("editor.project")

#: Имя каталога со всеми проектами (в корне репозитория/рабочего каталога).
PROJECTS_DIRNAME = "projects"

#: Каталог кэша импорта внутри проекта (не коммитится).
LIBRARY_DIRNAME = "Library"

#: Каталог исходных ассетов внутри проекта.
ASSETS_DIRNAME = "Assets"

#: Подкаталоги ассетов (внутри :data:`ASSETS_DIRNAME`).
SCRIPTS_DIRNAME = "Scripts"
SCENES_DIRNAME = "Scenes"
PREFABS_DIRNAME = "Prefabs"

#: Каталог настроек проекта.
SETTINGS_DIRNAME = "Settings"

#: Подкаталоги ассетов, которые создаются в новом проекте.
PROJECT_FOLDERS: tuple[str, ...] = (
    "Audio",
    "Materials",
    "Models",
    SCENES_DIRNAME,
    SCRIPTS_DIRNAME,
    "Shaders",
    "Sprites",
    "Textures",
)

#: Шаблоны нового проекта: имя совпадает с папкой в ``assets/templates``.
PROJECT_TEMPLATES: tuple[str, ...] = ("empty", "3d", "2d")

#: Каталог готовых шаблонов проекта.
TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "assets" / "templates"

#: Символы, недопустимые в имени каталога на Windows, плюс управляющие.
_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def projects_root(base: Path | str) -> Path:
    """Каталог со всеми проектами (``base/projects``).

    Args:
        base: Корень, внутри которого лежит папка проектов.

    Returns:
        Путь к каталогу ``projects`` (может ещё не существовать).
    """
    return Path(base) / PROJECTS_DIRNAME


def assets_root_for(project_root: Path | str) -> Path:
    """Каталог исходных ассетов проекта (``<проект>/Assets``).

    Плоская раскладка (ассеты прямо в корне проекта) не поддерживается.

    Args:
        project_root: Корень проекта.

    Returns:
        Каталог, который показывает панель ассетов и сканирует импорт
        (может ещё не существовать).
    """
    return Path(project_root) / ASSETS_DIRNAME


def scripts_root_for(project_root: Path | str) -> Path:
    """Каталог скриптов проекта (``<проект>/Assets/Scripts``)."""
    return assets_root_for(project_root) / SCRIPTS_DIRNAME


def scenes_root_for(project_root: Path | str) -> Path:
    """Каталог сцен проекта (``<проект>/Assets/Scenes``)."""
    return assets_root_for(project_root) / SCENES_DIRNAME


def prefabs_root_for(project_root: Path | str) -> Path:
    """Каталог префабов проекта (``<проект>/Assets/Prefabs``)."""
    return assets_root_for(project_root) / PREFABS_DIRNAME


def ensure_projects_dir(base: Path | str) -> Path:
    """Создать папку проектов, если её нет.

    Args:
        base: Корень, внутри которого лежит папка проектов.

    Returns:
        Существующий путь к каталогу ``projects``.
    """
    root = projects_root(base)
    root.mkdir(parents=True, exist_ok=True)
    return root


def validate_project_name(name: str) -> str:
    """Проверить имя проекта и вернуть его без лишних пробелов.

    Args:
        name: Предложенное имя проекта.

    Returns:
        Очищенное имя.

    Raises:
        ValueError: Если имя пустое, состоит из точек или содержит символы,
            недопустимые в имени каталога.
    """
    cleaned = str(name).strip()
    if not cleaned:
        raise ValueError("имя проекта пустое")
    if cleaned in {".", ".."} or set(cleaned) == {"."}:
        raise ValueError(f"недопустимое имя проекта: '{name}'")
    if _INVALID_CHARS.search(cleaned):
        raise ValueError(
            f"в имени проекта '{name}' есть недопустимые символы: < > : \" / \\ | ? *"
        )
    if cleaned.endswith("."):
        raise ValueError(f"имя проекта '{name}' не может заканчиваться точкой")
    return cleaned


def create_project(
    base: Path | str,
    name: str,
    *,
    exists_ok: bool = False,
    template: str = "empty",
) -> Path:
    """Создать проект со стандартным набором подкаталогов.

    Args:
        base: Корень, внутри которого лежит папка проектов.
        name: Имя проекта (становится именем каталога).
        exists_ok: Разрешить использовать существующий каталог.
        template: Шаблон из :data:`PROJECT_TEMPLATES`; файлы шаблона
            копируются в проект, существующие файлы не перезаписываются.

    Returns:
        Путь к созданному проекту.

    Raises:
        ValueError: Если имя проекта или шаблон некорректны.
        FileExistsError: Если такой проект уже есть, а ``exists_ok`` — ``False``.
        OSError: Если каталоги не удалось создать.
    """
    if template not in PROJECT_TEMPLATES:
        raise ValueError(f"неизвестный шаблон проекта: '{template}'")
    project = projects_root(base) / validate_project_name(name)
    if project.exists() and not exists_ok:
        raise FileExistsError(f"проект уже существует: {project}")

    for folder in PROJECT_FOLDERS:
        (project / ASSETS_DIRNAME / folder).mkdir(parents=True, exist_ok=True)
    for folder in (LIBRARY_DIRNAME, SETTINGS_DIRNAME):
        (project / folder).mkdir(parents=True, exist_ok=True)
    source = TEMPLATES_DIR / template
    if source.is_dir():
        shutil.copytree(source, project, dirs_exist_ok=True, copy_function=_copy_new)
    _write_project_file(project)
    _log.info("Проект '%s' готов: %s", project.name, project)
    return project


def _write_project_file(project: Path) -> None:
    """Записать ``<Имя>.project.json`` (если его ещё нет).

    В ``scene`` попадает первая сцена из ``Assets/Scenes``, если она есть.
    Новые проекты получают поля рендера A10a: HDR с пост-обработкой и
    динамическое разрешение (``render_scale_hint`` не задан — берётся профиль
    качества). Существующие проекты файл не перезаписывается.
    """
    path = project / f"{project.name}.project.json"
    if path.exists():
        return
    data: dict[str, Any] = {
        "name": project.name,
        "hdr": True,
        "dynamic_resolution": True,
    }
    scenes = sorted(scenes_root_for(project).glob("*.scene.json"))
    if scenes:
        data["scene"] = scenes[0].relative_to(project).as_posix()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", "utf-8")


def _copy_new(src: str, dst: str) -> None:
    """Копировать файл шаблона, не затирая уже существующий."""
    if not Path(dst).exists():
        shutil.copy2(src, dst)


def list_projects(base: Path | str) -> tuple[Path, ...]:
    """Перечислить существующие проекты.

    Args:
        base: Корень, внутри которого лежит папка проектов.

    Returns:
        Отсортированные пути к каталогам проектов (пустой кортеж, если папки
        проектов ещё нет).
    """
    root = projects_root(base)
    if not root.is_dir():
        return ()
    return tuple(sorted(path for path in root.iterdir() if path.is_dir()))
