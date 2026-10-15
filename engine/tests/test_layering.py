"""Тест слоёв: ядро не должно зависеть от GUI и графических библиотек.

Правило зависимостей из архитектуры (§3): `engine.core` — самый нижний слой,
он обязан работать в headless-режиме, поэтому импорты PySide6, OpenGL, pygame
и оконных библиотек в нём запрещены. Тест разбирает исходники по AST и валит
сборку на первом же нарушении.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

CORE_ROOT = Path(__file__).resolve().parents[1] / "core"

FORBIDDEN_ROOTS = {
    "PySide6",
    "PyQt5",
    "PyQt6",
    "OpenGL",
    "OpenGL_accelerate",
    "glfw",
    "pygame",
    "sdl2",
    "moderngl",
}


def iter_core_modules() -> list[Path]:
    """Собрать все модули ядра.

    Returns:
        Отсортированный список путей к файлам ``.py``.
    """
    return sorted(CORE_ROOT.rglob("*.py"))


def imported_roots(path: Path) -> set[str]:
    """Собрать корневые имена всех импортируемых модулей в файле.

    Args:
        path: Путь к файлу Python.

    Returns:
        Множество корневых имён (например, ``{"numpy", "engine"}``).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_core_modules_are_discovered() -> None:
    """Ядро должно состоять из нескольких модулей (защита от ошибки в пути)."""
    modules = iter_core_modules()
    assert len(modules) >= 10, f"найдено всего {len(modules)} модулей в {CORE_ROOT}"


@pytest.mark.parametrize("module_path", iter_core_modules(), ids=lambda path: path.name)
def test_core_does_not_import_gui_or_graphics(module_path: Path) -> None:
    """Ни один модуль ядра не импортирует GUI/графические библиотеки."""
    forbidden = imported_roots(module_path) & FORBIDDEN_ROOTS
    relative = module_path.relative_to(CORE_ROOT.parent.parent)
    assert not forbidden, (
        f"{relative} импортирует {sorted(forbidden)}: "
        "ядро обязано оставаться независимым от GUI и OpenGL"
    )


@pytest.mark.parametrize("module_path", iter_core_modules(), ids=lambda path: path.name)
def test_core_does_not_import_editor(module_path: Path) -> None:
    """Ядро не должно зависеть от пакета редактора."""
    source = module_path.read_text(encoding="utf-8")
    assert "myengine_editor" not in source, (
        f"{module_path.name} ссылается на редактор: зависимость направлена только вниз"
    )


PLAYER_ROOT = CORE_ROOT.parent / "player"


@pytest.mark.parametrize(
    "module_path", sorted(PLAYER_ROOT.rglob("*.py")), ids=lambda path: path.name
)
def test_player_does_not_import_editor_or_qt(module_path: Path) -> None:
    """Плеер живёт без редактора и Qt (offscreen-контекст берётся из rendering)."""
    forbidden = imported_roots(module_path) & {"PySide6", "PyQt5", "PyQt6", "editor"}
    assert not forbidden, (
        f"engine/player/{module_path.name} импортирует {sorted(forbidden)}: "
        "плееру Qt и редактор не нужны"
    )


PHYSICS_ROOT = CORE_ROOT.parent / "physics"


@pytest.mark.parametrize(
    "module_path", sorted(PHYSICS_ROOT.rglob("*.py")), ids=lambda path: path.name
)
def test_physics_does_not_import_gui_gl_or_rendering(module_path: Path) -> None:
    """Физика — чистые данные и numpy: без Qt, GL, рендера и редактора."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    bad = {
        name
        for name in modules
        if name.split(".")[0] in FORBIDDEN_ROOTS | {"editor"}
        or name.startswith(("engine.rendering", "engine.player", "engine.scripting"))
    }
    assert not bad, f"engine/physics/{module_path.name} импортирует {sorted(bad)}"
