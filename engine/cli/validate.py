"""Базовая проверка проекта перед сборкой (A13a п.7).

Ищет битые ссылки на ассеты в сценах (``{"guid": ...}``), отсутствующие
скрипты и ассеты с ошибкой импорта. Ничего не чинит и не пишет на диск.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Issue", "ValidationReport", "validate_project"]

_GUID = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True)
class Issue:
    """Одно замечание: ``level`` — ``error`` или ``warning``."""

    level: str
    where: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.where}: {self.message}"


@dataclass
class ValidationReport:
    """Итог проверки."""

    issues: list[Issue] = field(default_factory=list)
    scenes: int = 0

    @property
    def errors(self) -> list[Issue]:
        """Только ошибки (сборку блокируют)."""
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        """Только предупреждения."""
        return [i for i in self.issues if i.level == "warning"]


def _walk(node: Any, trail: str = "") -> Any:
    """Обойти JSON: ``(путь, словарь)`` для каждого словаря."""
    if isinstance(node, dict):
        yield trail, node
        for key, value in node.items():
            yield from _walk(value, f"{trail}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, f"{trail}[{index}]")


def validate_project(
    project: Path,
    db: Any,
    scene_files: list[Path],
    script_files: set[str],
    import_errors: dict[str, str] | None = None,
) -> ValidationReport:
    """Проверить сцены проекта.

    Args:
        project: Корень проекта.
        db: ``AssetDatabase`` (поддерживает ``guid in db``, ``db.get(guid)``).
        scene_files: Файлы ``*.scene.json``.
        script_files: Ключи скриптов, попадающих в пак (``Assets/Scripts/a.py``).
        import_errors: GUID → текст ошибки импорта (``ImportManager.errors()``).
    """
    report = ValidationReport(scenes=len(scene_files))
    for guid, text in (import_errors or {}).items():
        handle = db.get(guid)
        name = handle.path.name if handle is not None else guid
        report.issues.append(Issue("error", name, f"ошибка импорта: {text}"))
    for scene in scene_files:
        rel = scene.relative_to(project).as_posix()
        try:
            data = json.loads(scene.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as error:
            report.issues.append(Issue("error", rel, f"сцена не читается: {error}"))
            continue
        for trail, node in _walk(data):
            guid = node.get("guid")
            if isinstance(guid, str) and _GUID.match(guid) and guid not in db:
                report.issues.append(
                    Issue("error", f"{rel}{trail}", f"битая ссылка на ассет {guid}")
                )
            script = node.get("path")
            if (
                isinstance(script, str)
                and script.endswith(".py")
                and "guid" not in node
            ):
                key = script.replace("\\", "/")
                if key not in script_files:
                    report.issues.append(
                        Issue("error", f"{rel}{trail}", f"нет скрипта '{script}'")
                    )
    return report
