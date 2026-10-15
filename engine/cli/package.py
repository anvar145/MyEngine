"""Сборка ``data.pak`` проекта (A13a п.6): ассеты из ``Library/`` + ``files``.

В пак попадают импортированные ассеты (секция ``assets``) и файлы, которые
не являются ассетами (секция ``files``): сцены, скрипты, ``Settings/*.json``,
``*.project.json`` и ``build.json`` с параметрами сборки. После этого
``python -m engine.player --project <пусто> --pak data.pak`` работает без
каталога ``Assets/``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from engine import __version__
from engine.cli.validate import ValidationReport, validate_project
from engine.rendering.dynamic_resolution import read_hdr_hint, read_render_hints

__all__ = ["PackageError", "PackageResult", "build_pak", "collect_files"]

_log = logging.getLogger("engine.cli")

#: Имя пака собранной игры.
PAK_NAME = "data.pak"


class PackageError(RuntimeError):
    """Проект не собирается (нет ассетов, ошибки валидации и т.п.)."""


@dataclass
class PackageResult:
    """Итог сборки пака."""

    pak: Path
    assets: int
    files: int
    size: int
    report: ValidationReport
    by_type: dict[str, int] = field(default_factory=dict)


def _assets_dir(project: Path) -> Path:
    for name in ("Assets", "assets"):
        if (project / name).is_dir():
            return project / name
    raise PackageError(f"В проекте '{project}' нет каталога Assets/")


def collect_files(project: Path, assets_dir: Path) -> dict[str, Path]:
    """Файлы секции ``files``: ключ в паке → файл на диске.

    Сцены ``**/*.scene.json`` и скрипты ``**/Scripts/**/*.py`` под ассетами,
    ``Settings/*.json`` и ``*.project.json`` в корне проекта.
    """
    found: dict[str, Path] = {}
    sources = [
        *assets_dir.rglob("*.scene.json"),
        *(
            p
            for p in assets_dir.rglob("*.py")
            if "Scripts" in p.relative_to(assets_dir).parts[:-1]
        ),
        *(project / "Settings").glob("*.json"),
        *project.glob("*.project.json"),
    ]
    for path in sources:
        if path.is_file():
            found[path.relative_to(project).as_posix()] = path
    return dict(sorted(found.items()))


def _project_version(project: Path) -> str:
    """Версия игры из ``*.project.json`` (поле ``version``), иначе ``1.0.0``."""
    for file in sorted(project.glob("*.project.json")):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and isinstance(data.get("version"), str):
            return data["version"]
    return "1.0.0"


def build_pak(
    project: Path,
    out_dir: Path,
    *,
    quality: str = "medium",
    name: str | None = None,
    allow_errors: bool = False,
) -> PackageResult:
    """Собрать ``<out_dir>/data.pak``.

    Args:
        project: Корень проекта.
        out_dir: Каталог результата (создаётся).
        quality: Тир качества по умолчанию (пишется в ``build.json``).
        name: Имя игры (по умолчанию из ``*.project.json`` или имя каталога).
        allow_errors: Собирать, несмотря на ошибки валидации.

    Raises:
        PackageError: Нет ``Assets/`` или проверка нашла ошибки.
    """
    from engine.core.resources import AssetDatabase, ImportManager
    from engine.core.resources.pak import PakBuilder

    project = Path(project).resolve()
    assets_dir = _assets_dir(project)
    files = collect_files(project, assets_dir)
    database = AssetDatabase(assets_dir, library=project / "Library")
    manager = ImportManager(database)
    try:
        manager.open()
        manager.wait_idle(600)
        report = validate_project(
            project,
            database,
            [p for k, p in files.items() if k.endswith(".scene.json")],
            {k for k in files if k.endswith(".py")},
            manager.failures(),
        )
        for issue in report.issues:
            _log.warning("%s", issue)
        if report.errors and not allow_errors:
            raise PackageError(
                f"Проверка проекта нашла {len(report.errors)} ошибок: "
                + "; ".join(str(i) for i in report.errors[:5])
            )
        builder = PakBuilder(database).add_all()
        for key, path in files.items():
            builder.add_file_from_disk(path, key)
        _hint, dynamic, _hdr = read_render_hints(project)
        hdr = read_hdr_hint(project)
        build_info = {
            "engine_version": __version__,
            "quality": quality,
            "name": name or project.name,
            "version": _project_version(project),
        }
        if hdr is not None:  # нет поля - решает сцена (секция post)
            build_info["hdr"] = hdr
        if dynamic is not None:
            build_info["dynamic_resolution"] = dynamic
        builder.add_file("build.json", json.dumps(build_info).encode("utf-8"))
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pak = builder.build(out_dir / PAK_NAME)
        by_type: dict[str, int] = {}
        for guid in database.guids:
            handle = database.get(guid)
            if handle is not None:
                kind = handle.path.suffix.lower() or "?"
                by_type[kind] = by_type.get(kind, 0) + 1
        return PackageResult(
            pak=pak,
            assets=len(builder),
            files=len(files) + 1,
            size=pak.stat().st_size,
            report=report,
            by_type=dict(sorted(by_type.items())),
        )
    finally:
        manager.close()
        database.close()
