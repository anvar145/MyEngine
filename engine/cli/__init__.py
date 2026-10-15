"""Командная строка движка: ``myengine package`` (A13)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="myengine", description="MyEngine CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("package", help="собрать игру из проекта")
    pack.add_argument("--project", required=True, help="каталог проекта")
    pack.add_argument("--target", default="windows-x64", choices=["windows-x64"])
    pack.add_argument(
        "--mode",
        default="onedir",
        choices=["onedir", "onefile"],
        help="onedir (основной формат) или onefile; onefile требует прав на "
        "запись в системный TEMP, на ограниченных учётках (не-админ, "
        "Program Files) используйте onedir",
    )
    pack.add_argument("--quality", default="medium", choices=["low", "medium", "high"])
    pack.add_argument("--out", default=None, help="каталог результата")
    pack.add_argument("--icon", default=None, help="иконка .ico")
    pack.add_argument("--name", default=None, help="имя игры")
    pack.add_argument(
        "--pak-only",
        action="store_true",
        help="собрать только data.pak (без exe)",
    )
    pack.add_argument(
        "--allow-errors", action="store_true", help="не останавливаться на ошибках"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Точка входа ``myengine``.

    Returns:
        Код возврата процесса.
    """
    args = _build_parser().parse_args(argv)
    from engine.cli.package import PackageError, build_pak

    project = Path(args.project).resolve()
    out = Path(args.out) if args.out else project / "Build"
    try:
        result = build_pak(
            project,
            out,
            quality=args.quality,
            name=args.name,
            allow_errors=args.allow_errors,
        )
    except PackageError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    print(
        f"data.pak: {result.pak} ({result.size / 1048576:.1f} МБ, "
        f"ассетов {result.assets}, файлов {result.files}, "
        f"предупреждений {len(result.report.warnings)})"
    )
    if not args.pak_only:
        print(
            "Сборка exe (PyInstaller) — этап A13b; пока доступен только --pak-only.",
            file=sys.stderr,
        )
    return 0
