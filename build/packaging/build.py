"""Сборка exe игры: пак + иконка + версия + PyInstaller (A13b).

Запуск (из корня репозитория, в venv с ``pip install -e .[package]``)::

    python build/packaging/build.py --project projects/PlayModeTest --mode onedir

Результат: ``<out>/<имя>/<имя>.exe`` + ``data.pak`` (onedir) либо один
``<out>/<имя>.exe`` с вшитым паком (onefile). Промежуточные файлы - в
``build/artifacts/<имя>/`` (git-ignored).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SPEC = HERE / "myengine_player.spec"
ICON_SIZES = [(16, 16), (32, 32), (48, 48), (256, 256)]
DEFAULT_ICON_SVG = ROOT / "assets" / "icons" / "cube.svg"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы."""
    parser = argparse.ArgumentParser(description="Сборка exe игры MyEngine")
    parser.add_argument("--project", required=True, help="каталог проекта")
    parser.add_argument("--mode", choices=("onedir", "onefile"), default="onedir")
    parser.add_argument("--out", default=None, help="куда класть результат")
    parser.add_argument("--name", default=None, help="имя игры (по умолчанию проект)")
    parser.add_argument(
        "--quality", choices=("low", "medium", "high"), default="medium"
    )
    parser.add_argument("--icon", default=None, help="PNG/ICO иконка игры")
    parser.add_argument("--pak", default=None, help="готовый data.pak (без пересборки)")
    parser.add_argument("--console", action="store_true", help="оставить консоль")
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument("--clean", action="store_true", help="очистить кэш PyInstaller")
    return parser.parse_args(argv)


def make_icon(target: Path, source: Path | None) -> Path:
    """Собрать ``icon.ico`` (16/32/48/256); результат кэшируется по источнику."""
    from PIL import Image, ImageDraw

    src = source or DEFAULT_ICON_SVG
    if target.is_file() and target.stat().st_mtime >= src.stat().st_mtime:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".ico":
        shutil.copyfile(src, target)
        return target
    if src.suffix.lower() == ".svg":
        glyph = _render_svg(src, 256)
        base = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        ImageDraw.Draw(base).rounded_rectangle(
            (8, 8, 248, 248), radius=48, fill=(34, 40, 52, 255)
        )
        glyph = glyph.resize((176, 176), Image.LANCZOS)
        base.alpha_composite(glyph, (40, 40))
        image = base
    else:
        image = Image.open(src).convert("RGBA").resize((256, 256), Image.LANCZOS)
    image.save(target, format="ICO", sizes=ICON_SIZES)
    return target


def _render_svg(path: Path, size: int):
    """Растеризовать Lucide-SVG белым цветом (Qt SVG; только на машине сборки)."""
    import io

    from PIL import Image
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    text = path.read_text(encoding="utf-8").replace("currentColor", "#ffffff")
    renderer = QSvgRenderer(QByteArray(text.encode("utf-8")))
    canvas = QImage(size, size, QImage.Format_ARGB32)
    canvas.fill(0)
    painter = QPainter(canvas)
    renderer.render(painter)
    painter.end()
    buffer = QBuffer()
    buffer.open(QIODevice.WriteOnly)
    canvas.save(buffer, "PNG")
    return Image.open(io.BytesIO(bytes(buffer.data()))).convert("RGBA")


def version_tuple(text: str) -> tuple[int, int, int, int]:
    """``"1.2.3"`` -> ``(1, 2, 3, 0)``."""
    parts = [int("".join(c for c in p if c.isdigit()) or 0) for p in text.split(".")]
    a, b, c, d = ([*parts, 0, 0, 0, 0])[:4]
    return a, b, c, d


def write_version_file(target: Path, name: str, version: str, company: str) -> Path:
    """Ресурс версии Windows (``version_info.txt``) для PyInstaller."""
    vt = version_tuple(version)
    text = f"""# UTF-8
VSVersionInfo(
  ffi=FixedFileInfo(filevers={vt}, prodvers={vt}, mask=0x3f, flags=0x0, OS=0x40004,
    fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', {company!r}),
      StringStruct('FileDescription', {name!r}),
      StringStruct('FileVersion', {version!r}),
      StringStruct('InternalName', {name!r}),
      StringStruct('OriginalFilename', {name + '.exe'!r}),
      StringStruct('ProductName', {name!r}),
      StringStruct('ProductVersion', {version!r})])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _mb(value: int) -> str:
    return f"{value / 1024 / 1024:.1f} МБ"


def main(argv: list[str] | None = None) -> int:
    """Собрать игру; код возврата процесса."""
    args = parse_args(argv)
    project = Path(args.project).resolve()
    name = args.name or project.name
    out = Path(args.out).resolve() if args.out else project / "Build"
    work = ROOT / "build" / "artifacts" / name
    work.mkdir(parents=True, exist_ok=True)

    if args.pak:
        pak = Path(args.pak).resolve()
    else:
        from engine.cli.package import build_pak

        result = build_pak(
            project, work, quality=args.quality, name=name,
            allow_errors=args.allow_errors,
        )
        pak = result.pak
    with_pak = json.loads(_read_build_json(pak) or "{}")
    version = str(with_pak.get("version", "1.0.0"))
    icon = make_icon(work / "icon.ico", Path(args.icon) if args.icon else None)
    version_file = write_version_file(
        work / "version_info.txt", name, version, company=name
    )

    env = dict(os.environ)
    env.update(
        MYENGINE_PKG_NAME=name,
        MYENGINE_PKG_MODE=args.mode,
        MYENGINE_PKG_PAK=str(pak),
        MYENGINE_PKG_ICON=str(icon),
        MYENGINE_PKG_VERSION_FILE=str(version_file),
        MYENGINE_PKG_CONSOLE="1" if args.console else "0",
    )
    dist = out / args.mode
    command = [
        sys.executable, "-m", "PyInstaller", str(SPEC), "--noconfirm",
        "--distpath", str(dist), "--workpath", str(work / "pyi"),
    ]
    if args.clean:
        command.append("--clean")
    print("PyInstaller:", " ".join(command))
    code = subprocess.run(command, env=env, cwd=ROOT).returncode
    if code != 0:
        return code

    if args.mode == "onedir":
        target = dist / name
        shutil.copyfile(pak, target / "data.pak")
        exe = target / f"{name}.exe"
        print(f"Готово: {exe}")
        print(
            f"  onedir {_mb(_size(target))}; exe {_mb(_size(exe))}; "
            f"data.pak {_mb(_size(pak))}; _internal {_mb(_size(target / '_internal'))}"
        )
    else:
        exe = dist / f"{name}.exe"
        print(f"Готово: {exe}  ({_mb(_size(exe))}, data.pak {_mb(_size(pak))} внутри)")
    return 0


def _read_build_json(pak: Path) -> str | None:
    from engine.core.resources.pak import PakReader

    with PakReader(pak) as reader:
        data = reader.read_file("build.json")
    return data.decode("utf-8") if data else None


if __name__ == "__main__":
    sys.exit(main())
