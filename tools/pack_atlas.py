"""Упаковка PNG-спрайтов в атлас (PNG + JSON с кадрами) без запуска редактора.

Использует тот же cooker, что и импортёр ``*.atlas.json``, поэтому раскладка
совпадает. Удобно, чтобы посмотреть атлас глазами или отдать его вне проекта.

Запуск::

    .venv\\Scripts\\python.exe tools\\pack_atlas.py sprites\\ -o _work\\ui
    .venv\\Scripts\\python.exe tools\\pack_atlas.py a.png b.png -o _work\\ui --trim

Результат: ``<out>.png`` и ``<out>.json`` (формат ``atlas.json`` импортёра).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from engine.core.resources import AssetImportError
from engine.core.resources.cookers import (
    Sprite,
    decode_image,
    encode_png,
    pack_sprites,
    to_rgba,
)


def collect(inputs: list[Path]) -> list[Path]:
    """Развернуть каталоги в отсортированные списки PNG."""
    files: list[Path] = []
    for item in inputs:
        if item.is_dir():
            files += sorted(item.glob("*.png"), key=lambda path: path.name.lower())
        else:
            files.append(item)
    return list(dict.fromkeys(files))


def main(argv: list[str] | None = None) -> int:
    """Точка входа; возвращает код выхода."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("inputs", nargs="+", type=Path, help="PNG-файлы или каталоги")
    parser.add_argument("-o", "--out", type=Path, required=True, help="база имён")
    parser.add_argument("--padding", type=int, default=2, help="зазор между спрайтами")
    parser.add_argument("--max-size", type=int, default=2048, help="предел стороны")
    parser.add_argument("--trim", action="store_true", help="обрезать прозрачные поля")
    parser.add_argument(
        "--no-pot", action="store_true", help="не округлять высоту до степени двойки"
    )
    args = parser.parse_args(argv)

    try:
        files = collect(args.inputs)
        if not files:
            raise AssetImportError("Не найдено ни одного PNG")
        sprites = [Sprite(path.stem, to_rgba(decode_image(path))) for path in files]
        built = pack_sprites(
            sprites,
            padding=args.padding,
            max_size=args.max_size,
            power_of_two=not args.no_pot,
            trim=args.trim,
        )
    except AssetImportError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".png").write_bytes(encode_png(built.pixels))
    args.out.with_suffix(".json").write_text(
        json.dumps(built.document(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Спрайтов: {len(built.frames)}, атлас {built.width}x{built.height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
