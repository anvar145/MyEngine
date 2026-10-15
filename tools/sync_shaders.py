"""Синхронизация канонических GLSL-исходников движка с каталогом ассетов.

Освещение живёт в двух местах по необходимости: канонический исходник лежит
рядом с движком (``engine/rendering/shaders/``), а конвейер ресурсов читает
шейдеры только из ``assets/shaders/``. Чтобы копии не разъехались, они
синхронизируются этим скриптом, а совпадение проверяет тест
``test_asset_copy_matches_engine_source``.

Запуск::

    .venv\\Scripts\\python.exe tools\\sync_shaders.py           # скопировать
    .venv\\Scripts\\python.exe tools\\sync_shaders.py --check    # только проверить
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Пары «канонический исходник → копия в ассетах».
SHADER_PAIRS: tuple[tuple[Path, Path], ...] = (
    (
        ROOT / "engine" / "rendering" / "shaders" / "lighting.frag",
        ROOT / "assets" / "shaders" / "lighting.frag",
    ),
)


def sync(*, check_only: bool = False) -> int:
    """Скопировать исходники в ассеты (или только проверить совпадение).

    Args:
        check_only: Не писать файлы, а только доложить о расхождениях.

    Returns:
        Число расхождений (0 — всё синхронно).
    """
    differences = 0
    for source, target in SHADER_PAIRS:
        relative_source = source.relative_to(ROOT)
        relative_target = target.relative_to(ROOT)
        if not source.is_file():
            print(f"НЕТ ИСХОДНИКА: {relative_source}")
            differences += 1
            continue
        if check_only:
            if not target.is_file() or not filecmp.cmp(source, target, shallow=False):
                print(f"РАСХОЖДЕНИЕ: {relative_target} != {relative_source}")
                differences += 1
            else:
                print(f"ок: {relative_target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        print(f"скопировано: {relative_source} → {relative_target}")
    return differences


def main(argv: list[str] | None = None) -> int:
    """Разобрать аргументы и выполнить синхронизацию."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="только проверить совпадение, ничего не записывать",
    )
    args = parser.parse_args(argv)
    differences = sync(check_only=args.check)
    if args.check and differences:
        print(f"\nРасхождений: {differences}. Запустите без --check.")
    return 1 if (args.check and differences) else 0


if __name__ == "__main__":
    sys.exit(main())
