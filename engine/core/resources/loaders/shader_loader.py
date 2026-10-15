"""Загрузчик GLSL-шейдеров с разрешением ``#include``.

Шейдеры удобно держать модульными (общие функции в отдельных файлах), поэтому
загрузчик сам подставляет включения и возвращает **один готовый исходник**
одной стадии. Стадия определяется по расширению (``.vert``, ``.frag``,
``.geom``, ``.comp``) или директивой ``#stage <имя>`` в начале файла.

Поддерживаются директивы:

* ``#include "path"`` — включить файл (поиск: рядом с файлом, затем в
  ``include_dirs`` из параметров импорта);
* ``#pragma once`` — включать файл только один раз.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from engine.core.resources.asset import ShaderAsset, ShaderStage
from engine.core.resources.errors import ResourceFormatError
from engine.core.resources.loader import AssetLoader, LoadContext

__all__ = ["GlslShaderLoader", "detect_stage", "resolve_includes"]

_log = logging.getLogger("engine.core.resources.shader")

_INCLUDE_PATTERN = re.compile(r'^\s*#\s*include\s+"([^"]+)"\s*$')
_STAGE_PATTERN = re.compile(
    r"^\s*#\s*(?:pragma\s+)?stage\s+(\w+)", re.IGNORECASE | re.MULTILINE
)
_PRAGMA_ONCE = re.compile(r"^\s*#\s*pragma\s+once\s*$", re.IGNORECASE | re.MULTILINE)

_STAGE_BY_EXTENSION = {
    ".vert": ShaderStage.VERTEX,
    ".frag": ShaderStage.FRAGMENT,
    ".geom": ShaderStage.GEOMETRY,
    ".comp": ShaderStage.COMPUTE,
}
_STAGE_BY_NAME = {
    "vertex": ShaderStage.VERTEX,
    "fragment": ShaderStage.FRAGMENT,
    "pixel": ShaderStage.FRAGMENT,
    "geometry": ShaderStage.GEOMETRY,
    "compute": ShaderStage.COMPUTE,
}

MAX_INCLUDE_DEPTH = 32


class GlslShaderLoader(AssetLoader):
    """Загрузчик GLSL: ``.glsl``, ``.vert``, ``.frag``, ``.geom``, ``.comp``."""

    extensions = (".glsl", ".vert", ".frag", ".geom", ".comp")

    def load(self, context: LoadContext) -> ShaderAsset:
        """Прочитать шейдер и разрешить все включения.

        Поддерживаемые параметры импорта:
            include_dirs (iterable[Path]): Дополнительные каталоги поиска.
            defines (mapping[str, str]): Подстановки в начало файла.
        """
        include_dirs = [Path(item) for item in context.option("include_dirs", ()) or ()]
        try:
            source = context.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ResourceFormatError(
                f"Не удалось прочитать шейдер '{context.path}': {exc}"
            ) from exc

        source, includes = resolve_includes(
            context.path, source, include_dirs=include_dirs
        )
        stage = detect_stage(context.path, source)

        defines = context.option("defines", None)
        if defines:
            header = "\n".join(
                f"#define {name} {value}".rstrip()
                for name, value in dict(defines).items()
            )
            source = f"{header}\n{source}"

        asset = ShaderAsset(context.path, stage, source, tuple(includes))
        _log.debug(
            "Шейдер загружен: %s (стадия=%s, включений=%d)",
            context.path.name,
            stage.value,
            len(includes),
        )
        return asset


def resolve_includes(
    path: Path,
    source: str,
    *,
    include_dirs: list[Path] | None = None,
    _seen: set[Path] | None = None,
    _stack: tuple[Path, ...] = (),
    _once_cache: dict[Path, bool] | None = None,
) -> tuple[str, list[str]]:
    """Подставить содержимое ``#include`` в исходник шейдера.

    Args:
        path: Путь к текущему файлу.
        source: Его исходный текст.
        include_dirs: Дополнительные каталоги поиска включений.
        _seen: Служебное множество уже включённых файлов.
        _stack: Служебный стек включений (защита от циклов).
        _once_cache: Служебный кэш «файл объявлен как ``#pragma once``».

    Returns:
        Кортеж из итогового текста и списка путей включённых файлов.

    Raises:
        ResourceFormatError: Если включаемый файл не найден или обнаружен цикл.
    """
    search_dirs = [path.parent, *(include_dirs or [])]
    seen = _seen if _seen is not None else set()
    once_cache = _once_cache if _once_cache is not None else {}
    includes: list[str] = []
    stack = (*_stack, path)
    output_lines: list[str] = []

    for line in source.splitlines():
        if _PRAGMA_ONCE.match(line):
            # Директиву обрабатывает загрузчик; в GLSL её передавать не нужно.
            output_lines.append("// #pragma once (обработано загрузчиком)")
            continue
        match = _INCLUDE_PATTERN.match(line)
        if match is None:
            output_lines.append(line)
            continue

        target = _find_include(match.group(1), search_dirs, path)
        if target in stack:
            raise ResourceFormatError(
                f"Циклическое включение '{target.name}' в '{path.name}'"
            )
        if len(stack) > MAX_INCLUDE_DEPTH:
            raise ResourceFormatError(
                f"Слишком глубокая вложенность include в '{path.name}'"
            )
        if target in seen and once_cache.get(target, False):
            output_lines.append(f"// include пропущен (#pragma once): {target.name}")
            continue

        try:
            nested_source = target.read_text(encoding="utf-8")
        except OSError as exc:
            raise ResourceFormatError(
                f"Не удалось прочитать включение '{target}': {exc}"
            ) from exc
        once_cache[target] = bool(_PRAGMA_ONCE.search(nested_source))
        nested_text, nested_includes = resolve_includes(
            target,
            nested_source,
            include_dirs=include_dirs,
            _seen=seen,
            _stack=stack,
            _once_cache=once_cache,
        )
        seen.add(target)
        includes.append(str(target))
        includes.extend(nested_includes)
        output_lines.append(f"// >>> начало включения: {target.name}")
        output_lines.extend(nested_text.splitlines())
        output_lines.append(f"// <<< конец включения: {target.name}")

    return "\n".join(output_lines), includes


def _find_include(name: str, search_dirs: list[Path], current: Path) -> Path:
    """Найти включаемый файл в каталогах поиска.

    Raises:
        ResourceFormatError: Если файл не найден.
    """
    for directory in search_dirs:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise ResourceFormatError(
        f"Включение '{name}' из '{current.name}' не найдено "
        f"(искали в: {', '.join(str(item) for item in search_dirs)})"
    )


def detect_stage(path: Path, source: str) -> ShaderStage:
    """Определить стадию шейдера по расширению или директиве ``#stage``."""
    stage = _STAGE_BY_EXTENSION.get(path.suffix.lower())
    if stage is not None:
        return stage
    match = _STAGE_PATTERN.search(source)
    if match is not None:
        return _STAGE_BY_NAME.get(match.group(1).lower(), ShaderStage.UNKNOWN)
    return ShaderStage.UNKNOWN
