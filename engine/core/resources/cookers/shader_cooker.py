"""Cooker шейдеров: самодостаточный исходник, ключевые слова и варианты.

Cooker не знает про GL: он разворачивает ``#include``, проверяет баланс
условной компиляции, находит **ключевые слова** (имена из ``#ifdef``/``#if``,
которыми варианты включаются снаружи) и считает стабильный хэш исходника.
Готовые бинарники драйвера (``glGetProgramBinary``) кэширует графический слой
(``engine/rendering/program_cache.py``): он же использует :func:`variant_key`.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engine.core.resources.asset import ShaderStage
from engine.core.resources.errors import AssetImportError, ResourceFormatError
from engine.core.resources.loaders.shader_loader import (
    detect_stage,
    resolve_includes,
)

__all__ = [
    "CookedShader",
    "ShaderCookSettings",
    "ShaderVariant",
    "cook_shader",
    "scan_features",
    "source_hash",
    "variant_key",
]

_VERSION = re.compile(r"^\s*#\s*version\s+(\d+)", re.MULTILINE)
_DIRECTIVE = re.compile(r"^\s*#\s*(\w+)\b(.*)$")
_DEFINE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)")
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
_MAIN = re.compile(r"\bvoid\s+main\s*\(")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//[^\n]*")

_IGNORED_NAMES = frozenset({"defined", "true", "false"})


@dataclass(frozen=True, slots=True)
class ShaderVariant:
    """Именованный набор ключевых слов, который стоит собрать заранее.

    Attributes:
        name: Читаемое имя (``"instanced"``).
        defines: Пары «ключевое слово — значение»; ``None`` даёт ``#define X``.
    """

    name: str
    defines: Mapping[str, object] = field(default_factory=dict)

    def key(self) -> str:
        """Стабильный ключ набора определений."""
        return variant_key(self.defines)


@dataclass(frozen=True, slots=True)
class ShaderCookSettings:
    """Настройки cooker'а шейдера.

    Attributes:
        include_dirs: Дополнительные каталоги поиска ``#include``.
        strip_comments: Убрать комментарии и пустые строки (меньше текста для
            драйвера; номера строк в логах компиляции при этом сдвигаются).
        variants: Варианты, которые рантайм может собрать заранее.
    """

    include_dirs: tuple[Path, ...] = ()
    strip_comments: bool = False
    variants: tuple[ShaderVariant, ...] = ()


@dataclass(frozen=True, slots=True)
class CookedShader:
    """Результат cooker'а.

    Attributes:
        source: Исходник со всеми развёрнутыми ``#include``.
        stage: Стадия.
        glsl_version: Номер из ``#version`` или ``None`` (файл-библиотека).
        features: Ключевые слова, которыми шейдер управляется снаружи.
        defaults: Значения по умолчанию для слов с ``#ifndef X / #define X v``.
        includes: Пути включённых файлов.
        has_main: Есть ли ``void main()`` (иначе это библиотека функций).
        digest: Хэш исходника (:func:`source_hash`).
    """

    source: str
    stage: ShaderStage
    glsl_version: int | None
    features: tuple[str, ...]
    defaults: Mapping[str, str]
    includes: tuple[str, ...]
    has_main: bool
    digest: str


def cook_shader(
    path: Path, settings: ShaderCookSettings | None = None
) -> CookedShader:
    """Развернуть шейдер и собрать сведения о нём.

    Args:
        path: Путь к исходнику.
        settings: Настройки (по умолчанию — без дополнительных каталогов).

    Returns:
        Подготовленный шейдер.

    Raises:
        AssetImportError: Если файл не читается, включение не найдено или
            нарушен баланс ``#if``/``#endif``; если вариант использует
            ключевое слово, которого в шейдере нет.
    """
    config = settings or ShaderCookSettings()
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise AssetImportError(f"Не удалось прочитать шейдер '{path}': {exc}") from exc
    try:
        source, includes = resolve_includes(
            path, raw, include_dirs=list(config.include_dirs)
        )
    except ResourceFormatError as exc:
        raise AssetImportError(str(exc)) from exc

    clean = _strip(source)
    _check_balance(clean, path.name)
    if config.strip_comments:
        source = "\n".join(line for line in clean.splitlines() if line.strip()) + "\n"
    elif not source.endswith("\n"):
        source += "\n"

    features, defaults = scan_features(clean)
    for variant in config.variants:
        unknown = sorted(set(variant.defines) - set(features))
        if unknown:
            raise AssetImportError(
                f"Вариант '{variant.name}' шейдера '{path.name}' использует "
                f"неизвестные ключевые слова: {', '.join(unknown)} "
                f"(есть: {', '.join(features) or 'нет'})"
            )

    version = _VERSION.search(clean)
    return CookedShader(
        source=source,
        stage=detect_stage(path, raw),
        glsl_version=int(version.group(1)) if version else None,
        features=features,
        defaults=defaults,
        includes=tuple(includes),
        has_main=_MAIN.search(clean) is not None,
        digest=source_hash(source),
    )


def scan_features(source: str) -> tuple[tuple[str, ...], dict[str, str]]:
    """Найти ключевые слова вариантов и их значения по умолчанию.

    Ключевое слово — имя из ``#ifdef``/``#ifndef``/``#if``/``#elif``, которое
    шейдер не определяет безусловно сам. Имена, заданные ``#define`` на
    верхнем уровне (константы вроде ``LIGHT_DIRECTIONAL``), в список не входят.

    Args:
        source: Исходник (комментарии уже не мешают, но допустимы).

    Returns:
        Отсортированные слова и словарь значений по умолчанию (слова, для
        которых внутри ``#ifndef X`` есть ``#define X значение``).
    """
    used: set[str] = set()
    constants: set[str] = set()
    include_guards: set[str] = set()
    defaults: dict[str, str] = {}
    guards: list[str | None] = []  # имя из #ifndef для каждого открытого #if
    for line in _strip(source).splitlines():
        match = _DIRECTIVE.match(line)
        if match is None:
            continue
        name, rest = match.group(1), match.group(2)
        if name in ("ifdef", "ifndef"):
            found = _IDENTIFIER.findall(rest)
            used.update(found[:1])
            guards.append(found[0] if name == "ifndef" and found else None)
        elif name == "if":
            used.update(_identifiers(rest))
            guards.append(None)
        elif name == "elif":
            used.update(_identifiers(rest))
        elif name == "endif":
            if guards:
                guards.pop()
        elif name == "define":
            defined = _DEFINE.match(line)
            if defined is None:
                continue
            word = defined.group(1)
            if not guards:
                constants.add(word)
            elif guards[-1] == word:
                value = line.split(word, 1)[1].strip()
                if value:
                    defaults[word] = value
                else:
                    include_guards.add(word)  # защита от повторного включения
    features = sorted(used - constants - include_guards - _IGNORED_NAMES)
    features = [name for name in features if not name.startswith(("GL_", "__"))]
    return tuple(features), {k: v for k, v in defaults.items() if k in features}


def source_hash(source: str) -> str:
    """Короткий (16 hex) стабильный хэш исходника."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def variant_key(defines: Mapping[str, Any] | None) -> str:
    """Стабильный ключ набора определений (порядок слов не важен).

    Пустой набор даёт ``"base"``. Значение ``None`` и ``1`` — разные ключи,
    потому что порождают разный текст шейдера.
    """
    if not defines:
        return "base"
    parts = (
        f"{name}" if value is None else f"{name}={value}"
        for name, value in sorted(defines.items())
    )
    return ",".join(parts)


def _identifiers(expression: str) -> set[str]:
    return set(_IDENTIFIER.findall(expression))


def _strip(source: str) -> str:
    """Убрать комментарии, сохранив число строк."""

    def blank(match: re.Match[str]) -> str:
        return "\n" * match.group(0).count("\n")

    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub(blank, source))


def _check_balance(source: str, name: str) -> None:
    depth = 0
    for number, line in enumerate(source.splitlines(), start=1):
        match = _DIRECTIVE.match(line)
        if match is None:
            continue
        directive = match.group(1)
        if directive in ("if", "ifdef", "ifndef"):
            depth += 1
        elif directive in ("else", "elif") and depth == 0:
            raise AssetImportError(f"{name}:{number}: #{directive} без #if")
        elif directive == "endif":
            depth -= 1
            if depth < 0:
                raise AssetImportError(f"{name}:{number}: #endif без #if")
    if depth:
        raise AssetImportError(f"{name}: не закрыто #if/#ifdef ({depth})")
