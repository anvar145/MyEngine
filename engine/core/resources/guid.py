"""GUID ассетов: генерация, разбор и проверка.

GUID — это 32 hex-символа в нижнем регистре (``uuid4().hex``). Он хранится в
``<asset>.meta`` рядом с исходным файлом и не меняется при переименовании или
перемещении файла, поэтому именно GUID используется для ссылок между ассетами.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from engine.core.resources.errors import InvalidGuidError

__all__ = [
    "GUID_LENGTH",
    "is_valid_guid",
    "library_subdir",
    "new_guid",
    "parse_guid",
]

GUID_LENGTH = 32
_GUID_RE = re.compile(r"[0-9a-f]{32}")


def new_guid() -> str:
    """Сгенерировать новый GUID.

    Returns:
        Строка из 32 hex-символов в нижнем регистре.
    """
    return uuid.uuid4().hex


def is_valid_guid(value: object) -> bool:
    """``True``, если ``value`` — корректный GUID в каноничной форме."""
    return isinstance(value, str) and _GUID_RE.fullmatch(value) is not None


def parse_guid(value: str) -> str:
    """Привести строку к каноничному GUID.

    Допускает пробелы по краям, верхний регистр и дефисы (формат UUID).

    Args:
        value: Строка с GUID.

    Returns:
        GUID из 32 hex-символов в нижнем регистре.

    Raises:
        InvalidGuidError: Если строка не похожа на GUID.
    """
    if not isinstance(value, str):
        raise InvalidGuidError(f"GUID должен быть строкой, получено {type(value)}")
    normalized = value.strip().lower().replace("-", "")
    if _GUID_RE.fullmatch(normalized) is None:
        raise InvalidGuidError(f"Некорректный GUID: {value!r}")
    return normalized


def library_subdir(guid: str) -> Path:
    """Относительный путь каталога артефактов: ``<xx>/<guid>``.

    Args:
        guid: Каноничный GUID.

    Raises:
        InvalidGuidError: Если GUID некорректен.
    """
    canonical = parse_guid(guid)
    return Path(canonical[:2]) / canonical
