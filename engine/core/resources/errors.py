"""Ошибки подсистемы ресурсов.

Все исключения наследуются от :class:`ResourceError`, поэтому вызывающий код
может ловить как конкретную причину, так и всю группу.
"""

from __future__ import annotations

__all__ = [
    "AssetImportError",
    "ImporterNotFoundError",
    "InvalidGuidError",
    "MetaFormatError",
    "MissingLoaderError",
    "ResourceError",
    "ResourceFormatError",
    "ResourceLoadError",
    "ResourceNotFoundError",
]


class ResourceError(Exception):
    """Базовая ошибка работы с ресурсами."""


class ResourceNotFoundError(ResourceError):
    """Файл ресурса не найден ни по указанному пути, ни в корнях поиска."""


class MissingLoaderError(ResourceError):
    """Для расширения файла не зарегистрирован ни один загрузчик."""


class ResourceFormatError(ResourceError):
    """Содержимое файла не соответствует ожидаемому формату."""


class ResourceLoadError(ResourceError):
    """Загрузчик завершился ошибкой (обёртка исходного исключения)."""


class InvalidGuidError(ResourceError, ValueError):
    """Строка не является корректным GUID (ожидается 32 hex-символа)."""


class MetaFormatError(ResourceFormatError):
    """Файл ``.meta`` повреждён или содержит недопустимые данные."""


class ImporterNotFoundError(ResourceError):
    """Для расширения файла не зарегистрирован ни один импортёр."""


class AssetImportError(ResourceError):
    """Импортёр не смог обработать исходный файл ассета."""
