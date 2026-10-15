"""Локализация интерфейса редактора (русский и английский).

Строки лежат отдельно от кода — в JSON-файлах рядом с модулем, поэтому текст
можно править без пересборки. Любой виджет реализует метод ``retranslate()``,
а :class:`Translator` уведомляет подписчиков сигналом ``language_changed``;
благодаря этому язык переключается на лету, без перезапуска редактора.

Если в будущем понадобятся внешние переводы (Qt Linguist, ``.ts``/``.qm``),
достаточно заменить загрузчик внутри :class:`Translator`: интерфейс
``tr(...)`` и сигнал останутся теми же.

Example:
    >>> translator = Translator("ru")
    >>> translator.tr("menu.file.new")
    'Новая сцена'
    >>> translator.set_language("en")
    >>> translator.tr("menu.file.new")
    'New scene'
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PySide6.QtCore import QObject, Signal

__all__ = ["DEFAULT_LANGUAGE", "LANGUAGES", "Translator", "load_strings"]

_log = logging.getLogger("editor.i18n")

#: Каталог со словарями (``ru.json``, ``en.json``).
STRINGS_DIR = Path(__file__).resolve().parent

#: Поддерживаемые языки: код → ключ названия в словаре.
LANGUAGES: dict[str, str] = {"ru": "language.ru", "en": "language.en"}

DEFAULT_LANGUAGE = "ru"


def load_strings(language: str) -> dict[str, str]:
    """Прочитать словарь языка.

    Args:
        language: Код языка (``"ru"`` или ``"en"``).

    Returns:
        Отображение «ключ → строка». Если файла нет, возвращается пустой
        словарь: интерфейс покажет сами ключи, но не упадёт.
    """
    path = STRINGS_DIR / f"{language}.json"
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except OSError as exc:
        _log.error("Не удалось прочитать словарь '%s': %s", path, exc)
        return {}
    except json.JSONDecodeError as exc:  # pragma: no cover - защита от правок
        _log.error("Словарь '%s' содержит некорректный JSON: %s", path, exc)
        return {}
    return {str(key): str(value) for key, value in data.items()}


class Translator(QObject):
    """Хранилище строк текущего языка с уведомлением об изменении.

    Args:
        language: Начальный язык; неизвестный код заменяется на
            :data:`DEFAULT_LANGUAGE`.
        parent: Родительский QObject.

    Signals:
        language_changed: Испускается после смены языка (аргумент — код языка).
    """

    language_changed = Signal(str)

    def __init__(
        self, language: str = DEFAULT_LANGUAGE, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._language = language if language in LANGUAGES else DEFAULT_LANGUAGE
        self._strings = load_strings(self._language)
        self._fallback = (
            {}
            if self._language == DEFAULT_LANGUAGE
            else load_strings(DEFAULT_LANGUAGE)
        )

    # ------------------------------------------------------------------ #
    # Состояние
    # ------------------------------------------------------------------ #
    @property
    def language(self) -> str:
        """Текущий код языка."""
        return self._language

    @property
    def language_name(self) -> str:
        """Название текущего языка на нём самом (для статусной строки)."""
        return self.tr(LANGUAGES.get(self._language, "language.en"))

    @property
    def available_languages(self) -> tuple[tuple[str, str], ...]:
        """Доступные языки: пары «код, подпись на этом языке»."""
        return tuple(
            (code, load_strings(code).get(label_key, label_key))
            for code, label_key in LANGUAGES.items()
        )

    def set_language(self, language: str) -> bool:
        """Переключить язык интерфейса.

        Args:
            language: Код языка.

        Returns:
            ``True``, если язык действительно изменился.
        """
        if language not in LANGUAGES or language == self._language:
            return False
        self._language = language
        self._strings = load_strings(language)
        self._fallback = (
            {}
            if language == DEFAULT_LANGUAGE
            else load_strings(DEFAULT_LANGUAGE)
        )
        self.language_changed.emit(language)
        return True

    # ------------------------------------------------------------------ #
    # Перевод
    # ------------------------------------------------------------------ #
    def tr(self, key: str, **kwargs: object) -> str:
        """Перевести строку по ключу.

        Args:
            key: Ключ вида ``"menu.file.new"``.
            **kwargs: Подстановки для ``str.format`` (например, ``name="Cube"``).

        Returns:
            Переведённый текст; при отсутствии ключа возвращается сам ключ
            (так пропущенный перевод сразу видно в интерфейсе).
        """
        text = self._strings.get(key) or self._fallback.get(key) or key
        if kwargs:
            try:
                return text.format(**kwargs)
            except (KeyError, IndexError):
                _log.warning("В строке '%s' нет подстановок %s", key, sorted(kwargs))
                return text
        return text

    def keys(self) -> frozenset[str]:
        """Набор ключей текущего словаря (для тестов полноты перевода)."""
        return frozenset(self._strings)

    def __call__(self, key: str, **kwargs: object) -> str:
        """Сокращённая запись перевода: ``t("menu.file.new")``."""
        return self.tr(key, **kwargs)

    def __repr__(self) -> str:
        return f"<Translator language='{self._language}' strings={len(self._strings)}>"
