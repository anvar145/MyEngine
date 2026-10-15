"""Тесты локализации интерфейса редактора."""

from __future__ import annotations

from editor.i18n import DEFAULT_LANGUAGE, LANGUAGES, Translator, load_strings


class TestDictionaries:
    def test_both_languages_are_present(self) -> None:
        for code in LANGUAGES:
            strings = load_strings(code)
            assert strings, f"словарь '{code}' пуст"

    def test_keys_match_between_languages(self) -> None:
        ru = load_strings("ru")
        en = load_strings("en")
        missing_in_en = sorted(set(ru) - set(en))
        missing_in_ru = sorted(set(en) - set(ru))
        assert not missing_in_en, f"нет английского перевода: {missing_in_en}"
        assert not missing_in_ru, f"нет русского перевода: {missing_in_ru}"

    def test_no_empty_translations(self) -> None:
        for code, strings in (("ru", load_strings("ru")), ("en", load_strings("en"))):
            empty = [key for key, value in strings.items() if not value.strip()]
            assert not empty, f"пустые строки в '{code}': {empty}"


class TestTranslator:
    def test_default_language(self) -> None:
        translator = Translator()
        assert translator.language == DEFAULT_LANGUAGE

    def test_translation_and_placeholders(self) -> None:
        translator = Translator("ru")
        assert translator.tr("menu.file.new") == "Новая сцена"
        assert translator.tr("status.selected", name="Cube") == "Выбрано: Cube"
        assert translator.tr("status.deleted", count=3) == "Удалено объектов: 3"

    def test_unknown_key_returns_key(self) -> None:
        translator = Translator("ru")
        assert translator.tr("nonexistent.key") == "nonexistent.key"

    def test_switch_language_emits_signal(self, qapp) -> None:
        translator = Translator("ru")
        seen: list[str] = []
        translator.language_changed.connect(seen.append)

        assert translator.set_language("en") is True
        assert translator.language == "en"
        assert seen == ["en"]
        assert translator.tr("menu.file.new") == "New scene"

        # Повторная установка того же языка ничего не меняет.
        assert translator.set_language("en") is False
        assert seen == ["en"]

    def test_unknown_language_is_ignored(self) -> None:
        translator = Translator("ru")
        assert translator.set_language("de") is False
        assert translator.language == "ru"

    def test_callable_shortcut(self) -> None:
        translator = Translator("en")
        assert translator("menu.view") == "&View"

    def test_available_languages_have_labels(self) -> None:
        translator = Translator("ru")
        labels = dict(translator.available_languages)
        assert labels["ru"] == "Русский"
        assert labels["en"] == "English"

    def test_language_name_is_localized(self) -> None:
        assert Translator("ru").language_name == "Русский"
        assert Translator("en").language_name == "English"

    def test_keys_reports_dictionary_size(self) -> None:
        translator = Translator("ru")
        keys = translator.keys()
        assert "menu.file" in keys
        assert len(translator.keys()) > 50

    def test_repr(self) -> None:
        assert "Translator" in repr(Translator("en"))
