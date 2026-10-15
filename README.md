# MyEngine

Модульный 3D-движок с игровой логикой на Python и редактором на PySide6.
Только Windows, OpenGL 3.3 Core (ориентир — слабые встроенные GPU, например Intel HD 520).

## Возможности

- Сцена из `GameObject` и компонентов, фиксированный шаг физики и переменный шаг логики.
- OpenGL-рендерер: PBR, тени, инстансинг, спрайты, текст, UI, процедурное небо, пост-обработка.
- Собственная физика на numpy: коллайдеры, твёрдые тела, триггеры, `CharacterController`, mesh-коллайдеры.
- Импорт ассетов (OBJ, glTF, FBX, текстуры, звук) с GUID, `.meta` и кэшем `Library/`; упаковка в `.pak`.
- Скрипты на Python (функциональные хуки и классы `Behaviour`) с горячей перезагрузкой.
- Префабы, пулы объектов, планировщик задач и твины, сохранения, звук (в том числе 3D).
- Редактор: иерархия, инспектор, вьюпорт с гизмо, undo/redo, Play Mode в отдельном процессе.
- Сборка игры в `exe` через PyInstaller.

## Требования

- Windows, Python 3.10+
- Видеокарта с OpenGL 3.3

## Установка

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[editor,player,dev]"
```

Дополнительные группы зависимостей: `gltf`, `fbx`, `audio`, `package`.

## Запуск

```bash
# редактор (без --project сначала откроется менеджер проектов)
.venv\Scripts\python.exe main.py
.venv\Scripts\python.exe main.py --project projects/PlayModeTest

# игра без редактора
.venv\Scripts\python.exe -m engine.player --project projects/PlayModeTest
```

Сборка игры в `exe` (нужна группа зависимостей `package`):

```bash
.venv\Scripts\python.exe -m pip install -e ".[package]"
.venv\Scripts\myengine.exe package --project projects/PlayModeTest
```

## Тесты и линтер

```bash
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check
```

Часть тестов требует настоящего GL-контекста; для Qt-тестов задайте `QT_QPA_PLATFORM=offscreen`.

## Структура

| Каталог | Назначение |
|---|---|
| `engine/core/` | ядро без Qt и GL: сцена, компоненты, ресурсы, сохранения |
| `engine/rendering/` | OpenGL-рендерер |
| `engine/physics/` | физика |
| `engine/scripting/` | рантайм скриптов |
| `engine/audio/`, `engine/player/`, `engine/cli/` | звук, плеер, упаковка |
| `editor/` | редактор на PySide6 |
| `assets/` | шейдеры, иконки, шаблоны проектов |
| `projects/` | примеры проектов |
| `scripts/`, `tools/` | демо-сцены, проверки, бенчмарки |
| `build/packaging/` | сборка exe |
| `docs/` | документация |

Зависимости идут строго вниз: `core` ← `rendering` ← `editor`.

## Документация

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — архитектура и статус реализации
- [docs/CORE.md](docs/CORE.md) — ядро
- [docs/RENDERING.md](docs/RENDERING.md) — рендеринг
- [docs/EDITOR.md](docs/EDITOR.md) — редактор
- [docs/SCRIPTING.md](docs/SCRIPTING.md) — скрипты
