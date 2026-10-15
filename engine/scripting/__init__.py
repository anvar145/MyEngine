"""Скриптинг MyEngine: компонент-скрипт, песочница загрузки и подсветка.

Пакет состоит из двух слоёв:

* **headless-ядро** — :mod:`engine.scripting.runtime` и
  :mod:`engine.scripting.script_component`: изоляция пространств имён,
  жизненный цикл, горячая перезагрузка. PySide6 здесь не нужен;
* **слой Qt** — :mod:`engine.scripting.python_highlighter` (подсветка для
  ``QPlainTextEdit``). Он подтягивается отдельно и не ломает импорт пакета в
  окружении без PySide6, потому что PySide6 — необязательная зависимость.

Example:
    >>> from engine.scripting import ScriptComponent, ScriptRuntime
    >>> runtime = ScriptRuntime(project_root=Path("."))
    >>> cube.add_component(ScriptComponent("scripts/spin.py", runtime=runtime))
"""

from __future__ import annotations

from engine.scripting.behaviour import (
    HOOK_NAMES,
    Behaviour,
    event_handler,
    export,
    require_component,
)
from engine.scripting.fields import Prefab
from engine.scripting.runtime import (
    ALLOWED_MODULES,
    ALLOWED_ROOTS,
    BLOCKED_BUILTINS,
    ScriptError,
    ScriptHandle,
    ScriptRuntime,
    create_builtins,
    format_traceback,
    load_module,
    safe_import,
)
from engine.scripting.script_component import (
    API_NAMES,
    InputState,
    ScriptComponent,
    ScriptTime,
    default_runtime,
    runtime_for,
    set_default_runtime,
)
from engine.scripting.syntax import (
    BUILTIN_NAMES,
    ENGINE_API,
    ENGINE_API_ATTRIBUTES,
    KEYWORDS,
    LIFECYCLE_HOOKS,
    TOKEN_COLORS,
    completion_words,
)

__all__ = [
    "Prefab",
    # реестр и загрузка
    "ALLOWED_MODULES",
    "ALLOWED_ROOTS",
    "BLOCKED_BUILTINS",
    "ScriptError",
    "ScriptHandle",
    "ScriptRuntime",
    "create_builtins",
    "default_runtime",
    "format_traceback",
    "load_module",
    "runtime_for",
    "safe_import",
    "set_default_runtime",
    # Behaviour
    "HOOK_NAMES",
    "Behaviour",
    "event_handler",
    "export",
    "require_component",
    # компонент
    "API_NAMES",
    "InputState",
    "ScriptComponent",
    "ScriptTime",
    # данные подсветки
    "BUILTIN_NAMES",
    "ENGINE_API",
    "ENGINE_API_ATTRIBUTES",
    "KEYWORDS",
    "LIFECYCLE_HOOKS",
    "TOKEN_COLORS",
    "completion_words",
]
