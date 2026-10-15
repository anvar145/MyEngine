"""Изолированное исполнение скриптов: загрузка, песочница и горячая перезагрузка.

Модуль отвечает за всё, что происходит **вокруг** пользовательского кода:

* :func:`load_module` / :func:`compile_source` — превратить файл в объект-модуль
  с собственным пространством имён (``ModuleType``);
* :func:`create_builtins` — собрать безопасный ``__builtins__`` и предзагрузить
  модули, разрешённые для скриптов;
* :class:`ScriptRuntime` — реестр живых скриптов, перехват исключений с
  форматированием traceback и горячая перезагрузка при изменении файла.

Изоляция устроена так: каждый скрипт получает **свой** словарь-модуль и
**свой** ``__builtins__``. Скрипты не могут случайно затереть имена друг друга,
а ``import`` внутри скрипта не видит чужих модулей: белый список задаётся
параметром ``allowed_modules``.

Example:
    >>> runtime = ScriptRuntime(project_root=Path("."))
    >>> handle, module = runtime.load(Path("scripts/spin.py"))
    >>> module.on_start(instance)
    >>> runtime.close(handle)
"""

from __future__ import annotations

import builtins
import contextlib
import functools
import importlib
import logging
import sys
import time
import tokenize
import traceback
import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ALLOWED_MODULES",
    "BLOCKED_BUILTINS",
    "ScriptError",
    "ScriptHandle",
    "ScriptRuntime",
    "create_builtins",
    "format_traceback",
    "load_module",
    "module_name_for",
    "safe_import",
]

_log = logging.getLogger("engine.scripting")
#: Логгер пакета должен пропускать INFO: ``print()`` и ``self.log()`` пишут
#: именно на этом уровне, а корневой логгер по умолчанию режет всё ниже
#: WARNING ещё до обработчиков (в том числе консоли редактора).
if _log.level == logging.NOTSET:
    _log.setLevel(logging.INFO)

#: Корневые пакеты, которые разрешены скрипту целиком: собственный движок.
#: Скрипт пишется **для** MyEngine, поэтому ``from engine.core.math3d import
#: Vector3`` — обычное дело, а вот ``os``/``subprocess``/``sys`` остаются
#: закрытыми, чтобы опечатка не превратилась в работу с файлами и процессами.
ALLOWED_ROOTS: frozenset[str] = frozenset({"engine", "__future__"})

#: Модули, которые скрипт может импортировать (мягкая песочница).
ALLOWED_MODULES: frozenset[str] = frozenset(
    {
        "array",
        "bisect",
        "collections",
        "colorsys",
        "copy",
        "csv",
        "dataclasses",
        "enum",
        "functools",
        "heapq",
        "itertools",
        "json",
        "logging",
        "math",
        "numpy",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
        "time",
        "typing",
        "uuid",
    }
)

#: Встроенные функции, недоступные скриптам: произвольное исполнение кода и
#: управление процессом. Это защита «от опечатки», а не полноценная песочница:
#: Python в принципе не изолирует код внутри процесса.
BLOCKED_BUILTINS: frozenset[str] = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "exit",
        "globals",
        "input",
        "locals",
        "memoryview",
        "open",
        "quit",
        "vars",
    }
)

#: Сколько строк traceback попадает в лог (полный текст уходит в консоль).
TRACEBACK_LIMIT = 12

#: Максимальная длина одной строки лога скрипта.
LOG_LINE_LIMIT = 4000


class ScriptError(Exception):
    """Ошибка загрузки или исполнения скрипта.

    Args:
        message: Краткое описание («синтаксическая ошибка», «ошибка on_update»).
        name: Человекочитаемое имя скрипта.
        text: Полный отформатированный traceback.
        syntax: ``True``, если ошибка возникла на этапе компиляции.

    Attributes:
        name: Имя скрипта.
        text: Traceback целиком (для консоли редактора).
        syntax: Признак синтаксической ошибки.
    """

    def __init__(
        self,
        message: str,
        *,
        name: str = "",
        text: str = "",
        syntax: bool = False,
    ) -> None:
        super().__init__(message)
        self.name = name
        self.text = text
        self.syntax = syntax

    def summary(self) -> str:
        """Однострочная сводка: имя скрипта и причина."""
        prefix = f"[{self.name}] " if self.name else ""
        return f"{prefix}{self}"


# ---------------------------------------------------------------------- #
# Пространства имён и загрузка файлов
# ---------------------------------------------------------------------- #
def safe_import(
    name: str,
    globals_: dict[str, Any] | None = None,
    locals_: dict[str, Any] | None = None,  # noqa: A002 - подпись __import__
    fromlist: tuple[str, ...] = (),
    level: int = 0,
    *,
    allowed: frozenset[str] = ALLOWED_MODULES,
    allowed_roots: frozenset[str] = ALLOWED_ROOTS,
) -> types.ModuleType:
    """Импорт для скриптов: разрешены только модули белого списка.

    Args:
        name: Имя модуля (как в обычном ``import``).
        globals_: Словарь вызывающего кода (передаёт интерпретатор).
        locals_: Локальные имена (передаёт интерпретатор).
        fromlist: Список имён для ``from ... import ...``.
        level: Уровень относительного импорта (не поддерживается).
        allowed: Белый список корневых имён модулей.
        allowed_roots: Пакеты, разрешённые целиком (движок и ``__future__``,
            без которого не работает привычная шапка
            ``from __future__ import annotations``).

    Returns:
        Загруженный модуль.

    Raises:
        ImportError: Если модуль запрещён или не найден.
    """
    root = name.split(".")[0]
    if level:
        raise ImportError("относительный импорт в скриптах не поддерживается")
    if root not in allowed and root not in allowed_roots:
        raise ImportError(
            f"модуль '{name}' недоступен скриптам (разрешены: "
            f"{', '.join(sorted(allowed | allowed_roots))})"
        )
    return builtins.__import__(name, globals_, locals_, fromlist, level)


def _make_print(stream: types.ModuleType, fallback: Any) -> Any:
    """Собрать ``print``, который уходит и в stdout, и в лог движка.

    Args:
        stream: Модуль ``sys`` (нужен на момент вызова, а не создания).
        fallback: Исходный ``print`` — используется после закрытия потока.

    Returns:
        Функцию печати для скриптов.
    """
    logger = logging.getLogger("engine.scripting.print")

    def script_print(*values: object, **kwargs: object) -> None:
        """Напечатать значения в консоль редактора и в stdout."""
        try:
            text = " ".join(str(value) for value in values)
        except Exception:  # noqa: BLE001 - печать не должна падать
            text = "<непечатаемое значение>"
        logger.info("%s", text[:LOG_LINE_LIMIT])
        stdout = getattr(stream, "stdout", None)
        if stdout is not None:
            try:
                kwargs.setdefault("file", stdout)
                fallback(*values, **kwargs)  # type: ignore[arg-type]
            except (OSError, ValueError):
                pass

    return script_print


def create_builtins(
    *,
    allowed: frozenset[str] = ALLOWED_MODULES,
    allowed_roots: frozenset[str] = ALLOWED_ROOTS,
    preloaded: tuple[str, ...] = ("math", "random", "time"),
) -> dict[str, Any]:
    """Создать словарь ``__builtins__`` для скриптового модуля.

    Args:
        allowed: Белый список модулей для ``import``.
        allowed_roots: Пакеты, разрешённые целиком (движок).
        preloaded: Модули, сразу доступные по имени (``math``, ``random``, ...).

    Returns:
        Копия встроенных имён без опасных функций, с ограниченным ``import``
        и перехваченным ``print``.
    """
    namespace: dict[str, Any] = {
        name: getattr(builtins, name)
        for name in dir(builtins)
        if not name.startswith("_") or name in ("__build_class__", "__name__")
    }
    for name in BLOCKED_BUILTINS:
        namespace.pop(name, None)

    namespace["__import__"] = functools.partial(
        safe_import, allowed=allowed, allowed_roots=allowed_roots
    )
    namespace["print"] = _make_print(sys, builtins.print)

    for module_name in preloaded:
        if module_name not in allowed:
            continue
        try:
            namespace[module_name] = importlib.import_module(module_name)
        except ImportError:  # pragma: no cover - стандартная библиотека
            _log.debug("Предзагрузка модуля '%s' не удалась", module_name)
    return namespace


def module_name_for(path: Path, root: Path | None = None) -> str:
    """Собрать стабильное имя модуля по пути к файлу.

    Args:
        path: Путь к скрипту.
        root: Корень проекта; путь внутри него становится точечным именем.

    Returns:
        Имя вида ``myengine_script_scripts.spin``.
    """
    resolved = Path(path).resolve()
    stem = resolved.stem
    if root is not None:
        try:
            relative = resolved.relative_to(Path(root).resolve())
        except ValueError:
            relative = None
        if relative is not None and relative.parent != Path("."):
            parts = [
                part
                for part in relative.with_suffix("").parts
                if part.isidentifier()
            ]
            if parts:
                return "myengine_script_" + ".".join(parts)
    return f"myengine_script_{stem}"


def _read_source(path: Path) -> str:
    """Прочитать исходник с учётом объявленной кодировки.

    Args:
        path: Путь к файлу скрипта.

    Returns:
        Исходный текст.
    """
    with tokenize.open(str(path)) as stream:
        return stream.read()


def compile_source(
    source: str,
    name: str,
    filename: str = "<script>",
) -> types.CodeType:
    """Скомпилировать исходник скрипта.

    Args:
        source: Текст скрипта.
        name: Имя модуля (для трассировки).
        filename: Путь для сообщений об ошибках.

    Returns:
        Код-объект, готовый к исполнению.

    Raises:
        ScriptError: При синтаксической ошибке.
    """
    try:
        return compile(source, filename, "exec", dont_inherit=True)
    except SyntaxError as exc:
        raise ScriptError(
            f"синтаксическая ошибка: {exc.msg} (строка {exc.lineno})",
            name=name,
            text=format_traceback(exc, filename=filename),
            syntax=True,
        ) from exc


def load_module(
    name: str,
    source: str,
    filename: str = "<script>",
    *,
    previous: dict[str, Any] | None = None,
    allowed: frozenset[str] = ALLOWED_MODULES,
    allowed_roots: frozenset[str] = ALLOWED_ROOTS,
    preloaded: tuple[str, ...] = ("math", "random", "time"),
) -> types.ModuleType:
    """Создать изолированный модуль из исходника.

    Args:
        name: Имя модуля.
        source: Исходный текст.
        filename: Путь к файлу (попадает в traceback и ``__file__``).
        previous: Словарь предыдущей версии модуля: из него переносятся имена,
            которых нет в новом исходнике (состояние переживает горячую
            перезагрузку, если скрипт его не переопределяет).
        allowed: Белый список модулей для ``import``.
        allowed_roots: Пакеты, разрешённые целиком (движок).
        preloaded: Модули, доступные сразу по имени.

    Returns:
        Загруженный модуль со выполненным телом.

    Raises:
        ScriptError: При синтаксической ошибке или исключении на верхнем уровне.
    """
    code = compile_source(source, name, filename)
    module = types.ModuleType(name)
    namespace = module.__dict__
    namespace["__builtins__"] = create_builtins(
        allowed=allowed, allowed_roots=allowed_roots, preloaded=preloaded
    )
    namespace["__file__"] = filename
    namespace["__name__"] = name
    namespace["__package__"] = None
    namespace["__doc__"] = None
    # Имена для аннотаций полей (``target: GameObject = None``): без них
    # аннотация упала бы с NameError при исполнении тела модуля.
    from engine.core.game_object import GameObject
    from engine.scripting.fields import Asset, Prefab

    namespace["GameObject"] = GameObject
    namespace["Asset"] = Asset
    namespace["Prefab"] = Prefab

    if previous:
        carried = {
            key: value
            for key, value in previous.items()
            if not (key.startswith("__") and key.endswith("__"))
            and not isinstance(value, types.ModuleType)
        }
        namespace.update(carried)
        namespace["__builtins__"] = create_builtins(
            allowed=allowed, allowed_roots=allowed_roots, preloaded=preloaded
        )

    # Тело модуля исполняется в его собственном словаре: интерпретатор считает
    # этот словарь глобальным, поэтому функции после загрузки видят __globals__.
    try:
        exec(code, namespace)  # noqa: S102 - исполняем код пользователя по его просьбе
    except BaseException as exc:  # noqa: BLE001 - ошибку показываем пользователю
        raise ScriptError(
            f"{type(exc).__name__}: {exc}",
            name=name,
            text=format_traceback(exc, filename=filename),
        ) from exc
    return module


def format_traceback(
    error: BaseException,
    *,
    filename: str | None = None,
    limit: int = TRACEBACK_LIMIT,
) -> str:
    """Отформатировать traceback, отбросив служебные кадры загрузчика.

    Служебными считаются кадры самого загрузчика (:mod:`engine.scripting`) и
    кадры функции-обёртки ``ScriptRuntime.call``: пользователю нужен только его
    собственный код. Если кадр с искомым файлом не найден (например, ошибка
    возникла в импортированном модуле скрипта), отбрасываются только кадры
    загрузчика.

    Args:
        error: Исключение.
        filename: Путь к скрипту: кадры до него считаются служебными.
        limit: Максимальное число строк; остаток заменяется пометкой.

    Returns:
        Текст traceback, пригодный для консоли редактора.
    """
    frames = traceback.extract_tb(error.__traceback__)
    if filename:
        target = str(filename)
        for index, frame in enumerate(frames):
            if _same_file(frame.filename, target):
                frames = frames[index:]
                break
    frames = [frame for frame in frames if not _is_runtime_frame(frame.filename)]
    lines = ["Traceback (последний вызов — последним):"]
    lines.extend(traceback.format_list(frames))
    lines.extend(traceback.format_exception_only(type(error), error))
    text = "".join(lines).rstrip()
    parts = text.splitlines()
    if limit > 0 and len(parts) > limit:
        hidden = len(parts) - limit
        parts = [*parts[:limit], f"… ещё {hidden} строк(и) скрыто"]
    return "\n".join(parts)


def _is_runtime_frame(filename: str) -> bool:
    """Относится ли кадр к самому загрузчику скриптов.

    Args:
        filename: Путь из кадра трассировки.

    Returns:
        ``True`` для файлов модуля :mod:`engine.scripting`.
    """
    try:
        return Path(filename).resolve().parent == Path(__file__).resolve().parent
    except OSError:  # pragma: no cover - защита от экзотических путей
        return False


def _same_file(left: str, right: str) -> bool:
    """Сравнить пути кадров трассировки с точностью до нормализации."""
    if left == right:
        return True
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:  # pragma: no cover - защита от экзотических путей
        return False


# ---------------------------------------------------------------------- #
# Реестр исполняемых скриптов
# ---------------------------------------------------------------------- #
@dataclass
class ScriptHandle:
    """Учётная запись загруженного скрипта в реестре.

    Attributes:
        token: Уникальное имя владельца (путь или синтетический идентификатор).
        name: Человекочитаемое имя (обычно имя файла).
        path: Путь к файлу или ``None`` для скрипта из строки.
        module: Загруженный модуль.
        revision: Номер успешной загрузки (для горячей перезагрузки).
        modified: Время изменения файла на момент загрузки.
        errors: Сколько раз скрипт падал.
        active: Флаг «выполняется» — для кнопки «Стоп».
        meta: Произвольные данные владельца (например, компонент).
    """

    token: str
    name: str
    path: Path | None
    module: types.ModuleType
    revision: int = 1
    modified: float = 0.0
    errors: int = 0
    active: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> Path | str:
        """Источник скрипта: путь к файлу или синтетическая пометка."""
        return self.path if self.path is not None else f"<{self.name}>"

    def __repr__(self) -> str:
        state = "активен" if self.active else "остановлен"
        return (
            f"<ScriptHandle '{self.name}' r{self.revision} {state} "
            f"ошибок={self.errors}>"
        )


class ScriptRuntime:
    """Реестр скриптов: загрузка в изоляции, вызовы хуков и перезагрузка.

    Args:
        project_root: Корень проекта (для импорта соседних скриптов и имён).
        logger: Логгер движка; по умолчанию ``engine.scripting``.
        allowed_modules: Белый список модулей для ``import``.
        preloaded: Модули, доступные скрипту по имени.
        traceback_limit: Сколько строк traceback писать в лог.
        auto_reload: Проверять время изменения файлов в :meth:`tick`.

    Attributes:
        scripts: Отображение «токен → :class:`ScriptHandle`».
    """

    def __init__(
        self,
        *,
        project_root: Path | None = None,
        logger: logging.Logger | None = None,
        allowed_modules: frozenset[str] = ALLOWED_MODULES,
        allowed_roots: frozenset[str] = ALLOWED_ROOTS,
        preloaded: tuple[str, ...] = ("math", "random", "time"),
        traceback_limit: int = TRACEBACK_LIMIT,
        auto_reload: bool = True,
    ) -> None:
        self.project_root = Path(project_root) if project_root else None
        self.log = logger or _log
        self.allowed_modules = allowed_modules
        self.allowed_roots = allowed_roots
        self.preloaded = preloaded
        self.traceback_limit = int(traceback_limit)
        self.auto_reload = auto_reload
        self.scripts: dict[str, ScriptHandle] = {}
        self._by_path: dict[Path, str] = {}
        self._sequence = 0
        self._source_resolver: Callable[[str], bytes | None] | None = None

    def set_source_resolver(
        self, resolver: Callable[[str], bytes | None] | None
    ) -> None:
        """Задать источник исходников вместо диска (например, ``vfs.read_file``).

        Резолвер получает путь от корня проекта (``Assets/Scripts/a.py``, разделитель
        ``/``) и возвращает байты или ``None``; при ``None`` скрипт читается с диска.

        Args:
            resolver: Функция ``путь -> bytes | None``; ``None`` снимает резолвер.
        """
        self._source_resolver = resolver

    def read_virtual(self, path: Path | str) -> str | None:
        """Исходник скрипта через резолвер или ``None``, если его там нет.

        Raises:
            ScriptError: Если байты не декодируются как UTF-8.
        """
        resolver = self._source_resolver
        if resolver is None:
            return None
        target = Path(path)
        if target.is_absolute() and self.project_root is not None:
            try:
                target = target.relative_to(self.project_root.resolve())
            except ValueError:
                try:
                    target = target.relative_to(self.project_root)
                except ValueError:
                    return None
        elif target.is_absolute():
            return None
        data = resolver(target.as_posix())
        if data is None:
            return None
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ScriptError(
                f"не удалось прочитать файл: {exc}", name=target.name
            ) from exc

    # ------------------------------------------------------------------ #
    # Реестр
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        """Сколько скриптов сейчас зарегистрировано."""
        return len(self.scripts)

    def __contains__(self, token: object) -> bool:
        """Зарегистрирован ли скрипт с таким токеном."""
        return token in self.scripts

    def __iter__(self) -> Iterator[ScriptHandle]:
        """Итерация по учётным записям скриптов."""
        return iter(self.scripts.values())

    @property
    def active_scripts(self) -> tuple[ScriptHandle, ...]:
        """Скрипты, отмеченные как выполняющиеся."""
        return tuple(handle for handle in self.scripts.values() if handle.active)

    def next_token(self, prefix: str = "script") -> str:
        """Выдать уникальный токен для скрипта без файла.

        Args:
            prefix: Префикс токена.

        Returns:
            Строка вида ``script-3``.
        """
        self._sequence += 1
        return f"{prefix}-{self._sequence}"

    # ------------------------------------------------------------------ #
    # Загрузка
    # ------------------------------------------------------------------ #
    def load(
        self,
        path: Path | str,
        *,
        token: str | None = None,
        name: str | None = None,
        previous: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> ScriptHandle:
        """Загрузить скрипт из файла.

        Args:
            path: Путь к файлу ``*.py``.
            token: Ключ владельца (по умолчанию — путь файла).
            name: Отображаемое имя (по умолчанию — имя файла).
            previous: Словарь прежней версии модуля (для перезагрузки).
            meta: Произвольные данные владельца.

        Returns:
            Учётную запись загруженного скрипта.

        Raises:
            ScriptError: Если файл не читается, не компилируется или падает
                на верхнем уровне.
        """
        target = Path(path)
        source = self.read_virtual(target)
        if source is not None:
            # Файл из пака: путь виртуальный, делаем его абсолютным от корня.
            if not target.is_absolute() and self.project_root is not None:
                target = self.project_root / target
        else:
            with contextlib.suppress(OSError):
                # Читаем через resolve(), но не падаем на экзотических путях.
                target = target.resolve()
            if not target.is_file():
                raise ScriptError(f"файл скрипта не найден: {target}", name=target.name)
        try:
            if source is None:
                source = _read_source(target)
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            raise ScriptError(
                f"не удалось прочитать файл: {exc}",
                name=target.name,
                text=format_traceback(exc, filename=str(target)),
            ) from exc

        key = token or str(target)
        module = self.load_source(
            source,
            filename=str(target),
            name=self.module_name(target),
            previous=previous,
        )
        handle = self.register(
            key, module, name=name or target.name, path=target, meta=meta
        )
        self.log.info("Скрипт загружен: %s", handle.name)
        return handle

    def load_source(
        self,
        source: str,
        *,
        name: str | None = None,
        filename: str = "<script>",
        previous: dict[str, Any] | None = None,
    ) -> types.ModuleType:
        """Загрузить скрипт из строки (без файла).

        Args:
            source: Исходный текст.
            name: Имя модуля.
            filename: Имя, попадающее в traceback.
            previous: Словарь прежней версии модуля.

        Returns:
            Загруженный модуль.
        """
        module_name = name or self.module_name(None)
        return load_module(
            module_name,
            source,
            filename,
            previous=previous,
            allowed=self.allowed_modules,
            allowed_roots=self.allowed_roots,
            preloaded=self.preloaded,
        )

    def register(
        self,
        token: str,
        module: types.ModuleType,
        *,
        name: str = "",
        path: Path | None = None,
        meta: dict[str, Any] | None = None,
    ) -> ScriptHandle:
        """Взять готовый модуль под управление реестра.

        Args:
            token: Ключ владельца.
            module: Модуль скрипта.
            name: Отображаемое имя.
            path: Путь к файлу (для слежения за изменениями).
            meta: Произвольные данные владельца.

        Returns:
            Учётную запись (прежняя запись с тем же токеном заменяется).
        """
        existing = self.scripts.pop(token, None)
        handle = ScriptHandle(
            token=token,
            name=name or module.__name__,
            path=path,
            module=module,
            revision=(existing.revision + 1) if existing is not None else 1,
            modified=self.file_stamp(path) if path is not None else 0.0,
            active=existing.active if existing is not None else True,
            meta=meta if meta is not None else (existing.meta if existing else {}),
        )
        self.scripts[token] = handle
        if path is not None:
            self._by_path[path] = token
        return handle

    def unregister(self, token: str) -> bool:
        """Убрать скрипт из реестра (без вызова ``on_destroy``).

        Args:
            token: Ключ владельца.

        Returns:
            ``True``, если запись существовала.
        """
        handle = self.scripts.pop(token, None)
        if handle is None:
            return False
        if handle.path is not None and self._by_path.get(handle.path) == token:
            del self._by_path[handle.path]
        return True

    def get(self, token: str) -> ScriptHandle | None:
        """Найти учётную запись по токену."""
        return self.scripts.get(token)

    def by_path(self, path: Path | str) -> ScriptHandle | None:
        """Найти учётную запись по пути файла."""
        token = self._by_path.get(Path(path))
        return self.scripts.get(token) if token is not None else None

    def close(self) -> None:
        """Остановить и забыть все скрипты.

        Returns:
            Ничего; используется при закрытии редактора или смене сцены.
        """
        for handle in list(self.scripts.values()):
            handle.active = False
        self.scripts.clear()
        self._by_path.clear()
        self.log.debug("Реестр скриптов очищен")

    # ------------------------------------------------------------------ #
    # Вызовы хуков
    # ------------------------------------------------------------------ #
    def call(
        self,
        handle: ScriptHandle,
        function: str,
        *arguments: object,
        report: bool = True,
    ) -> Any:
        """Вызвать хук скрипта с перехватом исключений.

        Args:
            handle: Учётная запись скрипта.
            function: Имя функции (``on_start``, ``on_update``, ...).
            *arguments: Аргументы вызова (обычно экземпляр скрипта).
            report: Писать ли traceback в лог.

        Returns:
            Результат функции или ``None``, если её нет или она упала.
        """
        target = getattr(handle.module, function, None)
        if not callable(target):
            return None
        return self.invoke(handle, target, function, *arguments, report=report)

    def invoke(
        self,
        handle: ScriptHandle,
        target: Any,
        function: str,
        *arguments: object,
        report: bool = True,
    ) -> Any:
        """Вызвать произвольный вызываемый объект скрипта с перехватом исключений.

        Общая часть :meth:`call` и вызовов методов ``Behaviour``: ошибка
        учитывается в ``handle.errors`` и уходит в лог, наружу не пробрасывается.

        Args:
            handle: Учётная запись скрипта.
            target: Что вызвать (функция модуля, метод экземпляра, класс).
            function: Имя хука для сообщения об ошибке.
            *arguments: Аргументы вызова.
            report: Писать ли traceback в лог.

        Returns:
            Результат вызова или ``None``, если он упал.
        """
        try:
            return target(*arguments)
        except BaseException as exc:  # noqa: BLE001 - ошибку показываем пользователю
            handle.errors += 1
            error = ScriptError(
                f"ошибка {function}: {type(exc).__name__}: {exc}",
                name=handle.name,
                text=format_traceback(exc, filename=handle.source, limit=0),
            )
            if report:
                self._report(error, level=logging.ERROR)
            return None

    def call_all(
        self,
        function: str,
        *argument_factory: Any,
        report: bool = True,
    ) -> int:
        """Вызвать хук у всех активных скриптов.

        Args:
            function: Имя функции.
            *argument_factory: Функция ``token -> аргументы`` или готовые
                аргументы, общие для всех скриптов.
            report: Писать ли traceback в лог.

        Returns:
            Сколько скриптов вызвали хук.
        """
        called = 0
        for handle in list(self.scripts.values()):
            if not handle.active:
                continue
            arguments = (
                argument_factory[0](handle.token)
                if argument_factory and callable(argument_factory[0])
                else argument_factory
            )
            self.call(handle, function, *arguments, report=report)
            called += 1
        return called

    # ------------------------------------------------------------------ #
    # Горячая перезагрузка
    # ------------------------------------------------------------------ #
    @staticmethod
    def file_stamp(path: Path | None) -> float:
        """Время изменения файла (``0.0``, если файла нет).

        Args:
            path: Путь к файлу.

        Returns:
            Метка времени в секундах.
        """
        if path is None:
            return 0.0
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    def needs_reload(self, handle: ScriptHandle) -> bool:
        """Изменился ли файл скрипта после загрузки.

        Args:
            handle: Учётная запись скрипта.

        Returns:
            ``True``, если файл новее загруженной версии.
        """
        if handle.path is None:
            return False
        stamp = self.file_stamp(handle.path)
        return stamp > handle.modified + 1e-9

    def pending_reloads(self) -> list[ScriptHandle]:
        """Скрипты, файлы которых изменились и ждут перезагрузки."""
        if not self.auto_reload:
            return []
        return [handle for handle in self.scripts.values() if self.needs_reload(handle)]

    def reload(
        self,
        token: str,
        *,
        previous: dict[str, Any] | None = None,
    ) -> ScriptHandle:
        """Перезагрузить скрипт из файла, сохранив учётную запись и состояние.

        Предупреждение о возможных утечках выводится в лог: старые объекты
        функций остаются жить, пока на них ссылаются сигналы, таймеры или
        замыкания, поэтому память освобождается не сразу.

        Args:
            token: Ключ владельца.
            previous: Словарь прежней версии; по умолчанию берётся из реестра.

        Returns:
            Обновлённую учётную запись.

        Raises:
            ScriptError: Если файл не читается или не компилируется.
            KeyError: Если токен не зарегистрирован.
        """
        handle = self.scripts.get(token)
        if handle is None:
            raise KeyError(f"скрипт '{token}' не зарегистрирован")
        if handle.path is None:
            raise ScriptError(
                f"скрипт '{handle.name}' загружен из строки и не перезагружается",
                name=handle.name,
            )
        self.log.warning(
            "Горячая перезагрузка '%s': прежние функции могут остаться в памяти",
            handle.name,
        )
        source = previous if previous is not None else dict(handle.module.__dict__)
        reloaded = self.load(
            handle.path,
            token=token,
            name=handle.name,
            previous=source,
            meta=handle.meta,
        )
        reloaded.errors = handle.errors
        reloaded.active = handle.active
        return reloaded

    def refresh(
        self,
        on_destroy: Any = None,
        on_reload: Any = None,
    ) -> list[ScriptHandle]:
        """Перезагрузить все изменившиеся скрипты.

        Вызывается кадровым циклом или таймером редактора. Для каждого
        изменившегося файла сначала вызывается ``on_destroy`` (старая версия
        успевает освободить ресурсы), затем скрипт перезагружается и
        вызывается ``on_reload``.

        Args:
            on_destroy: Обработчик ``handle`` перед заменой версии.
            on_reload: Обработчик ``handle`` после успешной перезагрузки.

        Returns:
            Список успешно перезагруженных записей.
        """
        if not self.auto_reload:
            return []
        reloaded: list[ScriptHandle] = []
        for handle in self.pending_reloads():
            if callable(on_destroy):
                try:
                    on_destroy(handle)
                except Exception as exc:  # noqa: BLE001 - обработчик не должен ломать цикл
                    self.log.error("Обработчик остановки упал: %s", exc)
            try:
                updated = self.reload(handle.token)
            except ScriptError as exc:
                self._report(exc)
                continue
            except KeyError as exc:  # pragma: no cover - реестр меняется на ходу
                self.log.error("Скрипт исчез из реестра: %s", exc)
                continue
            if callable(on_reload):
                try:
                    on_reload(updated)
                except Exception as exc:  # noqa: BLE001
                    self.log.error("Обработчик перезагрузки упал: %s", exc)
            reloaded.append(updated)
        return reloaded

    # ------------------------------------------------------------------ #
    # Служебное
    # ------------------------------------------------------------------ #
    def module_name(self, path: Path | None) -> str:
        """Имя модуля для файла (использует корень проекта)."""
        if path is None:
            return f"myengine_script_{self.next_token('inline').replace('-', '_')}"
        return module_name_for(path, self.project_root)

    def _report(self, error: ScriptError, level: int = logging.ERROR) -> None:
        """Записать ошибку скрипта в лог редактора.

        Полный traceback идёт отдельной записью уровня ERROR с префиксом
        ``[script]``: консоль редактора раскрашивает его как ошибку, а краткая
        сводка остаётся в первой строке — её видно в списке сообщений.

        Args:
            error: Ошибка скрипта.
            level: Уровень логирования.
        """
        self.log.log(level, "%s", error.summary())
        if error.text:
            text = error.text
            if self.traceback_limit > 0:
                parts = text.splitlines()
                if len(parts) > self.traceback_limit:
                    hidden = len(parts) - self.traceback_limit
                    notice = f"… ещё {hidden} строк(и) скрыто"
                    text = "\n".join([*parts[: self.traceback_limit], notice])
            self.log.log(level, "[script] %s", text)

    def describe(self) -> str:
        """Сводка состояния реестра (для логов и статусной строки)."""
        active = len(self.active_scripts)
        errors = sum(handle.errors for handle in self.scripts.values())
        return (
            f"скриптов={len(self.scripts)} активных={active} ошибок={errors} "
            f"t={time.strftime('%H:%M:%S')}"
        )

    def __repr__(self) -> str:
        return f"<ScriptRuntime {self.describe()}>"
