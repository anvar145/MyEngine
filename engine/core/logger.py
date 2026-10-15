"""Минималистичный лог ошибок для ручного тестирования.

Один файл на запуск (``logs/ГГГГ-ММ-ДД_ЧЧ-ММ-СС.log``), только WARNING и выше,
без ротации и внешних зависимостей. Необработанные исключения — в том числе в
потоках и в Qt-слотах — попадают и в файл, и в консоль.
"""

from __future__ import annotations

import importlib
import logging
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

__all__ = ["LOG_DIR", "LOG_FORMAT", "setup_logger"]

#: Корень проекта: файл лежит в ``engine/core``, то есть двумя уровнями глубже.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
#: Куда складываются логи (папка создаётся при первом запуске).
LOG_DIR = PROJECT_ROOT / "logs"
#: ``[дата время] [УРОВЕНЬ] [модуль:строка] сообщение``.
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False
_log_path: Path | None = None


def setup_logger(level: int = logging.WARNING) -> Path:
    """Завести файл лога на этот запуск и поставить перехватчики исключений.

    Идемпотентна: повторные вызовы ничего не добавляют и возвращают тот же путь.
    ``level`` — минимальный уровень записи в файл: по умолчанию WARNING, чтобы
    INFO не раздувал лог. Возвращает путь к созданному файлу.
    """
    global _configured, _log_path  # однократная настройка на процесс
    if _configured and _log_path is not None:
        return _log_path

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _log_path = LOG_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.log"
    handler = logging.FileHandler(_log_path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logging.getLogger().addHandler(handler)

    sys.excepthook = _handle_uncaught
    threading.excepthook = _handle_thread
    _install_qt_handler()
    _configured = True
    return _log_path


def _handle_uncaught(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_tb: object,
) -> None:
    """Записать необработанное исключение в лог и вывести traceback в консоль."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    logging.getLogger("myengine").critical(
        "Необработанное исключение", exc_info=(exc_type, exc_value, exc_tb)
    )
    traceback.print_exception(exc_type, exc_value, exc_tb)


def _handle_thread(args: threading.ExceptHookArgs) -> None:
    """Записать исключение из потока: движок продолжает работу."""
    if issubclass(args.exc_type, SystemExit):
        return
    name = args.thread.name if args.thread is not None else "?"
    logging.getLogger("myengine").error(
        "Исключение в потоке '%s'",
        name,
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )
    traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)


def _install_qt_handler() -> None:
    """Перехватить ``qWarning``/``qCritical`` через ``qInstallMessageHandler``.

    Qt — необязательная зависимость, поэтому импорт динамический: ядро остаётся
    headless-совместимым и не тянет GUI статически (правило слоёв, см.
    ``engine/tests/test_layering.py``). Без PySide6 функция тихо ничего не делает.
    """
    try:
        qt_core = importlib.import_module("PySide6.QtCore")
    except ImportError:  # pragma: no cover - зависит от окружения
        return

    types = qt_core.QtMsgType
    quiet_modes = (types.QtDebugMsg, types.QtInfoMsg)
    critical_modes = (types.QtCriticalMsg, types.QtFatalMsg)

    def handler(mode: object, _context: object, message: str) -> None:
        if mode in quiet_modes:
            return
        level = logging.ERROR if mode in critical_modes else logging.WARNING
        logging.getLogger("qt").log(level, "%s", message)

    qt_core.qInstallMessageHandler(handler)
