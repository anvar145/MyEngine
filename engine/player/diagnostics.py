"""Диагностика плеера: журнал, пересылка логов в редактор и crash-лог."""

from __future__ import annotations

import collections
import datetime as _dt
import logging
import os
import sys
from pathlib import Path

__all__ = ["LogForwarder", "attach_file_log", "install_logging", "write_crash_log"]

#: Сколько последних строк журнала попадает в crash-лог.
TAIL_LINES = 200


class LogForwarder(logging.Handler):
    """Копит записи журнала для отправки редактору и хранит «хвост»."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.pending: collections.deque[dict] = collections.deque(maxlen=2000)
        self.tail: collections.deque[str] = collections.deque(maxlen=TAIL_LINES)
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        """Запомнить запись."""
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - журнал не должен падать
            message = str(record.msg)
        self.tail.append(self.format(record))
        self.pending.append(
            {
                "level": record.levelname,
                "msg": message,
                "source": record.name,
                "line": record.lineno,
            }
        )

    def drain(self, limit: int = 100) -> list[dict]:
        """Забрать до ``limit`` накопленных записей."""
        batch = []
        while self.pending and len(batch) < limit:
            batch.append(self.pending.popleft())
        return batch


def install_logging(log_file: Path | str | None = None) -> LogForwarder:
    """Настроить журнал плеера.

    Args:
        log_file: Файл журнала (``--log``); ``None`` — только консоль.

    Returns:
        Обработчик, собирающий записи для редактора.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    forwarder = LogForwarder()
    root.addHandler(forwarder)
    if sys.stderr is not None:  # exe без консоли (--windowed): stderr нет
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(logging.Formatter("[player] %(levelname)s %(message)s"))
        root.addHandler(console)
    if log_file:
        attach_file_log(log_file)
    return forwarder


def attach_file_log(log_file: Path | str) -> bool:
    """Дописать в журнал файловый обработчик; ``False``, если файл недоступен."""
    path = Path(log_file)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
    except OSError:
        return False
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(handler)
    return True


def write_crash_log(
    trace: str,
    forwarder: LogForwarder | None = None,
    driver: str = "",
    log_dir: Path | None = None,
) -> Path | None:
    """Записать crash-лог в ``log_dir`` (по умолчанию ``%APPDATA%/MyEngine/logs``).

    Args:
        trace: Текст traceback.
        forwarder: Источник «хвоста» журнала.
        driver: Описание драйвера GL.
        log_dir: Каталог логов (``<user_data>/<игра>/logs`` у собранной игры).

    Returns:
        Путь к файлу или ``None``, если записать не удалось.
    """
    default = Path(os.environ.get("APPDATA") or Path.home()) / "MyEngine" / "logs"
    base = log_dir or default
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = base / f"player_{stamp}.log"
    lines = [trace, f"Драйвер: {driver or 'неизвестно'}", "", "Последние записи:"]
    if forwarder is not None:
        lines.extend(forwarder.tail)
    try:
        base.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
    except OSError:
        return None
    return path
