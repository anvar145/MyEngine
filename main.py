"""MyEngine — запуск редактора со встроенным редактором скриптов.

Точка входа верхнего уровня: она собирает приложение Qt, тему, контекст
редактора и главное окно, а затем **встраивает панель скриптов** в окно рядом с
консолью. Логика самих панелей живёт в :mod:`editor.panels`, здесь только
сборка и запуск.

Запуск::

    .venv\\Scripts\\python.exe main.py                # менеджер проектов
    .venv\\Scripts\\python.exe main.py --project C:\\MyGame --language en
    .venv\\Scripts\\python.exe main.py --screenshot _work\\editor.png

Полезные горячие клавиши:

* ``Ctrl+Shift+E`` — показать или скрыть редактор скриптов;
* ``Ctrl+N`` / ``Ctrl+O`` / ``Ctrl+S`` / ``Ctrl+Shift+S`` — файл скрипта;
* ``F5`` / ``Shift+F5`` / ``F6`` — запуск, остановка и перезагрузка скриптов;
* ``Ctrl+F`` / ``F3`` — поиск и замена по открытому скрипту.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from editor.context import EditorContext
from editor.i18n import DEFAULT_LANGUAGE, LANGUAGES
from editor.main_window import MainWindow
from editor.project_manager import resolve_launch_project
from editor.theme import apply_theme
from engine.core.logger import setup_logger
from engine.rendering.qt_context import apply_surface_format
from engine.rendering.settings import QualityTier, RenderSettings

# Первое, что делает main.py: заводит файл лога на этот запуск и ставит
# перехватчики необработанных исключений (см. engine/core/logger.py).
setup_logger()

__all__ = ["build_app", "build_parser", "create_context", "main", "run"]

_log = logging.getLogger("myengine.main")

#: Корень репозитория — проект по умолчанию.
REPO_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    """Собрать разбор аргументов командной строки.

    Returns:
        Готовый парсер аргументов.
    """
    parser = argparse.ArgumentParser(
        prog="myengine",
        description="MyEngine — редактор сцен со встроенным редактором скриптов",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="корень проекта (без него при запуске открывается менеджер проектов)",
    )
    parser.add_argument(
        "--no-launcher",
        action="store_true",
        help="не показывать менеджер проектов, открыть корень репозитория",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        choices=sorted(LANGUAGES),
        help="язык интерфейса при запуске",
    )
    parser.add_argument(
        "--quality",
        default=QualityTier.MEDIUM.value,
        choices=[tier.value for tier in QualityTier],
        help="профиль качества рендеринга",
    )
    parser.add_argument("--width", type=int, default=1480, help="ширина окна")
    parser.add_argument("--height", type=int, default=900, help="высота окна")
    parser.add_argument(
        "--script",
        default="",
        help="скрипт, который нужно открыть во вкладке при запуске",
    )
    parser.add_argument(
        "--no-import",
        action="store_true",
        help="не следить за ассетами (не создавать .meta и Library)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="сразу запустить скрипты после открытия окна",
    )
    parser.add_argument(
        "--screenshot",
        default="",
        help="сохранить скриншот окна в этот файл и выйти",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="сколько кадров отрисовать перед скриншотом",
    )
    parser.add_argument("--verbose", action="store_true", help="подробный лог")
    return parser


def _configure_console(verbose: bool) -> None:
    """Настроить вывод в консоль, не снимая файловый обработчик лога ошибок.

    ``logging.basicConfig`` здесь не годится: он ничего не делает, если у
    корневого логгера уже есть обработчики, — а файл лога их добавляет.

    Args:
        verbose: Включить уровень DEBUG вместо INFO.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            return
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S"
        )
    )
    root.addHandler(console)


def build_app(argv: list[str] | None = None) -> tuple[QApplication, argparse.Namespace]:
    """Создать ``QApplication``, применить тему и разобрать аргументы.

    Args:
        argv: Аргументы командной строки; ``None`` — взять ``sys.argv``.

    Returns:
        Пару «приложение, разобранные аргументы».
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_console(args.verbose)
    settings = RenderSettings.for_quality(QualityTier(args.quality))
    apply_surface_format(settings)
    application = QApplication.instance() or QApplication(sys.argv[:1])
    application.setApplicationName("MyEngine")
    application.setOrganizationName("MyEngine")
    apply_theme(application)
    return application, args


def create_context(args: argparse.Namespace) -> EditorContext:
    """Собрать контекст редактора по аргументам командной строки.

    Args:
        args: Разобранные аргументы.

    Returns:
        Контекст с проектом, языком и словарём скриптов.
    """
    return EditorContext(
        project_root=Path(args.project or REPO_ROOT),
        workspace_root=REPO_ROOT,
        language=args.language,
        shader_cache=not args.no_import,
    )


def run(argv: list[str] | None = None) -> int:
    """Запустить редактор со встроенной панелью скриптов.

    Args:
        argv: Аргументы командной строки.

    Returns:
        Код возврата приложения.
    """
    application, args = build_app(argv)
    if not resolve_launch_project(args, REPO_ROOT):
        return 0
    context = create_context(args)
    window = MainWindow(context)
    window.resize(args.width, args.height)
    if not args.no_import:
        context.start_imports()

    if args.script:
        path = Path(args.script)
        if not path.is_absolute():
            path = Path(args.project) / path
        if path.is_file():
            window.set_script_panel_visible(True)
            window.script_editor.open_script(path)
            _log.info("Открыт скрипт: %s", path)
        else:
            _log.error("Скрипт не найден: %s", path)

    window.show()
    _log.info(
        "MyEngine запущен: проект %s, язык %s, панель скриптов %s",
        args.project,
        args.language,
        "видима" if window.script_editor.isVisible() else "скрыта",
    )

    if args.run:
        # Запуск откладываем на цикл событий: окно должно быть готово.
        QTimer.singleShot(0, window.script_editor.run_current)

    if args.screenshot:
        _schedule_screenshot(application, window, Path(args.screenshot), args.frames)

    return application.exec()


def _schedule_screenshot(
    application: QApplication,
    window: MainWindow,
    path: Path,
    frames: int,
) -> None:
    """Сделать скриншот окна после нескольких кадров и закрыть редактор.

    Args:
        application: Приложение Qt.
        window: Главное окно.
        path: Куда сохранить PNG.
        frames: Сколько кадров рендера подождать.
    """
    counter = {"left": max(1, frames)}

    def capture() -> None:
        """Сделать снимок окна вне обработчика кадра."""
        image = window.grab().toImage()
        path.parent.mkdir(parents=True, exist_ok=True)
        if image.save(str(path), "PNG"):
            _log.info(
                "Скриншот сохранён: %s (%dx%d)", path, image.width(), image.height()
            )
        else:  # pragma: no cover - зависит от файловой системы
            _log.error("Не удалось сохранить скриншот в '%s'", path)
        application.quit()

    def on_frame(_fps: float) -> None:
        """Отсчитать кадры и запланировать снимок."""
        counter["left"] -= 1
        if counter["left"] > 0:
            return
        window.viewport.viewport.stop_animation()
        QTimer.singleShot(0, capture)

    window.viewport.viewport.frame_rendered.connect(on_frame)
    window.viewport.viewport.start_animation(60)
    QTimer.singleShot(15_000, application.quit)


def main(argv: list[str] | None = None) -> int:
    """Точка входа ``python main.py``."""
    return run(argv)


if __name__ == "__main__":  # pragma: no cover - ручной запуск
    raise SystemExit(main())
