"""Запуск редактора MyEngine: приложение, тема, язык, главное окно.

Модуль собирает всё воедино и умеет работать в двух режимах:

* обычный интерактивный запуск — ``python editor/main.py``;
* снятие скриншота интерфейса без участия пользователя —
  ``python editor/main.py --screenshot _work/editor.png --frames 40``
  (используется для проверки оформления и автоматических отчётов).
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
from engine.core.frame_budget import resolve_quality, set_current_budget
from engine.rendering.qt_context import apply_surface_format
from engine.rendering.settings import QualityTier, RenderSettings

__all__ = ["build_parser", "create_application", "main", "run"]

_log = logging.getLogger("editor.app")

#: Корень репозитория (используется как проект по умолчанию).
REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    """Собрать разбор аргументов командной строки.

    Returns:
        Готовый парсер аргументов.
    """
    parser = argparse.ArgumentParser(
        prog="myengine-editor", description="Редактор сцен MyEngine"
    )
    parser.add_argument(
        "--project",
        default=None,
        help="корень проекта (без него открывается менеджер проектов)",
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
        default=None,
        choices=[tier.value for tier in QualityTier],
        help="профиль качества (иначе quality проекта, иначе medium)",
    )
    parser.add_argument("--width", type=int, default=1480, help="ширина окна")
    parser.add_argument("--height", type=int, default=900, help="высота окна")
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
    parser.add_argument(
        "--no-import",
        action="store_true",
        help="не следить за ассетами (не создавать .meta и Library)",
    )
    parser.add_argument("--verbose", action="store_true", help="подробный лог")
    return parser


def create_application(
    argv: list[str] | None = None,
) -> tuple[QApplication, argparse.Namespace]:
    """Создать ``QApplication`` и разобрать аргументы.

    Args:
        argv: Аргументы командной строки; ``None`` — взять ``sys.argv``.

    Returns:
        Пару «приложение, разобранные аргументы».
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Тот же выбор профиля и бюджета, что у плеера (Play Mode читает проект сам).
    args.quality_cli = args.quality
    quality = resolve_quality(getattr(args, "project", None), cli=args.quality)
    args.quality = quality.tier
    set_current_budget(quality.budget)
    settings = RenderSettings.for_quality(QualityTier(args.quality))
    apply_surface_format(settings)
    application = QApplication.instance() or QApplication(sys.argv[:1])
    application.setApplicationName("MyEngine Editor")
    application.setOrganizationName("MyEngine")
    apply_theme(application)
    return application, args


def run(argv: list[str] | None = None) -> int:
    """Запустить редактор.

    Args:
        argv: Аргументы командной строки.

    Returns:
        Код возврата приложения.
    """
    application, args = create_application(argv)
    if not resolve_launch_project(args, REPO_ROOT):
        return 0
    # Проект мог выбрать менеджер проектов: бюджет кадра берём уже из него.
    set_current_budget(
        resolve_quality(args.project, cli=args.quality_cli).budget
    )
    context = EditorContext(
        project_root=Path(args.project),
        workspace_root=REPO_ROOT,
        language=args.language,
        shader_cache=not args.no_import,
    )
    window = MainWindow(context)
    if not args.no_import:
        context.start_imports()
    window.resize(args.width, args.height)
    window.show()
    _log.info("Редактор запущен: проект %s, язык %s", args.project, args.language)

    if args.screenshot:
        _schedule_screenshot(application, window, Path(args.screenshot), args.frames)

    return application.exec()


def _schedule_screenshot(
    application: QApplication, window: MainWindow, path: Path, frames: int
) -> None:
    """Сделать скриншот окна после нескольких кадров и закрыть редактор.

    Args:
        application: Приложение Qt.
        window: Главное окно.
        path: Куда сохранить PNG.
        frames: Сколько кадров рендера подождать (чтобы вьюпорт успел ожить).
    """
    counter = {"left": max(1, frames)}

    def capture() -> None:
        """Сделать снимок окна вне обработчика кадра."""
        image = window.grab().toImage()
        path.parent.mkdir(parents=True, exist_ok=True)
        if image.save(str(path), "PNG"):
            _log.info(
                "Скриншот редактора сохранён: %s (%dx%d)",
                path,
                image.width(),
                image.height(),
            )
        else:
            _log.error("Не удалось сохранить скриншот в '%s'", path)
        application.quit()

    def on_frame(_fps: float) -> None:
        """Отсчитать кадры и запланировать снимок."""
        counter["left"] -= 1
        if counter["left"] > 0:
            return
        window.viewport.viewport.stop_animation()
        # Снимок делается следующим циклом событий: grab() внутри обработчика
        # кадра приводит к рекурсивной перерисовке.
        QTimer.singleShot(0, capture)

    window.viewport.viewport.frame_rendered.connect(on_frame)
    window.viewport.viewport.start_animation(60)
    # Страховка: если вьюпорт не отдаст кадры (нет GL), выходим по таймеру.
    QTimer.singleShot(15_000, application.quit)


def main(argv: list[str] | None = None) -> int:
    """Точка входа для ``python -m editor.app``."""
    return run(argv)


if __name__ == "__main__":  # pragma: no cover - ручной запуск
    raise SystemExit(main())
