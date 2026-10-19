"""Приложение плеера: окно/контекст, сцена, цикл и IPC.

Плеер не знает про Qt. Исключение — режим ``--headless``: offscreen-контекст
берётся из ``engine.rendering.qt_context`` и импортируется лениво.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from engine.audio import (
    BUSES,
    AudioSystem,
    ClipCache,
    create_backend,
    load_audio_settings,
)
from engine.audio.system import SETTINGS_FILE as AUDIO_SETTINGS
from engine.core.game_loop import GameLoop
from engine.core.input import (
    InputMap,
    InputState,
    ManualInputBackend,
    MouseState,
    default_input_map,
)
from engine.core.input.keys import KEY_F2, KEY_F5, KEY_F9, KEY_SHIFT_F5
from engine.core.platform_info import (
    is_writable_dir,
    user_cache_root,
    user_data_root,
)
from engine.core.resources import (
    NoBuildError,
    ResourceManager,
    WorkdirVFS,
    as_assets,
    detect_vfs,
)
from engine.core.save import (
    AUTOSAVE_SLOT,
    QUICK_SLOT,
    SaveManager,
    SaveStatus,
    autosave_before_switch,
    default_save_root,
    project_save_version,
)
from engine.core.save.paths import game_name
from engine.core.scene_load import PHASE_ACTIVATE, PHASE_BUILD
from engine.core.scene_manager import SceneManager
from engine.core.screen_fx import ScreenFx
from engine.core.ui_events import (
    QuitRequested,
    RenderScaleChanged,
    SceneLoadProgress,
    SceneLoadRequested,
    SceneUnloadRequested,
)
from engine.player.diagnostics import (
    LogForwarder,
    attach_file_log,
    install_logging,
    write_crash_log,
)
from engine.player.frame_pacing import FramePacer
from engine.player.input_backend import GlfwInputBackend
from engine.player.ipc_server import IpcServer
from engine.player.pak_locator import find_pak, is_frozen
from engine.player.scene_renderer import SceneRenderer
from engine.player.snapshot import SceneSnapshot
from engine.rendering import QualityTier, Renderer, RenderSettings
from engine.rendering.dynamic_resolution import (
    DynamicResolution,
    read_hdr_hint,
    read_render_hints,
)
from engine.rendering.gl_debug import get_gl
from engine.rendering.gpu_timer import GpuTimer
from engine.rendering.post import PostSettings, tier_render_scale
from engine.rendering.sky import SkyPass, SkySettings
from engine.scene_io import (
    iter_scene_from_dict,
    load_scene_full,
    read_post,
    read_scene_data,
    read_sky,
    scene_from_dict,
)
from engine.scripting.script_component import set_default_input

__all__ = ["PlayerApp", "main", "parse_args"]

_log = logging.getLogger("engine.player")

ROOT = Path(__file__).resolve().parents[2]
STATS_PERIOD = 0.2
LOG_PERIOD = 1.0 / 30.0
#: Кадры прогрева (компиляция шейдеров, загрузка), не входящие в замеры.
WARMUP_FRAMES = 5
#: Максимум фиксированных шагов за кадр: при долгом шаге физики мир замедляется,
#: а кадр не раздувается (см. docs/CORE.md §7.1).
PLAYER_MAX_SUBSTEPS = 3
#: Бюджет фиксированных шагов за кадр (мс) в оконном режиме: от сетки vsync
#: 16.7 мс остаётся запас на рендер и swap. В headless/--frames не применяется,
#: чтобы результат не зависел от скорости машины (docs/CORE.md §7.1).
PLAYER_FIXED_BUDGET_MS = 9.0
#: Максимум шагов оседания физики до первого кадра (≈4 с игрового времени).
PLAYER_SETTLE_MAX_STEPS = 240
#: Бюджет build-фазы за кадр экрана загрузки (мс): остаётся запас до 16.7 мс
#: на события окна и показ буфера.
STARTUP_LOAD_BUDGET_MS = 8.0


def _warm_imports() -> None:
    """Импортировать модули, которые сцена подтянет лениво при сборке."""
    import engine.core.physics_adapter  # noqa: F401


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки плеера."""
    parser = argparse.ArgumentParser(
        prog="engine.player", description="Плеер MyEngine (Play Mode)"
    )
    parser.add_argument(
        "--project",
        default=None,
        help="каталог проекта (без него: каталог пака или exe собранной игры)",
    )
    parser.add_argument("--scene", default=None, help="путь, имя или GUID сцены")
    parser.add_argument(
        "--additive",
        action="append",
        default=[],
        help="сцена, загружаемая рядом с основной (можно несколько раз)",
    )
    parser.add_argument("--pak", default=None, help=".pak с артефактами Library")
    parser.add_argument(
        "--use-library",
        action="store_true",
        help="читать Library/ проекта, даже если есть собранный пак",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--windowed", action="store_true", help="оконный режим")
    mode.add_argument("--fullscreen", action="store_true", help="на весь экран")
    parser.add_argument("--rpc-port", type=int, default=None, help="порт IPC (0=любой)")
    parser.add_argument(
        "--quality",
        default=None,
        choices=[tier.value for tier in QualityTier],
        help="профиль качества (иначе — build.json пака, иначе medium)",
    )
    parser.add_argument(
        "--user-dirs",
        action="store_true",
        help="писать кэш и логи в каталоги пользователя (как собранная игра)",
    )
    parser.add_argument(
        "--render-scale",
        type=float,
        default=None,
        help="масштаб разрешения 0.25..1 (иначе — подсказка проекта/профиль)",
    )
    parser.add_argument(
        "--no-dynamic-res",
        action="store_true",
        help="выключить динамическое разрешение",
    )
    parser.add_argument(
        "--hdr",
        default=None,
        choices=["on", "off"],
        help="HDR и пост-обработка (иначе — поле hdr проекта)",
    )
    parser.add_argument(
        "--fade",
        type=float,
        default=None,
        help="сразу затемнить кадр чёрным до непрозрачности 0..1 (проверка fade)",
    )
    parser.add_argument("--frames", type=int, default=0, help="выйти через N кадров")
    parser.add_argument("--headless", action="store_true", help="без окна")
    parser.add_argument("--log", default=None, help="файл журнала")
    parser.add_argument("--dump-state", default=None, help="JSON с состоянием сцены")
    parser.add_argument("--stats-json", default=None, help="JSON со статистикой")
    parser.add_argument("--snapshot", default=None, help="снимок сцены (JSON)")
    parser.add_argument("--screenshot", default=None, help="PNG последнего кадра")
    parser.add_argument(
        "--mouse-at",
        default=None,
        metavar="X,Y",
        help="(headless) поставить курсор в пиксели X,Y до первого кадра",
    )
    parser.add_argument(
        "--click-at",
        default=None,
        metavar="X,Y",
        help="(headless) клик левой кнопкой в пикселях X,Y на втором кадре",
    )
    parser.add_argument(
        "--hold-keys",
        default="",
        metavar="W,SPACE",
        help="(headless) клавиши, считающиеся нажатыми все кадры",
    )
    parser.add_argument(
        "--input-map",
        default=None,
        metavar="PATH",
        help="JSON маппинга действий ввода (формат InputMap.save)",
    )
    parser.add_argument(
        "--saves-dir", default=None, metavar="PATH", help="каталог сохранений"
    )
    parser.add_argument(
        "--use-project-saves",
        action="store_true",
        help="сохранения в <проект>/Saves вместо каталога данных пользователя",
    )
    parser.add_argument(
        "--no-autosave", action="store_true", help="отключить автосохранение"
    )
    parser.add_argument(
        "--quick-save-at",
        type=int,
        default=0,
        metavar="N",
        help="(отладка) быстрое сохранение (F5) на кадре N",
    )
    parser.add_argument(
        "--quick-load-at",
        type=int,
        default=0,
        metavar="N",
        help="(отладка) быстрая загрузка (F9) на кадре N",
    )
    parser.add_argument(
        "--show-colliders",
        action="store_true",
        help="показывать коллайдеры физики (переключается F2)",
    )
    parser.add_argument(
        "--frame-trace",
        default=None,
        metavar="PATH",
        help="JSON с пофазной трассой каждого кадра (стенд frame pacing)",
    )
    parser.add_argument(
        "--closed-loop",
        action="store_true",
        help="(headless) реальное dt кадра и бюджет физики, как в окне",
    )
    parser.add_argument(
        "--present-hz",
        type=float,
        default=0.0,
        help="(headless, с --closed-loop) эмуляция vsync: кадр ждёт границы 1/Hz",
    )
    parser.add_argument(
        "--sync-load",
        action="store_true",
        help="загружать стартовую сцену синхронно, до окна (тесты, отладка)",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-fps", type=float, default=0.0, help="лимит FPS")
    parser.add_argument(
        "--no-audio", action="store_true", help="не открывать аудиоустройство"
    )
    parser.add_argument(
        "--crash-after", type=int, default=0, help="(отладка) упасть через N кадров"
    )
    parser.add_argument(
        "--gl-debug",
        action="store_true",
        help="оставить проверку ошибок PyOpenGL (медленнее)",
    )
    args = parser.parse_args(argv)
    if args.project is None:
        if args.pak:
            args.project = str(Path(args.pak).resolve().parent)
        elif is_frozen():
            args.project = str(Path(sys.executable).resolve().parent)
        else:
            parser.error("нужен --project (или --pak)")
    return args


def _pick_dir(project: Path, *names: str) -> Path:
    for name in names:
        candidate = project / name
        if candidate.is_dir():
            return candidate
    return project / names[0]


def _plain(value: Any) -> Any:
    """Привести значение к JSON-совместимому виду или вернуть ``None``."""
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (list, tuple)) and len(value) < 64:
        items = [_plain(item) for item in value]
        return items if all(item is not None for item in items) else None
    return None


class PlayerApp:
    """Состояние и главный цикл плеера.

    Args:
        args: Разобранные аргументы командной строки.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.project = Path(args.project).resolve()
        self.forwarder: LogForwarder = install_logging(args.log)
        self.assets_dir = _pick_dir(self.project, "Assets", "assets")
        self.library_dir = self.project / "Library"
        self.db: Any = None
        self.vfs: Any = None
        self.scene = None
        # VFS нужен уже здесь: настройки, ``*.project.json`` и сцены собранной
        # игры лежат в паке, а не на диске.
        self.open_assets()
        #: Параметры сборки из ``build.json`` пака (``quality``, ``hdr``, ...).
        self.build_info = self._read_build_info()
        if args.quality is None:
            args.quality = str(self.build_info.get("quality") or "medium")
        #: Собранная игра пишет кэш и логи в каталоги пользователя, не рядом с exe.
        self.user_dirs = bool(args.user_dirs) or is_frozen()
        self.game_name = game_name(self.project, self.vfs)
        if self.user_dirs and not args.log:
            logs = user_data_root(self.game_name) / "logs"
            if not attach_file_log(logs / "player.log"):
                _log.warning("Каталог логов недоступен для записи: %s", logs)
        #: Аудио: бэкенд (создаётся в ``run``), кэш клипов и громкости шин.
        self.audio_backend: Any = None
        self.audio_cache = ClipCache()
        self.audio_gains: dict[str, float] = dict.fromkeys(BUSES, 1.0)
        self.mouse = MouseState()
        #: Ввод, общий для всех скриптов сцены (клавиатура окна → имена).
        self.input = InputState()
        self.input.input_map = self._load_input_map()
        self.input.settings_path = self.project / "Settings" / "input.json"
        #: Колбэки окна GLFW → ``self.input``.
        self.glfw_input = GlfwInputBackend(self.input)
        #: Инъекция клавиш (``--hold-keys``, IPC, тесты).
        self.manual_input = ManualInputBackend(self.input)
        self.show_colliders = bool(getattr(args, "show_colliders", False))
        self._click_at: tuple[float, float] | None = None
        self.save_manager = self._make_save_manager()
        self.scenes = SceneManager(
            self._load_file,
            parse=self._read_scene,
            build=self._build_scene,
            root=self.project,
            exists=self._file_exists,
        )
        self.scenes.before_activate.append(autosave_before_switch)
        self.scenes.events.subscribe(SceneLoadProgress, self._forward_progress)
        self.scenes.events.subscribe(SceneLoadProgress, self._log_progress)
        #: Последняя записанная в журнал точка прогресса: путь -> (фаза, четверть).
        self._progress_logged: dict[str, tuple[str, int]] = {}
        #: Длительности кадров стартовой загрузки (мс), для ``--frame-trace``.
        self._loading_frames: list[float] = []
        self._load_steps: list[float] = []
        #: Отложенные до конца кадра действия сохранений: ``save``/``load``.
        self._pending_quick: list[str] = []
        self._status_object: Any = None
        self._status_left = 0.0
        self._pending_scene: str | None = None
        #: Очередь ``(загрузить?, путь)`` для дополнительных сцен.
        self._pending_additive: list[tuple[bool, str]] = []
        self.sky = SkySettings()
        self.sky_pass: SkyPass | None = None
        #: Секция ``post`` активной сцены (``None`` — умолчания профиля).
        self.scene_post: PostSettings | None = None
        #: Общий экран-эффект: fade переживает смену сцены.
        self.screen_fx = ScreenFx()
        if args.fade is not None:
            self.screen_fx.fade_to((0.0, 0.0, 0.0), args.fade, 0.0)
            self.screen_fx.update(0.0)
        self.dynres: DynamicResolution | None = None
        self.gpu_timer = GpuTimer()
        self._post_hdr = False
        self.renderer: Renderer | None = None
        self.scene_renderer: SceneRenderer | None = None
        self.loop: GameLoop | None = None
        self.server: IpcServer | None = None
        self.window: Any = None
        self.target: Any = None
        self.paused = False
        self.stop_requested = False
        self.step_frames = 0
        self.frame = 0
        self._script_ms = 0.0
        #: Трасса кадров для ``--frame-trace`` (стенд frame pacing).
        self._trace: list[dict[str, float]] = []
        self._last_stats: Any = None
        self._samples: list[tuple[float, float, float]] = []
        self._stats_clock = 0.0
        self._log_clock = 0.0
        self._fps_frames = 0
        self._fps_clock = time.perf_counter()
        self._fps = 0.0
        self._cpu_ms = 0.0
        self._t_warm = self._t_last = time.perf_counter()
        self._exit_code = 0

    # ------------------------------------------------------------------ #
    # Загрузка
    # ------------------------------------------------------------------ #
    def _read_build_info(self) -> dict[str, Any]:
        """``build.json`` из пака (пустой словарь, если его нет или он битый)."""
        text = self._pak_text(self.project / "build.json")
        if text is None:
            return {}
        try:
            data = json.loads(text)
        except ValueError as error:
            _log.warning("build.json пропущен: %s", error)
            return {}
        return data if isinstance(data, dict) else {}

    def _render_hints(self) -> tuple[float | None, bool | None, bool]:
        """Подсказки рендера: проект, затем поля ``build.json`` поверх."""
        hint, dynamic, hdr = read_render_hints(self.project, self.vfs)
        info = self.build_info
        if isinstance(info.get("dynamic_resolution"), bool):
            dynamic = info["dynamic_resolution"]
        if isinstance(info.get("hdr"), bool):
            hdr = info["hdr"]
        return hint, dynamic, hdr

    def _pak_extract_root(self) -> Path | None:
        """Где распаковывать артефакты пака: user-cache (не системный temp)."""
        if not self.user_dirs:
            return None
        root = user_cache_root(self.game_name) / "pak_extract"
        return root if is_writable_dir(root) else None

    def _shader_cache_dir(self) -> Path | None:
        """Каталог кэша шейдеров; ``None`` — кэш выключен (нет прав на запись)."""
        if not self.user_dirs:
            return self.library_dir / "ShaderCache"
        cache = user_cache_root(self.game_name) / "shader_cache"
        if is_writable_dir(cache):
            return cache
        _log.warning("Кэш шейдеров отключён: нет доступа к %s", cache)
        return None

    def _project_key(self, path: Path) -> str | None:
        """Ключ файла в паке (путь от корня проекта) или ``None``, если вне проекта."""
        try:
            return path.relative_to(self.project).as_posix()
        except ValueError:
            return None

    def _pak_text(self, path: Path) -> str | None:
        """Текст файла проекта из VFS (пак) или ``None``."""
        key = self._project_key(path)
        if key is None or self.vfs is None:
            return None
        data = self.vfs.read_file(key)
        return None if data is None else data.decode("utf-8-sig")

    def _settings_text(self, path: Path) -> str | None:
        """Как :meth:`_pak_text`, но файл на диске (правка игрока) главнее пака."""
        return None if path.is_file() else self._pak_text(path)

    def _read_scene(self, path: Path) -> Any:
        """Прочитать данные сцены (из пака или с диска)."""
        return read_scene_data(path, self._pak_text)

    def _file_exists(self, path: Path) -> bool:
        """Есть ли файл на диске или в паке."""
        if path.is_file():
            return True
        key = self._project_key(path)
        return (
            key is not None
            and self.vfs is not None
            and key in self.vfs.list_files(key)
        )

    def _scene_files(self) -> list[Path]:
        """Все ``*.scene.json`` проекта: с диска и из пака."""
        found: set[Path] = set()
        if self.assets_dir.is_dir():
            found.update(self.assets_dir.rglob("*.scene.json"))
        if self.vfs is not None:
            for key in self.vfs.list_files(f"{self.assets_dir.name}/"):
                if key.endswith(".scene.json"):
                    found.add(self.project / key)
        return sorted(found)

    def open_assets(self) -> None:
        """Открыть базу ассетов проекта (без записи ``.meta``).

        Повторный вызов ничего не делает. Собранная игра берёт ``data.pak``
        (:func:`find_pak`) и работает без каталога ``Assets/``.
        """
        if self.vfs is not None:
            return
        pak = find_pak(self.args.pak, self.project)
        if pak is None and not self.assets_dir.is_dir():
            return
        try:
            self.vfs = detect_vfs(
                self.project,
                pak=pak,
                use_library=self.args.use_library,
                assets_dir=self.assets_dir,
                library=self.library_dir,
                pak_cache_root=self._pak_extract_root,
            )
        except NoBuildError as error:
            # Сцены на встроенных примитивах работают и без импорта.
            _log.warning(
                "%s. Режим разработки: ассеты читаются из рабочего каталога. "
                "В релизной сборке (нет .pak и нет Library/) игра не запустится.",
                error,
            )
            self.vfs = WorkdirVFS(self.assets_dir, self.library_dir)
        self.db = as_assets(self.vfs, self.assets_dir)

    def init_audio(self) -> None:
        """Создать аудио-бэкенд: ``Null`` при ``--headless``/``--no-audio``.

        Громкости шин берутся из ``Settings/audio.json`` проекта. Нет
        устройства/пакета - предупреждение и тишина, игра продолжает работу.
        """
        silent = self.args.headless or self.args.no_audio
        self.audio_backend = create_backend("null" if silent else "auto")
        settings = load_audio_settings(
            self.project / AUDIO_SETTINGS, self._settings_text
        )
        for bus, value in settings.items():
            self.audio_gains[bus] = value
            if bus == "master":
                self.audio_backend.set_master_gain(value)
            else:
                self.audio_backend.set_bus_gain(bus, value)

    def _attach_audio(self, scene: Any) -> None:
        """Добавить в сцену ``AudioSystem`` (общий бэкенд, кэш и громкости)."""
        if self.audio_backend is None:
            return
        scene.add_system(
            AudioSystem(
                self.audio_backend,
                self.vfs,
                self.audio_cache,
                bus_gains=self.audio_gains,
            )
        )

    def _make_save_manager(self) -> SaveManager:
        """Менеджер сохранений: ``--saves-dir`` > ``--use-project-saves`` > ОС."""
        args = self.args
        if args.saves_dir:
            root = Path(args.saves_dir)
        else:
            root = default_save_root(
                self.project, player=not args.use_project_saves, vfs=self.vfs
            )
        manager = SaveManager(
            root, version=project_save_version(self.project, self.vfs)
        )
        manager.autosave_enabled = not args.no_autosave
        return manager

    def _scene_path(self) -> Path | None:
        spec = self.args.scene
        if not spec:
            found = self._scene_files()
            return found[0] if found else None
        return self._resolve_scene(spec)

    def _hdr_choice(self) -> bool | None:
        """Явное решение автора про HDR (``build.json``, затем проект) или ``None``."""
        value = self.build_info.get("hdr")
        if isinstance(value, bool):
            return value
        return read_hdr_hint(self.project, self.vfs)

    def _startup_scene_has_post(self) -> bool:
        """Есть ли в стартовой сцене секция ``post`` (тогда нужен HDR-конвейер)."""
        try:
            path = self._scene_path()
            if path is None:
                return False
            return read_post(self._read_scene(path)) is not None
        except (OSError, ValueError, KeyError, FileNotFoundError):
            return False

    def _resolve_scene(self, spec: str) -> Path:
        """Найти файл сцены по пути, имени или GUID."""
        candidate = Path(spec)
        names = {candidate.name, f"{candidate.name}.scene.json"}
        for option in (
            candidate,
            self.project / spec,
            self.assets_dir / spec,
            self.assets_dir / "Scenes" / spec,
            *(p for p in self._scene_files() if p.name in names),
        ):
            if self._file_exists(option):
                return option
        if self.db is not None:
            handle = self.db.get(spec)
            if handle is not None:
                return handle.path
        raise FileNotFoundError(f"Сцена '{spec}' не найдена")

    def _load_file(self, path: Path) -> Any:
        """Загрузчик для :class:`SceneManager` (небо запоминается отдельно)."""
        scene, self.sky, self.scene_post = load_scene_full(
            path, self.db, self._pak_text
        )
        self._attach_audio(scene)
        return scene

    def _build_scene(self, data: Any, path: Path) -> Any:
        """Порционная сборка сцены для :class:`SceneManager` (небо запоминается)."""
        self.sky = read_sky(data)
        self.scene_post = read_post(data)
        scene = yield from iter_scene_from_dict(data, self.db)
        self._attach_audio(scene)
        _log.info("Сцена загружена: %s (%d объектов)", path, len(scene))
        return scene

    def _log_progress(self, event: SceneLoadProgress) -> None:
        """Записать в журнал смену фазы и каждую четверть build (без спама)."""
        quarter = 0
        if event.phase == PHASE_BUILD:
            local = (event.progress - 0.05) / 0.9
            quarter = min(3, max(0, int(local * 4)))
        point = (event.phase, quarter)
        if self._progress_logged.get(event.path) == point:
            return
        self._progress_logged[event.path] = point
        _log.info(
            "Загрузка сцены %s: %s, %d%%",
            Path(event.path).name,
            event.phase,
            round(event.progress * 100),
        )
        if event.phase == PHASE_ACTIVATE:
            self._progress_logged.pop(event.path, None)

    def _load_with_pump(self, target: Path) -> Any:
        """Загрузить сцену порциями, между шагами показывая чёрный экран."""
        operation = self.scenes.load_async(target)
        while True:
            started = time.perf_counter()
            done = operation.step(STARTUP_LOAD_BUDGET_MS)
            self._load_steps.append((time.perf_counter() - started) * 1000.0)
            if done:
                return operation.result()
            self._loading_frame()

    def _loading_frame(self) -> None:
        """Кадр экрана загрузки: события окна, чёрный фон, показ буфера."""
        started = time.perf_counter()
        window = self.window
        if window is not None:
            window.poll_events()
            gl = get_gl()
            gl.glClearColor(0.0, 0.0, 0.0, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            window.swap_buffers()
        self._loading_frames.append((time.perf_counter() - started) * 1000.0)

    def load_scene(
        self, path: Path | None = None, *, incremental: bool = False
    ) -> None:
        """Загрузить сцену из снимка или файла (скрипты стартуют вместе с ней).

        Args:
            path: Файл сцены; ``None`` — из ``--snapshot`` или первая в проекте.
            incremental: Грузить порциями между кадрами экрана загрузки (старт
                плеера); иначе целиком за один вызов.
        """
        from engine.scripting import ScriptRuntime
        from engine.scripting.script_component import set_default_runtime

        runtime = ScriptRuntime(project_root=self.project, auto_reload=False)
        if self.vfs is not None:
            runtime.set_source_resolver(self.vfs.read_file)
        set_default_runtime(runtime)
        set_default_input(self.input)
        self.input.contexts.reset()
        self.script_runtime = runtime
        old = self.scene
        if self.args.snapshot and path is None:
            snapshot = SceneSnapshot.load(self.args.snapshot)
            self.scene = scene_from_dict(snapshot.data, self.db)
            self.sky = read_sky(snapshot.data)
            self.scene_post = read_post(snapshot.data)
            if old is not None:
                old.shutdown()
        else:
            target = path or self._scene_path()
            if target is None:
                raise FileNotFoundError("В проекте нет сцен (*.scene.json)")
            if incremental:
                self.scene = self._load_with_pump(target)
            else:
                self.scene = self.scenes.load(target)
            for other in self.scenes.loaded:
                if other != target:
                    self.scenes.unload(other)
            _log.info("Сцена загружена: %s", target.name)
        self._subscribe_scene(self.scene)
        if self.sky_pass is not None:
            self.sky_pass.settings = self.sky
        self._apply_post()
        if self.scene_renderer is not None:
            self.scene_renderer.set_scene(self.scene)
            self._sync_extra_scenes()
            self._apply_collider_debug()
            self.scene_renderer.preload(self.viewport())

    def load_additive(self, path: Path) -> Any:
        """Загрузить сцену рядом с активной (обе обновляются и рисуются).

        Args:
            path: Файл сцены.

        Returns:
            Загруженная сцена.
        """
        sky, post = self.sky, self.scene_post  # небо и пост задаёт основная сцена
        scene = self.scenes.load(path, additive=True)
        self.sky, self.scene_post = sky, post
        self._subscribe_scene(scene)
        self._sync_extra_scenes()
        _log.info("Дополнительная сцена загружена: %s", Path(path).name)
        return scene

    def unload_additive(self, path: Path) -> None:
        """Выгрузить дополнительную сцену (активная не затрагивается)."""
        if Path(path) in self.scenes.additive_scenes:
            self.scenes.unload(Path(path))
            self._sync_extra_scenes()

    def _sync_extra_scenes(self) -> None:
        if self.scene_renderer is not None:
            self.scene_renderer.set_extra_scenes(
                list(self.scenes.additive_scenes.values())
            )

    def _apply_collider_debug(self) -> None:
        """Передать рендеру флаг показа коллайдеров (если он умеет)."""
        setter = getattr(self.scene_renderer, "set_physics_debug", None)
        if setter is not None:
            setter(self.show_colliders)

    def toggle_colliders(self) -> bool:
        """Переключить показ коллайдеров (F2); возвращает новое состояние."""
        self.show_colliders = not self.show_colliders
        self._apply_collider_debug()
        return self.show_colliders

    def _load_input_map(self) -> InputMap:
        """Маппинг из ``--input-map``, иначе ``Settings/input.json`` проекта.

        Raises:
            ValueError: Явно заданный ``--input-map`` не читается.
        """
        explicit = getattr(self.args, "input_map", None)
        if explicit:
            return InputMap.load(Path(explicit))
        user = self.project / "Settings" / "input.json"
        if self._file_exists(user):
            try:
                return InputMap.load(user, reader=self._settings_text)
            except ValueError as error:
                _log.warning("Settings/input.json пропущен: %s", error)
        return default_input_map()

    def _pump_keyboard(self) -> None:
        """Хоткеи плеера и инъекция ``--hold-keys`` (клавиши окна идут колбэками)."""
        window = self.window
        if window is not None:
            events, window.key_events = window.key_events, []
            if KEY_F2 in events:
                self.toggle_colliders()
            if KEY_SHIFT_F5 in events:
                _log.info("Shift+F5: остановка игры")
                self.stop_requested = True
            if KEY_F5 in events:
                self._pending_quick.append("save")
            if KEY_F9 in events:
                self._pending_quick.append("load")
        args = self.args
        if args.quick_save_at and self.frame == args.quick_save_at:
            self._pending_quick.append("save")
        if args.quick_load_at and self.frame == args.quick_load_at:
            self._pending_quick.append("load")
        hold = str(getattr(self.args, "hold_keys", "") or "")
        keys = [k.strip() for k in hold.split(",") if k.strip()]
        if keys or self.manual_input._held:
            self.manual_input.hold(keys)

    def _forward_progress(self, event: SceneLoadProgress) -> None:
        """Передать прогресс загрузки в шины загруженных сцен (экран загрузки)."""
        for scene in self.scenes.all_scenes:
            scene.events.emit(event)
        if self.server is not None:
            self.server.event(
                "scene_load_progress",
                progress=round(event.progress, 3),
                phase=event.phase,
            )
            if event.phase == PHASE_ACTIVATE:
                self.server.event("scene_loaded", path=Path(event.path).name)

    def _subscribe_scene(self, scene: Any) -> None:
        """Принимать запросы UI-переходов и выхода от сцены."""

        def on_load(event: SceneLoadRequested) -> None:
            if event.additive:
                self._pending_additive.append((True, event.path))
            else:
                self._pending_scene = event.path

        def on_unload(event: SceneUnloadRequested) -> None:
            self._pending_additive.append((False, event.path))

        def on_quit(_event: QuitRequested) -> None:
            self.stop_requested = True

        scene.save_manager = self.save_manager
        scene.screen = self.screen_fx
        scene.events.subscribe(SaveStatus, self._on_save_status)

        scene.events.subscribe(SceneLoadRequested, on_load)
        scene.events.subscribe(SceneUnloadRequested, on_unload)
        scene.events.subscribe(QuitRequested, on_quit)

    def inject_mouse(
        self, x: float, y: float, button: str | None = None, down: bool = False
    ) -> None:
        """Подать мышь без окна (headless, IPC, тесты).

        Args:
            x: Позиция в пикселях цели.
            y: Позиция в пикселях цели.
            button: Имя кнопки для нажатия/отпускания, ``None`` — только сдвиг.
            down: Нажата ли кнопка.
        """
        self.mouse.move(x, y)
        if button is not None:
            self.mouse.button(button, down)

    def _run_quick(self) -> None:
        """Быстрое сохранение/загрузка, запрошенные клавишей или IPC."""
        actions, self._pending_quick = self._pending_quick, []
        for action in actions:
            if action == "save":
                self.scene.quick_save()
            else:
                self._quick_load()

    def _quick_load(self) -> None:
        """F9: слот ``_quick`` (иначе ``_autosave``); набор сцен как в слоте."""
        manager = self.save_manager
        slot = next(
            (s for s in (manager.slot(QUICK_SLOT), manager.slot(AUTOSAVE_SLOT)) if s),
            None,
        )
        if slot is not None:
            self._restore_scene_set(slot.meta.scenes)
        self.scene.quick_load()

    def _restore_scene_set(self, refs: Any) -> None:
        """Привести загруженные сцены к ``meta.scenes`` слота (порядок сохраняется)."""
        files: list[Path] = []
        for ref in refs:
            file = self.scenes.resolve_path(ref.path) or self._scene_by_name(ref.path)
            if file is None:
                _log.warning("Сцена слота не найдена: %s", ref.path)
                return
            files.append(file)
        main, extras = files[0], files[1:]
        active = self.scenes.active_path
        current = [p for p in self.scenes.loaded if p != active]
        if active == main and current == extras:
            return
        if active != main:
            self.load_scene(main)
        else:
            for path in current:
                self.unload_additive(path)
        for path in extras:
            self.load_additive(path)

    def _scene_by_name(self, name: str) -> Path | None:
        """Файл сцены по имени из метаданных слота (без учёта регистра и ``_ -``)."""

        def norm(text: str) -> str:
            return text.lower().replace("_", "").replace("-", "").replace(" ", "")

        for path in self._scene_files():
            if norm(path.name.removesuffix(".scene.json")) == norm(name):
                return path
        return None

    def _on_save_status(self, event: SaveStatus) -> None:
        """Показать короткое сообщение в правом верхнем углу на 1 секунду."""
        from engine.core.ui_components import UIText

        scene = self.scene
        obj = self._status_object
        if obj is None or obj.scene is not scene or obj.is_destroyed:
            obj = scene.create_game_object("__SaveStatus")
            element = obj.add_component(
                UIText,
                text=event.text,
                anchor="top_right",
                pivot=(1.0, 0.0),
                position=(-16.0, 16.0),
                sorting_layer=1000,
            )
            element.text.size_px = 22
            self._status_object = obj
        element = obj.get_component(UIText)
        element.text.text = event.text
        element.visible = True
        self._status_left = 1.0

    def _tick_status(self, delta: float) -> None:
        if self._status_left <= 0.0:
            return
        self._status_left -= delta
        if self._status_left <= 0.0 and self._status_object is not None:
            from engine.core.ui_components import UIText

            element = self._status_object.get_component(UIText)
            if element is not None:
                element.visible = False

    def _apply_pending_scene(self) -> None:
        """Выполнить переход, запрошенный UI (в конце кадра)."""
        if self._pending_quick:
            self._run_quick()
        queue, self._pending_additive = self._pending_additive, []
        for load, extra in queue:
            try:
                target = self._resolve_scene(extra)
                if load:
                    self.load_additive(target)
                else:
                    self.unload_additive(target)
            except Exception as error:  # noqa: BLE001 - плохой путь не роняет игру
                _log.error("Дополнительная сцена '%s': %s", extra, error)
        spec, self._pending_scene = self._pending_scene, None
        if spec is None:
            return
        try:
            self.load_scene(self._resolve_scene(spec))
        except Exception as error:  # noqa: BLE001 - плохой путь не роняет игру
            _log.error("Переход на сцену '%s': %s", spec, error)

    # ------------------------------------------------------------------ #
    # Подготовка рендера
    # ------------------------------------------------------------------ #
    def _make_renderer(self, settings: RenderSettings) -> None:
        resources = ResourceManager(
            roots=[ROOT / "assets", self.project], name="Player"
        )
        cache = self._shader_cache_dir()
        self.renderer = Renderer(
            resources=resources,
            settings=settings,
            shader_root=ROOT / "assets",
            program_cache_dir=cache,
        )
        self.renderer.initialize()
        _log.info("Драйвер: %s", self.renderer.info.describe())
        self.sky_pass = SkyPass(self.sky)
        self.renderer.set_sky(self.sky_pass)
        if self.window is not None:
            self.mouse = self.window.mouse
        self.scene_renderer = SceneRenderer(
            self.renderer, self.scene, self.db, self.project, mouse=self.mouse
        )
        self._post_hdr = bool(
            self.renderer.hdr_enabled and self.renderer.capabilities.supports_hdr_target
        )
        self.scene_renderer.post = self._post_hdr
        self._init_render_scale()
        self._apply_post()
        self._sync_extra_scenes()
        self._apply_collider_debug()

    def _init_render_scale(self) -> None:
        """Выбрать начальный ``render_scale`` и включить динамическое разрешение."""
        assert self.renderer is not None
        tier = QualityTier(self.args.quality)
        hint, dynamic, _hdr = self._render_hints()
        if self.args.render_scale is not None:
            hint = self.args.render_scale
        if self.args.no_dynamic_res:
            dynamic = False
        hints = tier_render_scale(tier, hint=hint, dynamic=dynamic)
        if not self._post_hdr:
            hints.dynamic = False  # масштаб применяется только в HDR-пути
            hints.scale = 1.0
        self.dynres = DynamicResolution(hints)
        self.renderer.render_scale = self.dynres.scale
        if self._post_hdr:
            _log.info(
                "render_scale %.2f (dynamic=%s, %.2f..%.2f)",
                self.dynres.scale, hints.dynamic, hints.floor, hints.ceiling,
            )

    def _apply_post(self) -> None:
        """Передать рендереру пост-настройки сцены и общий ``ScreenFx``."""
        if self.renderer is None:
            return
        self.renderer.post_settings = self.scene_post or PostSettings.for_quality(
            QualityTier(self.args.quality)
        )
        self.renderer.screen_fx = self.screen_fx
        self.renderer.post.assets = self.db

    def _update_dynamic_resolution(self, cpu_ms: float) -> None:
        """Подстроить масштаб по времени кадра (GPU через запрос, иначе CPU)."""
        dyn = self.dynres
        if dyn is None or not dyn.enabled or self.renderer is None:
            return
        gpu_ms = self.gpu_timer.poll()
        if gpu_ms is None and not self.gpu_timer.available:
            gpu_ms = 0.0
        if gpu_ms is None:
            return
        new = dyn.update(max(gpu_ms, cpu_ms))
        if new is None:
            return
        old = self.renderer.render_scale
        self.renderer.render_scale = new
        _log.info("render_scale %.2f -> %.2f", old, new)
        if self.scene is not None:
            self.scene.events.emit(RenderScaleChanged(old=old, new=new))

    def _warm_up(self) -> None:
        """Прогреть рендер до первого показа: ресурсы сцены и пробный кадр.

        Драйвер доделывает программы при первом ``draw`` (небо ≈200 мс на
        HD 520), поэтому один кадр рисуется заранее, до первого ``swap``.
        """
        assert self.scene_renderer is not None
        self._settle_physics()
        started = time.perf_counter()
        viewport = self.viewport()
        self.scene_renderer.preload(viewport)
        self.scene_renderer.render(viewport, self.target)
        _log.info("Прогрев рендера: %.0f мс", (time.perf_counter() - started) * 1000)

    def _settle_physics(self) -> None:
        """Дать динамическим телам осесть до первого кадра (кроме детерминированного
        режима ``--frames N`` без ``--closed-loop``; ``--headless`` осаживает тела)."""
        if self.args.frames and not self.args.closed_loop:
            return
        physics = self.scene.physics if self.scene is not None else None
        if physics is None:
            return
        started = time.perf_counter()
        steps = physics.settle(PLAYER_SETTLE_MAX_STEPS)
        if steps:
            _log.info(
                "Физика осела за %d шагов (%.0f мс)",
                steps, (time.perf_counter() - started) * 1000,
            )

    def _make_loop(self) -> None:
        deterministic = (
            self.args.headless or bool(self.args.frames)
        ) and not self.args.closed_loop
        loop = GameLoop(
            scene=None,
            continue_on_error=True,
            max_substeps=PLAYER_MAX_SUBSTEPS,
            max_fixed_ms=None if deterministic else PLAYER_FIXED_BUDGET_MS,
        )
        self.loop = loop

        def timed(func: Any) -> Any:
            def hook(dt: float) -> None:
                started = time.perf_counter()
                getattr(self.scene, func)(dt)
                for extra in list(self.scenes.additive_scenes.values()):
                    getattr(extra, func)(dt)
                self._script_ms += (time.perf_counter() - started) * 1000.0

            return hook

        loop.add_fixed_update(timed("fixed_update"))
        loop.add_update(timed("update"))
        loop.add_late_update(timed("late_update"))

    # ------------------------------------------------------------------ #
    # IPC
    # ------------------------------------------------------------------ #
    def start_ipc(self) -> None:
        """Запустить сервер команд, если задан ``--rpc-port``."""
        if self.args.rpc_port is None:
            return
        self.server = IpcServer(self.args.rpc_port)
        self.server.start(
            {
                "pid": os.getpid(),
                "capabilities": {"stats": True, "pick": True, "step": True},
            }
        )
        print(f"RPC_PORT {self.server.port}", flush=True)

    def handle_commands(self) -> None:
        """Выполнить накопленные команды редактора."""
        if self.server is None:
            return
        for command in self.server.poll():
            name = command.get("cmd")
            try:
                self._dispatch(name, command)
            except Exception as error:  # noqa: BLE001 - плохая команда не роняет плеер
                _log.error("Команда %s: %s", name, error)
                self.server.send({"id": command.get("id"), "error": str(error)})

    def _dispatch(self, name: str | None, command: dict[str, Any]) -> None:
        server = self.server
        assert server is not None
        if name == "pause":
            self.paused = True
        elif name == "resume":
            self.paused = False
        elif name == "step":
            self.paused = True
            self.step_frames += max(1, int(command.get("frames", 1)))
        elif name == "stop":
            self.stop_requested = True
        elif name == "pick":
            result = self.pick(float(command["x"]), float(command["y"]))
            server.reply(command, result)
            if result:
                server.event("selection", entity=result["entity"])
        elif name == "load_scene":
            guid = str(command["guid"])
            handle = self.db.get(guid) if self.db is not None else None
            self.load_scene(handle.path if handle else Path(guid))
        elif name == "mouse":
            self.inject_mouse(
                float(command["x"]),
                float(command["y"]),
                command.get("button"),
                bool(command.get("down", False)),
            )
        elif name == "quick_save":
            self._pending_quick.append("save")
        elif name == "quick_load":
            self._pending_quick.append("load")
        elif name == "set_setting":
            self.set_setting(str(command["path"]), command.get("value"))
        else:
            _log.warning("Неизвестная команда: %r", name)

    def set_setting(self, path: str, value: Any) -> None:
        """Применить настройку «на лету» (``sky.*``, ``time_scale``)."""
        head, _, rest = path.partition(".")
        if head == "sky" and rest:
            self.sky.set_value(rest, value)
        elif head == "post" and rest:
            if self.scene_post is None:
                tier = QualityTier(self.args.quality)
                self.scene_post = PostSettings.for_quality(tier)
            self.scene_post.set_value(rest, value)
            self._apply_post()
        elif path == "render_scale" and self.dynres is not None and self.renderer:
            self.dynres.set_scale(float(value))
            self.dynres.enabled = False
            self.renderer.render_scale = self.dynres.scale
        elif path == "time_scale" and self.loop is not None:
            self.loop.time_scale = float(value)
        else:
            raise KeyError(f"Неизвестная настройка '{path}'")

    def pick(self, x: float, y: float) -> dict[str, Any] | None:
        """Найти объект под точкой экрана (по проекции центров)."""
        camera = self.scene_renderer.camera if self.scene_renderer else None
        if camera is None:
            return None
        width, height = self.viewport()
        best: tuple[float, dict[str, Any]] | None = None
        for name, entity, position in self.scene_renderer.pick_candidates():
            sx, sy, visible = camera.world_to_screen(position, width, height)
            if not visible:
                continue
            distance = ((sx - x) ** 2 + (sy - y) ** 2) ** 0.5
            if distance < 48.0 and (best is None or distance < best[0]):
                best = (distance, {"entity": entity, "name": name})
        return best[1] if best else None

    def publish(self, now: float) -> None:
        """Отправить статистику (5 Гц) и журнал (30 Гц)."""
        if self.server is None:
            return
        if now - self._log_clock >= LOG_PERIOD:
            self._log_clock = now
            for record in self.forwarder.drain():
                self.server.event("log", **record)
        if now - self._stats_clock >= STATS_PERIOD:
            self._stats_clock = now
            stats = self._last_stats
            self.server.event(
                "stats",
                fps=round(self._fps, 1),
                dc=getattr(stats, "draw_calls", 0),
                tri=getattr(stats, "triangles", 0),
                script_ms=round(self._script_ms, 3),
                cpu_ms=round(self._cpu_ms, 3),
                paused=self.paused,
            )

    # ------------------------------------------------------------------ #
    # Кадр
    # ------------------------------------------------------------------ #
    def viewport(self) -> tuple[int, int]:
        """Размер области вывода."""
        if self.window is not None:
            return self.window.size()
        return (self.args.width, self.args.height)

    def tick(self, delta: float) -> None:
        """Один кадр: логика и отрисовка (без показа буфера)."""
        started = time.perf_counter()
        self._script_ms = 0.0
        assert self.loop is not None and self.scene_renderer is not None
        if self._click_at is not None and self.frame >= 2:
            x, y = self._click_at
            self._click_at = None
            self.inject_mouse(x, y, "left", True)
            self.inject_mouse(x, y, "left", False)
        self._pump_keyboard()
        self.screen_fx.update(delta)
        if not self.paused or self.step_frames > 0:
            if self.step_frames > 0:
                self.step_frames -= 1
                delta = 1.0 / 60.0
            self.scene_renderer.update_ui(self.viewport())
            self.loop.step(delta)
            self.frame += 1
        logic_ms = (time.perf_counter() - started) * 1000.0
        if self.dynres is not None and self.dynres.enabled:
            self.gpu_timer.begin()
        self._last_stats = self.scene_renderer.render(self.viewport(), self.target)
        self.gpu_timer.end()
        elapsed = (time.perf_counter() - started) * 1000.0
        if not self.paused:
            self._update_dynamic_resolution(elapsed)
        if self.args.frame_trace:
            stats = self.loop.stats
            self._trace.append(
                {
                    "frame": self.frame,
                    "delta_ms": delta * 1000.0,
                    "tick_ms": elapsed,
                    "logic_ms": logic_ms,
                    "render_ms": elapsed - logic_ms,
                    "script_ms": self._script_ms,
                    "fixed_steps": stats.last_fixed_steps,
                    "fixed_ms": stats.last_fixed_ms,
                    "budget_stop": int(stats.last_budget_stop),
                    "overrun": stats.overrun_frames,
                }
            )
        self.input.end_frame()
        self._tick_status(delta)
        self._apply_pending_scene()
        self._cpu_ms = elapsed
        if self._last_stats is not None and not self.paused:
            self._samples.append(
                (
                    elapsed,
                    self._script_ms,
                    float(self._last_stats.draw_calls),
                )
            )
        now = time.perf_counter()
        if len(self._samples) == WARMUP_FRAMES:
            self._t_warm = now
        self._t_last = now
        self._samples_tri = getattr(self._last_stats, "triangles", 0)
        if self.args.crash_after and self.frame >= self.args.crash_after:
            raise RuntimeError("Преднамеренное падение (--crash-after)")

    def _update_fps(self, now: float) -> None:
        self._fps_frames += 1
        if now - self._fps_clock >= 0.5:
            self._fps = self._fps_frames / (now - self._fps_clock)
            self._fps_frames = 0
            self._fps_clock = now

    def run(self) -> int:
        """Главный цикл. Возвращает код выхода процесса."""
        args = self.args
        self.open_assets()
        self.init_audio()
        sync = args.sync_load
        # Ленивый импорт физики (~140 мс) не нарезать по бюджету: прогреваем его
        # в потоке, пока создаются окно и GL-контекст.
        warm = threading.Thread(target=_warm_imports, name="import-warm", daemon=True)
        warm.start()
        if sync:
            self._load_startup_scenes(incremental=False)
        self.start_ipc()
        settings = RenderSettings.for_quality(QualityTier(args.quality), samples=0)
        project_hdr = self._render_hints()[2]
        if self._hdr_choice() is None and self._startup_scene_has_post():
            _log.info("В сцене есть секция post: HDR включён автоматически")
            project_hdr = True
        elif self._hdr_choice() is False and self._startup_scene_has_post():
            _log.warning(
                "Пост выключен проектом (hdr: false), сцена содержит post: "
                "эффекты не применяются"
            )
        settings.hdr = project_hdr if args.hdr is None else args.hdr == "on"
        surface = self._open_surface(settings)
        try:
            warm.join()
            if not sync:
                self._load_startup_scenes(incremental=True)
            self._make_renderer(settings)
            self._make_loop()
            if args.headless:
                from engine.rendering import RenderTarget

                self.target = RenderTarget("player", args.width, args.height)
                if args.mouse_at:
                    x_text, _, y_text = args.mouse_at.partition(",")
                    self.inject_mouse(float(x_text), float(y_text))
                if args.click_at:
                    x_text, _, y_text = args.click_at.partition(",")
                    self._click_at = (float(x_text), float(y_text))
            self._warm_up()
            pacer = FramePacer(args.max_fps)
            self._main_loop(pacer)
        except Exception:  # noqa: BLE001 - падение сообщаем редактору
            trace = traceback.format_exc()
            _log.error("Плеер упал:\n%s", trace)
            driver = self.renderer.info.describe() if self.renderer else ""
            write_crash_log(
                trace,
                self.forwarder,
                driver,
                user_data_root(self.game_name) / "logs" if self.user_dirs else None,
            )
            if self.server is not None:
                self.server.crash(trace)
                time.sleep(0.2)
            self._exit_code = 1
        finally:
            self._finish()
            if self.audio_backend is not None:
                self.audio_backend.shutdown()
            if self.vfs is not None:
                self.vfs.close()
            if self.window is not None:
                self.window.close()
            elif surface is not None:
                surface.__exit__(None, None, None)
        return self._exit_code

    def _load_startup_scenes(self, *, incremental: bool) -> None:
        """Стартовая сцена (порциями или целиком) и ``--additive``."""
        started = time.perf_counter()
        self.load_scene(incremental=incremental)
        for spec in self.args.additive:
            self.load_additive(self._resolve_scene(spec))
        if self._load_steps:
            steps = self._load_steps
            _log.info(
                "Стартовая загрузка: %.0f мс, шагов %d (макс шаг %.1f мс, "
                "макс кадр загрузки %.1f мс)",
                (time.perf_counter() - started) * 1000.0,
                len(steps),
                max(steps),
                max(self._loading_frames, default=0.0),
            )

    def _open_surface(self, settings: RenderSettings) -> Any:
        if self.args.headless:
            # Собранная игра не содержит Qt (и плеер не импортирует его сам):
            # без PySide6 контекст даёт скрытое окно GLFW.
            forced = os.environ.get("MYENGINE_FORCE_HIDDEN_GLFW") == "1"
            if forced or importlib.util.find_spec("PySide6") is None:
                from engine.player.window import HiddenGlfwContext

                return HiddenGlfwContext().__enter__()
            from engine.rendering.qt_context import QtOffscreenContext

            os.environ.setdefault("QT_QPA_PLATFORM", "windows")
            context = QtOffscreenContext(
                (self.args.width, self.args.height), settings=settings
            )
            context.__enter__()
            return context
        from engine.player.window import create_window

        self.window = create_window(
            f"MyEngine — {self.project.name}",
            self.args.width,
            self.args.height,
            fullscreen=self.args.fullscreen,
        )
        if hasattr(self.window, "input_backend"):
            self.window.input_backend = self.glfw_input
        return None

    def _wait_vsync(self, hz: float) -> None:
        """Эмуляция vsync: ждать ближайшей границы периода ``1/hz``."""
        period = 1.0 / hz
        now = time.perf_counter()
        target = (int(now / period) + 1) * period
        while True:
            left = target - time.perf_counter()
            if left <= 0.0:
                return
            if left > 0.002:
                time.sleep(left - 0.001)

    def _main_loop(self, pacer: FramePacer) -> None:
        args = self.args
        last = time.perf_counter()
        while not self.stop_requested:
            pacer.begin()
            if self.window is not None and not self.window.poll_events():
                break
            self.handle_commands()
            if self.window is not None and self.window.is_iconified():
                pacer.idle()
                continue
            now = time.perf_counter()
            # В headless шаг фиксирован: результат не зависит от скорости машины.
            delta = (
                1.0 / 60.0 if args.headless and not args.closed_loop else now - last
            )
            last = now
            self.tick(delta)
            swap_started = time.perf_counter()
            if self.window is not None:
                self.window.swap_buffers()
            elif args.closed_loop and args.present_hz > 0.0:
                self._wait_vsync(args.present_hz)
            swap_ms = (time.perf_counter() - swap_started) * 1000.0
            self._update_fps(time.perf_counter())
            publish_started = time.perf_counter()
            self.publish(time.perf_counter())
            if self._trace and self._trace[-1]["frame"] == self.frame:
                self._trace[-1]["swap_ms"] = swap_ms
                self._trace[-1]["publish_ms"] = (
                    time.perf_counter() - publish_started
                ) * 1000.0
            if args.frames and self.frame >= args.frames and not self.paused:
                break
            pacer.end()

    # ------------------------------------------------------------------ #
    # Завершение
    # ------------------------------------------------------------------ #
    def _finish(self) -> None:
        args = self.args
        if args.screenshot and self.target is not None and self.target.is_created:
            from engine.rendering import save_png

            save_png(args.screenshot, self.target.read_pixels(), flip_vertical=True)
        if args.dump_state and self.scene is not None:
            self._write_json(args.dump_state, self.dump_state())
        if args.frame_trace:
            self._write_json(
                args.frame_trace,
                {
                    "frames": self._trace,
                    "loading_frames_ms": self._loading_frames,
                    "load_steps_ms": self._load_steps,
                },
            )
        if args.stats_json:
            self._write_json(args.stats_json, self.stats_report())
        if self.script_runtime_present():
            self._stop_scripts()
        if self.server is not None:
            self.server.event("exit", code=self._exit_code)
            self.server.shutdown()
        if self.scene_renderer is not None:
            self.scene_renderer.dispose()
        if self.renderer is not None:
            self.renderer.dispose()

    def script_runtime_present(self) -> bool:
        """Был ли создан реестр скриптов."""
        return getattr(self, "script_runtime", None) is not None

    def _stop_scripts(self) -> None:
        try:
            for extra in list(self.scenes.additive_scenes.values()):
                extra.shutdown()
            if self.scene is not None:
                self.scene.shutdown()
        except Exception:  # noqa: BLE001 - завершение не должно падать
            _log.exception("Ошибка остановки сцены")

    @staticmethod
    def _write_json(path: str, data: Any) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def dump_state(self) -> dict[str, Any]:
        """Состояние сцены: трансформы и публичные переменные скриптов."""
        from engine.scripting.script_component import ScriptComponent

        objects = []
        for game_object in self.scene.objects:
            transform = game_object.transform
            entry: dict[str, Any] = {
                "id": game_object.id,
                "name": game_object.name,
                "active": game_object.active_self,
                "position": list(transform.position.to_tuple()),
                "rotation": list(transform.rotation.to_tuple()),
                "scale": list(transform.scale.to_tuple()),
                "scripts": [],
            }
            for component in game_object.components:
                if not isinstance(component, ScriptComponent):
                    continue
                values: dict[str, Any] = {}
                module = component.module
                if module is not None:
                    for key, value in vars(module).items():
                        plain = None if key.startswith("_") else _plain(value)
                        if plain is not None:
                            values[key] = plain
                entry["scripts"].append(
                    {
                        "name": component.name,
                        "loaded": component.is_loaded,
                        "error": component.last_error,
                        "vars": values,
                    }
                )
            objects.append(entry)
        return {"frames": self.frame, "objects": objects}

    def _fps_average(self) -> float:
        """Средний FPS по настенному времени после прогрева."""
        frames = len(self._samples) - WARMUP_FRAMES
        span = self._t_last - self._t_warm
        return frames / span if frames > 0 and span > 0.0 else self._fps

    def stats_report(self) -> dict[str, Any]:
        """Итоги производительности для ``--stats-json``."""
        samples = self._samples[WARMUP_FRAMES:] or self._samples
        count = max(1, len(samples))
        cpu = sorted(s[0] for s in samples) or [0.0]
        return {
            "frames": self.frame,
            "cpu_ms_avg": sum(s[0] for s in samples) / count,
            "cpu_ms_p95": cpu[int(len(cpu) * 0.95) - 1 if len(cpu) > 1 else 0],
            "script_ms_avg": sum(s[1] for s in samples) / count,
            "script_ms_max": max((s[1] for s in samples), default=0.0),
            "draw_calls": max((s[2] for s in samples), default=0.0),
            "triangles": getattr(self, "_samples_tri", 0),
            "fps": self._fps_average(),
            "driver": self.renderer.info.describe() if self.renderer else "",
        }


def main(argv: list[str] | None = None) -> int:
    """Точка входа плеера."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    return PlayerApp(args).run()
