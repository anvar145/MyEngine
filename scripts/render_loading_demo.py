"""Рисует экран загрузки с прогрессом 45 % (headless) в ``_work/a8b_loading.png``.

Плеер запускается на ``loading.scene.json``; прогресс подаётся так же, как при
настоящей загрузке: событием ``SceneLoadProgress`` менеджера сцен. Сцена должна
быть создана ``scripts/seed_loading_demo.py``. Запуск:

    .venv\\Scripts\\python.exe scripts\\render_loading_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.core.ui_events import SceneLoadProgress  # noqa: E402
from engine.player.app import PlayerApp, parse_args  # noqa: E402
from engine.rendering import QualityTier, RenderSettings, RenderTarget  # noqa: E402

PROJECT = ROOT / "projects" / "PlayModeTest"
OUT = ROOT / "_work" / "a8b_loading.png"


def main() -> int:
    args = parse_args(
        ["--project", str(PROJECT), "--scene", "loading.scene.json", "--headless",
         "--screenshot", str(OUT)]
    )
    app = PlayerApp(args)
    app.open_assets()
    app.load_scene()
    settings = RenderSettings.for_quality(QualityTier(args.quality), samples=0)
    surface = app._open_surface(settings)
    try:
        app._make_renderer(settings)
        app._make_loop()
        app.target = RenderTarget("player", args.width, args.height)
        app._warm_up()
        app.scenes.events.emit(SceneLoadProgress(None, 0.45, "build", "big.scene.json"))
        for _ in range(3):
            app.tick(1.0 / 60.0)
    finally:
        app._finish()
        if surface is not None:
            surface.__exit__(None, None, None)
    print("Скриншот:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
