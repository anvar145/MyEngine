# -*- mode: python ; coding: utf-8 -*-
"""Спека PyInstaller для плеера MyEngine (A13b).

Параметры приходят из ``build.py`` через переменные окружения ``MYENGINE_PKG_*``:

* ``NAME`` - имя exe/каталога; ``MODE`` - ``onedir`` | ``onefile``;
* ``PAK`` - путь к ``data.pak`` (в onefile вшивается в exe, в onedir кладётся
  рядом ``build.py``-ом); ``ICON`` и ``VERSION_FILE`` - иконка и ресурс версии;
* ``CONSOLE`` - ``1`` оставляет консольное окно (для проверок с выводом).

Не включаются: PySide6/редактор, скрипты проекта (они в ``data.pak``),
``Assets/``, ``Library/``, ``docs/``, ``scripts/``, ``tools/``, ``_work/``.
"""

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

HERE = Path(SPECPATH).resolve()  # noqa: F821 - задаётся PyInstaller
ROOT = HERE.parents[1]

NAME = os.environ.get("MYENGINE_PKG_NAME", "MyEngineGame")
MODE = os.environ.get("MYENGINE_PKG_MODE", "onedir")
PAK = os.environ.get("MYENGINE_PKG_PAK", "")
ICON = os.environ.get("MYENGINE_PKG_ICON") or None
VERSION_FILE = os.environ.get("MYENGINE_PKG_VERSION_FILE") or None
CONSOLE = os.environ.get("MYENGINE_PKG_CONSOLE") == "1"
MANIFEST = str(HERE / "player.manifest")

# Модули, которым нужны PySide6/редактор, или которые не нужны игре.
SKIP_MODULES = (
    "engine.tests",
    "engine.cli",
    "engine.rendering.qt_context",
    "engine.rendering.qt_viewport",
    "engine.player.window_qt",
)


def _wanted(name: str) -> bool:
    return not any(name == s or name.startswith(s + ".") for s in SKIP_MODULES)


# Часть модулей движка подключается лениво (PEP 562, import_module по имени):
# анализатор их не видит, поэтому берём пакет целиком.
hiddenimports = [m for m in collect_submodules("engine") if _wanted(m)]
hiddenimports += [
    "OpenGL.platform.win32",
    "OpenGL.platform.glx",
    *collect_submodules("OpenGL.arrays"),
    "glfw",
    "miniaudio",
    "_miniaudio",
    "PIL.Image",
    "PIL.PngImagePlugin",
    "PIL.JpegImagePlugin",
    "PIL.BmpImagePlugin",
    "PIL.GifImagePlugin",
    "PIL.DdsImagePlugin",
]

# Данные, которые плеер ищет рядом с пакетом ``engine`` (см. ``ROOT / "assets"``).
datas = [
    (str(ROOT / "assets" / "shaders"), "assets/shaders"),
    (str(ROOT / "assets" / "fonts"), "assets/fonts"),
    (str(ROOT / "assets" / "default_materials"), "assets/default_materials"),
    (str(ROOT / "assets" / "primitives"), "assets/primitives"),
    (str(ROOT / "assets" / "default_input_map.json"), "assets"),
]
if MODE == "onefile" and PAK:
    datas.append((PAK, "."))

binaries = collect_dynamic_libs("glfw")

excludes = [
    "PySide6",
    "PySide6_Essentials",
    "PySide6_Addons",
    "shiboken6",
    "tkinter",
    "test",
    "unittest",
    "pytest",
    "_pytest",
    "pandas",
    "matplotlib",
    "scipy",
    "IPython",
    "jupyter",
    "notebook",
    "setuptools",
    "pkg_resources",
    "pip",
    "wheel",
    "PyInstaller",
    "ruff",
    "editor",
    "engine.rendering.qt_context",
    "engine.rendering.qt_viewport",
    "engine.player.window_qt",
]

a = Analysis(  # noqa: F821
    [str(HERE / "run_player.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821

_exe_args = dict(
    name=NAME,
    debug=False,
    strip=False,
    upx=False,
    console=CONSOLE,
    icon=ICON,
    version=VERSION_FILE,
    manifest=MANIFEST,
)

if MODE == "onefile":
    exe = EXE(  # noqa: F821
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        runtime_tmpdir=None,
        **_exe_args,
    )
else:
    exe = EXE(  # noqa: F821
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        **_exe_args,
    )
    coll = COLLECT(  # noqa: F821
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        name=NAME,
    )
