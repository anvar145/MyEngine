"""Замер стоимости пост-обработки по профилям качества (A10a).

Для каждого профиля (low/medium/high) и размера кадра меряется:

* ``post`` — чистое время ``PostStack.apply`` (tonemap [+FXAA] [+fade]);
* ``кадр`` — сцена + пост при ``render_scale`` профиля против кадра без поста.

Запуск: ``QT_QPA_PLATFORM=windows .venv/Scripts/python.exe tools/benchmark_post.py``.
Результат пишется в ``_work/benchmark_post.json``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "windows")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from OpenGL import GL  # noqa: E402

from engine.core.math3d import Vector3, compose_matrix  # noqa: E402
from engine.core.resources import ResourceManager, cube_mesh  # noqa: E402
from engine.core.screen_fx import ScreenFx  # noqa: E402
from engine.rendering import Camera, GpuMesh, Renderer, RenderTarget  # noqa: E402
from engine.rendering.lut import identity_lut  # noqa: E402
from engine.rendering.post import PostSettings, tier_render_scale  # noqa: E402
from engine.rendering.qt_context import QtOffscreenContext  # noqa: E402
from engine.rendering.settings import QualityTier, RenderSettings  # noqa: E402

SIZES = ((1280, 720), (960, 540))
WARMUP = 20
RUNS = 60


def _median(values: list[float]) -> float:
    values = sorted(values)
    return values[len(values) // 2]


def main() -> None:
    context = QtOffscreenContext((320, 240), settings=RenderSettings())
    assert context.make_current()
    resources = ResourceManager(roots=[ROOT / "assets"], name="bench")
    renderer = Renderer(resources=resources, shader_root=ROOT / "assets")
    renderer.initialize()
    renderer.hdr_enabled = True
    renderer.screen_fx = ScreenFx()
    print(
        "GPU:",
        renderer.info.renderer,
        "| HDR:",
        renderer.capabilities.supports_hdr_target,
    )
    camera = Camera(fov=50.0, far=100.0)
    camera.position = Vector3(0.0, 2.0, 6.0)
    camera.look_at(Vector3(0.0, 0.0, 0.0))
    mesh = GpuMesh.from_mesh_asset(cube_mesh(1.0))
    material = renderer.create_material("m", base_color=Vector3(0.8, 0.4, 0.2))
    matrices = [
        compose_matrix(position=Vector3(x * 1.5 - 6, 0, z * 1.5 - 6))
        for x in range(9)
        for z in range(9)
    ]
    results: list[dict] = []
    for size in SIZES:
        target = RenderTarget("final", *size)

        def frame(post: bool, size=size, target=target) -> None:
            renderer.begin_frame(camera, size, target=target, post=post)
            for matrix in matrices:
                renderer.submit(mesh, material, matrix)
            renderer.end_frame(target=target)
            GL.glFinish()

        renderer.render_scale = 1.0
        for _ in range(WARMUP):
            frame(False)
        plain = _median([_timed(frame, False) for _ in range(RUNS)])
        for tier in (QualityTier.LOW, QualityTier.MEDIUM, QualityTier.HIGH):
            hints = tier_render_scale(tier)
            renderer.render_scale = hints.scale
            for variant in _variants(tier):
                renderer.post_settings = variant[1]
                renderer.post.set_lut_pixels(identity_lut() if variant[2] else None)
                for _ in range(WARMUP):
                    frame(True)
                full = _median([_timed(frame, True) for _ in range(RUNS)])
                hdr = renderer.hdr_target
                post_ms = _median(
                    [_apply_ms(renderer, hdr, target, size) for _ in range(RUNS)]
                )
                row = {
                    "tier": tier.value,
                    "variant": variant[0],
                    "size": f"{size[0]}x{size[1]}",
                    "render_scale": hints.scale,
                    "tonemap": renderer.post_settings.tonemap,
                    "fxaa": renderer.post_settings.fxaa,
                    "bloom": renderer.post_settings.bloom,
                    "plain_ms": round(plain, 2),
                    "frame_ms": round(full, 2),
                    "post_ms": round(post_ms, 2),
                }
                results.append(row)
                print(
                    f"{row['size']:>9} {row['tier']:6s} {variant[0]:9s} "
                    f"scale {hints.scale:.2f} post {post_ms:5.2f} ms | "
                    f"кадр {full:6.2f} ms (без поста {plain:6.2f})"
                )
            renderer.post.set_lut_pixels(None)
        target.dispose()
    out = ROOT / "_work" / "benchmark_post.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2), "utf-8")
    renderer.dispose()
    context.dispose()


def _variants(tier: QualityTier) -> list[tuple[str, PostSettings, bool]]:
    """Варианты поста профиля: умолчания, без bloom, с LUT."""
    base = PostSettings.for_quality(tier)
    if tier is QualityTier.LOW:
        return [("default", base, False)]
    no_bloom = PostSettings.for_quality(tier)
    no_bloom.bloom = False
    return [
        ("default", base, False),
        ("no_bloom", no_bloom, False),
        ("with_lut", PostSettings.for_quality(tier), True),
    ]


def _timed(frame, post: bool) -> float:
    start = time.perf_counter()
    frame(post)
    return (time.perf_counter() - start) * 1000.0


def _apply_ms(renderer, hdr, target, size) -> float:
    GL.glFinish()
    start = time.perf_counter()
    renderer.post.apply(hdr, target, size)
    GL.glFinish()
    return (time.perf_counter() - start) * 1000.0


if __name__ == "__main__":
    main()
