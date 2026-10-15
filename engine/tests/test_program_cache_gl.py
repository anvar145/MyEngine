"""Тесты дискового кэша бинарников шейдерных программ (настоящий GL-контекст)."""

# ruff: noqa: F811 - фикстуры pytest подставляются по имени аргумента

from __future__ import annotations

from pathlib import Path

import pytest
from test_pbr_gl import draw, lights, sun
from test_rendering_gl import ASSETS, gl_context, renderer  # noqa: F401

VERTEX = """#version 330 core
layout(location = 0) in vec3 a_position;
void main() { gl_Position = vec4(a_position, 1.0); }
"""
FRAGMENT = """#version 330 core
out vec4 color;
void main() {
#ifdef RED
    color = vec4(1.0, 0.0, 0.0, 1.0);
#else
    color = vec4(0.0, 1.0, 0.0, 1.0);
#endif
}
"""


@pytest.fixture()
def cache(gl_context, tmp_path: Path):
    from engine.rendering import ProgramBinaryCache
    from engine.rendering.shader import set_program_cache

    instance = ProgramBinaryCache(tmp_path / "cache")
    if not instance.supported:
        pytest.skip("драйвер не поддерживает бинарники программ")
    set_program_cache(instance)
    yield instance
    set_program_cache(None)


def build(**defines):
    from engine.rendering import ShaderProgram

    return ShaderProgram.from_sources(
        "t", vertex=VERTEX, fragment=FRAGMENT, defines=defines or None
    )


class TestCache:
    def test_miss_then_hit(self, cache) -> None:
        first = build()
        assert (cache.stats.hits, cache.stats.misses, cache.stats.stores) == (0, 1, 1)
        assert len(list(cache.directory.glob("*.bin"))) == 1
        second = build()
        assert cache.stats.hits == 1 and cache.stats.stores == 1
        assert second.handle and second.handle != first.handle
        first.dispose()
        second.dispose()

    def test_variants_have_own_entries(self, cache) -> None:
        build().dispose()
        build(RED=1).dispose()
        assert len(list(cache.directory.glob("*.bin"))) == 2
        build(RED=1).dispose()
        assert cache.stats.hits == 1

    def test_key_depends_on_driver(self, cache, tmp_path: Path) -> None:
        from engine.core.resources import ShaderStage
        from engine.rendering import ProgramBinaryCache

        sources = {ShaderStage.VERTEX: VERTEX, ShaderStage.FRAGMENT: FRAGMENT}
        other = ProgramBinaryCache(tmp_path / "x", driver_id="другой драйвер")
        assert cache.key(sources) != other.key(sources)
        swapped = {ShaderStage.FRAGMENT: FRAGMENT, ShaderStage.VERTEX: VERTEX}
        assert cache.key(sources) == cache.key(swapped)

    def test_corrupt_entry_falls_back(self, cache) -> None:
        build().dispose()
        entry = next(cache.directory.glob("*.bin"))
        entry.write_bytes(entry.read_bytes()[:20])
        program = build()
        assert program.handle
        assert cache.stats.rejected == 1
        assert entry.stat().st_size > 20  # запись пересоздана компиляцией
        program.dispose()

    def test_garbage_entry_falls_back(self, cache) -> None:
        build().dispose()
        next(cache.directory.glob("*.bin")).write_bytes(b"garbage" * 10)
        program = build()
        assert program.handle and cache.stats.rejected == 1
        program.dispose()

    def test_cached_program_has_uniforms(self, cache) -> None:
        from engine.rendering import ShaderProgram

        source = FRAGMENT.replace(
            "out vec4 color;", "uniform vec3 u_tint;\nout vec4 color;"
        ).replace("void main() {", "void main() {\n    float tint = u_tint.x;")
        source = source.replace("vec4(1.0, 0.0", "vec4(1.0 + tint, 0.0")
        for _ in range(2):
            program = ShaderProgram.from_sources("u", vertex=VERTEX, fragment=source)
            assert program.uniform_location("u_tint") >= 0
            program.dispose()
        assert cache.stats.hits == 1

    def test_clear(self, cache) -> None:
        build().dispose()
        assert cache.clear() == 1
        assert not list(cache.directory.glob("*.bin"))

    def test_disabled_by_default(self, gl_context) -> None:
        from engine.rendering.shader import get_program_cache

        assert get_program_cache() is None

    def test_warm_variants(self, cache) -> None:
        from engine.rendering import warm_program_cache

        count = warm_program_cache(
            "w", vertex=VERTEX, fragment=FRAGMENT, variants=[{}, {"RED": 1}]
        )
        assert count == 2 and cache.stats.stores == 2
        build(RED=1).dispose()
        assert cache.stats.hits == 1


class TestRenderer:
    def _pixels(self, directory: Path):
        from engine.core.resources import ResourceManager, cube_mesh
        from engine.rendering import GpuMesh, Renderer

        resources = ResourceManager(roots=[ASSETS], name="CacheTests")
        instance = Renderer(
            resources=resources, shader_root=ASSETS, program_cache_dir=directory
        )
        instance.initialize()
        try:
            lights(instance, sun())
            material = instance.create_pbr_material(
                "m", roughness=0.5, metallic=0.0
            )
            mesh = GpuMesh.from_mesh_asset(cube_mesh(1.5))
            pixels = draw(instance, mesh, material)
            stats = (instance.program_cache.stats.hits, instance.program_cache.stats.stores)
        finally:
            instance.dispose()
            resources.clear()
        return pixels, stats

    def test_second_start_uses_cache_and_renders_same(
        self, gl_context, tmp_path: Path
    ) -> None:
        from engine.rendering.shader import get_program_cache

        probe = pytest.importorskip("engine.rendering").ProgramBinaryCache(tmp_path)
        if not probe.supported:
            pytest.skip("драйвер не поддерживает бинарники программ")
        directory = tmp_path / "programs"
        first, (hits1, stores1) = self._pixels(directory)
        second, (hits2, stores2) = self._pixels(directory)
        assert hits1 == 0 and stores1 >= 4  # lit, unlit, тень, pbr
        assert hits2 >= stores1 and stores2 == 0
        assert (first == second).all()
        assert get_program_cache() is None  # dispose отключил кэш
