"""Тесты PBR-шейдера и PbrMaterial на настоящем GL-контексте (offscreen)."""

# ruff: noqa: F811 - фикстуры pytest подставляются по имени аргумента

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_rendering_gl import (  # noqa: F401 - фикстуры pytest
    _color,
    _top_camera,
    count_visible,
    gl_context,
    render_to_target,
    renderer,
)

SIZE = 96


def lights(renderer, *items) -> None:
    renderer.shadows_enabled = False
    renderer.lights.set_lights(list(items))


def sun(direction=(0, -1, 0), intensity=1.0, cast_shadow=False):
    from engine.core.math3d import Vector3
    from engine.rendering import DirectionalLight

    return DirectionalLight(
        direction=Vector3(*direction), intensity=intensity, cast_shadow=cast_shadow
    )


def draw(renderer, mesh, material, *, camera=None, matrix=None):
    """Отрисовать один объект и вернуть пиксели ``(H, W, 4)`` как int."""
    from engine.core.math3d import compose_matrix

    target = render_to_target(renderer, camera or _top_camera(), SIZE, SIZE)
    renderer.submit(mesh, material, compose_matrix() if matrix is None else matrix)
    renderer.end_frame(target=target)
    return target.read_pixels().astype(int)


def center(pixels: np.ndarray) -> np.ndarray:
    return pixels[SIZE // 2, SIZE // 2, :3]


@pytest.fixture()
def sphere(renderer):
    from engine.core.resources import uv_sphere_mesh
    from engine.rendering import GpuMesh

    asset = uv_sphere_mesh(1.5, segments=32, rings=24)
    return GpuMesh.from_mesh_asset(asset)


@pytest.fixture()
def plane(renderer):
    from engine.core.resources import plane_mesh
    from engine.rendering import GpuMesh

    return GpuMesh.from_mesh_asset(plane_mesh(4.0))


def texture(name, array, **options):
    from engine.rendering import GpuTexture

    return GpuTexture.from_bytes(name, array.shape[1], array.shape[0], array, **options)


def solid(r, g, b, a=255, size=4):
    return np.broadcast_to(np.array([r, g, b, a], np.uint8), (size, size, 4)).copy()


class TestProgram:
    def test_pbr_program_and_variants_compile(self, renderer) -> None:
        base = renderer.program("pbr")
        assert base.handle != 0 and "pbr" in renderer.programs
        variants = [
            {"INSTANCED": 1},
            {"VERTEX_COLOR": 1},
            {"ALPHA_MASK": 1},
            {"DYNAMIC_LIGHTING": 0},
            {"INSTANCED": 1, "ALPHA_MASK": 1, "VERTEX_COLOR": 1},
        ]
        for defines in variants:
            assert renderer.pbr_variant(**defines).handle != 0
        assert renderer.pbr_variant() is base
        assert renderer.pbr_variant(ALPHA_MASK=1) is renderer.pbr_variant(ALPHA_MASK=1)

    def test_pbr_is_lazy(self, renderer) -> None:
        assert "pbr" not in renderer.programs
        renderer.create_pbr_material("m")
        assert "pbr" in renderer.programs

    def test_light_refactor_keeps_blinn_phong_working(self, renderer, plane) -> None:
        lights(renderer, sun())
        material = renderer.create_material("m", base_color=_color(0.8, 0.8, 0.8))
        assert center(draw(renderer, plane, material)).mean() > 60


class TestShading:
    def test_default_material_is_lit_and_gl_clean(self, renderer, sphere) -> None:
        from OpenGL import GL

        lights(renderer, sun())
        material = renderer.create_pbr_material("m", roughness=0.5)
        assert center(draw(renderer, sphere, material)).mean() > 60
        assert GL.glGetError() == 0

    def test_metal_has_no_diffuse_so_rough_metal_is_darker_than_dielectric(
        self, renderer, sphere
    ) -> None:
        lights(renderer, sun())
        color = _color(0.9, 0.2, 0.2)
        metal = draw(
            renderer, sphere,
            renderer.create_pbr_material("metal", base_color=color, metallic=1.0, roughness=1.0),
        )
        plastic = draw(
            renderer, sphere,
            renderer.create_pbr_material("plastic", base_color=color, metallic=0.0, roughness=1.0),
        )
        assert metal[:, :, :3].mean() < plastic[:, :, :3].mean()

    def test_metal_highlight_is_tinted_by_base_color(self, renderer, sphere) -> None:
        lights(renderer, sun(intensity=0.3))  # умеренный свет: блик не клиппится
        material = renderer.create_pbr_material(
            "gold", base_color=_color(1.0, 0.7, 0.1), metallic=1.0, roughness=0.8
        )
        r, g, b = center(draw(renderer, sphere, material))
        assert r > g > b

    def test_smoother_surface_has_brighter_highlight(self, renderer, sphere) -> None:
        lights(renderer, sun())
        values = []
        for roughness in (0.15, 0.9):
            material = renderer.create_pbr_material(
                f"r{roughness}", base_color=_color(1, 1, 1), metallic=1.0, roughness=roughness
            )
            values.append(center(draw(renderer, sphere, material)).mean())
        assert values[0] > values[1]

    def test_emissive_glows_without_lights(self, renderer, sphere) -> None:
        from engine.core.math3d import Vector3

        lights(renderer)
        black = _color(0, 0, 0)
        dark = draw(
            renderer, sphere, renderer.create_pbr_material("dark", base_color=black)
        )
        glow = draw(
            renderer,
            sphere,
            renderer.create_pbr_material(
                "glow", base_color=black, emissive=Vector3(0.8, 0.0, 0.0)
            ),
        )
        r, g, b = center(glow)
        assert r > 150 and g < 90 and b < 90
        assert r > center(dark)[0] + 80

    def test_double_sided_flips_backfaces(self, renderer, plane) -> None:
        from engine.core.math3d import Vector3, compose_matrix

        lights(renderer, sun())
        upside_down = compose_matrix(rotation=_flip_x())
        single = renderer.create_pbr_material("single", double_sided=False)
        double = renderer.create_pbr_material("double", double_sided=True)
        hidden = draw(renderer, plane, single, matrix=upside_down)
        shown = draw(renderer, plane, double, matrix=upside_down)
        assert count_visible_pixels(shown) > 0
        assert count_visible_pixels(hidden) == 0
        assert Vector3  # импорт нужен только для читаемости теста

    def test_baked_ambient_is_not_applied_twice(self, renderer) -> None:
        """Запечённый свет в PBR не умножается на u_ambient повторно."""
        from engine.core.math3d import Vector3, compose_matrix
        from engine.core.resources import uv_sphere_mesh
        from engine.rendering import AmbientLight, GpuMesh
        from engine.rendering.mesh import VertexLayout

        lights(renderer, AmbientLight(intensity=0.5))
        asset = uv_sphere_mesh(1.5, segments=24, rings=16)
        colors = renderer.lights.bake_to_vertex_colors(asset)
        layout = VertexLayout.from_names(["position", "normal", "uv", "color"])
        packed = layout.interleave(
            {
                "position": asset.positions,
                "normal": asset.normals,
                "uv": asset.uvs,
                "color": colors,
            },
            asset.vertex_count,
        )
        mesh = GpuMesh(layout, packed, asset.indices, name="baked", bounds=asset.bounds)
        material = renderer.create_pbr_material(
            "baked", base_color=Vector3(1, 1, 1), roughness=1.0
        )
        material.dynamic_lighting = False
        pixels = draw(renderer, mesh, material, matrix=compose_matrix())
        red = int(center(pixels)[0])
        # Linear 0.5 -> sRGB ≈ 188; с двойным ambient было бы заметно ярче.
        assert 175 <= red <= 200


def _flip_x():
    from engine.core.math3d import Quaternion, Vector3

    return Quaternion.from_axis_angle(Vector3(1, 0, 0), 180.0)  # градусы


def count_visible_pixels(pixels: np.ndarray) -> int:
    return int((~np.all(pixels[:, :, :3] < 40, axis=2)).sum())


class TestTextureMaps:
    def test_orm_occlusion_darkens_ambient_only(self, renderer, plane) -> None:
        from engine.core.math3d import Vector3
        from engine.rendering import AmbientLight

        lights(renderer, AmbientLight(color=Vector3(1, 1, 1), intensity=0.8))
        results = []
        for occlusion in (255, 0):
            orm = texture(f"orm{occlusion}", solid(occlusion, 255, 0))
            material = renderer.create_pbr_material(
                f"ao{occlusion}", roughness=1.0, orm_texture=orm
            )
            results.append(center(draw(renderer, plane, material)).mean())
        assert results[0] > results[1] + 20

    def test_orm_metallic_channel_scales_metallic(self, renderer, sphere) -> None:
        lights(renderer, sun())
        color = _color(0.9, 0.2, 0.2)
        means = []
        for metallic in (0, 255):
            orm = texture(f"m{metallic}", solid(255, 255, metallic))
            material = renderer.create_pbr_material(
                f"m{metallic}", base_color=color, metallic=1.0, roughness=1.0, orm_texture=orm
            )
            means.append(draw(renderer, sphere, material)[:, :, :3].mean())
        assert means[0] > means[1]

    def test_flat_normal_map_changes_nothing_but_tilted_one_does(
        self, renderer, plane
    ) -> None:
        lights(renderer, sun())
        plain = draw(renderer, plane, renderer.create_pbr_material("plain", roughness=1.0))
        flat = texture("flat", solid(128, 128, 255))
        tilted = texture("tilted", solid(255, 128, 128))
        with_flat = draw(
            renderer, plane,
            renderer.create_pbr_material("flat", roughness=1.0, normal_texture=flat),
        )
        with_tilt = draw(
            renderer, plane,
            renderer.create_pbr_material("tilt", roughness=1.0, normal_texture=tilted),
        )
        assert abs(with_flat[:, :, :3].mean() - plain[:, :, :3].mean()) < 4
        assert with_tilt[:, :, :3].mean() < plain[:, :, :3].mean() - 20

    def test_normal_scale_zero_disables_perturbation(self, renderer, plane) -> None:
        lights(renderer, sun())
        tilted = texture("tilted", solid(255, 128, 128))
        scaled = draw(
            renderer, plane,
            renderer.create_pbr_material(
                "s0", roughness=1.0, normal_texture=tilted, normal_scale=0.0
            ),
        )
        plain = draw(renderer, plane, renderer.create_pbr_material("p", roughness=1.0))
        assert abs(scaled[:, :, :3].mean() - plain[:, :, :3].mean()) < 4

    def test_base_color_texture_and_uv_scale(self, renderer, plane) -> None:
        lights(renderer, sun())
        pattern = np.zeros((8, 8, 4), np.uint8)
        pattern[:, :4] = (255, 0, 0, 255)
        pattern[:, 4:] = (0, 0, 255, 255)
        base = texture("pattern", pattern)
        material = renderer.create_pbr_material("tex", roughness=1.0, texture=base)
        pixels = draw(renderer, plane, material)
        assert pixels[:, :, :3].std() > 8
        tiled = renderer.create_pbr_material(
            "tiled", roughness=1.0, texture=base, uv_scale=(4.0, 4.0)
        )
        assert not np.array_equal(pixels, draw(renderer, plane, tiled))

    def test_alpha_mask_discards_and_blend_mixes(self, renderer, plane) -> None:
        from engine.rendering import TextureFilter

        lights(renderer, sun())
        cutout = solid(255, 255, 255, 255, size=8)
        cutout[:, :4, 3] = 0
        tex = texture("cutout", cutout, filter_mode=TextureFilter.NEAREST)
        opaque = renderer.create_pbr_material("o", roughness=1.0, texture=tex)
        masked = renderer.create_pbr_material(
            "m", roughness=1.0, texture=tex, alpha_mode="mask", alpha_cutoff=0.5
        )
        full = count_visible_pixels(draw(renderer, plane, opaque))
        half = count_visible_pixels(draw(renderer, plane, masked))
        assert 0.3 * full < half < 0.7 * full

        blend = renderer.create_pbr_material(
            "b", roughness=1.0, alpha_mode="blend", alpha=0.4
        )
        assert blend.transparent
        solid_mean = draw(
            renderer, plane, renderer.create_pbr_material("s", roughness=1.0)
        )[:, :, :3].mean()
        assert draw(renderer, plane, blend)[:, :, :3].mean() < solid_mean

    def test_invalid_alpha_mode_is_rejected(self, renderer) -> None:
        with pytest.raises(ValueError, match="alpha_mode"):
            renderer.create_pbr_material("bad", alpha_mode="glass")


class TestRenderPaths:
    def test_instanced_matches_loop(self, renderer, sphere) -> None:
        from engine.core.math3d import Vector3, compose_matrix
        from engine.rendering import RenderTarget

        lights(renderer, sun())
        material = renderer.create_pbr_material(
            "i", base_color=_color(0.7, 0.3, 0.3), metallic=0.5, roughness=0.5
        )
        matrices = np.stack(
            [compose_matrix(position=Vector3(x, 0, 0)) for x in (-1.6, 1.6)]
        )
        a = RenderTarget("a", SIZE, SIZE)
        renderer.begin_frame(_top_camera(), (SIZE, SIZE), target=a)
        assert renderer.draw_instanced(sphere, material, matrices) == 2
        stats = renderer.end_frame(target=a)
        assert stats.draw_calls == 1
        b = RenderTarget("b", SIZE, SIZE)
        renderer.begin_frame(_top_camera(), (SIZE, SIZE), target=b)
        for matrix in matrices:
            renderer.submit(sphere, material, matrix)
        renderer.end_frame(target=b)
        diff = np.abs(a.read_pixels().astype(int) - b.read_pixels().astype(int))
        assert diff.mean() < 1.0

    def test_materials_with_different_textures_do_not_share_sort_key(
        self, renderer
    ) -> None:
        one = renderer.create_pbr_material("a", orm_texture=texture("o1", solid(1, 2, 3)))
        two = renderer.create_pbr_material("b", orm_texture=texture("o2", solid(1, 2, 3)))
        same = renderer.create_pbr_material("c", orm_texture=one.orm_texture)
        assert one.sort_key != two.sort_key
        assert one.sort_key == same.sort_key

    def test_baked_mode_skips_dynamic_lights(self, renderer, sphere) -> None:
        from engine.core.math3d import Vector3

        material = renderer.create_pbr_material(
            "baked", base_color=Vector3(0.2, 0.2, 0.2), roughness=1.0
        )
        lights(renderer, sun())
        lit = center(draw(renderer, sphere, material)).mean()
        material.dynamic_lighting = False
        baked_sun = center(draw(renderer, sphere, material))
        lights(renderer)
        baked_dark = center(draw(renderer, sphere, material))
        assert lit != baked_sun.mean()
        assert (baked_sun == baked_dark).all()

    def test_shadow_pass_does_not_break_pbr(self, renderer, plane, sphere) -> None:
        """С включённым проходом теней PBR рисуется без ошибок GL и затемнений."""
        from OpenGL import GL

        from engine.core.math3d import Vector3, compose_matrix

        material = renderer.create_pbr_material("m", roughness=1.0)

        def frame(with_shadows: bool) -> np.ndarray:
            renderer.shadows_enabled = with_shadows
            renderer.lights.set_lights([sun(cast_shadow=True)])
            target = render_to_target(renderer, _top_camera(), SIZE, SIZE)
            renderer.submit(plane, material, compose_matrix(position=Vector3(0, -1, 0)))
            renderer.submit(sphere, material, compose_matrix())
            renderer.end_frame(target=target)
            return target.read_pixels().astype(int)[:, :, :3]

        with_shadows = frame(True)
        assert GL.glGetError() == 0
        without = frame(False)
        # Тень только затемняет: ни один пиксель не становится ярче.
        assert (with_shadows - without).max() <= 3
        assert with_shadows.std() > 5

    def _shadow_frame(
        self,
        renderer,
        plane,
        sphere,
        *,
        shadows=True,
        plane_flags=None,
        sphere_flags=None,
    ):
        """Кадр плоскость+сфера; ``*_flags`` уходят в ``submit`` объекта."""
        from engine.core.math3d import Vector3, compose_matrix

        material = renderer.create_pbr_material("m", roughness=1.0)
        renderer.shadows_enabled = shadows
        # Косой свет: тень сферы падает на плоскость рядом с ней, а не под неё.
        renderer.lights.set_lights([sun(direction=(0.5, -1, 0), intensity=0.2, cast_shadow=True)])
        target = render_to_target(renderer, _top_camera(), SIZE, SIZE)
        floor = compose_matrix(position=Vector3(0, -1, 0))
        renderer.submit(plane, material, floor, **(plane_flags or {}))
        ball = compose_matrix(position=Vector3(0, 1, 0), scale=Vector3(0.4, 0.4, 0.4))
        renderer.submit(sphere, material, ball, **(sphere_flags or {}))
        stats = renderer.end_frame(target=target)
        # Объект статистики общий для кадров: берём число сразу.
        return target.read_pixels().astype(int)[:, :, :3], stats.shadow_draws

    def test_cast_shadow_false_skips_shadow_map(self, renderer, plane, sphere) -> None:
        """Объект с cast_shadow=False не попадает в карту и не даёт тени."""
        _, both = self._shadow_frame(renderer, plane, sphere)
        no_cast, draws = self._shadow_frame(
            renderer, plane, sphere, sphere_flags={"cast_shadow": False}
        )
        unlit, _ = self._shadow_frame(renderer, plane, sphere, shadows=False)
        assert both == 2
        assert draws == both - 1
        assert np.abs(no_cast - unlit).max() <= 3

    def test_receive_shadow_false_ignores_shadows(
        self, renderer, plane, sphere
    ) -> None:
        """Объекты с receive_shadow=False не затемняются чужими тенями."""
        off = {"receive_shadow": False}
        no_receive, _ = self._shadow_frame(
            renderer, plane, sphere, plane_flags=off, sphere_flags=off
        )
        shadowed, _ = self._shadow_frame(renderer, plane, sphere)
        unlit, _ = self._shadow_frame(renderer, plane, sphere, shadows=False)
        assert np.abs(no_receive - unlit).max() <= 3
        assert (unlit - shadowed).max() > 5


class TestCookedTextures:
    def _cook(self, pixels, **settings):
        from engine.core.resources.cookers import (
            TextureCookSettings,
            cook_texture,
            read_cooked_texture,
        )

        blob = cook_texture(pixels, TextureCookSettings(**settings))
        return read_cooked_texture(blob)

    def test_cooked_mips_are_uploaded_and_sampled(self, renderer, plane) -> None:
        from OpenGL import GL

        from engine.rendering import GpuTexture

        lights(renderer, sun())
        pixels = np.zeros((16, 16, 4), np.uint8)
        pixels[..., 0] = 255
        pixels[..., 3] = 255
        cooked = self._cook(pixels)
        assert len(cooked.mips) == 5
        gpu = GpuTexture.from_cooked("red", cooked)
        assert gpu.srgb and gpu.upload() != 0 and gpu.has_mips
        material = renderer.create_pbr_material("c", roughness=1.0, texture=gpu)
        r, g, b = center(draw(renderer, plane, material))
        assert r > 100 and g < 40 and b < 40
        assert GL.glGetError() == 0

    def test_srgb_texture_is_decoded_to_linear(self, renderer, plane) -> None:
        from engine.rendering import GpuTexture

        lights(renderer, sun())
        gray = np.full((8, 8, 4), 128, np.uint8)
        gray[..., 3] = 255
        linear = GpuTexture.from_cooked("lin", self._cook(gray, srgb=False))
        srgb = GpuTexture.from_cooked("srgb", self._cook(gray, srgb=True))
        assert not linear.srgb and srgb.srgb
        a = center(draw(renderer, plane, renderer.create_pbr_material("a", roughness=1.0, texture=linear))).mean()
        b = center(draw(renderer, plane, renderer.create_pbr_material("b", roughness=1.0, texture=srgb))).mean()
        assert b < a  # sRGB-серый 128 в линейном пространстве темнее

    def test_single_level_gets_driver_mips(self, renderer) -> None:
        from engine.rendering import GpuTexture

        pixels = np.full((8, 8, 4), 200, np.uint8)
        cooked = self._cook(pixels, generate_mips=False)
        gpu = GpuTexture.from_cooked("one", cooked)
        gpu.upload()
        assert gpu.has_mips

    def test_block_compressed_texture_uploads(self, renderer, plane) -> None:
        from OpenGL import GL

        from engine.core.resources.cookers import (
            TextureCookSettings,
            cook_texture,
            read_cooked_texture,
        )
        from engine.core.resources.cookers.texture_format import mip_byte_size
        from engine.rendering import GpuTexture

        class Zeros:
            def compress(self, pixels, pixel_format, levels, srgb):
                h, w = pixels.shape[:2]
                out = []
                for _ in range(levels):
                    out.append(bytes(mip_byte_size(pixel_format, w, h)))
                    w, h = max(1, w // 2), max(1, h // 2)
                return out

        blob = cook_texture(
            np.full((8, 8, 4), 255, np.uint8), TextureCookSettings(compression="bc1"), Zeros()
        )
        cooked = read_cooked_texture(blob)
        assert cooked.compressed
        gpu = GpuTexture.from_cooked("bc1", cooked)
        gpu.upload()
        assert GL.glGetError() == 0 and gpu.has_mips


class TestAssetLoading:
    def _project(self, tmp_path: Path) -> Path:
        from engine.core.resources.cookers import encode_png

        root = tmp_path / "proj"
        root.mkdir()
        (root / "albedo.png").write_bytes(
            encode_png(np.broadcast_to(np.array([200, 40, 40], np.uint8), (4, 4, 3)).copy())
        )
        (root / "normal.png").write_bytes(
            encode_png(np.broadcast_to(np.array([128, 128, 255], np.uint8), (4, 4, 3)).copy())
        )
        (root / "rough.png").write_bytes(encode_png(np.full((4, 4, 1), 128, np.uint8)))
        (root / "paint.mat.json").write_text(
            json.dumps(
                {
                    "name": "Paint",
                    "base_color": [1, 1, 1, 0.5],
                    "metallic": 0.2,
                    "roughness": 0.7,
                    "emissive": [0.1, 0.0, 0.0],
                    "alpha_mode": "blend",
                    "sorting_layer": 2,
                    "double_sided": True,
                    "uv_scale": [2, 2],
                    "textures": {
                        "albedo": "albedo.png",
                        "normal": "normal.png",
                        "roughness": "rough.png",
                    },
                }
            )
        )
        return root

    def test_material_is_built_from_imported_asset(self, renderer, plane, tmp_path) -> None:
        from engine.core.resources import AssetDatabase
        from engine.rendering import PbrMaterial, load_pbr_material

        lights(renderer, sun())
        root = self._project(tmp_path)
        with AssetDatabase(root) as db:
            db.scan()
            db.reimport_dirty(wait=True)
            guid = db.guid_of(root / "paint.mat.json")
            cache: dict = {}
            material = load_pbr_material(renderer, db, guid, textures=cache)
            again = load_pbr_material(renderer, db, guid, textures=cache)

        assert isinstance(material, PbrMaterial)
        assert material.name == "Paint" and material.alpha_mode == "blend"
        assert material.transparent and material.double_sided
        assert material.roughness == pytest.approx(0.7) and material.alpha == pytest.approx(0.5)
        assert material.uv_scale == (2.0, 2.0)
        assert material.sorting_layer == 2
        assert material.texture is not None and material.texture.srgb
        assert material.normal_texture is not None and not material.normal_texture.srgb
        assert material.orm_texture is not None and not material.orm_texture.srgb
        # Кэш: те же GPU-текстуры у второго материала.
        assert again.texture is material.texture and again.orm_texture is material.orm_texture
        pixels = draw(renderer, plane, material)
        assert count_visible(_Pixels(pixels)) > 0

    def test_vfs_gives_same_material_as_database(self, renderer, tmp_path) -> None:
        from engine.core.resources import AssetDatabase, LibraryVFS, PakBuilder, PakVFS
        from engine.rendering import load_pbr_material

        root = self._project(tmp_path)
        with AssetDatabase(root, library=tmp_path / "Library") as db:
            db.scan()
            db.reimport_dirty(wait=True)
            guid = db.guid_of(root / "paint.mat.json")
            expected = load_pbr_material(renderer, db, guid)
            pak = PakBuilder(db).add_all().build(tmp_path / "p.pak")
            with PakVFS(pak) as pak_vfs:
                for vfs in (LibraryVFS(tmp_path / "Library"), pak_vfs):
                    got = load_pbr_material(renderer, vfs, guid)
                    assert got.name == expected.name
                    assert got.roughness == expected.roughness
                    assert got.alpha_mode == expected.alpha_mode
                    assert got.uv_scale == expected.uv_scale
                    for slot in ("texture", "normal_texture", "orm_texture"):
                        a, b = getattr(got, slot), getattr(expected, slot)
                        assert (a is None) == (b is None), (type(vfs).__name__, slot)

    def test_missing_texture_is_skipped_not_fatal(self, renderer, tmp_path, caplog) -> None:
        from engine.core.resources import AssetDatabase
        from engine.rendering import load_pbr_material

        root = self._project(tmp_path)
        with AssetDatabase(root) as db:
            db.scan()
            db.reimport_dirty(wait=True)
            (root / "albedo.png").unlink()
            (root / "albedo.png.meta").unlink()
            with AssetDatabase(root) as fresh:
                fresh.scan()
                guid = fresh.guid_of(root / "paint.mat.json")
                material = load_pbr_material(renderer, fresh, guid)
        assert material.texture is None and material.normal_texture is not None


class _Pixels:
    """Минимальная обёртка под ``count_visible`` (нужен ``read_pixels``)."""

    def __init__(self, pixels: np.ndarray) -> None:
        self._pixels = pixels

    def read_pixels(self) -> np.ndarray:
        return self._pixels
