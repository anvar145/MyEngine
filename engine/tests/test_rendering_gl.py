"""Тесты рендеринга с настоящим OpenGL-контекстом (offscreen).

Тесты создают контекст OpenGL 3.3 Core без окна, компилируют шейдеры, грузят
меши и проверяют результат по прочитанным пикселям. Если контекст недоступен
(нет GPU, offscreen-платформа Qt не умеет GL), тесты помечаются пропущенными —
это позволяет запускать набор где угодно, но проверять рендер там, где можно.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"


@pytest.fixture(scope="session")
def gl_context():
    """Общий на весь набор offscreen-контекст OpenGL.

    Yields:
        Активированный :class:`QtOffscreenContext`.
    """
    pytest.importorskip("PySide6", reason="нужен PySide6 для GL-контекста")
    # Платформа Qt "offscreen" не создаёт GL-контекст, поэтому нужна оконная.
    os.environ.setdefault("QT_QPA_PLATFORM", "windows")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from engine.rendering.qt_context import QtOffscreenContext
    from engine.rendering.settings import RenderSettings

    context = QtOffscreenContext((320, 240), settings=RenderSettings())
    if not context.make_current():
        context.dispose()
        pytest.skip("OpenGL-контекст недоступен в этой среде")
    yield context
    context.dispose()


@pytest.fixture()
def renderer(gl_context):
    """Рендерер, собранный в текущем контексте.

    Yields:
        Инициализированный :class:`Renderer`.
    """
    from engine.core.resources import ResourceManager
    from engine.rendering import Renderer

    resources = ResourceManager(roots=[ASSETS], name="RenderTests")
    instance = Renderer(resources=resources, shader_root=ASSETS)
    instance.initialize()
    yield instance
    instance.dispose()
    resources.clear()


@pytest.fixture()
def camera():
    """Камера, смотрящая на начало координат."""
    from engine.core.math3d import Vector3
    from engine.rendering import Camera

    view = Camera(fov=50.0, far=100.0)
    view.position = Vector3(0.0, 2.0, 6.0)
    view.look_at(Vector3(0.0, 0.0, 0.0))
    return view


def render_to_target(renderer, camera, width: int = 160, height: int = 120):
    """Отрисовать кадр в RenderTarget и вернуть его."""
    from engine.rendering import RenderTarget

    target = RenderTarget("test", width, height)
    renderer.begin_frame(camera, (width, height), target=target)
    return target


def count_visible(target) -> int:
    """Сколько пикселей отличаются от фона."""
    pixels = target.read_pixels()
    background = np.all(pixels[:, :, :3] < 40, axis=2)
    return int((~background).sum())


class TestContextAndPrograms:
    def test_context_reports_version(self, renderer) -> None:
        assert renderer.info.major >= 3
        assert renderer.info.minor >= 3
        assert "GL" in renderer.info.version or renderer.info.renderer
        assert renderer.capabilities.max_texture_size > 0
        assert "instancing" in renderer.capabilities.describe()

    def test_default_programs_are_compiled(self, renderer) -> None:
        programs = renderer.programs
        assert "lit" in programs
        assert "lit_instanced" in programs
        assert programs["lit"].handle != 0
        assert "INSTANCED" in "".join(programs["lit_instanced"].sources.values())

    def test_unlit_program_available(self, renderer) -> None:
        assert "unlit" in renderer.programs

    def test_unknown_program_raises(self, renderer) -> None:
        from engine.rendering import RenderError

        with pytest.raises(RenderError):
            renderer.program("nonexistent")

    def test_shader_compile_error_is_reported(self, renderer) -> None:
        from engine.rendering import ShaderCompileError, ShaderProgram

        with pytest.raises(ShaderCompileError):
            ShaderProgram.from_sources(
                "broken",
                vertex="#version 330 core\nvoid main() { this is not glsl }\n",
                fragment=(
                    "#version 330 core\nout vec4 c; void main() "
                    "{ c = vec4(1.0); }\n"
                ),
            )


class TestMeshUpload:
    def test_cube_is_uploaded_and_drawn(self, renderer, camera) -> None:
        from engine.core.math3d import compose_matrix
        from engine.core.resources import cube_mesh
        from engine.rendering import GpuMesh

        mesh = GpuMesh.from_mesh_asset(cube_mesh(2.0))
        material = renderer.create_material("red", base_color=_color(1.0, 0.2, 0.2))
        target = render_to_target(renderer, camera)
        assert renderer.submit(mesh, material, compose_matrix())
        stats = renderer.end_frame(target=target)

        assert stats.draw_calls == 1
        assert stats.triangles == 12
        assert count_visible(target) > 500
        assert mesh.is_uploaded

    def test_mesh_upload_is_cached(self, renderer) -> None:
        from engine.core.resources import cube_mesh
        from engine.rendering import GpuMesh

        mesh = GpuMesh.from_mesh_asset(cube_mesh())
        first = mesh.upload()
        assert mesh.upload() == first
        mesh.dispose()
        assert not mesh.is_uploaded

    def test_empty_mesh_is_rejected(self, renderer) -> None:
        from engine.rendering import GpuMesh, MeshUploadError

        with pytest.raises(MeshUploadError):
            GpuMesh.from_mesh_asset(_empty_mesh()).upload()


class TestLighting:
    def _render_plane_with_light(self, renderer, direction, color):
        """Отрисовать плоскость сверху с одним направленным источником.

        Свет задаётся через :class:`LightManager` (так же, как в редакторе и
        плеере), а карта теней отключается: тест проверяет вклад источника, а
        не тени.
        """
        from engine.core.math3d import compose_matrix
        from engine.core.resources import plane_mesh
        from engine.rendering import DirectionalLight, GpuMesh

        renderer.shadows_enabled = False
        renderer.lights.set_lights(
            [DirectionalLight(direction=direction, color=color, intensity=1.0)]
        )
        mesh = GpuMesh.from_mesh_asset(plane_mesh(2.0))
        material = renderer.create_material(
            f"lit_{direction.x}_{direction.y}_{direction.z}", base_color=color
        )
        target = render_to_target(renderer, _top_camera())
        renderer.submit(mesh, material, compose_matrix())
        renderer.end_frame(target=target)
        return target.read_pixels().astype(int)

    def test_directional_light_brightens_front_face(self, renderer) -> None:
        from engine.core.math3d import Vector3

        bright = self._render_plane_with_light(
            renderer, Vector3(0, -1, 0), _color(0.8, 0.8, 0.8)
        )        # Светит вбок: нормаль плоскости (+Y) не освещена.
        dark = self._render_plane_with_light(
            renderer, Vector3(1, 0, 0), _color(0.8, 0.8, 0.8)
        )
        assert bright[:, :, :3].mean() > dark[:, :, :3].mean()

    def test_light_not_overwritten_by_other_material(self, renderer) -> None:
        """Смена материала не затирает первый источник света сцены."""
        from engine.core.math3d import Vector3, compose_matrix
        from engine.core.resources import plane_mesh
        from engine.rendering import DirectionalLight, GpuMesh

        renderer.shadows_enabled = False
        mesh = GpuMesh.from_mesh_asset(plane_mesh(1.0))
        color = _color(0.8, 0.8, 0.8)
        renderer.lights.set_lights(
            [DirectionalLight(direction=Vector3(1, 0, 0), color=color)]
        )
        # Свет идёт вдоль +X, плоскость (+Y) не освещена: яркость = ambient.
        first = renderer.create_material("a_same", base_color=color)
        second = renderer.create_material("b_same", base_color=color)
        target = render_to_target(renderer, _top_camera())
        renderer.submit(mesh, first, compose_matrix(position=Vector3(-0.6, 0, 0)))
        renderer.submit(mesh, second, compose_matrix(position=Vector3(0.6, 0, 0)))
        renderer.end_frame(target=target)
        pixels = target.read_pixels().astype(int)[:, :, :3]
        half = pixels.shape[1] // 2
        left = pixels[:, :half][pixels[:, :half] > 60]
        right = pixels[:, half:][pixels[:, half:] > 60]
        assert left.size and right.size
        assert abs(left.mean() - right.mean()) < 3.0

    def test_point_light_falls_off_with_distance(self, renderer) -> None:
        """Затухание точечного света: вблизи ярче, за радиусом — фон."""
        from engine.core.math3d import Vector3, compose_matrix
        from engine.core.resources import plane_mesh
        from engine.rendering import GpuMesh, PointLight

        renderer.shadows_enabled = False
        mesh = GpuMesh.from_mesh_asset(plane_mesh(4.0))
        material = renderer.create_material("near_far", base_color=_color(0.9, 0.9, 0.9))
        view = _top_camera()

        renderer.lights.set_lights(
            [PointLight(position=Vector3(0, 1.0, 0), range=3.0, intensity=1.0)]
        )
        near = render_to_target(renderer, view)
        renderer.submit(mesh, material, compose_matrix())
        renderer.end_frame(target=near)

        renderer.lights.set_lights(
            [PointLight(position=Vector3(0, 40.0, 0), range=3.0, intensity=1.0)]
        )
        far = render_to_target(renderer, view)
        renderer.submit(mesh, material, compose_matrix())
        renderer.end_frame(target=far)

        assert near.read_pixels()[:, :, :3].mean() > far.read_pixels()[:, :, :3].mean()

    def test_light_culling_skips_distant_objects(self, renderer) -> None:
        """Отсечение по радиусу: далёкий объект не получает источников."""
        from engine.core.math3d import Vector3
        from engine.rendering import PointLight

        renderer.lights.set_lights(
            [PointLight(position=Vector3(0, 0, 0), range=5.0, intensity=1.0)]
        )
        near = renderer.lights.select(Vector3(0, 0, 0), 0.5)
        far = renderer.lights.select(Vector3(100, 0, 0), 0.5)
        assert len(near) == 1
        assert len(far) == 0

    def test_ambient_light_affects_everything(self, renderer) -> None:
        """Фоновый свет доходит даже туда, куда направленный не достаёт."""
        from engine.core.math3d import Vector3
        from engine.rendering import AmbientLight, DirectionalLight

        renderer.lights.set_lights(
            [DirectionalLight(direction=Vector3(0, -1, 0), intensity=1.0)]
        )
        without = renderer.lights.ambient_color
        renderer.lights.set_lights(
            [
                DirectionalLight(direction=Vector3(0, -1, 0), intensity=1.0),
                AmbientLight(color=Vector3(1.0, 1.0, 1.0), intensity=0.4),
            ]
        )
        with_ambient = renderer.lights.ambient_color
        assert with_ambient.x > without.x

    def test_ambient_only_when_light_is_away(self, renderer) -> None:
        from engine.core.math3d import Vector3
        from engine.rendering import DirectionalLight

        renderer.light = DirectionalLight(direction=Vector3(0, 1, 0), ambient=0.5)
        assert renderer.light.unit_direction.almost_equals(Vector3(0, 1, 0), 1e-6)
        # Фоновый вклад запасного света тоже стал цветом.
        assert renderer.light.ambient_color.x > 0.0


class TestCulling:
    def test_object_behind_camera_is_culled(self, renderer, camera) -> None:
        from engine.core.math3d import Vector3, compose_matrix
        from engine.core.resources import cube_mesh
        from engine.rendering import GpuMesh

        mesh = GpuMesh.from_mesh_asset(cube_mesh())
        material = renderer.create_material("m")
        target = render_to_target(renderer, camera)

        assert renderer.submit(mesh, material, compose_matrix())
        assert not renderer.submit(
            mesh, material, compose_matrix(position=Vector3(0, 0, 200))
        )
        stats = renderer.end_frame(target=target)
        assert stats.culled == 1
        assert stats.draw_calls == 1

    def test_culling_can_be_disabled(self, renderer, camera) -> None:
        from engine.core.math3d import Vector3, compose_matrix
        from engine.core.resources import cube_mesh
        from engine.rendering import GpuMesh

        mesh = GpuMesh.from_mesh_asset(cube_mesh())
        material = renderer.create_material("m")
        target = render_to_target(renderer, camera)
        assert renderer.submit(
            mesh, material, compose_matrix(position=Vector3(0, 0, 200)), cull=False
        )
        stats = renderer.end_frame(target=target)
        assert stats.culled == 0
        assert stats.submitted == 1


class TestBatching:
    def test_same_material_is_one_batch(self, renderer, camera) -> None:
        from engine.core.math3d import Vector3, compose_matrix
        from engine.core.resources import cube_mesh
        from engine.rendering import GpuMesh

        mesh = GpuMesh.from_mesh_asset(cube_mesh())
        material = renderer.create_material("same")
        target = render_to_target(renderer, camera)
        for index in range(4):
            renderer.submit(
                mesh,
                material,
                compose_matrix(position=Vector3(index * 0.6 - 1.0, 0, 0)),
            )
        stats = renderer.end_frame(target=target)
        assert stats.draw_calls == 4
        assert stats.batches == 1  # состояние переключено один раз

    def test_same_program_materials_stay_one_batch(self, renderer, camera) -> None:
        """Разные материалы на одной программе рисуются одной пачкой."""
        from engine.core.math3d import Vector3, compose_matrix
        from engine.core.resources import cube_mesh
        from engine.rendering import GpuMesh

        mesh = GpuMesh.from_mesh_asset(cube_mesh())
        first = renderer.create_material("a", base_color=_color(1, 0, 0))
        second = renderer.create_material("b", base_color=_color(0, 1, 0))
        target = render_to_target(renderer, camera)
        renderer.submit(mesh, first, compose_matrix(position=Vector3(-0.7, 0, 0)))
        renderer.submit(mesh, second, compose_matrix(position=Vector3(0.7, 0, 0)))
        stats = renderer.end_frame(target=target)
        assert stats.draw_calls == 2
        assert stats.batches == 1  # программа и текстура те же — состояние не менялось

    def test_different_programs_switch_state(self, renderer, camera) -> None:
        from engine.core.math3d import Vector3, compose_matrix
        from engine.core.resources import cube_mesh
        from engine.rendering import GpuMesh

        mesh = GpuMesh.from_mesh_asset(cube_mesh())
        lit = renderer.create_material("lit", base_color=_color(1, 0, 0))
        unlit = renderer.create_material(
            "unlit", program="unlit", base_color=_color(0, 1, 0)
        )
        target = render_to_target(renderer, camera)
        renderer.submit(mesh, lit, compose_matrix(position=Vector3(-0.7, 0, 0)))
        renderer.submit(mesh, unlit, compose_matrix(position=Vector3(0.7, 0, 0)))
        stats = renderer.end_frame(target=target)
        assert stats.batches == 2

    def test_state_cache_saves_calls(self, renderer, camera) -> None:
        from engine.core.math3d import Vector3, compose_matrix
        from engine.core.resources import cube_mesh
        from engine.rendering import GpuMesh

        mesh = GpuMesh.from_mesh_asset(cube_mesh())
        material = renderer.create_material("m")
        target = render_to_target(renderer, camera)
        for index in range(6):
            renderer.submit(
                mesh,
                material,
                compose_matrix(position=Vector3(index * 0.5 - 1.5, 0, 0)),
            )
        stats = renderer.end_frame(target=target)
        assert stats.state_saved > 0  # кэш действительно экономит вызовы


class TestInstancing:
    def test_instanced_matches_loop_drawing(self, renderer, camera) -> None:
        from engine.core.math3d import Vector3, compose_matrix
        from engine.core.resources import cube_mesh
        from engine.rendering import GpuMesh

        mesh = GpuMesh.from_mesh_asset(cube_mesh())
        material = renderer.create_material("m")

        positions = [Vector3(-1.0, 0, 0), Vector3(1.0, 0, 0)]
        matrices = np.stack([compose_matrix(position=point) for point in positions])

        target_instanced = render_to_target(renderer, camera)
        drawn = renderer.draw_instanced(mesh, material, matrices)
        stats = renderer.end_frame(target=target_instanced)
        assert drawn == 2
        assert stats.draw_calls == 1  # одна пачка на все экземпляры
        assert stats.instances == 2
        instanced_pixels = count_visible(target_instanced)

        target_loop = render_to_target(renderer, camera)
        for matrix in matrices:
            renderer.submit(mesh, material, matrix)
        renderer.end_frame(target=target_loop)
        assert instanced_pixels == count_visible(target_loop)

    def test_instancing_handles_many_objects(self, renderer, camera) -> None:
        from engine.core.math3d import Vector3, compose_matrix
        from engine.core.resources import cube_mesh
        from engine.rendering import GpuMesh

        mesh = GpuMesh.from_mesh_asset(cube_mesh(0.2))
        material = renderer.create_material("m")
        matrices = np.stack(
            [
                compose_matrix(
                    position=Vector3((i % 10) * 0.5 - 2.5, 0, (i // 10) * 0.5 - 2.5)
                )
                for i in range(400)
            ]
        )
        target = render_to_target(renderer, camera, 320, 240)
        renderer.draw_instanced(mesh, material, matrices)
        stats = renderer.end_frame(target=target)
        assert stats.draw_calls == 1
        assert stats.instances == 400
        assert count_visible(target) > 100

    def test_instance_group_is_culled_as_whole(self, renderer, camera) -> None:
        from engine.core.math3d import Vector3, compose_matrix
        from engine.core.resources import cube_mesh
        from engine.rendering import GpuMesh

        mesh = GpuMesh.from_mesh_asset(cube_mesh())
        material = renderer.create_material("m")
        far = np.stack([compose_matrix(position=Vector3(0, 0, 500))])
        target = render_to_target(renderer, camera)
        assert renderer.draw_instanced(mesh, material, far) == 0
        stats = renderer.end_frame(target=target)
        assert stats.culled == 1


class TestTextures:
    def test_texture_upload_and_sampling(self, renderer, camera) -> None:
        from engine.core.math3d import compose_matrix
        from engine.core.resources import plane_mesh
        from engine.rendering import GpuMesh, GpuTexture

        pixels = np.zeros((16, 16, 4), dtype=np.uint8)
        pixels[:8, :, 0] = 255  # половина текстуры — красная
        pixels[8:, :, 2] = 255  # половина — синяя
        texture = GpuTexture.from_bytes("checker", 16, 16, pixels)
        assert texture.upload() != 0
        assert texture.is_uploaded
        assert texture.gpu_bytes >= 16 * 16 * 4

        mesh = GpuMesh.from_mesh_asset(plane_mesh(2.0))
        material = renderer.create_material(
            "textured", texture=texture, base_color=_color(1, 1, 1)
        )
        target = render_to_target(renderer, _top_camera(), 128, 128)
        renderer.submit(mesh, material, compose_matrix())
        renderer.end_frame(target=target)

        rendered = target.read_pixels().astype(int)
        assert rendered[:, :, :3].std() > 5.0  # текстура дала разные цвета
        texture.dispose()
        assert not texture.is_uploaded

    def test_texture_without_pixels_is_reported(self, renderer) -> None:
        from engine.rendering import GpuTexture, RenderError

        texture = GpuTexture("empty", 4, 4)
        with pytest.raises(RenderError):
            texture.upload()


class TestRenderTargetAndScreenshot:
    def test_read_pixels_returns_frame(self, renderer, camera) -> None:
        from engine.core.math3d import compose_matrix
        from engine.core.resources import cube_mesh
        from engine.rendering import GpuMesh

        mesh = GpuMesh.from_mesh_asset(cube_mesh(2.0))
        material = renderer.create_material("m", base_color=_color(1, 0.5, 0))
        target = render_to_target(renderer, camera, 96, 72)
        renderer.submit(mesh, material, compose_matrix())
        renderer.end_frame(target=target)

        pixels = target.read_pixels()
        assert pixels.shape == (72, 96, 4)
        assert pixels.dtype == np.uint8
        assert pixels.max() > 0

    def test_target_resize_recreates_buffers(self, renderer) -> None:
        from engine.rendering import RenderTarget

        target = RenderTarget("resizable", 32, 32)
        target.create()
        target.resize(64, 48)
        # Драйвер вправе выдать те же имена объектов, поэтому проверяем размеры
        # и работоспособность цели, а не идентификаторы.
        assert (target.width, target.height) == (64, 48)
        assert target.is_created
        assert target.aspect == pytest.approx(64 / 48)
        pixels = target.read_pixels()
        assert pixels.shape == (48, 64, 4)
        target.dispose()
        assert not target.is_created

    def test_target_resize_is_noop_for_same_size(self, renderer) -> None:
        from engine.rendering import RenderTarget

        target = RenderTarget("same", 16, 16)
        target.create()
        first = target.color_textures[0]
        target.resize(16, 16)
        assert target.color_textures[0] == first  # буферы не пересоздавались
        target.dispose()

    def test_save_screenshot(self, renderer, camera, tmp_path) -> None:
        from engine.core.math3d import compose_matrix
        from engine.core.resources import cube_mesh
        from engine.rendering import GpuMesh, save_png

        mesh = GpuMesh.from_mesh_asset(cube_mesh(2.0))
        material = renderer.create_material("m", base_color=_color(1, 1, 1))
        target = render_to_target(renderer, camera, 64, 64)
        renderer.submit(mesh, material, compose_matrix())
        renderer.end_frame(target=target)

        path = save_png(tmp_path / "shot.png", target.read_pixels(), flip_vertical=True)
        assert path.exists()
        assert path.read_bytes().startswith(b"\x89PNG")

    def test_gl_error_check_passes_after_frame(self, renderer, camera) -> None:
        from engine.core.math3d import compose_matrix
        from engine.core.resources import cube_mesh
        from engine.rendering import GpuMesh, check_gl_error

        mesh = GpuMesh.from_mesh_asset(cube_mesh())
        material = renderer.create_material("m")
        target = render_to_target(renderer, camera)
        renderer.submit(mesh, material, compose_matrix())
        renderer.end_frame(target=target)
        check_gl_error("после кадра")  # не должно бросить исключение


class TestRenderGraphIntegration:
    def test_custom_pass_runs_after_geometry(self, renderer, camera) -> None:
        from engine.core.math3d import compose_matrix
        from engine.core.resources import cube_mesh
        from engine.rendering import GpuMesh

        order: list[str] = []

        class Marker:
            name = "marker"

            def execute(self, context) -> None:
                order.append("marker")

        mesh = GpuMesh.from_mesh_asset(cube_mesh())
        material = renderer.create_material("m")
        renderer.add_pass(Marker())
        target = render_to_target(renderer, camera)
        renderer.submit(mesh, material, compose_matrix())
        renderer.end_frame(target=target)

        assert order == ["marker"]
        pass_names = [item.name for item in renderer.graph.passes]
        # Порядок проходов: очистка, карта теней, геометрия, затем свои проходы.
        assert pass_names[:3] == ["clear", "shadow", "geometry"]
        assert pass_names[-1] == "marker"

    def test_transparent_objects_drawn_last(self, renderer, camera) -> None:
        from engine.core.math3d import Vector3, compose_matrix
        from engine.core.resources import cube_mesh
        from engine.rendering import GpuMesh

        mesh = GpuMesh.from_mesh_asset(cube_mesh())
        opaque = renderer.create_material("opaque", base_color=_color(0.2, 0.6, 0.9))
        glass = renderer.create_material(
            "glass", base_color=_color(1, 1, 1), transparent=True, alpha=0.4
        )
        target = render_to_target(renderer, camera)
        renderer.submit(mesh, glass, compose_matrix(position=Vector3(0, 0, 1.2)))
        renderer.submit(mesh, opaque, compose_matrix(position=Vector3(0, 0, -1.2)))
        stats = renderer.end_frame(target=target)
        assert stats.draw_calls == 2
        assert count_visible(target) > 500


    def test_sorting_layer_overrides_distance_for_transparent(
        self, renderer, camera
    ) -> None:
        """Слой сортировки важнее расстояния: больший слой рисуется позже."""
        from engine.core.math3d import Vector3, compose_matrix
        from engine.core.resources import cube_mesh
        from engine.rendering import GpuMesh

        mesh = GpuMesh.from_mesh_asset(cube_mesh())

        def pixel(far_layer: int) -> np.ndarray:
            near = renderer.create_material(
                "near", program="unlit", base_color=_color(0, 0, 1),
                transparent=True, alpha=0.5,
            )  # fmt: skip
            far = renderer.create_material(
                "far", program="unlit", base_color=_color(1, 0, 0),
                transparent=True, alpha=0.5,
            )  # fmt: skip
            far.sorting_layer = far_layer
            target = render_to_target(renderer, camera)
            renderer.submit(mesh, near, compose_matrix(position=Vector3(0, 0, 1.0)))
            renderer.submit(mesh, far, compose_matrix(position=Vector3(0, 0, -1.0)))
            renderer.end_frame(target=target)
            return target.read_pixels()[60, 80, :3].astype(int)

        by_distance = pixel(0)  # дальний рисуется первым, ближний — поверх
        by_layer = pixel(1)  # слой дальнего выше: он рисуется последним
        assert by_distance[2] > by_distance[0]
        assert np.abs(by_distance - by_layer).max() > 10


    def test_msaa_target_resolves_and_smooths_edges(self, renderer, camera) -> None:
        """MSAA-цель после resolve читается и даёт больше оттенков на кромке."""
        from engine.core.math3d import Quaternion, compose_matrix
        from engine.core.resources import cube_mesh
        from engine.rendering import GpuMesh, RenderTarget

        mesh = GpuMesh.from_mesh_asset(cube_mesh())
        material = renderer.create_material("m", program="unlit",
                                            base_color=_color(1, 1, 1))  # fmt: skip
        turn = Quaternion.from_euler(0.0, 0.0, 30.0)

        def shades(samples: int) -> tuple[int, RenderTarget]:
            target = RenderTarget("aa", 160, 120, samples=samples)
            renderer.begin_frame(camera, (160, 120), target=target)
            renderer.submit(mesh, material, compose_matrix(rotation=turn))
            renderer.end_frame(target=target)
            return len(np.unique(target.read_pixels()[:, :, 0])), target

        plain, _ = shades(0)
        smooth, target = shades(4)
        assert target.msaa_samples >= 2
        assert smooth >= plain + 2


class TestDepthBuffer:
    def test_nearer_object_hides_farther(self, renderer) -> None:
        """Проверка буфера глубины: ближний куб перекрывает дальний."""
        from engine.core.math3d import Vector3, compose_matrix
        from engine.core.resources import cube_mesh
        from engine.rendering import Camera, GpuMesh

        mesh = GpuMesh.from_mesh_asset(cube_mesh(2.0))
        near = renderer.create_material("near", base_color=_color(1.0, 0.1, 0.1))
        far = renderer.create_material("far", base_color=_color(0.1, 1.0, 0.1))

        view = Camera(fov=45.0, far=100.0)
        view.position = Vector3(0.0, 0.0, 8.0)
        view.look_at(Vector3.zero())

        target = render_to_target(renderer, view, 96, 96)
        # Дальний рисуем первым: если бы глубины не было, он перекрыл бы ближний.
        renderer.submit(mesh, far, compose_matrix(position=Vector3(0, 0, -3)))
        renderer.submit(mesh, near, compose_matrix(position=Vector3(0, 0, 0)))
        renderer.end_frame(target=target)

        pixels = target.read_pixels()
        height, width = pixels.shape[:2]
        center = pixels[height // 2, width // 2][:3].astype(int)
        message = f"в центре должен быть красный куб, получено {center}"
        assert center[0] > center[1], message


class TestProgramStateDesync:
    """Рендер не должен падать, если активную программу сменили в обход кэша.

    Регрессия из логов редактора: ``GL_INVALID_OPERATION`` (1282) на
    ``glUniformMatrix4fv``. Причина — :class:`GLState` считал активной одну
    программу, а драйвер удерживал другую; рендерер пропускал
    ``glUseProgram`` и писал uniform-переменные камеры в чужую программу.
    """

    #: Минимальная «сторонняя» программа: о ней кэш состояний не знает.
    WITNESS_VERTEX = """#version 330 core
layout(location = 0) in vec3 in_position;
uniform mat4 u_model;
uniform mat4 u_view_projection;
void main() {
    gl_Position = u_view_projection * u_model * vec4(in_position, 1.0);
}
"""

    WITNESS_FRAGMENT = """#version 330 core
out vec4 out_color;
void main() { out_color = vec4(1.0, 0.0, 1.0, 1.0); }
"""

    def _witness(self):
        """Собрать стороннюю программу (о ней кэш состояний не знает)."""
        from engine.rendering import ShaderProgram

        program = ShaderProgram.from_sources(
            "witness",
            vertex=self.WITNESS_VERTEX,
            fragment=self.WITNESS_FRAGMENT,
        )
        program.build()
        return program

    def _draw_frame(self, renderer, camera, target, mesh, material) -> None:
        """Отрисовать один кадр одним объектом."""
        from engine.core.math3d import compose_matrix

        renderer.begin_frame(camera, (64, 64), target=target)
        renderer.submit(mesh, material, compose_matrix())
        renderer.end_frame(target=target)

    def test_foreign_program_bound_out_of_band(self, renderer, camera) -> None:
        """Внешний код привязал свою программу — рендер обязан выжить."""
        from OpenGL import GL

        from engine.core.math3d import compose_matrix
        from engine.core.resources import cube_mesh
        from engine.rendering import GpuMesh, RenderTarget

        mesh = GpuMesh.from_mesh_asset(cube_mesh(1.0))
        lit = renderer.create_material("lit_object")
        unlit = renderer.create_material("unlit_object", program="unlit")
        target = RenderTarget("desync", 64, 64)

        # Прогрев: рендерер запоминает активную программу.
        self._draw_frame(renderer, camera, target, mesh, lit)

        witness = self._witness()
        try:
            # Ломаем состояние: в драйвере — чужая программа, в кэше — прежняя.
            GL.glUseProgram(witness.handle)
            renderer.state._program = lit.program.handle  # noqa: SLF001
            renderer.begin_frame(camera, (64, 64), target=target)
            renderer.submit(mesh, unlit, compose_matrix())
            renderer.end_frame(target=target)
        finally:
            witness.dispose()

        # После кадра драйвер и кэш согласованы.
        current = int(GL.glGetIntegerv(GL.GL_CURRENT_PROGRAM))
        assert current == unlit.program.handle

    def test_deleted_program_left_in_cache(self, renderer, camera) -> None:
        """В кэше остался удалённый идентификатор — состояние лечится."""
        from OpenGL import GL

        from engine.core.math3d import compose_matrix
        from engine.core.resources import cube_mesh
        from engine.rendering import GpuMesh, RenderTarget

        mesh = GpuMesh.from_mesh_asset(cube_mesh(1.0))
        material = renderer.create_material(
            "lit_object", base_color=_color(0.9, 0.2, 0.2)
        )
        target = RenderTarget("dead_program", 64, 64)

        witness = self._witness()
        dead_handle = witness.handle
        GL.glDeleteProgram(dead_handle)
        witness.handle = 0
        renderer.state._program = dead_handle  # noqa: SLF001

        renderer.begin_frame(camera, (64, 64), target=target)
        renderer.submit(mesh, material, compose_matrix())
        stats = renderer.end_frame(target=target)

        assert stats.draw_calls == 1
        current = int(GL.glGetIntegerv(GL.GL_CURRENT_PROGRAM))
        assert current == material.program.handle

    def test_program_use_with_state_keeps_cache_in_sync(self, renderer) -> None:
        """``ShaderProgram.use(state)`` не рассинхронизирует кэш."""
        program = renderer.program("unlit")
        assert program.use(renderer.state) == program.handle
        assert renderer.state._program == program.handle  # noqa: SLF001

    def test_missing_program_reports_clear_error(self, gl_context, camera) -> None:
        """Несобранная программа даёт понятную ошибку, а не падение драйвера."""
        from engine.core.math3d import compose_matrix
        from engine.core.resources import ResourceManager, cube_mesh
        from engine.rendering import GpuMesh, Renderer, RenderError, RenderTarget

        # Отдельный рендерер: тест намеренно ломает программу 'unlit'.
        resources = ResourceManager(roots=[ASSETS], name="BrokenProgram")
        renderer = Renderer(resources=resources, shader_root=ASSETS)
        renderer.initialize()
        try:
            mesh = GpuMesh.from_mesh_asset(cube_mesh(1.0))
            broken = renderer.create_material(
                "broken", program="unlit", base_color=_color(0.5, 0.5, 0.5)
            )
            broken.program.dispose()  # handle = 0
            target = RenderTarget("broken_program", 64, 64)

            renderer.begin_frame(camera, (64, 64), target=target)
            renderer.submit(mesh, broken, compose_matrix())
            with pytest.raises(RenderError, match="не собрана"):
                renderer.end_frame(target=target)
        finally:
            renderer.dispose()
            resources.clear()


class TestStateCacheWithContext:
    def test_repeated_state_calls_are_skipped(self, renderer) -> None:
        state = renderer.state
        state.invalidate()
        # Прогрев: доводим состояние до целевого (тут вызовы драйвера нужны).
        state.set_depth_test(True)
        state.set_cull(True)
        state.set_blend(False)
        calls_after_warmup = state.stats.calls
        saved_before = state.stats.saved
        for _ in range(20):
            state.set_depth_test(True)
            state.set_cull(True)
            state.set_blend(False)
        assert state.stats.calls == calls_after_warmup  # ни одного лишнего вызова
        assert state.stats.saved - saved_before >= 60


def _color(red: float, green: float, blue: float):
    """Создать Vector3-цвет (короткая запись для тестов)."""
    from engine.core.math3d import Vector3

    return Vector3(red, green, blue)


def _top_camera():
    """Камера, смотрящая на плоскость XZ сверху."""
    from engine.core.math3d import Vector3
    from engine.rendering import Camera

    view = Camera(fov=50.0, far=100.0)
    view.position = Vector3(0.0, 6.0, 0.001)
    view.look_at(Vector3.zero())
    return view


def _empty_mesh():
    """Меш без вершин (для проверки ошибки загрузки)."""
    from pathlib import Path as _Path

    from engine.core.resources import MeshAsset

    return MeshAsset(
        path=_Path("<test>/empty"),
        positions=np.zeros((0, 3), dtype=np.float32),
        name="empty",
    )


class TestSpriteBatch:
    def test_sprite_is_drawn(self, renderer, camera) -> None:
        from pathlib import Path

        from engine.core.resources import SpriteAtlas
        from engine.core.resources.cookers import SpriteFrame
        from engine.rendering import GpuTexture, SpriteBatch, SpritePass

        pixels = np.full((8, 8, 4), 255, dtype=np.uint8)
        texture = GpuTexture.from_bytes(
            "atlas", 8, 8, pixels, wrap_repeat=False, generate_mips=False
        )
        atlas = SpriteAtlas(8, 8, {"s": SpriteFrame("s", 0, 0, 8, 8)}, Path("x"))
        batch = SpriteBatch(atlas, pixels_per_unit=4.0, texture=texture)
        batch.add("s", (0.0, 1.0, 0.0), scale=2.0)
        renderer.add_pass(SpritePass(batch))
        target = render_to_target(renderer, camera)
        renderer.end_frame(target=target)
        assert count_visible(target) > 300
        batch.dispose()


class TestBakedLighting:
    def test_baked_ambient_is_not_applied_twice(self, renderer, camera) -> None:
        """Запечённый свет не умножается на u_ambient повторно."""
        from engine.core.math3d import Vector3, compose_matrix
        from engine.core.resources import uv_sphere_mesh
        from engine.rendering import AmbientLight, GpuMesh
        from engine.rendering.mesh import VertexLayout

        renderer.shadows_enabled = False
        renderer.lights.set_lights([AmbientLight(intensity=0.5)])
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
        material = renderer.create_material("baked", base_color=Vector3(1, 1, 1))
        material.dynamic_lighting = False
        camera.position = Vector3(0.0, 0.0, 6.0)
        camera.look_at(Vector3.zero())
        target = render_to_target(renderer, camera)
        renderer.submit(mesh, material, compose_matrix())
        renderer.end_frame(target=target)
        red = int(target.read_pixels()[60, 80, 0])
        # Ожидаем linear 0.5 -> sRGB ≈ 188; с двойным ambient было бы ≈ 224.
        assert 175 <= red <= 200
