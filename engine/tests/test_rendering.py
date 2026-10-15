"""Тесты рендеринга без OpenGL: форматы вершин, камера, шейдерные утилиты.

Эти тесты не требуют графического контекста и потому работают в любой среде
(в том числе в CI без GPU). Реальный рендер проверяется в
``test_rendering_gl.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.core.components import BUILTIN_MESHES
from engine.core.math3d import Quaternion, Vector3, compose_matrix
from engine.core.resources import (
    cube_mesh,
    cylinder_mesh,
    line_grid_mesh,
    plane_mesh,
    primitives,
    uv_sphere_mesh,
)
from engine.rendering.camera import Camera, OrbitCameraController, ProjectionMode
from engine.rendering.capabilities import Capabilities
from engine.rendering.errors import RenderError
from engine.rendering.mesh import GpuInstanceBuffer, GpuMesh, VertexLayout
from engine.rendering.render_graph import FrameContext, RenderGraph, RenderTarget
from engine.rendering.renderer import RenderStats, _transformed_bounds
from engine.rendering.screenshot import save_png
from engine.rendering.settings import QualityTier, RenderSettings, ShadowSettings
from engine.rendering.shader import inject_defines
from engine.rendering.state import GLState


class TestVertexLayout:
    def test_from_names_builds_offsets(self) -> None:
        layout = VertexLayout.from_names(("position", "normal", "uv"))
        assert layout.names == ("position", "normal", "uv")
        assert layout.stride == (3 + 3 + 2) * 4
        assert layout.attribute("normal").offset == 3 * 4
        assert layout.attribute("uv").offset == 6 * 4
        assert layout.components == 8

    def test_unknown_attribute_rejected(self) -> None:
        with pytest.raises(ValueError):
            VertexLayout.from_names(("position", "wibble"))

    def test_empty_layout_rejected(self) -> None:
        with pytest.raises(ValueError):
            VertexLayout.from_names(())

    def test_interleave_packs_in_order(self) -> None:
        layout = VertexLayout.from_names(("position", "uv"))
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        uvs = np.array([[0.25, 0.75], [0.5, 1.0]], dtype=np.float32)
        packed = layout.interleave({"position": positions, "uv": uvs})
        assert packed.shape == (2, 5)
        assert packed[0].tolist() == [1.0, 2.0, 3.0, 0.25, 0.75]
        assert packed[1].tolist() == [4.0, 5.0, 6.0, 0.5, 1.0]

    def test_missing_attribute_is_zero_filled(self) -> None:
        layout = VertexLayout.from_names(("position", "normal"))
        packed = layout.interleave({"position": np.ones((2, 3), dtype=np.float32)})
        assert packed.shape == (2, 6)
        assert packed[:, 3:].tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    def test_mismatched_vertex_count(self) -> None:
        layout = VertexLayout.from_names(("position",))
        with pytest.raises(ValueError):
            layout.interleave({"position": np.zeros((3, 3), dtype=np.float32)}, 5)

    def test_mismatched_components(self) -> None:
        layout = VertexLayout.from_names(("position",))
        with pytest.raises(ValueError):
            layout.interleave({"position": np.zeros((2, 2), dtype=np.float32)})


class TestGpuMeshData:
    def test_mesh_from_arrays_computes_bounds(self) -> None:
        mesh = GpuMesh.from_arrays(
            np.array([[-1, -1, -1], [1, 1, 1], [0, 0, 0]], dtype=np.float32)
        )
        minimum, maximum = mesh.bounds
        assert minimum.to_tuple() == (-1.0, -1.0, -1.0)
        assert maximum.to_tuple() == (1.0, 1.0, 1.0)
        assert mesh.vertex_count == 3
        assert mesh.triangle_count == 1
        assert not mesh.is_uploaded
        assert mesh.gpu_bytes > 0

    def test_mesh_from_cube_asset(self) -> None:
        asset = cube_mesh(2.0)
        mesh = GpuMesh.from_mesh_asset(asset)
        assert mesh.vertex_count == 24
        assert mesh.triangle_count == 12
        assert mesh.is_indexed
        assert mesh.layout.names == ("position", "normal", "uv")

    def test_repr_and_dispose_without_context(self) -> None:
        mesh = GpuMesh.from_mesh_asset(plane_mesh(1.0))
        assert "GpuMesh" in repr(mesh)
        mesh.dispose()  # без контекста ничего не делает и не падает

    def test_transformed_bounds_follow_matrix(self) -> None:
        mesh = GpuMesh.from_arrays(
            np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=np.float32)
        )
        matrix = compose_matrix(position=Vector3(10, 0, 0), scale=Vector3(2, 2, 2))
        center, radius = _transformed_bounds(mesh, matrix)
        assert center.almost_equals(Vector3(10, 0, 0), 1e-5)
        # Половина диагонали куба (sqrt(3)/2) с учётом масштаба 2.
        assert radius == pytest.approx(np.sqrt(3.0), abs=0.01)

    def test_instance_buffer_validates_shape(self) -> None:
        buffer = GpuInstanceBuffer(capacity=2)
        with pytest.raises(ValueError):
            buffer.upload(np.zeros((2, 3), dtype=np.float32))
        with pytest.raises(ValueError):
            buffer.upload(np.zeros((5, 4, 4), dtype=np.float32))
        with pytest.raises(ValueError):
            GpuInstanceBuffer(capacity=0)


class TestProceduralMeshes:
    def test_cube_has_24_vertices(self) -> None:
        asset = cube_mesh(1.0)
        assert asset.vertex_count == 24
        assert asset.triangle_count == 12
        assert asset.normals is not None
        minimum, maximum = asset.bounds
        assert minimum.to_tuple() == (-0.5, -0.5, -0.5)
        assert maximum.to_tuple() == (0.5, 0.5, 0.5)

    def test_sphere_is_unit_and_smooth(self) -> None:
        asset = uv_sphere_mesh(1.0, segments=16, rings=8)
        lengths = np.linalg.norm(asset.positions, axis=1)
        assert np.allclose(lengths, 1.0, atol=1e-4)
        assert asset.normals is not None
        assert asset.triangle_count == 16 * 8 * 2

    def test_sphere_rejects_bad_resolution(self) -> None:
        with pytest.raises(ValueError):
            uv_sphere_mesh(1.0, segments=2, rings=8)

    def test_plane_subdivisions(self) -> None:
        asset = plane_mesh(4.0, subdivisions=3)
        assert asset.vertex_count == 16
        assert asset.triangle_count == 18

    def test_cylinder_has_outward_winding_and_caps(self) -> None:
        """Цилиндр замкнут и обходится против часовой стрелки (лицо наружу)."""
        asset = cylinder_mesh(0.5, 1.0, segments=24)
        assert asset.normals is not None
        assert asset.vertex_count == 2 * 25 + 2 * 26
        assert asset.triangle_count == 24 * 2 + 24 * 2

        points = np.asarray(asset.positions, dtype=np.float64)
        indices = np.asarray(asset.indices, dtype=np.int64).reshape(-1, 3)
        normals = np.asarray(asset.normals, dtype=np.float64)
        first, second, third = (points[indices[:, i]] for i in range(3))
        geometric = np.cross(second - first, third - first)
        averaged = (normals[indices[:, 0]] + normals[indices[:, 1]] + normals[indices[:, 2]])
        assert np.all(np.einsum("ij,ij->i", geometric, averaged) > 0.0)

        minimum, maximum = asset.bounds
        assert minimum.almost_equals(Vector3(-0.5, -0.5, -0.5), 1e-6)
        assert maximum.almost_equals(Vector3(0.5, 0.5, 0.5), 1e-6)
        assert asset.uvs is not None
        assert asset.uvs.min() >= 0.0
        assert asset.uvs.max() <= 1.0

    def test_cylinder_rejects_bad_resolution(self) -> None:
        with pytest.raises(ValueError):
            cylinder_mesh(0.5, 1.0, segments=2)

    def test_every_builtin_mesh_has_a_generator(self) -> None:
        """У каждого имени из ``BUILTIN_MESHES`` есть рабочий генератор геометрии.

        Список встроенных мешей объявлен в ядре (его показывает инспектор),
        поэтому генератор обязан существовать для каждого имени: иначе редактор
        предложит меш, который не загрузится.
        """
        generators = {
            "cube": "cube_mesh",
            "sphere": "uv_sphere_mesh",
            "plane": "plane_mesh",
            "cylinder": "cylinder_mesh",
        }
        assert set(BUILTIN_MESHES) == set(generators), "список встроенных мешей изменился"
        for mesh_name, generator_name in generators.items():
            factory = getattr(primitives, generator_name, None)
            assert callable(factory), f"нет генератора для '{mesh_name}'"
            asset = factory()
            assert asset.vertex_count > 0
            assert asset.triangle_count > 0
            assert asset.normals is not None

    def test_grid_is_line_pairs(self) -> None:
        asset = line_grid_mesh(2.0, divisions=2)
        assert asset.vertex_count == 12  # (2+1) линий * 2 направления * 2 вершины
        assert asset.indices is None


class TestShaderUtilities:
    def test_inject_defines_after_version(self) -> None:
        source = "#version 330 core\nvoid main() {}\n"
        result = inject_defines(source, {"INSTANCED": 1})
        lines = result.splitlines()
        assert lines[0] == "#version 330 core"
        assert lines[1] == "#define INSTANCED 1"

    def test_inject_defines_without_version(self) -> None:
        result = inject_defines("void main() {}\n", {"FLAG": None})
        assert result.startswith("#define FLAG\n")

    def test_inject_defines_without_defines(self) -> None:
        source = "#version 330 core\nvoid main() {}\n"
        assert inject_defines(source, None) == source
        assert inject_defines(source, {}) == source


class TestRenderSettings:
    def test_quality_tier_defaults(self) -> None:
        low = QualityTier.LOW
        assert low.shadow_map_size == 512
        assert low.msaa_samples == 0
        assert not low.enable_post_effects
        assert QualityTier.HIGH.shadow_cascades == 3

    def test_for_quality_applies_overrides(self) -> None:
        settings = RenderSettings.for_quality(
            QualityTier.HIGH, samples=0, swap_interval=0
        )
        assert settings.samples == 0
        assert settings.swap_interval == 0
        assert settings.quality is QualityTier.HIGH

    def test_describe_mentions_profile(self) -> None:
        settings = RenderSettings()
        text = settings.describe()
        assert "GL 3.3" in text
        assert "core" in text

    def test_shadow_settings_defaults(self) -> None:
        shadows = ShadowSettings()
        assert shadows.enabled
        assert shadows.map_size == 1024


class TestCamera:
    def test_default_projection_uses_aspect(self) -> None:
        camera = Camera(fov=90.0, near=0.1, far=100.0, aspect=1.0)
        matrix = camera.projection_matrix
        assert matrix[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert matrix[1, 1] == pytest.approx(1.0, abs=1e-5)
        assert matrix[3, 2] == pytest.approx(-1.0, abs=1e-6)

    def test_invalid_projection_parameters(self) -> None:
        with pytest.raises(ValueError):
            _ = Camera(fov=0.0).projection_matrix
        with pytest.raises(ValueError):
            _ = Camera(near=10.0, far=1.0).projection_matrix

    def test_look_at_preserves_distance(self) -> None:
        camera = Camera()
        camera.position = Vector3(0, 0, 5)
        camera.look_at(Vector3.zero())
        assert camera.forward.almost_equals(Vector3(0, 0, -1), 1e-5)
        assert camera.world_to_view(Vector3.zero()).z == pytest.approx(-5.0, abs=1e-4)

    def test_view_projection_transforms_center(self) -> None:
        camera = Camera(fov=60.0, aspect=1.0, near=1.0, far=10.0)
        camera.position = Vector3(0, 0, 4)
        camera.look_at(Vector3.zero())
        x, y, z, w = camera.world_to_clip(Vector3.zero())
        assert w == pytest.approx(4.0, abs=1e-4)
        assert (x, y) == pytest.approx((0.0, 0.0), abs=1e-4)
        # NDC-глубина точки между плоскостями отсечения лежит в [-1, 1].
        assert -1.0 <= z / w <= 1.0

    def test_projection_maps_near_and_far_planes(self) -> None:
        camera = Camera(fov=60.0, aspect=1.0, near=2.0, far=20.0)
        camera.position = Vector3(0, 0, 0)
        camera.look_at(Vector3(0, 0, -1))
        _, _, near_z, near_w = camera.world_to_clip(Vector3(0, 0, -2.0))
        _, _, far_z, far_w = camera.world_to_clip(Vector3(0, 0, -20.0))
        assert near_z / near_w == pytest.approx(-1.0, abs=1e-4)
        assert far_z / far_w == pytest.approx(1.0, abs=1e-4)

    def test_set_aspect_updates_projection(self) -> None:
        camera = Camera(aspect=1.0)
        before = camera.projection_matrix.copy()
        camera.set_aspect(800, 400)
        assert camera.aspect == pytest.approx(2.0)
        assert not np.allclose(before, camera.projection_matrix)

    def test_orthographic_mode(self) -> None:
        camera = Camera(mode=ProjectionMode.ORTHOGRAPHIC, ortho_height=10.0, aspect=1.0)
        matrix = camera.projection_matrix
        assert matrix[2, 2] < 0.0
        # В ортографии w всегда 1: масштаб не зависит от глубины.
        _, _, _, w = camera.world_to_clip(Vector3(0, 0, -5))
        assert w == pytest.approx(1.0, abs=1e-6)

    def test_frustum_culling(self) -> None:
        camera = Camera(fov=60.0, aspect=1.0, near=1.0, far=100.0)
        camera.position = Vector3(0, 0, 10)
        camera.look_at(Vector3.zero())
        assert camera.is_sphere_visible(Vector3.zero(), 0.5)
        assert not camera.is_sphere_visible(Vector3(0, 0, 50), 0.5)  # за спиной
        assert not camera.is_sphere_visible(Vector3(200, 0, 0), 0.5)  # сбоку

    def test_screen_ray_points_through_center(self) -> None:
        camera = Camera(fov=60.0)
        camera.position = Vector3(0, 0, 5)
        camera.look_at(Vector3.zero())
        origin, direction = camera.ray_from_screen(400, 300, 800, 600)
        assert origin.almost_equals(Vector3(0, 0, 5), 1e-5)
        assert direction.almost_equals(Vector3(0, 0, -1), 1e-4)

    def test_repr(self) -> None:
        assert "Camera" in repr(Camera())


class TestOrbitController:
    def test_orbit_changes_angles(self) -> None:
        controller = OrbitCameraController()
        yaw_before, pitch_before = controller.yaw, controller.pitch
        controller.orbit(10.0, 5.0)
        assert controller.yaw != yaw_before
        assert controller.pitch > pitch_before

    def test_pitch_is_clamped(self) -> None:
        controller = OrbitCameraController()
        controller.orbit(0.0, 10_000.0)
        assert controller.pitch <= controller.max_pitch
        controller.orbit(0.0, -20_000.0)
        assert controller.pitch >= controller.min_pitch

    def test_zoom_is_clamped(self) -> None:
        controller = OrbitCameraController(min_distance=1.0, max_distance=10.0)
        controller.zoom(100.0)
        assert controller.distance == pytest.approx(1.0)
        controller.zoom(-100.0)
        assert controller.distance == pytest.approx(10.0)

    def test_apply_places_camera_around_target(self) -> None:
        controller = OrbitCameraController(
            target=Vector3(1, 2, 3), distance=5.0, yaw=0.0, pitch=0.0
        )
        camera = Camera()
        controller.apply(camera)
        offset = (camera.position - Vector3(1, 2, 3)).length
        assert offset == pytest.approx(5.0, rel=1e-5)

    def test_pan_moves_target_perpendicular_to_view(self) -> None:
        controller = OrbitCameraController(target=Vector3.zero(), distance=10.0)
        camera = Camera()
        controller.apply(camera)
        before = controller.target
        controller.pan(50.0, 0.0, camera)
        assert not controller.target.almost_equals(before, 1e-6)

    def test_focus_on_sets_distance(self) -> None:
        controller = OrbitCameraController()
        controller.focus_on(Vector3(5, 0, 5), radius=2.0)
        assert controller.target.almost_equals(Vector3(5, 0, 5), 1e-6)
        assert controller.distance == pytest.approx(6.0)

    def test_reset_restores_defaults(self) -> None:
        controller = OrbitCameraController()
        controller.orbit(120.0, 40.0)
        controller.zoom(3.0)
        controller.reset()
        assert controller.distance == pytest.approx(6.0)
        assert controller.yaw == pytest.approx(45.0)


class TestRenderGraph:
    def test_passes_execute_in_order(self) -> None:
        order: list[str] = []

        class Recorder:
            def __init__(self, name: str) -> None:
                self._name = name

            @property
            def name(self) -> str:
                return self._name

            def execute(self, context: FrameContext) -> None:
                order.append(self._name)

        graph = RenderGraph("test")
        graph.add_pass(Recorder("first"))
        graph.add_pass(Recorder("second"))
        context = FrameContext(
            camera=Camera(), light=_light(), state=GLState(), viewport=(64, 64)
        )
        assert graph.execute(context) == 2
        assert order == ["first", "second"]
        assert graph.frame_index == 1
        assert [item.name for item in graph.passes] == ["first", "second"]
        assert graph.pass_by_name("second") is not None

    def test_remove_and_clear(self) -> None:
        class Pass:
            name = "p"

            def execute(self, context: FrameContext) -> None:
                pass

        graph = RenderGraph()
        render_pass = Pass()
        graph.add_pass(render_pass)
        assert graph.remove_pass(render_pass)
        assert not graph.remove_pass(render_pass)
        graph.add_pass(render_pass)
        graph.clear()
        assert len(graph) == 0


class TestTargetAndScreenshot:
    def test_target_is_lazy(self) -> None:
        target = RenderTarget("t", 32, 16)
        assert not target.is_created
        assert target.aspect == pytest.approx(2.0)
        assert "RenderTarget" in repr(target)

    def test_read_pixels_requires_creation(self) -> None:
        with pytest.raises(RenderError):
            RenderTarget("t", 4, 4).read_pixels()

    def test_save_png_roundtrip(self, tmp_path) -> None:
        pixels = np.zeros((4, 3, 4), dtype=np.uint8)
        pixels[:, :, 0] = 255
        path = save_png(tmp_path / "frame.png", pixels)
        data = path.read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert data.endswith(b"IEND\xaeB`\x82")
        assert path.stat().st_size > 50

    def test_save_png_rejects_bad_shape(self, tmp_path) -> None:
        with pytest.raises(RenderError):
            save_png(tmp_path / "bad.png", np.zeros((4, 4), dtype=np.uint8))


class TestStatsAndCapabilities:
    def test_stats_reset(self) -> None:
        stats = RenderStats()
        stats.draw_calls = 5
        stats.triangles = 100
        stats.reset()
        assert stats.draw_calls == 0
        assert "draws 0" in stats.describe()

    def test_capabilities_can_defer(self) -> None:
        caps = Capabilities(
            supports_multiple_render_targets=True, max_texture_size=4096
        )
        assert caps.can_defer_shading
        caps.max_texture_size = 512
        assert not caps.can_defer_shading
        assert "MRT" in caps.describe()


def _light():
    """Источник света по умолчанию (для контекста кадра в тестах)."""
    from engine.rendering.material import DirectionalLight

    return DirectionalLight()


def test_quaternion_from_forward_handles_opposite_direction() -> None:
    from engine.rendering.camera import quaternion_from_forward

    forward = quaternion_from_forward(Vector3(0, 0, -1))
    assert forward.rotate_vector(Vector3.forward()).almost_equals(
        Vector3(0, 0, -1), 1e-5
    )
    up = quaternion_from_forward(Vector3.up())
    assert up.rotate_vector(Vector3.forward()).almost_equals(Vector3.up(), 1e-5)


def test_compose_matrix_rotation_matches_quaternion() -> None:
    rotation = Quaternion.from_euler(0, 90, 0)
    matrix = compose_matrix(rotation=rotation)
    rotated = matrix.astype(np.float32) @ np.array(
        [0.0, 0.0, 1.0, 1.0], dtype=np.float32
    )
    assert Vector3.from_numpy(rotated[:3]).almost_equals(Vector3(1, 0, 0), 1e-5)
