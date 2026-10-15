"""Подсистема рендеринга: OpenGL 3.3 Core, forward-путь, камера и вьюпорт.

Слои внутри пакета:

* :mod:`engine.rendering.context` — порт графического контекста (кто владеет GL);
* :mod:`engine.rendering.state` — кэш состояний: меньше вызовов драйвера;
* :mod:`engine.rendering.shader`, :mod:`~engine.rendering.mesh`,
  :mod:`~engine.rendering.texture`, :mod:`~engine.rendering.material` — ресурсы GPU;
* :mod:`engine.rendering.camera` — камеры: компонент, стек, фрустум,
  рендер в текстуру и орбитальное управление редактора;
* :mod:`engine.rendering.render_graph` — проходы и цели вывода (путь к deferred);
* :mod:`engine.rendering.renderer` — forward-рендерер с очередью и батчингом;
* :mod:`engine.rendering.qt_context`, :mod:`~engine.rendering.qt_viewport` —
  адаптеры PySide6 (импортируются только редактором).

Пакет не импортирует Qt на уровне модуля: плеер и headless-сборка работают
только с PyOpenGL.
"""

from __future__ import annotations

from engine.rendering.camera import (
    ASPECT_FIXED,
    ASPECT_VIEWPORT,
    Camera,
    CameraComponent,
    CameraPreview,
    CameraStack,
    EditorCameraController,
    EditorCameraSession,
    EditorCameraState,
    Frustum,
    OrbitCameraController,
    ProjectionMode,
)
from engine.rendering.capabilities import Capabilities, query_capabilities
from engine.rendering.context import (
    ContextInfo,
    IRenderContextProvider,
    IRenderSurface,
    query_context_info,
)
from engine.rendering.errors import (
    MeshUploadError,
    RenderError,
    ShaderCompileError,
    ShaderLinkError,
    UnsupportedFeatureError,
)
from engine.rendering.gl_debug import check_gl_error, enable_debug_output
from engine.rendering.lighting import (
    ALL_LAYERS_MASK,
    DEFAULT_MAX_LIGHTS,
    DEFAULT_SHADOW_MAP_SIZE,
    LAYERS_ALL,
    LIGHT_TYPES,
    AmbientLight,
    DirectionalLight,
    Light,
    LightBuffer,
    LightCacheStats,
    LightComponent,
    LightManager,
    LightSample,
    LightSet,
    PointLight,
    SpotLight,
    create_light,
    light_from_dict,
)
from engine.rendering.material import Material
from engine.rendering.material_loader import load_pbr_material
from engine.rendering.mesh import (
    GpuInstanceBuffer,
    GpuMesh,
    VertexAttribute,
    VertexLayout,
)
from engine.rendering.pbr_material import PbrMaterial
from engine.rendering.program_cache import ProgramBinaryCache, warm_program_cache
from engine.rendering.render_graph import (
    FrameContext,
    RenderGraph,
    RenderPass,
    RenderTarget,
)
from engine.rendering.renderer import DrawItem, InstanceGroup, Renderer, RenderStats
from engine.rendering.screenshot import capture_framebuffer, read_framebuffer, save_png
from engine.rendering.settings import QualityTier, RenderSettings, ShadowSettings
from engine.rendering.shader import Shader, ShaderProgram, load_program
from engine.rendering.shadow import ShadowMap
from engine.rendering.sprite_batch import SpriteBatch, SpritePass
from engine.rendering.state import GLState, StateStats
from engine.rendering.texture import GpuTexture, TextureFilter

__all__ = [
    "ALL_LAYERS_MASK",
    "ASPECT_FIXED",
    "ASPECT_VIEWPORT",
    "AmbientLight",
    "Camera",
    "CameraComponent",
    "CameraPreview",
    "CameraStack",
    "Capabilities",
    "ContextInfo",
    "DEFAULT_MAX_LIGHTS",
    "DEFAULT_SHADOW_MAP_SIZE",
    "DirectionalLight",
    "DrawItem",
    "EditorCameraController",
    "EditorCameraSession",
    "EditorCameraState",
    "FrameContext",
    "Frustum",
    "GLState",
    "GpuInstanceBuffer",
    "GpuMesh",
    "GpuTexture",
    "IRenderContextProvider",
    "IRenderSurface",
    "InstanceGroup",
    "LAYERS_ALL",
    "LIGHT_TYPES",
    "Light",
    "LightBuffer",
    "LightCacheStats",
    "LightComponent",
    "LightManager",
    "LightSample",
    "LightSet",
    "Material",
    "load_pbr_material",
    "PbrMaterial",
    "ProgramBinaryCache",
    "MeshUploadError",
    "OrbitCameraController",
    "PointLight",
    "ProjectionMode",
    "QualityTier",
    "RenderError",
    "RenderGraph",
    "RenderPass",
    "RenderSettings",
    "RenderStats",
    "RenderTarget",
    "Renderer",
    "warm_program_cache",
    "Shader",
    "ShaderCompileError",
    "ShaderLinkError",
    "ShaderProgram",
    "ShadowMap",
    "ShadowSettings",
    "SpotLight",
    "SpriteBatch",
    "SpritePass",
    "StateStats",
    "TextureFilter",
    "UnsupportedFeatureError",
    "VertexAttribute",
    "VertexLayout",
    "capture_framebuffer",
    "check_gl_error",
    "create_light",
    "enable_debug_output",
    "light_from_dict",
    "load_program",
    "query_capabilities",
    "query_context_info",
    "read_framebuffer",
    "save_png",
]
