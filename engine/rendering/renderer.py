"""Forward-рендерер: освещение, очередь отрисовки, батчинг и статистика.

Устройство:

* **Очередь** (:class:`DrawItem`) собирается вызовами :meth:`Renderer.submit` и
  сортируется: непрозрачные — по материалу (минимум переключений состояния),
  прозрачные — от дальних к ближним.
* **Отсечение**: если у камеры задан усечённый объём, объекты вне него не
  попадают в очередь (`stats.culled`).
* **Батчинг**: одинаковые материалы рисуются подряд, а повторяющаяся геометрия
  может рисоваться одним вызовом через инстансинг
  (:meth:`Renderer.draw_instanced`) — фундамент для травы, деревьев и пропсов.
* **Граф проходов** (:class:`~engine.rendering.render_graph.RenderGraph`):
  очистка и геометрия — это проходы, поэтому тени, G-буфер и пост-обработка
  добавляются как новые элементы графа, без правок этого файла.

Example:
    >>> renderer = Renderer(resources=resources)
    >>> renderer.initialize()
    >>> material = renderer.create_material("crate", base_color=Vector3(0.8, 0.5, 0.2))
    >>> renderer.begin_frame(camera, viewport=(1280, 720))
    >>> renderer.submit(cube, material, cube.transform.world_matrix)
    >>> stats = renderer.end_frame()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from engine.core.math3d import (
    MATRIX_DTYPE,
    Vector3,
    extract_translation,
    identity_matrix,
)
from engine.rendering.camera import Camera, CameraStack
from engine.rendering.capabilities import Capabilities, query_capabilities
from engine.rendering.context import ContextInfo, query_context_info
from engine.rendering.errors import RenderError
from engine.rendering.lighting import (
    DEFAULT_MAX_LIGHTS,
    DEFAULT_SHADOW_MAP_SIZE,
    LightManager,
)
from engine.rendering.material import DirectionalLight, Material
from engine.rendering.mesh import GpuInstanceBuffer, GpuMesh
from engine.rendering.pbr_material import PbrMaterial
from engine.rendering.post import PostSettings, PostStack
from engine.rendering.program_cache import ProgramBinaryCache
from engine.rendering.render_graph import (
    FrameContext,
    RenderGraph,
    RenderPass,
    RenderTarget,
)
from engine.rendering.shader import ShaderProgram, get_program_cache, set_program_cache
from engine.rendering.shadow import ShadowMap
from engine.rendering.state import GLState
from engine.rendering.texture import GpuTexture

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.resources import ResourceManager
    from engine.core.screen_fx import ScreenFx
    from engine.rendering.settings import RenderSettings
    from engine.rendering.sky import SkyPass

__all__ = ["DrawItem", "InstanceGroup", "Renderer", "RenderStats"]

_log = logging.getLogger("engine.rendering.renderer")

DEFAULT_VERTEX_SHADER = "shaders/simple.vert"
DEFAULT_FRAGMENT_SHADER = "shaders/simple.frag"


@dataclass(slots=True)
class RenderStats:
    """Статистика кадра (для панели Statistics и бюджетов производительности)."""

    submitted: int = 0
    draw_calls: int = 0
    batches: int = 0
    triangles: int = 0
    vertices: int = 0
    instances: int = 0
    culled: int = 0
    state_calls: int = 0
    state_saved: int = 0
    frame_index: int = 0
    #: Освещение: сколько источников передано объектам и что дала карта теней.
    lights_bound: int = 0
    shadow_draws: int = 0
    shadow_triangles: int = 0

    def reset(self) -> None:
        """Обнулить счётчики кадра."""
        self.submitted = 0
        self.draw_calls = 0
        self.batches = 0
        self.triangles = 0
        self.vertices = 0
        self.instances = 0
        self.culled = 0
        self.lights_bound = 0
        self.shadow_draws = 0
        self.shadow_triangles = 0

    def describe(self) -> str:
        """Краткое описание для строки статуса."""
        shadow = (
            f", shadow {self.shadow_draws} draws/{self.shadow_triangles} tri"
            if self.shadow_draws
            else ""
        )
        return (
            f"draws {self.draw_calls} (batches {self.batches}), "
            f"tri {self.triangles}, verts {self.vertices}, "
            f"inst {self.instances}, culled {self.culled}, "
            f"lights {self.lights_bound}{shadow}"
        )


@dataclass(slots=True)
class DrawItem:
    """Элемент очереди отрисовки."""

    mesh: GpuMesh
    material: Material
    model_matrix: np.ndarray
    sort_key: tuple[int, int, int]
    distance: float = 0.0
    center: Vector3 = field(default_factory=Vector3.zero)
    radius: float = 0.0
    mode: int | None = None
    cast_shadow: bool = True
    receive_shadow: bool = True


@dataclass(slots=True)
class InstanceGroup:
    """Группа экземпляров одной геометрии с общим материалом."""

    mesh: GpuMesh
    material: Material
    matrices: list[np.ndarray] = field(default_factory=list)
    center: Vector3 = field(default_factory=Vector3.zero)
    radius: float = 0.0

    @property
    def count(self) -> int:
        """Число экземпляров в группе."""
        return len(self.matrices)


@dataclass(slots=True)
class _InstanceDraw:
    """Группа экземпляров, поставленная в очередь текущего кадра."""

    mesh: GpuMesh
    material: Material
    matrices: np.ndarray
    sort_key: tuple[int, int, int]
    center: Vector3
    radius: float
    distance: float = 0.0
    cast_shadow: bool = True
    receive_shadow: bool = True


class _ClearPass:
    """Проход очистки: заполняет цвет и глубину перед геометрией.

    Флаги ``clears_color`` и ``clears_depth`` позволяют стеку камер рисовать
    UI-камеру поверх игровой: верхний слой очищает только глубину.
    """

    __slots__ = (
        "clear_color",
        "clear_depth",
        "clears_color",
        "clears_depth",
        "enabled",
    )

    def __init__(self, clear_color: tuple[float, float, float, float]) -> None:
        self.clear_color = clear_color
        self.clear_depth = 1.0
        self.clears_color = True
        self.clears_depth = True
        self.enabled = True

    @property
    def name(self) -> str:
        """Имя прохода."""
        return "clear"

    def execute(self, context: FrameContext) -> None:
        """Очистить цветовой буфер и буфер глубины (что разрешено)."""
        if not self.enabled:
            return
        from OpenGL import GL

        mask = 0
        if self.clears_color:
            GL.glClearColor(*self.clear_color)
            mask |= GL.GL_COLOR_BUFFER_BIT
        if self.clears_depth:
            context.state.set_depth_write(True)
            GL.glClearDepth(self.clear_depth)
            mask |= GL.GL_DEPTH_BUFFER_BIT
        if mask:
            GL.glClear(mask)


class _ShadowPass:
    """Проход карты теней: рисует непрозрачную геометрию со стороны солнца.

    Проход дешёвый по построению: цвет не пишется, освещение не считается,
    шейдеры минимальны. Инстансные пачки рисуются тем же инстансным вариантом,
    что и основной проход, поэтому тень не требует распада на отдельные вызовы.
    """

    __slots__ = ("_renderer",)

    def __init__(self, renderer: Renderer) -> None:
        self._renderer = renderer

    @property
    def name(self) -> str:
        """Имя прохода."""
        return "shadow"

    def execute(self, context: FrameContext) -> None:
        """Отрисовать карту теней для активного солнца."""
        self._renderer._draw_shadow_map(context)  # noqa: SLF001 - проход часть рендерера


class _GeometryPass:
    """Проход геометрии: рисует очередь, собранную рендерером.

    Проход держит ссылку на рендерер, чтобы использовать его программы,
    свет и статистику — сам он отвечает только за порядок и состояние.
    """

    __slots__ = ("_renderer",)

    def __init__(self, renderer: Renderer) -> None:
        self._renderer = renderer

    @property
    def name(self) -> str:
        """Имя прохода."""
        return "geometry"

    def execute(self, context: FrameContext) -> None:
        """Отрисовать непрозрачную и прозрачную части очереди."""
        self._renderer._draw_queue(context)  # noqa: SLF001 - проход часть рендерера


class _TransparentPass:
    """Проход прозрачной геометрии: идёт после неба, чтобы смешиваться с ним."""

    __slots__ = ("_renderer",)

    def __init__(self, renderer: Renderer) -> None:
        self._renderer = renderer

    @property
    def name(self) -> str:
        """Имя прохода."""
        return "transparent"

    def execute(self, context: FrameContext) -> None:
        """Отрисовать прозрачную часть очереди."""
        self._renderer._draw_transparent(context)  # noqa: SLF001 - часть рендерера


class Renderer:
    """Forward-рендерер с направленным светом.

    Args:
        resources: Менеджер ресурсов для загрузки шейдеров; если не задан,
            программы собираются из встроенных исходников.
        settings: Настройки рендеринга (используются для параметров по умолчанию).
        shader_root: Каталог с шейдерами по умолчанию.
        cull_by_frustum: Отсекать объекты вне усечённого объёма камеры.
        program_cache_dir: Каталог дискового кэша бинарников программ
            (``glGetProgramBinary``); ``None`` — без кэша.
    """

    def __init__(
        self,
        *,
        resources: ResourceManager | None = None,
        settings: RenderSettings | None = None,
        shader_root: Path | str | None = None,
        cull_by_frustum: bool = True,
        program_cache_dir: Path | str | None = None,
        name: str = "ForwardRenderer",
    ) -> None:
        self.name = name
        self.resources = resources
        self.settings = settings
        self.shader_root = Path(shader_root) if shader_root is not None else None
        self.cull_by_frustum = cull_by_frustum
        self.program_cache: ProgramBinaryCache | None = (
            ProgramBinaryCache(program_cache_dir)
            if program_cache_dir is not None
            else None
        )

        self.state = GLState()
        self.graph = RenderGraph("forward")
        self.light = DirectionalLight()
        self.stats = RenderStats()
        self.info = ContextInfo()
        self.capabilities = Capabilities()
        self.clear_color: tuple[float, float, float, float] = (0.09, 0.10, 0.13, 1.0)

        self._programs: dict[str, ShaderProgram] = {}
        self._items: list[DrawItem] = []
        self._instances: list[_InstanceDraw] = []
        self._pending_transparent: list[DrawItem | _InstanceDraw] = []
        self._instance_buffers: dict[int, GpuInstanceBuffer] = {}
        self._camera: Camera | None = None
        self._stack: CameraStack | None = None
        self._context: FrameContext | None = None
        self._active_material: Material | None = None
        self._active_program: ShaderProgram | None = None
        self._viewport: tuple[int, int] = (1, 1)
        self._clear_pass: _ClearPass | None = None
        self._initialized = False
        # Программы, чья привязка уже сверена с драйвером в этом кадре.
        self._verified_programs: set[int] = set()
        # Варианты основного шейдера освещения: ключ — набор ``#define``.
        self._lit_sources: dict[object, str | None] = {}
        self._variants: dict[tuple, ShaderProgram] = {}
        # PBR-программа собирается лениво (первый PBR-материал): сцены без
        # PBR не платят за вторую компиляцию крупного шейдера.
        self._pbr_sources: dict[object, str | None] = {}
        self._pbr_variants: dict[tuple, ShaderProgram] = {}
        # Освещение: менеджер света можно подменить снаружи (см. ``lighting``).
        self.lights = LightManager(
            max_lights_per_object=DEFAULT_MAX_LIGHTS,
            cast_shadows=bool(settings.quality.shadow_map_size) if settings else True,
            shadow_map_size=(
                settings.quality.shadow_map_size
                if settings is not None
                else DEFAULT_SHADOW_MAP_SIZE
            ),
        )
        self.shadow_map: ShadowMap | None = None
        self.shadows_enabled = True
        self._shadow_pass: _ShadowPass | None = None
        self._shadow_pass_ready = False
        self._shadow_applied: set[int] = set()
        #: Последнее значение «объект принимает тени», записанное в программу.
        self._receive_applied: dict[int, bool] = {}
        self._shadow_drawn = 0
        #: Цель вывода текущего кадра (``None`` — экран).
        self._frame_target: RenderTarget | None = None
        self._light_set_key: tuple | None = None
        #: HDR-кадр (пост-обработка): запрос и состояние (A10, docs/RENDERING.md).
        self.hdr_enabled = bool(settings.hdr) if settings is not None else False
        #: Доля разрешения сцены относительно окна (0.5–1.0).
        self.render_scale = 1.0
        self._hdr_target: RenderTarget | None = None
        self._final_target: RenderTarget | None = None
        self._post_active = False
        self._full_viewport: tuple[int, int] = (1, 1)
        #: Fused tonemap + FXAA + fade/flash (``post_settings``/``screen_fx``).
        self.post = PostStack(self)
        # Кэш uniform-переменных камеры: матрицы пересчитываются только при
        # смене камеры или её параметров, а не на каждый объект очереди.
        self._camera_uniform_key: tuple[int, int, int] | None = None
        #: handle программы -> ключ камеры, с которым её uniform уже отправлены.
        self._camera_sent: dict[int, tuple[int, int, int] | None] = {}
        self._camera_uniform_values: (
            tuple[np.ndarray, np.ndarray, np.ndarray] | None
        ) = None

    # ------------------------------------------------------------------ #
    # Инициализация
    # ------------------------------------------------------------------ #
    def initialize(self, *, probe_driver: bool = True) -> Renderer:
        """Подготовить драйвер, программы и граф проходов.

        Требует активного контекста OpenGL.

        Args:
            probe_driver: Читать сведения о драйвере и его возможностях.

        Returns:
            Сам рендерер (для цепочки вызовов).
        """
        if self._initialized:
            return self
        if probe_driver:
            self.info = query_context_info()
            self.capabilities = query_capabilities()
            _log.info("Контекст: %s", self.info.describe())
            _log.info("Возможности: %s", self.capabilities.describe())

        if self.program_cache is not None:
            set_program_cache(self.program_cache)
        self._ensure_default_programs()
        self._clear_pass = _ClearPass(self.clear_color)
        self._shadow_pass = _ShadowPass(self)
        self.graph = RenderGraph("forward")
        self.graph.add_pass(self._clear_pass)
        self.graph.add_pass(self._shadow_pass)
        self.graph.add_pass(_GeometryPass(self))
        self.graph.add_pass(_TransparentPass(self))
        self._initialized = True
        return self

    def _ensure_default_programs(self) -> None:
        """Собрать программы по умолчанию: ``lit``, ``unlit`` и тень."""
        if "lit" in self._programs:
            return
        program = self._load_default_program()
        self._programs["lit"] = program
        self._lit_sources = {
            _stage_vertex(): program.sources.get(_stage_vertex()),
            _stage_fragment(): program.sources.get(_stage_fragment()),
        }
        # Инстансный вариант — тот же код с ключевым словом INSTANCED:
        # так работает механизм вариантов шейдеров (см. §7 архитектуры).
        # Собирается через :meth:`variant`, чтобы имя ключевого слова совпадало
        # с тем, что просят материалы: иначе для одной и той же комбинации
        # собирались бы две разные программы (проверено тестом).
        vertex = self._lit_sources[_stage_vertex()]
        fragment = self._lit_sources[_stage_fragment()]
        if vertex is not None and fragment is not None:
            self._programs["lit_instanced"] = self.variant(INSTANCED=1)
        unlit = self._load_program_from_files(
            "unlit", "shaders/unlit.vert", "shaders/unlit.frag"
        )
        if unlit is not None:
            self._programs["unlit"] = unlit
        sprite = self._load_program_from_files(
            "sprite", "shaders/sprite.vert", "shaders/sprite.frag"
        )
        if sprite is not None:
            self._programs["sprite"] = sprite
        text = self._load_program_from_files(
            "text", "shaders/text.vert", "shaders/text.frag"
        )
        if text is not None:
            self._programs["text"] = text
        ui = self._load_program_from_files("ui", "shaders/ui.vert", "shaders/ui.frag")
        if ui is not None:
            self._programs["ui"] = ui
        sky = self._load_program_from_files(
            "sky", "shaders/sky.vert", "shaders/sky.frag"
        )
        if sky is not None:
            self._programs["sky"] = sky
        for name in ("tonemap", "fxaa", "fade", "bloom_bright", "blur", "bloom_up"):
            program = self._load_program_from_files(
                name, "shaders/post/fullscreen.vert", f"shaders/post/{name}.frag"
            )
            if program is not None:
                self._programs[name] = program
        shadow = self._load_program_from_files(
            "shadow", "shaders/shadow.vert", "shaders/shadow.frag"
        )
        if shadow is not None:
            self._programs["shadow"] = shadow
            vertex_source = shadow.sources.get(_stage_vertex())
            fragment_source = shadow.sources.get(_stage_fragment())
            if vertex_source is not None and fragment_source is not None:
                # Теневая программа INSTANCED не участвует в кэше вариантов
                # освещения: у неё своя пара исходников.
                self._programs["shadow_instanced"] = ShaderProgram.from_sources(
                    "shadow_instanced",
                    vertex=vertex_source,
                    fragment=fragment_source,
                    defines={"INSTANCED": 1},
                )

    def variant(self, **defines: object) -> ShaderProgram:
        """Собрать (или взять из кэша) вариант основного шейдера освещения.

        Варианты — это ``#define``, вставленные после ``#version``. Их набор
        определяется свойствами материала и формата меша:

        * ``INSTANCED`` — рисование пачкой экземпляров;
        * ``VERTEX_COLOR`` — меш принёс вершинные цвета (запечённый свет);
        * ``USE_LIGHTMAP`` — освещение берётся из лайтмапы;
        * ``DYNAMIC_LIGHTING=0`` — материал переведён на запечённый свет.

        Программы собираются по требованию и живут до :meth:`dispose`: на
        слабом GPU лишний вариант означает лишнюю компиляцию при старте, а
        нужен он далеко не всегда.

        Args:
            **defines: Ключевые слова варианта (порядок не важен).

        Returns:
            Скомпилированная программа варианта.

        Raises:
            RenderError: Если исходники основного шейдера не загружены.
        """
        key = tuple(sorted((name, str(value)) for name, value in defines.items()))
        if not key:
            # Базовый вариант уже собран при инициализации: не тратим на него
            # вторую компиляцию (на слабом GPU каждая лишняя программа заметна).
            return self.program("lit")
        cached = self._variants.get(key)
        if cached is not None:
            return cached
        vertex = self._lit_sources.get(_stage_vertex())
        fragment = self._lit_sources.get(_stage_fragment())
        if vertex is None or fragment is None:
            raise RenderError("Исходники основного шейдера не загружены")
        suffix = "".join(f"_{define.lower()}" for define, _ in key)
        name = f"lit{suffix}" if suffix else "lit"
        program = ShaderProgram.from_sources(
            name, vertex=vertex, fragment=fragment, defines=dict(defines)
        )
        self._variants[key] = program
        return program

    def pbr_variant(self, **defines: object) -> ShaderProgram:
        """Собрать (или взять из кэша) вариант PBR-шейдера.

        Ключевые слова: ``INSTANCED``, ``VERTEX_COLOR``, ``ALPHA_MASK``,
        ``DYNAMIC_LIGHTING=0`` (см. ``pbr.frag``). Базовый вариант без
        ключевых слов — программа ``pbr``.

        Raises:
            RenderError: Если шейдеры PBR не найдены или не собираются.
        """
        base = self._ensure_pbr_program()
        key = tuple(sorted((name, str(value)) for name, value in defines.items()))
        if not key:
            return base
        cached = self._pbr_variants.get(key)
        if cached is not None:
            return cached
        vertex = self._pbr_sources.get(_stage_vertex())
        fragment = self._pbr_sources.get(_stage_fragment())
        if vertex is None or fragment is None:
            raise RenderError("Исходники PBR-шейдера не загружены")
        suffix = "".join(f"_{define.lower()}" for define, _ in key)
        program = ShaderProgram.from_sources(
            f"pbr{suffix}", vertex=vertex, fragment=fragment, defines=dict(defines)
        )
        self._pbr_variants[key] = program
        return program

    def _ensure_pbr_program(self) -> ShaderProgram:
        """Собрать базовую PBR-программу при первом обращении."""
        program = self._programs.get("pbr")
        if program is not None:
            return program
        program = self._load_program_from_files(
            "pbr", DEFAULT_VERTEX_SHADER, "shaders/pbr.frag"
        )
        if program is None:
            raise RenderError(
                "Не удалось загрузить shaders/pbr.frag: PBR-материалы недоступны"
            )
        self._programs["pbr"] = program
        self._pbr_sources = {
            _stage_vertex(): program.sources.get(_stage_vertex()),
            _stage_fragment(): program.sources.get(_stage_fragment()),
        }
        return program

    def _load_program_from_files(
        self, name: str, vertex_path: str, fragment_path: str
    ) -> ShaderProgram | None:
        """Собрать программу из файлов, вернув ``None`` при неудаче.

        Вспомогательные программы (например, ``unlit`` для сетки) не должны
        ломать инициализацию рендерера.

        Returns:
            Программа или ``None``, если файлы не найдены.
        """
        from engine.core.resources import ResourceManager, ShaderAsset

        manager = self.resources
        if manager is None:
            root = self.shader_root or _default_shader_root()
            manager = ResourceManager(roots=[root] if root else [])
        try:
            vertex = manager.load(vertex_path)
            fragment = manager.load(fragment_path)
        except Exception as exc:  # noqa: BLE001 - программа необязательна
            _log.debug("Программа '%s' недоступна: %s", name, exc)
            return None
        if not isinstance(vertex, ShaderAsset) or not isinstance(fragment, ShaderAsset):
            return None
        return ShaderProgram.from_assets(vertex, fragment, name=name)

    def _load_default_program(self) -> ShaderProgram:
        """Загрузить шейдеры по умолчанию и собрать программу."""
        from engine.core.resources import ResourceManager, ShaderAsset

        root = self.shader_root or _default_shader_root()
        manager = self.resources
        if manager is None:
            manager = ResourceManager(roots=[root] if root else [])
        vertex_path = DEFAULT_VERTEX_SHADER
        fragment_path = DEFAULT_FRAGMENT_SHADER
        try:
            vertex = manager.load(vertex_path)
            fragment = manager.load(fragment_path)
        except Exception as exc:  # noqa: BLE001 - сообщаем понятную причину
            raise RenderError(
                f"Не удалось загрузить шейдеры '{vertex_path}'/'{fragment_path}': {exc}"
            ) from exc
        if not isinstance(vertex, ShaderAsset) or not isinstance(fragment, ShaderAsset):
            raise RenderError("Шейдеры по умолчанию должны быть GLSL-ассетами")
        self.shader_root = root
        return ShaderProgram.from_assets(vertex, fragment, name="lit")

    # ------------------------------------------------------------------ #
    # Программы и материалы
    # ------------------------------------------------------------------ #
    def register_program(self, name: str, program: ShaderProgram) -> ShaderProgram:
        """Зарегистрировать программу под именем (``lit``, ``unlit``, ...)."""
        self._programs[name] = program
        return program

    def program(self, name: str = "lit") -> ShaderProgram:
        """Вернуть программу по имени.

        Raises:
            RenderError: Если программа не зарегистрирована.
        """
        program = self._programs.get(name)
        if program is None and name == "pbr":
            return self._ensure_pbr_program()
        if program is None:
            raise RenderError(
                f"Программа '{name}' не зарегистрирована; доступны: "
                f"{', '.join(sorted(self._programs)) or 'нет'}"
            )
        return program

    @property
    def programs(self) -> dict[str, ShaderProgram]:
        """Зарегистрированные программы."""
        return dict(self._programs)

    def create_material(
        self,
        name: str = "material",
        *,
        program: str = "lit",
        base_color: Vector3 | None = None,
        roughness: float = 0.5,
        metallic: float = 0.0,
        transparent: bool = False,
        double_sided: bool = False,
        texture: GpuTexture | None = None,
        alpha: float = 1.0,
        depth_test: bool = True,
    ) -> Material:
        """Создать материал на базе зарегистрированной программы.

        Args:
            name: Имя материала.
            program: Имя программы (``lit``, ``lit_instanced``, ``unlit``).
            base_color: Базовый цвет.
            roughness: Шероховатость.
            metallic: Металличность.
            transparent: Признак прозрачности.
            double_sided: Рисовать обе стороны.
            texture: Диффузная текстура.
            alpha: Прозрачность (используется программами без освещения и
                полупрозрачными материалами).
            depth_test: Проверять ли глубину (``False`` — рисовать поверх).

        Returns:
            Новый материал.
        """
        return Material(
            name=name,
            program=self.program(program),
            base_color=base_color or Vector3(0.82, 0.82, 0.86),
            roughness=roughness,
            metallic=metallic,
            transparent=transparent,
            double_sided=double_sided,
            texture=texture,
            alpha=alpha,
            depth_test=depth_test,
        )

    def create_pbr_material(
        self,
        name: str = "pbr",
        *,
        base_color: Vector3 | None = None,
        roughness: float = 1.0,
        metallic: float = 0.0,
        alpha_mode: str = "opaque",
        double_sided: bool = False,
        **options: object,
    ) -> PbrMaterial:
        """Создать PBR-материал на программе ``pbr``.

        Args:
            name: Имя материала.
            base_color: Множитель base color (RGB).
            roughness: Множитель шероховатости (0..1).
            metallic: Множитель металличности (0..1).
            alpha_mode: ``opaque`` | ``mask`` | ``blend``.
            double_sided: Рисовать обе стороны.
            **options: Остальные поля :class:`PbrMaterial` (текстуры, свечение...).

        Returns:
            Новый материал.
        """
        return PbrMaterial(
            name=name,
            program=self.program("pbr"),
            base_color=base_color or Vector3(1.0, 1.0, 1.0),
            roughness=roughness,
            metallic=metallic,
            alpha_mode=alpha_mode,
            double_sided=double_sided,
            **options,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------ #
    # Кадр
    # ------------------------------------------------------------------ #
    def begin_frame(
        self,
        camera: Camera | CameraStack,
        viewport: tuple[int, int] | None = None,
        *,
        clear_color: tuple[float, float, float, float] | None = None,
        light: DirectionalLight | None = None,
        target: RenderTarget | None = None,
        reset_viewport: bool = True,
        post: bool = False,
    ) -> FrameContext:
        """Начать кадр: подготовить контекст и очистить очередь.

        Args:
            camera: Активная камера или стек камер. При передаче стека
                активной становится его активная камера, а матрицы
                фиксируются один раз на кадр (быстрый путь отсечения).
            viewport: Размер области вывода; ``None`` — взять текущий viewport.
            clear_color: Цвет очистки фона; ``None`` — цвет камеры. Для
                камер, которые не очищают цвет (UI поверх игры), очистка
                пропускается.
            light: Источник света; ``None`` — оставить прежний.
            target: Цель вывода (FBO); ``None`` — экран.
            reset_viewport: Сбрасывать ли viewport на весь размер области
                (``False`` нужен, когда стек сам управляет прямоугольниками).
            post: Рисовать сцену в HDR-цель и пропустить через пост-обработку
                (нужны ``hdr_enabled`` и поддержка RGBA16F; иначе флаг
                игнорируется). Проходы с ``after_post`` (UI) рисуются уже в
                ``target``/на экран после обработки.

        Returns:
            Контекст кадра (он же передаётся проходам графа).
        """
        if not self._initialized:
            self.initialize()
        if light is not None:
            self.light = light
        viewport = self._resolve_viewport(viewport)
        self._full_viewport = viewport
        self._final_target = target
        self._post_active = bool(
            post and self.hdr_enabled and self.capabilities.supports_hdr_target
        )
        if self._post_active:
            scale = min(1.0, max(0.25, float(self.render_scale)))
            viewport = (
                max(1, round(viewport[0] * scale)),
                max(1, round(viewport[1] * scale)),
            )
            target = self._acquire_hdr_target(*viewport)
        width, height = viewport

        resolved = self._resolve_camera(camera)
        resolved.set_aspect(width, height)

        # Камера может занимать только часть области вывода (стек камер,
        # предпросмотр): тогда всё, кроме прямоугольника, не перезаписывается.
        rect = resolved.pixel_rect(width, height)
        if rect[2] <= 0 or rect[3] <= 0 or rect == (0, 0, width, height):
            rect = (0, 0, width, height)
        camera_clears_color = resolved.clears_color and rect == (0, 0, width, height)

        color = clear_color if clear_color is not None else resolved.clear_color_rgba
        if clear_color is not None or self._clear_pass is None:
            self.clear_color = color
        if self._clear_pass is not None:
            self._clear_pass.clear_color = color
            # Слои поверх игры (UI-камера) не должны стирать уже нарисованное.
            self._clear_pass.clears_color = camera_clears_color
            self._clear_pass.clears_depth = resolved.clears_depth

        self._camera = resolved
        # Сцена может задать фоновую засветку прямо у солнца: переносим её в
        # менеджер света, чтобы шейдер получил нужный u_ambient.
        self.lights.use_fallback_ambient(self.light)
        self._items.clear()
        self._instances.clear()
        # Новая ревизия привязок: проверку состояния драйвера повторяем.
        self._verified_programs.clear()
        self.stats.reset()
        self.stats.frame_index = self.graph.frame_index

        if target is not None:
            target.bind(self.state)
        self._frame_target = target
        if reset_viewport:
            self.state.set_viewport(*rect)
        self.state.apply_default_3d()

        self._context = FrameContext(
            camera=resolved,
            light=self.light,
            state=self.state,
            viewport=viewport,
            frame_index=self.graph.frame_index,
            clear_color=color,
            stats=self.stats,
            renderer=self,
            stack=self._stack,
        )
        return self._context

    def _resolve_camera(self, camera: Camera | CameraStack) -> Camera:
        """Выбрать камеру кадра и зафиксировать матрицы стека.

        Args:
            camera: Камера или стек камер.

        Returns:
            Камера, по которой считается кадр.

        Raises:
            RenderError: Если передан пустой стек камер.
        """
        if isinstance(camera, CameraStack):
            self._stack = camera
            active = camera.begin_frame()
            if active is None:
                raise RenderError("Стек камер пуст: нечего рендерить")
            return active
        self._stack = None
        return camera

    def _resolve_viewport(self, viewport: tuple[int, int] | None) -> tuple[int, int]:
        """Определить размер области вывода."""
        if viewport is not None:
            self._viewport = (max(1, int(viewport[0])), max(1, int(viewport[1])))
        return self._viewport

    def submit(
        self,
        mesh: GpuMesh,
        material: Material,
        model_matrix: np.ndarray,
        *,
        cull: bool | None = None,
        mode: int | None = None,
        layer: int | None = None,
        cast_shadow: bool = True,
        receive_shadow: bool = True,
    ) -> bool:
        """Поставить объект в очередь отрисовки.

        Args:
            mesh: Геометрия.
            material: Материал.
            model_matrix: Матрица модели ``(4, 4)``.
            cull: Отсекать ли объект по усечённому объёму; ``None`` — по
                настройке рендерера.
            mode: Примитив OpenGL (``GL_LINES`` для сетки и gizmos);
                ``None`` — треугольники.
            layer: Слой объекта: если он не попадает в маску отсечения
                камеры, объект отбрасывается без теста объёма.
            cast_shadow: Попадает ли объект в карту теней.
            receive_shadow: Затемняется ли объект чужими тенями.

        Returns:
            ``True``, если объект попал в очередь.
        """
        if self._camera is None:
            raise RenderError("submit() вызван до begin_frame()")
        if layer is not None and not self._camera.layer_visible(layer):
            self.stats.culled += 1
            return False
        matrix = np.asarray(model_matrix, dtype=MATRIX_DTYPE).reshape(4, 4)
        center, radius = _transformed_bounds(mesh, matrix)

        use_culling = self.cull_by_frustum if cull is None else cull
        if use_culling and not self._camera.is_sphere_visible(center, radius):
            self.stats.culled += 1
            return False

        distance = (center - self._camera.position).length
        self._items.append(
            DrawItem(
                mesh=mesh,
                material=material,
                model_matrix=matrix,
                sort_key=material.sort_key,
                distance=distance,
                center=center,
                radius=radius,
                mode=mode,
                cast_shadow=cast_shadow,
                receive_shadow=receive_shadow,
            )
        )
        self.stats.submitted += 1
        return True

    def submit_transform(
        self, mesh: GpuMesh, material: Material, transform: Any
    ) -> bool:
        """Поставить в очередь объект по его трансформу.

        Args:
            mesh: Геометрия.
            material: Материал.
            transform: :class:`~engine.core.transform.Transform`.

        Returns:
            ``True``, если объект попал в очередь.
        """
        return self.submit(mesh, material, transform.world_matrix)

    def draw_instanced(
        self,
        mesh: GpuMesh,
        material: Material,
        matrices: list[np.ndarray] | np.ndarray,
        *,
        cull: bool | None = None,
        fallback_to_loop: bool = True,
        cast_shadow: bool = True,
        receive_shadow: bool = True,
    ) -> int:
        """Поставить в очередь отрисовку одной геометрии много раз (instancing).

        Отрисовка выполняется в проходе геометрии — после очистки экрана, —
        поэтому вызов безопасен в любом месте между ``begin_frame`` и
        ``end_frame``.

        Args:
            mesh: Геометрия.
            material: Материал.
            matrices: Матрицы экземпляров ``(N, 4, 4)``.
            cull: Отсекать ли всю группу по усечённому объёму.
            fallback_to_loop: При отсутствии поддержки инстансинга поставить
                экземпляры обычными объектами очереди.
            cast_shadow: Отбрасывают ли экземпляры тень.
            receive_shadow: Получают ли экземпляры чужие тени.

        Returns:
            Число принятых экземпляров.
        """
        if self._camera is None:
            raise RenderError("draw_instanced() вызван до begin_frame()")
        array = np.asarray(matrices, dtype=MATRIX_DTYPE)
        if array.ndim == 2:
            array = array.reshape(1, 4, 4)
        if array.ndim != 3 or array.shape[1:] != (4, 4):
            raise RenderError(f"Ожидался массив (N, 4, 4), получено {array.shape}")
        count = int(array.shape[0])
        if count == 0:
            return 0

        supports_instancing = bool(self.capabilities.supports_instancing) and (
            "lit_instanced" in self._programs or material.program.handle != 0
        )
        if not supports_instancing:
            if not fallback_to_loop:
                return 0
            for matrix in array:
                self.submit(
                    mesh,
                    material,
                    matrix,
                    cull=False,
                    cast_shadow=cast_shadow,
                    receive_shadow=receive_shadow,
                )
            return count

        center, radius = _group_bounds(mesh, array)
        use_culling = self.cull_by_frustum if cull is None else cull
        if use_culling and not self._camera.is_sphere_visible(center, radius):
            self.stats.culled += count
            return 0

        self._instances.append(
            _InstanceDraw(
                mesh=mesh,
                material=material,
                matrices=np.ascontiguousarray(array),
                sort_key=material.sort_key,
                center=center,
                radius=radius,
                distance=(center - self._camera.position).length,
                cast_shadow=cast_shadow,
                receive_shadow=receive_shadow,
            )
        )
        self.stats.submitted += count
        return count

    def _instance_buffer_for(
        self, mesh: GpuMesh, *, capacity: int
    ) -> GpuInstanceBuffer:
        """Получить (или создать) буфер экземпляров для меша."""
        key = id(mesh)
        buffer = self._instance_buffers.get(key)
        if buffer is None or buffer.capacity < capacity:
            if buffer is not None:
                buffer.dispose()
            buffer = GpuInstanceBuffer(
                capacity=max(64, capacity), name=f"instances:{mesh.name}"
            )
            self._instance_buffers[key] = buffer
        return buffer

    def end_frame(self, *, target: RenderTarget | None = None) -> RenderStats:
        """Выполнить граф проходов и вернуть статистику кадра.

        Args:
            target: Цель вывода; ``None`` — экран.

        Returns:
            Статистика кадра.

        Raises:
            RenderError: Если кадр не был начат.
        """
        if self._context is None:
            raise RenderError("end_frame() вызван до begin_frame()")

        if self._post_active:
            self._finish_post_frame()
        else:
            self.graph.execute(self._context)
        if target is not None:
            target.resolve()
            target.unbind(self.state)

        self.stats.state_calls = self.state.stats.calls
        self.stats.state_saved = self.state.stats.saved
        self._context = None
        return self.stats

    def _acquire_hdr_target(self, width: int, height: int) -> RenderTarget:
        """Вернуть HDR-цель нужного размера (кэш, пересоздание при resize)."""
        target = self._hdr_target
        if target is None:
            target = RenderTarget("hdr", width, height, hdr=True)
            self._hdr_target = target
        target.resize(width, height)
        target.create()
        return target

    @property
    def hdr_target(self) -> RenderTarget | None:
        """HDR-цель кадра (``None`` до первого кадра с ``post=True``)."""
        return self._hdr_target

    @property
    def post_frame_active(self) -> bool:
        """``True``, если текущий кадр рисуется в HDR-цель."""
        return self._post_active

    def _finish_post_frame(self) -> None:
        """Завершить HDR-кадр: проходы сцены, вывод, проходы поверх (UI)."""
        context = self._context
        hdr = self._hdr_target
        final = self._final_target
        assert context is not None and hdr is not None  # noqa: S101
        self.graph.execute(context, "main")
        hdr.resolve()

        self._frame_target = final
        self.post.apply(hdr, final, self._full_viewport)

        late = replace(context, viewport=self._full_viewport)
        self._context = late
        self.graph.execute(late, "late")
        self._post_active = False

    @property
    def post_settings(self) -> PostSettings:
        """Настройки tonemap/FXAA текущего кадра."""
        return self.post.settings

    @post_settings.setter
    def post_settings(self, value: PostSettings) -> None:
        self.post.settings = value

    @property
    def screen_fx(self) -> ScreenFx | None:
        """Fade/flash, рисуемые поверх кадра (``None`` — нет)."""
        return self.post.screen_fx

    @screen_fx.setter
    def screen_fx(self, value: ScreenFx | None) -> None:
        self.post.screen_fx = value

    def render(
        self,
        camera: Camera | CameraStack,
        items: list[tuple[GpuMesh, Material, np.ndarray]],
        viewport: tuple[int, int] | None = None,
        *,
        target: RenderTarget | None = None,
    ) -> RenderStats:
        """Удобный вызов: начать кадр, отправить объекты, завершить кадр.

        Args:
            camera: Камера или стек камер.
            items: Список ``(меш, материал, матрица модели)``.
            viewport: Размер области вывода.
            target: Цель вывода; ``None`` — экран.

        Returns:
            Статистика кадра.
        """
        self.begin_frame(camera, viewport, target=target)
        for mesh, material, matrix in items:
            self.submit(mesh, material, matrix)
        return self.end_frame(target=target)

    # ------------------------------------------------------------------ #
    # Внутренняя отрисовка
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # Тени
    # ------------------------------------------------------------------ #
    def _draw_shadow_map(self, context: FrameContext) -> None:
        """Построить карту теней направленного источника.

        Проход выполняется перед геометрией: он пишет глубину со стороны
        солнца, а затем шейдер сравнивает с ней глубину фрагмента. Если в сцене
        нет источника с ``cast_shadow`` или тени выключены, проход мгновенно
        завершается — карта не создаётся и кадр ничего не платит.
        """
        light = self.lights.shadow_caster
        if light is None or not self.shadows_enabled or not self.shadow_programs:
            self._shadow_pass_ready = False
            return

        if self.shadow_map is None:
            self.shadow_map = ShadowMap(
                size=light.shadow_map_size, name="directional_shadow"
            )
        shadow = self.shadow_map
        shadow.prepare(light, context.camera, state=self.state)
        self._shadow_drawn = 0

        shadow.bind(self.state)
        try:
            # В карту пишется только глубина: тест и запись глубины обязаны быть
            # включены, а отсечение граней выключено — иначе геометрия,
            # обращённая к свету задней стороной (тонкие плоскости, листья),
            # просто исчезла бы из карты теней.
            self.state.set_depth_test(True)
            self.state.set_depth_write(True)
            self.state.set_depth_func()
            self.state.set_cull(False)
            # Запись цвета выключена: карта содержит только глубину.
            self.state.set_color_write(False)
            shadow.clear()
            # Привязки программ сверяются заново: цель вывода сменилась, и
            # состояние драйвера после прохода теней уже другое.
            self._verified_programs.clear()
            for item in self._items:
                if item.material.transparent or item.mode is not None:
                    # Прозрачные объекты и линии тень не отбрасывают: первые
                    # всё равно рисуются поверх, вторые не имеют толщины.
                    continue
                if not item.cast_shadow:
                    continue
                # Отсечение по объёму источника — главная экономия прохода:
                # объект вне карты тени всё равно не отбрасывает, а рисовать
                # его второй раз на слабом GPU дорого.
                if not shadow.is_visible(item.center, item.radius):
                    continue
                self._draw_shadow_item(item)
            for group in self._instances:
                if group.material.transparent or not group.cast_shadow:
                    continue
                if not shadow.is_visible(group.center, group.radius):
                    continue
                self._draw_shadow_group(group)
            # Счётчик объектов в карте ведёт сама карта: по нему видно, что
            # проход действительно что-то нарисовал, а не просто очистил цель.
            shadow.set_drawn(self._shadow_drawn)
        finally:
            # Возвращаем вывод туда, куда рисовался кадр: у кадра это может
            # быть цель вывода (RenderTarget), а не экран. Простой unbind
            # отправил бы всю геометрию на экран мимо цели.
            self._unbind_shadow()
            self._restore_frame_target(context)
            self.state.set_cull(True)
            # Viewport после прохода теней равен разрешению карты: возвращаем
            # область вывода кадра, иначе сцена рисуется «мимо» кадра.
            self.state.set_viewport(*context.viewport_rect)
        self._shadow_pass_ready = True

    def _unbind_shadow(self) -> None:
        """Вернуть состояние вывода после прохода теней.

        Делается напрямую в драйвере, а не через :class:`GLState`: кэш
        состояний мог обновиться во время прохода, и его ``set_color_write``
        тогда решил бы, что маска уже верна, и не включил бы запись цвета.
        """
        from OpenGL import GL

        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
        self.state.invalidate_color_write()

    def _restore_frame_target(self, context: FrameContext) -> None:
        """Вернуть цель вывода кадра после смены framebuffer.

        Проход, который рисовал в собственную цель (карта теней, G-буфер),
        обязан вернуть привязку кадра — иначе следующий проход нарисует мимо.

        Args:
            context: Контекст кадра.
        """
        target = self._frame_target
        if target is not None and target.is_created:
            target.bind(self.state, viewport=False)
        else:
            self.state.bind_default_framebuffer()

    @property
    def shadow_programs(self) -> tuple[ShaderProgram, ShaderProgram] | None:
        """Программы прохода теней: обычная и инстансная.

        Returns:
            Пару программ или ``None``, если шейдеры теней не загрузились.
        """
        base = self._programs.get("shadow")
        instanced = self._programs.get("shadow_instanced")
        if base is None or instanced is None:
            return None
        return base, instanced

    def _draw_shadow_item(self, item: DrawItem) -> None:
        """Нарисовать один объект в карту теней."""
        programs = self.shadow_programs
        if programs is None:
            return
        program = programs[0]
        if not self._bind_program(program):
            return
        program.set_uniform("u_light_space_matrix", self._shadow_matrix())
        program.set_uniform("u_model", item.model_matrix)
        item.mesh.draw(item.mode)
        self.stats.shadow_draws += 1
        self.stats.shadow_triangles += item.mesh.triangle_count
        self._shadow_drawn += 1

    def _draw_shadow_group(self, group: _InstanceDraw) -> None:
        """Нарисовать инстансную пачку в карту теней."""
        programs = self.shadow_programs
        if programs is None:
            return
        program = programs[1]
        if not self._bind_program(program):
            return
        count = int(group.matrices.shape[0])
        buffer = self._instance_buffer_for(group.mesh, capacity=count)
        buffer.upload(group.matrices)
        vao = group.mesh.upload()
        if vao and group.mesh.layout.has("position"):
            buffer.attach(vao)
        program.set_uniform("u_light_space_matrix", self._shadow_matrix())
        group.mesh.draw_instanced(count)
        self.stats.shadow_draws += 1
        self.stats.shadow_triangles += group.mesh.triangle_count * count

    def _shadow_matrix(self) -> np.ndarray:
        """Матрица мира → пространство источника (или единичная, если теней нет)."""
        if self.shadow_map is None:
            return np.identity(4, dtype=float)
        return self.shadow_map.light_space_matrix

    def _apply_shadow_uniforms(self, program: ShaderProgram) -> None:
        """Передать в программу параметры карты теней.

        Когда карта не построена, шейдер получает ``u_shadow_enabled = 0`` и
        вообще не трогает текстуру: на слабом GPU это заметная экономия, а
        результат тот же — свет проходит беспрепятственно.
        """
        from OpenGL import GL

        # Параметры теней одинаковы для всех объектов кадра, а значения uniform
        # живут в программе: повторно грузить их на каждый draw — лишняя работа.
        shadow = self.shadow_map
        if id(program) in self._shadow_applied:
            if self._shadow_pass_ready and shadow is not None and shadow.is_created:
                self.state.bind_texture(1, GL.GL_TEXTURE_2D, shadow.texture)
            return
        self._shadow_applied.add(id(program))
        if not self._shadow_pass_ready or shadow is None or not shadow.is_created:
            program.set_uniform("u_shadow_enabled", 0)
            return
        program.set_uniform("u_shadow_enabled", 1)
        program.set_uniform("u_light_space_matrix", shadow.light_space_matrix)
        program.set_uniform("u_shadow_bias", float(shadow.bias))
        program.set_uniform("u_shadow_normal_bias", float(shadow.normal_bias))
        program.set_uniform("u_shadow_strength", float(shadow.strength))
        texel = np.array(shadow.texel_size, dtype=np.float32)
        program.set_uniform("u_shadow_texel", texel)
        self.state.bind_texture(1, GL.GL_TEXTURE_2D, shadow.texture)
        program.set_uniform("u_shadow_map", 1)

    # ------------------------------------------------------------------ #
    # Освещение
    # ------------------------------------------------------------------ #
    def _apply_lights(self, program: ShaderProgram, item: DrawItem) -> None:
        """Выбрать свет для объекта и загрузить его в программу.

        Набор света считается для каждого объекта: это и есть отсечение по
        радиусу влияния. Повторяющиеся наборы (а их в сцене большинство) не
        загружаются в драйвер повторно — см.
        :meth:`~engine.rendering.lighting.LightManager.prepare`.
        """
        light_set = self.lights.prepare(item.center, item.radius)
        self.lights.bind_set(program, light_set)
        self._apply_shadow_uniforms(program)
        self._apply_receive_shadow(program, item.receive_shadow)
        self.stats.lights_bound += light_set.count

    def _apply_receive_shadow(self, program: ShaderProgram, receive: bool) -> None:
        """Включить или выключить приём теней для следующего draw.

        Значение uniform хранится в программе и переживает кадры, поэтому в
        начале кадра кэш сбрасывается и первое значение уходит всегда; дальше
        в драйвер идёт только смена: большинство объектов принимает тени.
        """
        key = id(program)
        if self._receive_applied.get(key) != receive:
            program.set_uniform("u_no_receive_shadow", 0 if receive else 1)
            self._receive_applied[key] = receive

    def _apply_instance_lights(
        self, program: ShaderProgram, group: _InstanceDraw
    ) -> None:
        """Выбрать свет для инстансной пачки."""
        light_set = self.lights.prepare(group.center, group.radius)
        self.lights.bind_set(program, light_set)
        self._apply_shadow_uniforms(program)
        self._apply_receive_shadow(program, group.receive_shadow)
        self.stats.lights_bound += light_set.count

    def _draw_queue(self, context: FrameContext) -> None:
        """Отрисовать непрозрачное и инстансы; прозрачное — в своём проходе."""
        camera = context.camera
        opaque = [item for item in self._items if not item.material.transparent]
        transparent = [item for item in self._items if item.material.transparent]
        opaque.sort(key=lambda item: item.sort_key)

        instance_groups = sorted(self._instances, key=lambda group: group.sort_key)
        opaque_groups = [g for g in instance_groups if not g.material.transparent]
        transparent_groups = [g for g in instance_groups if g.material.transparent]

        self._active_program = None
        self._active_material = None
        self._shadow_applied.clear()
        self._receive_applied.clear()

        previous_key: tuple[int, int, int] | None = None
        for item in opaque:
            if previous_key != item.sort_key:
                self.stats.batches += 1
                previous_key = item.sort_key
            self._draw_item(item, camera)

        for group in opaque_groups:
            self.stats.batches += 1
            self._draw_instance_group(group, camera)

        # Прозрачное рисуется по слоям, внутри слоя — от дальнего к ближнему;
        # одиночные объекты и инстансные группы сортируются вместе.
        queue: list[DrawItem | _InstanceDraw] = [*transparent, *transparent_groups]
        queue.sort(key=lambda entry: (entry.material.sorting_layer, -entry.distance))
        self._pending_transparent = queue

    def invalidate_active_program(self) -> None:
        """Сбросить кэш «текущей программы/материала» очереди.

        Нужно проходам, которые между непрозрачной и прозрачной частью
        включают собственную программу (например, небо): иначе uniform-ы
        следующего материала уйдут в чужую программу.
        """
        self._active_program = None
        self._active_material = None

    def _draw_transparent(self, context: FrameContext) -> None:
        """Отрисовать прозрачную часть очереди (после неба)."""
        queue = self._pending_transparent
        self._pending_transparent = []
        if not queue:
            return
        camera = context.camera
        context.state.set_blend(True)
        for entry in queue:
            if isinstance(entry, _InstanceDraw):
                self._draw_instance_group(entry, camera)
            else:
                self._draw_item(entry, camera)
        context.state.set_blend(False)
        self.stats.batches += 1

    def _draw_instance_group(self, group: _InstanceDraw, camera: Camera) -> None:
        """Нарисовать группу экземпляров одним инстансным вызовом."""
        material = group.material
        program = self._instance_program(material, group.mesh)
        count = int(group.matrices.shape[0])

        buffer = self._instance_buffer_for(group.mesh, capacity=count)
        buffer.upload(group.matrices)
        # Порядок важен и проверен на интегрированном драйвере: сначала VAO меша,
        # затем программа и uniform-переменные, и только потом атрибуты
        # экземпляров. Атрибуты (locations 8..11) хранятся в состоянии VAO,
        # поэтому отрисовка обязана идти с тем же привязанным VAO.
        group.mesh.bind()

        self.state.set_cull(True)
        if not self._bind_program(program):
            return
        material.apply(self.light, camera.position, program=program)
        self._apply_camera_uniforms(program)
        self._apply_instance_lights(program, group)
        self._bind_material_texture(material, program)

        if group.mesh.layout.has("position"):
            buffer.attach(group.mesh.upload(), keep_bound=True)
        group.mesh.draw_instanced(count)

        self.stats.draw_calls += 1
        self.stats.instances += count
        self.stats.triangles += group.mesh.triangle_count * count
        self.stats.vertices += group.mesh.vertex_count * count
        self._active_program = program

    def _instance_program(self, material: Material, mesh: GpuMesh) -> ShaderProgram:
        """Выбрать программу для инстансной отрисовки.

        Если материал создан на базовом ``lit``, берётся его инстансный вариант
        с теми же ключевыми словами, что и одиночная отрисовка (вершинные цвета,
        лайтмапа, запечённый свет). Программы без такого варианта используются
        как есть: шейдер сам читает матрицы экземпляров.
        """
        base = material.program
        if base.name == "pbr":
            return self.pbr_variant(INSTANCED=1, **self._pbr_defines(material, mesh))
        if base.name != "lit":
            return base
        defines = self._feature_defines(material, mesh)
        # Ключевое слово пишется тем же именем, что и в исходниках шейдеров:
        # иначе для одной комбинации собирались бы две программы, и материал
        # рисовался бы не тем вариантом.
        defines["INSTANCED"] = 1
        return self.variant(**defines)

    def _feature_defines(
        self, material: Material, mesh: GpuMesh | None
    ) -> dict[str, object]:
        """Собрать ключевые слова варианта для материала и меша.

        Args:
            material: Материал.
            mesh: Меш (нужен, чтобы узнать про вершинные цвета); ``None`` —
                считать, что вершинных цветов нет.

        Returns:
            Словарь ``#define`` без ``INSTANCED`` (его добавляет вызывающий).
        """
        defines: dict[str, object] = {}
        if mesh is not None and mesh.layout.has("color"):
            # Меш принёс вершинные цвета: запечённый свет умножится в шейдере.
            defines["VERTEX_COLOR"] = 1
        if material.uses_lightmap:
            defines["USE_LIGHTMAP"] = 1
        if not material.dynamic_lighting:
            # Запечённый режим: шейдер не считает источники вовсе.
            defines["DYNAMIC_LIGHTING"] = 0
        return defines

    def _program_for(self, material: Material, item: DrawItem) -> ShaderProgram:
        """Подобрать программу варианта под материал и меш.

        Материалы на базовом ``lit`` получают вариант с нужными ключевыми
        словами; всё остальное (``unlit``, пользовательские программы)
        используется как есть.
        """
        base = material.program
        if base.name == "pbr":
            return self.pbr_variant(**self._pbr_defines(material, item.mesh))
        if base.name != "lit":
            return base
        return self.variant(**self._feature_defines(material, item.mesh))

    @staticmethod
    def _pbr_defines(material: Material, mesh: GpuMesh | None) -> dict[str, object]:
        """Ключевые слова PBR-варианта (без ``INSTANCED``)."""
        defines: dict[str, object] = {}
        if mesh is not None and mesh.layout.has("color"):
            defines["VERTEX_COLOR"] = 1
        if getattr(material, "alpha_mode", "opaque") == "mask":
            defines["ALPHA_MASK"] = 1
        if not material.dynamic_lighting:
            defines["DYNAMIC_LIGHTING"] = 0
        return defines

    def _draw_item(self, item: DrawItem, camera: Camera) -> None:
        """Нарисовать один элемент очереди."""
        material = item.material
        program = self._program_for(material, item)

        self.state.set_cull(not material.double_sided)
        self.state.set_depth_test(material.depth_test)
        if self._active_program is not program:
            if not self._bind_program(program):
                return
            self._apply_camera_uniforms(program)
            material.apply(self.light, camera.position, program=program)
            self._apply_lights(program, item)
            self._active_material = material
            self._active_item = item
            self._active_program = program
        else:
            # Набор света зависит от объекта, поэтому его нужно обновить даже
            # при той же программе: LightManager пропустит повторную загрузку,
            # если у объекта тот же набор.
            self._apply_lights(program, item)
            if self._active_material is not material:
                material.apply(self.light, camera.position, program=program)
                self._active_material = material
            self._active_item = item

        self._bind_material_texture(material, program)
        program.set_uniform("u_model", item.model_matrix)
        item.mesh.draw(item.mode)

        self.stats.draw_calls += 1
        self.stats.triangles += item.mesh.triangle_count
        self.stats.vertices += item.mesh.vertex_count

    def _camera_uniforms(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Вернуть ``(view, projection, view_projection)`` матрицы кадра.

        Матрицы считаются один раз на камеру и её ревизию: в горячем цикле
        (сотни объектов) повторные умножения матриц не выполняются.
        """
        camera = self._camera
        if camera is None:
            raise RenderError("Матрицы камеры запрошены до begin_frame()")
        stack = self._stack
        key = (id(camera), camera.version, id(stack) if stack is not None else 0)
        if key != self._camera_uniform_key or self._camera_uniform_values is None:
            if stack is not None and stack.active is camera:
                # Стек уже зафиксировал матрицы кадра — берём готовые.
                self._camera_uniform_values = (
                    stack.view_matrix,
                    stack.projection_matrix,
                    stack.view_projection_matrix,
                )
            else:
                view = camera.get_view_matrix()
                projection = camera.get_projection_matrix()
                self._camera_uniform_values = (view, projection, projection @ view)
            self._camera_uniform_key = key
        return self._camera_uniform_values

    def _apply_camera_uniforms(self, program: ShaderProgram) -> None:
        """Передать в программу матрицы камеры (один раз на смену программы).

        Кроме произведения ``view * projection`` отправляются отдельные
        матрицы вида и проекции: шейдерам нужны позиция во view-space и
        линейная глубина (туман, эффекты глубины, отладочная визуализация).

        Перед отправкой проверяется, что программа действительно привязана в
        драйвере: если состояние контекста изменили в обход :class:`GLState`
        (скрипт, отладочный проход, чужой виджет), установка uniform-переменной
        даёт невнятную ``GL_INVALID_OPERATION`` на произвольном вызове — а
        камера обязана либо рисовать, либо честно сообщить о проблеме.
        """
        if self._camera is None:
            return
        if not self._bind_program(program):
            return
        view, projection, view_projection = self._camera_uniforms()
        key = self._camera_uniform_key
        if self._camera_sent.get(program.handle) == key:
            return
        self._camera_sent[program.handle] = key
        program.set_uniform("u_view_projection", view_projection)
        program.set_uniform("u_view", view)
        program.set_uniform("u_projection", projection)
        program.set_uniform("u_camera_position", self._camera.position)

    def _bind_program(self, program: ShaderProgram) -> bool:
        """Привязать программу и убедиться, что она активна в драйвере.

        Проверка состояния драйвера делается **один раз на программу за кадр**:
        она ловит рассинхронизацию кэша состояний (внешний код привязал свою
        программу, программа была удалена) и при этом не платит за readback на
        каждой смене материала — в сцене с двумя сотнями материалов это была бы
        заметная доля кадра.

        Args:
            program: Программа, в которую будут писаться uniform-переменные.

        Returns:
            ``True``, если программу можно использовать для отрисовки.

        Raises:
            RenderError: Если программа не собрана или драйвер её не принимает.
        """
        if not program.handle:
            raise RenderError(
                f"Программа '{program.name}' не собрана: рисовать нечем"
            )
        from OpenGL import GL

        self.state.use_program(program.handle)
        if program.handle in self._verified_programs:
            return True
        current = int(GL.glGetIntegerv(GL.GL_CURRENT_PROGRAM))
        if current != program.handle:
            # Кэш ошибся — лечим состояние и привязываем программу заново.
            self.state.invalidate_program()
            self.state.use_program(program.handle)
            current = int(GL.glGetIntegerv(GL.GL_CURRENT_PROGRAM))
        if current != program.handle:
            raise RenderError(
                f"Не удалось привязать программу '{program.name}' "
                f"(драйвер удерживает {current}, ожидалась {program.handle})"
            )
        self._verified_programs.add(program.handle)
        return True

    def invalidate_camera_cache(self) -> None:
        """Сбросить кэш матриц камеры (после ручной правки uniform-переменных)."""
        self._camera_uniform_key = None
        self._camera_uniform_values = None
        self._camera_sent.clear()

    def _bind_material_texture(
        self, material: Material, program: ShaderProgram | None = None
    ) -> None:
        """Загрузить (если нужно) и привязать текстуры материала.

        Диффузная текстура идёт в юнит 0, лайтмапа — в юнит 2 (юнит 1 занят
        картой теней).
        """
        from OpenGL import GL

        texture = material.texture
        if texture is not None:
            if not texture.is_uploaded:
                texture.upload()
            self.state.bind_texture(0, GL.GL_TEXTURE_2D, texture.handle)
        for unit, extra in material.extra_textures():
            if not extra.is_uploaded:
                extra.upload()
            self.state.bind_texture(unit, GL.GL_TEXTURE_2D, extra.handle)
        lightmap = material.lightmap
        if lightmap is None:
            return
        if not lightmap.is_uploaded:
            lightmap.upload()
        self.state.bind_texture(2, GL.GL_TEXTURE_2D, lightmap.handle)

    # ------------------------------------------------------------------ #
    # Обслуживание
    # ------------------------------------------------------------------ #
    def resize(self, width: int, height: int) -> None:
        """Сообщить рендереру новый размер области вывода."""
        self._viewport = (max(1, int(width)), max(1, int(height)))

    def set_sky(self, sky_pass: SkyPass | None) -> None:
        """Поставить (или убрать) проход неба между ``geometry`` и ``transparent``.

        Args:
            sky_pass: Новый проход неба; ``None`` — убрать небо.
        """
        current = self.graph.pass_by_name("sky")
        if current is not None:
            self.graph.remove_pass(current)
        if sky_pass is None:
            return
        names = [p.name for p in self.graph.passes]
        index = names.index("transparent") if "transparent" in names else None
        self.graph.add_pass(sky_pass, index=index)

    def add_pass(
        self, render_pass: RenderPass, *, index: int | None = None
    ) -> RenderPass:
        """Добавить проход в граф (например, тени или пост-обработку)."""
        return self.graph.add_pass(render_pass, index=index)

    def dispose(self) -> None:
        """Освободить программы, буферы экземпляров и карту теней."""
        for program in self._programs.values():
            program.dispose()
        self._programs.clear()
        if get_program_cache() is self.program_cache and self.program_cache:
            set_program_cache(None)
        for program in self._variants.values():
            program.dispose()
        self._variants.clear()
        for program in self._pbr_variants.values():
            program.dispose()
        self._pbr_variants.clear()
        self._pbr_sources.clear()
        self._lit_sources.clear()
        if self.shadow_map is not None:
            self.shadow_map.dispose()
            self.shadow_map = None
        for buffer in self._instance_buffers.values():
            buffer.dispose()
        self._instance_buffers.clear()
        if self._hdr_target is not None:
            self._hdr_target.dispose()
            self._hdr_target = None
        self.post.dispose()
        self.graph.clear()
        self._verified_programs.clear()
        self._camera_sent.clear()
        self._initialized = False

    @property
    def queue_size(self) -> int:
        """Сколько объектов в текущей очереди отрисовки."""
        return len(self._items)

    def __repr__(self) -> str:
        return (
            f"<Renderer '{self.name}' programs={len(self._programs)} "
            f"queued={len(self._items)} {self.stats.describe()}>"
        )


def _transformed_bounds(mesh: GpuMesh, matrix: np.ndarray) -> tuple[Vector3, float]:
    """Посчитать мировые центр и радиус сферы для меша с матрицей модели.

    Args:
        mesh: Меш (используется его AABB).
        matrix: Матрица модели.

    Returns:
        Пару «центр сферы, радиус».
    """
    minimum, maximum = mesh.bounds
    # Чистая арифметика на float: для одного объекта numpy-вызовы дороже счёта.
    cx = (minimum.x + maximum.x) * 0.5
    cy = (minimum.y + maximum.y) * 0.5
    cz = (minimum.z + maximum.z) * 0.5
    hx = (maximum.x - minimum.x) * 0.5
    hy = (maximum.y - minimum.y) * 0.5
    hz = (maximum.z - minimum.z) * 0.5
    (a, b, c, tx), (d, e, f, ty), (g, h, i, tz) = (
        row for row in matrix[:3].tolist()
    )
    center = Vector3(
        a * cx + b * cy + c * cz + tx,
        d * cx + e * cy + f * cz + ty,
        g * cx + h * cy + i * cz + tz,
    )
    largest = max(
        (a * a + d * d + g * g) ** 0.5,
        (b * b + e * e + h * h) ** 0.5,
        (c * c + f * f + i * i) ** 0.5,
    )
    radius = (hx * hx + hy * hy + hz * hz) ** 0.5 * max(largest, 1e-4)
    return center, radius


def _group_bounds(mesh: GpuMesh, matrices: np.ndarray) -> tuple[Vector3, float]:
    """Посчитать общий центр и радиус группы экземпляров.

    Args:
        mesh: Геометрия (используется её AABB).
        matrices: Матрицы экземпляров ``(N, 4, 4)``.

    Returns:
        Пару «центр группы, радиус описанной сферы».
    """
    minimum, maximum = mesh.bounds
    local_center = (minimum + maximum) * 0.5
    half_extent = (maximum - minimum) * 0.5
    stack = matrices.astype(float, copy=False)
    point = np.array([*local_center.to_tuple(), 1.0])
    centers = (stack @ point)[:, :3]
    largest_scale = float(np.linalg.norm(stack[:, :3, :3], axis=1).max())
    group_center = centers.mean(axis=0)
    offsets = np.linalg.norm(centers - group_center, axis=1)
    group_radius = float(offsets.max()) + half_extent.length * max(largest_scale, 1e-4)
    return (
        Vector3(float(group_center[0]), float(group_center[1]), float(group_center[2])),
        group_radius,
    )


def _stage_vertex():
    """Стадия вершинного шейдера (ленивый импорт ради отсутствия циклов)."""
    from engine.core.resources import ShaderStage

    return ShaderStage.VERTEX


def _stage_fragment():
    """Стадия фрагментного шейдера."""
    from engine.core.resources import ShaderStage

    return ShaderStage.FRAGMENT


def _default_shader_root() -> Path:
    """Каталог шейдеров репозитория (``assets/`` рядом с пакетом)."""
    return Path(__file__).resolve().parents[2] / "assets"


def identity_model_matrix() -> np.ndarray:
    """Единичная матрица модели (удобно для отладочной геометрии)."""
    return identity_matrix()


def translation_of(matrix: np.ndarray) -> Vector3:
    """Сдвиг матрицы модели (для отладки и инспектора)."""
    return extract_translation(matrix)
