# Рендеринг MyEngine — устройство и API

Подсистема живёт в `engine/rendering/` и работает на **OpenGL 3.3 Core**
(PyOpenGL). Она не зависит от PySide6: Qt подключается отдельными модулями
(`qt_context`, `qt_viewport`) только в редакторе. Игра, headless-рендер и тесты
обходятся без Qt-виджета.

Текущий этап — forward-рендеринг с одним направленным источником света,
очередью отрисовки, отсечением по усечённому объёму, сортировкой по материалам
и GPU-инстансингом. Граф проходов уже поддерживает цели вывода с несколькими
цветовыми привязанностями (MRT), то есть переход на deferred shading не требует
менять интерфейсы.

Проверить всё можно так:

```bash
.venv\Scripts\python.exe scripts\render_offscreen.py --mesh cube --out _work/cube.png
.venv\Scripts\python.exe scripts\render_offscreen.py --mesh sphere --instances 400 --out _work/forest.png
.venv\Scripts\python.exe scripts\render_showcase.py --out _work/showcase.png   # витринная сцена
.venv\Scripts\python.exe scripts\check_cameras.py                             # проверка камер (стек, гизмо, превью)
.venv\Scripts\python.exe scripts\run_viewer.py --mesh sphere      # интерактивное окно
.venv\Scripts\python.exe tools\benchmark_render.py                # замеры рендера
.venv\Scripts\python.exe -m pytest -q                             # 886 тестов
```

---

## 1. Слои пакета

| Модуль | Ответственность |
|---|---|
| `context.py` | порт `IRenderContextProvider` / `IRenderSurface`, сведения о драйвере |
| `settings.py` | `RenderSettings`, `QualityTier`, `ShadowSettings` |
| `capabilities.py` | лимиты драйвера и флаги технологий (instancing, MRT, …) |
| `state.py` | `GLState` — кэш состояния: меньше вызовов драйвера |
| `gl_debug.py` | проверка `glGetError`, отладочный вывод `KHR_debug` |
| `shader.py` | `Shader`, `ShaderProgram`, варианты через `#define`, uniform-кэш |
| `mesh.py` | `VertexLayout`, `GpuMesh` (VAO/VBO/EBO), `GpuInstanceBuffer` |
| `texture.py` | `GpuTexture`: мипы, фильтрация, анизотропия, BC-данные |
| `material.py` | `Material`, `DirectionalLight` |
| `camera.py` | `Camera`, `Frustum`, `CameraComponent`, `CameraStack`, `CameraPreview`, `EditorCameraController`, `EditorCameraSession` |
| `render_graph.py` | `RenderTarget` (FBO, MRT), `RenderPass`, `RenderGraph`, `FrameContext` |
| `renderer.py` | `Renderer`: очередь, батчинг, инстансинг, статистика |
| `screenshot.py` | чтение кадрового буфера, свой PNG-кодировщик |
| `qt_context.py` | `QSurfaceFormat`, `QtOffscreenContext`, `QtRenderSurface` |
| `qt_viewport.py` | `ViewportWidget(QOpenGLWidget)`: ввод, камера, скриншоты |

---

## 2. Контекст: кто владеет OpenGL

Рендерер не создаёт контекст сам — он получает его от владельца. В редакторе это
`QOpenGLWidget`, в плеере — оконный бэкенд, в тестах и при генерации превью —
offscreen-поверхность:

```python
from engine.rendering.qt_context import QtOffscreenContext, apply_surface_format
from engine.rendering import RenderSettings

apply_surface_format(RenderSettings())          # до создания QApplication
with QtOffscreenContext((960, 640)) as context:
    renderer.initialize()                        # читает драйвер и собирает программы
    ...
```

Важные детали, выясненные на практике:

* платформенный плагин Qt `offscreen` **не умеет** создавать GL-контекст
  («This plugin does not support createPlatformOpenGLContext») — для headless
  нужна оконная платформа, окно при этом не показывается;
* `apply_surface_format` задаёт `QSurfaceFormat` по умолчанию: версия 3.3,
  Core Profile, глубина 24, трафарет 8, MSAA по профилю качества, VSync;
* `QtOffscreenContext` сам создаёт `QApplication`, если его ещё нет.

---

## 3. Кэш состояния — самый дешёвый выигрыш

Слабые драйверы дорого платят за переключение состояния. `GLState` помнит, что
уже установлено, и пропускает повторные вызовы, собирая статистику:

```python
renderer.state.stats.calls     # сколько вызовов реально ушло в драйвер
renderer.state.stats.saved     # сколько вызовов удалось не делать
```

В кадре с шестью кубами одного материала экономится больше десяти вызовов
(`test_state_cache_saves_calls`), а `invalidate()` сбрасывает кэш при смене
контекста.

**Программу можно сменить только через кэш.** Если привязать программу в обход
(`ShaderProgram.use()` без аргумента или «сырой» `glUseProgram`), кэш продолжит
считать активной прежнюю — и рендерер пропустит `glUseProgram`, отправив
uniform-переменные в чужую программу. Драйвер ответит
`GL_INVALID_OPERATION (1282)` на произвольном вызове (в редакторе это ловилось
на `glUniformMatrix4fv` для `u_view_projection`, см.
`TestProgramStateDesync`). Поэтому:

* из кода рендера используйте `Renderer._bind_program` — он привязывает
  программу и сверяет результат с драйвером (проверка выполняется один раз на
  программу за кадр, поэтому смена материала её не оплачивает);
* из внешнего кода передавайте кэш: `program.use(renderer.state)`;
* `GLState.invalidate_program()` сбрасывает запомненную программу, если она
  была удалена в обход рендерера.

**Текстуры в обход кэша.** Загрузка `GpuTexture`, создание `RenderTarget` и карты
теней привязывают текстуру напрямую (`glBindTexture` на *текущем* юните). Если
текущим был юнит материала, кэш `GLState` терял синхронизацию: следующий кадр
пропускал привязку и сэмплировал чужую текстуру. Поэтому такие операции идут
через служебный юнит `SCRATCH_TEXTURE_UNIT` (15, см. `select_scratch_unit()` в
`state.py`); материалы используют юниты 0–5.

---

## 4. Шейдеры и варианты

Загрузку исходников (с `#include` и `#pragma once`) делает подсистема ресурсов,
а `ShaderProgram` компилирует, линкует и кэширует расположения uniform-переменных:

```python
from engine.rendering import ShaderProgram

program = ShaderProgram.from_assets(vertex_asset, fragment_asset, name="lit")
program.use(renderer.state)                         # через кэш состояний
program.set_uniform("u_model", model_matrix)        # матрица (4, 4)
program.set_uniform("u_base_color", Vector3(1, 0.5, 0))
program.set_uniform("u_use_texture", True)
```

* `set_uniform` сам выбирает `glUniform1i/1f/2f/3f/4f/Matrix4fv` по типу значения
  (`bool`, `int`, `float`, `Vector3`, `Quaternion`, кортеж, numpy-массив);
* матрицы транспонируются перед отправкой: движок хранит их построчно
  (row-major, сдвиг в последнем столбце), а OpenGL ждёт столбцовую раскладку;
* **варианты шейдеров** — это `#define`, вставляемый сразу после `#version`
  (`inject_defines`). Так собран `lit_instanced`: тот же код с `INSTANCED`,
  читающий матрицу модели из атрибутов экземпляра.

Базовый шейдер `assets/shaders/simple.vert|frag`: ламберт + блик Блинна,
фоновая подсветка, опциональная диффузная текстура, перевод в sRGB.
Вспомогательный `unlit.vert|frag` — сплошной цвет для сетки и gizmos.

### PBR-шейдер и `PbrMaterial`

`assets/shaders/pbr.frag` (вершинный шейдер общий — `simple.vert`) — модель
металличность/шероховатость: Cook—Torrance (GGX, Смит—Шлик, Френель—Шлик) для
источников и аналитический BRDF окружения (Лазарев) для фона вместо IBL
(карты окружения дороги для iGPU). Источники и тени — из `lighting.frag`:
общая часть вынесена в `light_incidence()` (тип, затухание, конус), поэтому
Блинн—Фонг и PBR считают падающий свет одинаково. Яркости источников заданы под
ламберт без деления на π, поэтому диффузная часть PBR тоже без `1/π`, а
зеркальная умножена на π — оба шейдера одинаково ярки.

```python
material = renderer.create_pbr_material(
    "brick", roughness=0.8, texture=albedo, normal_texture=normal, orm_texture=orm
)
renderer.submit(mesh, material, matrix)
# или из ассета (.mat.json → AssetDatabase):
material = load_pbr_material(renderer, db, guid, textures=cache)
```

* карты: base color — юнит 0 (sRGB декодирует железо), нормали — 3, ORM
  (R = occlusion, G = roughness, B = metallic) — 4, свечение — 5; наличие карты
  сообщают `u_has_*`, поэтому вариантов шейдера немного: `INSTANCED`,
  `VERTEX_COLOR`, `ALPHA_MASK`, `DYNAMIC_LIGHTING=0`;
* карта нормалей без вершинных касательных: базис строится по производным
  позиции и UV (Шюлер); конвенция — OpenGL (зелёный вверх);
* `alpha_mode`: `opaque`, `mask` (вариант `ALPHA_MASK`, `discard`), `blend`
  (проход прозрачности); `double_sided` переворачивает нормаль у задних граней;
* программа `pbr` собирается лениво — сцены без PBR не платят компиляцией;
* `Material.extra_textures()` — общий хук: рендерер привязывает пары
  `(юнит, текстура)`, а `sort_key` PBR учитывает весь набор текстур;
* `GpuTexture.from_cooked()` загружает `texture.bin` как есть: мипы, sRGB-формат,
  BC1/BC3/BC5 (S3TC/RGTC). Не реализовано: лайтмапа и запечённый свет в PBR,
  освещение IBL, анизотропия по умолчанию.

### Кэш бинарников программ

`ProgramBinaryCache` (`program_cache.py`) сохраняет линкованные программы на диск
(`glGetProgramBinary`, `ARB_get_program_binary`) и при следующем запуске поднимает их
без компиляции. Включается так: `Renderer(program_cache_dir=<каталог>)` (рекомендуемо —
`Library/ShaderCache` проекта) или `shader.set_program_cache(cache)`. Ключ записи —
хэш исходников всех стадий **с `#define` варианта** и строки драйвера
(`GL_VENDOR|RENDERER|VERSION`): смена драйвера даёт промах, а не чужой бинарник.
Битый файл или отказ драйвера (`rejected`) → запись удаляется, программа собирается
обычным путём. Нет расширения — кэш молча отключён (`supported == False`).
`warm_program_cache(name, vertex=, fragment=, variants=[...])` собирает варианты заранее
(список берётся из `info["variants"]` артефакта шейдера).

Оба вершинных шейдера получают **отдельные** матрицы камеры: `u_view`,
`u_projection` и готовое произведение `u_view_projection`. Отдельные матрицы
дают `v_view_position` (позиция в пространстве камеры) — это основа для
эффектов глубины, тумана и контуров, а `u_view_projection` остаётся быстрым
путём для самой отрисовки.

---

## 5. Меши и инстансинг

```python
from engine.core.resources import cube_mesh
from engine.rendering import GpuMesh, GpuInstanceBuffer

mesh = GpuMesh.from_mesh_asset(cube_mesh(1.0))     # данные ещё на CPU
mesh.upload()                                      # VAO + VBO + EBO
mesh.draw()                                        # glDrawElements
mesh.draw_instanced(1000)                          # glDrawElementsInstanced
```

* `VertexLayout` описывает interleaved-формат (`position`, `normal`, `uv`,
  `tangent`, `color`, кости), сам упаковывает массивы и вычисляет смещения;
* AABB считается один раз при загрузке и используется для отсечения;
* `GpuInstanceBuffer` раскладывает матрицы по четырём vec4-атрибутам
  (locations 8–11) с делителем 1 — это и есть механизм «одна геометрия, тысячи
  объектов»; матрицы транспонируются при загрузке по той же причине, что и
  uniform-матрицы.

---

## 6. Камера и управление

```python
from engine.rendering import Camera, OrbitCameraController

camera = Camera(fov=50, near=0.1, far=300)
camera.position = Vector3(4, 3, 4)
camera.look_at(Vector3.zero())

orbit = OrbitCameraController(target=Vector3.zero(), distance=8)
orbit.orbit(dx, dy); orbit.pan(dx, dy, camera); orbit.zoom(steps)
orbit.apply(camera)
```

* перспектива и ортография (`ProjectionMode`), кэш видовой и проекционной матриц;
* `is_sphere_visible(center, radius)` — тест по шести плоскостям усечённого
  объёма (извлекаются из `view_projection`), на нём работает отсечение рендерера;
* `ray_from_screen(...)` — луч для выбора объектов мышью (пригодится редактору);
* `OrbitCameraController` ограничивает наклон и дистанцию, умеет фокус на объект.

Во вьюпорте редактора: ЛКМ — орбита, ПКМ/СКМ — панорама, колесо — зум, `R` —
сброс вида.

---

## 7. Камеры: компонент, стек, отсечение и предпросмотр

Камера — это самостоятельный компонент (`engine/rendering/camera.py`) с типом
проекции, параметрами отсечения, прямоугольником вьюпорта, цветом очистки и
маской слоёв. Все параметры, влияющие на матрицы, объявлены **свойствами**, и
любая запись сбрасывает кэш: иначе отсечение и отрисовка работали бы по
устаревшей матрице.

```python
from engine.rendering import Camera, CameraStack, ProjectionMode

camera = Camera(fov=50, near=0.1, far=300, name="Main")
camera.position = Vector3(4, 3, 4)
camera.look_at(Vector3.zero())

camera.get_view_matrix()          # world → view  (кэш)
camera.get_projection_matrix()    # view → clip   (кэш, перспектива или ортография)
camera.world_to_screen(point, 1280, 720)      # → (x, y, visible)
camera.screen_to_world(x, y, 1280, 720)       # ← точка на луче
camera.ray_from_screen(x, y, 1280, 720)       # луч для выбора объектов
```

Что важно в реализации:

* **Проекция**: `ProjectionMode.PERSPECTIVE` / `ORTHOGRAPHIC`. В ортографии
  высота объёма задаётся `ortho_height`, ширина выводится из `aspect`.
* **Aspect**: `aspect_mode="viewport"` (по умолчанию) следует за размером
  области вывода; `lock_aspect()` закрепляет отношение сторон — это нужно
  рендеру в текстуру, где кадр не должен «плыть» при растягивании панели.
* **Прямоугольник вьюпорта** (`viewport_rect`) позволяет камере занимать часть
  области вывода: так работает UI-камера в стеке и предпросмотр. Значения
  меньше либо равные 1.5 трактуются как доли, большие — как пиксели.
* **Цвет очистки и флаги** (`clear_color`, `clear_flags`): камера, у которой
  остался только `"depth"`, рисуется поверх уже нарисованного — это и есть
  механизм «UI поверх игры».
* **Маска отсечения** (`culling_mask`): слой `N` соответствует биту `N`;
  значение по умолчанию `0xFFFFFFFF` пропускает все 32 слоя.
* **Усечённый объём** (`camera.frustum`) — один объект `Frustum` на камеру,
  который обновляется **на месте** и только при изменении матриц. Горячий цикл
  отсечения не создаёт мусора; `frustum_planes()` возвращает тот же массив, а не
  новую копию. Умеет `intersects_sphere`, `intersects_aabb` (по `p`-vertex) и
  `frustum_corners()` — последнее используется гизмо камеры в редакторе.

Обратная матрица `view_projection` собирается **аналитически** из матрицы
проекции и видовой матрицы (`invert_view_projection`), а не через
`np.linalg.inv`: движок хранит матрицы в float32, и численное обращение
произведения теряет точность на элементах порядка `1/(far - near)` — точка
перестаёт попадать под курсор. Аналитическая формула не вызывает LAPACK, а
`screen_to_world` и `ray_from_screen` строятся через неё, поэтому выбор объектов
согласован с проекцией **по построению** (это проверяется случайными камерами в
`TestScreenWorldInvariant`).

### Камеры игровых объектов

`CameraComponent` связывает камеру с `GameObject`: позиция и ориентация берутся
из мирового трансформа владельца, а `local_offset` / `local_rotation` позволяют
сместить камеру относительно объекта (кокпит, вид от третьего лица).

```python
from engine.rendering import Camera, CameraComponent

component = cube.add_component(CameraComponent)   # камера следует за объектом
component.camera.fov = 40
component.sync_to_camera()                        # вызывается и каждый кадр
```

### Стек камер

`CameraStack` хранит несколько камер, выбирает активную и задаёт порядок
отрисовки по `depth` — камера с большим `depth` рисуется позже:

```python
stack = CameraStack([game_camera])
stack.add(ui_camera, make_active=False)     # ui_camera.depth = 10
stack.set_active("Main")                     # или индексом, или объектом
stack.begin_frame()                          # матрицы кадра фиксируются один раз

renderer.begin_frame(stack, (1280, 720))
...
```

### Рендер в текстуру и предпросмотр

`CameraPreview` — камера плюс `RenderTarget`: пересчитывает цель при смене
размера, умеет переключаться между камерами и отдавать пиксели кадра.

```python
from engine.rendering import CameraPreview

preview = CameraPreview(ui_camera, width=320, height=180)
preview.render(renderer, items)          # рисует сцену в свою текстуру
pixels = preview.read_pixels()           # (180, 320, 4) uint8
preview.set_camera(other_camera)         # переключение между камерами
```

### Производительность камер

* матрицы камеры передаются в шейдер **один раз на смену программы**
  (`Renderer._camera_uniforms` кэширует `view`, `projection` и их произведение
  по ревизии камеры);
* усечённый объём пересчитывается только при изменении камеры, а не на каждый
  submitted-объект;
* `frustum_planes()` переиспользует массив, `is_sphere_visible` не аллоцирует;
* отдельные uniform-матрицы `u_view` и `u_projection` дают шейдеру
  view-space-позицию (`v_view_position`) для эффектов глубины.

---

## 8. Рендерер: очередь, свет, батчинг

```python
renderer = Renderer(resources=resources, shader_root=ROOT / "assets")
renderer.initialize()

material = renderer.create_material("crate", base_color=Vector3(0.8, 0.5, 0.2), roughness=0.4)

renderer.begin_frame(camera, (1280, 720))
renderer.submit(mesh, material, model_matrix)                 # обычный объект
renderer.draw_instanced(mesh, material, matrices)             # пачка экземпляров
stats = renderer.end_frame()
print(stats.describe())
```

Что происходит внутри:

1. `begin_frame` готовит `FrameContext`, кэширует viewport, включает стандартное
   3D-состояние, очищает очереди и статистику;
2. `submit` считает мировые границы объекта и отбрасывает невидимые
   (`stats.culled`), остальные складывает в очередь; `draw_instanced` кладёт
   группу экземпляров в отдельную очередь (рисовать сразу нельзя — проход
   очистки выполняется позже и стёр бы кадр);
3. `end_frame` запускает граф проходов: `clear` → `shadow` → `geometry`;
4. проход геометрии сортирует объекты по материалу (непрозрачные) и по
   расстоянию (прозрачные), считает `batches` — число реальных переключений
   программы/текстуры, и рисует инстансные группы одним вызовом каждая.

Статистика кадра: `submitted`, `draw_calls`, `batches`, `triangles`, `vertices`,
`instances`, `culled`, `lights_bound`, `shadow_draws`, `shadow_triangles`,
`state_calls`, `state_saved`.

Расширение — добавление прохода:

```python
class ShadowPass:
    name = "shadow"

    def execute(self, context):        # context: FrameContext
        ...

renderer.add_pass(ShadowPass())        # выполнится после геометрии
```

### Спрайты

`SpriteBatch(atlas, pixels_per_unit=100)` (`sprite_batch.py`) копит спрайты одного
`SpriteAtlas` (`add(name, position, scale=, rotation=, color=, flip_x=, flip_y=)`) и
собирает вершины (9 float: позиция, UV, цвет) одним векторным вызовом numpy без цикла по
вершинам; `build()` отдаёт `(vertices, indices)` и проверяется без OpenGL. Геометрия
кадра учитывает точку привязки (pivot от верха исходного спрайта) и сдвиг обрезанных
прозрачных полей. `draw()` загружает всё в один динамический буфер и рисует одним
`glDrawElements`: программа `sprite` (`assets/shaders/sprite.vert|frag`), мировые
координаты через `u_view_projection` камеры, смешивание включено, запись глубины
выключена, отсечение граней выключено. Сортировка по глубине — на вызывающем
(порядок `add` = порядок рисования). `SpritePass(batch, ...)` — проход графа кадра
(`renderer.add_pass`), рисует после геометрии.

### Текст

Изолированная подсистема `engine/rendering/text/` (без Qt и без шейпинга; только
латиница и кириллица). Компонент `TextRenderer` (`engine/core/components.py`) — только
данные; сохраняется в сцене как обычный компонент (`scene_io`).

- `TextAtlas` (`atlas.py`) — Pillow рисует глифы набора символов в один атлас
  (полки 512→1024, белый RGB + альфа), `GlyphMetrics` на глиф; GL-текстура создаётся
  лениво (`texture()`), сам атлас тестируется без GL. LRU-кэш 16 атласов по ключу
  «путь + размер + bold + italic + charset». Шрифт по умолчанию — DejaVu Sans 2.37
  (`assets/fonts/`, лицензия рядом); bold/italic синтетические.
- `layout_text` (`layout.py`) — чистая функция: `
`, перенос по ширине, выравнивание,
  межстрочный интервал; результат кэшируется (`lru_cache`).
- `TextBatch` (`batch.py`) — векторизованная сборка вершин (9 float, как у спрайтов),
  один `glDrawElements` на атлас, сортировка `(sorting_layer, -distance)`, billboard
  берёт оси из видовой матрицы камеры, обводка — 8 смещённых копий под основным
  текстом. Кэш сборки: пока камера и содержимое не менялись, вершины не
  пересобираются и не загружаются в GPU заново.
- `TextPass` (`pass_.py`) — рисует после `transparent` (не чередуется с прозрачными
  мешами); программа `text` (`assets/shaders/text.vert|frag`).
- `fill_text_batch(batch, objects, roots, scene)` (`scene.py`) — общий для плеера и
  редактора сбор `TextRenderer`; при заданной `scene` пропускается, если не менялись
  `TransformStore.version`, `Scene.structure_version`/`activity_version` и
  `TextRenderer.revision`.

Замер (`tools/benchmark_text.py`, 100 текстов, 1290 квадов, 1280×720, HD 520):
+0,85 мс/кадр в статичной сцене, 1 draw call; при движущейся камере пересборка
вершин каждый кадр даёт ≈ +2,7 мс. Редактор-вьюпорт текст не пикает.

### UI / HUD

Экранные виджеты (`UIPanel`, `UIImage`, `UIText`, `UIButton`; данные — в
`engine/core/ui_components.py`, раскладка — `ui_layout.py`, ввод — `ui_system.py`)
рисует `UIPass` (`engine/rendering/ui/`): последний проход кадра, без depth,
ортографика в единицах UI (по умолчанию холст 1280×720, режим `fixed`; `UIScreen`
переводит UI ↔ пиксели и мышь). Порядок внутри прохода: квады (`UIBatch`: панели,
рамки, кнопки, картинки; один `glDrawElements` на текстуру/clip), затем текст
(`TextBatch.draw(projection=...)`, тот же шейдер `text`). Чередования квад/текст
нет; `clip` (scissor) действует только на квады. Вершины и индексы `UIBatch`
пересобираются и заливаются в буфер лишь при изменении пакета; `UIFiller` пропускает
сборку, пока раскладка не менялась (токен версий). Картинки читаются Pillow прямо из
файла (не через `AssetDatabase`). Редактор-вьюпорт UI не рисует.

Замер (`tools/benchmark_ui.py`, 50 виджетов, 1280×720, HD 520, проверки ошибок
PyOpenGL выключены как в плеере): ≈ +1,0 мс/кадр (медиана) в статике, 3 draw call;
при ежекадровой смене текста ≈ +4,8 мс (пересборка раскладки и текста).

---

## 9. Освещение: forward, отсечение и запекание

Свет живёт в `engine/rendering/lighting.py`. Четыре типа источников
(`DirectionalLight`, `PointLight`, `SpotLight`, `AmbientLight`) наследуют общий
`Light` и умеют сериализоваться, копироваться и описывать свою зону влияния.

```python
from engine.rendering import (
    AmbientLight, DirectionalLight, PointLight, SpotLight,
)

sun = DirectionalLight(direction=Vector3(-0.5, -1.0, -0.3), cast_shadow=True)
sun.shadow_map_size = 512            # значение по умолчанию — под слабые GPU
renderer.lights.set_lights([
    sun,
    AmbientLight(intensity=0.18),
    PointLight(position=Vector3(-3, 3, 0), range=8.0, falloff=2.0),
    SpotLight(position=Vector3(3, 4, 2), direction=Vector3(0, -1, 0), outer_angle=26.0),
])
```

### Как это работает за кадр

1. `LightManager.collect(objects)` обходит сцену и собирает источники из
   компонентов `LightComponent`, вызывая `sync_to_light()`: свет следует за
   трансформом объекта. Приоритет — направленный, фоновый, затем местные по
   убыванию силы;
2. для каждого объекта `select(center, radius, layer)` отсекает источники по
   сфере влияния. Точечный свет за своим `range` даёт ровно ноль, прожектор —
   ещё и по конусу, направленный и фоновый влияют всегда;
3. результат кладётся в кэш по ключу границ объекта, поэтому повторные кадры
   без движения не пересчитывают набор;
4. `LightBuffer` раскладывает источники по uniform-массивам (structure of
   arrays) и загружает их одним вызовом на массив. Загрузка пропускается, если
   тот же набор уже загружен для этой программы (`stats.upload_hits`);
5. `shaders/lighting.frag` считает Blinn-Phong по каждому источнику. Цикл имеет
   постоянную границу `MAX_LIGHTS` (по умолчанию 8), поэтому драйвер
   разворачивает его без ветвлений.

Потолок источников на объект настраивается: `renderer.lights.max_lights = 4`.
Это первый рычаг для слабого GPU — уменьшение потолка уменьшает цикл во
фрагментном шейдере.

### Тени

Тени строятся только для направленного источника: `cast_shadow = True` на
`DirectionalLight`. Проход выполняется перед геометрией и пишет глубину со
стороны солнца в карту 512×512 (`shadow_map_size`).

Матрица `directional_light_matrix` даёт **NDC** (`-1..1` по глубине): перевод в
`0..1` делает сам OpenGL по формуле `z_win = (z_ndc + 1) / 2`. Если записать
`0..1` уже в матрицу, драйвер применит преобразование второй раз, диапазон
уедет вдвое и сравнение с текстурой перестанет срабатывать — шейдер обязан
делать то же преобразование перед сравнением.

Область карты подбирается по дальности теней и переднему плану кадра, а не по
дальней плоскости камеры: при `far = 100` ортобокс раздувался до ~59 единиц, и
одноточечная тень занимала единицы текселей. Дополнительно объекты вне объёма
источника отсекаются (`ShadowMap.is_visible`): проход рисует геометрию второй
раз, и это самая дорогая его часть.

Качество карты: PCF 3×3 вместо усреднения глубины, привязка центра области к
сетке текселей (тени не «плывут» при движении камеры), near-плоскость отсечения
по `m[3]+m[2]`, `shadow_distance` не ограничен 25. Экранный FBO у
`QOpenGLWidget` не равен 0: его id хранится в `GLState` (из `paintGL`,
`defaultFramebufferObject()`), и проходы теней/RT возвращаются именно в него.
`Material.apply` не затирает `u_light_*[0]`; набор источников кэшируется по
программе.

`shadow_bias` убирает «акне» на пологих поверхностях, `normal_bias` —
самозатенение; значения по умолчанию (0.0015 и 0.02) подходят для сцен
масштаба единиц.

### Запекание под слабые устройства

```python
renderer.lights.dynamic = False                 # шейдер не считает источники
colors = renderer.lights.bake_to_vertex_colors(mesh, ambient_only=False)
```

Запекание считает освещение по нормалям меша и складывает результат в цвета
вершин. Материал с `dynamic_lighting = False` компилируется с
`DYNAMIC_LIGHTING=0`: цикл по источникам исчезает из шейдера целиком. Второй
вариант — `Material.lightmap` (текстура в юните 2, define `USE_LIGHTMAP`).

### Производительность редактора

Вьюпорт редактора сам инстансит одинаковые (меш, материал) от 4 штук
(`auto_instance_min`), uniform-ы теней ставятся раз на программу за кадр.
`scripts/visual_test.py` (окно целиком, HD 520, low): пустая сцена ~54 FPS,
1000 кубов + 16 источников ~22 FPS, 5000 — ~10, 10000 — ~6 FPS; до
оптимизаций 1000 объектов давали ~8 FPS. Отклик: выбор→инспектор ~38 мс,
создание куба ~77 мс при 1000 объектов, `notify_modified` <1 мс.

### Производительность

Замеры: `tools/benchmark_lighting.py` (по умолчанию поштучная отрисовка,
`--instanced` — пачкой). На Intel HD 520, кадр 1280×720, 200 объектов:

| Режим | Время кадра |
| --- | --- |
| 2 источника (фон + солнце), инстансинг | ~11 мс |
| 10 источников (фон + солнце + 8 местных), инстансинг | ~11.7 мс |
| то же + карта теней 512×512, инстансинг | ~12.3 мс |
| запечённый свет, инстансинг | ~9.2 мс |

При инстансной отрисовке переход от 1 к 8 местным источникам не меняет время
кадра — разница в пределах шума, потому что отсечение оставляет в наборе
каждого объекта лишь несколько источников, а цикл разворачивается без
ветвлений. Карта теней 512×512 добавляет около 1.2 мс.

При поштучной отрисовке (200 отдельных `submit`) кадр стоит ~170 мс и **не
зависит** от числа источников: узкое место — число вызовов драйвера, а не свет.
Поэтому на слабой графике сначала включайте инстансинг, затем снижайте
`max_lights`, и только потом трогайте параметры карты теней.

---

## 10. Цели вывода и путь к deferred
```python
from engine.rendering import RenderTarget

g_buffer = RenderTarget("gbuffer", 1280, 720, color_attachments=3, samples=0)
target = RenderTarget("hdr", 1280, 720)

renderer.begin_frame(camera, (1280, 720), target=g_buffer)
...
renderer.end_frame(target=g_buffer)
pixels = g_buffer.read_pixels(attachment=0)      # (H, W, 4) uint8
```

`RenderTarget` умеет MRT (`glDrawBuffers`), буфер глубины,
`resize()` с пересозданием и чтение любой цветовой привязки. `Capabilities`
подсказывает, стоит ли идти в deferred: `capabilities.can_defer_shading`.

План развития (следующие шаги, интерфейсы уже готовы):

1. **Тени**: `ShadowPass` с картой глубины на `RenderTarget`, каскады по тирам.
2. **Deferred**: `GBufferPass` (3 привязки) + `LightingPass` (полноэкранный
   квад) — граф и MRT на месте.
3. **Пост-обработка**: тонемаппинг и FXAA сделаны (§10a), bloom — A10b.
4. **Отсечение**: BVH/иерархия сцены поверх `Camera.is_sphere_visible`,
   GPU-запросы окклюзии.
5. **Батчинг**: статическое объединение геометрии по материалу и
   `GpuInstanceBuffer` для динамики.

---

## 10a. Пост-обработка (A10a)

Код: `engine/rendering/post.py` (`PostStack`, `PostSettings`, `tier_render_scale`),
`dynamic_resolution.py` (`DynamicResolution`, `read_render_hints`), `gpu_timer.py`
(`GpuTimer`, `GL_TIME_ELAPSED`), шейдеры `assets/shaders/post/`
(`fullscreen.vert`, `tonemap.frag`, `fxaa.frag`, `fade.frag`).

- **HDR**: `RenderSettings.hdr` (по умолчанию `False`; новые проекты при medium — `True`).
  Поле `hdr` в `*.project.json` трёхсостоянное (`read_hdr_hint`): нет поля — сцена с
  секцией `post` включает HDR сама (плеер и вьюпорт), `true` — включён, `false` —
  явное решение автора, сцена его не перекрывает (баннер в инспекторе, WARNING в плеере).
  Сцена рисуется в RGBA16F; буфер хранит гамма-кодированные значения без верхнего
  клампа, tonemap декодирует, применяет экспозицию и кривую, кодирует обратно.
- **Порядок**: сцена (HDR, масштаб `render_scale`) → fused-проход tonemap
  (`linear`/`reinhard`/`aces` + экспозиция; ветка `u_use_lut` есть, но выключена) →
  FXAA (после tonemap, в LDR; при выключенном FXAA tonemap пишет сразу в вывод) →
  fade/flash (`ScreenFx`, отдельный микро-проход, альфа-смешивание).
- **Секция сцены `post`**: `tonemap`, `exposure`, `fxaa`, `bloom`, `bloom_threshold`,
  `bloom_intensity`, `bloom_levels`, `vignette`, `vignette_inner/outer/intensity`,
  `lut` (`{"guid","path"}` | `null`), `lut_intensity`; нет секции — умолчания тира
  (`PostSettings.for_quality`). Старые секции без новых полей читаются с выключенными
  bloom/виньеткой/LUT. Формат сцены v2 не менялся.
- **Bloom (A10b)**: `engine/rendering/bloom.py` (`BloomChain`), шейдеры `post/bloom_bright.frag`
  (`c·smoothstep(threshold, threshold+soft_knee, luma)`, soft_knee 0.5), `blur.frag`
  (5-тап, `u_direction`), `bloom_up.frag`. Каскад линейных RGBA16F: bright-pass →
  L0 (1/2 кадра), на каждом уровне два прохода blur (H, V), уровень i+1 — 1/2 от i;
  upsample аддитивным блендом (ONE, ONE) от меньшего к большему в L0. Цели каскада
  переиспользуются при resize (создаются заново только при росте числа уровней).
  Тонемап читает L0 (texture unit 1) и складывает с сценой **до** сжатия кривой:
  формулировка «после tonemap» противоречит «bloom считается в HDR», поэтому bloom
  считается из HDR и добавляется внутри fused-прохода tonemap, а FXAA остаётся в LDR.
- **Виньетка**: в том же fused-проходе, после LUT и до fade:
  `color *= mix(1, 1 - intensity, smoothstep(inner, outer, r))`
  (`u_inner`=0.5, `u_outer`=0.9, `u_intensity`=0.3).
- **LUT**: только PNG 256×16 без сжатия (16 срезов по B вдоль X, внутри среза r по X,
  g по строкам; тождественная — `(r,g,b)/15`, `engine/rendering/lut.py`: `make_lut`,
  `identity_lut`, `save_lut_png`). Ассет с GUID (`.meta`, Pillow), ссылка
  `AssetRef`; загрузка через VFS (`as_assets` → `texture`-артефакт →
  `GpuTexture`, texture unit 2, BILINEAR, без sRGB). `u_lut_intensity` 0..1:
  `mix(original, lutted, intensity)`. `PostStack.assets` задаёт VFS (плеер — `db`,
  редактор — `context.asset_db`). `.cube` не поддерживается.
- **Масштаб и dyn-res**: поля проекта `render_scale_hint`, `dynamic_resolution` в
  `*.project.json`; флаги плеера `--render-scale`, `--no-dynamic-res`, `--hdr on|off`.
  `DynamicResolution`: EMA 0.2, шаг 0.05; вниз — среднее > 0.95·16.6·0.9 мс за 4 кадра
  (cooldown 12), вверх — < 0.75·цель за 90 кадров (cooldown 60); вход —
  `max(gpu_ms, cpu_ms)`. Смена масштаба шлёт `RenderScaleChanged`.

| эффект | low | medium | high |
| --- | --- | --- | --- |
| HDR | off | on | on |
| Tonemap | linear | aces | aces |
| FXAA | off | on | on |
| Bloom | off | **off** (`BLOOM_IN_MEDIUM=False`, см. замеры) | on, 4 уровня |
| Vignette | off | on | on |
| LUT | off | on, если задан | on, если задан |
| Render scale | 0.5 | 0.75 | 1.0 |
| Dynamic | on (0.5–0.75) | on (0.5–1.0) | off |

Бюджеты post — **целевые, не контракт**. Реальные замеры HD 520
(`tools/benchmark_post.py`, мс на проход поста, медиана 60 кадров; шум iGPU ±10 %):

| tier / вариант | бюджет 720p / 540p | факт 720p | факт 540p | оценка |
| --- | --- | --- | --- | --- |
| low | 1.0 / 0.7 | 0.90–1.10 | 0.77 | граница шума |
| medium без bloom | 3.5 / 2.5 | 2.08–2.23 | 1.49–1.67 | в бюджете |
| medium + LUT | 3.5 / 2.5 | +0.05…0.16 | +0.0…0.2 | в бюджете |
| medium с bloom (3 ур.) | 3.5 / 2.5 | 4.09 | 2.99 | не влезает → bloom в medium не включён |
| high (bloom 4 ур.) | 5.0 / 3.5 | 4.69–5.50 | 3.90–4.28 | 720p на границе шума; 540p не гарантируется |
| high без bloom | 5.0 / 3.5 | 2.38 | 1.86 | в бюджете |

На iGPU уровня HD 520 high — компромисс: bloom стоит ≈2.4–3 мс, поэтому он только в high.
Правило: bloom в medium, только если укладывается в бюджет medium. Dyn-res в окне с
vsync не проверялся.

Не делалось: `.cube`, SSAO, motion blur, DOF; превью камеры в редакторе без
пост-обработки.

---

## 11. Производительность и замеры

Рендер-тесты и демо гонялись на **Intel HD Graphics 520** (OpenGL 3.3, GLSL 3.30) —
это ровно тот класс «слабого железа», под который проектируется движок.

| Сцена | Вызовов отрисовки | Треугольников | Комментарий |
|---|---|---|---|
| Куб + сетка | 2 | 26 | базовый каркас |
| UV-сфера + сетка | 2 | 1 294 | процедурная геометрия |
| 400 кубов инстансингом | 1 (для пачки) | 4 800 | одна пачка на 400 объектов |
| 144 сферы (48×32) | 2 | 185 600 | ~1 290 тр/объект, один вызов |

Замер `tools/benchmark_render.py` (кадр 1280×720, 20 кадров, тот же Intel HD 520):
одинаковые кубы, разложенные сеткой, — сначала одной инстансной пачкой, затем
поштучно через `submit`.

| Объектов | Инстансинг | Поштучно | Ускорение | Draw calls |
|---|---|---|---|---|
| 500 | 26.5 мс | 195.5 мс | 7.4× | 1 против 500 |
| 2 000 | 94.5 мс | 842.4 мс | 8.9× | 1 против 2 000 |
| 5 000 | 230.9 мс | 2 036.6 мс | 8.8× | 1 против 5 000 |

Выигрыш складывается из двух частей: исчезает Python-оверхед постановки в
очередь (около 30 мкс на объект) и исчезают тысячи вызовов драйвера. Остаток
времени — уже чистая нагрузка GPU на заполнение пикселей, её уменьшают
LOD, отсечение и батчинг статики (следующие этапы).

Практические выводы, которые уже учтены в коде:

* сортировка по материалу и кэш состояния убирают большинство переключений;
* инстансинг снимает главное узкое место — количество вызовов отрисовки;
* отсечение по усечённому объёму работает до draw call (в тесте объект за
  камерой не попадает в очередь, а группа из 1000 экземпляров за камерой
  отбрасывается целиком: `stats.culled == 1000`).

Замеры CPU плеера (`cpu_ms`) сравнимы только в одном режиме: `--headless` даёт ≈6,4–6,6 мс (~150 FPS), старые 7,28 мс (137 FPS) сняты тем же способом до кэша uniform-ов и `_transformed_bounds` (разброс прогонов ±0,2 мс); в окне кадр упирается в vsync (16,5 мс, 60 FPS) и не отражает стоимость CPU.

---

## 12. Ограничения текущего этапа

* PBR без IBL: фон плоский, металлы отражают только источники и грубую
  аналитику; нет лайтмап и параллакса.
* Тени — только от направленного света (карта теней одна); у точечных и
  прожекторов теней нет. Флаги `cast_shadow`/`receive_shadow` задаются
  на `MeshRenderer` и работают независимо.
* Прозрачность сортируется по `(Material.sorting_layer, -расстояние)`;
  слой читается из `.mat.json` (`sorting_layer`) и переопределяется полем `MeshRenderer.sorting_layer` (≠0, инспектор «Слой сортировки»).
* MSAA: offscreen-`RenderTarget` с `samples>0` рисует в multisample-FBO, а
  `resolve()` (`glBlitFramebuffer`) переносит результат в текстуру; вызывается
  из `end_frame`. Пост-обработка по-прежнему не использует MSAA-цели.
* Нет скелетной анимации в рендере (паллета костей поддержана в `set_uniform`,
  понадобится на этапе анимации).
* Камеры рисуются в один проход: стек задаёт порядок и прямоугольники вьюпорта,
  но полноценные «слои поверх с перерисовкой глубины» (отдельный проход с
  очисткой) — следующий шаг; сейчас верхняя камера очищает глубину целиком.
