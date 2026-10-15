# Ядро MyEngine — API и устройство

Документ описывает реализованное ядро (`engine/core/`). Ядро не зависит ни от PySide6, ни от
OpenGL: игровую логику, иерархию, ресурсы и цикл можно запускать headless (тесты, серверные
сборки, инструменты импорта). Рендер и редактор подключаются поверх через порты.

Быстрая проверка:

```bash
.venv\Scripts\python.exe scripts\run_demo.py      # демонстрация всех подсистем
.venv\Scripts\python.exe -m pytest                # 146 тестов
.venv\Scripts\python.exe tools\benchmark_core.py  # замеры производительности
```

---

## 1. Состав

| Модуль | Назначение |
|---|---|
| `engine.core.math3d` | `Vector3`, `Quaternion`, матрицы 4×4, batch-функции `compose_matrices`, `invert_rigid`, `trs_from_matrix`, `transform_point` |
| `engine.core.game_object` | `GameObject`: имя, тег, слой, трансформ, компоненты, иерархия, отложенное уничтожение |
| `engine.core.component` | `Component`: жизненный цикл (`on_attach` → `on_enable` → `on_start` → кадры → `on_destroy`) |
| `engine.core.transform` | `Transform` (фасад) и `TransformStore` (numpy-хранилище мировых матриц) |
| `engine.core.scene` | `Scene`: реестры, запросы, системы, фазы кадра, статистика |
| `engine.core.system` | `System`: пакетная обработка сцены с приоритетом |
| `engine.core.event_bus` | `EventBus`, `Event`, `Subscription`: приоритеты, слабые ссылки, отложенная доставка |
| `engine.core.game_loop` | `GameLoop`, `Time`, `LoopStats`: фиксированный шаг физики + переменный шаг логики |
| `engine.core.resources` | `ResourceManager`, `AssetLoader`, `LoaderRegistry`, ассеты и загрузчики |
| `engine.core.resources.primitives` | процедурные меши: `cube_mesh`, `uv_sphere_mesh`, `plane_mesh`, `line_grid_mesh`, `wire_box_mesh`, `compute_normals` |
| `engine.core.components` | `MeshRenderer` — описание отрисовки объекта (меш, цвет, шероховатость, металличность) без привязки к GPU |
| `engine.core.log` | единая настройка логирования |

Рендеринг (`engine.rendering`) описан в `docs/RENDERING.md`, редактор
(`editor/`) — в `docs/EDITOR.md`.

---

## 2. Игровые объекты и компоненты

```python
from engine.core import Component, Scene, Vector3

class Spinner(Component):
    def __init__(self, speed: float = 90.0) -> None:
        super().__init__()
        self.speed = speed

    def on_start(self) -> None:
        print(f"{self.game_object.name} готов")

    def on_update(self, dt: float) -> None:
        self.transform.rotate(Vector3.up(), self.speed * dt)

scene = Scene("Level")
cube = scene.create_game_object("Cube", position=Vector3(0, 1, 0), tag="Prop")
cube.add_component(Spinner, speed=180.0)
child = scene.create_game_object("Child", parent=cube, position=Vector3(0, 0, 1))

scene.update(1 / 60)
print(child.transform.world_position)   # мировая позиция с учётом родителя
```

Особенности:

* **Несколько компонентов одного типа разрешены**; `get_component` возвращает первый,
  `get_components` — все в порядке добавления.
* **Объект можно создать вне сцены** (`GameObject("Prefab")`), собрать поддерево и позже
  добавить его в сцену: `scene.add_game_object(obj)` переносит поддерево с сохранением
  локальных координат.
* **Уничтожение отложено до конца кадра** (`obj.destroy()`), поэтому удалять объекты можно
  прямо из логики и обработчиков событий. `destroy(immediate=True)` удаляет сразу.
* **Активность**: `obj.active_self = False` выключает поведение всего поддерева; компоненты
  неактивных объектов не обновляются и не попадают в `scene.active_components`.

### Идентичность объекта: `uid` и версионирование сцен (A4.1)

* У каждого `GameObject` два идентификатора. `id` — рантайм-номер (строка
  `TransformStore`, ключ `Scene._objects`): сцена выдаёт его заново при каждой
  загрузке и в файл он не пишется. `uid` — стабильный идентификатор: 16 hex-символов
  (64 бита, `secrets.token_hex(8)`), создаётся при конструировании объекта, хранится
  в сцене и переживает сохранение, снимки Play Mode и undo/redo.
* `Scene.find_by_uid(uid)` ищет объект по индексу `Scene._by_uid`. Индекс производный:
  не сериализуется, ведётся при создании/усыновлении поддерева, уничтожении (в том числе
  отложенном) и `clear()`. Присвоение `obj.uid = ...` проверяет формат (`ValueError`) и
  уникальность в сцене; конфликт uid при добавлении поддерева или загрузке — ERROR в лог
  и выдача нового uid.
* **Формат сцены v2** (`SCENE_VERSION = 2`): у объекта `uid`, `parent` —
  `{"uid": ...}` или `null`, ссылки на объекты в полях скриптов — `{"uid": ...}`;
  целого `id` в файле нет. `save_scene` всегда пишет v2.
* **Чтение v1**: числовые `id`/`parent` и ссылки `{"object": id}` читаются (в том числе в
  смешанных файлах). Объектам v1 uid присваивается детерминированно:
  `blake2b(f"v1:{id}", digest_size=8).hexdigest()` (`engine.scene_io.migrated_uid`), так что
  повторная загрузка того же файла даёт те же uid. Файл сам по себе при загрузке не меняется.
  Версия новее 2 — `ValueError`.
* **Бэкап**: перед записью `save_scene` смотрит целевой файл на диске. Если он существует,
  `version == 1` и рядом нет `<имя>.v1.bak` — файл копируется в `<имя>.v1.bak` (лог
  «Сцена мигрирована v1->v2, бэкап: …»). Существующий `.v1.bak` не перезаписывается;
  для нового файла, v2-файла или нечитаемого файла бэкап не создаётся; если в целевом файле
  версия > 2, он перезаписывается без бэкапа с предупреждением в лог.

### Префабы (A4.2)

* **Файл** `*.prefab.json` (`format: "myengine.prefab"`, `version: 1`, `name`, `root`,
  `objects`). Объект — запись формата сцены, но вместо `uid` у него `local_id`, `parent` —
  `{"local": ...}` и ссылки в полях — `{"local": ...}`. Валидация — `prefab_data.validate_prefab_data`
  (`PrefabError`); чистые данные и граф зависимостей (`check_prefab_graph`) лежат в
  `engine/core/prefab_data.py`, сборка и instantiate — в `engine/prefab.py`.
* **Импорт**: `PrefabImporter` (`*.prefab.json`) валидирует файл и копирует байты в
  `Library/<guid>/prefab.json` без нормализации; `info["nested"]` — GUID вложенных префабов.
  Зависимости регистрируются **по пути** (`ctx.add_dependency` принимает путь, не GUID),
  поэтому правка вложенного префаба переимпортирует внешний.
* **`Scene.instantiate(prefab, parent=None, position=None, rotation=None)`** возвращает корень.
  `prefab` — `Prefab` либо строка guid/путь (разрешается через `Scene.prefab_loader`, затем
  загрузчик по умолчанию `set_default_prefab_loader`; `make_prefab_loader(assets, root)`).
  Порядок: проверка родителя -> `check_prefab_graph` -> подбор `spawn_serial` без коллизий uid ->
  `build_objects` -> `set_parent` -> `position`/`rotation` на корень. При ошибке сцена не меняется.
* **uid** детерминированы: `instance_uid = blake2b("inst:"+guid+":"+serial, 8)`,
  `uid объекта = blake2b(instance_uid+":"+local_id, 8)`. `Scene._spawn_serial` в файл не пишется;
  при коллизии (например, после загрузки сцены) serial растёт до свободного.
* **`PrefabInstance`** (`engine/core/prefab_components.py`) — маркер на корне экземпляра:
  `prefab_guid`, `prefab_path`, `overrides`, `instance_uid` (A4.3; пусто — запись A4.2).
  Сериализуется в сцене как обычный компонент; версия сцены остаётся 2.
* **Overrides (A4.3)** (`engine/prefab_overrides.py` — чистые функции над dict, `engine/prefab.py`
  — применение). Запись: `{"target": "L0", "component": "MeshRenderer", "field": "color",
  "value": [1,0,0]}`. `target` — путь по `local_id` (вложенные: `placeholder/subpath`, `/` в
  `local_id` запрещён); `Transform` — особый компонент; поля корня `name/active/tag/layer` —
  `component="Root"`; `field` через точку для вложенных dict, списки и векторы — целиком.
  Трансформ корня экземпляра — размещение, а не override.
  * **Сохранение**: `scene_to_dict` схлопывает экземпляр в одну запись корня
    (`uid`, `parent`, `transform`, маркер с `instance_uid` и `overrides`); члены экземпляра
    не пишутся, добавленные объекты — обычные объекты с `parent {"uid"}`.
  * **Дифф**: база — нормализованное раскрытие префаба с каноническим `instance_uid="0"*16`
    во временной сцене (LRU 32); `new = diff + stale_from_previous`. Запись, поля которой нет
    в префабе, получает `"stale": true`, сохраняется и оживает при загрузке/пересборке
    (один WARNING на экземпляр).
  * **Загрузка**: раскрытие экземпляров -> применение overrides -> разрешение ссылок по uid.
    Нужен загрузчик префабов (`scene.prefab_loader`, `set_default_prefab_loader` или
    fallback `make_prefab_loader(assets)`). Если префаб не грузится — остаётся заглушка корня;
    при сбое диффа экземпляр пишется развёрнуто (как в A4.2).
  * **API**: `Prefab.get_overrides`, `Prefab.apply_override_to_asset`, `Prefab.revert_override`,
    `capture_instances`/`restore_instances` (пересборка с сохранением overrides).
  * **Не выразимо**: удаление компонентов и объектов экземпляра, перенос члена к другому
    родителю (при загрузке воскреснут) — Unpack/Remove в A4.4+. Пересборка
    (`restore_instances`) сохраняет родителя и позицию корня среди братьев (A12 п.4):
    позиции снимаются до первой правки, регионы вставляются по возрастанию индекса.
* **Вложенность**: объект с `PrefabInstance` внутри префаба — заместитель; он заменяется
  корнем вложенного экземпляра (name/tag/layer/active/transform/parent берутся из заместителя),
  его дети подвешиваются к этому корню. `nest:`-uid вложенного экземпляра зависит от uid
  объекта-заместителя, а не от корня внешнего. Глубина <= 8, повтор GUID в цепочке —
  `PrefabCycleError`. `prefab_to_dict` схлопывает вложенные экземпляры в один лист.

### Unpack и сохранение как префаб (A4.5)

`engine/prefab_save.py`: `save_as_prefab(scene, root, path, assets=, project_root=)` пишет
`*.prefab.json`, регистрирует его в `AssetDatabase` (нужен GUID для вложенных префабов),
создаёт на месте поддерева экземпляр и перепривязывает внешние ссылки `old_uid -> new_uid`
(поля `ScriptComponent` через `set_field`, `_pending_refs`, значения overrides других
экземпляров); `unpack_instance(obj)` снимает `PrefabInstance` с корня экземпляра (корень
ищется вверх, `find_instance_root`), оставляя объекты, `uid` и значения как есть,
`pooled=False` для членов (вложенные экземпляры не затрагиваются). Overrides при Unpack
«запекаются» - в сцене они и так уже применены к объектам.

### Пулы объектов и запросы (A4.4)
* **`ObjectPool(scene, prefab, *, prewarm=0, max_size=None, on_exhausted="grow"|"reuse_oldest"|"return_none")`**
  (`engine/core/pool.py`): `get(position, rotation)`, `release(obj)`, `size` (выдано), `free`.
  Свободные хранятся LIFO. `get` включает объект (`active_self=True`), затем зовёт `on_spawn`;
  `release` зовёт `on_despawn` **до** выключения, возвращает корень к исходному родителю и
  локальным трансформам. Хуки вызываются у всех компонентов поддерева (DFS); исключение
  хука логируется и пул не ломает. `max_size` + `grow` - `get` вернёт `None` с предупреждением,
  `reuse_oldest` - самый давний выданный, `return_none` - `None` молча.
* **Не сохраняется в сцену**: корень экземпляра получает `GameObject.pooled=True`,
  `serialize_objects` пропускает такое поддерево (состояние пула восстанавливает владелец).
  Ручные `Scene.instantiate` сохраняются как раньше.
* Скорости `RigidBody` обнуляются при `get` (`ScenePhysicsAdapter.stop_body`, без
  синхронизации структуры). Уничтоженные вне пула объекты обнаруживаются при следующем `get`
  (ERROR в лог, создаётся новый). Прогрев синхронный (~3 мс на экземпляр): делайте при
  загрузке сцены; >50 - предупреждение. Замеры: `get+release` x100 6 мс против 307 мс
  у `instantiate` (~51x).
* **Запросы**: `Scene.find_all_with_tag(tag)` (только активные; индекс по тегу обновляется
  сеттером `GameObject.tag`), `GameObject.get_component_in_children`,
  `get_components_in_children(include_inactive=False)`, `get_component_in_parent`.

### Команды отмены (A12, `engine/core/commands/`)

Headless-слой undo/redo (без Qt); редактор владеет стеком на документ
(`SceneDocument.commands`). Запись пропорциональна правке, а не сцене.

* `CommandStack`: `execute`/`push` (уже выполненное)/`undo`/`redo`, транзакции
  (`transaction(name)` - одна запись, внутри слушатель молчит, по выходу один сигнал),
  `top`, `state_id`, `is_clean`/`mark_clean`, лимит `DEFAULT_LIMIT=200`. Слияние: правки одного
  поля одного компонента в окне `MERGE_WINDOW=1.0` с (`SetFieldCommand`,
  `SetScriptFieldCommand` - ключ компонент + имя поля).
* Команды: `CreateObject/DeleteObject/Rename/Reparent/SetActive/SetTag/SetLayer`,
  `AddComponent/RemoveComponent/SetField/SetScriptField`, `ReplaceSubtreesCommand`
  (замена поддеревьев по записям сцены v2: пересборка экземпляров, Unpack, «Сохранить как
  префаб»; области не вкладываются). `Scene.set_parent` / `GameObject.set_parent(index=)`
  задаёт позицию среди братьев: reparent и undo возвращают порядок A-B-C.
* **Идентичность - по `uid`**: undo создания/удаления пересоздаёт объект (тот же `uid`, новый
  runtime-`id`); долгоживущие ссылки хранить как `uid` и искать `Scene.find_by_uid`.
  `ReplaceSubtreesCommand` и `_restore_subtree` строят объекты с `expand_instances=False`.
* Удаление командой уничтожает объекты **сразу** (`immediate=True`), не в конце кадра.
* Ограничения: ссылки полей других скриптов на удалённый объект после undo не
  восстанавливаются; порядок `scene.objects` после undo удаления может отличаться (порядок
  братьев сохраняется); undo «Применить к префабу» откатывает только сцену, файл
  `*.prefab.json` остаётся изменённым; композитные записи не сливаются (каждое поле
  скрипта - своя запись, но повторные правки одного поля сливаются).
* Play Mode использует собственный `SceneSnapshot.freeze/restore` (не затронут).

### Иерархия трансформов

Компоненты держат данные, а иерархия — отдельное плоское хранилище:

* локальные позиция/кватернион/масштаб лежат в numpy-массивах сцены;
* мировые матрицы пересчитываются **пакетно по уровням** (`np.matmul`), а не поэлементно;
* пересчитываются только изменённые поддеревья; если ничего не менялось,
  `TransformStore.update_world_matrices()` возвращает `0`;
* матрицы — column-vector (`v' = M @ v`), композиция `world = parent_world @ local`,
  сдвиг в последнем столбце (совместимо с OpenGL/Unity-подобной конвенцией).

---

## 3. Запросы (ECS-стиль) и системы

```python
for obj in scene.query(Spinner, Health):        # объекты, у которых есть ВСЕ компоненты
    ...

class MoveSystem(System):
    priority = 100                              # больше — раньше в фазе

    def on_added(self, scene): ...
    def on_fixed_update(self, scene, dt): ...
    def on_update(self, scene, dt): ...
    def on_late_update(self, scene, dt): ...
```

`Scene.query` кэширует результат до следующего структурного изменения сцены и возвращает
**тот же неизменяемый кортеж** — без копирования (22.7 мкс на вызов при 5 000 объектов).

Порядок фаз кадра:

```
Scene.fixed_update(dt)  → системы → компоненты → пересчёт иерархии
Scene.update(dt)        → scheduler.update_game → process_pending → events.flush → системы → компоненты
                          → пересчёт иерархии → process_pending
Scene.late_update(dt)   → системы → компоненты → пересчёт иерархии
```

---

## 4. События

```python
from dataclasses import dataclass
from engine.core import Event

@dataclass(slots=True)
class Pickup(Event):
    points: int

scene.events.subscribe(Pickup, on_pickup, priority=10)
scene.events.emit(Pickup(5))            # немедленная доставка
scene.events.emit_deferred(Pickup(7))   # доставка в начале следующего update
```

* Обработчики вызываются по убыванию `priority`; подписка на базовый класс получает события
  наследников.
* Методы удерживаются **слабо**: подписка исчезает вместе с объектом-подписчиком (утечек нет).
* Ошибка обработчика логируется и не мешает остальным; вложенные `emit` защищены лимитом
  глубины.

### 4.1 UI, мышь и сцены

* `MouseState` (`core/input.py`) — позиция, кнопки и edge-события за кадр; заполняется
  GLFW-коллбеками плеера или `inject_mouse` (headless/IPC). `InputState` скриптов
  (клавиши) не менялся.
* `UISystem` делает hit-test виджетов (слой ↓, порядок создания ↓) и шлёт события
  `UIHoverEnter/Exit`, `UIPress`, `UIRelease`, `UIClick`; кнопка с `on_click`
  `event:<имя>` порождает `UIEventFired`, с `load_scene:<путь>` — `SceneLoadRequested`.
  `QuitRequested` из скрипта останавливает плеер.
* `SceneManager(loader)` (`core/scene_manager.py`): `load/unload/set_active`,
  события `SceneLoaded/Unloaded/Activated`. Загрузчик внедряется (плеер подставляет
  `engine.scene_io`), чтобы core не зависел от rendering/scripting.

---

## 5. Ресурсы

```python
from engine.core import ResourceManager

resources = ResourceManager(roots=[Path("assets")])
mesh = resources.load("primitives/cube.obj")      # читается с диска один раз
same = resources.load("primitives/cube.obj")      # берётся из кэша
resources.release(mesh)
resources.unload_unused()                         # освобождает свободные ресурсы
```

* Ключ кэша — нормализованный путь **и** параметры импорта: один файл с разными настройками
  даёт разные ресурсы.
* Счётчик ссылок: ресурс выгружается, когда на него никто не ссылается (`unload_unused`) или
  принудительно (`unload(..., force=True)`); `reload()` перечитывает файл (hot reload).
* Форматы «из коробки»: `.obj` (парсер с дедупликацией вершин и триангуляцией), изображения
  `.png/.jpg/.jpeg/.bmp/.gif/.dds` (метаданные читаются из заголовка, BC-форматы не
  распаковываются), GLSL `.glsl/.vert/.frag/.geom/.comp` с `#include` и `#pragma once`.
* Новый формат — наследник `AssetLoader` + `register_loader`; код менеджера не меняется.

### 5.1 Пайплайн импорта (`AssetDatabase`)

Над `ResourceManager` (загрузка в рантайме) лежит офлайн-слой импорта; менеджер он не
заменяет, а после реимпорта перечитывает уже загруженный ассет (`manager.reload`).

```python
from engine.core.resources import AssetDatabase

with AssetDatabase(Path("MyGame/Assets"), event_bus=bus) as db:   # Library — рядом с Assets
    db.scan()                          # GUID + <asset>.meta для каждого известного файла
    db.reimport_dirty(wait=True)       # импорт в JobSystem, применение — в главном потоке
    handle = db.resolve(guid)          # dict-lookup O(1), годится на каждый кадр
    path = handle.artifact_path("mesh")  # Library/<xx>/<guid>/mesh.npz
```

* `<asset>.meta` (коммитится) хранит `guid`, `importer`, `version` (последний импорт) и
  `settings`; `Library/` — кэш, в git не попадает. Копия ассета вместе с `.meta` получает
  новый GUID.
* «Грязный» ассет (`is_dirty`): нет артефактов, выросла `version` импортёра, изменились
  настройки или содержимое исходника (быстрая проверка по размеру+mtime, затем SHA-256).
  Повторный запуск на чистой базе ничего не пишет.
* Поток исполнения: импортёр работает в рабочем потоке `JobSystem` и пишет во временный
  каталог `Library/<xx>/<guid>.tmp`; в главном потоке (`pump`/`wait_idle`) каталог
  подменяется, обновляются граф зависимостей и `.meta`, публикуется `AssetImported(guid)`
  (при ошибке — `AssetImportFailed`, прежние артефакты сохраняются).
* Импортёры: `ImporterRegistry` (ключ — расширение), встроенные — `obj`, `image`
  (png/jpg/bmp/gif), `dds` (только BC1/BC3/BC5, проверка размера блоков), `audio`
  (wav/ogg: проверка заголовка, без декодирования → `audio_clip_bin`), `gltf` (.gltf/.glb,
  extra `[gltf]` = `pygltflib`; без него импорт даёт понятную ошибку, `inspect` работает). Сторонние форматы —
  entry point `myengine.plugins` на функцию `register(registry)`.
* Метод импортёра называется `import_asset` (`import` — ключевое слово Python).
* Зависимости: `ctx.add_dependency(path)` → `db.dependencies/dependents/dependency_closure`.
* **Mesh cooker** (`resources/cookers/`): `cook_mesh` → `mesh.bin` (формат — докстринг
  `mesh_format.py`, секции выровнены по 16 байт, читается `read_cooked_mesh` без копирования).
  LOD1+ — vertex clustering под долю треугольников из `lod_ratios` (по умолчанию 0.5, 0.25;
  швы UV на LOD1+ не сохраняются), вершины переупорядочиваются по первому использованию,
  позиции/нормали квантуются в int16 (snorm относительно центра и полуразмеров bbox), UV — float32.
  Настройки `generate_lods`/`lod_ratios` задаются в `.meta` любого меш-импортёра.
* **glTF**: узлы сцены запекаются в вершины (матрицы, TRS, зеркальные узлы переворачивают обход),
  каждый (узел, примитив) → `mesh_NNN.bin` с описанием материала; файл `baseColorTexture`
  становится зависимостью ассета. Не поддержаны: sparse, скиннинг, морфы, анимации.
* **Texture cooker**: `cook_texture` → `texture.bin` (MYTX v1: заголовок 32 байта, таблица мипов,
  уровни выровнены по 16, `read_cooked_texture` без копирования). Форматы `r8`/`rg8`/`rgba8`
  (RGB дополняется альфой 255) и `bc1`/`bc3`/`bc5`. Мипы — box-фильтр 2x2, у `srgb`-текстур цвет
  усредняется в линейном пространстве; `normal_map` перенормирует векторы и отключает sRGB;
  `max_size` ограничивает большую сторону. Настройки `.meta`: `srgb`, `generate_mips`,
  `max_size`, `compression` (`none` по умолчанию | `auto` | `bc1|bc3|bc5`), `normal_map`.
  Пиксели PNG декодирует собственный `png_codec` (zlib + numpy, без чересстрочных);
  JPEG/BMP/GIF — через Pillow (основная зависимость; без него исходник копируется как есть,
  `cooked=False`). BC-сжатие — внешний `texconv` из `PATH` (`TexconvCompressor`); не найден или
  упал → текстура без сжатия + предупреждение в лог. DDS по-прежнему копируется как есть.
* **PBR-материалы** (`.mat.json`, `MaterialImporter`): скаляры (`base_color`, `metallic`,
  `roughness`, `emissive`, `normal_scale`, `occlusion_strength`, `alpha_mode`/`alpha_cutoff`,
  `double_sided`, `uv_scale`/`uv_offset`) и слоты `textures` (`base_color`≡`albedo`, `normal`,
  `emissive` — ссылки на текстурные ассеты; `ao`, `roughness`, `metallic` или готовая
  `metallic_roughness` — упаковываются `pack_orm` в одну текстуру `material_orm.bin`:
  R = occlusion, G = roughness, B = metallic, линейная, с мипами; карты разных размеров
  приводятся к крупнейшей). Неизвестные ключи — ошибка (опечатки). Артефакты: `material`
  (`material.json`) и `orm_texture`. `load_material(db, guid)` / `load_material_artifact`
  → `ResolvedMaterial` (параметры, GUID текстур, путь ORM). Примеры — `assets/default_materials/`.
  Источники упаковки регистрируются как *запечённые* зависимости (`ctx.add_baked_input`): их
  изменение делает ассет грязным (запись `baked` в `artifacts.json`), а простые ссылки — нет.
  `srgb`/`normal_map` для albedo/normal задаются в `.meta` самих текстур.
  Реестр импортёров понимает составные расширения (`.mat.json`).
* **glTF-материалы** (`GltfImporter` v2, `extract_materials`): каждый используемый материал →
  `material_NNN.json` (+ `material_NNN_orm.bin`); меш ссылается на него через
  `info["material_artifact"]`. Сбой в материале (нет файла, битая текстура) не валит импорт модели.
  Встроенные (data-URI/bufferView) текстуры не извлекаются.
* **Shader cooker** (`ShaderImporter`: `.vert|.frag|.geom|.comp|.glsl`, `cook_shader`):
  разворачивает `#include`/`#pragma once` (`resolve_includes`), проверяет баланс
  `#if/#endif`, находит **ключевые слова** вариантов (`scan_features`: имена из
  `#ifdef/#ifndef/#if/#elif`, кроме констант, заданных `#define` на верхнем уровне и
  `GL_*`/`__*`) и их значения по умолчанию (`#ifndef X / #define X v`). Артефакт `shader`
  (`shader.glsl`) с `info`: `stage`, `glsl_version`, `features`, `defaults`, `includes`,
  `library` (нет `main` — файл-библиотека), `digest`, `variants`. Все включённые файлы —
  *запечённые* зависимости: правка `common.glsl` переимпортирует шейдеры, которые его
  включают. Настройки `.meta`: `include_dirs`, `strip_comments`, `variants`
  (`[{"name": ..., "defines": {...}}]`; ключевое слово, которого в шейдере нет, — ошибка
  импорта, защита от опечаток). `variant_key(defines)` — стабильный ключ набора. Бинарники
  программ cooker не строит (он без GL): их кэширует `engine/rendering/program_cache.py`.
  `load_shader(db, guid)` → `ShaderAsset`, готовый к `ShaderProgram.from_assets`.
* **FBX** (`FbxImporter`, `.fbx` бинарный и ASCII, extra `[fbx]` = `ufbx`): работает как glTF —
  обход узлов, мировые матрицы запекаются в вершины, на каждую пару (узел, материал меша)
  один артефакт `mesh_NNN.bin` + `material_NNN.json` (PBR-слоты `ufbx`, текстуры ищутся
  только внутри каталога модели: по относительному пути, имени файла, `textures/`,
  `<каталог>.fbm/`). `ufbx` сам приводит оси к правым Y-вверх и единицы к метрам (файл в
  сантиметрах даст куб 1 см). Многоугольники триангулируются веером (`triangulate_faces`),
  углы свариваются по индексам позиции/нормали/UV (`weld_corners`). Настройки как у glTF:
  `scale`, `generate_normals`, `generate_lods`, `lod_ratios`, `extract_materials`. Без
  `ufbx` импортёр зарегистрирован, а импорт даёт понятную ошибку про extra `[fbx]`.
  Не поддержано: скиннинг, морфы, анимации, NURBS (меш берётся в позе привязки).
  **Осторожно с `ufbx` 0.0.5:** временная обёртка элемента (`mesh.materials[i]`), которую
  освободили, а затем читали тот же элемент через другую (`scene.materials[i].pbr`), роняет
  процесс (access violation). Поэтому импортёр держит все обёртки (`hold`) до конца разбора
  и сам вызывает `gc.collect()` до освобождения сцены; при правках не цепляйте `ufbx`-объекты
  цепочкой через временные списки.
* **Спрайт-атласы** (`SpriteAtlasImporter`, файл `*.atlas.json`): режим `grid` (сетка
  `cell`, `margin`, `spacing`, пустые ячейки пропускаются), `alpha` (связные области
  непрозрачных пикселей, 8-связность, `threshold`, `min_size`) и `pack` (набор `sources`
  или каталог `folder` PNG упаковывается «полками» с экструзией зазора `padding`, опции
  `trim`, `max_size`, `power_of_two`). Артефакты: `texture` (`texture.bin`, мипы по
  умолчанию выключены) и `sprite_atlas` (`atlas.json`: `size` и кадры
  `{x, y, w, h, uv, pivot, source_size, offset}`; пиксели от верхнего левого угла, `v` растёт
  вниз; имена кадров дублируются в `info["names"]`). Исходные картинки — *запечённые*
  зависимости. `load_sprite_atlas(db, guid)` → `SpriteAtlas` (`frame/uv/texture()`).
  Пути в описании только внутри каталога файла. Офлайн-вариант: `tools/pack_atlas.py`.
* **Менеджер импортов** (`ImportManager`, `import_manager.py`) надстроен над `AssetDatabase`
  и живёт в главном потоке. `open()` сканирует проект и ставит грязные ассеты в очередь;
  `poll()` (по таймеру, не чаще `poll_interval`) находит новые, изменённые, удалённые и
  переехавшие (по GUID из `.meta`) файлы — сравнением `stat`, без `watchdog`; упавший
  импорт не повторяется, пока не изменится исходник. Очередь ограничена `max_parallel`,
  поддерживает приоритет (`reimport(..., priority=True)`) и отмену ожидающих; состояния
  `ImportState` (`idle/queued/running/ok/failed`), `progress()`, `failures()`. События:
  `ImportBatchStarted/Progress/Finished`, `AssetRemoved`, `AssetMoved` (плюс
  `AssetImported/Failed` базы). `set_settings(guid, ...)` проверяет значения по схеме
  импортёра (`settings_schema()` — своя у импортёра или выведенная из `default_settings()`;
  `SettingSpec`, `validate_settings`) и пишет в `.meta` только отличия от умолчаний.
  `orphans()`/`gc_library()` убирают в `Library` лишь каталоги `<xx>/<guid>[.tmp]` без
  ассета (чужое, например кэш шейдеров, не трогают); `clear_library()`/`rebuild_library()`.
  В `AssetDatabase` добавлены `iter_source_files`, `unregister` (файлы остаются),
  `move`, `invalidate`.
* **Ссылки на ассеты по GUID.** `MeshRenderer.mesh_guid` (необязателен; в словарь пишется
  только непустой) позволяет сцене пережить переименование файла; путь `mesh` остаётся
  запасным. Разрешение GUID↔путь делает `editor/scene_io.py`. `SpriteRenderer` — данные
  спрайта (`sprite`, `atlas_guid`, `color`, `flip_x/y`, `visible`); положение берётся из
  `Transform`, рисует `SpriteBatch` из `engine/rendering/`.
* **Пакет ассетов** (`pak.py`): `PakBuilder(db).add(guid)/add_all().build(path, compress=)`
  складывает артефакты `Library/` в один `.pak` (заголовок `MEPK`, блобы сырые или zlib,
  JSON-индекс в конце; запись атомарна через `.tmp`, у каждого блоба CRC32). `PakReader`
  читает индекс, `read(guid, name)` проверяет сумму, а `resolve/get` распаковывает
  артефакты в кэш `<xx>/<guid>/` и возвращает `AssetHandle`, поэтому `load_sprite_atlas`,
  `load_pbr_material` и др. принимают `PakReader` вместо `AssetDatabase`.
* **FBX в подпроцессе** (`importers/fbx_worker.py`): `FbxImporter.import_asset` по
  умолчанию (`isolated = True`) запускает `python -m ...fbx_worker`, а нативный краш
  `ufbx` (или таймаут `worker_timeout`) превращается в `AssetImportError` с кодом
  завершения. Зависимости и записи лога переносятся из ответа дочернего процесса.
  `import_in_process` — прежняя логика без изоляции (её и вызывает воркер). Проверка
  `find_spec("ufbx")` остаётся в родителе, чтобы сообщение про extra `[fbx]` не терялось.
* Ещё не сделано: проверка `texconv` на реальной утилите (на этой машине её нет),
  GLSL-валидация при импорте (нет `glslangValidator`), подключение `SpriteRenderer` к
  сцене (сбор компонентов в `SpriteBatch`) и к `scene_io`, экспорт exe поверх `.pak`.

---

### 5.2 VFS: источник артефактов (A6)

`engine/core/resources/vfs.py`. Протокол `VFS`: `read(guid, kind)`, `resolve(guid, kind)`
(путь к файлу; `.pak` распаковывает в кэш), `exists`, `list_artifacts`, `list_guids`.
Реализации: `LibraryVFS(Library/)`, `PakVFS(.pak)`, `OverlayVFS(base, patch)` (патч важнее
базы), `WorkdirVFS(Assets/, Library/)` (+ исходные файлы и `.meta`). Все `load_*`
(`load_material`, `load_sprite_atlas`, `read_cooked_mesh`...) принимают `AssetDatabase`,
`PakReader` или любой `VFS`; `as_assets(source, root)` даёт общий адаптер
(`resolve/get/guid_of/guids/root`).

`detect_vfs(project, pak=, use_library=)` выбирает источник для плеера: явный `--pak` →
`Build/<имя>.pak` → `Library/` (`WorkdirVFS`, если есть `Assets/`). Нет ни пака, ни
`Library/` → `NoBuildError`; плеер пишет громкое предупреждение и остаётся на `WorkdirVFS`
(в сборке игра не запустится, для разработки это норма). `--use-library` игнорирует пак.
Покадровое совпадение `--dump-state` через Library и через `.pak` проверяется на
верификации A6 (`_work/a6_vfs_report.md`).

### 5.3 AssetRef: ссылка на ассет (A6)

`engine/core/resources/asset_ref.py`: `AssetRef(guid, path)` (frozen, пустые строки →
`None`, `\` → `/`, без guid и path — `ValueError`). Пишутся оба поля
`{"guid", "path"}`; `resolve(vfs)` ищет сначала по GUID, затем по пути (через базу, затем по
`.meta` рядом с файлом), для битой ссылки и примитива возвращает `None`. Примитив —
`AssetRef(guid=None, path="cube")` (`is_primitive`). `AssetRef.coerce` принимает
`AssetRef`, строку-путь, `{"guid","path"}` и старое `{"asset": путь}`.

Где используется: `MeshRenderer.mesh_ref`, `SpriteRenderer.sprite_ref` (ссылка на атлас;
`sprite` остаётся именем кадра), `PrefabInstance.prefab_ref`, поля скриптов
(`Asset`/`Prefab` — это `AssetRef`; в скрипте значение — `AssetRef`, путь в `.path`).
Старые имена `mesh`, `mesh_guid`, `atlas_guid`, `prefab_guid`, `prefab_path` — read-only
property; присваивать нужно `*_ref`. Конструкторы по-прежнему принимают `mesh=`,
`mesh_guid=`, `atlas_guid=`.

Формат сцены остаётся v2: читаются обе формы (`mesh`/`mesh_guid`, `atlas_guid`,
`prefab_guid`/`prefab_path`, `{"asset"}`), пишется только новая (`mesh_ref`, `sprite_ref`,
`prefab_ref`, `{"guid","path"}`). Файл не переписывается при загрузке. Путь пишется
относительно `Assets/`; при загрузке главнее GUID (путь обновляется после переименования),
при сохранении — путь (GUID берётся из базы). Нерезолвимая ссылка не стирается: инспектор
подсвечивает её красным. Миграции v1→v2→v3 нет.

## 6. Игровой цикл

```python
from engine.core import GameLoop

loop = GameLoop(scene, fixed_delta=1 / 60, max_substeps=5, target_fps=60)
loop.add_fixed_update(on_physics)          # после сцены
loop.add_render(lambda dt, alpha: draw(alpha))
loop.run(max_frames=600)                   # или while: loop.step(dt)
```

* Фиксированный шаг выполняется столько раз, сколько нужно (обычно 0–5 за кадр); остаток
  времени отдаётся в `Time.alpha` для интерполяции картинки.
* `max_substeps` защищает от «спирали смерти»: лишнее время отбрасывается, а не накапливается.
* `pause()` останавливает симуляцию, продолжая рендер; `time_scale` включает слоу-мо.
* `step(dt)` детерминирован при явном `dt` — так тестируются игровые механики без часов.
* Статистика: `loop.stats.fps`, `last_frame_ms`, `dropped_time`, `fixed_steps`.

---

## 7. Производительность (замеры `tools/benchmark_core.py`)

Сцена: 5 000 объектов с компонентами, 60 кадров, Windows/Python 3.13, numpy 2.5.

| Что измерялось | Время | На единицу |
|---|---|---|
| Создание 5 000 объектов с компонентами | 155 мс | 31 мкс на объект |
| 60 кадров `scene.update()` | 127 мс | **1.70 мс на кадр** |
| Пересчёт иерархии трансформов | 23 мс | 384 мкс на кадр |
| `scene.query(...)` по кэшу | 4.5 мс | **22.7 мкс на запрос** |
| 100 000 событий через шину | 208 мс | 2.1 мкс на событие |
| Уничтожение 1 000 объектов | 17 мс | 17 мкс на объект |

Что дало ускорение (первый замер до оптимизации — в скобках):

* корни сцены и активные компоненты — словари, а не списки: удаление O(1) вместо O(n)
  (уничтожение 1 000 объектов: 470 → 17 мс);
* кэш запросов возвращает неизменяемый кортеж без копии (237 → 22.7 мкс);
* пакетный пересчёт матриц по уровням + пересчёт только грязных поддеревьев
  (иерархия: 768 → 384 мкс на кадр).

Практический вывод: Python-логика в компонентах стоит примерно 0.3 мкс на компонент в кадре —
при бюджете 6 мс на CPU это ~10–15 тысяч активных компонентов. Всё, что дороже, выносится в
системы с numpy-векторизацией или в нативные модули (см. §7 архитектуры).

---

## 7.1 Физика (`engine.physics`)

Чистый numpy, без GL/Qt. Основные модули: `shape` (Box/Sphere/Capsule: aabb,
объём, опорная функция, тензор инерции), `broadphase` (SAP по X),
`narrowphase` (`ContactManifold`: точки, нормаль A→B, глубина; скалярный
эталон всех пар), `narrowphase_batch` (`box_box_batch` — куб–куб пачкой),
`material` (`PhysicsMaterial`, combine), `store` (SoA-хранилища `BodyStore`/
`ColliderStore`), `body`/`collider` (фасады над строками хранилищ), `solver`
(sequential impulse; `solve_island` — скалярный эталон), `solver_batch`
(`solve_batch`, `ContactBatch`, `WarmStart` — тот же солвер на массивах),
`island`, `sleeping`, `events`, `raycast`, `world` (`PhysicsWorld`),
`character` (`CharacterController`). Кватернионы — xyzw.

Связь со сценой — `engine/core/physics_components.py` (компоненты с
`to_dict/from_dict`) и `engine/core/physics_adapter.py`
(`ScenePhysicsAdapter`: структура, перенос поз туда и обратно, шаг, события).
`Scene.physics` — ленивый `PhysicsWorld`; `Scene.physics_if_exists` не создаёт его.
Размер коллайдера умножается на мировой масштаб объекта. События физики
(`TriggerEnter/Stay/Exit`, `CollisionEnter/Exit`) доставляются через
`scene.events.emit_deferred`. Скриптам — `self.physics`
(`engine/scripting/physics_api.py`).
Замеры чистой физики (без рендера, медиана мс/шаг, HD 520 / i3-6006U, сцены
`_work/phys_bench.py`; до → после SoA): 100 ящиков 60.2 → 14.5; 500 ящиков 404 → 48.4;
стопка 20 ящиков 29.2 → 10.4; 100 шаров 27.8 → 12.8; 100 капсул 78.0 → 51.4.
Демо-сцена в плеере (`--headless`): CPU стоя ≈ 14.3 мс/кадр (прежнее дерево в тех же
условиях — 15.6 мс).

`max_substeps=3` (плеер, `PLAYER_MAX_SUBSTEPS`): при step > 16.7 мс мир замедляется, но кадр не раздувается до 5×. Осознанный выбор в пользу отзывчивости.

**Бюджет времени (`GameLoop(max_fixed_ms=…)`).** Кадр = k·S + R (k шагов физики по S мс + рендер R): без обратной связи кадр, удлинённый тяжёлым шагом, даёт больше `delta`, больше подшагов и ещё более длинный кадр (плюс квантование vsync 16.7 → 33 мс). `max_fixed_ms` режет число шагов по времени: первый шаг кадра выполняется всегда, следующий — только если `потрачено + оценка шага ≤ бюджет` (оценка = максимум из последней стоимости и сглаженной: быстро растёт, медленно спадает). При остановке по бюджету остаток отбрасывается с сохранением фазы (`accumulator %= fixed_delta`), счётчик — `stats.budget_stops`. Плеер в оконном режиме передаёт `PLAYER_FIXED_BUDGET_MS = 9.0`; в `--headless`/`--frames` бюджет выключен (результат не зависит от скорости машины). При срабатывании игровое время идёт медленнее реального — скрипты должны считать время по `self.time.fixed_frame * self.time.fixed_delta`, а не по `elapsed`. Предупреждение «Кадр не успевает» выводится сразу, затем не чаще раза в секунду со сводкой. Остановка по бюджету тоже логируется («Физика не укладывается в бюджет … игровое время замедлено», не чаще раза в секунду) — раньше замедление было молчаливым.

**Стенд frame pacing** (`tools/benchmark_frame_pacing.py`, сцены `tools/_frame_pacing_scenes.py`): плеер `--headless --closed-loop --present-hz 60 --frame-trace PATH` — реальное dt предыдущего кадра, бюджет включён, эмуляция vsync; метрики `sim_rate` (игровое время / реальное, в т.ч. худшее окно 1 с), поздние кадры, шаг p50/p95, остановки по бюджету. Сцены «щель» между двумя боксами/капсулами (`gap_*`). В реальном окне с vsync не проверялось.

**Стартовое оседание** (`ScenePhysicsAdapter.settle`, плеер `_settle_physics`, `PLAYER_SETTLE_MAX_STEPS=240`): перед первым кадром (в `_warm_up`) мир шагает без скриптов/персонажей/событий, пока все динамические тела не заснут. Иначе бодрствующие пропсы 1–2 с держат шаг на 17–25 мс. Не вызывается только в детерминированном режиме `--frames N` без `--closed-loop`; `--headless` теперь осаживает тела (A14a, B6). `sync_out` пишет позу заснувшего тела один раз после засыпания (`_sleep_synced`; сбрасывается, когда тело снова бодрствует или пересобирается раскладка), иначе тело, заснувшее на шаге, оставалось в трансформе на позе предыдущего шага («висело в воздухе»). **Пакетные хуки**: `Scene.add_fixed_batch(key, func, item)` + скриптовый `on_fixed_update_batch` (см. `docs/SCRIPTING.md`); `realistic`: `sim_rate` 0.85 → 0.95 при пакетном патруле (машина занята, 3 прогона).

**Свип капсулы по боксу.** `PhysicsWorld._sweep_one` для пары «капсула — бокс» ведёт марш и бисекцию по булеву предикату `narrowphase._capsule_box_overlap`, а полный `collide` вызывается один раз в найденной точке (при расхождении предиката и `collide` — прежний путь). Результат побитово равен прежнему (`test_sweep_capsule_box.py`); `_depenetrate` предфильтрует боксы тем же предикатом. Персонаж у стены: char 8.9 → 2.1 мс, `collide` на кадр 16 → 3; угол: 18 → 7.7 мс.

**SoA-хранилище (`engine/physics/store.py`).** Состояние мира лежит в плотных
numpy-массивах, а не в объектах: горячие пути работают сразу со всеми строками.

* `BodyStore`: `position (n,3)`, `rotation (n,4)` xyzw, `vel (n,6)` (линейная |
  угловая), `mass`, `inv_mass`, `inv_inertia_local/world (n,3,3)`, демпфирование,
  `gravity_scale`, `sleep_timer`, `body_type` (u1: static 0 / kinematic 1 /
  dynamic 2), флаги `sleeping`, `can_sleep`, `use_gravity`, `freeze_rotation`,
  `handle` (i8) и `row_of_handle` (handle → строка).
* `ColliderStore`: `cid`, `owner_handle`, `kind` (u1: box 0 / sphere 1 / capsule 2 /
  прочее 3), `half`, `offset`, `rot_off`, `layer`/`mask` (uint64), `is_trigger`,
  мировые `world_pos`, `world_rot`, `aabb_lo/hi` и `row_of_cid`.
* Тип чисел — `FLOAT = float64` (одно место смены на float32). Строки плотные
  `0..count`; ёмкость стартует с 64 и **удваивается**; удаление — **swap-remove**
  (последняя строка встаёт на место удалённой, хвост обнуляется). Порядок, от
  которого зависит результат, задаёт ключ, а не номер строки: тела — по `handle`
  (`BodyStore.order`), коллайдеры и пары — по `cid`. Поэтому два мира с разной
  историей add/remove и одинаковым составом дают одинаковый шаг.
* `RigidBody` и `Collider` — **фасады** над строкой: публичный API не менялся
  (`position`, `linear_velocity`, `mass`, `layer`, … читают/пишут строку). Вне мира
  тело живёт в собственном хранилище на одну строку; `add_body` переносит строку в
  хранилище мира (тот же Python-объект, меняются `_store`/`_row`), `remove_body`
  возвращает её обратно. Форма и смещение коллайдера копируются в строку при
  добавлении; `layer`/`mask`/`is_trigger` пишутся в строку сразу.
* **Ограничение: устаревший view.** `np.ndarray`, полученный из свойства
  (`body.position`, `body.linear_velocity`), указывает на строку буфера и валиден
  только до ближайшего `add_body`/`remove_body`: после роста ёмкости он смотрит в
  старый буфер (запись не дойдёт до мира), после swap-remove строка может достаться
  другому телу. Кэшировать view нельзя — свойство читается заново.

Конвейер `PhysicsWorld.step`: гравитация и демпфирование массивами → широкая фаза
(`broadphase.query_pair_arrays()`, пары отсортированы по `(cid_a, cid_b)`) →
фильтр пар массивами (владелец, слои/маски, триггеры, static–static,
спящее–неподвижное) → пробуждение (цикл по немногим парам) → узкая фаза →
события пар → `solve_batch` → `_integrate` массивами → засыпание островов.

* **Векторизовано:** гравитация/демпфирование, фильтр пар, мировые позы и AABB
  коллайдеров, **куб–куб** (`box_box_batch`: 15 осей SAT, клиппинг
  Сазерленда — Ходжмана, редукция полигона >4 точек до 4 по тем же правилам, что у
  скалярного `_reduce_points`), **солвер** (`solve_batch`), интегрирование.
* **Не векторизовано (остаётся поштучным Python):** пары шар–куб, шар–капсула,
  капсула–куб, капсула–капсула, шар–шар (`PhysicsWorld._scalar_contacts` вызывает
  скалярные функции `narrowphase`), триггеры (`_overlap`), алгоритм SAP широкой фазы
  (сама отдаёт массивы пар, но sweep — прежний), острова и засыпание
  (`build_islands`, `update_island_sleep`), события пар, смешивание материалов пары,
  кинематики-«двигатели» в `_integrate`, раскраска контактов (`assign_colors`).
* **Солвер.** `solve_batch` решает все острова одним вызовом. Жадная раскраска
  даёт тот же порядок Гаусса — Зейделя, что у `solve_island`; контакты
  переставляются так, чтобы цвета лежали подряд (срезы без gather/scatter);
  якобианы и отклики считаются один раз как матрицы `(m,3,12)`; в итерации остаётся
  пара `einsum` на «цвет × итерацию». Остаток стоимости — накладные расходы вызовов
  numpy (≈58 тыс. вызовов `einsum` за 250 шагов в сцене A), поэтому у стопки
  (8 цветов × 8 итераций, зависимость строго последовательная) шаг ≈ 10 мс.
  Ключ warm start — int64 `(cid_a << 33) | (cid_b << 3) | слот`.
* **Детерминизм.** Побитовая повторяемость запуск → запуск обязательна (тесты
  `test_physics_store.py`, `test_solver_batch.py`, `test_narrowphase_batch.py`).
  В `box_box_batch` только поэлементная арифметика (без BLAS): результат пары не
  зависит от состава и порядка пакета (тест). Скалярные `box_box`/`solve_island`
  остались эталоном: пакетные версии совпадают с ними до округления (контракт тестов
  — 1e-4; конечные позы бенч-сцен отличаются от прежних на 1e-14…1e-8).

Поведение тела (проверено `engine/tests/test_rigidbody_*.py`):

* Демпфирование `v ← v / (1 + d·dt)` — e-затухание за `1/d` с (d=1 → ≈0.37 за 1 с,
  ≈0.61 за 0.5 с); знак скорости не меняется. Предельная скорость падения `g/d`.
* Полунеявный Эйлер при 60 Гц: путь свободного падения за 0.5 с на ≈3 % больше
  аналитического `g·t²/2`; скорость точная.
* `freeze_rotation` обнуляет обратную инерцию, а в начале `PhysicsWorld.step`
  ещё и угловую скорость динамического тела (в том числе заданную скриптом).
  `RigidBody.mass` / `RigidBody.freeze_rotation` — свойства: запись на живом теле
  пересчитывает `inv_mass` и инерцию (при смене массы тензор масштабируется).
* Пороги сна — `PhysicsWorld.sleep_settings` (`SleepSettings`: 0.05 м/с, 0.05 рад/с,
  0.5 с); общие для всего мира, на уровне тела настраивается только `can_sleep`.
* **Неупругий контакт (`restitution=0`).** В контактах с глубиной проникновения
  ~0.07 м за кадр остаётся разделение ~18 % от относительной скорости — артефакт
  позиционной коррекции (она работает как скоростная добавка); импульс при этом
  сохраняется точно (центр масс не меняет скорость). Это штатное поведение, не баг.

Поведение коллайдера (проверено `engine/tests/test_collider_*.py`):

* **Слои и маски.** Пара взаимодействует, если выполнены оба условия: слой A
  входит в маску B **и** слой B входит в маску A (AND). Фильтр применяется до
  проверки триггера. `raycast`/`overlap_*` используют только `layer_mask`
  запроса и маски коллайдеров не учитывают. Маска — 32 бита (в инспекторе hex-поле
  и кнопки «Все»/«Никто» — и у коллайдера, и у контроллера персонажа); `layer`
  в инспекторе — тот же 32-битный `MaskEditor`.
* **Смешивание материалов.** По умолчанию трение — `multiply`, упругость — `max`.
  Режим пары: побеждает режим, стоящий позже в `multiply < min < max < average`
  (`material._pick`); «главного» коллайдера нет, результат не зависит от порядка.
  Поля компонента: `friction_combine`, `restitution_combine`; неверное имя режима
  → `ValueError` в `PhysicsMaterial`, объект пропускается с предупреждением.
* **Смещение и поворот.** `offset` — в локальных осях объекта, `rotation_offset`
  (кватернион xyzw, в инспекторе углы Эйлера) поворачивает форму относительно
  объекта; identity в сцену не пишется. Отрицательный `size` берётся по модулю.
  `null` в числовых полях JSON → значение по умолчанию.
* **Рост статики/кинематики** (изменение размера, спавн, сужение/удаление опоры)
  будит спящие тела в зоне нового/старого AABB (`PhysicsWorld.wake_overlapping`).
* **Капсула и тонкий пол.** Если ось капсулы пересекает бокс, выталкивание идёт
  по ближайшей грани (`narrowphase._capsule_inside_exit`), дрожания нет.
* **Известные ограничения.** (а) При `restitution=1` полная энергия отскока
  превышает исходную на ≈4.4 % — артефакт полунеявного Эйлера (как и путь падения
  выше), не баг контакта. (б) Шар, центр которого лежит ровно на оси капсулы, не
  выталкивается (вырожденный случай, нормаль не определена). (в) Центр тела,
  глубоко проникший в статику, выталкивается «сквозь» пол по ближайшей грани.

### Mesh-коллайдеры и BVH (A11)

* **Модель.** `MeshColliderComponent(mesh: AssetRef, is_trigger, friction,
  restitution, layer, mask)` — статический коллайдер по LOD0 cooked-меша
  (все части склеены). Мировой scale запекается в вершины, поза — только
  поворот + смещение, поэтому движение объекта не перестраивает BVH
  (`PhysicsWorld.set_mesh_pose`); смена scale — пересборка.
  **`mesh = null` — ERROR** (`ScenePhysicsAdapter._mesh_bvh`, один раз на
  объект: «не задан меш, коллизии нет»). При создании (`on_attach`) пустой
  `mesh` копируется из `MeshRenderer.mesh_ref` объекта (только ссылка с GUID;
  копия, не связь — позднее изменение рендерера коллайдер не трогает);
  `inheritable_mesh()` даёт ту же ссылку инспектору: при пустом `mesh`
  показывается кнопка «Унаследовать от MeshRenderer» (A14a, B2).
* **BVH** (`engine/physics/bvh.py`): median-split по самой длинной оси, лист ≤ 8
  треугольников (лист 4 даёт 12 уровней с листьями по 2–3 tri; на shapecast лист
  8 не хуже), плоские массивы узлов; `raycast` (Мёллер — Трамбор, обе стороны
  грани, обход «ближний потомок первым»), `raycast_batch`, `query_aabb`,
  `closest_point_triangle`. Строится в рантайме при загрузке в мир: ≈13 мкс/
  треугольник (10k tri ≈ 131 мс), кэш по `(guid, path, scale)` — одинаковые
  объекты делят дерево. Cook `collision.bin` не делался.
* **Пакеты лучей** (`Bvh.raycast_batch`, `PhysicsWorld.raycast_batch`):
  пакеты >= 16 лучей идут numpy-обходом; меньше — циклом (numpy overhead не
  окупается). 100 лучей по 10k tri: 2.6 мс против 4.9 мс циклом.
* **Запросы.** `PhysicsWorld.add_mesh/remove_mesh/set_mesh_pose/iter_meshes`;
  mesh видят `raycast`, `overlap`, `shapecast` и `CharacterController`
  (`mesh_contacts` в `_depenetrate`, `prepare_sweep` в свипе). В широкую фазу,
  солвер и события столкновений mesh не входит, учтён в `region_fingerprint`
  (персонаж в покое по-прежнему пропускает шаг).
* **Ограничения.** Динамические тела сквозь mesh проваливаются (для динамики —
  `ColliderComponent`); mesh не может быть динамическим; box-форма персонажа/
  тела на mesh не поддержана в `contacts` (для box — только overlap по
  треугольникам); треугольники — двусторонние.
* **Адаптер** (`ScenePhysicsAdapter._sync_meshes/_sync_mesh_poses`): геометрию
  берёт `mesh_geometry.load_mesh_geometry(scene.asset_source, ref)`; источник
  ассетов `scene.asset_source` ставят `iter_scene_from_dict` и редактор. Нет
  ассета/меша — один WARNING, коллайдер пропускается.
* **Замеры** (`tools/benchmark_mesh_collider.py`, HD 520, 10k tri, A11b):
  `raycast` ≈64 мкс/луч — потолок чистого Python (было ≈187 мкс); `shapecast`
  капсулы 0.5 м ≈0.8 мс (с самим персонажем в мире) / ≈1.1 мс (без него, полный
  свип); `character.move` по холмистому мешу p50 ≈3.0 / p95 ≈6.7 мс (цели 0.2 мс
  и 0.8 мс не достигнуты: `query_aabb` и `capsule_triangle` на Python). Реальный
  уровень `mesh_level` (92 tri) — player CPU ≈4.5 мс/кадр. Точность луча по
  рампе: box по габариту ошибается на 6 см, mesh — 0; 1000 лучей до/после A11b
  совпадают (0.0 ошибка расстояния, те же треугольники).
* **Не сделано (A11c):** cook BVH в `Library/`; shapecast/`character.move` быстрее
  — только через векторный или нативный обход (в чистом Python упёрлись).

## 7.2 Character controller (`engine.physics.character`)

Капсульный kinematic-контроллер: собственный collide-and-slide по
`PhysicsWorld.shapecast` (не участвует в солвере), затем перенос кинематического
двойника в мире. Компонент — `CharacterControllerComponent` (16 полей, см. таблицу
ниже), скриптам — `self.physics.is_grounded() / move_character(motion) /
simple_move(velocity=None) / jump(speed=None) / sprint(on) / set_air_control(value)`
(подробнее — `docs/SCRIPTING.md`). Эталонный скрипт —
`projects/PlayModeTest/Assets/Scripts/player_controller.py`.
Поведение проверено `engine/tests/test_character_*.py`.

**Параметры и валидация**

Геометрия и слои (меняются только пересозданием двойника):

| Параметр | По умолч. | Поведение |
|---|---|---|
| `radius` | 0.4 | `> 0`, иначе `ValueError`. В адаптере — `ERROR`, объект не регистрируется |
| `height` | 1.8 | строго `> 2*radius`, иначе `ValueError("height must be > 2*radius (got height=..., radius=...)")`. В адаптере — `ERROR`, не регистрируется |
| `step_offset` | 0.3 | `>= 0`; предельная высота ступени равна `step_offset` и не зависит от скорости ходьбы (шаг замера 0.05 м) |
| `slope_limit` | 45 | градусы; адаптер зажимает в `[0, 90]` с предупреждением (сам контроллер не проверяет) |
| `skin_width` | 0.02 | `> 0`, иначе `ValueError`; адаптер при `<= 0` подставляет `0.001` с предупреждением. Зазор до стены и пола равен `skin_width`, центр в покое — `height/2 + skin_width` |
| `layer`/`mask` | 1 / все | слой двойника и маска столкновений (маска 32 бита) |

Движение, гравитация и земля («живые» поля: правка на работающем компоненте
**не пересоздаёт** двойник и не сбрасывает вертикальную скорость; адаптер
переносит значения в контроллер при следующем шаге):

| Параметр | По умолч. | Допустимо | Назначение |
|---|---|---|---|
| `move_speed` | 4.5 | `>= 0` | скорость ходьбы для `simple_move()` без аргумента, м/с |
| `sprint_multiplier` | 1.6 | `>= 1` | множитель скорости после `sprint(True)` |
| `jump_velocity` | 5.0 | `> 0` | скорость для `jump()` без аргумента, м/с |
| `use_gravity` | `True` | bool | `False` — вертикальная скорость всегда 0, прыжок невозможен |
| `gravity_scale` | 1.0 | `>= 0` | множитель гравитации мира (`world.gravity`) |
| `ground_check_distance` | 0.03 | `> 0` | дальность пробы земли сверх `skin_width` |
| `air_control` | 0.3 | `[0, 1]` | доля управляемости в воздухе для `simple_move()` без аргумента |
| `snap_to_ground` | `True` | bool | прилипание к земле при спуске по склону и ступеням |
| `alignment_to_ground` | `False` | bool | наклон трансформа по нормали земли (см. ниже) |

Конструктор компонента и контроллера строг: значение вне диапазона — `ValueError`
с именем поля (`move_speed=-1.0: допустимо значение >= 0`). Значения из сцены
(`from_dict`) и присвоенные полям позже **приводятся к диапазону** (ближайшая
граница, для открытого минимума — `1e-3`, `nan`/`inf` — значение по умолчанию)
с `WARNING` в лог: битая сцена не обрывает загрузку. Отсутствующие в старой сцене
поля берут значения по умолчанию, так что прежние сцены (7 полей) читаются без
изменений. `to_dict` пишет все 16 полей.

`ERROR` пишется один раз на объект (имя объекта и причина); пока параметры
невалидны, `move_character`/`simple_move`/`jump` бросают
`RuntimeError("компонент не зарегистрирован: <причина>")`; после исправления
параметров объект регистрируется заново.

**Гравитация без скрипта (пассивный шаг).** Персонаж сам применяет гравитацию,
если скрипт его в этом физическом шаге не двигал: `ScenePhysicsAdapter.step`
вызывает `CharacterController.step(dt)` для каждого контроллера, у которого не
было вызова `move`/`simple_move`/`move_input` после предыдущего шага (флаг
`binding.moved`). Скрипт, который зовёт `simple_move` каждый кадр, гравитацию уже
получил — второй раз она не применяется. Объект, созданный в воздухе, падает на
землю без единой строчки скрипта (раньше он зависал: «объект поднят над землёй»,
см. `_work/character_elevation_bug.md`). Отключается `use_gravity = False`.
Покоящийся персонаж трансформ не переписывает: пассивный шаг пишет позу, только
если позиция сместилась больше чем на `1e-7` (или нужно обновить наклон).

**Быстрый путь покоя.** Полный шаг стоящего на земле персонажа (`shapecast` вниз,
проба земли, `move_kinematic`) возвращает его в ту же точку, а в сцене таких
персонажей обычно большинство, поэтому `CharacterController.step` его пропускает.
Шаг — чистая функция входов: позы, скоростей, параметров контроллера, гравитации
мира, `dt` и коллайдеров вблизи капсулы. Пока после полного шага состояние
побитово то же (`vertical_velocity == 0`, позиция не сдвинулась) и отпечаток этих
входов не изменился, повтор даёт тот же результат, и его не считают; возвращается
нулевое смещение. Отпечаток окрестности — `PhysicsWorld.region_fingerprint(lo, hi,
ignore)`: коллайдеры, чьи AABB пересекают капсулу, расширенную на
`step_offset + skin_width + ground_check_distance` с запасом; он включает `cid`,
вид, размеры, слой, маску, признак триггера, позу и вращение. Следствия:

* результат побитово совпадает с полным шагом каждый кадр (дифференциальные тесты
  `engine/tests/test_character_rest.py`: убрали пол, поднялась платформа, появилась
  стена, упала коробка, перевернули гравитацию, включили `use_gravity`, телепорт);
* движение вдали (кинематический лифт, падающие ящики за пределами капсулы) кэш не
  сбивает, а движение рядом пересчитывает персонажа сразу;
* явные `move`/`simple_move`/`move_input` быстрого пути не имеют, считаются всегда;
* если вблизи есть коллайдер формы вне `Box/Sphere/Capsule` (`KIND_OTHER`),
  отпечаток не строится (`None`) и персонаж считается каждый шаг;
* пока `vertical_velocity != 0` (падение, прыжок) шаг считается полностью; парение
  с `use_gravity = False` пропускается так же, как стояние.

Стоимость остаётся ~100 мкс на стоящего персонажа за кадр (отпечаток и разбор
состояния), против ~1.5 мс/кадр без быстрого пути (замеры —
`_work/character_improvements_report.md`).

**Проба земли и первый кадр.** Проба — `skin_width + ground_check_distance`
(по умолчанию `0.05`). При создании двойника адаптер вызывает
`CharacterController.refresh_ground()` **после** привязки всех объектов сцены:
иначе `grounded` считался бы в неполном мире (пол регистрируется позже
персонажа) и первый кадр ложно давал `False`.

**Выравнивание по земле.** При `alignment_to_ground = True` трансформ наклоняется
так, что его «вверх» совпадает с нормалью земли под ногами; рыскание (поворот
вокруг вертикали) сохраняется. Физическая капсула **остаётся вертикальной** —
наклоняется только визуал. Центр вращения — центр капсулы (трансформ объекта). В
воздухе наклон держится прежним; выключение флага возвращает объект в
вертикальное положение (с сохранением рыскания). Стоимость — один кватернион на
шаг и только у стоящих на земле персонажей с включённым флагом; при `False`
ветка не выполняется.

**Масштаб объекта.** Капсула строится по `radius`/`height` компонента, умноженным
на **мировой** масштаб объекта (с учётом родителей), раздельно, по формуле
`ColliderComponent`: `radius × max(|sx|, |sz|)`, `height × |sy|`. Масштаб берётся по
модулю (длины столбцов 3×3 мировой матрицы), отрицательный — зеркало, размер от
знака не зависит; компонента с нулевым `|s|` нет — `|s| < 1e-3` зажимается в `1e-3`
с одним `WARNING` на объект. Примеры для умолчаний (0.4 / 1.8):

| `scale` | радиус | высота |
|---|---|---|
| (1, 1, 1) | 0.4 | 1.8 |
| (2, 2, 2) | 0.8 | 3.6 |
| (2, 1, 2) | 0.8 | 1.8 |
| (0.5, 0.5, 0.5) | 0.2 | 0.9 |
| (4, 1, 4) | 1.6 | 3.202 (подъём, см. ниже; без него было бы 1.8) |
| (0, 1, 1) | 0.4 | 1.8 (`sx` зажат в `1e-3`, один `WARNING`; на размер не влияет, т. к. берётся `max(sx, sz)`) |
| (1, 0, 1) | 0.4 | 0.802 (`sy` зажат в `1e-3`, высота 0.0018 поднята до `2r + 2e-3`; два `WARNING`: масштаб и высота) |

Масштаб **не** применяется к `skin_width`, `step_offset`, `slope_limit`,
`ground_check_distance` и скоростям (`move_speed`, `jump_velocity`): это игровые
величины в метрах мира, а не размеры формы.

* **Узкая капсула.** Если после масштаба `height ≤ 2·radius` (например, `(4, 1, 4)`:
  `r = 1.6`, `h = 1.8`), капсулы не существует. Это не ошибка: высота поднимается до
  `2·radius + 2e-3` и один раз на объект пишется `WARNING` (объект при этом
  регистрируется и двигается). Ошибка `ValueError`/`ERROR` остаётся только для
  собственных значений полей компонента (`height ≤ 2·radius` без учёта масштаба).
* **Ноги остаются на месте.** При смене размера (поле компонента, масштаб объекта или
  родителя) нижняя точка капсулы не двигается, сдвигается центр на `Δheight/2`:
  персонаж не проваливается в пол и не подскакивает при росте/уменьшении. Новый
  центр записывается в трансформ **только позицией** (вращение не трогается, поэтому
  зеркальные масштабы вращение не искажают).
* **Пересоздание.** Двойник пересоздаётся, когда пара `(radius, height)` после
  масштаба изменилась больше, чем на `1e-6` (для крупных размеров допуск
  относительный — `1e-6·max(1, size)`), либо изменились `skin_width`/`step_offset`/
  `slope_limit`. Изменения меньше допуска (дрожание `float32` мировой матрицы) не
  пересчитываются. При пересоздании переносятся `vertical_velocity` и
  `horizontal_velocity`: прыжок в воздухе не прерывается сменой масштаба.
* **Стоимость.** Чтение масштаба — один доступ к `world_matrix` на персонажа за
  шаг, который уже был нужен для позы; сравнение 3×3 — по байтам. В покое прирост
  в пределах шума (≈0), при вращении объекта каждый кадр ≈ +4 мкс/персонаж
  (пересчёт трёх длин столбцов, чистый Python).
* **Анимация масштаба.** Масштаб, меняющийся каждый кадр, пересоздаёт двойника
  каждый кадр (капсулу нельзя масштабировать на месте); для «дыхания» или приседания
  меняйте `height` целиком по условию, а не плавно.
* **Не относится к `ColliderComponent`.** Обычные коллайдеры читают мировой масштаб
  только при создании (`_bind`): смена масштаба родителя на лету AABB коллайдера не
  обновляет. Это отдельный известный техдолг (`CLAUDE.md`, «Status»), персонажей он
  не касается.

**CharacterController и RigidBody несовместимы.** Оба компонента управляют
движением объекта: контроллер переносит kinematic-двойника сам (collide-and-slide,
гравитация, прыжок), тело ведёт решатель по силам и контактам. На одном объекте
они дали бы два источника позы за шаг и непредсказуемый результат, поэтому
объект с **активными** `CharacterControllerComponent` и `RigidBodyComponent`
в `PhysicsWorld` не попадает (проверка — в `ScenePhysicsAdapter.sync_structure`,
раз при изменении состава сцены, в кадре ничего не стоит):

* в лог один раз уходит `ERROR` с именем объекта и причиной
  («CharacterController и RigidBody несовместимы»); если конфликт снят и вернулся,
  `ERROR` пишется снова;
* ни двойник, ни тело, ни `ColliderComponent` такого объекта в мир не добавляются,
  трансформ не меняется (объект остаётся там, где стоял, и не падает);
* `move_character` / `simple_move` / `move_input` / `jump` / `is_grounded` и
  `apply_impulse` / `set_velocity` / `get_velocity` бросают
  `RuntimeError("CharacterController и RigidBody несовместимы")` — вместо прежнего
  «У объекта … нет физического тела».

Конфликт снимается удалением любого из двух компонентов или выключением одного
(`enabled=False`): неактивный компонент конфликтом не считается. Какой из двух
остаётся в мире — таблица «Кто управляет объектом» ниже. Выключенный объект
(`active_self=False`) неактивен целиком, в мире его нет вовсе, конфликта тоже.

Было «тихое игнорирование RigidBody»: контроллер выигрывал, тело и коллайдер молча
отбрасывались, а `apply_impulse` жаловался на «нет физического тела». Стало «`ERROR` +
объект не зарегистрирован». Побочное следствие: объект в конфликте инертен, а не
«ведётся контроллером» — исправлять нужно сцену (разбор —
`_work/character_rigidbody_conflict.md`). Редактор подсвечивает конфликт красной
подписью в секциях контроллера и тела (`docs/EDITOR.md`); решение «конфликт или нет»
одно на адаптер и редактор — `has_character_rigidbody_conflict` в
`engine/core/physics_components.py`.

**Три уровня: активен / заморожен / удалён.** Состав мира пересматривается в
`ScenePhysicsAdapter.sync_structure` при любом структурном изменении сцены (в том
числе при переключении `enabled`: `on_enable`/`on_disable` компонента вызывают
`mark_toggled`), отдельных обработчиков `on_remove` не нужно. У
`CharacterControllerComponent` три состояния:

| | активен (`enabled=True`) | заморожен (`enabled=False`) | удалён |
|---|---|---|---|
| Как получить | по умолчанию | `component.enabled = False` | `remove_component(...)`, уничтожение объекта, `active_self=False` |
| Двойник в `PhysicsWorld` | есть | **есть, тот же** (не пересоздаётся) | нет |
| Коллизии, `raycast`/`overlap_sphere`/`shapecast` | видят, солвер учитывает | видят, солвер учитывает (как kinematic) | не видят |
| Движение и гравитация | да | нет, поза не меняется | — |
| `simple_move`/`move_character`/`move_input`/`jump` | работают | `(0, 0, 0)` / `False`, **без исключений** | `RuntimeError` |
| `is_grounded()` | пересчитывается | последнее значение, не пересчитывается | `RuntimeError` |
| Следует за `transform` | да | да (двойник переезжает при переносе объекта) | — |

* **Заморозка** (`enabled = False`) оставляет двойника в мире: он остаётся
  препятствием (враг в выключенной зоне, персонаж-манекен), но не двигается и не
  падает. Вертикальная и горизонтальная скорости при заморозке обнуляются —
  размораживание не «доигрывает» падение с накопленной скоростью. Скрипты
  замороженного персонажа продолжают тикать и звать `self.physics.simple_move(...)`:
  вызовы безвредны и возвращают нулевой вектор. `sprint(on)` и `set_air_control(v)`
  только записывают состояние в компонент и работают всегда.
* **Разморозка** (`enabled = True`) продолжает движение **с текущей позы**
  (двойник тот же, пересоздания и сброса позы нет): персонаж, замороженный в
  воздухе, падает с места заморозки со скоростью 0. Проба земли выполняется сразу
  (`refresh_ground()`): пока двойник стоял, пол мог исчезнуть. Если объект
  переносили, пока он был заморожен, персонаж стоит на новом месте.
* **Удаление** (`remove_component(CharacterControllerComponent)` по типу или
  экземпляру, уничтожение объекта) убирает двойника: коллизии и запросы его больше
  не видят, другие персонажи проходят сквозь это место, трансформ не меняется.
  `move_character`/`simple_move` на объекте бросают
  `RuntimeError("У объекта '…' нет CharacterControllerComponent")`. Вернуть
  компонент можно только `add_component`; в редакторе удаление без undo — кнопка
  спрашивает подтверждение (`docs/EDITOR.md`).
* **`active_self = False`** у объекта (или у родителя) выключает объект целиком:
  двойника в мире нет вовсе, как при удалении (замороженный компонент на
  выключенном объекте тоже снимается). Это **не** заморозка: заморозка — свойство
  компонента, а не объекта. Поведение API у выключенного объекта прежнее —
  включённый компонент на неактивном объекте по-прежнему даёт `RuntimeError`
  (двойника нет), выключенный — no-op.

**Кто управляет объектом: CC и RB.** Правило одно: побеждает **активный**
компонент, замороженный контроллер не отбирает объект у тела.

| CharacterController | RigidBody | В мире |
|---|---|---|
| активен | активен | **конфликт**: `ERROR`, объект не в мире (см. выше) |
| активен | выключен / нет | двойник контроллера |
| выключен | активен | тело (замороженного двойника нет) |
| выключен | выключен | объекта нет в физике (если нет и активного `ColliderComponent`; с ним — статический двойник) |
| выключен | нет | **замороженный** двойник контроллера |

Почему так. Выключенный `RigidBody` рядом с активным контроллером — обычная
ситуация (тело оставили на объекте «на потом»), контроллер из-за него не ломается.
Обратное — выключенный контроллер рядом с активным телом: решает тело, иначе
«заморозка» превращала бы динамический объект в неподвижную стену, а тот, кто
включил `RigidBody`, ждёт физику. Замороженный контроллер появляется только когда
объекту больше нечем управляться, то есть тела нет вовсе; если тело есть, но
выключено, объект выходит из физики целиком (выключены оба компонента движения).
Раньше для «заморозки с сохранением коллизий» приходилось не вызывать `simple_move`
в скрипте и ставить `use_gravity=False` — теперь достаточно `enabled = False`.

**Сериализация.** `enabled=False` пишется в сцену (`scene_io`) только при `False`
и только у физических компонентов (`RigidBody`, `Collider`, `CharacterController`):
включённые компоненты остаются в прежнем формате, старые сцены читаются без
изменений. Заодно исправлена прежняя потеря `enabled=False` у `Collider` и
`RigidBody` при save/load. Заморозка переживает save/load: персонаж загружается
замороженным и остаётся на месте.

**Ступени.** `_try_step` поднимает капсулу на `step_offset`, двигает вперёд и
опускает; если нижний sweep нашёл непроходимую поверхность (ребро), шаг
повторяется с вылетом на `radius/2`. Откат происходит при пустоте под ногами,
потолке, стене и ступени выше `step_offset`; сквозь стены шаг не телепортирует.
Без `step_offset` (`0`) низкие бордюры (≈0.1 м) всё равно проезжаются скольжением
по нижней полусфере капсулы — это геометрия, а не `_try_step`.

**Прыжок.** `jump(speed=None)` (без аргумента — `jump_velocity`, по умолчанию 5.0):
`speed <= 0` — `ValueError`; только с земли, двойной и воздушный прыжок запрещены
(в воздухе вернёт `False`), как и прыжок при `use_gravity=False`. Высота прыжка —
`v²/(2g) − v·dt/2` (гравитация применяется до сдвига): при 60 Гц на `v=5` ниже
аналитики на ≈3 %, на `v=15` — на ≈12 %. Осознанно не исправляется.

**Осознанные ограничения (дизайн kinematic-контроллера, не баги)**

* Персонаж **не толкает динамические тела** (для него это стена, при любой массе).
* Персонаж **не ездит на горизонтально движущихся платформах**; вертикальное
  движение платформы он повторяет.
* `simple_move` использует `world.fixed_dt` как шаг: вызывать из `on_fixed_update`,
  а не из `on_update` (иначе скорость привязана к FPS).
* Поза пишется в трансформ сразу, **без интерполяции** — на мониторах с частотой
  выше физического шага (например, 144 Гц) заметно дрожание; отдельная задача.
* Явный `simple_move(velocity)` диагональ не нормализует: `(v, v)` даёт `v·√2`
  (нормализуйте в скрипте, как `player_controller.py`). `simple_move()` без
  аргумента (`move_input`) сжимает вектор ввода длиной больше 1 до единицы.
* **Меш привязывайте к низу капсулы.** Трансформ — центр капсулы, а в покое центр
  стоит на `height/2 + skin_width` над полом. Меш с началом координат у «подошв»
  должен быть дочерним объектом со смещением `y = -(height/2 + skin_width)`
  (для умолчаний `-0.92`); иначе модель «висит» над землёй на эту величину. Если у
  персонажа или его родителя есть масштаб, смещение считается с учётом `sy`
  (см. «Масштаб объекта» выше и `docs/EDITOR.md`).

**Измеренные особенности (game feel)**

* Проба земли = `skin_width + ground_check_distance` (по умолчанию `0.02 + 0.03`):
  зазор до пола 0.04 при `skin=0.02` ещё `grounded`, 0.05 — уже нет.
* Неровности пола 1–3 см: мерцание `grounded` ≤ 5 кадров за 2 с; лестница
  5×0.2 м на 4.5 м/с: `grounded` пропадает на ≈6 кадрах из 180.
* Склон 44° проходится, 46° — нет (упирается у подножия), 60° — персонаж
  соскальзывает; границы `slope_limit` точны (29°/31° для склона 30°).
* Тонкая стена 0.1 м при скорости 400 м/с **не туннелируется** (swept-проверка).
* Потолок над головой на `grounded` не влияет, прыжок упирается в потолок.
* **Сход с уступа.** При сходе с кромки высотой 0.2 м персонаж ≈0.13 с (≈8 кадров)
  `is_grounded = False`: `_snap_to_ground` консервативен намеренно (не прилипает на
  непроходимой нормали ребра). Известный game feel, осознанно не исправляется;
  для «времени койота» используйте таймер в скрипте.
* `jump()` без аргумента берёт `jump_velocity` одинаково у компонента, адаптера и
  `self.physics.jump`; `speed <= 0` → `ValueError`. `slope_limit` в инспекторе —
  0…90°, как у адаптера.
* `set_air_control(value)` — переопределение до `set_air_control(None)`; поле
  `air_control` не меняется и в сцену не пишется.

---

## 7.3 Reflection полей (`engine.core.reflection`)

Единый источник правды о данных компонента: `FieldDescriptor` (имя, `kind`,
умолчание, `range`, `step`, `tooltip`, `choices`, `serialize`, `live`,
`omit_if_default`, `exclusive_min`, `on_invalid`, `ctor_error`, `visible_if`).
Реестр `registry` (`FieldRegistry`) хранит дескрипторы по классам и даёт:

* `serialize(obj)` / `deserialize(cls, data)` - словарь сцены; порядок ключей =
  порядок объявления (формат `.scene.json` v1 не менялся, побайтно сверено со
  старыми `to_dict`); отсутствующее и `null` - умолчание;
* `clamp(cls, name, value, strict=, warn=)` - диапазон: нечисло -> умолчание,
  вне диапазона -> граница (открытый минимум -> `1e-3`) с предупреждением;
  `strict` (конструктор) бросает `ValueError` с именем поля;
* `enum` проверяется по `choices` только в `clamp_descriptor` (Behaviour и
  инспектор); `registry.clamp` зарегистрированных компонентов его пропускает -
  сцены читаются как есть, неизвестное значение (напр. форма коллайдера) не
  заменяется умолчанием;
* `validate_dependencies(obj)` - методы `@validator` для связанных полей.

Объявление: аннотации `имя: тип = умолчание` / `field(...)` + `@component_fields`;
для классов со `__slots__` - `__field_specs__ = (declare(...), ...)`.
`describe_class_fields(cls)` собирает дескрипторы без регистрации;
`engine.scripting.fields.descriptors_from_behaviour(cls)` - то же для класса
скрипта (заготовка под `Behaviour`, пока нигде не вызывается).

Мигрированы `RigidBodyComponent`, `ColliderComponent`,
`CharacterControllerComponent` (`to_dict`/`from_dict` - обёртки над реестром;
`_live_fields` выводится из `live`). Инспектор строит секцию по дескрипторам
(`editor/panels/auto_section.py`) для компонентов без ручной секции.

Не делается: валидации, которых не было в коде (`radius`/`height` ERROR,
`height > 2*radius`); ссылки на ассеты хранятся путём (GUID - позже); UI списков.

## 8. Ограничения текущего этапа

* Рендера ещё нет: `render/`, `platform/` — следующий этап (M1). Ядро к нему готово:
  мировые матрицы, меши и шейдеры уже загружаются в удобном для GPU виде.
* Импорт ассетов синхронный; асинхронная загрузка и стриминг — вместе с `render`/`resources`
  GUID-кэшем (M2).
* Скриптовый слой (`Behaviour`, hot reload) появится на M4 — API компонента уже совпадает
  с будущим скриптовым циклом (`on_start/on_update/on_fixed_update/on_destroy`).

---

## 7.4 Планировщик (`engine.core.scheduling`, A3a/A3b)

`Scene.scheduler` — таймеры и корутины. Задачи (`ScheduleItem`) лежат в куче `heapq` по `(wake_time, порядок вставки)`: равные моменты срабатывают в порядке добавления (детерминизм), кадр без созревших задач стоит O(1). Отмена ленивая (флаги + `_compact`).

**Часы** (`TimeCategory`):

| Категория | Кто ведёт | Пауза | `time_scale` |
|---|---|---|---|
| `GAME` | `Scene.update` → `scheduler.update_game(dt)` | стоит | да |
| `GAME_RAW` | `update_ui` (сырой dt) | стоит | нет (`Wait(unscaled=True)`) |
| `UI` | `GameLoop.step` → `scheduler.update_ui(raw)` | идёт | нет |

* **Порядок фаз:** `update_game` выполняется в `Scene.update` до систем и `on_update` компонентов; `update_ui` — в `GameLoop.step` до `_run_update`. `Scene.fixed_update` только зовёт `tick_fixed` (счётчик `fixed_frame`, будит `WaitForFixed`) и время **не двигает** (иначе оно росло бы дважды за кадр).
* **Пауза:** `GameLoop.pause/resume/attach_scene` ставят `scheduler.game_paused`; на паузе `update_game` ничего не делает (счётчик `frame` не растёт), `UI` идёт. Без `GameLoop` (тесты, редактор) `UI`/`GAME_RAW` не тикают — звать `Scene.tick_scheduler_ui(dt)` (только UI/GAME_RAW: игровые часы двигает `Scene.update`, двойной вызов удвоил бы время).
* **Таймеры:** `after(scheduler, s, cb, owner=, category=, on_error=)`, `every(..., immediate=False)` (момент k-го вызова `start + k·period`, без дрейфа, пропущенные догоняются; потолок `MAX_FIRES_PER_UPDATE`). Ошибка callback — лог с трассировкой или `on_error`; `every` продолжает, `after` завершён.
* **Корутины** (`coroutines.py`): `Wait(s, unscaled=False)`, `WaitForFrame`, `WaitForFixed(n)`, `WaitUntil(pred)` (раз в кадр, со следующего), число = `Wait`, `Wait(<=0)` = следующий кадр. Ожидающие кадр/шаг лежат в отдельных списках (`wake_frame` = номер кадра/шага; очередь задаёт `ScheduleItem.queue`). Первый шаг — при ближайшем `update`. `return x` → `Handle.result`; исключение — лог и конец; отмена (`cancel`, `cancel_all(owner)`, `clear`) закрывает генератор (`finally` выполнятся).
* `Handle` держит задачу сильной ссылкой; `cancel_all(owner)` — по индексу владельцев.
* **Три хранилища:** куча по времени (`QUEUE_HEAP`, по одной на категорию), список ждущих следующего кадра (`QUEUE_FRAME`) и список ждущих фиксированного шага (`QUEUE_FIXED`). `TimeCategory` — `IntEnum` (хеш в C: ключ словарей в горячем пути).
* **Твины** (`tweens.py`, `easing.py`): `TweenItem` каждый апдейт пишет `setattr(target, attr, lerp(start, end, ease(t)))`, финал — ровно `end`. После шага `wake_time = nextafter(now, inf)`: при `dt = 0` твин не зацикливается, для любой категории. 8 функций сглаживания в `EASES`, `get_ease(имя|callable)`. `TweenHandle.on_complete/on_update` — цепочка. Типы: число, `Vector3` (`lerp`), `Quaternion` (`slerp`), кортеж чисел. Ошибки `on_complete`/`setattr` — лог, твин завершён; `on_update` после ошибки отключается.
* **Стоимость** (`tools/benchmark_scheduler.py`, `_work/a3b_scheduler_bench.md`, мс/кадр): 1000 `after`/`every`/корутин ≈ 0.03/0.4/0.13; 100 твинов float/Vector3/`transform.position` ≈ 0.28/0.41/0.74; 1000 твинов (стресс) ≈ 3.5/4.7/8.2 (≈3.5–8 мкс на твин, без узкого места в профиле). Долг P2: плоский список для непрерывных задач, если 200+ одновременных твинов станут узким местом.

## 7.5 Ввод: действия, контексты, бэкенды (`engine.core.input`, A5)

- Слои: сырые устройства (`pressed_keys`, `mouse_buttons`, `mouse_delta`,
  `gamepad_*`) → `InputAction` (button/axis/axis2) с `Binding(device, code, scale,
  axis)` → `InputMap` (JSON, поверх `assets/default_input_map.json`).
- `InputState` — единый класс для скриптов; бэкенды пишут события
  (`key_down/key_up/mouse_button/mouse_move/gamepad_*`), `end_frame()` в конце
  кадра фиксирует «прошлое» состояние и сбрасывает `mouse_delta`.
- `InputContextStack`: приоритеты game 0, paused 50, ui 100, cutscene 200;
  `absorb` блокирует действия не из набора; `Pause` разрешён в `ui`/`paused`.
- Бэкенды: `ManualInputBackend` (core; тесты, IPC, `--hold-keys`, `feed_key`),
  `GlfwInputBackend` (`engine/player/input_backend.py`, колбэки `GlfwWindow`),
  `QtInputBackend` (`editor/play_mode_qt_input.py`, только при `enabled`; Play
  Mode сейчас — отдельный процесс плеера, поэтому потребителя в редакторе нет).
- Таблица GLFW-кодов — `engine/core/input/keys.py`. Не сделано: `scene.input_map`,
  геймпад-бэкенд (п.4).
- Ремаппинг (п.3): `InputState.begin_rebind/cancel_rebind/on_rebind`,
  `save_bindings/load_bindings`, `InputMap.add_binding/unbind/save(only_changes)`;
  события при ожидании привязки поглощаются, `Escape` отменяет. Файл
  `Settings/input.json` содержит только отличия от дефолтов (см. SCRIPTING.md).

## 7.6 Save/Load (`engine.core.save`, A7a)

Игровое сохранение отдельно от `.scene.json`: сцена — статический шаблон,
слот — дельта поверх неё плюс глобальное состояние игры. Формат сцены не
менялся.

**Модель.**
- `SaveContext` — ключ-значение с вложенными ключами `"a.b.c"`. Типы: `None`,
  `bool`, `int`, `float`, `str`, `list`, `dict` (ключи — строки без `.` и `$`),
  `tuple`, `Vector3`; остальное → `TypeError` при `set`. `get` отдаёт копию,
  `get_typed(key, type)` → `ValueError` при отсутствии/несовпадении типа
  (`int` приводится к `float`, `bool` не `int`).
- `GameState` (`scene.state`, лениво) — обёртка над `SaveContext` без схемы:
  `set/get/get_typed/has/remove`, `on_changed(key, cb)` (слышны ключ, родитель
  и потомки; возвращает отписку; `cb(key, value)`). Привязан к `Scene`.
- `ISaveable` (Protocol): `on_save(ctx)` / `on_load(ctx)`. Любой компонент с
  такими методами; `Behaviour` и функциональные скрипты получают их как хуки.
- Поля с `save=True` в `FieldDescriptor` (`@component_fields` и
  `@export(..., save=True)`) идут в сохранение автоматически, значение из
  слота перекрывает сцену. Поля без `save` берутся из сцены.

**Файл** `<имя>.save` — zip (CRC32 записей, как блобы `.pak`), записи по
порядку: `version` (целое), `meta.json` (`version`, `timestamp`, `scene_name`,
`playtime_sec`, `screenshot_path`, `scenes`), `state.json`, `objects.json`, опционально
`screenshot.png`. `objects.json`:
`{uid: {"components": {"Тип#n": {"fields": {...}, "data": {...}}}}}`; `Тип#n` —
тип и номер среди компонентов того же типа на объекте (порядок однотипных
компонентов в шаблоне сцены влияет на адресацию в save; не переставляйте
вручную). Объект находится по
`GameObject.uid`; нет в сцене → один WARNING на загрузку, пропуск. Объекты
пула (`pooled`) не сохраняются. Запись атомарна: `<имя>.save.tmp` → `fsync` →
`os.replace`; при сбое `.tmp` удаляется, старый слот цел. Zip детерминирован
(фиксированная дата записей, ключи отсортированы).

**`SaveManager(root, *, version, migrations=None)`**: `save(name, scene,
global_state, *, playtime_sec, screenshot) -> SaveSlot`, `load(name, scene)`,
`slot(name)`, `list_slots()` (новые первыми, битые пропускаются с WARNING),
`delete(name)`, `read(name)`, `migrate(slot, from, to)`. Имя слота —
`[\w\- ]+`. `root` задаёт хост: плеер — `%APPDATA%/<Game>/saves`, тесты —
`_work/saves`. Ошибки: `SaveError` → `SaveCorruptedError`, `SaveVersionError`.
Порядок загрузки: `GameState` → поля `@save` всех объектов → `on_load`
(поэтому ссылки на объекты в `on_load` уже разрешены).

**Несколько сцен (A8b).** `meta.scenes` = `[{"path", "additive"}]`, активная
первой; `scene_name` остаётся и равен `scenes[0].path`. Путь — относительно
корня проекта (`SceneManager.save_refs()`), для сцены без менеджера — её имя.
Слот без `scenes` (A7a) читается как `[{"path": scene_name, "additive": false}]`
без миграции версии. `Scene.save`/`Scene.load` при наличии `scene_manager`
охватывают все загруженные сцены (объекты всех сцен — в одном `objects.json`,
uid разных сцен не пересекаются; `GameState` — активной).
`SceneManager.restore(save_manager, slot, ...)` приводит набор сцен к слоту
(активная + аддитивные, порядок сохраняется; совпадает — ничего не
пересоздаёт), затем применяет слот; нет файла сцены → `FileNotFoundError`.
Плеер: F9/`--quick-load-at` делают это через `_restore_scene_set`.
Не сохраняются: запущенные задачи планировщика, скорости RigidBody.

**Версии.** `version` лежит в `meta.json` и в записи `version`. Старше текущей →
цепочка миграций (`engine/core/save/migrations.py`: `MIGRATIONS`,
`register_migration(from, to)`, `migrate_chain(from, to)` — кратчайший путь);
функция получает `{"meta","state","objects"}` и возвращает его; файл не
переписывается до следующего `save`. Новее текущей → `SaveVersionError`
(«создано более новой версией игры … Обновите игру»); нет цепочки → тоже.

**Сцена.** `Scene.save_manager` (ставится снаружи), `Scene.save(slot, **kw)`,
`Scene.load(slot)`; без менеджера `RuntimeError`. Не сохраняется: активные
задачи планировщика, физика (скорости), камеры/свет/меш, аддитивные сцены.
Позиции — через `@save`-поля компонента (Transform сам не компонент).

### Каталог, чекпоинты, автосейв, быстрые слоты (A7b)

**Каталог.** `default_save_root(project, *, player)` (`engine/core/save/paths.py`):
плеер — `%APPDATA%/<game_name>/saves` (Linux `~/.local/share/<game>/saves`,
macOS `~/Library/Application Support/<game>/saves`; нет `APPDATA` →
`<проект>/Saves`), редактор и Play Mode редактора — `<проект>/Saves`; тесты —
`_work/saves`. `game_name` — поле `name` из `*.project.json`. Версия схемы —
`save_version` из `*.project.json` (по умолчанию 1, `project_save_version`).
Флаги плеера: `--saves-dir PATH` (приоритетнее всего), `--use-project-saves`
(редактор передаёт его сам), `--no-autosave`.

**Служебные слоты** (`engine/core/save/service.py`): `_autosave` (один,
перезаписывается), `_quick` (F5/F9). Запись идёт через `safe_save`: нет менеджера,
уже идёт запись (`SaveManager.busy`) или `OSError`/`SaveError` → `None` и запись
в журнал, игра не падает. `state_hash` (sha256 от `GameState` + данных объектов)
хранится в `SaveManager.hashes` (в памяти, не на диске).

**Компоненты** (`engine/core/save/components.py`, сцена v2 не менялась,
инспектор — группа «Сохранения»): `CheckpointComponent(slot_name="checkpoint",
idempotent=True, player_tag="player")` — на триггере; вход игрока (тег без учёта
регистра) пишет слот и объявляет `CheckpointReached(slot_name)`; идемпотентный
не пишет повторно, пока хеш состояния тот же и слот есть на диске.
`SafeZone` — триггер «безопасной зоны». `AutoSavePolicy(on_scene_change=True,
interval_sec=0, require_safe_zone=True)` — таймер N с (только пока игрок в
`SafeZone`) и запись перед сменой активной сцены (`SceneManager.before_activate`
→ `autosave_before_switch`; нет активной сцены или идёт запись → ничего).
Триггерный автосейв — это `CheckpointComponent` со слотом `_autosave`.

**F5/F9.** `Scene.quick_save()`/`quick_load()`; F9 грузит `_quick`, а без него
`_autosave`; статус («Сохранено», «Загружено», «Нет сохранения», ошибки) — событие
`SaveStatus`, плеер показывает его в правом верхнем углу на 1 с. Если слот
сделан в другой сцене, плеер сначала загружает её (по имени из `meta`). В Play Mode
редактора — «Игра → Быстрое сохранение/загрузка» (Ctrl+F5/Ctrl+F9) через IPC
`quick_save`/`quick_load`. Отладка: `--quick-save-at N`, `--quick-load-at N`.
Не входит: сетевые сохранения.

## 7.7 Асинхронная загрузка, аддитивные сцены, переходы (A8)

**Порционная загрузка** (`engine/core/scene_load.py`). Настоящей многопоточности
создания объектов в Python нет (GIL), поэтому «асинхронно» = порциями по кадрам.
`SceneLoadOperation(path, parse, build)`: *parse* (чтение + `json.loads`) идёт в
потоке `JobSystem`, *build* — в главном потоке порциями, пока не вышел бюджет
`step(budget_ms)`, *activate* — регистрация в менеджере. Прогресс: parse 5 %,
build 90 %, activate 5 %. Билдер — генератор: `scene_io.iter_build_objects` /
`iter_scene_from_dict` отдают долю 0..1, готовая сцена в `StopIteration.value`;
`build_objects`/`scene_from_dict` — синхронные обёртки (результат тот же).
`cancel()` закрывает генератор (`GeneratorExit` выгружает недостроенную сцену).
`adaptive_budget_ms(frame_ms, load_ms)` = 0.8 × свободное время кадра, зажим
2–10 мс. Замер 1000 объектов (A8b, `_work/a8b_bench_load.py`): синхронно
≈68–112 мс; порциями шаг med 5.0 мс, p95 10.1, max 12.9, всего ≈54 мс (фон
машины 9–28 %; выбросы 20–120 мс возникали только при сильной фоновой
нагрузке); пустая сцена 0.8 мс.

**Старт плеера (A8b).** Стартовая сцена грузится через `load_async` с насосом
`_load_with_pump` (окно/GL уже созданы, прогресс виден); `--sync-load` —
старый синхронный путь до окна (тесты, отладка). Ленивый импорт
`engine.core.physics_adapter` (≈140–180 мс, не делится бюджетом) прогревается
потоком `_warm_imports` параллельно созданию окна. Прогресс уходит в шины всех
сцен (`SceneLoadProgress`), в лог (`Загрузка сцены X: build, 31%`) и в IPC
(`scene_load_progress {progress, phase}`, `scene_loaded {path}`). Итог старта
пишется в лог (`Стартовая загрузка: N мс, шагов K`), `--frame-trace` содержит
`load_steps_ms`.

**SceneManager.** `SceneManager(loader, parse=, build=, jobs=)`; `load(path,
activate=None, additive=False)` (через `load_async` с бесконечным бюджетом),
`load_async(...)`, `update_loading(budget_ms)`, `additive_scenes`, `all_scenes`
(активная первой), `loading`. Ключ сцены — путь. Событие `SceneLoadProgress`;
`SceneLoaded/Unloaded/Activated` получили `additive`. Старая активная при
`set_active` становится дополнительной.

**Аддитивные сцены** (вариант A): каждая сцена — свой `Scene` со своим
планировщиком; ссылки между сценами (uid, поля скриптов) запрещены — uid
разрешается только внутри своей сцены. Плеер вызывает `fixed_update/update/
late_update` у всех сцен менеджера; `SceneRenderer.set_extra_scenes` рисует меши,
свет, спрайты и текст всех сцен (камера — первая активная, начиная с основной),
UI каждой дополнительной сцены — отдельный `_UiLayer` поверх основной. Текстовый
кэш пакета при дополнительных сценах не используется (пересборка каждый кадр).
Небо задаёт только основная сцена.

**Переход** `SceneManager.transition(to_scene, from_scene=None, unload_from=True,
loading_scene=None)` → `SceneTransition` (`is_done`, `progress`, `error`,
`cancel()`): старая сцена остаётся активной, пока новая грузится порциями
(продвигается `update_loading`); затем `set_active`, выгрузка старой и экрана
загрузки. Ошибка/отмена — остаётся старая сцена.

## 7.8 Audio (`engine.audio`, A9a)

Ядро (`engine.core`) от аудио не зависит. Пакет `engine/audio/`:

- **Бэкенд** (`IAudioBackend`: `init/shutdown/play/stop/set_*_gain`). `SoftwareBackend` —
  микшер без устройства (`advance(seconds)` для тестов), основа `MiniaudioBackend`
  (extra `[audio]`, miniaudio 1.71; устройство тянет PCM у микшера). `NullBackend` —
  ничего не играет (плеер при `--headless`/`--no-audio`). `create_backend(kind)`
  создаёт и инициализирует; `"auto"` при ошибке устройства → Null + WARNING.
- **Формат PCM**: int16, interleaved, стерео, 44100 Гц. `AudioClip` приводит любой
  декодированный клип (`to_format`); WAV декодируется своим кодом, OGG — miniaudio.
  `ClipCache` — LRU по размеру (`CLIP_CACHE_LIMIT`, 64 МБ). Загрузка из VFS по `AssetRef`.
- **Шины**: `master/music/sfx/voice/ui`; итоговая громкость
  `min(1, source * bus * master)`. Громкости лежат в `Settings/audio.json`
  (`{шина: 0..1}`, битый файл/неизвестные шины игнорируются), читаются плеером.
  Сглаживание (A9b): коэффициенты левого/правого канала линейно рампятся внутри
  блока (`_Voice.last_lr`), смена gain/позиции доходит до цели за один блок.
- **Компоненты**: `AudioSourceComponent` (`clip`, `gain`, `pitch`, `loop`, `spatial`,
  `min_distance`, `max_distance`, `falloff`, `rolloff`, `cue`, `bus`, `autoplay`,
  `play_state` — не пишется в сцену, `save=True`), `AudioListenerComponent` (`active`).
  Формат сцены не менялся (новые поля — значения по умолчанию).
- **`AudioSystem`** (priority −100) синхронизирует голоса: autoplay, смена клипа →
  рестарт, `clip=None`/неактивный объект → стоп, gain только при изменении. Систему
  добавляет плеер (`scene.add_system`); в редакторе её нет — звука нет. Карта громкостей
  шин общая у систем всех сцен плеера. Стоимость `update` на 32 голоса ≈ 0.17 мс/кадр.
- Скрипты: `self.audio.play(clip, gain, bus, loop, pitch)`, `stop`, `set_bus_gain`,
  `get_bus_gain`, `master_gain`; без системы вызовы безопасны.
- Инспектор: группа «Звук», `AutoSection` с `ReferenceField` для клипа (drop `.wav/.ogg`),
  кью (`.cue.json`) и 3D-полей.

### 7.8.1 3D-звук, слушатель, музыка, SoundCue (A9b)

- **Затухание** (`engine/audio/spatial.py`): `Attenuation(min_distance=1, max_distance=50,
  falloff, rolloff=1)`, модели `linear`/`logarithmic`/`inverse` как в OpenAL; дальше
  `max_distance` — 0, ближе `min_distance` — 1; logarithmic/inverse зажаты в [min, max].
  Панорама: `pan_of` = проекция направления на «вправо» слушателя (`cross(up, forward)`,
  левосторонняя система, вперёд +Z). Отступление от наброска: `pan_gains` при pan=0
  оставляет оба канала 1 (громкость центра не падает вдвое).
- **Микшер**: позиция голоса и слушатель хранятся в `SoftwareBackend`
  (`play(..., position, attenuation)`, `set_position`, `set_listener(position, forward, up)`);
  затухание и панорама всех 3D-голосов считаются одним батчем numpy, группировка по
  `Attenuation`. `spatial_of(voice)` — (pan, atten) для тестов/отладки.
- **Слушатель**: `AudioSystem` каждый кадр зовёт `set_listener`. Приоритет: первый
  активный `AudioListenerComponent` → `CameraComponent` → один WARNING при 3D без слушателя.
- **Музыка** (`engine/audio/music.py`, `MusicPlayer`): `*.music.json`
  (`tracks[{clip,gain,fade_in,fade_out,stem?}]`, `loop`, `shuffle`), `play(playlist,
  crossfade, bus)`, `stop(crossfade)`, `current_track`, `set_intensity(0..1)` — один
  дополнительный «тревожный» стем (опциональное поле трека `stem`), громкости основной
  дорожки и стема рампятся от интенсивности. Кроссфейд — два голоса. Роль артефакта
  `music`.
- **SoundCue** (`engine/audio/cue.py`): `*.cue.json` (`clips[{clip,gain,pitch,weight}]`,
  `randomize_gain`, `randomize_pitch`, `avoid_repeat`). `SoundCue.pick(rng)` делает ровно
  3 обращения к `rng` (`Scene.random`, `seed_random`) — выбор детерминирован при
  фиксированном seed. `AudioSystem` кэширует кью, чтобы `avoid_repeat` помнил прошлый
  вариант. Роль артефакта `sound_cue`.
- **Импорт**: `CueImporter`/`MusicImporter` (`importers/audio_json_importer.py`) валидируют
  файл (`parse_cue`/`parse_playlist`) и кладут его в `Library`; чтение — `load_json_asset`
  (по роли, из `Library`/`.pak`/рабочего каталога).
- Стоимость: 32 3D-голоса + музыка (2 голоса) + кью, `AudioSystem.on_update` ≈ 0.4 мс/кадр
  (порог теста 0.5), микшер блока 16 мс ≈ 2.7 мс.
- Не делалось: реверберация, окклюзия, дакинг, HRTF, многослойные стемы, субтитры (A9c).

## 7.9 Экранные эффекты и масштаб рендера (A10a)

- `ScreenFx` (`engine/core/screen_fx.py`, `Scene.screen`, создаётся лениво): `fade_to(цвет,
  alpha, длительность)`, `flash(цвет, интенсивность, длительность)`, `clear()`,
  `update(dt)`, `fade_alpha`/`fade_color`/`active`. Без GL; рисует `PostStack` последним
  проходом. Плеер подставляет один экземпляр во все сцены, поэтому fade переживает
  смену сцены. Аргумент плеера `--fade ALPHA` — чёрный fade для скриншота.
- Скрипты: `self.screen.fade_to(цвет, alpha, время)`, `.flash()`, `.clear()`,
  `.fade_alpha` (`engine/scripting/screen_api.py`; и в прокси `Behaviour`).
- Событие `RenderScaleChanged(old, new)` (`engine/core/ui_events.py`) — шлёт плеер при
  смене масштаба внутреннего разрешения (dyn-res или `set_setting render_scale`).
- Dyn-res и параметры тиров — `docs/RENDERING.md` §10a. `set_setting` плеера принимает
  `post.tonemap|exposure|fxaa` и `render_scale` (редактор шлёт их из секции «Пост-процесс»).

## 7.10 Сборка игры в exe (A13)

* **Пак v2** (`pak.py`, заголовок `MEPK`, `PAK_VERSION = 2`): кроме артефактов `Library/`
  индекс содержит `files` (`build.json`, сцены, скрипты, настройки проекта). `build.json`:
  `engine_version, quality, name, version, hdr[, dynamic_resolution]`. Приоритет качества:
  `--quality` > `build.json` > `medium`; `hdr`/`dynamic_resolution` из `build.json` главнее
  `*.project.json`, `--hdr on|off` главнее всего. v1 читается как раньше.
* **Поиск пака** (`find_pak`): `--pak` → `<exe>/data.pak` → `sys._MEIPASS/data.pak`
  (onefile) → `<exe>/<project>/data.pak`. `--project` без значения — каталог пака/exe.
* **Пользовательские каталоги** (`engine/core/platform_info.py`): `user_data_root(game)`
  (`%APPDATA%`: сейвы, `logs/player.log`, краш-логи), `user_cache_root(game)`
  (`%LOCALAPPDATA%`: `shader_cache/`, `pak_extract/<guid>/`). В frozen-сборке включаются
  всегда (`--user-dirs` — и в обычном запуске). Нет доступа на запись — кэш шейдеров
  отключается с одним предупреждением.
* **Headless без PySide6**: `HiddenGlfwContext` (скрытое окно GLFW, GL 3.3 core) +
  `RenderTarget`; принудительно `MYENGINE_FORCE_HIDDEN_GLFW=1`.
* **Сборка** (`build/packaging/build.py`, extra `[package]`):
  `python build/packaging/build.py --project projects/X --mode onedir|onefile --out DIR`.
  Пак → иконка (`.ico` 16/32/48/256, по умолчанию из `cube.svg`) → `version_info.txt` →
  PyInstaller по `myengine_player.spec` (excludes: PySide6/shiboken6/tkinter/pytest/
  unittest/pandas/matplotlib/scipy/setuptools и др.; `collect_submodules("engine")` из-за
  ленивых импортов; манифест `player.manifest`: DPI-aware PerMonitorV2 + longPathAware;
  `--windowed`, без UPX). onedir: `data.pak` копируется рядом с exe; onefile — вшивается.
  В сборку не попадают `Assets/`, `Library/`, `docs/`, `scripts/`, `tools/`, `_work/`.
  Подробности и размеры — `_work/a13b_pyinstaller_report.md`.
