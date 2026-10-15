# Скрипты: справочник API

Скрипты — функции-хуки модуля; `self` — компонент (`game_object`, `scene`,
`transform`, `input`, `time`, `physics`, `log()`). Подробности устройства —
в `docs/ARCHITECTURE.md` (§ скриптинг) и `docs/CORE.md`.

Хуки: `on_start(self)`, `on_update(self, delta_time)`,
`on_fixed_update(self, dt)`, `on_destroy(self)`, `on_reload`.

Пакетный хук `on_fixed_update_batch(self, items, dt)` (вместо `on_fixed_update`):
сцена собирает все компоненты с этим скриптом и вызывает функцию **один раз**
на шаг (после обхода компонентов, до физики); `self` — первый компонент,
`items` — список всех. Для однотипной логики сотен объектов (патруль врагов)
это на порядок дешевле вызова на каждый объект; в модуле доступен `numpy`
(`ALLOWED_MODULES`), массовая запись трансформов — `scene.transforms.get_local_batch`/
`set_local_batch`. Пример: `projects/PlayModeTest/Assets/Scripts/patrol.py`.

## Физика (`self.physics`)

Фасад `engine/scripting/physics_api.py`. Векторы принимаются как `Vector3`,
кортеж или список из трёх чисел; возвращаются простые типы (`dict`, кортежи,
объекты `GameObject`). Физический шаг идёт после `on_fixed_update`.
Для движения на фиксированной сетке (патруль kinematic-тел) время берите из `self.time.fixed_frame * self.time.fixed_delta` — `elapsed` идёт по кадрам и при срабатывании бюджета плеера расходится с физикой.

### Запросы и мир

| Метод | Возвращает |
|---|---|
| `raycast(origin, direction, max_distance=1000, layer_mask=0xFFFFFFFF, include_triggers=False)` | `dict(object, point, normal, distance)` или `None` |
| `overlap_sphere(center, radius, layer_mask=…, include_triggers=False)` | список `GameObject` |
| `set_gravity(x, y, z)` / `set_gravity(vec)`, `get_gravity()` | кортеж из трёх float |

### Тела

`apply_impulse(impulse, point=None, target=None)`, `set_velocity(v, target=None)`,
`get_velocity(target=None)` — по умолчанию работают с телом своего объекта.

### Персонаж (нужен `CharacterControllerComponent` на том же объекте)

| Метод | Описание |
|---|---|
| `is_grounded() -> bool` | Стоит ли персонаж на проходимой поверхности (проба `skin_width + ground_check_distance` вниз) |
| `simple_move() -> (dx, dy, dz)` | Один физический шаг **по вводу**: направление из WASD (W = −Z, D = +X в мировых осях, диагональ сжимается до длины 1), скорость `move_speed`, после `sprint(True)` — ещё `× sprint_multiplier`; в воздухе действует `air_control`. Гравитация и прыжок применяются автоматически |
| `simple_move(velocity) -> (dx, dy, dz)` | Один физический шаг с **явной скоростью**: `velocity` — горизонтальная (XZ) в м/с, Y игнорируется; `move_speed`, бег и `air_control` не применяются. Как и раньше |
| `move_character(motion) -> (dx, dy, dz)` | Сдвиг на `motion` (метры) с коллизиями и скольжением, **без** гравитации. Возвращает фактическое смещение |
| `jump(speed=None) -> bool` | Прыжок: без аргумента — `jump_velocity` компонента, иначе `speed` м/с; `True` — прыжок начат, `False` — персонаж не на земле (или `use_gravity=False`) |
| `sprint(on=True)` | Включает/выключает бег для `simple_move()` без аргумента; состояние хранится до следующего вызова |
| `set_air_control(value)` | Переопределяет `air_control` ([0, 1], иначе `ValueError`) на время работы скрипта; `None` возвращает значение поля. Поле компонента и сцена не меняются |

Правила:

- `simple_move` вызывают **раз за физический шаг** (из `on_fixed_update`). Каждый
  вызов — отдельный шаг `dt`; два вызова за кадр дадут двойное смещение и двойную
  гравитацию. `dt` берётся из физического мира (`1/60`), не из `on_update`.
- **Гравитация действует и без скрипта.** Если скрипт не двигал персонажа в этом
  физическом шаге (`simple_move`/`move_character`), адаптер сам делает «пустой»
  шаг с гравитацией: объект, созданный в воздухе, падает на землю. Отключается
  `use_gravity = False` в компоненте. Скрипт, зовущий `simple_move` каждый шаг,
  гравитацию получает один раз.
- `jump(speed)`: `speed <= 0` → `ValueError`; в воздухе возвращает `False` (двойной
  прыжок запрещён). Высота прыжка ≈ `v²/2g − v·dt/2` (при `v = 5`, `g = 9.81`
  ≈ 1.2 м).
- Без `CharacterControllerComponent` все методы персонажа бросают
  `RuntimeError("У объекта '<имя>' нет CharacterControllerComponent")`. Если
  компонент есть, но не создан из-за неверных параметров (`radius <= 0`,
  `height <= 2·radius`), текст: `RuntimeError("компонент не зарегистрирован:
  <причина>")`; причина также пишется в лог как ERROR (один раз на объект). То же
  `RuntimeError("У объекта … нет CharacterControllerComponent")` получит скрипт
  объекта, выключенного целиком (`active_self = False` у него или у родителя) при
  включённом компоненте: двойника в мире нет (`docs/CORE.md` §7.2).
- **Выключенный (`enabled = False`) контроллер — «заморожен»**, а не удалён:
  двойник остаётся в мире (коллизии, `raycast`, `overlap_sphere` его видят), но
  персонаж не двигается и не падает. Методы движения не бросают исключений:
  `simple_move`/`move_character` возвращают `(0.0, 0.0, 0.0)`, `jump` — `False`,
  `is_grounded` — последнее значение до заморозки; `sprint` и `set_air_control`
  работают как обычно. Скрипт можно не менять: он продолжает звать
  `simple_move()` каждый шаг. При `enabled = True` персонаж продолжает с текущей
  позы, скорости обнулены. Удалённый компонент (`remove_component`) — по-прежнему
  `RuntimeError` (см. выше).
- **`CharacterControllerComponent` и `RigidBodyComponent` на одном объекте
  несовместимы.** Такой объект в физику не попадает: в лог один раз уходит `ERROR`
  с именем объекта, а `simple_move`/`jump`/`is_grounded`/`move_character` и
  `apply_impulse`/`set_velocity`/`get_velocity` бросают
  `RuntimeError("CharacterController и RigidBody несовместимы")`. Уберите один из
  компонентов или выключите его (`enabled = False`): активный остаётся в мире,
  выключенный контроллер рядом с активным телом замороженным двойником не
  становится (`docs/CORE.md` §7.2, «Кто управляет объектом»).
- Нет скриптового доступа к ударам (`hits`) и флагам столкновений (`MoveResult` не
  делаем): `CharacterController.move()` возвращает `ndarray(3,)` — фактическое
  смещение, а скриптовые `move_character`/`simple_move` — кортеж из трёх `float`.
  Нормаль земли и данные о стенах из скрипта не видны — используйте `raycast`.
- Персонаж не толкает динамические тела и не едет на платформах (см. `docs/CORE.md`
  §7.2, ограничения F5/F6).
- Вектор из двух чисел → `ValueError`.

Пример (WASD + бег + прыжок, `on_fixed_update`; скорости и прыжок берутся из
полей компонента, настраиваются в инспекторе):

```python
def on_fixed_update(self, dt):
    self.physics.sprint(self.input.is_key_down("shift"))
    if self.input.is_key_down("space"):
        self.physics.jump()          # на земле — True, в воздухе — False
    self.physics.simple_move()       # ввод WASD, move_speed, гравитация
```

Прежний вариант с явной скоростью работает без изменений (эталонный
`player_controller.py` правок не потребовал):

```python
SPEED = 4.5


def on_fixed_update(self, dt):
    x = (1 if self.input.is_key_down("d") else 0) - (1 if self.input.is_key_down("a") else 0)
    z = (1 if self.input.is_key_down("s") else 0) - (1 if self.input.is_key_down("w") else 0)
    if self.input.is_key_down("space"):
        self.physics.jump()
    self.physics.simple_move((x * SPEED, 0, z * SPEED))
```

### Параметры `CharacterControllerComponent`

Всего 16 полей (все сериализуются в сцену; старые сцены с 7 полями читаются,
недостающие берут значения по умолчанию).

| Поле | По умолч. | Допустимо | Для чего |
|---|---|---|---|
| `radius` / `height` | 0.4 / 1.8 | `> 0` / `> 2·radius` | капсула; трансформ объекта — её центр |
| `step_offset` | 0.3 | `>= 0` | максимальная высота ступени |
| `slope_limit` | 45 | 0…90° | максимальный проходимый уклон |
| `skin_width` | 0.02 | `> 0` | зазор до препятствий |
| `layer` / `mask` | 1 / `0xFFFFFFFF` | 32 бита | слой двойника и маска столкновений |
| `move_speed` | 4.5 | `>= 0` | скорость ходьбы для `simple_move()`, м/с |
| `sprint_multiplier` | 1.6 | `>= 1` | множитель скорости при `sprint(True)` |
| `jump_velocity` | 5.0 | `> 0` | скорость для `jump()` без аргумента, м/с |
| `use_gravity` | `True` | bool | гравитация (в т.ч. пассивная); `False` — персонаж не падает и не прыгает |
| `gravity_scale` | 1.0 | `>= 0` | множитель гравитации мира |
| `ground_check_distance` | 0.03 | `> 0` | дальность пробы земли сверх `skin_width` |
| `air_control` | 0.3 | 0…1 | управляемость в воздухе для `simple_move()` без аргумента |
| `snap_to_ground` | `True` | bool | прилипание к земле на склонах и ступенях |
| `alignment_to_ground` | `False` | bool | наклон трансформа по нормали земли; капсула остаётся вертикальной |

Конструктор строг (вне диапазона — `ValueError` с именем поля), сцена и
присваивание после создания приводятся к диапазону с предупреждением в лог. Поля
скорости, гравитации, пробы земли и флаги «живые»: правка на работающем компоненте
двойник не пересоздаёт. Детали, ограничения (`scale` объекта игнорируется) и
поведение при ошибках — `docs/CORE.md` §7.2.

**Меш привязывайте к низу капсулы.** Центр капсулы в покое лежит на
`height/2 + skin_width` над землёй (для умолчаний `0.92`), поэтому модель с началом
координат у подошв делайте **дочерним** объектом со смещением
`y = -(height/2 + skin_width)`; иначе она «висит» над землёй.

## Переменные скрипта (поля в инспекторе)

Публичные глобальные переменные скрипта (имя не начинается с `_`) показываются в
инспекторе и настраиваются вручную; значение в коде — значение по умолчанию, ручное
значение его переопределяет (как поля `MonoBehaviour` в Unity). Скрипт при этом не
исполняется: поля ищутся разбором исходника (`engine/scripting/fields.py`).

```python
speed = 3.0                  # float
count = 5                    # int
label = "hi"                 # str
alive = True                 # bool
ratio: float = 1             # аннотация приводит к float
target: GameObject = None    # ссылка на объект сцены
icon: Asset = None           # ссылка на файл (в скрипте — `AssetRef`: `.guid`, `.path`)
```

- Полем становится только литерал (`bool`/`int`/`float`/`str`, в т. ч. `-4`) либо
  `None` с аннотацией `GameObject`/`Asset`. Функции, списки, вызовы, вычисления и
  `_private` пропускаются. `GameObject` и `Asset` доступны без импорта.
- Ссылки задаются перетаскиванием: объект — из иерархии, файл — из панели ассетов.
  Для проверки типа ссылки используйте `AssetRef`, не `Asset` (последнее — только аннотация).
  Кнопка `×` очищает ссылку, кнопка с иконкой «вернуть» (`rotate-ccw`) возвращает значение из кода; изменённое поле жирное.
- Значения применяются к модулю перед `on_start`, у каждого экземпляра свои
  (один скрипт на двух объектах — два набора значений). Переживают горячую
  перезагрузку; если тип поля в коде сменили, ручное значение сбрасывается с
  предупреждением, а поле, пропавшее из кода, ждёт возвращения.
- В сцене: `"fields"` в компоненте; объект — `{"object": id}`, файл —
  `{"asset": путь от корня проекта, "guid": ...}` (приоритет GUID). Ссылка на
  удалённый объект или отсутствующий файл становится `None`. Если у скрипта есть
  ссылки на объекты, `on_start` вызывается после загрузки всей сцены.
- Play Mode и плеер читают те же данные из снимка сцены; правка в инспекторе во
  время игры на неё не влияет.

## Behaviour: скрипт-класс с `@export` (A2a)

Альтернатива функциям-хукам. Существующие скрипты из функций работают без
изменений; формат `.scene.json` v1 не менялся (добавлен только необязательный ключ
`script_class` у компонента-скрипта, для функций его нет).

```python
from engine.scripting import Behaviour, export

class Spinner(Behaviour):
    speed: float = export(90.0, range=(0, 360), tooltip="град/с")
    target: GameObject = export(None)

    def on_update(self, delta_time):
        self.transform.rotate(Vector3.up(), self.speed * delta_time)
```

- **Хуки:** `on_attach`, `on_enable`, `on_disable`, `on_start`, `on_update(dt)`,
  `on_fixed_update(dt)`, `on_late_update(dt)`, `on_destroy`, `on_reload` и
  пакетный `@classmethod on_fixed_update_batch(cls, instances, dt)` (один вызов на
  все экземпляры класса, поштучный `on_fixed_update` тогда не зовётся). Вызываются
  только переопределённые (`Behaviour._overridden_hooks`). Порядок при добавлении
  к активному объекту: создание → поля → `on_attach` → `on_enable` → `on_start`.
- **Пакетный хук** `on_fixed_update_batch` вызывается у класса, который взял первый
  компонент в списке; обычно все компоненты списка — экземпляры одного и того же Behaviour.
- **Горячая перезагрузка:** экземпляр пересоздаётся, нематериализованное состояние
  (приватные атрибуты, не `export`) теряется. Сохранение состояния — задача A2b.
- **Доступ к движку:** `self.game_object/scene/transform/input/time/physics`
  и `self.log("...")` — прокси по явному списку `Behaviour.API_NAMES`; опечатка
  даёт `AttributeError`.
- **`export(default, **options)`** — обёртка над `reflection.field`
  (`range`, `step`, `tooltip`, `choices`, `kind`, ...). Вид выводится из аннотации
  или умолчания. Значения хранятся в `ScriptComponent.fields` в JSON-виде
  (`{"object": id}`, `{"asset": путь}`), в сцену пишутся только отличающиеся от
  умолчания. Число зажимается по `range`.
- **Приоритет:** если в модуле есть класс `Behaviour`, он выигрывает у функций-хуков.
  Берутся классы, объявленные в самом модуле и не являющиеся предками других. Если
  их несколько, а `script_class` не задан, берётся первый по порядку объявления
  (с предупреждением); имя пишется в `script_class`, редактор делает это сам
  выпадающим списком в строке скрипта. Неизвестное имя — предупреждение и режим
  функций.
- **Behaviour не использует `@component_fields` и не регистрируется в глобальном
  `registry`.** Иначе при горячей перезагрузке класс создаётся заново, а реестр
  копил бы устаревшие версии (утечка). Дескрипторы лежат в самом классе
  (`__behaviour_fields__`), читать их — `descriptors_from_behaviour`.
- **Инспектор:** поля показывает `ScriptFieldsForm` (bool/int/float/str/enum/vec3,
  ссылки на объект и файл; границы, шаг и подсказка из `export`; `color`, `quat`,
  списки пока только показываются). Правка попадает в undo/redo.
- **Не сделано (A3 и далее):** корутины/таймеры/tweens, prefab/uid.

### События, зависимости, hot reload (A2b)

- **`@event_handler(EventType, priority=0)`** помечает метод `Behaviour` как
  подписчика на `scene.events`. Подписки ставятся при добавлении компонента
  (не ждут `on_start`) и снимаются до `on_destroy`; декораторы стекуются (один метод
  на несколько типов), событие базового типа получают и подтипы, порядок —
  по приоритету. Переопределение метода без декоратора в наследнике снимает
  подписку. Ошибка хендлера попадает в лог скрипта и не ломает остальных.
  Ручные `scene.events.subscribe` в `on_start` движок **не отслеживает** —
  для них отписку делайте сами, лучше используйте `@event_handler`.
- **`@require_component(Type)`** (классовый, складывается с унаследованными).
  Если на объекте нет компонента, скрипт не стартует, а на первом кадре пишется
  один ERROR; блок снимается сам, когда компонент появляется. Порядок
  добавления компонентов не важен (при загрузке сцены проверка тихая до первого
  кадра). Удаление требуемого компонента в рантайме — no-op: скрипт продолжает
  работать, `get_component` вернёт `None`. Инспектор показывает красную подпись.
  В прокси есть `get_component(Type)` и `get_components(Type)`.
- **Hot reload с сохранением состояния.** Новый экземпляр создаётся
  `__init__`, затем в него переносится `__dict__` старого (кроме служебных
  `_component`, `_subscriptions` и `@export`-полей), затем применяются export-поля
  из сцены, `on_attach` и подписки. `on_destroy` зовётся у старого экземпляра,
  `on_reload` у нового, `on_start` второй раз **не** вызывается (если
  экземпляр ещё не стартовал — вызывается как обычно). Если тип перенесённого
  атрибута изменился, один warning на перезагрузку. Ограничения: ссылки на
  замыкания и объекты старой версии сохраняются как есть; `on_attach`
  выполняется заново и может затереть перенесённое — инициализируйте состояние в
  `__init__`.

  Что переносится: все атрибуты экземпляра, в том числе `_`-приватные.
  Что не переносится: `@export`-поля (берутся из сцены) и служебные
  `_component`/`_subscriptions`. Что затирается: всё, что присваивает `on_attach`.

  ```python
  class Door(Behaviour):
      speed: float = export(2.0)

      def __init__(self):          # верно: переживёт hot reload
          super().__init__()
          self._opened = 0.0

      def on_attach(self):         # неверно: _opened обнулится при reload
          self._opened = 0.0
  ```
- **Инспектор авто-секций:** header «Включён/Удалить» (`ComponentSection`),
  `quat` в углах Эйлера, `list` (`list[примитив]` и `list[Vector3]`, кнопки
  +/−/↑/↓), валидация `enum` (вне `choices` — warning и default, при политике
  `ERROR` — `ValueError`). У `ScriptFieldsForm` header нет: в строке скрипта уже
  есть «авто-старт» и «убрать».

## Пулы, запросы и тип `Prefab` (A4.4)

* **`on_spawn(self)` / `on_despawn(self)`** - хуки `Behaviour`/функциональных скриптов для
  объектов из `ObjectPool`: `on_spawn` после включения объекта, `on_despawn` до выключения
  (задачи планировщика компонента отменяются автоматически). Вне пула не вызываются.
* **`@export` тип `Prefab`**: `bullet: Prefab` - поле принимает только `*.prefab.json`
  (перетаскивание из панели ассетов). Значение - путь, годится для `scene.instantiate(self.bullet)`.
* Запросы: `scene.find_all_with_tag("Enemy")`, `obj.get_component_in_children(T)`,
  `obj.get_components_in_children(T, include_inactive=False)`, `obj.get_component_in_parent(T)`.
* Пул (`from engine.core.pool import ObjectPool`): `ObjectPool(self.scene, self.bullet, prewarm=20, max_size=50, on_exhausted="reuse_oldest")`,
  `bullet = pool.get(position)`, `pool.release(bullet)`. Прогрев делайте в `on_start`.

## Планировщик: таймеры, корутины, твины (A3a/A3b)

Одинаковый API у `Behaviour` и у функциональных скриптов (там `self` — компонент). Это тонкие обёртки над `self.scene.scheduler` с `owner=компонент`:

| Метод | Что делает |
|---|---|
| `self.after(seconds, cb, *, category=GAME)` | один вызов `cb()` через `seconds` |
| `self.every(seconds, cb, *, category=GAME, immediate=False)` | повтор каждые `seconds` |
| `self.start_coroutine(gen, *, category=GAME)` | запуск генератора-корутины |
| `self.tween(target, attr, end, duration, *, ease="ease_in_out_quad", category=GAME)` | плавное изменение атрибута, возвращает `TweenHandle` |
| `self.wait(s)` / `wait_unscaled(s)` / `wait_for_frame()` / `wait_for_fixed(n=1)` / `wait_until(pred)` | инструкции для `yield` в корутине |
| `self.cancel(handle)` / `self.cancel_all()` | отмена одной / всех задач компонента |

```python
class Door(Behaviour):
    def on_start(self):
        self.after(2.0, lambda: self.log("прошло 2 с"))
        self.every(0.5, self.tick, immediate=True)
        self.start_coroutine(self.sequence())
        self.tween(self.transform, "position", Vector3(0, 3, 0), 1.5,
                   ease="ease_out_cubic").on_complete(lambda: self.log("открыта"))

    def sequence(self):
        yield self.wait(1.0)            # игровое время (с time_scale, стоит на паузе)
        yield self.wait_unscaled(1.0)   # без time_scale, на паузе тоже стоит
        yield self.wait_for_frame()     # следующий кадр update
        yield self.wait_for_fixed(2)    # 2 фиксированных шага
        yield self.wait_until(lambda: self.ready)
        return "готово"                 # -> handle.result
```

### Твины
* Типы атрибута: `float`/`int` (в конце ровно `end`), `Vector3`, `Quaternion` (slerp), кортеж чисел (поэлементно, длины должны совпасть). Другой тип — `TypeError` при создании.
* `ease` — имя (`linear`, `ease_in_quad`, `ease_out_quad`, `ease_in_out_quad` (по умолчанию), `ease_in_cubic`, `ease_out_cubic`, `ease_in_out_cubic`, `smoothstep`) или функция `f(t)`; неизвестное имя — `ValueError` со списком.
* Цепочка: `.on_complete(fn)` (без аргументов, один раз; при отмене не вызывается) и `.on_update(fn)` (получает текущее значение). `duration <= 0`: значение ставится сразу, `on_complete` — при ближайшем update. Два твина одного атрибута допустимы, применяются в порядке создания (побеждает последний).
* Стартовое значение — снимок в момент вызова; запись начинается при ближайшем update.

### Жизненный цикл и ошибки
* **Автоочистка:** при удалении компонента/объекта (`on_destroy`) все задачи отменяются **до** пользовательского `on_destroy`.
* **Hot reload:** задачи старой версии отменяются и не переносятся. У функциональных скриптов новый `on_start` создаст свои; у `Behaviour` `on_start` не повторяется (состояние перенесено) — новые задачи создавайте в `on_reload`.
* **`enabled=False` не останавливает задачи:** таймер и корутина сработают на выключенном компоненте, хотя `on_update` не вызывается. Нужна остановка — `self.cancel_all()` в `on_disable`.
* **Пауза и UI:** игровые `after`/`Wait` на паузе стоят. Для меню паузы ставьте `category=TimeCategory.UI`; `wait_unscaled` — игровое время без `time_scale`, на паузе тоже стоит.
* Первый шаг корутины — при ближайшем `update`, а не внутри `start_coroutine`.
* **Ошибки:** исключение в callback таймера/корутине пишется в лог с трассировкой (`handle.errors` растёт), компонент и планировщик продолжают работать; `every` продолжается, `after` и корутина завершаются. Одинаковая ошибка (тип + сообщение) логируется 2 раза, затем один раз сообщается о подавлении повторов. Ошибка в `on_complete` твина — лог, твин завершён; ошибка в `on_update` — лог, `on_update` отключается.
* **Ошибки в твине:** ошибки в `on_complete`/`on_update` твина попадают в лог с трассировкой без throttling (у таймеров и корутин — с throttling через `ScriptComponent._task_error`). Если в `on_complete` критичная логика — оборачивайте её в `self.after(0, wrapper)`, где wrapper логируется как задача Behaviour.
* Вне `GameLoop` (тесты, утилиты) UI-часы двигайте `scene.tick_scheduler_ui(dt)`; игровые двигает `scene.update(dt)`.
* Без сцены (`scene is None`) методы создания задач бросают `RuntimeError`.

## Ввод: действия (`self.input`, A5)

`self.input` — `engine.core.input.InputState`. Старый API (`is_key_down("w")`,
`pressed_keys`) работает как раньше и контекстами не фильтруется.

```python
def on_update(self, delta_time):
    if self.input.pressed("Jump"):        # только в первом кадре нажатия
        self.physics.jump()
    move = self.input.axis2("Move")       # (x, y), длина <= 1, WASD/стик
    look = self.input.axis2("Look")       # дельта мыши за кадр
    if self.input.held("Sprint"): ...
    if self.input.released("Fire"): ...
    v = self.input.vec2("MoveX", "MoveY") # Vector3(x, y, 0) из двух осей
```

- Методы: `held`, `pressed`, `released` (кнопки), `axis` (-1..1), `axis2`,
  `vec2`. Неизвестное действие даёт нейтральное значение и одно WARNING.
- Дефолтные действия: `assets/default_input_map.json` (Jump, Fire, Interact,
  Sprint, Pause, Menu, Move, Look, UIClick, UICancel).
- `self.input.contexts` — стек контекстов (`game`, `paused`, `ui`, `cutscene`):
  `push(ui_context())` при открытии меню, `pop("ui")` при закрытии. `ui` и
  `paused` пропускают `Pause`, чтобы меню закрывалось той же клавишей.
  Стек сбрасывается до `game` при загрузке сцены.
- Кадр: плеер вызывает `input.end_frame()` после `tick()`; `pressed`/`released`
  считаются относительно прошлого кадра, нажатие+отпускание внутри кадра не теряется.
- Плеер: `--input-map PATH` (JSON формата `InputMap.save`, поверх дефолтов),
  иначе `Settings/input.json` проекта; `--hold-keys w,space` держит клавиши
  каждый кадр через `ManualInputBackend`.

### Ремаппинг (A5 п.3)

```python
def on_start(self):
    self.input.on_rebind("Jump", lambda a, i, b: self.log(f"{a}[{i}] -> {b.code}"))
    self.input.begin_rebind("Jump")       # следующая клавиша/кнопка станет Jump

def on_update(self, delta_time):
    if self.input.rebinding is None and self.input.pressed("Pause"):
        self.input.save_bindings()        # в Settings/input.json (только отличия)
```

- `begin_rebind(action, index=0)`: следующий `key_down` / кнопка мыши / кнопка
  геймпада заменяет привязку `index` (индекс = числу привязок добавляет новую).
  Нажатие поглощается (игре не уходит), `Escape` отменяет, `cancel_rebind()` тоже.
  У оси (`Move`) сохраняются `scale`/`axis`, поэтому W→I остаётся «вперёд».
  Мышиную дельту и `axis:` записать нельзя (нужна кнопка/клавиша).
- `on_rebind(action | None, callback(action, index, binding))`.
- `InputMap`: `rebind`, `unbind`/`remove_binding`, `add_binding`,
  `reset_to_default(action)`, `reset_all()`, `is_default(action)`,
  `save(path, only_changes=False)`, `load(path)`.
- `save_bindings(path=None)` / `load_bindings(path=None)`: путь по умолчанию —
  `InputState.settings_path` (плеер ставит `<проект>/Settings/input.json`);
  без пути `ValueError`.
- Формат: `{"version": 1, "actions": {"Jump": {"kind": "button", "bindings":
  [{"device", "code", "scale", "axis"}]}}}`. Сохраняются только действия, чьи
  привязки отличаются от дефолтов; отсутствующее в файле действие берётся из
  дефолта. Действие с удалёнными привязками пишется с пустым `bindings`.

## Сохранения: `@export(save=True)`, `on_save`/`on_load` (A7a)

```python
class Hero(Behaviour):
    hp: int = export(100, save=True)      # значение из слота перекрывает сцену
    speed: float = export(5.0)            # без save — всегда из сцены
    buddy: GameObject = export(None, save=True)   # ссылка по uid

    def on_save(self, ctx):               # дополнительные данные
        ctx.set("inventory.items", self.items)

    def on_load(self, ctx):               # поля save=True уже применены
        self.items = ctx.get("inventory.items", [])
```

- Функциональный скрипт: `def on_save(self, ctx)` / `def on_load(self, ctx)`.
- `ctx` — `SaveContext` этого компонента (вложенные ключи `"a.b"`, типы: числа,
  строки, bool, list/dict/tuple, `Vector3`; `ctx.get_typed(key, int)`).
- Глобальное состояние — `self.scene.state.set("quest.stage", 2)`,
  `self.scene.state.on_changed("quest", cb)`; сохраняется целиком.
- `self.scene.save("slot1")` / `self.scene.load("slot1")` (нужен
  `scene.save_manager`, ставит хост). Подробности формата — `docs/CORE.md` §7.6.
- Не сохраняются: активные `after/every/tween`, скорости RigidBody — восстановите
  в `on_load` (например, перезапустите таймер).
- `self.checkpoint.save("name", idempotent=False)` — записать слот из скрипта
  (`True`, если записан; `idempotent=True` пропускает запись без изменений).
  Компоненты `CheckpointComponent`/`SafeZone`/`AutoSavePolicy`, F5/F9 и каталог
  сейвов — `docs/CORE.md` §7.6. Пример слотов на UI — `scripts/seed_save_demo.py`
  и `Assets/Scripts/save_slots.py` в PlayModeTest (сцена `save_menu`).

## Звук: `self.audio` (A9a)

```python
def on_start(self):
    self.audio.set_bus_gain("music", 0.5)       # master/music/sfx/voice/ui, 0..1
    self.voice = self.audio.play("Audio/bell.wav", gain=0.8, bus="sfx", pitch=1.0)

def on_destroy(self):
    self.audio.stop(self.voice)                 # или self.voice.stop()
```

`play(clip, gain=1, bus="sfx", loop=False, pitch=1)` принимает `Asset`/`AssetRef`/путь и
возвращает `VoiceHandle` (`stop()`, `set_gain()`, `is_playing`); неизвестная шина →
`ValueError`. Без `AudioSystem` (редактор, `--no-audio` не влияет — Null просто молчит)
вызовы безопасны. Для постоянных источников — компонент `AudioSourceComponent`.

### 3D-звук, музыка, SoundCue (A9b)

```python
def on_start(self):
    # 3D: позиция в мире, затухание по полям min/max_distance (по умолчанию 1/50)
    self.audio.play_at("Audio/boom.wav", self.transform.position, gain=0.9, bus="sfx")

    # SoundCue: случайный вариант из *.cue.json (веса, gain/pitch-разброс, без повторов)
    self.audio.play_cue("Audio/step.cue.json", gain=1.0, bus="sfx")
    self.audio.play_cue("Audio/step.cue.json", position=(3.0, 0.0, 1.0))

    # Музыка: плейлист *.music.json, кроссфейд в секундах
    self.audio.music.play("Audio/level1.music.json", crossfade=2.0)
    self.audio.music.set_intensity(0.7)         # 0..1: подмешивает «тревожный» стем
    print(self.audio.music.current_track)       # None, если музыка не играет
    self.audio.music.stop(crossfade=1.0)
```

`play_at(clip, position, gain=1, bus="sfx", loop=False, pitch=1)` и
`play_cue(cue, gain=1, bus="sfx", position=None)` возвращают `VoiceHandle`; `position` —
`Vector3` или кортеж из трёх чисел. Слушатель — `AudioListenerComponent` (поле `active`),
иначе камера сцены; без слушателя 3D-голос играет как обычный и пишется один WARNING.
Случайность кью берётся из `scene.random` (`scene.seed_random(n)` — воспроизводимо).
Без `AudioSystem` все вызовы — безопасные no-op (`music.play` возвращает `False`).
У компонента `AudioSourceComponent` поле `cue` заменяет `clip`: каждый запуск выбирает
вариант кью.
