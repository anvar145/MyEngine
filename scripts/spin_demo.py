"""Демонстрационные скрипты для встроенного редактора MyEngine.

Каждый скрипт — обычный модуль Python с функциями жизненного цикла
``on_start(self)`` / ``on_update(self, delta_time)`` / ``on_destroy(self)``.
Доступ к движку идёт от лица ``self`` (компонента-скрипта):

``self.game_object``
    Игровой объект-владелец (``None`` при разовом запуске из редактора).
``self.scene``
    Сцена: ``self.scene.find("Cube")``, ``self.scene.query(...)``.
``self.transform``
    Трансформ владельца: позиция, поворот, масштаб.
``self.input``
    Состояние ввода: ``self.input.is_key_down("space")``.
``self.time``
    Тайминги: ``self.time.delta``, ``self.time.elapsed``, ``self.time.frame``.
``self.log(...)`` / ``print(...)``
    Сообщение в консоль редактора.

Дополнительно в пространство имён скрипта попадают имена ``Vector3``,
``Quaternion``, ``scene``, ``game_object``, ``input`` и ``time`` — короткая
запись ``Vector3.up()`` работает без импорта.

Example:
    Откройте панель «Скрипты» (Ctrl+Shift+E), выберите файл в дереве и нажмите
    F5, чтобы запустить его разово, либо «Прикрепить скрипт к выбранному
    объекту» — тогда ``on_update`` будет вызываться каждый кадр.
"""

__all__ = ["on_destroy", "on_reload", "on_start", "on_update"]

#: Скорость вращения в градусах в секунду.
SPEED = 90.0

#: Сколько времени скрипт работает до остановки (секунды).
LIFETIME = 5.0


def on_start(self) -> None:
    """Вызывается один раз при запуске скрипта."""
    owner = self.game_object.name if self.game_object is not None else "без объекта"
    self.log(f"spin начал работу на объекте '{owner}'")
    print("print() из скрипта тоже попадает в консоль редактора")
    if self.transform is None:
        self.log("объект не привязан — вращать нечего (разовый запуск из редактора)")


def on_update(self, delta_time: float) -> None:
    """Вызывается каждый кадр: delta_time — секунды с прошлого кадра.

    Args:
        delta_time: Длительность кадра в секундах.
    """
    if self.transform is None:
        # Разовый запуск из редактора (F5): у скрипта ещё нет объекта.
        return
    self.transform.rotate(Vector3.up(), SPEED * delta_time)  # noqa: F821 - имя даёт движок

    if self.time.elapsed >= LIFETIME:
        self.log(f"отпущенные {LIFETIME:g} с истекли — останавливаюсь")
        self.enabled = False


def on_destroy(self) -> None:
    """Вызывается при остановке скрипта (кнопка «Стоп» или удаление объекта)."""
    frames = self.time.frame
    elapsed = self.time.elapsed
    self.log(f"spin остановлен: кадров {frames}, время {elapsed:.2f} с")


def on_reload(self) -> None:
    """Необязательный хук: вызывается после горячей перезагрузки файла."""
    self.log("spin перезагружен — состояние сохранено")
