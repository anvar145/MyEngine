"""Реестр отладочных оверлеев: что показывать поверх кадра (F3 в плеере).

Оверлей — это имя канала :class:`~engine.debug.draw.DebugDraw` плюс, по
желанию, провайдер: функция, которая раз в кадр сама рисует в буфер (каркасы
навмеша, кости скелетов). Общий выключатель :attr:`OverlayRegistry.visible`
скрывает всё разом, флаг оверлея — только его канал.

Встроенные оверлеи (:data:`BUILTIN_OVERLAYS`):

- ``colliders`` — каркасы коллайдеров. Рисует их отдельный
  ``PhysicsDebugPass`` (F2 в плеере, меню «Вид → Коллайдеры» в редакторе);
  здесь оверлей только перечислен, чтобы все отладочные слои были в одном
  списке. Его флаг не зависит от общего выключателя F3;
- ``nav`` — навмеш и порталы (провайдер появится в B3);
- ``bones`` — кости скелетов (провайдер появится в B1);
- ``sight`` — конусы зрения (:meth:`DebugDraw.cone`);
- ``paths`` — пути и маршруты (:meth:`DebugDraw.path`);
- ``script`` — всё, что скрипты рисуют без явного канала.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from engine.debug.draw import DEFAULT_CHANNEL, DebugDraw

__all__ = ["BUILTIN_OVERLAYS", "Overlay", "OverlayRegistry"]

_log = logging.getLogger("engine.debug.overlays")

#: Провайдер: ``(буфер, контекст) -> None``; контекст задаёт владелец прохода
#: (сцена плеера или редактора), провайдер может его игнорировать.
Provider = Callable[[DebugDraw, Any], None]

#: ``(имя, подпись, включён по умолчанию, внешний)``.
BUILTIN_OVERLAYS: tuple[tuple[str, str, bool, bool], ...] = (
    ("colliders", "Коллайдеры", False, True),
    ("nav", "Навмеш", True, False),
    ("bones", "Кости", True, False),
    ("sight", "Конусы зрения", True, False),
    ("paths", "Пути", True, False),
    (DEFAULT_CHANNEL, "Скрипты", True, False),
)


@dataclass(slots=True)
class Overlay:
    """Один оверлей.

    Attributes:
        name: Имя (и канал в :class:`DebugDraw`).
        label: Подпись в меню.
        enabled: Включён ли оверлей (при включённом общем выключателе).
        provider: Функция, рисующая оверлей каждый кадр, или ``None``.
        external: Оверлей рисует собственный проход (коллайдеры): реестр
            хранит флаг и сообщает об изменениях, но в ``DebugPass`` не идёт.
    """

    name: str
    label: str
    enabled: bool = True
    provider: Provider | None = None
    external: bool = False


class OverlayRegistry:
    """Набор оверлеев и общий выключатель.

    Args:
        builtin: Зарегистрировать :data:`BUILTIN_OVERLAYS`.
    """

    def __init__(self, *, builtin: bool = True) -> None:
        #: Общий выключатель (F3). Выключен по умолчанию: игра без отладки.
        self.visible = False
        self._overlays: dict[str, Overlay] = {}
        self._listeners: list[Callable[[OverlayRegistry], None]] = []
        if builtin:
            for name, label, enabled, external in BUILTIN_OVERLAYS:
                self.register(name, label, enabled=enabled, external=external)

    # ------------------------------------------------------------------ #
    # Состав
    # ------------------------------------------------------------------ #
    def register(
        self,
        name: str,
        label: str | None = None,
        provider: Provider | None = None,
        *,
        enabled: bool = True,
        external: bool = False,
    ) -> Overlay:
        """Добавить оверлей (или обновить провайдер/подпись существующего).

        Флаг ``enabled`` уже зарегистрированного оверлея не сбрасывается:
        повторная регистрация провайдера (пересоздание сцены) не должна
        включать то, что пользователь выключил.
        """
        overlay = self._overlays.get(name)
        if overlay is None:
            overlay = Overlay(name, label or name, enabled, provider, external)
            self._overlays[name] = overlay
        else:
            if label:
                overlay.label = label
            if provider is not None:
                overlay.provider = provider
        self._notify()
        return overlay

    def set_provider(self, name: str, provider: Provider | None) -> None:
        """Задать или снять провайдер оверлея (оверлей должен существовать)."""
        self._overlays[name].provider = provider

    def get(self, name: str) -> Overlay | None:
        """Оверлей по имени."""
        return self._overlays.get(name)

    def __iter__(self):
        return iter(list(self._overlays.values()))

    def __len__(self) -> int:
        return len(self._overlays)

    def names(self) -> list[str]:
        """Имена оверлеев в порядке регистрации."""
        return list(self._overlays)

    # ------------------------------------------------------------------ #
    # Флаги
    # ------------------------------------------------------------------ #
    def set_visible(self, visible: bool) -> None:
        """Включить/выключить все оверлеи ``DebugPass`` разом."""
        if self.visible != bool(visible):
            self.visible = bool(visible)
            self._notify()

    def toggle(self) -> bool:
        """Переключить общий выключатель (F3); вернуть новое состояние."""
        self.set_visible(not self.visible)
        return self.visible

    def set_enabled(self, name: str, enabled: bool) -> None:
        """Включить/выключить один оверлей."""
        overlay = self._overlays[name]
        if overlay.enabled != bool(enabled):
            overlay.enabled = bool(enabled)
            self._notify()

    def is_shown(self, name: str) -> bool:
        """Виден ли оверлей сейчас (внешние не зависят от F3)."""
        overlay = self._overlays.get(name)
        if overlay is None or not overlay.enabled:
            return False
        return overlay.external or self.visible

    def active_channels(self) -> list[str]:
        """Каналы, которые рисует ``DebugPass`` в этом кадре."""
        if not self.visible:
            return []
        return [o.name for o in self._overlays.values() if o.enabled and not o.external]

    def subscribe(self, listener: Callable[[OverlayRegistry], None]) -> None:
        """Слушатель изменений флагов (меню редактора, проход коллайдеров)."""
        self._listeners.append(listener)

    def unsubscribe(self, listener: Callable[[OverlayRegistry], None]) -> None:
        """Снять слушателя."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify(self) -> None:
        for listener in list(self._listeners):
            listener(self)

    # ------------------------------------------------------------------ #
    # Кадр
    # ------------------------------------------------------------------ #
    def run_providers(self, draw: DebugDraw, context: Any = None) -> int:
        """Вызвать провайдеры видимых оверлеев; вернуть число вызванных.

        Ошибка провайдера пишется в журнал и не роняет кадр.
        """
        if not self.visible:
            return 0
        called = 0
        for overlay in list(self._overlays.values()):
            if overlay.provider is None or not overlay.enabled or overlay.external:
                continue
            try:
                overlay.provider(draw, context)
            except Exception as error:  # noqa: BLE001 - отладка не роняет кадр
                _log.warning("Оверлей %s: %s", overlay.name, error)
            called += 1
        return called
