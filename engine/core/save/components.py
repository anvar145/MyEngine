"""Компоненты сохранений: контрольная точка, безопасная зона, политика автосейва."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from engine.core.component import Component
from engine.core.reflection import component_fields, declare, registry
from engine.core.save.service import AUTOSAVE_SLOT, save_checkpoint
from engine.physics.events import TriggerEnter, TriggerExit

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.core.event_bus import Subscription
    from engine.core.game_object import GameObject
    from engine.core.scene import Scene

__all__ = [
    "AutoSavePolicy",
    "CheckpointComponent",
    "SafeZone",
    "autosave_before_switch",
]


def _is_player(obj: Any, tag: str) -> bool:
    """Объект с тегом ``tag`` (без учёта регистра)."""
    return (
        obj is not None
        and bool(tag)
        and str(getattr(obj, "tag", "")).lower() == tag.lower()
    )


class _TriggerWatcher(Component):
    """База: подписка на события триггера этого объекта (``a`` - триггер)."""

    __slots__ = ("_subs",)

    def __init__(self) -> None:
        super().__init__()
        self._subs: list[Subscription] = []

    def _event_types(self) -> tuple[type, ...]:
        return (TriggerEnter,)

    def on_attach(self) -> None:
        scene = self.scene
        if scene is None:
            return
        for event_type in self._event_types():
            self._subs.append(scene.events.subscribe(event_type, self._on_event))

    def on_detach(self) -> None:
        for sub in self._subs:
            sub.cancel()
        self._subs.clear()

    def _on_event(self, event: Any) -> None:
        if event.user_a is not self.game_object or not self.enabled:
            return
        self._handle(event, event.user_b)

    def _handle(self, event: Any, other: GameObject | None) -> None:
        raise NotImplementedError


@component_fields
class CheckpointComponent(_TriggerWatcher):
    """Контрольная точка: вход игрока в триггер записывает слот.

    Нужен ``ColliderComponent`` с ``is_trigger`` на том же объекте. Вошедший
    должен иметь тег ``player_tag``.

    Args:
        slot_name: Имя записываемого слота.
        idempotent: Не перезаписывать слот, если с прошлой записи не менялись
            ни ``GameState``, ни ``@save``-данные сцены.
        player_tag: Тег игрока.
    """

    __slots__ = ("idempotent", "player_tag", "slot_name")

    __field_specs__ = (
        declare("slot_name", "checkpoint", tooltip="Слот, куда пишется точка"),
        declare("idempotent", True, tooltip="Не писать повторно без изменений"),
        declare("player_tag", "player", tooltip="Тег игрока"),
    )

    def __init__(
        self,
        slot_name: str = "checkpoint",
        idempotent: bool = True,
        player_tag: str = "player",
    ) -> None:
        super().__init__()
        self.slot_name = str(slot_name)
        self.idempotent = bool(idempotent)
        self.player_tag = str(player_tag)

    def _handle(self, event: Any, other: GameObject | None) -> None:
        scene = self.scene
        if scene is not None and _is_player(other, self.player_tag):
            save_checkpoint(scene, self.slot_name, idempotent=self.idempotent)

    def to_dict(self) -> dict[str, object]:
        """Сериализовать компонент (для сцены)."""
        return registry.serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointComponent:
        """Восстановить компонент; отсутствующие поля берутся по умолчанию."""
        return registry.deserialize(cls, data)


@component_fields
class SafeZone(_TriggerWatcher):
    """Безопасная зона: пока в триггере игрок, работает автосейв по таймеру.

    Args:
        player_tag: Тег игрока.
    """

    __slots__ = ("occupants", "player_tag")

    __field_specs__ = (declare("player_tag", "player", tooltip="Тег игрока"),)

    def __init__(self, player_tag: str = "player") -> None:
        super().__init__()
        self.player_tag = str(player_tag)
        self.occupants: set[int] = set()

    def _event_types(self) -> tuple[type, ...]:
        return (TriggerEnter, TriggerExit)

    @property
    def occupied(self) -> bool:
        """В зоне есть игрок."""
        return bool(self.occupants)

    def _handle(self, event: Any, other: GameObject | None) -> None:
        if not _is_player(other, self.player_tag):
            return
        if isinstance(event, TriggerEnter):
            self.occupants.add(id(other))
        else:
            self.occupants.discard(id(other))

    def to_dict(self) -> dict[str, object]:
        """Сериализовать компонент (для сцены)."""
        return registry.serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SafeZone:
        """Восстановить компонент."""
        return registry.deserialize(cls, data)


@component_fields
class AutoSavePolicy(Component):
    """Политика автосохранения сцены (слот ``_autosave``, один, перезаписывается).

    Автосейв по триггеру - это ``CheckpointComponent`` со слотом ``_autosave``.

    Args:
        on_scene_change: Писать перед сменой активной сцены.
        interval_sec: Период таймера, с (0 - таймер выключен).
        require_safe_zone: Таймер пишет, только пока игрок в ``SafeZone``.
    """

    __slots__ = ("_clock", "interval_sec", "on_scene_change", "require_safe_zone")

    __field_specs__ = (
        declare("on_scene_change", True, tooltip="Писать перед сменой сцены"),
        declare("interval_sec", 0.0, tooltip="Период таймера, с (0 - выкл.)"),
        declare("require_safe_zone", True, tooltip="Таймер только в безопасной зоне"),
    )

    def __init__(
        self,
        on_scene_change: bool = True,
        interval_sec: float = 0.0,
        require_safe_zone: bool = True,
    ) -> None:
        super().__init__()
        self.on_scene_change = bool(on_scene_change)
        self.interval_sec = max(0.0, float(interval_sec))
        self.require_safe_zone = bool(require_safe_zone)
        self._clock = 0.0

    def in_safe_zone(self) -> bool:
        """В какой-либо ``SafeZone`` сцены сейчас находится игрок."""
        scene = self.scene
        if scene is None:
            return False
        return any(
            zone.occupied
            for obj in scene.objects
            for zone in obj.get_components(SafeZone)
        )

    def on_update(self, dt: float) -> None:
        """Таймер автосейва."""
        scene = self.scene
        if self.interval_sec <= 0.0 or scene is None or scene.save_manager is None:
            return
        self._clock += dt
        if self._clock < self.interval_sec:
            return
        self._clock = 0.0
        if self.require_safe_zone and not self.in_safe_zone():
            return
        save_checkpoint(scene, AUTOSAVE_SLOT)

    def to_dict(self) -> dict[str, object]:
        """Сериализовать компонент (для сцены)."""
        return registry.serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AutoSavePolicy:
        """Восстановить компонент."""
        return registry.deserialize(cls, data)


def autosave_before_switch(old: Scene | None, new: Scene | None = None) -> bool:
    """Автосейв уходящей сцены (хук ``SceneManager.before_activate``).

    Args:
        old: Активная сцена до переключения.
        new: Сцена, которая станет активной (если совпадает с ``old`` - ничего).

    Returns:
        ``True``, если слот записан.
    """
    if old is None or old is new or old.save_manager is None:
        return False
    for obj in old.objects:
        for policy in obj.get_components(AutoSavePolicy):
            if policy.enabled and policy.on_scene_change:
                return save_checkpoint(old, AUTOSAVE_SLOT)
    return False
