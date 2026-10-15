"""Слоты сохранений на готовом UI: события ``save_N``/``load_N``/``delete_N``.

Пример игрового кода поверх ``Scene.save/load`` и ``GameState``: компонент движка
не нужен. Событие ``gold`` прибавляет 10 к ``GameState['gold']``.
"""

from __future__ import annotations

import time

from engine.core.ui_components import UIText
from engine.core.ui_events import UIEventFired

SLOTS = (1, 2, 3)


def _slot_name(number):
    return f"slot{number}"


def _refresh(self):
    manager = self.scene.save_manager
    gold = self.scene.state.get("gold", 0)
    header = self.scene.find("GoldLabel")
    if header is not None:
        header.get_component(UIText).text.text = f"Золото: {gold}"
    for number in SLOTS:
        label = self.scene.find(f"SlotLabel_{number}")
        if label is None:
            continue
        slot = manager.slot(_slot_name(number)) if manager else None
        if slot is None:
            text = f"Слот {number}: пусто"
        else:
            stamp = time.strftime("%H:%M:%S", time.localtime(slot.meta.timestamp))
            text = f"Слот {number}: {stamp}"
        label.get_component(UIText).text.text = text


def on_start(self):
    def handler(event):
        name = event.event_name
        scene = self.scene
        if name == "gold":
            scene.state.set("gold", scene.state.get("gold", 0) + 10)
        elif name.startswith("save_"):
            scene.save(_slot_name(int(name[5:])))
        elif name.startswith("load_"):
            slot = _slot_name(int(name[5:]))
            if scene.save_manager.slot(slot) is not None:
                scene.load(slot)
        elif name.startswith("delete_"):
            scene.save_manager.delete(_slot_name(int(name[7:])))
        else:
            return
        _refresh(self)

    self.scene.events.subscribe(UIEventFired, handler)
    _refresh(self)
