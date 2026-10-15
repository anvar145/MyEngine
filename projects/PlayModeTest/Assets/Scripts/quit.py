"""Выход из игры по событию UI ``event:quit`` (кнопка «Выход» в меню)."""

from __future__ import annotations

from engine.core.ui_events import QuitRequested, UIEventFired


def on_start(self):
    def handler(event):
        if event.event_name == "quit":
            self.log("quit: выход по кнопке")
            self.scene.events.emit(QuitRequested())

    self.scene.events.subscribe(UIEventFired, handler)
