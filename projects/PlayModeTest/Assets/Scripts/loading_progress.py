"""Экран загрузки: текст «Загрузка… N%» по событиям ``SceneLoadProgress``.

Плеер пробрасывает прогресс менеджера сцен в шину каждой загруженной сцены,
поэтому скрипту экрана загрузки достаточно подписаться на ``scene.events``.
Объект с `UIText` называется ``LoadingLabel``.
"""

from __future__ import annotations

from engine.core.ui_components import UIText
from engine.core.ui_events import SceneLoadProgress


def on_start(self):
    label = self.scene.find("LoadingLabel")
    element = label.get_component(UIText) if label is not None else None
    if element is None:
        self.log("loading_progress: нет объекта LoadingLabel с UIText")
        return

    def handler(event):
        element.text.text = f"Загрузка... {round(event.progress * 100)}%"

    self.scene.events.subscribe(SceneLoadProgress, handler)
