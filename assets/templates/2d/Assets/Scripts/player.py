"""Пример скрипта 2D-проекта: объект плавно движется вправо."""

from __future__ import annotations

SPEED = 1.0


def on_start(self):
    self.log("Скрипт запущен")


def on_update(self, delta_time):
    position = self.transform.position
    position.x += SPEED * delta_time
    self.transform.position = position
