"""«Собирает» предметы: раз в две секунды пишет в журнал счётчик."""

from __future__ import annotations

INTERVAL = 2.0

_elapsed = 0.0
_collected = 0


def on_start(self):
    global _elapsed, _collected
    _elapsed = 0.0
    _collected = 0
    self.log(f"collector: старт, предметов в сцене: {len(self.scene)}")


def on_update(self, delta_time):
    global _elapsed, _collected
    _elapsed += delta_time
    if _elapsed >= INTERVAL:
        _elapsed -= INTERVAL
        _collected += 1
        self.log(f"collector: собрано {_collected}")


def on_destroy(self):
    self.log(f"collector: итого собрано {_collected}")
