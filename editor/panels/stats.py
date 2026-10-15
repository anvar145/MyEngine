"""Панель статистики: состояние сцены, показатели рендера и камеры."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from editor.panels.base import DockPanel
from editor.widgets import SectionBox

__all__ = ["StatsPanel"]

#: Метрики сцены: ключ перевода → имя атрибута SceneStats.
SCENE_FIELDS: tuple[tuple[str, str], ...] = (
    ("stats.objects", "game_objects"),
    ("stats.components", "components"),
    ("stats.active_components", "active_components"),
    ("stats.systems", "systems"),
)

#: Метрики рендера: ключ перевода → имя атрибута RenderStats.
RENDER_FIELDS: tuple[tuple[str, str], ...] = (
    ("stats.draw_calls", "draw_calls"),
    ("stats.batches", "batches"),
    ("stats.triangles", "triangles"),
    ("stats.instances", "instances"),
    ("stats.culled", "culled"),
    ("stats.lights", "lights_bound"),
    ("stats.shadow_draws", "shadow_draws"),
)

#: Метрики внешнего плеера: ключ события → ключ перевода подписи.
PLAYER_FIELDS: tuple[tuple[str, str], ...] = (
    ("fps", "stats.fps"),
    ("dc", "stats.draw_calls"),
    ("tri", "stats.triangles"),
    ("script_ms", "play.script_ms"),
)


class StatsPanel(DockPanel):
    """Сводка по сцене и рендереру, обновляемая таймером редактора."""

    def __init__(self, context, parent: QWidget | None = None) -> None:
        super().__init__("panel.stats", context, parent)
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignTop)

        self.scene_section = SectionBox(self.tr("stats.scene"), container)
        self.scene_form = QFormLayout()
        self.scene_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.scene_labels: dict[str, QLabel] = {}
        for key, _attr in SCENE_FIELDS:
            label = QLabel("0", container)
            self.scene_labels[key] = label
            self.scene_form.addRow(self.tr(key), label)
        self.scene_section.add_layout(self.scene_form)
        layout.addWidget(self.scene_section)

        self.render_section = SectionBox(self.tr("stats.render"), container)
        self.render_form = QFormLayout()
        self.render_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.render_labels: dict[str, QLabel] = {}
        for key, _attr in RENDER_FIELDS:
            label = QLabel("0", container)
            self.render_labels[key] = label
            self.render_form.addRow(self.tr(key), label)
        self.fps_label = QLabel("—", container)
        self.render_labels["stats.fps"] = self.fps_label
        self.render_form.addRow(self.tr("stats.fps"), self.fps_label)
        self.render_section.add_layout(self.render_form)
        layout.addWidget(self.render_section)

        self.camera_section = SectionBox(self.tr("stats.camera"), container)
        self.camera_form = QFormLayout()
        self.camera_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.camera_position = QLabel("—", container)
        self.camera_distance = QLabel("—", container)
        self.camera_form.addRow(self.tr("stats.camera_position"), self.camera_position)
        self.camera_form.addRow(self.tr("stats.camera_distance"), self.camera_distance)
        self.camera_section.add_layout(self.camera_form)
        layout.addWidget(self.camera_section)

        self.player_section = SectionBox(self.tr("play.player_stats"), container)
        self.player_form = QFormLayout()
        self.player_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.player_labels: dict[str, QLabel] = {}
        for key, caption in PLAYER_FIELDS:
            label = QLabel("—", container)
            self.player_labels[key] = label
            self.player_form.addRow(self.tr(caption), label)
        self.player_section.add_layout(self.player_form)
        self.player_section.setVisible(False)
        layout.addWidget(self.player_section)

        self.driver_label = QLabel("—", container)
        self.driver_label.setWordWrap(True)
        self.driver_label.setProperty("role", "hint")
        layout.addWidget(self.driver_label)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(container)
        self.scroll_area = scroll
        self.setWidget(scroll)
        self.retranslate()

    # ------------------------------------------------------------------ #
    # Локализация
    # ------------------------------------------------------------------ #
    def retranslate(self, *_args: object) -> None:
        """Перевести подписи метрик."""
        super().retranslate()
        self.scene_section.set_title(self.tr("stats.scene"))
        self.render_section.set_title(self.tr("stats.render"))
        self.camera_section.set_title(self.tr("stats.camera"))
        self.player_section.set_title(self.tr("play.player_stats"))
        for row, (_key, caption) in enumerate(PLAYER_FIELDS):
            self._set_row_label(self.player_form, row, self.tr(caption))
        for row, (key, _attr) in enumerate(SCENE_FIELDS):
            self._set_row_label(self.scene_form, row, self.tr(key))
        for row, (key, _attr) in enumerate(RENDER_FIELDS):
            self._set_row_label(self.render_form, row, self.tr(key))
        self._set_row_label(self.render_form, len(RENDER_FIELDS), self.tr("stats.fps"))
        self._set_row_label(self.camera_form, 0, self.tr("stats.camera_position"))
        self._set_row_label(self.camera_form, 1, self.tr("stats.camera_distance"))

    def set_player_stats(self, stats: dict | None) -> None:
        """Показать статистику внешнего плеера (``None`` — скрыть секцию).

        Args:
            stats: Событие ``stats`` плеера: ``fps``, ``dc``, ``tri``, ``script_ms``.
        """
        self.player_section.setVisible(stats is not None)
        if stats is None:
            return
        for key, _caption in PLAYER_FIELDS:
            value = stats.get(key, "—")
            self.player_labels[key].setText(
                f"{value:.1f}" if isinstance(value, float) else str(value)
            )

    @staticmethod
    def _set_row_label(form: QFormLayout, row: int, text: str) -> None:
        """Заменить подпись строки формы."""
        item = form.itemAt(row, QFormLayout.LabelRole)
        if item is not None and item.widget() is not None:
            item.widget().setText(text)

    # ------------------------------------------------------------------ #
    # Данные
    # ------------------------------------------------------------------ #
    def update_stats(
        self,
        scene_stats: object,
        render_stats: object | None = None,
        *,
        fps: float = 0.0,
        camera_position: str = "—",
        camera_distance: str = "—",
        driver: str = "",
    ) -> None:
        """Обновить показатели панели.

        Args:
            scene_stats: :class:`~engine.core.scene.SceneStats`.
            render_stats: :class:`~engine.rendering.renderer.RenderStats` или ``None``.
            fps: Сглаженный FPS вьюпорта.
            camera_position: Текстовая позиция камеры.
            camera_distance: Текстовая дистанция камеры до цели.
            driver: Строка с описанием драйвера.
        """
        for key, attr in SCENE_FIELDS:
            self.scene_labels[key].setText(str(getattr(scene_stats, attr, 0)))
        if render_stats is not None:
            for key, attr in RENDER_FIELDS:
                self.render_labels[key].setText(str(getattr(render_stats, attr, 0)))
        self.fps_label.setText(f"{fps:.0f}")
        self.camera_position.setText(camera_position)
        self.camera_distance.setText(camera_distance)
        if driver:
            self.driver_label.setText(f"{self.tr('stats.driver')}: {driver}")
