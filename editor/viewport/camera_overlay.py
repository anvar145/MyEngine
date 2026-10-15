"""Оверлей предпросмотра камеры: маленькое окно в углу вьюпорта.

Показывает, что видит выбранная игровая камера, прямо в окне сцены — не нужно
открывать отдельную панель. Виджет прозрачен для мыши: орбита, панорама и
выбор объектов работают и под ним.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

__all__ = ["CameraPreviewOverlay", "OVERLAY_SIZE"]

#: Размер изображения предпросмотра (ширина, высота) в пикселях.
OVERLAY_SIZE = (240, 135)

#: Отступ окна от правого нижнего угла вьюпорта.
OVERLAY_MARGIN = 12


class CameraPreviewOverlay(QFrame):
    """Окно предпросмотра камеры в углу вьюпорта.

    Args:
        parent: Виджет, поверх которого рисуется оверлей (вьюпорт).
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("cameraPreviewOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setStyleSheet(
            "#cameraPreviewOverlay { background: rgba(20, 19, 18, 235);"
            " border: 1px solid #d97757; border-radius: 8px; }"
            "#cameraPreviewOverlay QLabel { color: #e8e4de; background: transparent; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)
        self.title = QLabel(self)
        layout.addWidget(self.title)
        self.image = QLabel(self)
        self.image.setFixedSize(*OVERLAY_SIZE)
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image)
        self.adjustSize()
        self.hide()

    def set_title(self, text: str) -> None:
        """Подпись окна (имя камеры)."""
        self.title.setText(text)

    def set_pixels(self, pixels: np.ndarray) -> None:
        """Показать кадр предпросмотра (OpenGL отдаёт строки снизу вверх)."""
        array = np.ascontiguousarray(np.flipud(pixels))
        height, width = array.shape[:2]
        image = QImage(
            array.data, width, height, width * 4, QImage.Format.Format_RGBA8888
        ).copy()
        self.image.setPixmap(QPixmap.fromImage(image))

    def reposition(self) -> None:
        """Прижать окно к правому нижнему углу родителя."""
        parent = self.parentWidget()
        if parent is None:
            return
        self.adjustSize()
        self.move(
            max(0, parent.width() - self.width() - OVERLAY_MARGIN),
            max(0, parent.height() - self.height() - OVERLAY_MARGIN),
        )
        self.raise_()
