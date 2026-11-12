from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel


class VideoPreview(QLabel):
    """Render decoded pixels without a native video surface or GPU presentation."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QImage()
        self.frame_count = 0
        self._last_frame = 0.0
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(160, 200)
        self.setStyleSheet("background: black; color: #9aafbd;")
        self.setText("Выберите видео — здесь появится кадр исходника")

    def clear_image(self):
        self.image = QImage()
        self.frame_count = 0
        self._last_frame = 0
        self.clear()
        self.setText("Загрузка кадра…")

    def set_image(self, image: QImage):
        if image.isNull():
            return
        self.image = image.copy()
        self._draw()

    def set_frame(self, frame):
        now = time.monotonic()
        if not frame.isValid() or now - self._last_frame < 1 / 24:
            return
        image = frame.toImage()
        if not image.isNull():
            self.frame_count += 1
            self._last_frame = now
            self.set_image(image)

    def _draw(self):
        if not self.image.isNull():
            self.setPixmap(QPixmap.fromImage(self.image).scaled(self.contentsRect().size(), Qt.KeepAspectRatio,
                                                              Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._draw()
