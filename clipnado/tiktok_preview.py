"""Optional, approximate TikTok chrome for the export preview only."""

from PySide6.QtCore import QRectF, QSizeF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap

from .preview import VideoPreview


class TikTokPreview(VideoPreview):
    """Keep the decoded export frame intact; draw guides on a separate pixmap."""

    def __init__(self, parent=None):
        self.guides_enabled = False
        super().__init__(parent)

    def set_guides_enabled(self, enabled):
        self.guides_enabled = bool(enabled)
        self._draw()

    def _draw(self):
        if not self.guides_enabled or self.image.isNull():
            super()._draw()
            return
        size = self.contentsRect().size()
        if size.isEmpty():
            return
        canvas = QPixmap(size)
        canvas.fill(Qt.black)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        # Simulate a portrait viewport even for square/landscape exports.
        scale = min(size.width() / 1080, size.height() / 1920)
        painter.translate((size.width() - 1080 * scale) / 2,
                          (size.height() - 1920 * scale) / 2)
        painter.scale(scale, scale)
        painter.setClipRect(QRectF(0, 0, 1080, 1920))
        frame = QSizeF(self.image.size()).scaled(1080, 1920, Qt.KeepAspectRatio)
        painter.drawImage(QRectF((1080 - frame.width()) / 2,
                                 (1920 - frame.height()) / 2,
                                 frame.width(), frame.height()), self.image)
        self._paint_chrome(painter)
        painter.end()
        self.setPixmap(canvas)

    @staticmethod
    def _paint_chrome(painter):
        # These are illustrative areas, not platform specifications or a guarantee
        # of safety. Real chrome moves with the device, description and app mode.
        zones = (QRectF(0, 0, 1080, 230), QRectF(900, 650, 180, 920),
                 QRectF(0, 1510, 1080, 410))
        for zone in zones:
            painter.fillRect(zone, QColor(18, 21, 29, 125))
        painter.setPen(QPen(QColor(255, 202, 103, 210), 3, Qt.DashLine))
        for zone in zones:
            painter.drawRect(zone.adjusted(2, 2, -2, -2))

        def text(x, y, value, pixels=36, bold=False):
            font = QFont("Segoe UI")
            font.setPixelSize(pixels)
            font.setBold(bold)
            painter.setFont(font)
            painter.setPen(Qt.white)
            painter.drawText(QRectF(x, y, 850, pixels * 1.5), Qt.AlignLeft, value)

        text(45, 105, "Подписки     Рекомендации", 42, True)
        painter.setPen(QPen(Qt.white, 5))
        painter.drawEllipse(QRectF(970, 115, 40, 40))
        painter.drawLine(1005, 150, 1025, 170)

        painter.drawEllipse(QRectF(942, 685, 90, 90))
        painter.drawLine(987, 708, 987, 752)
        painter.drawLine(965, 730, 1009, 730)

        heart = QPainterPath()
        heart.moveTo(987, 945)
        heart.cubicTo(905, 892, 939, 834, 987, 874)
        heart.cubicTo(1035, 834, 1069, 892, 987, 945)
        painter.fillPath(heart, Qt.white)
        text(945, 962, "Лайки", 27)

        painter.setPen(QPen(Qt.white, 6))
        painter.drawRoundedRect(QRectF(948, 1080, 80, 60), 16, 16)
        painter.drawLine(965, 1140, 954, 1160)
        text(945, 1170, "Ответы", 27)
        bookmark = QPainterPath()
        bookmark.moveTo(958, 1280)
        bookmark.lineTo(1018, 1280)
        bookmark.lineTo(1018, 1350)
        bookmark.lineTo(988, 1330)
        bookmark.lineTo(958, 1350)
        bookmark.closeSubpath()
        painter.fillPath(bookmark, Qt.white)
        text(930, 1380, "Избранное", 25)
        share = QPainterPath()
        share.moveTo(945, 1510)
        share.quadTo(950, 1450, 1000, 1460)
        share.lineTo(1000, 1430)
        share.lineTo(1040, 1480)
        share.lineTo(1000, 1520)
        share.lineTo(1000, 1490)
        share.quadTo(970, 1480, 945, 1510)
        painter.fillPath(share, Qt.white)
        text(925, 1530, "Поделиться", 23)
        text(48, 1580, "@ваш_канал", 42, True)
        text(48, 1645, "Описание видео и хэштеги…", 36)
        text(48, 1710, "Звук видео", 32)
        painter.setPen(QPen(QColor(255, 255, 255, 150), 2))
        painter.drawLine(0, 1810, 1080, 1810)
        text(48, 1840, "Главная       Поиск       +       Входящие       Профиль", 28)
