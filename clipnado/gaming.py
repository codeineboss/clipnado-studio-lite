"""Select two source regions once and preview a vertical gaming composition."""

from __future__ import annotations

from pathlib import Path
import tempfile

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .models import crop_region
from .preview import VideoPreview
from .process import Cancellation, run
from .workers import Worker

DEFAULT_CAMERA = [0.76, 0.0, 0.24, 0.31]
FULL_FRAME = [0.0, 0.0, 1.0, 1.0]


def source_frame(media, at_ms, tools, token):
    """Decode the original display geometry, never an already exported crop."""
    at_ms = max(0, min(at_ms, media.duration_ms - 1))
    previous = max(0, at_ms - max(100, 1000 / media.fps if media.fps else 1000))
    with tempfile.TemporaryDirectory(prefix="clipnado-gaming-frame-") as tmp:
        path = Path(tmp) / "frame.png"
        for at in (at_ms, previous):
            run(
                [
                    tools.ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{at/1000:.3f}",
                    "-i",
                    media.path,
                    "-an",
                    "-sn",
                    "-dn",
                    "-map",
                    "0:v:0",
                    "-frames:v",
                    "1",
                    "-filter_threads",
                    str(tools.threads),
                    "-vf",
                    "scale=trunc(iw*sar/2)*2:ih,setsar=1,scale=1280:720:force_original_aspect_ratio=decrease:force_divisible_by=2",
                    "-threads",
                    str(tools.threads),
                    str(path),
                ],
                token,
                timeout=60,
            )
            if path.is_file():
                return path.read_bytes()
    raise ValueError("В выбранном месте нет кадра. Выберите другое время.")


def trim_black_bars(image):
    """Suggest only narrow black bars; center columns exclude corner webcams."""
    image = image.convertToFormat(QImage.Format_RGB888)
    w, h = image.width(), image.height()
    raw, stride = image.constBits(), image.bytesPerLine()
    limit = max(1, int(h * 0.15))

    def dark(y):
        return (
            max(
                raw[y * stride + int(w * 0.25) * 3 : y * stride + int(w * 0.75) * 3],
                default=0,
            )
            < 18
        )

    top = bottom = 0
    while top < limit and dark(top):
        top += 1
    while bottom < limit and dark(h - 1 - bottom):
        bottom += 1
    # A black intro/scene is not evidence of a letterbox boundary.
    if top == limit:
        top = 0
    if bottom == limit:
        bottom = 0
    return [0.0, top / h, 1.0, (h - top - bottom) / h]


def region_pixels(rect, width, height):
    x, y, w, h = crop_region(rect)
    left, top = min(width - 1, int(x * width)), min(height - 1, int(y * height))
    right, bottom = min(width, max(left + 1, int((x + w) * width))), min(
        height, max(top + 1, int((y + h) * height))
    )
    return left, top, right - left, bottom - top


def gaming_preview(image, camera, game, fraction, width=270, height=480):
    result = QImage(width, height, QImage.Format_RGB32)
    result.fill(Qt.black)
    painter = QPainter(result)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)
    top = int(height * fraction) // 2 * 2
    for region, target in (
        (camera, QRectF(0, 0, width, top)),
        (game, QRectF(0, top, width, height - top)),
    ):
        cropped = image.copy(*region_pixels(region, image.width(), image.height()))
        scaled = cropped.scaled(
            int(target.width()),
            int(target.height()),
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation,
        )
        painter.save()
        painter.setClipRect(target)
        painter.drawImage(
            QPointF(
                target.x() + (target.width() - scaled.width()) / 2,
                target.y() + (target.height() - scaled.height()) / 2,
            ),
            scaled,
        )
        painter.restore()
    painter.end()
    return result


class RegionCanvas(QWidget):
    changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QImage()
        self.camera, self.game = DEFAULT_CAMERA.copy(), FULL_FRAME.copy()
        self.target = 0
        self.drag = None
        self.setMinimumSize(400, 260)
        self.setMouseTracking(True)

    def sizeHint(self):
        return QSize(760, 430)

    def image_rect(self):
        if self.image.isNull():
            return QRectF()
        scale = min(
            self.width() / self.image.width(), self.height() / self.image.height()
        )
        w, h = self.image.width() * scale, self.image.height() * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def normalized(self, point):
        r = self.image_rect()
        return QPointF(
            max(0.0, min(1.0, (point.x() - r.x()) / r.width())),
            max(0.0, min(1.0, (point.y() - r.y()) / r.height())),
        )

    def active(self):
        return self.camera if self.target == 0 else self.game

    def set_active(self, rect):
        rect = crop_region(rect)
        if self.target == 0:
            self.camera = rect
        else:
            self.game = rect
        self.changed.emit(rect)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#080d12"))
        r = self.image_rect()
        if r.isEmpty():
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "Подготовка исходного кадра…")
            return
        painter.drawImage(r, self.image)
        for index, (rect, color, name) in enumerate(
            ((self.camera, "#65cab7", "КАМЕРА"), (self.game, "#f0bf67", "ИГРА"))
        ):
            x, y, w, h = rect
            box = QRectF(
                r.x() + x * r.width(),
                r.y() + y * r.height(),
                w * r.width(),
                h * r.height(),
            )
            painter.setPen(QPen(QColor(color), 3 if index == self.target else 1))
            fill = QColor(color)
            fill.setAlpha(35 if index == self.target else 10)
            painter.setBrush(fill)
            painter.drawRect(box)
            painter.drawText(
                box.adjusted(6, 5, -2, -2), Qt.AlignTop | Qt.AlignLeft, name
            )
            if index == self.target:
                painter.setBrush(QColor(color))
                for point in (
                    box.topLeft(),
                    box.topRight(),
                    box.bottomLeft(),
                    box.bottomRight(),
                ):
                    painter.drawRect(QRectF(point.x() - 4, point.y() - 4, 8, 8))

    def mousePressEvent(self, event):
        if (
            event.button() != Qt.LeftButton
            or self.image.isNull()
            or not self.image_rect().contains(event.position())
        ):
            return
        point = self.normalized(event.position())
        x, y, w, h = self.active()
        r = self.image_rect()
        corners = [
            (x, y, x + w, y + h),
            (x + w, y, x, y + h),
            (x, y + h, x + w, y),
            (x + w, y + h, x, y),
        ]
        for cx, cy, ax, ay in corners:
            if (
                abs(point.x() - cx) * r.width() <= 10
                and abs(point.y() - cy) * r.height() <= 10
            ):
                self.drag = ("draw", QPointF(ax, ay), None)
                return
        if x <= point.x() <= x + w and y <= point.y() <= y + h:
            self.drag = ("move", point, self.active().copy())
        else:
            self.drag = ("draw", point, None)

    def mouseMoveEvent(self, event):
        if not self.drag or self.image.isNull():
            return
        point = self.normalized(event.position())
        kind, anchor, rect = self.drag
        if kind == "move":
            x, y, w, h = rect
            self.set_active(
                [
                    max(0.0, min(1 - w, x + point.x() - anchor.x())),
                    max(0.0, min(1 - h, y + point.y() - anchor.y())),
                    w,
                    h,
                ]
            )
        else:
            left, right = sorted((point.x(), anchor.x()))
            top, bottom = sorted((point.y(), anchor.y()))
            if right - left >= 0.01 and bottom - top >= 0.01:
                self.set_active([left, top, right - left, bottom - top])

    def mouseReleaseEvent(self, event):
        self.drag = None


class GamingRegionsDialog(QDialog):
    def __init__(self, media, tools, settings, at_ms=0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Игровая нарезка · камера и игра")
        self.media, self.tools = media, tools
        self.closed = False
        self.worker = None
        self.token = None
        self.loading_numbers = False
        self.suggest_game = (
            settings.gaming_camera is None and settings.gaming_game == FULL_FRAME
        )
        outer = QVBoxLayout(self)
        hint = QLabel(
            "Выделите камеру стримера и область игры на исходнике. Перетаскивайте рамку и её углы или задавайте проценты.\nРазметка сохранится для всех клипов этого видео. Внизу игры можно выбрать центральную область, исключив веб-камеру и чёрные полосы."
        )
        hint.setWordWrap(True)
        outer.addWidget(hint)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        body = QHBoxLayout(content)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)
        left = QVBoxLayout()
        body.addLayout(left, 3)
        self.canvas = RegionCanvas()
        self.canvas.camera = list(settings.gaming_camera or DEFAULT_CAMERA)
        self.canvas.game = list(settings.gaming_game)
        left.addWidget(self.canvas, 1)
        self.target = QComboBox()
        self.target.addItems(["Камера стримера", "Игра"])
        left.addWidget(self.target)
        numbers = QHBoxLayout()
        left.addLayout(numbers)
        self.fields = []
        for label, minimum, maximum in (
            ("X, %", 0, 99),
            ("Y, %", 0, 99),
            ("Ширина, %", 1, 100),
            ("Высота, %", 1, 100),
        ):
            numbers.addWidget(QLabel(label))
            control = QDoubleSpinBox()
            control.setRange(minimum, maximum)
            control.setDecimals(2)
            numbers.addWidget(control)
            self.fields.append(control)
            control.valueChanged.connect(self.from_numbers)
        presets = QHBoxLayout()
        left.addLayout(presets)
        for label, rect in [
            ("Камера справа сверху", DEFAULT_CAMERA),
            ("Камера слева сверху", [0.0, 0.0, 0.24, 0.31]),
        ]:
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, r=rect: self.camera_preset(r))
            presets.addWidget(b)
        reset = QPushButton("Игра: исходный кадр без чёрных полос")
        reset.clicked.connect(self.reset_game)
        left.addWidget(reset)
        seek = QHBoxLayout()
        left.addLayout(seek)
        seek.addWidget(QLabel("Другой кадр, сек:"))
        self.position = QDoubleSpinBox()
        self.position.setRange(0, max(0, (media.duration_ms - 1) / 1000))
        self.position.setDecimals(3)
        self.position.setValue(at_ms / 1000)
        seek.addWidget(self.position)
        self.refresh = QPushButton("Показать кадр")
        self.refresh.clicked.connect(self.request_frame)
        seek.addWidget(self.refresh)
        right = QVBoxLayout()
        body.addLayout(right, 1)
        right.addWidget(QLabel("Компоновка 9:16"))
        self.preview = VideoPreview()
        self.preview.setFixedSize(270, 480)
        right.addWidget(self.preview, 0, Qt.AlignHCenter | Qt.AlignTop)
        right.addStretch(1)
        self.fraction = QSpinBox()
        self.fraction.setRange(25, 60)
        self.fraction.setSuffix("%")
        self.fraction.setValue(round(settings.gaming_fraction * 100))
        form = QFormLayout()
        form.addRow("Доля камеры сверху", self.fraction)
        right.addLayout(form)
        self.status = QLabel()
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Применить к видео")
        self.buttons.button(QDialogButtonBox.Cancel).setText("Отмена")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        outer.addWidget(self.buttons)
        self.target.currentIndexChanged.connect(self.target_changed)
        self.canvas.changed.connect(self.regions_changed)
        self.fraction.valueChanged.connect(self.update_preview)
        self.target_changed()
        self.resize(1150, 720)
        if parent:
            self.resize(
                min(1150, parent.screen().availableGeometry().width() - 50),
                min(760, parent.screen().availableGeometry().height() - 60),
            )
        self.request_frame()

    def target_changed(self, *_):
        self.canvas.target = self.target.currentIndex()
        self.canvas.update()
        self.regions_changed()

    def regions_changed(self, *_):
        self.loading_numbers = True
        for field, value in zip(self.fields, self.canvas.active()):
            field.setValue(value * 100)
        self.loading_numbers = False
        self.update_preview()

    def from_numbers(self, *_):
        if self.loading_numbers:
            return
        x, y, w, h = [control.value() / 100 for control in self.fields]
        self.canvas.set_active([min(x, 1 - w), min(y, 1 - h), w, h])

    def camera_preset(self, rect):
        self.target.setCurrentIndex(0)
        self.canvas.set_active(list(rect))

    def reset_game(self):
        if self.canvas.image.isNull():
            return
        self.target.setCurrentIndex(1)
        self.canvas.set_active(trim_black_bars(self.canvas.image))

    def update_preview(self, *_):
        if self.canvas.image.isNull():
            return
        self.preview.set_image(
            gaming_preview(
                self.canvas.image,
                self.canvas.camera,
                self.canvas.game,
                self.fraction.value() / 100,
            )
        )

    def request_frame(self, *_):
        if self.worker or self.closed:
            return
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self.refresh.setEnabled(False)
        self.status.setText("Подготовка исходного кадра…")
        self.token = Cancellation()
        at_ms = round(self.position.value() * 1000)
        self.worker = Worker(
            lambda token, progress: source_frame(self.media, at_ms, self.tools, token),
            self.token,
            self,
        )
        self.worker.result.connect(self.frame_ready)
        self.worker.error.connect(
            lambda text: self.status.setText(text) if not self.closed else None
        )
        self.worker.finished.connect(self.frame_finished)
        self.worker.start()

    def frame_ready(self, data):
        if self.closed:
            return
        image = QImage.fromData(data)
        if image.isNull():
            self.status.setText("Не удалось прочитать кадр. Выберите другое время.")
            return
        self.canvas.image = image
        if self.suggest_game:
            self.canvas.game = trim_black_bars(image)
            self.suggest_game = False
        self.canvas.update()
        self.regions_changed()
        self.status.setText(
            "Бирюзовая рамка — камера, жёлтая — игра. Обе части используют один исходный кадр и один звук."
        )
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(True)

    def frame_finished(self):
        if self.worker:
            self.worker.wait()
            self.worker.deleteLater()
            self.worker = None
        if not self.closed:
            self.refresh.setEnabled(True)

    def regions(self):
        return (
            list(self.canvas.camera),
            list(self.canvas.game),
            self.fraction.value() / 100,
        )

    def done(self, result):
        self.closed = True
        if self.token:
            self.token.cancel()
        if self.worker:
            self.worker.wait()
        super().done(result)
