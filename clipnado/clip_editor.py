"""Source-time editing, independent of automatic search duration limits."""
from __future__ import annotations

import copy
from PySide6.QtCore import Qt, Slot
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
                              QLabel, QLineEdit, QPushButton, QSlider, QVBoxLayout)

from .models import parse_time, timecode
from .preview import VideoPreview

MANUAL_NOTE = "Границы изменены вручную; оценка относится к автоматическому выбору."


def edited_candidate(media, segments, candidate, start_ms, end_ms):
    """Validate a copy before applying both boundaries atomically."""
    result = copy.deepcopy(candidate)
    result.start_ms, result.end_ms = start_ms, end_ms
    result.validate(media)
    if (start_ms, end_ms) != (candidate.start_ms, candidate.end_ms):
        result.transcript_excerpt = " ".join(
            s.text for s in segments if s.start_ms < end_ms and s.end_ms > start_ms)[:2000]
        if MANUAL_NOTE not in result.reason:
            result.reason += " · " + MANUAL_NOTE
    return result


def editor_time(value):
    # Also accept the short source notation used in the editor: 08:20.
    value = value.strip()
    return parse_time("00:" + value if value.count(":") == 1 else value)


class ClipEditorDialog(QDialog):
    def __init__(self, parent, media, candidate, player):
        super().__init__(parent)
        self.media, self.original = copy.deepcopy(media), copy.deepcopy(candidate)
        self.preview_end = None
        self.pending_play = None
        self.stopped = False
        self.setWindowTitle("Изменить отрезок · исходное видео")
        self.resize(960, 700)
        layout = QVBoxLayout(self)
        label = QLabel("Посмотрите соседние секунды и выберите начало и конец. "
                       "Можно расширять и сокращать клип в пределах всего исходника.")
        label.setWordWrap(True)
        layout.addWidget(label)
        self.video = VideoPreview()
        layout.addWidget(self.video, 1)
        # Borrow the source player. Opening/closing an editor must not create
        # and destroy another FFmpeg decoder while frames are being delivered.
        self.player = player
        self.sink = player.videoSink()
        self.sink.videoFrameChanged.connect(self.video.set_frame, Qt.QueuedConnection)
        self.player.positionChanged.connect(self.position_changed, Qt.QueuedConnection)
        self.player.mediaStatusChanged.connect(self.media_status_changed, Qt.QueuedConnection)
        self.player.errorOccurred.connect(self.media_error, Qt.QueuedConnection)

        transport = QHBoxLayout()
        for text, action in (("▶ Отрезок", self.play_clip), ("▶ Соседние ±10 с", self.play_context),
                             ("⏸ Пауза", self.player.pause),
                             ("−1 с", lambda: self.seek(self.player.position()-1000)),
                             ("+1 с", lambda: self.seek(self.player.position()+1000))):
            b = QPushButton(text); b.clicked.connect(action); transport.addWidget(b)
        layout.addLayout(transport)
        row = QHBoxLayout()
        self.position = QSlider(Qt.Horizontal)
        self.position.setRange(0, media.duration_ms)
        self.position.sliderMoved.connect(self.seek)
        row.addWidget(self.position, 1)
        self.clock = QLabel(timecode(candidate.start_ms))
        row.addWidget(self.clock)
        layout.addLayout(row)
        form = QFormLayout()
        self.start = QLineEdit(timecode(candidate.start_ms))
        self.end = QLineEdit(timecode(candidate.end_ms))
        for label, field, mark in (("Начало", self.start, "начало"), ("Конец", self.end, "конец")):
            field.setPlaceholderText("08:20 или 00:08:20.000")
            line = QHBoxLayout(); line.addWidget(field)
            b = QPushButton("Текущий кадр → " + mark)
            b.clicked.connect(lambda _checked=False, target=field: target.setText(timecode(self.player.position())))
            line.addWidget(b); form.addRow(label, line)
        layout.addLayout(form)
        self.duration = QLabel()
        layout.addWidget(self.duration)
        self.error = QLabel(); self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #ffb7a1;")
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel | QDialogButtonBox.Reset)
        self.apply_button = buttons.button(QDialogButtonBox.Apply)
        self.apply_button.setText("Применить границы")
        self.apply_button.clicked.connect(self.accept)
        buttons.button(QDialogButtonBox.Cancel).setText("Отмена")
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Reset).setText("Вернуть найденный отрезок")
        buttons.button(QDialogButtonBox.Reset).clicked.connect(self.reset)
        layout.addWidget(buttons)
        self.start.textChanged.connect(self.validate_draft)
        self.end.textChanged.connect(self.validate_draft)
        # Release subscriptions after QDialog's native close operation. The
        # caller also releases them immediately after the modal loop returns.
        self.finished.connect(self.stop, Qt.QueuedConnection)
        self.validate_draft()
        self.play_clip()

    def bounds(self):
        start, end = editor_time(self.start.text()), editor_time(self.end.text())
        if not 0 <= start < end <= self.media.duration_ms:
            raise ValueError("Нужно: 0 ≤ начало < конец ≤ длительности исходника (" + timecode(self.media.duration_ms) + ").")
        return start, end

    def validate_draft(self, *_):
        self.preview_end = None
        self.pending_play = None
        self.player.pause()
        try:
            start, end = self.bounds()
            self.duration.setText(f"Длительность: {(end-start)/1000:g} с · найдено: "
                                  f"{timecode(self.original.start_ms)}–{timecode(self.original.end_ms)}")
            self.error.clear(); self.apply_button.setEnabled(True)
        except ValueError as exc:
            self.error.setText(str(exc)); self.apply_button.setEnabled(False)

    def reset(self):
        self.start.setText(timecode(self.original.start_ms))
        self.end.setText(timecode(self.original.end_ms))

    def play_range(self, start, end):
        self.preview_end = end
        self.pending_play = (start, end)
        self.media_status_changed()

    @Slot()
    def media_status_changed(self):
        status = self.player.mediaStatus()
        if self.pending_play and status in (QMediaPlayer.LoadedMedia, QMediaPlayer.BufferedMedia, QMediaPlayer.EndOfMedia):
            start, self.preview_end = self.pending_play
            self.pending_play = None
            self.player.setPosition(start)
            self.player.play()

    def play_clip(self):
        try:
            self.play_range(*self.bounds())
        except ValueError as exc:
            self.error.setText(str(exc))

    def play_context(self):
        try:
            start, end = self.bounds()
            self.play_range(max(0, start-10000), min(self.media.duration_ms, end+10000))
        except ValueError as exc:
            self.error.setText(str(exc))

    def seek(self, at):
        self.pending_play = None
        self.preview_end = None
        self.player.setPosition(max(0, min(at, self.media.duration_ms)))

    @Slot(int)
    def position_changed(self, at):
        if not self.position.isSliderDown():
            self.position.setValue(at)
        self.clock.setText(timecode(at))
        if self.preview_end is not None and at >= self.preview_end:
            self.preview_end = None
            self.player.pause()

    def accept(self):
        try:
            self.bounds()
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        super().accept()

    @Slot()
    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        self.preview_end = None
        self.pending_play = None
        self.player.pause()
        self.sink.videoFrameChanged.disconnect(self.video.set_frame)
        self.player.positionChanged.disconnect(self.position_changed)
        self.player.mediaStatusChanged.disconnect(self.media_status_changed)
        self.player.errorOccurred.disconnect(self.media_error)

    @Slot()
    def media_error(self):
        self.error.setText("Не удалось воспроизвести исходник: " + self.player.errorString())
