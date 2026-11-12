from __future__ import annotations

import copy
import os
import sys
import tempfile
from pathlib import Path

# Use software decoding for the pixel preview; it avoids black native D3D surfaces
# on older drivers. The source and FFmpeg exports are unaffected.
os.environ.setdefault("QT_FFMPEG_DECODING_HW_DEVICE_TYPES", "")

from PySide6.QtCore import QLibraryInfo, QTimer, Qt, QTranslator, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QImage
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoSink
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea,
    QSlider, QSpinBox, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem,
    QTextEdit, QVBoxLayout, QWidget,
)

from . import __version__
from .analysis import LocalFallbackProvider, recognize
from .export import export_clip
from .media import Tools, probe, thumbnail
from .models import Candidate, ExportSettings, Project, SearchSettings, Segment, parse_time, timecode
from .process import Cancellation, Cancelled, run
from .preview import VideoPreview
from .projects import load_project, save_project
from .providers import GeminiFreeProvider, MODELS
from .settings import data_root, load_key, read_settings, restore_tools, save_key, write_settings
from .subtitles import from_srt
from .workers import Worker

STYLE = """
QWidget { background: #111820; color: #e7edf3; font-family: 'Segoe UI', 'DejaVu Sans'; font-size: 13px; }
QMainWindow { background: #111820; }
QGroupBox { border: 1px solid #303f4d; border-radius: 9px; margin-top: 17px; padding: 13px 10px 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; color: #86d5c3; }
QPushButton { background: #22313e; border: 1px solid #3a4b5a; padding: 8px 13px; border-radius: 6px; }
QPushButton:hover { background: #304554; border-color: #78cbb9; }
QPushButton:disabled { color: #64717e; background: #19222c; }
QPushButton#primary { background: #66c7ac; color: #08251e; font-weight: 700; border: none; padding: 11px; }
QPushButton#primary:hover { background: #8de1c9; }
QPushButton#primary:disabled { background: #263b37; color: #86a398; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit { background: #19242f; border: 1px solid #384958; border-radius: 4px; padding: 5px; selection-background-color: #397965; }
QComboBox QAbstractItemView { background: #19242f; selection-background-color: #397965; }
QTableWidget { background: #151f29; alternate-background-color: #1b2732; border: 1px solid #303f4d; gridline-color: #2a3846; selection-background-color: #294f4c; }
QHeaderView::section { background: #22313e; border: none; padding: 9px 5px; font-weight: 600; }
QTabWidget::pane { border: 1px solid #303f4d; }
QTabBar::tab { background: #1c2935; padding: 10px 15px; }
QTabBar::tab:selected { background: #304850; color: #a0eed9; }
QProgressBar { background: #19242f; border: 1px solid #384958; border-radius: 4px; text-align: center; height: 18px; }
QProgressBar::chunk { background: #58aa95; }
QLabel#muted { color: #9aafbd; }
QLabel#title { font-size: 25px; font-weight: 700; }
QLabel#notice { background: #27322c; color: #c5d6ba; padding: 10px; border-radius: 6px; }
QSplitter::handle { background: #263542; width: 5px; }
QToolTip { background: #22313e; color: white; border: 1px solid #78cbb9; }
"""


def button(label, callback, primary=False):
    result = QPushButton(label)
    if primary:
        result.setObjectName("primary")
    result.clicked.connect(callback)
    return result


def spin(low, high, value):
    result = QSpinBox()
    result.setRange(low, high)
    result.setValue(value)
    return result


def combo(values, default=None):
    result = QComboBox()
    result.addItems(values)
    if default is not None:
        result.setCurrentText(str(default))
    return result


def note(text):
    result = QLabel(text)
    result.setWordWrap(True)
    result.setObjectName("muted")
    return result


class SettingsDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Инструменты и хранение")
        self.resize(760, 540)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.fields = {}
        for name, label, value, folder in [
            ("ffmpeg", "FFmpeg", parent.tools.ffmpeg, False),
            ("ffprobe", "ffprobe", parent.tools.ffprobe, False),
            ("whisper", "whisper.cpp CLI", parent.tools.whisper, False),
            ("model", "Модель Whisper (.bin)", parent.tools.model, False),
            ("cache", "Кэш", str(parent.cache), True),
            ("output", "Результаты", str(parent.output), True),
        ]:
            line = QLineEdit(value)
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(line)
            def choose(_=False, field=line, directory=folder):
                path = QFileDialog.getExistingDirectory(self, "Выберите папку", field.text()) if directory else QFileDialog.getOpenFileName(self, "Выберите файл", field.text())[0]
                if path:
                    field.setText(path)
            row_layout.addWidget(button("…", choose))
            form.addRow(label, row)
            self.fields[name] = line
        self.threads = spin(1, 32, parent.tools.threads)
        self.gpu = QCheckBox("CUDA — только для совместимой CUDA-сборки whisper.cpp")
        self.gpu.setChecked(parent.tools.gpu)
        form.addRow("Потоки CPU", self.threads)
        form.addRow(self.gpu)
        layout.addLayout(form)
        layout.addWidget(note("CPU включён по умолчанию. Для Xeon без AVX2 используйте комплектный Whisper или scripts/build_windows.ps1; обычные готовые сборки могут требовать AVX2. GTX 1660 требует отдельной CUDA-сборки. Настройки распознавания применяются к новым проектам; сохранённый текст можно очистить вручную."))
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Clipnado Studio Lite · {__version__}")
        self.resize(1460, 940)
        self.setMinimumSize(1080, 720)
        settings = read_settings()
        self.tools = restore_tools(settings.get("tools"))
        self.cache = Path(settings.get("cache") or str(data_root() / "cache"))
        self.output = Path(settings.get("output") or str(Path.home() / "Videos" / "Clipnado"))
        self.project = None
        self.project_path = None
        self.dirty = False
        self.task_thread = None
        self.token = None
        self.worker = None
        self.preview_end = None
        self.poster_jobs = []
        self.poster_serial = 0
        self.repopulating = False
        self.visible_count = 15
        self.api_key = load_key()
        self.build_ui()
        self.model.setCurrentText(settings.get("model", MODELS[0]))
        self.connect_player()
        recovery = data_root() / "recovery.clipnado.json"
        if recovery.exists():
            QTimer.singleShot(300, lambda: self.offer_recovery(recovery))

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(22, 18, 22, 16)
        header = QHBoxLayout()
        title = QLabel("CLIPNADO  /  STUDIO LITE")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()
        self.open_button = button("Открыть проект", self.open_project)
        self.save_button = button("Сохранить проект", self.save)
        self.settings_button = button("Настройки", self.settings)
        for widget in (self.open_button, self.save_button, self.settings_button):
            header.addWidget(widget)
        root.addLayout(header)
        root.addWidget(note("Длинное видео → цельные эпизоды → заготовки для CapCut. Рейтинг — внутренняя оценка, не прогноз просмотров."))
        source = QHBoxLayout()
        self.import_button = button("＋ Выбрать видео", self.import_media, True)
        source.addWidget(self.import_button)
        self.media_label = QLabel("Выберите локальное видео, чтобы начать")
        self.media_label.setWordWrap(True)
        source.addWidget(self.media_label, 1)
        self.relink_button = button("Перепривязать видео", self.relink)
        source.addWidget(self.relink_button)
        root.addLayout(source)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 8, 0)
        search = QGroupBox("01   Поиск эпизодов")
        form = QFormLayout(search)
        self.count = spin(1, 100, 15)
        self.minimum = spin(1, 600, 60)
        self.maximum = spin(1, 600, 90)
        self.sensitivity = spin(0, 100, 55)
        form.addRow("До кандидатов", self.count)
        durations = QWidget()
        durations_row = QHBoxLayout(durations)
        durations_row.setContentsMargins(0, 0, 0, 0)
        durations_row.addWidget(self.minimum)
        durations_row.addWidget(QLabel("—"))
        durations_row.addWidget(self.maximum)
        form.addRow("Длина, секунд", durations)
        form.addRow("Чувствительность", self.sensitivity)
        self.categories = []
        for label in ("История", "Юмор", "Реакция", "Неожиданность"):
            check = QCheckBox(label)
            check.setChecked(True)
            self.categories.append(check)
            form.addRow(check)
        self.provider = combo(["Локально · без сети", "Gemini API · по согласию"])
        form.addRow("Анализ", self.provider)
        self.model = combo(list(MODELS))
        form.addRow("Модель Gemini", self.model)
        self.key = QLineEdit(self.api_key)
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText("Ключ только в Credential Manager")
        form.addRow("API-ключ", self.key)
        api_row = QWidget()
        api_layout = QHBoxLayout(api_row)
        api_layout.setContentsMargins(0, 0, 0, 0)
        api_layout.addWidget(button("Проверить", self.check_api))
        api_layout.addWidget(button("Очистить", self.clear_api))
        form.addRow(api_row)
        self.api_status = note("API выключен. Локальная работа не требует ключа.")
        form.addRow(self.api_status)
        self.analyze_button = button("Найти эпизоды", self.analyze, True)
        form.addRow(self.analyze_button)
        sidebar_layout.addWidget(search)
        self.tools_button = button("Проверить FFmpeg / Whisper", self.check_tools)
        sidebar_layout.addWidget(self.tools_button)
        sidebar_layout.addWidget(note("В локальном режиме оцениваются текст и аудиособытия. Число результатов зависит от записи. Для видео без речи доступен ручной отбор."))
        sidebar_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(sidebar)
        scroll.setMinimumWidth(395)
        splitter.addWidget(scroll)
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)
        splitter.setSizes([405, 1055])
        self.create_candidates_tab()
        self.create_transcript_tab()
        self.create_export_tab()
        bottom = QHBoxLayout()
        self.status = QLabel("Готово к работе")
        self.status.setWordWrap(True)
        bottom.addWidget(self.status, 1)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(280)
        bottom.addWidget(self.progress)
        self.cancel_button = button("Отменить задачу", self.cancel_task)
        self.cancel_button.setEnabled(False)
        bottom.addWidget(self.cancel_button)
        root.addLayout(bottom)
        self.mutating = [self.open_button, self.save_button, self.settings_button, self.import_button, self.relink_button,
                         self.analyze_button, self.tools_button, sidebar, self.candidate_actions, self.table,
                         self.transcript_controls, self.transcript, self.export_group, self.export_button, self.more_button]
        for shortcut, callback in (("Ctrl+O", self.open_project), ("Ctrl+S", self.save)):
            action = QAction(self)
            action.setShortcut(shortcut)
            action.triggered.connect(callback)
            self.addAction(action)

    def create_candidates_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        controls = QHBoxLayout()
        self.results_label = QLabel("Эпизоды появятся после анализа")
        controls.addWidget(self.results_label, 1)
        self.sort = combo(["По хронологии", "По рейтингу"])
        self.sort.currentIndexChanged.connect(self.refresh_candidates)
        controls.addWidget(self.sort)
        controls.addWidget(button("＋ Вручную", self.add_manual))
        controls.addWidget(button("Отметить все", lambda: self.select_all(True)))
        controls.addWidget(button("Снять отметки", lambda: self.select_all(False)))
        self.candidate_actions = QWidget()
        self.candidate_actions.setLayout(controls)
        layout.addWidget(self.candidate_actions)
        inner = QSplitter(Qt.Vertical)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["✓", "Начало", "Конец", "Оценка", "Категория", "Почему выбран"])
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        for col, width in enumerate((40, 125, 125, 65, 130)):
            self.table.setColumnWidth(col, width)
        self.table.itemChanged.connect(self.candidate_changed)
        self.table.itemSelectionChanged.connect(self.candidate_selected)
        inner.addWidget(self.table)
        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        self.video = VideoPreview()
        preview_layout.addWidget(self.video, 1)
        transport = QHBoxLayout()
        self.play_button = button("▶ Видео", self.play_candidate)
        self.play_button.setEnabled(False)
        transport.addWidget(self.play_button)
        transport.addWidget(button("⏸ Пауза", lambda: self.player.pause()))
        self.position = QSlider(Qt.Horizontal)
        self.position.sliderMoved.connect(self.seek)
        transport.addWidget(self.position, 1)
        self.position_label = QLabel("00:00:00.000")
        transport.addWidget(self.position_label)
        preview_layout.addLayout(transport)
        self.preview_status = note("Кадр появится сразу после импорта. Просмотр доступен и во время поиска.")
        preview_layout.addWidget(self.preview_status)
        self.still_timer = QTimer(self)
        self.still_timer.setSingleShot(True)
        self.still_timer.timeout.connect(lambda: self.request_poster(self.player.position()))
        self.reason = note("Выберите строку. Начало и конец можно изменить прямо в таблице: ЧЧ:ММ:СС.мс.")
        preview_layout.addWidget(self.reason)
        inner.addWidget(preview)
        inner.setSizes([310, 340])
        layout.addWidget(inner, 1)
        self.more_button = button("Показать ещё", self.show_more)
        layout.addWidget(self.more_button)
        self.tabs.addTab(tab, "01  Эпизоды")

    def create_transcript_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(note("Исправляйте текст и временные метки перед экспортом. Изменения применяются к субтитрам. Для нового поиска нажмите «Найти эпизоды»: сохранённый текст будет использован повторно."))
        row = QHBoxLayout()
        self.srt_import = button("Импорт SRT", self.import_srt)
        row.addWidget(self.srt_import)
        row.addWidget(button("＋ Реплика", self.add_segment))
        row.addWidget(button("Удалить реплику", self.remove_segment))
        row.addWidget(button("Очистить расшифровку", self.clear_transcript))
        row.addStretch()
        self.transcript_controls = QWidget()
        self.transcript_controls.setLayout(row)
        layout.addWidget(self.transcript_controls)
        self.transcript = QTableWidget(0, 3)
        self.transcript.setHorizontalHeaderLabels(["Начало", "Конец", "Текст (можно исправить)"])
        self.transcript.setColumnWidth(0, 135)
        self.transcript.setColumnWidth(1, 135)
        self.transcript.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.transcript.setSelectionBehavior(QTableWidget.SelectRows)
        self.transcript.setSelectionMode(QTableWidget.SingleSelection)
        self.transcript.itemChanged.connect(self.segment_changed)
        layout.addWidget(self.transcript)
        self.tabs.addTab(tab, "02  Расшифровка")

    def create_export_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        group = QGroupBox("03   Подготовка клипов для CapCut")
        self.export_group = group
        form = QFormLayout(group)
        self.aspect = combo(["9:16", "16:9", "1:1"])
        self.resolution = combo(["720", "1080"], "1080")
        self.fps = combo(["24", "25", "30", "50", "60"], "30")
        self.quality = combo(["Высокое · CRF 18", "Стандарт · CRF 20", "Компактно · CRF 25"], "Стандарт · CRF 20")
        self.top = spin(20, 100, 40)
        self.video_y = spin(0, 80, 0)
        self.subtitle_y = spin(0, 99, 31)
        self.subtitle_mode = combo(["Выключены", "Отдельный SRT", "Встроить в видео"])
        for label, widget in [("Холст", self.aspect), ("Размер, p", self.resolution), ("FPS", self.fps), ("Качество H.264", self.quality),
                              ("Высота зоны видео, %", self.top), ("Отступ зоны сверху, %", self.video_y),
                              ("Субтитры", self.subtitle_mode), ("Позиция текста от верха, %", self.subtitle_y)]:
            form.addRow(label, widget)
        self.export_folder = QLineEdit(str(self.output))
        folder_row = QWidget()
        folder_layout = QHBoxLayout(folder_row)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        folder_layout.addWidget(self.export_folder)
        folder_layout.addWidget(button("Выбрать…", self.choose_output))
        form.addRow("Папка результата", folder_row)
        layout.addWidget(group)
        notice = QLabel("Видео вписывается с сохранением пропорций. Нижняя зона — чистое чёрное поле. MP4 не содержит прозрачности. Исходный звук сохраняется; музыка и эффекты не добавляются.")
        notice.setWordWrap(True)
        notice.setObjectName("notice")
        layout.addWidget(notice)
        row = QHBoxLayout()
        self.export_button = button("Экспортировать отмеченные", self.export_selected, True)
        row.addWidget(self.export_button)
        row.addWidget(button("Открыть папку", self.open_output))
        layout.addLayout(row)
        self.queue = QTextEdit()
        self.queue.setReadOnly(True)
        self.queue.setPlaceholderText("Здесь появятся очередь и пути готовых файлов. Каждый MP4 проверяется после кодирования.")
        layout.addWidget(self.queue, 1)
        self.tabs.addTab(tab, "03  Экспорт")

    def connect_player(self):
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.audio.setVolume(.7)
        self.player.setAudioOutput(self.audio)
        self.video_sink = QVideoSink(self)
        self.video_sink.videoFrameChanged.connect(self.video.set_frame)
        self.player.setVideoSink(self.video_sink)
        self.player.positionChanged.connect(self.position_changed)
        self.player.errorOccurred.connect(lambda error, text: self.preview_status.setText(
            "Ошибка воспроизведения: " + text + ". Кадры и экспорт доступны через FFmpeg."))

    def request_poster(self, position=0):
        if not self.project or self.player.playbackState() == QMediaPlayer.PlayingState:
            return
        self.poster_serial += 1
        serial = self.poster_serial
        for _, token in self.poster_jobs:
            token.cancel()
        media, tools = copy.copy(self.project.media), copy.copy(self.tools)
        position = max(0, min(int(position), max(0, media.duration_ms - 1000)))
        folder = self.cache / "preview"
        target = folder / f"{media.source_file_id}-{position}.png"
        token = Cancellation()
        def execute(cancel, progress):
            folder.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with tempfile.TemporaryDirectory(dir=folder) as tmp:
                    image_file = Path(tmp) / "frame.png"
                    thumbnail(media, position, image_file, tools, cancel)
                    cancel.check()
                    if not image_file.is_file():
                        raise RuntimeError("FFmpeg не извлёк кадр. Попробуйте другую позицию.")
                    os.replace(image_file, target)
            return str(target)
        worker = Worker(execute, token, self)
        self.poster_jobs.append((worker, token))
        def ready(path):
            if (self.project and self.project.media.source_file_id == media.source_file_id
                    and serial == self.poster_serial and self.player.playbackState() != QMediaPlayer.PlayingState):
                image = QImage(path)
                if not image.isNull():
                    self.video.set_image(image)
                    self.preview_status.setText("Кадр исходника · нажмите «Видео» или выберите фрагмент")
        worker.result.connect(ready)
        worker.error.connect(lambda message: self.preview_status.setText(message) if serial == self.poster_serial else None)
        worker.finished.connect(lambda: self.poster_done(worker))
        worker.start()

    def poster_done(self, worker):
        worker.wait()
        self.poster_jobs = [(job, token) for job, token in self.poster_jobs if job is not worker]
        worker.deleteLater()

    def start_task(self, function, callback):
        if self.task_thread is not None:
            return
        self.player.pause()
        self.progress.setRange(0, 0)
        self.status.setText("Задача запущена — подготовка…")
        self.token = Cancellation()
        self.worker = Worker(function, self.token, self)
        self.task_thread = self.worker
        self.worker.progress.connect(self.task_progress)
        self.worker.result.connect(callback)
        self.worker.error.connect(self.task_error)
        self.worker.cancelled.connect(lambda: self.status.setText(
            "Задача отменена. Следующий поиск продолжит сохранённые блоки; готовые клипы сохранены."))
        self.task_thread.finished.connect(self.task_done)
        for widget in self.mutating:
            widget.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.task_thread.start()

    def task_progress(self, percent, message):
        self.progress.setRange(0, 100 if percent > 0 else 0)
        self.progress.setValue(percent)
        self.status.setText(message)
        if message.startswith("Готов: "):
            self.queue.append(message)

    def task_error(self, message):
        # Never display an entered secret even if an unexpected backend embeds it in an error.
        if self.key.text():
            message = message.replace(self.key.text(), "[ключ скрыт]")
        self.status.setText("Операция не завершена")
        self.show_error(message)

    def task_done(self):
        thread = self.task_thread
        # The QObject stays in the UI thread; join the Python worker before deletion.
        thread.wait()
        self.task_thread = None
        self.worker = None
        self.token = None
        thread.deleteLater()
        self.progress.setRange(0, 100)
        for widget in self.mutating:
            widget.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.analyze_button.setText("Найти эпизоды")
        self.more_button.setEnabled(bool(self.project and len(self.project.candidates) > self.visible_count))

    def cancel_task(self):
        if self.token:
            self.token.cancel()
            self.status.setText("Отмена… Внешний процесс останавливается; сетевой запрос завершится по таймауту до 40 секунд.")

    def show_error(self, message):
        QMessageBox.warning(self, "Clipnado", str(message))

    def changed(self):
        self.dirty = True
        if self.project:
            try:
                save_project(self.project, data_root() / "recovery.clipnado.json")
            except (OSError, ValueError) as exc:
                self.status.setText("Не удалось создать копию восстановления: " + str(exc))

    def confirm_discard(self):
        if self.task_thread is not None:
            return False
        if not self.dirty:
            return True
        return QMessageBox.question(self, "Несохранённые изменения", "Заменить текущий проект? Сохраните его отдельной кнопкой, если он нужен.", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes

    def import_media(self):
        if not self.confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Выберите видео", "", "Видео (*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.ts);;Все файлы (*)")
        if path:
            self.start_task(lambda token, progress: probe(Path(path), self.tools, token), self.media_loaded)

    def media_loaded(self, media):
        self.project = Project(media)
        self.project_path = None
        self.visible_count = 15
        self.load_ui()
        self.changed()
        self.status.setText("Видео загружено" + (". Аудиодорожка отсутствует — используйте ручной отбор." if not media.has_audio else ". Настройте поиск или импортируйте SRT."))

    def load_ui(self):
        p = self.project
        m = p.media
        self.media_label.setText(f"{Path(m.path).name}  ·  {timecode(m.duration_ms)}  ·  {m.width}×{m.height}  ·  {m.fps:.3f} FPS  ·  {'со звуком' if m.has_audio else 'без звука'}")
        self.media_label.setToolTip(m.path)
        self.count.setValue(p.search.count)
        self.minimum.setValue(p.search.min_seconds)
        self.maximum.setValue(p.search.max_seconds)
        self.sensitivity.setValue(p.search.sensitivity)
        for c in self.categories:
            c.setChecked(c.text() in p.search.categories)
        self.aspect.setCurrentText(p.export.aspect)
        self.resolution.setCurrentText(str(p.export.resolution))
        self.fps.setCurrentText(str(p.export.fps))
        self.quality.setCurrentIndex(min(range(3), key=lambda i: abs([18, 20, 25][i] - p.export.quality)))
        self.top.setValue(round(p.export.top_fraction * 100))
        self.video_y.setValue(round(p.export.video_y * 100))
        self.subtitle_y.setValue(round(p.export.subtitle_y * 100))
        self.subtitle_mode.setCurrentIndex(["off", "srt", "burn"].index(p.export.subtitles))
        url = QUrl.fromLocalFile(m.path)
        self.preview_end = None
        if self.player.source() != url:
            self.player.stop()
            self.video.clear_image()
            self.player.setSource(url)
        self.play_button.setEnabled(True)
        self.request_poster()
        self.position.setRange(0, m.duration_ms)
        self.refresh_candidates()
        self.refresh_transcript()

    def search_settings(self):
        result = SearchSettings(self.count.value(), self.minimum.value(), self.maximum.value(), self.sensitivity.value(),
                                [c.text() for c in self.categories if c.isChecked()])
        result.validate()
        return result

    def export_settings(self):
        result = ExportSettings(self.aspect.currentText(), int(self.resolution.currentText()), int(self.fps.currentText()),
                                [18, 20, 25][self.quality.currentIndex()], self.top.value() / 100, self.video_y.value() / 100,
                                ["off", "srt", "burn"][self.subtitle_mode.currentIndex()], self.subtitle_y.value() / 100)
        result.validate()
        return result

    def analyze(self):
        if not self.project or self.task_thread is not None:
            return
        try:
            settings = self.search_settings()
            provider = LocalFallbackProvider()
            if self.provider.currentIndex() == 1:
                consent = QMessageBox.question(self, "Передача расшифровки в Google", "Будет отправлен только текст с таймкодами, без видео и аудио. На бесплатном уровне Google может использовать данные для улучшения продуктов, в том числе с проверкой людьми.\n\nНе отправляйте чувствительные записи без осознанного решения.\n\nПродолжить для этого анализа?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if consent != QMessageBox.Yes:
                    return
                free = QMessageBox.question(self, "Только бесплатный проект Google", "API не сообщает программе тариф вашего ключа. Программа не подключает биллинг и останавливает онлайн-анализ при 429. Для исключения оплаты нужен проект БЕЗ платёжного аккаунта.\n\nВы проверили Free Tier и отсутствие привязанного биллинга в AI Studio?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if free != QMessageBox.Yes:
                    return
                provider = GeminiFreeProvider(self.key.text(), True, True, self.model.currentText())
                self.persist_key()
            project = copy.deepcopy(self.project)
            tools, cache = copy.copy(self.tools), self.cache
            def execute(token, progress):
                segments, energy, warnings = recognize(project.media, tools, cache, token, progress, project.segments,
                                                       existing_energy=project.audio_energy)
                progress(82, "Поиск цельных эпизодов…")
                candidates = provider.analyze(project.media, segments, settings, energy, token)
                project.segments, project.candidates, project.search = segments, candidates, settings
                project.audio_energy = energy
                project.warnings = warnings
                project.validate()
                return project
            self.analyze_button.setText("Поиск выполняется…")
            self.start_task(execute, self.analysis_ready)
        except (ValueError, RuntimeError) as exc:
            self.show_error(exc)

    def analysis_ready(self, project):
        self.project = project
        self.visible_count = project.search.count
        self.refresh_candidates()
        self.refresh_transcript()
        self.changed()
        self.progress.setValue(100)
        self.status.setText(f"Анализ завершён: {len(project.candidates)} кандидатов. " + " ".join(project.warnings))
        if project.warnings:
            QMessageBox.information(self, "Качество поиска", "\n\n".join(project.warnings))

    def current_candidate(self):
        row = self.table.currentRow()
        if self.project and row >= 0 and self.table.item(row, 0):
            identity = self.table.item(row, 0).data(Qt.UserRole)
            return next((c for c in self.project.candidates if c.id == identity), None)
        return None

    def refresh_candidates(self, *_):
        self.repopulating = True
        self.table.setRowCount(0)
        if self.project:
            candidates = sorted(self.project.candidates, key=lambda c: (-c.score, c.start_ms))
            visible = sorted(candidates[:self.visible_count], key=(lambda c: (-c.score, c.start_ms)) if self.sort.currentIndex() else lambda c: c.start_ms)
            self.table.setRowCount(len(visible))
            for row, c in enumerate(visible):
                for col, text in enumerate(("", timecode(c.start_ms), timecode(c.end_ms), str(c.score), c.category, c.reason)):
                    item = QTableWidgetItem(text)
                    item.setToolTip(c.reason + "\n" + c.transcript_excerpt)
                    if col == 0:
                        item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                        item.setCheckState(Qt.Checked if c.selected else Qt.Unchecked)
                        item.setData(Qt.UserRole, c.id)
                    elif col not in (1, 2):
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.table.setItem(row, col, item)
            selected = sum(c.selected for c in self.project.candidates)
            self.results_label.setText(f"Показано {len(visible)} из {len(candidates)} · отмечено {selected}")
            self.more_button.setEnabled(len(visible) < len(candidates))
        self.repopulating = False

    def candidate_changed(self, item):
        if self.repopulating or not self.project:
            return
        identity = self.table.item(item.row(), 0).data(Qt.UserRole)
        c = next(c for c in self.project.candidates if c.id == identity)
        previous = (c.start_ms, c.end_ms, c.selected)
        try:
            c.selected = self.table.item(item.row(), 0).checkState() == Qt.Checked
            c.start_ms = parse_time(self.table.item(item.row(), 1).text())
            c.end_ms = parse_time(self.table.item(item.row(), 2).text())
            c.validate(self.project.media)
            self.changed()
            self.results_label.setText(f"Показано {self.table.rowCount()} из {len(self.project.candidates)} · отмечено {sum(c.selected for c in self.project.candidates)}")
        except ValueError as exc:
            c.start_ms, c.end_ms, c.selected = previous
            self.show_error(exc)
            self.refresh_candidates()

    def candidate_selected(self):
        c = self.current_candidate()
        if c:
            self.reason.setText(c.reason + "\n" + c.transcript_excerpt[:500])
            self.player.pause()
            self.player.setPosition(c.start_ms)
            self.play_button.setText("▶ Фрагмент")
            self.request_poster(c.start_ms)

    def select_all(self, selected):
        if self.project:
            visible = {self.table.item(row, 0).data(Qt.UserRole) for row in range(self.table.rowCount())}
            for c in self.project.candidates:
                if c.id in visible:
                    c.selected = selected
            self.refresh_candidates()
            self.changed()

    def show_more(self):
        self.visible_count += self.count.value()
        self.refresh_candidates()

    def add_manual(self):
        if self.project:
            start = min(self.player.position(), max(0, self.project.media.duration_ms - 1000))
            end = min(self.project.media.duration_ms, start + self.minimum.value() * 1000)
            self.project.candidates.append(Candidate(self.project.media.source_file_id, start, end, 0, "Вручную", "Интервал выбран пользователем", provider="manual", selected=True))
            self.visible_count = len(self.project.candidates)
            self.refresh_candidates()
            self.changed()

    def play_candidate(self):
        c = self.current_candidate()
        if c:
            self.preview_end = c.end_ms
            self.player.setPosition(c.start_ms)
        elif self.project:
            self.preview_end = None
            if self.player.position() >= self.project.media.duration_ms - 100:
                self.player.setPosition(0)
        else:
            return
        self.still_timer.stop()
        for _, token in self.poster_jobs:
            token.cancel()
        self.player.play()

    def seek(self, position):
        self.preview_end = None
        self.player.setPosition(position)
        if self.player.playbackState() != QMediaPlayer.PlayingState:
            self.still_timer.start(200)

    def position_changed(self, position):
        if not self.position.isSliderDown():
            self.position.setValue(position)
        self.position_label.setText(timecode(position))
        if self.preview_end is not None and position >= self.preview_end:
            self.player.pause()
            self.preview_end = None

    def refresh_transcript(self):
        self.repopulating = True
        self.transcript.setRowCount(0)
        if self.project:
            self.transcript.setRowCount(len(self.project.segments))
            for row, s in enumerate(self.project.segments):
                for col, text in enumerate((timecode(s.start_ms), timecode(s.end_ms), s.text)):
                    self.transcript.setItem(row, col, QTableWidgetItem(text))
        self.repopulating = False

    def segment_changed(self, item):
        if self.repopulating or not self.project:
            return
        row = item.row()
        try:
            segment = Segment(parse_time(self.transcript.item(row, 0).text()), parse_time(self.transcript.item(row, 1).text()), self.transcript.item(row, 2).text())
            segment.validate(self.project.media.duration_ms)
            self.project.segments[row] = segment
            self.changed()
        except ValueError as exc:
            self.show_error(exc)
            self.refresh_transcript()

    def import_srt(self):
        if not self.project:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Импортировать расшифровку", "", "SubRip (*.srt)")
        if path:
            try:
                self.project.segments = from_srt(Path(path).read_text(encoding="utf-8-sig"), self.project.media.duration_ms)
                self.refresh_transcript()
                self.changed()
            except (OSError, ValueError) as exc:
                self.show_error(exc)

    def add_segment(self):
        if self.project:
            start = min(self.player.position(), max(0, self.project.media.duration_ms - 1000))
            self.project.segments.append(Segment(start, min(self.project.media.duration_ms, start + 3000), "Новая реплика"))
            self.refresh_transcript()
            self.changed()

    def remove_segment(self):
        row = self.transcript.currentRow()
        if self.project and row >= 0:
            del self.project.segments[row]
            self.refresh_transcript()
            self.changed()

    def clear_transcript(self):
        if self.project and QMessageBox.question(self, "Очистить текст", "Удалить расшифровку? Следующий анализ использует кэш распознавания, если он есть. Для полного пересчёта удалите кэш в выбранной папке.") == QMessageBox.Yes:
            self.project.segments = []
            self.refresh_transcript()
            self.changed()

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, "Папка результата", self.export_folder.text())
        if path:
            self.export_folder.setText(path)

    def open_output(self):
        folder = Path(self.export_folder.text())
        try:
            folder.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder.resolve())))
        except OSError as exc:
            self.show_error(exc)

    def export_selected(self):
        if not self.project or self.task_thread is not None:
            return
        try:
            settings = self.export_settings()
            selected = [copy.deepcopy(c) for c in self.project.candidates if c.selected]
            if not selected:
                raise ValueError("Сначала отметьте клипы в таблице эпизодов")
            if not self.export_folder.text().strip():
                raise ValueError("Выберите папку результата")
            self.output = Path(self.export_folder.text()).resolve()
            self.project.export = settings
            self.changed()
            self.persist_settings()
            project, tools, output = copy.deepcopy(self.project), copy.copy(self.tools), self.output
            self.queue.clear()
            for i, c in enumerate(selected, 1):
                self.queue.append(f"{i}. {timecode(c.start_ms)} → {timecode(c.end_ms)} — в очереди")
            def execute(token, progress):
                results = []
                for index, c in enumerate(selected):
                    token.check()
                    path = export_clip(project.media, c, project.segments, settings, output, tools, token,
                                       lambda value, text: progress(int((index + value / 100) / len(selected) * 100), f"{index + 1}/{len(selected)} · {text}"))
                    results.append(str(path))
                    progress(int((index + 1) / len(selected) * 100), "Готов: " + str(path))
                return results
            self.start_task(execute, self.exports_ready)
        except (ValueError, OSError) as exc:
            self.show_error(exc)

    def exports_ready(self, paths):
        self.queue.setPlainText("Проверенные файлы:\n" + "\n".join(paths))
        self.status.setText(f"Экспорт завершён: {len(paths)} файлов. MP4 проверены декодированием.")
        self.progress.setValue(100)

    def persist_settings(self):
        write_settings(self.tools, str(self.cache), str(self.output), self.model.currentText())

    def persist_key(self):
        self.api_key = self.key.text()
        try:
            save_key(self.api_key)
        except Exception:
            self.api_status.setText("Хранилище ключей недоступно; ключ только в памяти сеанса.")

    def clear_api(self):
        try:
            save_key("")
            self.key.clear()
            self.api_key = ""
            self.api_status.setText("Ключ удалён из памяти и безопасного хранилища.")
        except Exception:
            self.key.clear()
            self.api_key = ""
            self.api_status.setText("Ключ очищен из памяти; хранилище недоступно. Проверьте Credential Manager.")

    def check_api(self):
        if self.task_thread is not None:
            return
        try:
            provider = GeminiFreeProvider(self.key.text(), model=self.model.currentText())
            self.persist_key()
            self.start_task(lambda token, progress: provider.check_connection(token), lambda message: self.api_status.setText(message))
        except RuntimeError as exc:
            self.show_error(exc)

    def settings(self):
        if self.task_thread is not None:
            return
        dialog = SettingsDialog(self)
        if dialog.exec() == QDialog.Accepted:
            values = {name: line.text().strip() for name, line in dialog.fields.items()}
            self.tools = Tools(**{name: values[name] for name in ("ffmpeg", "ffprobe", "whisper", "model")},
                               gpu=dialog.gpu.isChecked(), threads=dialog.threads.value())
            self.cache = Path(values["cache"] or str(data_root() / "cache"))
            self.output = Path(values["output"] or self.export_folder.text())
            self.export_folder.setText(str(self.output))
            try:
                self.persist_settings()
            except OSError as exc:
                self.show_error(exc)

    def check_tools(self):
        tools = copy.copy(self.tools)
        def execute(token, progress):
            result = []
            for executable, flag in ((tools.ffmpeg, "-version"), (tools.ffprobe, "-version")):
                text = run([executable, flag], token, timeout=15)
                result.append(text.splitlines()[0])
            if tools.model and Path(tools.model).is_file():
                run([tools.whisper, "--help"], token, timeout=15)
                result.append("Whisper запускается, модель найдена. Проверка реального распознавания выполняется при анализе.")
            else:
                result.append("Модель Whisper не настроена. Импорт SRT и экспорт доступны.")
            return "\n\n".join(result)
        self.start_task(execute, lambda text: QMessageBox.information(self, "Инструменты", text))

    def save(self):
        if not self.project or self.task_thread is not None:
            return
        try:
            self.project.search = self.search_settings()
            self.project.export = self.export_settings()
            path = self.project_path
            if path is None:
                name, _ = QFileDialog.getSaveFileName(self, "Сохранить проект", "project.clipnado.json", "Проект Clipnado (*.json)")
                if not name:
                    return
                path = Path(name)
            if path.resolve() == Path(self.project.media.path).resolve():
                raise ValueError("Нельзя записывать проект поверх исходного видео")
            snapshot = copy.deepcopy(self.project)
            def saved(_):
                self.project_path = path
                self.dirty = False
                self.status.setText("Проект сохранён: " + str(path))
            self.start_task(lambda token, progress: save_project(snapshot, path), saved)
        except (ValueError, OSError) as exc:
            self.show_error(exc)

    def open_project(self):
        if not self.confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Открыть проект", "", "Проект Clipnado (*.json)")
        if path:
            self.load_project_path(Path(path))

    def load_project_path(self, path):
        try:
            project = load_project(path)
            source = Path(project.media.path)
            if not source.is_absolute():
                source = path.parent / source
            if not source.exists():
                name, _ = QFileDialog.getOpenFileName(self, "Исходник перемещён. Укажите то же видео")
                if not name:
                    return
                source = Path(name)
            def execute(token, progress):
                media = probe(source, self.tools, token)
                if media.source_file_id != project.media.source_file_id or abs(media.duration_ms - project.media.duration_ms) > 100:
                    raise ValueError("Выбран другой исходник; перепривязка отменена")
                project.media = media
                return project
            def ready(value):
                self.project = value
                self.project_path = path if path.name != "recovery.clipnado.json" else None
                self.visible_count = value.search.count
                self.dirty = self.project_path is None
                self.load_ui()
                self.status.setText("Проект открыт. Распознавание и API не запускались.")
            self.start_task(execute, ready)
        except (ValueError, OSError) as exc:
            self.show_error(exc)

    def relink(self):
        if not self.project or self.task_thread is not None:
            return
        name, _ = QFileDialog.getOpenFileName(self, "Укажите то же исходное видео")
        if name:
            def execute(token, progress):
                media = probe(Path(name), self.tools, token)
                if media.source_file_id != self.project.media.source_file_id:
                    raise ValueError("Файл отличается от исходника")
                return media
            def ready(media):
                self.project.media = media
                self.load_ui()
                self.changed()
            self.start_task(execute, ready)

    def offer_recovery(self, path):
        if self.project is None and self.task_thread is None and QMessageBox.question(self, "Восстановление", "Восстановить последний рабочий проект из автоматической копии?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
            self.load_project_path(path)

    def closeEvent(self, event):
        if self.task_thread is not None:
            self.show_error("Сначала отмените задачу и дождитесь её остановки.")
            event.ignore()
            return
        if self.dirty and QMessageBox.question(self, "Закрыть Clipnado", "Есть несохранённые изменения. Автоматическая копия сохранена для восстановления. Закрыть программу?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            event.ignore()
            return
        self.player.stop()
        self.player.setSource(QUrl())
        self.still_timer.stop()
        for worker, token in self.poster_jobs:
            token.cancel()
        for worker, _ in list(self.poster_jobs):
            worker.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    translator = QTranslator(app)
    if translator.load("qtbase_ru", QLibraryInfo.path(QLibraryInfo.TranslationsPath)):
        app.installTranslator(translator)
    app.setApplicationName("ClipnadoStudioLite")
    app.setOrganizationName("Clipnado")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
