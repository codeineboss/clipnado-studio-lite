from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import tempfile
import time
from uuid import uuid4
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
from .analysis import recognize
from .export import clip_export_settings, composition_frame, export_clip
from .gaming import GamingRegionsDialog
from .media import Tools, probe, thumbnail
from .models import EXPORT_LAYOUTS, Candidate, ExportSettings, Project, SearchSettings, Segment, parse_time, timecode
from .process import Cancellation, Cancelled, run
from .preview import VideoPreview
from .projects import load_project, save_project
from .selection import shortfall_message
from .settings import data_root, read_settings, restore_tools, write_settings
from .subtitles import from_srt
from .workers import Worker
from .local_llm import LocalModelProvider, prepare_backend
from .assist import LocalClipAssistant
from .suggestions import ClipSuggestionsDialog

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
            ("llama", "llama.cpp CLI", parent.tools.llama, False),
            ("semantic_model", "Модель Qwen (.gguf)", parent.tools.semantic_model, False),
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
        self.acceleration = combo(["Авто: Vulkan → CPU", "Только CPU"])
        self.acceleration.setCurrentIndex(1 if parent.tools.acceleration == "cpu" else 0)
        form.addRow("Потоки CPU", self.threads)
        form.addRow("Ускорение", self.acceleration)
        layout.addLayout(form)
        layout.addWidget(note("Все модели работают локально. Для Xeon E5-2670 используйте комплектные сборки без AVX2/FMA/F16C. Режим «Авто» проверяет Vulkan и использует CPU при недоступном ускорении. Qwen оценивает смысл расшифровки, изображение она не анализирует."))
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
        self.assistant_dialog = None
        self.export_preview_dialog = None
        self.gaming_camera = None
        self.gaming_game = [0., 0., 1., 1.]
        self.gaming_fraction = .4
        self.boundary_undo = []
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
        self.build_ui()
        self.task_clock = QTimer(self)
        self.task_clock.setInterval(1000)
        self.task_clock.timeout.connect(self.update_task_elapsed)
        self.update_local_hint()
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
        self.target_count = spin(1, 100, 10)
        self.min_score = spin(0, 100, 60)
        self.minimum = spin(1, 600, 60)
        self.maximum = spin(1, 600, 90)
        self.sensitivity = spin(0, 100, 55)
        form.addRow("До кандидатов", self.count)
        form.addRow("Минимум (цель)", self.target_count)
        form.addRow("Оценка от", self.min_score)
        self.target_count.setToolTip("Ищем столько разных эпизодов, если позволяет материал. Оценки ради цели не повышаются.")
        self.target_count.valueChanged.connect(lambda value: self.count.setValue(max(value, self.count.value())))
        self.count.valueChanged.connect(lambda value: self.target_count.setValue(min(value, self.target_count.value())))
        durations = QWidget()
        durations_row = QHBoxLayout(durations)
        durations_row.setContentsMargins(0, 0, 0, 0)
        durations_row.addWidget(self.minimum)
        durations_row.addWidget(QLabel("—"))
        durations_row.addWidget(self.maximum)
        form.addRow("Длина, секунд", durations)
        self.categories = []
        for label in ("История", "Юмор", "Реакция", "Неожиданность"):
            check = QCheckBox(label)
            check.setChecked(True)
            self.categories.append(check)
            form.addRow(check)
        form.addRow("Анализ", QLabel("Qwen · локальная модель"))
        self.local_hint = note("")
        form.addRow(self.local_hint)
        form.addRow(button("Указать модель Qwen (.gguf)", self.choose_semantic_model))
        self.analyze_button = button("Найти эпизоды", self.analyze, True)
        form.addRow(self.analyze_button)
        self.content_query = QLineEdit()
        self.content_query.setMaxLength(500)
        self.content_query.setPlaceholderText("Например: где спорят с учителем")
        form.addRow("Найти по смыслу", self.content_query)
        self.content_button = button("Найти по содержанию", self.find_by_content)
        form.addRow(self.content_button)
        self.query_results_button = button("Результаты последнего запроса", self.show_query_results)
        form.addRow(self.query_results_button)
        form.addRow(note("Поиск по содержанию проверяет всю расшифровку. Совпадения показываются отдельно и добавляются в эпизоды по вашему выбору."))
        sidebar_layout.addWidget(search)
        self.tools_button = button("Проверить FFmpeg / Whisper", self.check_tools)
        sidebar_layout.addWidget(self.tools_button)
        sidebar_layout.addWidget(button("Отчёт о последнем анализе", self.show_analysis_report))
        sidebar_layout.addWidget(note("Цель по умолчанию: 10 разных клипов с оценкой от 60. При недоборе показывается причина. Локально оцениваются текст и звук; интересность требует просмотра."))
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
                         self.transcript_controls, self.transcript, self.export_group, self.export_button, self.more_button,
                         self.export_clips, self.export_clip_controls]
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
        actions_layout = QVBoxLayout(self.candidate_actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.addLayout(controls)
        refinement = QHBoxLayout()
        self.refine_button = button("Подобрать границы", self.refine_selected)
        refinement.addWidget(self.refine_button)
        self.undo_boundaries_button = button("Отменить границы", self.undo_boundaries)
        refinement.addWidget(self.undo_boundaries_button)
        refinement.addStretch()
        actions_layout.addLayout(refinement)
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
        tab_layout = QVBoxLayout(tab)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)
        tab_layout.addWidget(scroll, 1)
        group = QGroupBox("03   Настройки экспорта")
        self.export_group = group
        form = QFormLayout(group)
        self.export_layout = combo(list(EXPORT_LAYOUTS.values()))
        self.gaming_summary = QLabel("Области камеры и игры ещё не выбраны")
        self.gaming_summary.setWordWrap(True)
        self.aspect = combo(["9:16", "16:9", "1:1"])
        self.resolution = combo(["720", "1080"], "1080")
        self.fps = combo(["24", "25", "30", "50", "60"], "30")
        self.quality = combo(["Высокое · CRF 18", "Стандарт · CRF 20", "Компактно · CRF 25"], "Высокое · CRF 18")
        self.top = spin(20, 100, 40)
        self.video_y = spin(0, 80, 0)
        self.subtitle_y = spin(0, 99, 31)
        self.subtitle_mode = combo(["Выключены", "Отдельный SRT", "Встроить в видео"])
        for label, widget in [("Общее оформление", self.export_layout), ("Игровая нарезка", self.gaming_summary), ("Холст для первых двух вариантов", self.aspect), ("Размер, p", self.resolution), ("FPS", self.fps), ("Качество H.264", self.quality),
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
        self.export_notice = QLabel()
        self.export_notice.setWordWrap(True)
        self.export_notice.setObjectName("notice")
        layout.addWidget(self.export_notice)
        self.export_clip_controls = QWidget()
        clip_controls = QHBoxLayout(self.export_clip_controls)
        clip_controls.setContentsMargins(0, 0, 0, 0)
        self.export_clips_label = QLabel("Отметьте эпизоды во вкладке «Эпизоды»")
        self.export_clips_label.setWordWrap(True)
        clip_controls.addWidget(self.export_clips_label, 1)
        clip_controls.addWidget(button("Применить общее ко всем", self.apply_layout_to_selected))
        self.gaming_button = button("Камера и игра…", self.configure_gaming)
        clip_controls.addWidget(self.gaming_button)
        clip_controls.addWidget(button("Предпросмотр кадра", self.preview_export_clip))
        layout.addWidget(self.export_clip_controls)
        self.export_clips = QTableWidget(0, 4)
        self.export_clips.setHorizontalHeaderLabels(["Клип · оценка", "Начало", "Конец", "Оформление этого клипа"])
        self.export_clips.verticalHeader().hide()
        self.export_clips.setColumnWidth(0, 115)
        self.export_clips.setColumnWidth(1, 115)
        self.export_clips.setColumnWidth(2, 115)
        self.export_clips.setSelectionBehavior(QTableWidget.SelectRows)
        self.export_clips.setSelectionMode(QTableWidget.SingleSelection)
        self.export_clips.setEditTriggers(QTableWidget.NoEditTriggers)
        self.export_clips.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.export_clips.setMinimumHeight(130)
        layout.insertWidget(0, self.export_clip_controls)
        layout.insertWidget(1, note("Игровая нарезка: стример сверху, игра снизу, формат 9:16. Выделите области кнопкой «Камера и игра…» один раз для видео. Для каждой строки можно выбрать своё оформление."))
        layout.insertWidget(2, self.export_clips, 2)
        self.export_layout.currentIndexChanged.connect(self.export_layout_changed)
        self.export_layout_changed()
        for widget in (self.aspect, self.resolution, self.fps, self.quality, self.subtitle_mode):
            widget.currentIndexChanged.connect(self.close_export_preview)
        for widget in (self.top, self.video_y, self.subtitle_y):
            widget.valueChanged.connect(self.close_export_preview)
        row = QHBoxLayout()
        self.export_button = button("Экспортировать отмеченные", self.export_selected, True)
        row.addWidget(self.export_button)
        row.addWidget(button("Открыть папку", self.open_output))
        tab_layout.addLayout(row)
        self.queue = QTextEdit()
        self.queue.setReadOnly(True)
        self.queue.setPlaceholderText("Здесь появятся очередь и пути готовых файлов. Каждый MP4 проверяется после кодирования.")
        self.queue.setMaximumHeight(110)
        tab_layout.addWidget(self.queue)
        self.tabs.addTab(tab, "03  Экспорт")

    def export_layout_changed(self, *_):
        # A mixed queue may contain the upper-zone layout even when the common
        # default is another layout; keep its controls available in that case.
        code = list(EXPORT_LAYOUTS)[self.export_layout.currentIndex()]
        upper = code == "top_black" or bool(self.project and any(
            c.selected and c.export_layout == "top_black" for c in self.project.candidates))
        gaming = code == "gaming_split" or bool(self.project and any(
            c.selected and c.export_layout == "gaming_split" for c in self.project.candidates))
        self.gaming_button.setEnabled(bool(self.project) and gaming)
        self.gaming_summary.setText(("Камера и игра выбраны · камера сверху " + str(round(self.gaming_fraction*100)) + "%") if self.gaming_camera else "Нажмите «Камера и игра…» и выделите области на исходнике")
        self.top.setEnabled(upper)
        self.video_y.setEnabled(upper)
        notices = {
            "top_black": "Видео целиком вписано сверху, нижняя тёмная зона остаётся для анимаций.",
            "blur_center": "Видео целиком по центру, фон — его синхронная увеличенная размытая копия.",
            "gaming_split": "Игровая нарезка: стример сверху, игра снизу. Обе области берутся из одного отрывка и остаются синхронными; результат 9:16 без растяжения.",
        }
        self.export_notice.setText("Общее оформление: " + notices[code] + " В таблице можно выбрать свой вариант для каждого клипа. Области камеры и игры общие для выбранного видео. Для высокого качества выберите 1080p, CRF 18 и FPS исходника. Разрешение, FPS, качество и субтитры — общие. Исходный звук сохраняется.")
        self.refresh_export_clips()

    def refresh_export_clips(self):
        self.close_export_preview()
        current = self.current_export_clip()
        identity = current.id if current else None
        self.export_clips.setRowCount(0)
        clips = sorted((c for c in self.project.candidates if c.selected), key=lambda c: c.start_ms) if self.project else []
        self.export_clips.setRowCount(len(clips))
        for row, c in enumerate(clips):
            for col, text in enumerate((f"{row + 1} · {c.score}/100", timecode(c.start_ms), timecode(c.end_ms))):
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, c.id)
                item.setToolTip(c.reason + "\n" + c.transcript_excerpt)
                self.export_clips.setItem(row, col, item)
            control = combo(list(EXPORT_LAYOUTS.values()))
            control.setCurrentIndex(list(EXPORT_LAYOUTS).index(c.export_layout) if c.export_layout else self.export_layout.currentIndex())
            control.setToolTip("Индивидуальное оформление" if c.export_layout else "Используется общее оформление")
            control.currentIndexChanged.connect(lambda index, clip_id=c.id: self.clip_layout_changed(clip_id, index))
            self.export_clips.setCellWidget(row, 3, control)
        if clips:
            self.export_clips.selectRow(next((i for i, c in enumerate(clips) if c.id == identity), 0))
        self.export_clips_label.setText(f"Отмеченных клипов: {len(clips)} · выберите оформление в каждой строке")

    def current_export_clip(self):
        row = self.export_clips.currentRow()
        item = self.export_clips.item(row, 0) if row >= 0 else None
        return next((c for c in self.project.candidates if c.id == item.data(Qt.UserRole)), None) if self.project and item else None

    def clip_layout_changed(self, identity, index):
        if not self.project or self.task_thread is not None:
            return
        candidate = next((c for c in self.project.candidates if c.id == identity), None)
        if candidate:
            candidate.export_layout = list(EXPORT_LAYOUTS)[index]
            for row in range(self.export_clips.rowCount()):
                if self.export_clips.item(row, 0).data(Qt.UserRole) == identity:
                    self.export_clips.selectRow(row)
                    break
            self.export_layout_changed()
            self.changed()

    def apply_layout_to_selected(self):
        if not self.project or self.task_thread is not None:
            return
        for c in self.project.candidates:
            if c.selected:
                c.export_layout = None
        self.export_layout_changed()
        self.changed()

    def configure_gaming(self):
        if not self.project or self.task_thread is not None:
            return
        try:
            settings = self.export_settings()
            clip = self.current_export_clip()
            at = min(clip.start_ms+1000, (clip.start_ms+clip.end_ms)//2) if clip else self.player.position()
            dialog = GamingRegionsDialog(copy.copy(self.project.media), copy.copy(self.tools), settings, at, self)
            if dialog.exec() == QDialog.Accepted:
                self.gaming_camera, self.gaming_game, self.gaming_fraction = dialog.regions()
                self.project.export = self.export_settings()
                self.export_layout_changed()
                self.changed()
            dialog.deleteLater()
        except (ValueError, OSError) as exc:
            self.show_error(exc)

    def close_export_preview(self, *_):
        if self.export_preview_dialog is not None:
            self.export_preview_dialog.close()
            self.export_preview_dialog = None

    def preview_export_clip(self):
        if not self.project or self.task_thread is not None:
            return
        candidate = self.current_export_clip()
        if candidate is None:
            self.show_error("Отметьте клипы во вкладке «Эпизоды» и выберите строку в таблице экспорта.")
            return
        try:
            settings = clip_export_settings(self.export_settings(), candidate)
            media, candidate, tools = copy.deepcopy(self.project.media), copy.deepcopy(candidate), copy.copy(self.tools)
            self.close_export_preview()
            settings.require_gaming_regions()
            def completed(data):
                image = QImage.fromData(data)
                if image.isNull():
                    self.show_error("Не удалось прочитать кадр оформления.")
                    return
                dialog = QDialog(self)
                dialog.setAttribute(Qt.WA_DeleteOnClose)
                dialog.setWindowTitle("Предпросмотр оформления")
                layout = QVBoxLayout(dialog)
                heading = note(f"{timecode(candidate.start_ms)}–{timecode(candidate.end_ms)} · {EXPORT_LAYOUTS[settings.layout]}\nОдин кадр выбранного клипа, без субтитров. Положение камеры и игры совпадает с экспортом; качество MP4 задаётся в настройках.")
                layout.addWidget(heading)
                preview = VideoPreview()
                preview.set_image(image)
                layout.addWidget(preview, 1)
                layout.addWidget(button("Закрыть", dialog.close))
                dialog.resize(max(400, image.width() + 50), min(image.height() + 160, self.screen().availableGeometry().height() - 60))
                self.export_preview_dialog = dialog
                dialog.destroyed.connect(lambda: setattr(self, "export_preview_dialog", None) if self.export_preview_dialog is dialog else None)
                dialog.show()
            self.start_task(lambda token, progress: composition_frame(media, candidate, settings, tools, token), completed)
        except ValueError as exc:
            self.show_error(exc)

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
        self.task_started = time.monotonic()
        self.task_base_message = "Задача выполняется"
        self.task_clock.start()
        def completed(result):
            self.task_clock.stop()
            callback(result)
        self.worker.result.connect(completed)
        self.worker.error.connect(self.task_error)
        self.worker.cancelled.connect(lambda: (self.task_clock.stop(), self.status.setText(
            "Задача отменена. Следующий поиск продолжит сохранённые блоки; готовые клипы сохранены.")))
        self.task_thread.finished.connect(self.task_done)
        for widget in self.mutating:
            widget.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.task_thread.start()

    def task_progress(self, percent, message):
        self.progress.setRange(0, 100 if percent > 0 else 0)
        self.progress.setValue(percent)
        self.task_base_message = message
        self.update_task_elapsed()
        if message.startswith("Готов: "):
            self.queue.append(message)

    def update_task_elapsed(self):
        if self.task_thread is not None:
            elapsed = timecode(int((time.monotonic() - self.task_started) * 1000))[:-4]
            self.status.setText(f"{self.task_base_message} · всего прошло {elapsed}")

    def task_error(self, message):
        self.task_clock.stop()
        self.status.setText("Операция не завершена")
        self.show_error(message)

    def task_done(self):
        self.task_clock.stop()
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
            self.task_clock.stop()
            self.status.setText("Отмена… Локальные движки завершаются, проверенные участки остаются в кэше.")

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
        self.boundary_undo.clear()
        self.close_assistant_dialog()
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
        self.target_count.setValue(p.search.target_count)
        self.min_score.setValue(p.search.min_score)
        self.minimum.setValue(p.search.min_seconds)
        self.maximum.setValue(p.search.max_seconds)
        self.sensitivity.setValue(p.search.sensitivity)
        self.content_query.setText(p.last_query)
        self.close_assistant_dialog()
        self.boundary_undo.clear()
        for c in self.categories:
            c.setChecked(c.text() in p.search.categories)
        self.export_layout.setCurrentIndex(list(EXPORT_LAYOUTS).index(p.export.layout))
        self.gaming_camera = copy.deepcopy(p.export.gaming_camera)
        self.gaming_game = list(p.export.gaming_game)
        self.gaming_fraction = p.export.gaming_fraction
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
                                [c.text() for c in self.categories if c.isChecked()],
                                target_count=self.target_count.value(), min_score=self.min_score.value())
        result.validate()
        return result

    def export_settings(self):
        result = ExportSettings(self.aspect.currentText(), int(self.resolution.currentText()), int(self.fps.currentText()),
                                [18, 20, 25][self.quality.currentIndex()], self.top.value() / 100, self.video_y.value() / 100,
                                ["off", "srt", "burn"][self.subtitle_mode.currentIndex()], self.subtitle_y.value() / 100,
                                list(EXPORT_LAYOUTS)[self.export_layout.currentIndex()],
                                copy.deepcopy(self.gaming_camera), list(self.gaming_game), self.gaming_fraction)
        result.validate()
        return result

    def analyze(self, query=None):
        if not self.project or self.task_thread is not None:
            return
        try:
            settings = self.search_settings()
            settings.query = query if isinstance(query, str) else ""
            settings.validate()
            self.close_assistant_dialog()
            self.persist_settings()
            project = copy.deepcopy(self.project)
            cpu_tools, cache = copy.copy(self.tools), self.cache
            def execute(token, progress):
                started = time.monotonic()
                tools, backend_warnings = prepare_backend(cpu_tools, token, progress)
                try:
                    segments, energy, warnings = recognize(project.media, tools, cache, token, progress, project.segments,
                                                           existing_energy=project.audio_energy)
                except RuntimeError:
                    if not tools.gpu:
                        raise
                    tools.whisper, tools.gpu = cpu_tools.whisper, False
                    progress(0, "Whisper: Vulkan не запустился, продолжаем на CPU…")
                    segments, energy, warnings = recognize(project.media, tools, cache, token, progress, project.segments,
                                                           existing_energy=project.audio_energy)
                    backend_warnings.append("Whisper не запустился на Vulkan; распознавание выполнено на CPU.")
                speech_seconds = time.monotonic() - started
                stage = time.monotonic()
                result = LocalModelProvider(tools, cpu_tools).analyze(project.media, segments, settings, cache, token, progress)
                candidates = result.candidates
                warnings = [*warnings, *backend_warnings, *result.warnings]
                project.analysis_info = {**result.info, "speech_seconds": round(speech_seconds, 2),
                    "selection_seconds": round(time.monotonic() - stage, 2), "total_seconds": round(time.monotonic() - started, 2),
                    "speech_backend": "Vulkan" if tools.gpu else "CPU", "network_used": False}
                shortfall = (("" if settings.query else shortfall_message(project.media, settings, candidates, bool(segments))) if result.info.get("complete", True)
                    else f"Проверено {result.info.get('checked_windows', 0)} из {result.info.get('total_windows', 0)} участков. Количество найденных клипов промежуточное; повторный поиск продолжит пропущенные участки.")
                if shortfall:
                    warnings.append(shortfall)
                project.segments, project.candidates, project.search = segments, candidates, settings
                project.audio_energy, project.warnings = energy, warnings
                project.validate()
                return project
            self.analyze_button.setText("Поиск выполняется…")
            self.start_task(execute, self.query_search_ready if settings.query else self.analysis_ready)
        except (ValueError, RuntimeError) as exc:
            self.show_error(exc)

    def close_assistant_dialog(self):
        if self.assistant_dialog is not None:
            self.assistant_dialog.close()
            self.assistant_dialog = None

    def open_suggestions(self, candidates, heading, mode, apply):
        self.close_assistant_dialog()
        dialog = ClipSuggestionsDialog(self, candidates, heading, mode, self.preview_suggestion, self.player.pause, apply)
        self.assistant_dialog = dialog
        def released(*_):
            if self.assistant_dialog is dialog:
                self.assistant_dialog = None
        dialog.destroyed.connect(released)
        dialog.show()

    def preview_suggestion(self, candidate):
        if not self.project or candidate.source_file_id != self.project.media.source_file_id:
            return
        self.player.pause()
        self.preview_end = candidate.end_ms
        self.player.setPosition(candidate.start_ms)
        self.still_timer.stop()
        for _, token in self.poster_jobs:
            token.cancel()
        self.player.play()

    def find_by_content(self):
        query = self.content_query.text().strip()
        if not query:
            self.show_error("Введите, какие моменты найти: например, «где спорят с учителем».")
            return
        self.analyze(query)

    def query_search_ready(self, result):
        if not self.project or self.project.media.source_file_id != result.media.source_file_id:
            return
        self.project.segments, self.project.audio_energy = result.segments, result.audio_energy
        self.project.query_results = result.candidates
        self.project.last_query = result.search.query
        self.project.query_info = {**result.analysis_info, "warnings": result.warnings,
                                   "transcript_digest": self.transcript_digest(result.segments)}
        self.refresh_transcript()
        self.changed()
        self.progress.setValue(100)
        state = "Поиск по содержанию завершён" if result.analysis_info.get("complete", True) else "Поиск по содержанию завершён частично"
        self.status.setText(f"{state}: {len(result.candidates)} совпадений. Основные эпизоды сохранены.")
        self.show_query_results()

    def show_query_results(self):
        if not self.project or not self.project.last_query:
            self.show_error("Сначала выполните поиск по содержанию.")
            return
        source_id = self.project.media.source_file_id
        expected_segments = copy.deepcopy(self.project.segments)
        searched_digest = self.project.query_info.get("transcript_digest")
        stale = searched_digest is not None and searched_digest != self.transcript_digest(expected_segments)
        query = self.project.last_query
        warnings = self.project.query_info.get("warnings", [])
        heading = f"Запрос: «{query}». Найдено {len(self.project.query_results)} совпадений. Основные эпизоды не заменяются."
        if warnings:
            heading += "\n" + "\n".join(warnings[:3])
        if stale:
            heading += "\nРасшифровка изменена. Для добавления совпадений повторите поиск."
        if not self.project.query_results:
            heading += "\nСовпадений с заданной оценкой интересности и соответствием запросу от 70/100 нет."
        def add(candidates):
            if (stale or not self.project or self.project.media.source_file_id != source_id or self.project.segments != expected_segments
                or self.project.last_query != query):
                self.show_error("Видео, расшифровка или запрос изменены. Повторите поиск по содержанию.")
                return False
            known = {(c.start_ms, c.end_ms) for c in self.project.candidates}
            added = 0
            for candidate in candidates:
                if (candidate.start_ms, candidate.end_ms) not in known:
                    c = copy.deepcopy(candidate)
                    c.id, c.selected = uuid4().hex, True
                    self.project.candidates.append(c)
                    known.add((c.start_ms, c.end_ms))
                    added += 1
            self.visible_count = max(self.visible_count, len(self.project.candidates))
            self.refresh_candidates()
            self.changed()
            self.status.setText(f"Добавлено эпизодов из поиска по содержанию: {added}.")
            return True
        self.open_suggestions(copy.deepcopy(self.project.query_results), heading, "query", add)

    @staticmethod
    def transcript_digest(segments):
        text = json.dumps([(s.start_ms, s.end_ms, s.text) for s in segments], ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def refine_selected(self):
        anchor = self.current_candidate()
        if not self.project or self.task_thread is not None:
            return
        if anchor is None:
            self.show_error("Выберите клип в таблице эпизодов.")
            return
        try:
            settings = self.search_settings()
            project, original = copy.deepcopy(self.project), copy.deepcopy(anchor)
            cpu_tools, cache = copy.copy(self.tools), self.cache
            self.close_assistant_dialog()
            def execute(token, progress):
                tools, warnings = prepare_backend(cpu_tools, token, progress)
                try:
                    segments, energy, asr_warnings = recognize(project.media, tools, cache, token, progress, project.segments, existing_energy=project.audio_energy)
                except RuntimeError:
                    if not tools.gpu:
                        raise
                    tools.whisper, tools.gpu = cpu_tools.whisper, False
                    segments, energy, asr_warnings = recognize(project.media, tools, cache, token, progress, project.segments, existing_energy=project.audio_energy)
                    warnings.append("Whisper обработан на CPU после ошибки Vulkan.")
                result = LocalClipAssistant(tools, cpu_tools).refine(project.media, segments, original, settings, cache, token, progress)
                result.warnings = [*warnings, *asr_warnings, *result.warnings]
                return result, segments, energy
            self.start_task(execute, lambda value: self.boundaries_ready(original, project.media.source_file_id, *value))
        except (ValueError, RuntimeError) as exc:
            self.show_error(exc)

    def boundaries_ready(self, anchor, source_id, result, segments, energy):
        if not self.project or self.project.media.source_file_id != source_id:
            return
        self.project.segments, self.project.audio_energy = segments, energy
        self.refresh_transcript()
        self.changed()
        expected_segments = copy.deepcopy(segments)
        heading = f"Текущий клип: {timecode(anchor.start_ms)}–{timecode(anchor.end_ms)}. Сдвиг каждой границы — не более 30 секунд."
        if result.info.get("current_recommended"):
            heading += "\nМодель рекомендует также текущие границы."
        if not result.candidates:
            heading += "\nПодходящих альтернатив не предложено. Текущий клип сохранён."
        if result.warnings:
            heading += "\n" + "\n".join(result.warnings[:3])
        def apply(candidates):
            target = next((c for c in self.project.candidates if c.id == anchor.id), None) if self.project else None
            if (target is None or self.project.media.source_file_id != source_id or self.project.segments != expected_segments
                or (target.start_ms, target.end_ms) != (anchor.start_ms, anchor.end_ms)):
                self.show_error("Клип или расшифровка изменены. Подберите границы заново.")
                return False
            self.boundary_undo.append(copy.deepcopy(target))
            chosen = candidates[0]
            for name in ("start_ms", "end_ms", "score", "reason", "category", "transcript_excerpt", "provider", "version", "match_score"):
                setattr(target, name, getattr(chosen, name))
            target.validate(self.project.media)
            self.refresh_candidates()
            self.changed()
            self.status.setText("Новые границы применены. Кнопка «Отменить границы» вернёт прежний вариант.")
            return True
        self.progress.setValue(100)
        self.status.setText(f"Подбор границ завершён: {len(result.candidates)} альтернатив. Текущий клип сохранён до применения.")
        self.open_suggestions(result.candidates, heading, "bounds", apply)

    def undo_boundaries(self):
        if not self.project or not self.boundary_undo:
            self.show_error("Нет применённых изменений границ для отмены в этом сеансе.")
            return
        previous = self.boundary_undo.pop()
        target = next((c for c in self.project.candidates if c.id == previous.id), None)
        if target is None:
            self.show_error("Исходный клип больше не находится в проекте.")
            return
        selected, layout = target.selected, target.export_layout
        for name, value in vars(previous).items():
            setattr(target, name, copy.deepcopy(value))
        target.selected = selected
        target.export_layout = layout
        self.refresh_candidates()
        self.changed()
        self.status.setText("Прежние границы клипа восстановлены.")

    def analysis_ready(self, project):
        self.project = project
        self.boundary_undo.clear()
        self.visible_count = project.search.count
        self.refresh_candidates()
        self.refresh_transcript()
        self.changed()
        self.progress.setValue(100)
        heading = "Анализ завершён" if project.analysis_info.get("complete", True) else "Анализ завершён частично"
        self.status.setText(f"{heading}: {len(project.candidates)} клипов, оценка от {project.search.min_score}. "
                            f"Цель: {project.search.target_count}. " + " ".join(project.warnings))
        if project.warnings:
            QMessageBox.information(self, "Качество поиска", "\n\n".join(project.warnings))

    def update_local_hint(self):
        path = self.tools.semantic_model
        self.local_hint.setText(("Модель: " + Path(path).name if path and Path(path).is_file() else
            "Qwen не установлена. Скачайте GGUF-файл из локального выпуска и укажите его здесь.") +
            "\nРаспознавание и смысловой отбор выполняются на вашем ПК. Интернет и ключи не требуются.")

    def choose_semantic_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Локальная модель Qwen", self.tools.semantic_model, "Модель GGUF (*.gguf)")
        if path:
            self.tools.semantic_model = path
            self.persist_settings()
            self.update_local_hint()

    def show_analysis_report(self):
        if not self.project or not self.project.analysis_info:
            QMessageBox.information(self, "Отчёт анализа", "Отчёт появится после следующего поиска.")
            return
        import json
        info = self.project.analysis_info
        text = (f"Режим: {info.get('mode', '?')}\nВсего: {info.get('total_seconds', '?')} с\n"
                f"Распознавание: {info.get('speech_backend', '?')}; Qwen: {info.get('llm_backend', '?')}\n\n" +
                json.dumps(info, ensure_ascii=False, indent=2))
        QMessageBox.information(self, "Отчёт анализа · время и покрытие", text)

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
        self.refine_button.setEnabled(self.current_candidate() is not None)
        self.undo_boundaries_button.setEnabled(bool(self.boundary_undo))
        self.export_layout_changed()

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
            self.export_layout_changed()
        except ValueError as exc:
            c.start_ms, c.end_ms, c.selected = previous
            self.show_error(exc)
            self.refresh_candidates()

    def candidate_selected(self):
        c = self.current_candidate()
        self.refine_button.setEnabled(c is not None)
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
            plan = []
            for c in selected:
                try:
                    effective = clip_export_settings(settings, c)
                    effective.require_gaming_regions()
                except ValueError as exc:
                    raise ValueError(f"Клип {timecode(c.start_ms)}–{timecode(c.end_ms)}: {exc}") from exc
                plan.append((c, effective))
            if not self.export_folder.text().strip():
                raise ValueError("Выберите папку результата")
            self.output = Path(self.export_folder.text()).resolve()
            self.project.export = settings
            self.changed()
            self.persist_settings()
            project, tools, output = copy.deepcopy(self.project), copy.copy(self.tools), self.output
            self.queue.clear()
            for i, (c, effective) in enumerate(plan, 1):
                width, height = effective.dimensions()
                self.queue.append(f"{i}. {timecode(c.start_ms)} → {timecode(c.end_ms)} · {EXPORT_LAYOUTS[effective.layout]} · {width}×{height} — в очереди")
            def execute(token, progress):
                results = []
                for index, (c, effective) in enumerate(plan):
                    token.check()
                    path = export_clip(project.media, c, project.segments, effective, output, tools, token,
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
        write_settings(self.tools, str(self.cache), str(self.output))




    def settings(self):
        if self.task_thread is not None:
            return
        dialog = SettingsDialog(self)
        if dialog.exec() == QDialog.Accepted:
            values = {name: line.text().strip() for name, line in dialog.fields.items()}
            for name in ("ffmpeg", "ffprobe", "whisper", "model", "llama", "semantic_model"):
                setattr(self.tools, name, values[name])
            self.tools.threads = dialog.threads.value()
            self.tools.acceleration = "cpu" if dialog.acceleration.currentIndex() else "auto"
            self.tools.gpu = False
            self.cache = Path(values["cache"] or str(data_root() / "cache"))
            self.output = Path(values["output"] or self.export_folder.text())
            self.export_folder.setText(str(self.output))
            try:
                self.persist_settings()
                self.update_local_hint()
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
            try:
                from .local_llm import validate_tools
                validate_tools(tools)
                run([tools.llama, "--version"], token, timeout=20)
                result.append("llama.cpp запускается, GGUF-модель Qwen найдена.")
            except (ValueError, RuntimeError) as exc:
                result.append(str(exc))
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
                self.status.setText("Проект открыт. Повторный анализ не запускался.")
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
        self.close_export_preview()
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
