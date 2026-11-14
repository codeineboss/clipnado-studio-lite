"""Review local-model suggestions before changing the project."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QHeaderView, QLabel, QPushButton,
                              QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout)
from .models import timecode


class ClipSuggestionsDialog(QDialog):
    def __init__(self, parent, candidates, heading, mode, preview, pause, apply):
        super().__init__(parent)
        self.candidates, self.mode, self.apply_callback = candidates, mode, apply
        self.setWindowTitle("Поиск по содержанию" if mode == "query" else "Соседние границы клипа")
        self.resize(950, 570)
        self.setAttribute(Qt.WA_DeleteOnClose)
        layout = QVBoxLayout(self)
        label = QLabel(heading)
        label.setWordWrap(True)
        label.setTextFormat(Qt.PlainText)
        layout.addWidget(label)
        self.table = QTableWidget(len(candidates), 6)
        self.table.setHorizontalHeaderLabels(["✓", "Начало", "Конец", "Оценка", "Соответствие", "Почему выбран"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.setColumnHidden(0, mode != "query")
        self.table.setColumnHidden(4, not any(c.match_score is not None for c in candidates))
        for i, c in enumerate(candidates):
            for j, text in enumerate(("", timecode(c.start_ms), timecode(c.end_ms), str(c.score),
                                      "—" if c.match_score is None else f"{c.match_score}/100", c.reason)):
                item = QTableWidgetItem(text)
                if j == 0:
                    item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    item.setCheckState(Qt.Checked)
                item.setToolTip(c.reason + "\n" + c.transcript_excerpt)
                self.table.setItem(i, j, item)
        self.table.setColumnWidth(0, 35)
        self.table.setColumnWidth(1, 115)
        self.table.setColumnWidth(2, 115)
        layout.addWidget(self.table, 1)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(140)
        layout.addWidget(self.details)
        row = QHBoxLayout()
        self.preview_button = QPushButton("▶ Просмотреть")
        self.preview_button.clicked.connect(lambda: preview(self.current()) if self.current() else None)
        row.addWidget(self.preview_button)
        pause_button = QPushButton("⏸ Пауза")
        pause_button.clicked.connect(pause)
        row.addWidget(pause_button)
        row.addStretch()
        self.apply_button = QPushButton("Добавить отмеченные" if mode == "query" else "Применить границы")
        self.apply_button.clicked.connect(self.apply_selected)
        row.addWidget(self.apply_button)
        close = QPushButton("Закрыть")
        close.clicked.connect(self.reject)
        row.addWidget(close)
        layout.addLayout(row)
        self.table.itemSelectionChanged.connect(self.selection_changed)
        self.table.itemChanged.connect(self.selection_changed)
        if candidates:
            self.table.selectRow(0)
        self.selection_changed()

    def current(self):
        i = self.table.currentRow()
        return self.candidates[i] if 0 <= i < len(self.candidates) else None

    def selection_changed(self, *_):
        c = self.current()
        self.details.setPlainText(c.reason + "\n\n" + c.transcript_excerpt if c else "Подходящих вариантов нет.")
        self.preview_button.setEnabled(c is not None)
        self.apply_button.setEnabled(bool(self.selected_candidates()))

    def selected_candidates(self):
        if self.mode != "query":
            return [self.current()] if self.current() else []
        return [c for i, c in enumerate(self.candidates) if self.table.item(i, 0).checkState() == Qt.Checked]

    def apply_selected(self):
        candidates = self.selected_candidates()
        if candidates and self.apply_callback(candidates):
            self.accept()
