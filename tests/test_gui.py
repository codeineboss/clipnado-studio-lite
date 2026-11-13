import time
import subprocess
from pathlib import Path

import pytest
from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from clipnado.app import MainWindow, STYLE
from clipnado.models import Segment


@pytest.fixture(scope="session")
def app():
    result = QApplication.instance() or QApplication([])
    result.setStyleSheet(STYLE)
    return result


@pytest.fixture
def window(app, monkeypatch, tmp_path):
    monkeypatch.setenv("CLIPNADO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(QMessageBox, "information", lambda *_: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.Yes)
    w = MainWindow()
    w.errors = []
    w.show_error = w.errors.append
    w.show()
    app.processEvents()
    yield w
    assert w.task_thread is None
    w.dirty = False
    w.close()


def wait_task(app, window, seconds=30):
    limit = time.monotonic() + seconds
    while window.task_thread is not None and time.monotonic() < limit:
        app.processEvents()
        QTest.qWait(15)
    app.processEvents()
    assert window.task_thread is None, "Task did not finish"
    assert not window.errors, window.errors


def test_gui_manual_export_edit_and_save(app, window, media, tmp_path, monkeypatch):
    window.media_loaded(media)
    window.minimum.setValue(2)
    window.add_manual()
    assert window.table.rowCount() == 1
    window.table.item(0, 2).setText("00:00:03.000")
    window.table.item(0, 1).setText("00:00:01.000")
    assert window.project.candidates[0].start_ms == 1000
    assert window.project.candidates[0].end_ms == 3000
    assert window.project.candidates[0].selected
    window.project.segments = [Segment(1000, 3000, "Привет, это проверка графического интерфейса.")]
    window.refresh_transcript()
    window.transcript.item(0, 2).setText("Исправленный текст")
    assert window.project.segments[0].text == "Исправленный текст"
    window.resolution.setCurrentText("720")
    window.subtitle_mode.setCurrentText("Отдельный SRT")
    window.export_folder.setText(str(tmp_path / "results"))
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(10)
    window.export_selected()
    assert not window.import_button.isEnabled()
    assert window.cancel_button.isEnabled()
    wait_task(app, window)
    timer.stop()
    assert len(ticks) >= 5, "GUI was blocked during export"
    assert list((tmp_path / "results").glob("*.mp4"))
    assert "Исправленный" in next((tmp_path / "results").glob("*.srt")).read_text(encoding="utf-8-sig")
    # Change clip limits and export again, reusing transcript and no provider call.
    window.table.item(0, 2).setText("00:00:02.500")
    window.export_selected()
    wait_task(app, window)
    assert len(list((tmp_path / "results").glob("*.mp4"))) == 2
    path = tmp_path / "saved.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_: (str(path), ""))
    window.maximum.setValue(4)
    window.save()
    wait_task(app, window)
    assert path.exists() and not window.dirty
    window.load_project_path(path)
    wait_task(app, window)
    assert window.project.candidates[0].end_ms == 2500


def test_gui_sort_selection_and_invalid_limits(app, window, media):
    window.media_loaded(media)
    window.minimum.setValue(1)
    window.add_manual()
    window.player.setPosition(3000)
    window.add_manual()
    window.project.candidates[0].score = 20
    window.project.candidates[1].score = 90
    window.visible_count = 1
    window.refresh_candidates()
    assert window.table.item(0, 3).text() == "90", "Initial chronological view must include the highest-ranked clips"
    window.visible_count = 2
    window.sort.setCurrentText("По рейтингу")
    assert window.table.item(0, 3).text() == "90"
    window.select_all(False)
    assert not any(c.selected for c in window.project.candidates)
    window.table.item(0, 0).setCheckState(Qt.Checked)
    assert sum(c.selected for c in window.project.candidates) == 1
    old = window.project.candidates[1].start_ms
    window.table.item(0, 1).setText("00:00:59.000")
    assert window.errors
    assert window.project.candidates[1].start_ms == old
    window.errors.clear()


def test_many_consecutive_background_tasks_release_threads(app, window):
    results = []
    for index in range(40):
        window.start_task(lambda token, progress, value=index: value, results.append)
        wait_task(app, window)
    assert results == list(range(40))


def test_local_only_ui_and_analysis_work_without_network(app, window, media, tmp_path, monkeypatch):
    import socket
    from PySide6.QtWidgets import QLineEdit, QComboBox
    window.media_loaded(media)
    window.minimum.setValue(1)
    window.maximum.setValue(8)
    assert not hasattr(window, "key") and not hasattr(window, "provider")
    assert not any("Gemini" in c.currentText() or "Groq" in c.currentText() for c in window.findChildren(QComboBox))
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF-fixture")
    window.tools.semantic_model = str(model)
    import sys
    window.tools.llama = sys.executable
    window.tools.acceleration = "cpu"
    segments = [Segment(0, 1000, "Вопрос ученика."), Segment(1000, 3000, "Необычный ответ учителя."), Segment(3000, 5000, "Ответная реплика.")]
    monkeypatch.setattr("clipnado.app.recognize", lambda *_a, **_k: (segments, [], []))
    def forbidden(*_a, **_kw):
        pytest.fail("Local UI analysis attempted a network connection")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr("clipnado.local_llm.run", lambda *_a, **_kw:
        '{"clips":[{"clip_id":0,"hook":20,"event":20,"emotion":20,"completeness":20,"category":"Юмор","reason":"Вопрос и ответ"}]}')
    window.analyze()
    wait_task(app, window)
    assert window.project.candidates and window.project.analysis_info["network_used"] is False
    assert window.project.analysis_info["mode"] == "local-qwen"


def test_goal_controls_persist_and_analysis_reports_shortfall(app, window, media, tmp_path, monkeypatch):
    window.media_loaded(media)
    assert window.target_count.value() == 10 and window.min_score.value() == 60
    window.target_count.setValue(20)
    assert window.count.value() == 20
    window.count.setValue(15)
    assert window.target_count.value() == 15
    window.target_count.setValue(10)
    window.min_score.setValue(65)
    path = tmp_path / "goal.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_: (str(path), ""))
    window.save()
    wait_task(app, window)
    window.min_score.setValue(60)
    window.load_project_path(path)
    wait_task(app, window)
    assert window.target_count.value() == 10 and window.min_score.value() == 65
    monkeypatch.setattr("clipnado.app.recognize", lambda *_args, **_kwargs: ([], [], []))
    monkeypatch.setattr("clipnado.app.prepare_backend", lambda tools, *_: (tools, []))
    window.analyze()
    wait_task(app, window)
    assert not window.project.candidates
    assert any("0 из целевых 10" in warning for warning in window.project.warnings)
    assert "целевых 10" in window.status.text()






def wait_until(app, predicate, seconds=15):
    limit = time.monotonic() + seconds
    while not predicate() and time.monotonic() < limit:
        app.processEvents()
        QTest.qWait(20)
    assert predicate()


def test_import_shows_real_pixels_without_candidates_and_can_play_during_analysis(app, window, media):
    window.media_loaded(media)
    assert window.table.rowCount() == 0
    wait_until(app, lambda: not window.video.image.isNull() and not window.poster_jobs)
    color = window.video.image.pixelColor(10, 10)
    assert color.red() > 180 and color.green() < 40
    assert window.video.pixmap() and not window.video.pixmap().isNull()
    window.start_task(lambda token, progress: (progress(0, "Whisper CPU: ждём первый результат"), token.wait(3)), lambda *_: None)
    assert window.progress.minimum() == window.progress.maximum() == 0
    assert not window.import_button.isEnabled()
    assert window.tabs.isEnabled() and window.play_button.isEnabled()
    QTest.mouseClick(window.play_button, Qt.LeftButton)
    wait_until(app, lambda: window.video.frame_count > 0 and window.player.position() > 100)
    window.player.pause()
    window.seek(2400)
    wait_until(app, lambda: not window.still_timer.isActive() and not window.poster_jobs
               and window.video.image.pixelColor(10, 10).green() > 180)
    wait_task(app, window)


def test_1080p_60fps_preview_with_cyrillic_filename(app, window, tmp_path, tools):
    from clipnado.media import probe
    path = tmp_path / "60 кадров кириллица.mp4"
    subprocess.run([tools.ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=c=lime:s=1920x1080:r=60:d=2",
                    "-c:v", "libx264", "-preset", "ultrafast", "-threads", "2", "-pix_fmt", "yuv420p", str(path)], check=True)
    media = probe(path, tools)
    assert media.width == 1920 and media.height == 1080 and round(media.fps) == 60
    window.media_loaded(media)
    wait_until(app, lambda: not window.video.image.isNull() and not window.poster_jobs)
    assert window.video.image.pixelColor(10, 10).green() > 180
    QTest.mouseClick(window.play_button, Qt.LeftButton)
    wait_until(app, lambda: window.video.frame_count > 0 and window.video.image.width() == 1920)
    assert window.video.image.pixelColor(10, 10).green() > 180
    window.player.pause()
