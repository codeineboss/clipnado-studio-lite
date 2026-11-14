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
    monkeypatch.setattr("clipnado.local_llm.prompt_tokens", lambda *_a, **_k: 1000)
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


def test_content_search_preserves_main_clips_and_adds_only_on_click(app, window, media, tmp_path, monkeypatch):
    import copy
    from clipnado.models import Candidate
    from clipnado.local_llm import LocalAnalysisResult
    window.media_loaded(media)
    window.minimum.setValue(1); window.maximum.setValue(8)
    window.add_manual()
    original = copy.deepcopy(window.project.candidates)
    segments = [Segment(0, 4000, "Не согласен с твоим объяснением."), Segment(4000, 8000, "Почему? Давай обсудим.")]
    result_clip = Candidate(media.source_file_id, 2000, 6000, 80, "История", "Совпадение по смыслу", match_score=90)
    calls = []
    monkeypatch.setattr("clipnado.app.prepare_backend", lambda tools, *_: (tools, []))
    monkeypatch.setattr("clipnado.app.recognize", lambda *_a, **_k: (segments, [], []))
    def analyze(self, m, segs, settings, *args):
        calls.append(settings.query)
        return LocalAnalysisResult([copy.deepcopy(result_clip)], [], {"complete": True, "mode": "local-qwen", "query": settings.query})
    monkeypatch.setattr("clipnado.app.LocalModelProvider.analyze", analyze)
    window.content_query.setText("где спорят")
    window.find_by_content(); wait_task(app, window)
    assert calls == ["где спорят"] and window.project.candidates == original
    assert window.project.query_results[0].match_score == 90 and window.project.last_query == "где спорят"
    dialog = window.assistant_dialog
    add = dialog.apply_callback
    dialog.apply_selected(); app.processEvents()
    assert len(window.project.candidates) == 2 and window.project.candidates[0] == original[0]
    assert window.project.candidates[1].selected and window.project.candidates[1].id != result_clip.id
    assert add([result_clip]) and len(window.project.candidates) == 2
    window.analyze(); wait_task(app, window)
    assert calls[-1] == "", "Typed content query must not change the normal search mode"
    assert window.project.last_query == "где спорят" and window.project.query_results


def test_boundary_review_apply_and_undo_keep_clip_identity(app, window, media):
    import copy
    from clipnado.models import Candidate
    from clipnado.local_llm import LocalAnalysisResult
    window.media_loaded(media)
    window.minimum.setValue(2); window.add_manual()
    anchor = copy.deepcopy(window.project.candidates[0])
    alternative = Candidate(media.source_file_id, 1000, 4000, 80, "Юмор", "Сохранена развязка")
    segments = [Segment(0, 8000, "Диалог для предпросмотра")]
    window.boundaries_ready(anchor, media.source_file_id, LocalAnalysisResult([alternative], [], {"complete": True}), segments, [])
    assert window.project.candidates[0] == anchor
    window.assistant_dialog.apply_selected(); app.processEvents()
    changed = window.project.candidates[0]
    assert changed.id == anchor.id and changed.selected == anchor.selected and changed.start_ms == 1000 and changed.end_ms == 4000
    window.undo_boundaries()
    assert window.project.candidates[0] == anchor


def test_stale_boundary_proposal_is_rejected(app, window, media):
    import copy
    from clipnado.models import Candidate
    from clipnado.local_llm import LocalAnalysisResult
    window.media_loaded(media)
    window.minimum.setValue(2); window.add_manual()
    anchor = copy.deepcopy(window.project.candidates[0])
    alternative = Candidate(media.source_file_id, 1000, 4000, 80, "Юмор", "Вариант")
    window.boundaries_ready(anchor, media.source_file_id, LocalAnalysisResult([alternative]), [Segment(0, 8000, "Диалог")], [])
    window.project.candidates[0].end_ms = 2500
    window.assistant_dialog.apply_selected()
    assert window.errors and window.project.candidates[0].end_ms == 2500 and not window.boundary_undo
    window.errors.clear()
    window.close_assistant_dialog()


def test_blur_mode_gui_export_and_project_save(app, window, media, tmp_path, monkeypatch):
    window.media_loaded(media)
    window.minimum.setValue(1); window.add_manual()
    assert window.export_layout.currentIndex() == 0 and window.top.isEnabled()
    window.export_layout.setCurrentIndex(1)
    assert not window.top.isEnabled() and not window.video_y.isEnabled()
    window.resolution.setCurrentText("720")
    window.export_folder.setText(str(tmp_path / "blurred"))
    window.export_selected(); wait_task(app, window)
    assert list((tmp_path / "blurred").glob("*.mp4")) and window.project.export.layout == "blur_center"
    target = tmp_path / "blur-project.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_: (str(target), ""))
    window.save(); wait_task(app, window)
    window.load_project_path(target); wait_task(app, window)
    assert window.export_layout.currentIndex() == 1 and window.project.export.layout == "blur_center"


def test_refinement_background_task_uses_transcript_and_requires_apply(app, window, media, monkeypatch):
    import copy
    from clipnado.models import Candidate
    from clipnado.local_llm import LocalAnalysisResult
    window.media_loaded(media)
    window.minimum.setValue(2); window.maximum.setValue(8); window.add_manual()
    window.table.selectRow(0)
    original = copy.deepcopy(window.project.candidates[0])
    window.project.segments = [Segment(0, 8000, "Уже исправленный текст")]
    calls = []
    monkeypatch.setattr("clipnado.app.prepare_backend", lambda tools, *_: (tools, []))
    def recognize(media, tools, cache, token, progress, existing, **_):
        calls.append(copy.deepcopy(existing))
        return existing, [], []
    monkeypatch.setattr("clipnado.app.recognize", recognize)
    alternative = Candidate(media.source_file_id, 1000, 4000, 80, "Юмор", "Более полный ответ")
    monkeypatch.setattr("clipnado.app.LocalClipAssistant.refine", lambda *_: LocalAnalysisResult([alternative], [], {"complete": True}))
    QTest.mouseClick(window.refine_button, Qt.LeftButton)
    wait_task(app, window)
    assert calls == [window.project.segments] and window.project.candidates == [original]
    assert window.assistant_dialog and window.assistant_dialog.current() == alternative
    window.assistant_dialog.reject()
    app.processEvents()
    assert window.project.candidates == [original]


def test_reopening_query_results_after_transcript_edit_rejects_addition(app, window, media):
    from clipnado.models import Candidate, Project, SearchSettings
    result = Project(media, segments=[Segment(0, 8000, "Старый текст")],
                     candidates=[Candidate(media.source_file_id, 0, 6000, 80, "Юмор", "Совпадение", match_score=90)],
                     search=SearchSettings(query="смешной ответ"))
    window.media_loaded(media)
    window.query_search_ready(result)
    window.close_assistant_dialog()
    window.project.segments[0].text = "Новый текст"
    window.show_query_results()
    window.assistant_dialog.apply_selected()
    assert window.errors and not window.project.candidates
    window.errors.clear()
    window.close_assistant_dialog()
