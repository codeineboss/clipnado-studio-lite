import time
import subprocess
from pathlib import Path

import pytest
from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from clipnado.app import MainWindow, configure_application
from clipnado.models import Segment


@pytest.fixture(scope="session")
def app():
    result = QApplication.instance() or QApplication([])
    configure_application(result)
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
    assert not list((tmp_path / "results").glob("*.srt"))
    assert window.project.segments[0].text == "Исправленный текст"
    assert not hasattr(window, "subtitle_mode") and not hasattr(window, "readable_subtitles")
    # Change clip limits and export again, reusing transcript and no provider call.
    window.table.item(0, 2).setText("00:00:02.500")
    window.export_selected()
    wait_task(app, window)
    assert len(list((tmp_path / "results").glob("*.mp4"))) == 2
    path = tmp_path / "saved.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_: (str(path), ""))
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


def test_local_only_ui_and_analysis_work_without_network(app, window, long_media, tmp_path, monkeypatch):
    media=long_media
    import socket
    from PySide6.QtWidgets import QLineEdit, QComboBox
    window.media_loaded(media)
    window.minimum.setValue(1)
    assert not hasattr(window, "key") and not hasattr(window, "provider")
    assert not any("Gemini" in c.currentText() or "Groq" in c.currentText() for c in window.findChildren(QComboBox))
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF-fixture")
    window.tools.semantic_model = str(model)
    import sys
    window.tools.llama = sys.executable
    window.tools.acceleration = "cpu"
    monkeypatch.setattr("clipnado.local_llm.prompt_tokens", lambda *_a, **_k: 1000)
    segments = [Segment(0, 15000, "Вопрос ученика."), Segment(15000, 35000, "Необычный ответ учителя."), Segment(35000, 55000, "Ответная реплика.")]
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
    assert all(c.end_ms-c.start_ms>=15000 for c in window.project.candidates)
    assert window.search_settings().max_seconds == 90


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
    window.minimum.setValue(30)
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
    window.minimum.setValue(30); window.add_manual()
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


def test_mixed_export_uses_individual_layouts_and_saves_choices(app, window, media, tmp_path, monkeypatch):
    from clipnado.models import Candidate
    from clipnado.media import probe
    from test_export import frame
    window.media_loaded(media)
    clips = [Candidate(media.source_file_id, start, start+1000, 80-i, 'Юмор', 'Момент', selected=True)
             for i, start in enumerate((0, 1000, 4000))]
    window.project.candidates = clips
    window.visible_count = 1  # export includes all checked clips, including those hidden in the main table
    window.refresh_candidates()
    assert window.table.rowCount() == 1 and window.export_clips.rowCount() == 3
    window.export_clips.cellWidget(1, 3).setCurrentIndex(1)
    window.export_clips.cellWidget(2, 3).setCurrentIndex(2)
    window.gaming_camera = [.75,0,.25,1/3]
    window.gaming_game = [0,1/3,1,2/3]
    assert [c.export_layout for c in clips] == [None, 'blur_center', 'gaming_split']
    window.aspect.setCurrentText('1:1')
    window.resolution.setCurrentText('720')
    window.export_folder.setText(str(tmp_path/'mixed'))
    monkeypatch.setattr('clipnado.app.LocalModelProvider.analyze', lambda *_a, **_k: pytest.fail('Export started a new analysis'))
    window.export_selected(); wait_task(app, window)
    paths = sorted((tmp_path/'mixed').glob('*.mp4'))
    assert len(paths) == 3
    assert [(probe(p, window.tools).width, probe(p, window.tools).height) for p in paths] == [(720,720),(720,720),(720,1280)]
    pixels = [frame(p, window.tools, .2) for p in paths]
    assert max(pixels[0][-720*100*3:]) <= 5  # reserved animation zone
    assert pixels[1][-3] > 180 and pixels[2][-3] > 180  # both modes fill the lower canvas
    assert 'Экспорт завершён: 3' in window.status.text()
    assert window.gaming_button.isEnabled()
    path=tmp_path/'individual-layouts.json'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *_: (str(path), ''))
    window.save(); wait_task(app, window)
    window.load_project_path(path); wait_task(app, window)
    assert window.gaming_camera == [.75,0,.25,1/3]
    assert window.project.export.gaming_game == [0,1/3,1,2/3]
    assert [c.export_layout for c in window.project.candidates] == [None,'blur_center','gaming_split']
    assert [window.export_clips.cellWidget(i,3).currentIndex() for i in range(3)] == [0,1,2]
    # Common defaults change inherited rows; explicit clip choices remain stable.
    window.export_layout.setCurrentIndex(2)
    assert [window.export_clips.cellWidget(i,3).currentIndex() for i in range(3)] == [2,1,2]
    window.apply_layout_to_selected()
    assert all(c.export_layout is None for c in window.project.candidates)
    assert all(window.export_clips.cellWidget(i,3).currentIndex()==2 for i in range(3))


def test_export_choices_follow_clip_identity_across_selection_and_sort(app, window, media):
    from clipnado.models import Candidate
    window.media_loaded(media)
    a=Candidate(media.source_file_id, 0, 1000, 60, 'Юмор', 'Первый', selected=True)
    b=Candidate(media.source_file_id, 4000, 5000, 90, 'Юмор', 'Второй', selected=True)
    window.project.candidates=[a,b];window.visible_count=2;window.refresh_candidates()
    window.export_clips.cellWidget(0,3).setCurrentIndex(2)
    window.sort.setCurrentText('По рейтингу')
    assert a.export_layout=='gaming_split' and b.export_layout is None
    window.table.item(1,0).setCheckState(Qt.Unchecked)
    assert window.export_clips.rowCount()==1
    assert window.export_clips.item(0,0).data(Qt.UserRole)==b.id
    window.table.item(1,0).setCheckState(Qt.Checked)
    assert window.export_clips.rowCount()==2 and window.export_clips.cellWidget(0,3).currentIndex()==2


def test_mixed_export_validates_all_clips_before_creating_files(app, window, media, tmp_path):
    from clipnado.models import Candidate
    window.media_loaded(media)
    window.project.candidates=[
        Candidate(media.source_file_id,0,1000,80,'Юмор','Фон',selected=True,export_layout='blur_center'),
        Candidate(media.source_file_id,4000,5000,80,'Юмор','Зона',selected=True,export_layout='top_black')]
    window.project.segments=[Segment(0,8000,'Текст')]
    window.refresh_candidates()
    window.export_layout.setCurrentIndex(1)
    assert window.top.isEnabled(), 'An individually selected upper-zone layout still needs its controls'
    window.top.setValue(80);window.video_y.setValue(50)
    window.export_folder.setText(str(tmp_path/'invalid-mixed'))
    window.export_selected()
    assert window.task_thread is None and window.errors and '00:00:04.000' in str(window.errors[0])
    assert not list((tmp_path/'invalid-mixed').glob('*.mp4'))
    window.errors.clear()


def test_export_framing_preview_uses_chosen_clip_and_mode(app, window, media):
    from clipnado.models import Candidate
    from clipnado.preview import VideoPreview
    window.media_loaded(media)
    window.project.candidates=[Candidate(media.source_file_id,0,1000,80,'Юмор','Кадр',selected=True,export_layout='gaming_split')]
    window.refresh_candidates()
    window.gaming_camera = [.75,0,.25,.3]
    window.preview_export_clip();wait_task(app,window)
    dialog=window.export_preview_dialog
    assert dialog and dialog.isVisible()
    preview=dialog.findChild(VideoPreview)
    assert (preview.image.width(),preview.image.height())==(360,640)
    assert preview.image.pixelColor(10,10).red()>180
    window.export_clips.cellWidget(0,3).setCurrentIndex(1)
    assert window.export_preview_dialog is None


def test_boundary_undo_preserves_later_export_choice(app, window, media):
    import copy
    from clipnado.models import Candidate
    from clipnado.local_llm import LocalAnalysisResult
    window.media_loaded(media)
    window.minimum.setValue(2);window.add_manual()
    anchor=copy.deepcopy(window.project.candidates[0])
    alternative=Candidate(media.source_file_id,1000,4000,80,'Юмор','Границы')
    window.boundaries_ready(anchor,media.source_file_id,LocalAnalysisResult([alternative]),[Segment(0,8000,'Диалог')],[])
    window.assistant_dialog.apply_selected();app.processEvents()
    window.export_clips.cellWidget(0,3).setCurrentIndex(2)
    window.undo_boundaries()
    clip=window.project.candidates[0]
    assert (clip.start_ms,clip.end_ms)==(anchor.start_ms,anchor.end_ms)
    assert clip.export_layout=='gaming_split'


def test_gaming_region_editor_drag_apply_save_and_cancel(app,window,gaming_source,tools,tmp_path,monkeypatch):
    from PySide6.QtCore import QPoint
    from clipnado.gaming import GamingRegionsDialog
    from clipnado.media import probe
    from clipnado.projects import load_project
    from clipnado.models import Candidate
    window.media_loaded(probe(gaming_source,tools))
    window.project.candidates=[Candidate(window.project.media.source_file_id,1000,4000,80,'Реакция','Игра',selected=True,export_layout='gaming_split')]
    window.refresh_candidates();assert window.gaming_button.isEnabled()
    assert not hasattr(window,'tiktok_tracking')
    seen=[]
    timer=QTimer()
    def edit():
        dialog=window.findChild(GamingRegionsDialog)
        if dialog is None or dialog.worker is not None or dialog.canvas.image.isNull():return
        canvas=dialog.canvas;r=canvas.image_rect()
        def point(x,y):return QPoint(round(r.x()+r.width()*x),round(r.y()+r.height()*y))
        # Draw an entirely new camera away from the default corner. Coordinates
        # must account for the letterboxing of the original image in the widget.
        QTest.mousePress(canvas,Qt.LeftButton,pos=point(.15,.2))
        QTest.mouseMove(canvas,point(.4,.5),20)
        QTest.mouseRelease(canvas,Qt.LeftButton,pos=point(.4,.5))
        assert canvas.camera==pytest.approx([.15,.2,.25,.3],abs=.005)
        dialog.fraction.setValue(35)
        seen.append(dialog.regions());timer.stop();dialog.accept()
    timer.timeout.connect(edit);timer.start(30)
    window.configure_gaming();app.processEvents()
    assert seen and window.gaming_camera==seen[0][0]
    assert window.project.export.gaming_fraction==.35
    path=tmp_path/'gaming.json'
    monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *_:(str(path),''))
    window.save();wait_task(app,window)
    saved=load_project(path);assert saved.export.gaming_camera==window.gaming_camera
    before=window.export_settings()
    def cancel():
        dialog=window.findChild(GamingRegionsDialog)
        if dialog is not None and not dialog.closed:
            dialog.canvas.camera=[0,0,.2,.2];timer.stop();dialog.reject()
    timer.timeout.disconnect();timer.timeout.connect(cancel);timer.start(30)
    window.configure_gaming();app.processEvents()
    assert window.export_settings()==before
    # A new source has independent camera coordinates, never the previous stream's.
    window.media_loaded(window.project.media)
    assert window.gaming_camera is None


def test_gaming_mixed_queue_requires_regions_before_any_export(app,window,media,tmp_path):
    from clipnado.models import Candidate
    window.media_loaded(media)
    window.project.candidates=[Candidate(media.source_file_id,0,1000,80,'','',selected=True,export_layout='blur_center'),
                               Candidate(media.source_file_id,4000,5000,80,'','',selected=True,export_layout='gaming_split')]
    window.refresh_candidates();window.export_folder.setText(str(tmp_path/'no-partial-exports'))
    window.export_selected()
    assert window.task_thread is None and window.errors and 'Камера и игра' in str(window.errors[0])
    assert not list((tmp_path/'no-partial-exports').glob('*.mp4'))
    window.errors.clear()


def test_comfort_controls_preview_export_and_project_roundtrip(app, window, media, tmp_path, monkeypatch):
    from clipnado.models import Candidate
    from clipnado.projects import load_project, save_project
    from clipnado.preview import VideoPreview
    window.media_loaded(media)
    c=Candidate(media.source_file_id,1000,4000,80,'Юмор','Комфортный клип',selected=True)
    window.project.candidates=[c]
    window.project.segments=[Segment(0,8000,'Крупные субтитры на русском языке без потери слов и смысла.')]
    window.refresh_candidates()
    window.resolution.setCurrentText('720');window.color_mode.setCurrentIndex(1)
    window.normalize_audio.setChecked(True)
    window.background_dim.setValue(35)
    window.preview_export_clip();assert not window.comfort_group.isEnabled()
    wait_task(app,window)
    assert window.export_preview_dialog and not hasattr(window, "animation_mode")
    image=window.export_preview_dialog.findChild(VideoPreview).image
    assert image.pixelColor(180,500).value()<=5
    window.close_export_preview()
    window.export_folder.setText(str(tmp_path/'output'))
    window.export_selected();wait_task(app,window,60)
    assert not window.errors and len(list((tmp_path/'output').glob('*.mp4')))==1
    assert 'Проверенные файлы' in window.queue.toPlainText()
    path=tmp_path/'project.json';save_project(window.project,path)
    project=load_project(path);window.project=project;window.load_ui()
    assert window.color_mode.currentIndex()==1 and window.normalize_audio.isChecked()
    assert window.background_dim.value()==35


def test_search_profiles_new_defaults_change_without_analysis_and_restore_legacy(app,window,media,tmp_path):
    import json
    from clipnado.models import Candidate
    from clipnado.projects import load_project,save_project
    window.media_loaded(media)
    assert window.search_settings().profile=='mixed'
    assert window.minimum.value()==15 and window.search_settings().max_seconds == 90
    c=Candidate(media.source_file_id,0,3000,80,'Юмор','Существующий клип',selected=True)
    window.project.candidates=[c];window.refresh_candidates()
    window.search_profile.setCurrentIndex(window.search_profile.findData('quote'))
    assert window.minimum.value()==15 and window.search_settings().max_seconds == 60
    assert window.project.candidates==[c] and window.task_thread is None
    # Manual duration overrides survive roundtrip without reapplying defaults.
    window.minimum.setValue(35);window.project.search=window.search_settings()
    path=tmp_path/'profiles.json';save_project(window.project,path)
    window.project=load_project(path);window.load_ui()
    assert window.search_profile.currentData()=='quote' and window.minimum.value()==35
    data=window.project.to_dict();data['search'].pop('profile');data['search'].update(min_seconds=60,max_seconds=90)
    data['export'].pop('preset');path.write_text(json.dumps(data),encoding='utf-8')
    window.project=load_project(path);window.load_ui()
    assert window.search_profile.currentData()=='custom' and window.minimum.value()==60 and window.search_settings().max_seconds == 60
    data['search'].update(min_seconds=8,max_seconds=20);path.write_text(json.dumps(data),encoding='utf-8')
    window.project=load_project(path);window.load_ui()
    assert window.minimum.value()==15 and window.search_settings().max_seconds==60
    assert window.project.candidates[0].id==c.id and window.task_thread is None


def test_export_profiles_apply_preserve_individual_choices_and_roundtrip(app,window,media,tmp_path):
    from clipnado.models import Candidate
    from clipnado.projects import load_project,save_project
    from scripts.gaming_smoke import CAMERA,GAME
    window.media_loaded(media)
    window.project.segments=[Segment(0,media.duration_ms,'Проверяем полный короткий мем с реакцией.')]
    a=Candidate(media.source_file_id,1000,4000,80,'Юмор','Мем',selected=True)
    b=Candidate(media.source_file_id,4000,7000,80,'История','Другой формат',selected=True,export_layout='blur_center')
    window.project.candidates=[a,b];window.refresh_candidates()
    window.gaming_camera=CAMERA.copy();window.gaming_game=GAME.copy()
    window.boundary_undo=[('keep',a)]
    window.export_preset.setCurrentIndex(window.export_preset.findData('gaming'));window.apply_export_preset()
    settings=window.export_settings()
    assert settings.preset=='gaming' and settings.layout=='gaming_split' and settings.gaming_fraction==.35
    assert (settings.resolution,settings.fps,settings.quality)==(1080,30,18)
    assert settings.normalize_audio
    assert settings.gaming_camera==CAMERA and settings.gaming_game==GAME
    assert [c.export_layout for c in window.project.candidates]==[None,'blur_center'] and window.boundary_undo
    path=tmp_path/'preset.json';save_project(window.project,path)
    window.project=load_project(path);window.load_ui()
    assert window.export_preset.currentData()=='gaming' and window.applied_export_preset=='gaming'
    assert window.export_settings()==settings
    window.resolution.setCurrentText('720')
    assert window.export_settings().preset=='custom' and window.export_preset.currentData()=='custom'
    # A new preset fixes invalid intermediate zone controls, without losing ROIs.
    window.video_y.setValue(80)
    window.export_preset.setCurrentIndex(window.export_preset.findData('story'));window.apply_export_preset()
    assert window.export_settings().video_y==0 and window.export_settings().layout=='top_black'
    assert window.gaming_camera==CAMERA


def test_export_profile_real_mp4_clean_video_and_missing_camera_preflight(app,window,media,tmp_path):
    from clipnado.models import Candidate
    window.media_loaded(media)
    window.project.candidates=[Candidate(media.source_file_id,1000,4000,80,'Юмор','Мем',selected=True)]
    window.project.segments=[Segment(0,media.duration_ms,'Проверяем субтитры и готовый профиль экспорта.')]
    window.refresh_candidates();window.export_folder.setText(str(tmp_path/'out'))
    window.export_preset.setCurrentIndex(window.export_preset.findData('gaming'));window.apply_export_preset()
    window.export_selected()
    assert window.task_thread is None and window.errors and 'Камера и игра' in str(window.errors[-1])
    assert not list((tmp_path/'out').glob('*.mp4'));window.errors.clear()
    window.export_preset.setCurrentIndex(window.export_preset.findData('dialogue'));window.apply_export_preset()
    window.export_selected();wait_task(app,window,60)
    assert len(list((tmp_path/'out').glob('*.mp4')))==1
    assert window.project.export.preset=='dialogue' and not list((tmp_path/'out').glob('*.srt'))


def test_search_story_switch_updates_upper_limit_without_losing_clips(app,window,media):
    from clipnado.models import Candidate
    window.media_loaded(media)
    c=Candidate(media.source_file_id,0,3000,80,'Юмор','Существующий клип',selected=True)
    window.project.candidates=[c]
    window.minimum.setValue(25)
    window.search_profile.setCurrentIndex(window.search_profile.findData('story'))
    assert window.minimum.value()==25 and window.search_settings().max_seconds==90
    assert '90' in window.search_maximum.text()
    window.minimum.setValue(75)
    window.search_profile.setCurrentIndex(window.search_profile.findData('memes'))
    assert window.minimum.value()==60 and window.search_settings().max_seconds==60
    assert window.project.candidates==[c] and window.task_thread is None


def test_video_frames_are_deferred_until_gui_event_loop(app, window):
    from PySide6.QtGui import QImage
    from PySide6.QtMultimedia import QVideoFrame
    image = QImage(32,18,QImage.Format_RGB32)
    image.fill(0xffff0000)
    assert window.video.frame_count==0
    window.video_sink.setVideoFrame(QVideoFrame(image))
    # Frame painting must not re-enter a widget/table update synchronously.
    assert window.video.frame_count==0
    app.processEvents()
    assert window.video.frame_count==1
    assert window.video.image.pixelColor(0,0).red()==255


@pytest.mark.parametrize("image_size", [(360,640), (640,360), (400,400)])
@pytest.mark.parametrize("widget_size", [(400,800), (800,400)])
def test_tiktok_guides_fit_viewport_and_preserve_frame(app, image_size, widget_size):
    from PySide6.QtGui import QImage
    from clipnado.tiktok_preview import TikTokPreview

    preview = TikTokPreview()
    image = QImage(*image_size, QImage.Format_RGB32)
    image.fill(0xffff0000)
    preview.set_image(image)
    preview.resize(*widget_size)
    preview.show()
    app.processEvents()
    original_key = preview.image.cacheKey()
    plain = preview.grab().toImage()
    preview.set_guides_enabled(True)
    guided = preview.grab().toImage()
    assert guided != plain
    assert preview.image.cacheKey() == original_key and preview.image == image
    assert guided.pixelColor(guided.width()//2, guided.height()//2).red() == 255
    assert guided.pixelColor(guided.width()//2, guided.height()//2).green() == 0
    # All chrome stays inside the simulated portrait screen, including when
    # the preview widget itself is wider than the phone.
    if widget_size[0] > widget_size[1]:
        assert guided.pixelColor(10, guided.height()//2).red() == 0
        assert guided.pixelColor(guided.width()-10, guided.height()//2).red() == 0
    preview.set_guides_enabled(False)
    assert preview.grab().toImage() == plain
    preview.close()


@pytest.mark.parametrize("layout", ["top_black", "blur_center", "gaming_split"])
def test_tiktok_guides_toggle_and_real_export_stays_clean(app, window, media, tmp_path, layout):
    from PySide6.QtWidgets import QCheckBox
    from clipnado.models import Candidate
    from clipnado.tiktok_preview import TikTokPreview
    from test_export import frame

    window.media_loaded(media)
    window.project.candidates = [Candidate(media.source_file_id, 0, 1000, 80,
                                            "Юмор", "Кадр", selected=True, export_layout=layout)]
    window.gaming_camera = [.75,0,.25,.3]
    window.gaming_game = [0,0,.7,.9]
    window.resolution.setCurrentText('720')
    window.refresh_candidates()
    before = window.export_settings()
    window.preview_export_clip()
    wait_task(app, window)
    dialog = window.export_preview_dialog
    preview = dialog.findChild(TikTokPreview)
    toggle = dialog.findChild(QCheckBox, 'tiktok_guides')
    assert toggle.isChecked() and preview.guides_enabled
    assert not window.video.__class__ is TikTokPreview
    original = preview.image.copy()
    guided = preview.grab().toImage()
    toggle.setChecked(False)
    assert not preview.guides_enabled and preview.grab().toImage() != guided
    toggle.setChecked(True)
    assert preview.guides_enabled and preview.image == original
    assert window.export_settings() == before
    # Leave the chrome ON while creating a real MP4. No white guide text or
    # icons may appear in the solid-red/black source; each layout is checked.
    window.export_folder.setText(str(tmp_path / 'clean-export'))
    window.export_selected()
    wait_task(app, window)
    paths = list((tmp_path / 'clean-export').glob('*.mp4'))
    assert len(paths) == 1 and preview.guides_enabled
    pixels = frame(paths[0], window.tools, .2)
    assert not any(min(pixels[i:i+3]) > 200 for i in range(0, len(pixels), 3))
    assert preview.image == original
    window.close_export_preview()
