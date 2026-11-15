"""Verify the embedded Windows runtime offline, including one actual GGUF token."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--semantic-model", required=True, type=Path)
    args = parser.parse_args()
    root = args.folder.resolve()
    python = root / "runtime" / "python.exe"
    with tempfile.TemporaryDirectory(prefix="clipnado-offline-smoke-") as tmp:
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["CLIPNADO_DATA_DIR"] = str(Path(tmp) / "settings")
        env["CLIPNADO_QWEN_MODEL"] = str(args.semantic_model.resolve())
        diagnostic = subprocess.run([str(python), str(root / "scripts" / "diagnostics.py")], cwd=root, env=env,
                                    capture_output=True, encoding="utf-8", errors="replace", timeout=90)
        print(diagnostic.stdout, flush=True)
        if diagnostic.returncode or "ERROR" in diagnostic.stdout or "MISSING" in diagnostic.stdout:
            raise RuntimeError("Packaged diagnostics failed: " + diagnostic.stderr)
        code = r'''
import json, sys, socket
from pathlib import Path
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox
from clipnado.app import MainWindow
from clipnado.media import probe
from clipnado.models import Candidate, Segment
from clipnado.process import run
import clipnado.local_llm as llm
app = QApplication([])
QMessageBox.information = lambda *_: QMessageBox.Ok
w = MainWindow()
assert not hasattr(w, 'key') and not hasattr(w, 'provider')
assert Path(w.tools.llama_vulkan).is_file() and Path(w.tools.whisper_vulkan).is_file()
w.tools.acceleration = 'cpu'
w.show_error = lambda message: (print(message, flush=True), app.exit(2))
real = run([w.tools.llama, '-m', w.tools.semantic_model, '-p', 'Привет', '-n', '1', '-c', '256',
            '-b', '32', '-t', '2', '-ngl', '0', '--conversation', '--single-turn', '--simple-io',
            '--no-display-prompt', '--no-warmup', '--offline'], timeout=180)
assert 'eval time' in real
print('PACKAGED_ACTUAL_QWEN_TOKEN_OK', flush=True)
def no_network(*_a, **_kw):
    raise RuntimeError('Unexpected network connection')
socket.socket.connect = no_network
socket.create_connection = no_network
w.media_loaded(probe(Path('demo/demo.mp4'), w.tools))
w.show()
w.play_candidate()
stage = [0]
timer = QTimer()
def protocol_fixture(args, *_a, **_kw):
    if args[args.index("-n") + 1] == "0":
        return "main: number of tokens in prompt = 1000"
    item = {'clip_id': 0, 'hook': 20, 'event': 20, 'emotion': 20,
            'completeness': 20, 'category': 'Юмор', 'reason': 'Protocol fixture'}
    if 'Это поиск по содержанию' in (Path(_kw['cwd']) / 'system.txt').read_text(encoding='utf-8'):
        item['relevance'] = 90
    return json.dumps({'clips': [item]})
def check():
    if stage[0] == 0:
        if w.video.frame_count < 1 or w.player.position() < 100 or w.video.image.isNull():
            return
        print('PACKAGED_VIDEO_PIXELS_OK', flush=True)
        w.player.pause()
        w.player.setPosition(0)
        w.minimum.setValue(2)
        w.project.candidates = [Candidate(w.project.media.source_file_id, 0, 2000, 80, 'Юмор', 'Portable export', selected=True) for _ in range(3)]
        w.refresh_candidates()
        w.export_clips.cellWidget(1, 3).setCurrentIndex(1)
        w.export_clips.cellWidget(2, 3).setCurrentIndex(2)
        w.resolution.setCurrentText('720')
        w.aspect.setCurrentText('1:1')
        w.export_folder.setText(sys.argv[1])
        stage[0] = 1
        w.export_selected()
    elif stage[0] == 1 and w.task_thread is None:
        files = sorted(Path(sys.argv[1]).glob('*.mp4'))
        assert len(files) == 3
        infos = [probe(p, w.tools) for p in files]
        assert [(m.width, m.height) for m in infos] == [(720,720), (720,720), (720,1280)]
        assert [c.export_layout for c in w.project.candidates] == [None, 'blur_center', 'tt_fill']
        print('PACKAGED_GUI_EXPORT_OK', flush=True)
        print('PACKAGED_MIXED_EXPORT_TIKTOK_OK', flush=True)
        w.project.segments = [Segment(0, 1000, 'Вопрос'), Segment(1000, 2000, 'Ответ')]
        llm.run = protocol_fixture
        stage[0] = 2
        w.analyze()
    elif stage[0] == 2 and w.task_thread is None:
        assert w.project.analysis_info['network_used'] is False
        assert w.project.candidates and w.project.candidates[0].provider.startswith('local-qwen:')
        print('PACKAGED_LOCAL_WORKFLOW_OK', flush=True)
        stage[0] = 3
        w.content_query.setText('необычный ответ')
        w.find_by_content()
    elif stage[0] == 3 and w.task_thread is None:
        assert w.project.last_query == 'необычный ответ' and w.project.query_results[0].match_score == 90
        assert len(w.project.candidates) == 1 and w.assistant_dialog
        w.assistant_dialog.apply_selected()  # identical intervals are deduplicated
        assert len(w.project.candidates) == 1
        print('PACKAGED_CONTENT_QUERY_PROTOCOL_OK', flush=True)
        w.table.selectRow(0)
        stage[0] = 4
        w.refine_selected()
    elif stage[0] == 4 and w.task_thread is None:
        assert w.assistant_dialog and len(w.project.candidates) == 1
        assert (w.project.candidates[0].start_ms, w.project.candidates[0].end_ms) == (0, 2000)
        w.close_assistant_dialog()
        print('PACKAGED_BOUNDARY_REVIEW_PROTOCOL_OK', flush=True)
        stage[0] = 5
        w.export_clips.cellWidget(0, 3).setCurrentIndex(2)
        w.preview_export_clip()
    elif stage[0] == 5 and w.task_thread is None:
        from clipnado.preview import VideoPreview
        assert w.export_preview_dialog
        image = w.export_preview_dialog.findChild(VideoPreview).image
        assert (image.width(), image.height()) == (360, 640)
        w.close_export_preview()
        print('PACKAGED_EXPORT_PREVIEW_OK', flush=True)
        timer.stop()
        w.dirty = False
        w.close()
        app.quit()
timer.timeout.connect(check)
timer.start(50)
QTimer.singleShot(150000, lambda: app.exit(3))
sys.exit(app.exec())
'''
        result = subprocess.run([str(python), "-c", code, str(Path(tmp) / "export")], cwd=root, env=env,
                                capture_output=True, encoding="utf-8", errors="replace", timeout=360)
        print(result.stdout, flush=True)
        required = ("PACKAGED_ACTUAL_QWEN_TOKEN_OK", "PACKAGED_VIDEO_PIXELS_OK", "PACKAGED_GUI_EXPORT_OK", "PACKAGED_LOCAL_WORKFLOW_OK",
                    "PACKAGED_CONTENT_QUERY_PROTOCOL_OK", "PACKAGED_BOUNDARY_REVIEW_PROTOCOL_OK",
                    "PACKAGED_MIXED_EXPORT_TIKTOK_OK", "PACKAGED_EXPORT_PREVIEW_OK")
        if result.returncode or not all(tag in result.stdout for tag in required):
            raise RuntimeError("Packaged local runtime verification failed: " + result.stderr[-3000:])


if __name__ == "__main__":
    main()
