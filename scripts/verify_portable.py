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
from clipnado.models import Segment
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
def protocol_fixture(*_a, **_kw):
    return json.dumps({'clips': [{'first': 0, 'last': 1, 'hook': 20, 'event': 20, 'emotion': 20,
                                'completeness': 20, 'category': 'Юмор', 'reason': 'Protocol fixture'}]})
def check():
    if stage[0] == 0:
        if w.video.frame_count < 1 or w.player.position() < 100 or w.video.image.isNull():
            return
        print('PACKAGED_VIDEO_PIXELS_OK', flush=True)
        w.player.pause()
        w.player.setPosition(0)
        w.minimum.setValue(2)
        w.add_manual()
        w.resolution.setCurrentText('720')
        w.export_folder.setText(sys.argv[1])
        stage[0] = 1
        w.export_selected()
    elif stage[0] == 1 and w.task_thread is None:
        assert list(Path(sys.argv[1]).glob('*.mp4'))
        print('PACKAGED_GUI_EXPORT_OK', flush=True)
        w.project.segments = [Segment(0, 1000, 'Вопрос'), Segment(1000, 2000, 'Ответ')]
        llm.run = protocol_fixture
        stage[0] = 2
        w.analyze()
    elif stage[0] == 2 and w.task_thread is None:
        assert w.project.analysis_info['network_used'] is False
        assert w.project.candidates and w.project.candidates[0].provider.startswith('local-qwen:')
        print('PACKAGED_LOCAL_WORKFLOW_OK', flush=True)
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
        required = ("PACKAGED_ACTUAL_QWEN_TOKEN_OK", "PACKAGED_VIDEO_PIXELS_OK", "PACKAGED_GUI_EXPORT_OK", "PACKAGED_LOCAL_WORKFLOW_OK")
        if result.returncode or not all(tag in result.stdout for tag in required):
            raise RuntimeError("Packaged local runtime verification failed: " + result.stderr[-3000:])


if __name__ == "__main__":
    main()
