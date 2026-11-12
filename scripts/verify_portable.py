"""Exercise the actual embedded Windows Python and packaged Qt/FFmpeg offline."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    root = args.folder.resolve()
    python = root / "runtime" / "python.exe"
    with tempfile.TemporaryDirectory(prefix="clipnado-windows-smoke-") as tmp:
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["CLIPNADO_DATA_DIR"] = str(Path(tmp) / "settings")
        diagnostic = subprocess.run([str(python), str(root / "scripts" / "diagnostics.py")],
                                    cwd=root, env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=90)
        print(diagnostic.stdout, flush=True)
        if diagnostic.returncode or "ERROR" in diagnostic.stdout or "MISSING" in diagnostic.stdout:
            raise RuntimeError("Packaged runtime diagnostics failed: " + diagnostic.stderr)
        code = r'''
import sys
from pathlib import Path
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from clipnado.app import MainWindow
from clipnado.media import probe
from clipnado.models import Candidate, SearchSettings
from clipnado.process import Cancellation
from clipnado.selection import choose_candidates
app = QApplication([])
w = MainWindow()
assert w.target_count.value() == 10 and w.min_score.value() == 60
short = [Candidate('x', i * 60000, (i + 1) * 60000, 80, 'Реакция', 'Fixture') for i in range(10)]
long = [Candidate('x', i * 120000, i * 120000 + 90000, 95, 'Реакция', 'Fixture') for i in range(5)]
selected = choose_candidates(short + long, SearchSettings(), Cancellation())
assert len(selected) == 10 and all(c.score == 80 for c in selected)
print('PACKAGED_TARGET_SELECTION_OK', flush=True)
w.show_error = lambda message: (print(message, flush=True), app.exit(2))
w.media_loaded(probe(Path('demo/demo.mp4'), w.tools))
w.show()
w.play_candidate()
stage = [0]
timer = QTimer()
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
    elif w.task_thread is None:
        if not list(Path(sys.argv[1]).glob('*.mp4')):
            app.exit(4)
            return
        print('PACKAGED_GUI_EXPORT_OK', flush=True)
        timer.stop()
        w.dirty = False
        w.close()
        app.quit()
timer.timeout.connect(check)
timer.start(50)
QTimer.singleShot(90000, lambda: app.exit(3))
sys.exit(app.exec())
'''
        result = subprocess.run([str(python), "-c", code, str(Path(tmp) / "export")], cwd=root, env=env,
                                capture_output=True, encoding="utf-8", errors="replace", timeout=120)
        print(result.stdout, flush=True)
        if (result.returncode or "PACKAGED_GUI_EXPORT_OK" not in result.stdout
                or "PACKAGED_VIDEO_PIXELS_OK" not in result.stdout or "PACKAGED_TARGET_SELECTION_OK" not in result.stdout):
            raise RuntimeError("Packaged GUI/export failed: " + result.stderr[-6000:])


if __name__ == "__main__":
    main()
