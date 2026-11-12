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
app = QApplication([])
w = MainWindow()
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
        if result.returncode or "PACKAGED_GUI_EXPORT_OK" not in result.stdout or "PACKAGED_VIDEO_PIXELS_OK" not in result.stdout:
            raise RuntimeError("Packaged GUI/export failed: " + result.stderr[-6000:])


if __name__ == "__main__":
    main()
