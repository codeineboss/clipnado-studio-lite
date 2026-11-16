"""Offline diagnostics: only local engines and models are inspected."""
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clipnado.media import Tools
from clipnado.process import run
from clipnado.settings import default_tools


def main():
    print("Clipnado offline diagnostics")
    print(platform.platform(), sys.version)
    tools = default_tools()
    for name, flag in [("ffmpeg", "-version"), ("ffprobe", "-version"), ("whisper", "--help"), ("llama", "--version")]:
        try:
            result = run([getattr(tools, name), flag], timeout=20)
            print(name, "OK", next((line for line in result.splitlines() if line), ""))
        except Exception as exc:
            print(name, "ERROR", str(exc)[-600:])
    print("Speech model:", "OK" if Path(tools.model).is_file() else "MISSING")
    print("Qwen model:", "OK" if tools.semantic_model and Path(tools.semantic_model).is_file() else "Separate GGUF asset must be installed in models")
    try:
        from PySide6.QtCore import qVersion
        from PySide6.QtWidgets import QApplication
        from PySide6.QtMultimedia import QMediaPlayer
        from PySide6.QtMultimediaWidgets import QVideoWidget
        print("Qt imports OK:", qVersion())
    except Exception as exc:
        print("Qt ERROR", str(exc))
    try:
        from clipnado.reframe import MODEL_FILES, model_bytes
        for name in MODEL_FILES:
            model_bytes(name)
            print("Autocrop model OK:", name)
    except Exception as exc:
        print("Autocrop ERROR", str(exc))
    print("No data was sent over the network.")


if __name__ == "__main__":
    main()
