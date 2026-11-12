from __future__ import annotations

import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

from .media import Tools
from .projects import atomic_json


def application_root() -> Path:
    return Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent


def data_root() -> Path:
    explicit = os.environ.get("CLIPNADO_DATA_DIR")
    root = Path(explicit) if explicit else Path(os.environ.get("LOCALAPPDATA", os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))) / "ClipnadoStudioLite"
    root.mkdir(parents=True, exist_ok=True)
    return root


def default_tools() -> Tools:
    root = application_root()
    def find(name):
        ext = ".exe" if os.name == "nt" else ""
        for path in (root / "tools" / (name + ext), root / "tools" / "bin" / (name + ext)):
            if path.exists():
                return str(path)
        return shutil.which(name) or name
    return Tools(ffmpeg=find("ffmpeg"), ffprobe=find("ffprobe"), whisper=find("whisper-cli"),
                 model=str(root / "models" / "ggml-base.bin") if (root / "models" / "ggml-base.bin").exists() else "")


def read_settings() -> dict:
    import json
    try:
        data = json.loads((data_root() / "settings.json").read_text(encoding="utf-8"))
        # Explicit whitelist: never read/write an API key in a JSON file.
        return {k: v for k, v in data.items() if k in ("tools", "cache", "output", "model")}
    except (OSError, ValueError):
        return {}


def write_settings(tools: Tools, cache: str, output: str, model: str):
    atomic_json(data_root() / "settings.json", {"tools": asdict(tools), "cache": cache, "output": output, "model": model})


def load_key() -> str:
    try:
        import keyring
        return keyring.get_password("ClipnadoStudioLite", "gemini") or ""
    except Exception:
        return ""


def save_key(key: str):
    import keyring
    if os.name == "nt":
        from keyring.backends.Windows import WinVaultKeyring
        backend = WinVaultKeyring()
    else:
        backend = keyring.get_keyring()
        if "SecretService" not in type(backend).__name__ and "Keyring" != type(backend).__name__:
            raise RuntimeError("Безопасное хранилище недоступно; ключ останется только в памяти до закрытия программы.")
    if key:
        backend.set_password("ClipnadoStudioLite", "gemini", key)
    else:
        try:
            backend.delete_password("ClipnadoStudioLite", "gemini")
        except keyring.errors.PasswordDeleteError:
            pass
