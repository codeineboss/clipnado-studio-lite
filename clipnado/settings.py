from __future__ import annotations

import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

from .media import Tools
from .projects import atomic_json

SEMANTIC_MODEL_NAME = "Qwen_Qwen3-4B-Instruct-2507-Q3_K_M.gguf"


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
    configured = Path(os.environ["CLIPNADO_QWEN_MODEL"]) if os.environ.get("CLIPNADO_QWEN_MODEL") else None
    semantic = next((str(p) for p in (configured, root / "models" / SEMANTIC_MODEL_NAME, root / SEMANTIC_MODEL_NAME)
                     if p is not None and p.is_file()), "")
    extension = ".exe" if os.name == "nt" else ""
    def vulkan(name):
        path = root / "tools" / "vulkan" / (name + extension)
        return str(path) if path.is_file() else ""
    return Tools(ffmpeg=find("ffmpeg"), ffprobe=find("ffprobe"), whisper=find("whisper-cli"),
                 model=str(root / "models" / "ggml-base.bin") if (root / "models" / "ggml-base.bin").exists() else "",
                 llama=find("llama-cli"), semantic_model=semantic, llama_vulkan=vulkan("llama-cli"),
                 whisper_vulkan=vulkan("whisper-cli"))


def restore_tools(saved: dict | None) -> Tools:
    defaults = default_tools()
    try:
        values = asdict(defaults)
        if isinstance(saved, dict):
            values.update({k: v for k, v in saved.items() if k in values})
        result = Tools(**values)
    except TypeError:
        return defaults
    for field in ("ffmpeg", "ffprobe", "whisper", "model", "llama", "semantic_model", "llama_vulkan", "whisper_vulkan"):
        value, bundled = getattr(result, field), getattr(defaults, field)
        if not isinstance(value, str):
            setattr(result, field, bundled)
            continue
        exists = bool(value and (Path(value).is_file() or (field not in ("model", "semantic_model") and shutil.which(value))))
        if (not exists and value and bundled and Path(bundled).is_file()
                and Path(value).name.lower() == Path(bundled).name.lower()):
            # Standard bundled tools from an older extracted version were removed.
            # Custom models with another name are left as chosen by the user.
            setattr(result, field, bundled)
    if result.acceleration not in ("auto", "cpu"):
        result.acceleration = "auto"
    return result


def read_settings() -> dict:
    import json
    try:
        data = json.loads((data_root() / "settings.json").read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if k in ("tools", "cache", "output")}
    except (OSError, ValueError):
        return {}


def write_settings(tools: Tools, cache: str, output: str):
    atomic_json(data_root() / "settings.json", {"tools": asdict(tools), "cache": cache, "output": output,
                                               })
