from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .models import MediaInfo
from .process import Cancellation, run


@dataclass
class Tools:
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    whisper: str = "whisper-cli"
    model: str = ""
    gpu: bool = False
    threads: int = 8


def fingerprint(path: Path, token: Cancellation | None = None) -> str:
    h = hashlib.sha256()
    size = path.stat().st_size
    h.update(str(size).encode())
    with path.open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            if token:
                token.check()
            h.update(block)
    return h.hexdigest()


def probe(path: Path, tools: Tools, token: Cancellation | None = None) -> MediaInfo:
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError("Исходное видео не найдено")
    data = json.loads(run([tools.ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)], token, timeout=60))
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")), None)
    if not video:
        raise ValueError("В файле отсутствует видеодорожка")
    try:
        duration = float(data.get("format", {}).get("duration") or video.get("duration", 0))
        fps = float(Fraction(video.get("avg_frame_rate", "0/1")))
    except (ValueError, ZeroDivisionError):
        raise ValueError("Невозможно определить длительность или FPS")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Видео имеет неверную длительность")
    width, height = int(video["width"]), int(video["height"])
    rotation = next((s.get("rotation", 0) for s in video.get("side_data_list", []) if "rotation" in s), 0)
    if abs(rotation) % 180 == 90:
        width, height = height, width
    return MediaInfo(str(path), fingerprint(path, token), round(duration * 1000), width, height, fps,
                     any(s.get("codec_type") == "audio" for s in data["streams"]))


def require_space(folder: Path, bytes_required: int):
    folder.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(folder).free < bytes_required:
        raise OSError("Недостаточно свободного места в выбранной папке")


def extract_audio(media: MediaInfo, start_ms: int, duration_ms: int, target: Path, tools: Tools, token: Cancellation):
    require_space(target.parent, duration_ms * 32 + 10_000_000)
    run([tools.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start_ms / 1000:.3f}",
         "-i", media.path, "-t", f"{duration_ms / 1000:.3f}", "-vn", "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target)], token)


def thumbnail(media: MediaInfo, time_ms: int, target: Path, tools: Tools, token: Cancellation):
    run([tools.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", str(time_ms / 1000), "-i", media.path,
         "-frames:v", "1", "-vf", "scale=640:360:force_original_aspect_ratio=decrease", str(target)], token, timeout=60)
