"""Conservative picture and whole-mix loudness adjustments."""

from __future__ import annotations

import json
import math
import re

from .process import run


def color_filter(settings):
    return {
        "off": "",
        "soft": "eq=contrast=1.025:saturation=1.035:gamma=1.015,",
        "bright": "eq=gamma=1.10:saturation=1.02,",
    }[settings.color]


def audio_filter(media, candidate, settings, tools, token, progress):
    base = "aresample=async=1:first_pts=0"
    if not settings.normalize_audio or not media.has_audio:
        return base
    progress(0, "Измерение громкости выбранного отрывка…")
    args = [
        tools.ffmpeg,
        "-hide_banner",
        "-nostats",
        "-ss",
        f"{candidate.start_ms/1000:.3f}",
        "-i",
        media.path,
        "-t",
        f"{(candidate.end_ms-candidate.start_ms)/1000:.3f}",
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
        "-f",
        "null",
        "-",
    ]
    output = run(
        args,
        token,
        on_output=lambda _, elapsed: progress(
            0, f"Измерение громкости · {int(elapsed)} с…"
        ),
    )
    blocks = re.findall(r'\{\s*"input_i".*?\}', output, re.S)
    if not blocks:
        raise RuntimeError("FFmpeg не сообщил результаты измерения громкости")
    data = json.loads(blocks[-1])
    names = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    values = {name: float(data[name]) for name in names}
    if not all(math.isfinite(v) for v in values.values()):
        # A silent clip has -inf loudness. Keep it silent.
        progress(0, "В отрывке тишина; усиление звука пропущено")
        return base
    # Cap gain at 12 dB instead of bringing very quiet noise up to speech level.
    if values["input_i"] < -28:
        gain = min(12.0, -1.5 - values["input_tp"])
        return f"volume={gain:.4f}dB," + base
    parts = [
        "loudnorm=I=-16:TP=-1.5:LRA=11:linear=true",
        f"measured_I={values['input_i']}",
        f"measured_TP={values['input_tp']}",
        f"measured_LRA={values['input_lra']}",
        f"measured_thresh={values['input_thresh']}",
        f"offset={values['target_offset']}",
    ]
    return ":".join(parts) + "," + base
