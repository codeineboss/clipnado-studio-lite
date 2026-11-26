"""Read the native Whisper response without losing a completed recording.

whisper.cpp writes token text bytes directly into JSON. A broken UTF-8 token
must not discard otherwise valid segments. Only the native response is read
lossily; application projects and caches remain strict UTF-8.
"""
from __future__ import annotations

import json
from pathlib import Path

from .models import timecode


def read_whisper_output(path: Path, diagnostic: Path, start_ms: int, end_ms: int):
    raw = path.read_bytes()
    damaged = False
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        damaged = True
        text = raw.decode('utf-8-sig', errors='replace')

    saved = False

    def save_raw():
        nonlocal saved
        if saved:
            return ''
        try:
            diagnostic.parent.mkdir(parents=True, exist_ok=True)
            diagnostic.write_bytes(raw)
            saved = True
            return ' Диагностика: ' + str(diagnostic)
        except OSError:
            return ' Не удалось сохранить диагностический файл.'

    interval = timecode(start_ms) + '–' + timecode(end_ms)
    try:
        data = json.loads(text)
        if not isinstance(data, dict) or not isinstance(data.get('transcription', []), list):
            raise ValueError('неверная структура ответа')
    except ValueError as exc:
        detail = save_raw()
        raise RuntimeError('Whisper: не удалось прочитать результат на участке ' + interval
                           + '. Уже сохранённая часть распознавания остаётся в кэше; повторите поиск.'
                           + detail) from exc

    warnings = []
    if damaged:
        warnings.append('Whisper: в расшифровке участка ' + interval
                        + ' повреждённые символы заменены на «�». Проверьте текст этого участка.'
                        + save_raw())
    return data, warnings
