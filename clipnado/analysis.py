from __future__ import annotations

import array
import hashlib
import json
import math
import re
import shutil
import tempfile
import wave
from pathlib import Path

from .media import Tools, extract_audio, fingerprint, require_space
from .models import Candidate, MediaInfo, SearchSettings, Segment, timecode
from .process import Cancellation, Progress, run
from .projects import atomic_json

LEXICON = {
    "История": ("однажды", "истори", "помню", "случилось", "в итоге", "оказалось", "потом", "потому что"),
    "Юмор": ("смешн", "сме", "шутк", "ахах", "хаха", "прикол", "ржу", "анекдот"),
    "Реакция": ("невероят", "серьёзно", "серьезно", "офиг", "вот это", "победа", "получилось", "обалд", "не верю"),
    "Неожиданность": ("внезапно", "вдруг", "не ожидал", "оказалось", "но тут", "представьте", "секрет", "не повер"),
}

CHUNK_MS = 60000


class SpeechProgress:
    """Parse real whisper.cpp progress/segment timestamps, including split writes."""
    def __init__(self, duration_ms: int):
        self.duration_ms = duration_ms
        self.processed_ms = 0
        self.tail = ""

    def feed(self, output: str) -> int:
        self.tail = (self.tail + output)[-8192:]
        for value in re.findall(r"progress\s*=\s*(\d+)%", self.tail):
            self.processed_ms = max(self.processed_ms, self.duration_ms * min(100, int(value)) // 100)
        for h, m, s, ms in re.findall(r"-->\s*(\d+):(\d+):(\d+)[.,](\d{3})", self.tail):
            self.processed_ms = max(self.processed_ms, (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms))
        self.processed_ms = min(self.duration_ms, self.processed_ms)
        return self.processed_ms


def deduplicate(candidates: list[Candidate], count: int) -> list[Candidate]:
    selected = []
    for c in sorted(candidates, key=lambda c: (-c.score, c.start_ms)):
        if all(max(0, min(c.end_ms, s.end_ms) - max(c.start_ms, s.start_ms)) /
               min(c.end_ms - c.start_ms, s.end_ms - s.start_ms) <= .15 for s in selected):
            selected.append(c)
            if len(selected) >= count:
                break
    return sorted(selected, key=lambda c: c.start_ms)


def audio_energy(path: Path, offset_ms: int, token: Cancellation | None = None) -> list[tuple[int, float]]:
    result = []
    with wave.open(str(path), "rb") as source:
        while data := source.readframes(source.getframerate()):
            if token:
                token.check()
            values = array.array("h", data)
            rms = math.sqrt(sum(x * x for x in values) / max(1, len(values))) / 32768
            result.append((offset_ms + len(result) * 1000, rms))
    return result


def recognize(media: MediaInfo, tools: Tools, cache: Path, token: Cancellation, progress: Progress,
              existing: list[Segment] | None = None) -> tuple[list[Segment], list[tuple[int, float]], list[str]]:
    if existing:
        return sorted(existing, key=lambda s: (s.start_ms, s.end_ms)), [], ["Использована сохранённая расшифровка; повторное распознавание не запускалось."]
    if not media.has_audio:
        return [], [], ["В видео нет аудио. Смысловой поиск недоступен; добавьте интервалы вручную."]
    available = bool(tools.model and Path(tools.model).is_file() and (Path(tools.whisper).is_file() or shutil.which(tools.whisper)))
    # Content identity survives extracting an update to another folder and model relocation.
    engine_key = fingerprint(Path(tools.model), token) if available else "audio-only"
    identity = hashlib.sha256((media.source_file_id + engine_key + ":ru:v4").encode()).hexdigest()
    cache_file = cache / f"{identity}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            segments = [Segment(**s) for s in data["segments"]]
            for segment in segments:
                segment.validate(media.duration_ms)
            return segments, data["energy"], data["warnings"]
        except (ValueError, KeyError, TypeError):
            pass
    warnings = [] if available else ["Whisper или модель не настроены: используется только звук. Качество отбора низкое. Можно импортировать SRT."]
    segments, energy = [], []
    completed_ms = 0
    checkpoint = cache / f"{identity}.checkpoint.json"
    if checkpoint.exists():
        try:
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            completed_ms = saved["next_start_ms"]
            if (saved["chunk_ms"] != CHUNK_MS or not isinstance(completed_ms, int)
                    or not 0 <= completed_ms <= media.duration_ms
                    or (completed_ms != media.duration_ms and completed_ms % CHUNK_MS)):
                raise ValueError("Несовместимая контрольная точка")
            segments = [Segment(**s) for s in saved["segments"]]
            for segment in segments:
                segment.validate(completed_ms)
            energy = saved["energy"]
            if not isinstance(energy, list) or any(not isinstance(t, (int, float)) or not 0 <= t < completed_ms
                                                  or not isinstance(e, (int, float)) or not math.isfinite(e)
                                                  for t, e in energy):
                raise ValueError("Повреждённый кэш звука")
        except (KeyError, TypeError, ValueError):
            completed_ms, segments, energy = 0, [], []
    progress(int(completed_ms / media.duration_ms * 80),
             (f"Продолжение с {timecode(completed_ms)[:-4]}. " if completed_ms else "")
             + ("Подготовка Whisper · CPU без AVX2" if available and not tools.gpu else "Подготовка распознавания/звука…"))
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as tmp:
        folder = Path(tmp)
        if available:
            # Some Windows whisper.cpp builds use narrow argv/fopen. Relative ASCII inputs
            # avoid code-page failures even when the cache, model and username are Cyrillic.
            staged_model = folder / "model.bin"
            try:
                import os
                os.link(Path(tools.model).resolve(), staged_model)
            except OSError:
                require_space(folder, Path(tools.model).stat().st_size + 30_000_000)
                with open(tools.model, "rb") as source, staged_model.open("wb") as destination:
                    while block := source.read(8 * 1024 * 1024):
                        token.check()
                        destination.write(block)
        # Chunking bounds RAM/disk use for multi-hour recordings; overlapping audio preserves boundary context.
        for chunk_start in range(completed_ms, media.duration_ms, CHUNK_MS):
            token.check()
            start = max(0, chunk_start - 1500)
            chunk_end = min(media.duration_ms, chunk_start + CHUNK_MS)
            duration = chunk_end - start
            percent = int(chunk_start / media.duration_ms * 80)
            block = f"{chunk_start // CHUNK_MS + 1}/{math.ceil(media.duration_ms / CHUNK_MS)}"
            progress(percent, f"Извлечение аудио · блок {block} · {timecode(chunk_start)[:-4]}–{timecode(chunk_end)[:-4]}")
            wav = folder / "audio.wav"
            extract_audio(media, start, duration, wav, tools, token,
                          progress=lambda value, _: progress(percent, f"Извлечение аудио · блок {block} · {value}%"))
            progress(percent, f"Анализ звука · блок {block}")
            energy.extend((t, e) for t, e in audio_energy(wav, start, token) if chunk_start <= t < chunk_end)
            if available:
                output = folder / "speech"
                args = [tools.whisper, "-m", "model.bin", "-f", "audio.wav", "-l", "ru", "-oj", "-of", "speech",
                        "-t", str(tools.threads), "-np", "-pp", "-bs", "1", "-bo", "1", "-nf"]
                if not tools.gpu:
                    args.append("-ng")
                parser = SpeechProgress(duration)
                def speech_output(text, elapsed):
                    recognized = max(chunk_start, min(chunk_end, start + parser.feed(text)))
                    known = int(recognized / media.duration_ms * 80)
                    waiting = " · ждём первый результат" if parser.processed_ms == 0 else ""
                    mode = "CPU" if not tools.gpu else "GPU/CPU"
                    progress(known, f"Whisper {mode} · {timecode(recognized)[:-4]} / {timecode(media.duration_ms)[:-4]}"
                             f" · блок {block} · прошло {timecode(int(elapsed * 1000))[:-4]}{waiting}")
                run(args, token, cwd=folder, on_output=speech_output, timeout=1800)
                data = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
                for item in data.get("transcription", []):
                    a = max(chunk_start, int(item["offsets"]["from"]) + start)
                    b = min(chunk_end, int(item["offsets"]["to"]) + start)
                    text = item["text"].strip()
                    if b <= a or b <= chunk_start or not text or re.fullmatch(r"[\[\(].*[\]\)]", text):
                        continue
                    if segments:
                        a = max(a, segments[-1].end_ms)
                    if b > a:
                        segments.append(Segment(a, b, text))
                output.with_suffix(".json").unlink(missing_ok=True)
            wav.unlink(missing_ok=True)
            atomic_json(checkpoint, {"chunk_ms": CHUNK_MS, "next_start_ms": chunk_end,
                                    "segments": [s.__dict__ for s in segments], "energy": energy})
            progress(int(chunk_end / media.duration_ms * 80),
                     f"Сохранено {timecode(chunk_end)[:-4]} / {timecode(media.duration_ms)[:-4]} · блок {block}")
    if available and not segments:
        warnings.append("Речь не обнаружена. Звуковой поиск не оценивает смысл; проверьте интервалы вручную.")
    elif available and media.duration_ms >= 10000 and sum(len(s.text.split()) for s in segments) < 3:
        warnings.append("Распознано очень мало текста. Проверьте звук, качество модели и расшифровку; смысловой отбор ненадёжен.")
    atomic_json(cache_file, {"segments": [s.__dict__ for s in segments], "energy": energy, "warnings": warnings})
    checkpoint.unlink(missing_ok=True)
    return segments, energy, warnings


class LocalFallbackProvider:
    def analyze(self, media: MediaInfo, segments: list[Segment], settings: SearchSettings,
                energy: list[tuple[int, float]], token: Cancellation) -> list[Candidate]:
        settings.validate()
        lo, hi = settings.min_seconds * 1000, settings.max_seconds * 1000
        if media.duration_ms < lo:
            lo = media.duration_ms
        hi = min(hi, media.duration_ms)
        candidates = []
        cutoff = 66 - settings.sensitivity * .45
        if not segments:
            # Event-triggered suggestions only: no manufactured regular intervals for silent video.
            nonzero = [e for _, e in energy if e > .002]
            if not nonzero:
                return []
            mean = sum(nonzero) / len(nonzero)
            for t, level in sorted(energy, key=lambda p: -p[1])[:300]:
                token.check()
                if level < max(.06 - settings.sensitivity * .0004, mean * (2.4 - settings.sensitivity * .012)):
                    continue
                start = max(0, min(media.duration_ms - lo, t - lo // 3))
                end = min(media.duration_ms, start + lo)
                candidates.append(Candidate(media.source_file_id, start, end, min(30, round(15 + level * 30)),
                                            "Аудиособытие", "Резкое изменение звука. Смысл не оценён; требуется ручная проверка."))
            return deduplicate(candidates, settings.count * 3)
        weights = []
        for segment in segments:
            text = segment.text.lower()
            weights.append({cat: sum(1 for word in LEXICON[cat] if word in text) for cat in settings.categories if cat in LEXICON})
        for index, segment in enumerate(segments):
            token.check()
            if not any(weights[index].values()):
                continue
            # Start at a phrase before the event and stop at a phrase boundary inside the duration window.
            ideal = max(0, segment.start_ms - lo // 3)
            prior = [s.start_ms for s in segments[max(0, index - 50):index + 1] if s.start_ms <= ideal]
            start = prior[-1] if prior else segments[max(0, index - 1)].start_ms
            if media.duration_ms - start < lo:
                start = max(0, media.duration_ms - lo)
            ends = [s.end_ms for s in segments[index:] if start + lo <= s.end_ms <= start + hi and s.end_ms >= segment.end_ms]
            if not ends:
                continue
            end = min(ends, key=lambda value: abs(value - (start + lo)))
            included = [(s, w) for s, w in zip(segments, weights) if s.start_ms < end and s.end_ms > start]
            totals = {cat: sum(w.get(cat, 0) for _, w in included) for cat in settings.categories}
            category = max(totals, key=totals.get)
            hits = sum(totals.values())
            text = " ".join(s.text for s, _ in included)
            words = len(text.split())
            development = min(12, sum(3 for marker in ("потом", "в итоге", "оказалось", "поэтому", "но ") if marker in text.lower()))
            levels = [v for t, v in energy if start <= t < end]
            sound = min(8, (max(levels) - sum(levels) / len(levels)) * 30) if levels else 0
            score = round(min(96, 20 + min(36, hits * 8) + min(12, words / 8) + development + sound))
            if score < cutoff or words < 8:
                continue
            candidates.append(Candidate(media.source_file_id, start, end, score, category,
                                        f"{category}: {hits} смысловых маркеров, {words} слов; границы по репликам. Локальная эвристика, проверьте контекст.",
                                        text[:1000]))
        return deduplicate(candidates, settings.count * 3)
