from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4


EXPORT_LAYOUTS = {
    "top_black": "Видео сверху · тёмная зона",
    "blur_center": "Видео по центру · размытый фон",
    "gaming_split": "Игровая нарезка · стример сверху, игра снизу",
}


def integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value:
        raise ValueError(f"Некорректное число: {name}")
    return int(value)


def normalized_energy(points, duration_ms: int) -> list[list[float]]:
    if not isinstance(points, list):
        raise ValueError("Некорректные аудиопризнаки проекта")
    result = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Некорректные аудиопризнаки проекта")
        at, level = point
        at = integer(at, "время аудиопризнака")
        if not 0 <= at < duration_ms or not isinstance(level, (int, float)) or not math.isfinite(level) or not 0 <= level <= 1:
            raise ValueError("Некорректные аудиопризнаки проекта")
        result.append([at, float(level)])
    return result


def crop_region(value, name="область"):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"Некорректная {name}: нужны x, y, ширина и высота")
    if not all(type(v) in (int, float) and math.isfinite(v) for v in value):
        raise ValueError(f"Некорректная {name}")
    x, y, w, h = map(float, value)
    if not (0 <= x < 1 and 0 <= y < 1 and .01 <= w <= 1 and .01 <= h <= 1
            and x + w <= 1 + 1e-9 and y + h <= 1 + 1e-9):
        raise ValueError(f"{name.capitalize()} выходит за границы исходного кадра")
    return [x, y, w, h]


@dataclass
class MediaInfo:
    path: str
    source_file_id: str
    duration_ms: int
    width: int
    height: int
    fps: float
    has_audio: bool

    def validate(self):
        if not isinstance(self.path, str) or not isinstance(self.source_file_id, str) or not self.source_file_id:
            raise ValueError("Некорректный путь или идентификатор исходника")
        for name in ("duration_ms", "width", "height"):
            value = integer(getattr(self, name), name)
            if value <= 0:
                raise ValueError("Некорректные метаданные исходника")
            setattr(self, name, value)
        if not isinstance(self.fps, (int, float)) or not math.isfinite(self.fps) or not 0 <= self.fps <= 1000:
            raise ValueError("Некорректный FPS")
        if not isinstance(self.has_audio, bool):
            raise ValueError("Некорректные метаданные звука")


@dataclass
class Segment:
    start_ms: int
    end_ms: int
    text: str

    def validate(self, duration: int) -> None:
        self.start_ms = integer(self.start_ms, "начало реплики")
        self.end_ms = integer(self.end_ms, "конец реплики")
        if not 0 <= self.start_ms < self.end_ms <= duration or not isinstance(self.text, str):
            raise ValueError("Некорректный интервал расшифровки")


@dataclass
class Candidate:
    source_file_id: str
    start_ms: int
    end_ms: int
    score: int
    category: str
    reason: str
    transcript_excerpt: str = ""
    provider: str = "local"
    version: int = 1
    id: str = field(default_factory=lambda: uuid4().hex)
    selected: bool = False
    match_score: int | None = None
    export_layout: str | None = None

    def validate(self, media: MediaInfo) -> None:
        self.start_ms = integer(self.start_ms, "начало клипа")
        self.end_ms = integer(self.end_ms, "конец клипа")
        self.score = integer(self.score, "рейтинг")
        if self.export_layout is not None and (not isinstance(self.export_layout, str) or self.export_layout not in EXPORT_LAYOUTS):
            raise ValueError("Неизвестное оформление клипа")
        if self.source_file_id != media.source_file_id or not 0 <= self.start_ms < self.end_ms <= media.duration_ms:
            raise ValueError("Клип выходит за границы видео или относится к другому исходнику")
        if self.match_score is not None:
            self.match_score = integer(self.match_score, "соответствие запросу")
            if not 0 <= self.match_score <= 100:
                raise ValueError("Соответствие запросу должно быть от 0 до 100")
        if not 0 <= self.score <= 100:
            raise ValueError("Рейтинг должен быть от 0 до 100")
        if not all(isinstance(x, str) for x in (self.category, self.reason, self.transcript_excerpt, self.provider, self.id)):
            raise ValueError("Некорректное описание клипа")


@dataclass
class SearchSettings:
    count: int = 15
    min_seconds: int = 60
    max_seconds: int = 90
    sensitivity: int = 55
    categories: list[str] = field(default_factory=lambda: ["История", "Юмор", "Реакция", "Неожиданность"])
    target_count: int = 10
    min_score: int = 60
    query: str = ""

    def validate(self) -> None:
        if not isinstance(self.query, str) or len(self.query) > 500:
            raise ValueError("Поисковый запрос: не более 500 символов")
        self.query = self.query.strip()
        for name in ("count", "min_seconds", "max_seconds", "sensitivity", "target_count", "min_score"):
            setattr(self, name, integer(getattr(self, name), name))
        if not 1 <= self.count <= 100 or not 1 <= self.min_seconds <= self.max_seconds <= 600:
            raise ValueError("Проверьте количество и длительности (1–600 секунд)")
        if not 0 <= self.sensitivity <= 100 or not isinstance(self.categories, list) or not self.categories or not all(isinstance(c, str) for c in self.categories):
            raise ValueError("Выберите категории и чувствительность")
        if not 1 <= self.target_count <= self.count or not 0 <= self.min_score <= 100:
            raise ValueError("Цель должна быть от 1 до количества кандидатов, оценка — от 0 до 100")


@dataclass
class ExportSettings:
    aspect: str = "9:16"
    resolution: int = 1080
    fps: int = 30
    quality: int = 18
    top_fraction: float = 0.4
    video_y: float = 0.0
    subtitles: str = "off"
    subtitle_y: float = 0.31
    layout: str = "top_black"
    gaming_camera: list[float] | None = None
    gaming_game: list[float] = field(default_factory=lambda: [0., 0., 1., 1.])
    gaming_fraction: float = .4

    def validate(self) -> None:
        for name in ("resolution", "fps", "quality"):
            setattr(self, name, integer(getattr(self, name), name))
        if not all(isinstance(x, (float, int)) and math.isfinite(x) for x in (self.top_fraction, self.video_y, self.subtitle_y)):
            raise ValueError("Некорректные размеры композиции")
        if self.aspect not in ("9:16", "16:9", "1:1") or self.resolution not in (720, 1080):
            raise ValueError("Неверный формат экспорта")
        if self.fps not in (24, 25, 30, 50, 60) or not 16 <= self.quality <= 28:
            raise ValueError("Неверная частота кадров или качество")
        if not isinstance(self.layout, str) or self.layout not in EXPORT_LAYOUTS:
            raise ValueError("Неизвестное оформление видео")
        if self.gaming_camera is not None:
            self.gaming_camera = crop_region(self.gaming_camera, "область камеры")
        self.gaming_game = crop_region(self.gaming_game, "область игры")
        if type(self.gaming_fraction) not in (int, float) or not math.isfinite(self.gaming_fraction) or not .25 <= self.gaming_fraction <= .6:
            raise ValueError("Высота камеры должна быть от 25 до 60%")
        if self.layout == "top_black" and (not 0.2 <= self.top_fraction <= 1 or not 0 <= self.video_y <= 1 - self.top_fraction + 1e-9):
            raise ValueError("Верхняя зона выходит за холст")
        if self.subtitles not in ("off", "srt", "burn") or not 0 <= self.subtitle_y <= 1:
            raise ValueError("Неверные настройки субтитров")
        if self.layout == "top_black" and self.subtitles == "burn" and not self.video_y <= self.subtitle_y < self.video_y + self.top_fraction:
            raise ValueError("Субтитры должны находиться в зоне видео")

    def require_gaming_regions(self):
        self.validate()
        if self.layout == "gaming_split" and self.gaming_camera is None:
            raise ValueError("Для игровой нарезки нажмите «Камера и игра…» и один раз выделите области на исходном видео.")

    def dimensions(self) -> tuple[int, int]:
        return {"9:16": (self.resolution, self.resolution * 16 // 9),
                "16:9": (self.resolution * 16 // 9, self.resolution),
                "1:1": (self.resolution, self.resolution)}["9:16" if self.layout == "gaming_split" else self.aspect]


@dataclass
class Project:
    media: MediaInfo
    segments: list[Segment] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    search: SearchSettings = field(default_factory=SearchSettings)
    export: ExportSettings = field(default_factory=ExportSettings)
    schema_version: int = 1
    warnings: list[str] = field(default_factory=list)
    audio_energy: list = field(default_factory=list)
    analysis_info: dict = field(default_factory=dict)
    query_results: list[Candidate] = field(default_factory=list)
    last_query: str = ""
    query_info: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Эта версия проекта не поддерживается")
        self.media.validate()
        self.search.validate()
        self.export.validate()
        self.audio_energy = normalized_energy(self.audio_energy, self.media.duration_ms)
        if not isinstance(self.analysis_info, dict):
            raise ValueError("Некорректная диагностика анализа")
        if not isinstance(self.last_query, str) or len(self.last_query) > 500 or not isinstance(self.query_info, dict):
            raise ValueError("Некорректные данные поиска по содержанию")
        for candidate in self.query_results:
            candidate.validate(self.media)
        for segment in self.segments:
            segment.validate(self.media.duration_ms)
        ids = set()
        for candidate in self.candidates:
            candidate.validate(self.media)
            if candidate.id in ids:
                raise ValueError("В проекте повторяются идентификаторы клипов")
            ids.add(candidate.id)

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


def timecode(ms: int, comma: bool = False) -> str:
    ms = max(0, int(ms))
    h, remainder = divmod(ms, 3600000)
    m, remainder = divmod(remainder, 60000)
    s, fraction = divmod(remainder, 1000)
    return f"{h:02}:{m:02}:{s:02}{',' if comma else '.'}{fraction:03}"


def parse_time(value: str) -> int:
    parts = value.strip().replace(",", ".").split(":")
    if len(parts) != 3:
        raise ValueError("Таймкод: ЧЧ:ММ:СС.мс")
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    if h < 0 or not 0 <= m < 60 or not 0 <= s < 60 or not math.isfinite(s):
        raise ValueError("Неверный таймкод")
    return round((h * 3600 + m * 60 + s) * 1000)
