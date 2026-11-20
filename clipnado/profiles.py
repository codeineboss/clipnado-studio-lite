"""Local editorial presets, not predictions of platform reach."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SearchProfile:
    label: str
    minimum: int
    maximum: int
    description: str
    instruction: str


SEARCH_PROFILE_VERSION = 1

SEARCH_PROFILES = {
    "memes": SearchProfile(
        "Мемы · универсальный",
        10,
        40,
        "10–40 секунд. Понятный смешной эпизод: короткая завязка, событие и реакция. Диапазон можно изменить.",
        "Стиль: самостоятельная мемная нарезка. Предпочитай неожиданный ответ, контраст ожидания и результата, "
        "событие с эмоциональной реакцией. Начало должно объяснять ситуацию новому зрителю, конец — сохранять "
        "смешной итог и ответную реакцию. Не выбирай одно восклицание без причины. Не предсказывай просмотры.",
    ),
    "quote": SearchProfile(
        "Короткая реплика или шутка",
        8,
        20,
        "8–20 секунд. Один вопрос и неожиданный ответ либо одна законченная шутка.",
        "Стиль: короткая законченная шутка или реплика. Сохрани вопрос или минимальную завязку, без которой "
        "ответ непонятен, сам неожиданный ответ и короткую ответную реакцию. Исключай длинные подводки и повторы. "
        "Не обрывай предложение и не выдавай случайный крик за самостоятельную шутку.",
    ),
    "reaction": SearchProfile(
        "Событие и реакция",
        20,
        40,
        "20–40 секунд. Ожидание или действие → неожиданный результат → реакция.",
        "Стиль: событие и реакция. Предпочитай цельную последовательность: обещание или действие, неожиданный "
        "результат, затем реакция участников. Сохраняй достаточно причины, чтобы реакция была понятна. "
        "По тексту не придумывай невидимые игровые события; оценивай только подтверждённые реплики.",
    ),
    "story": SearchProfile(
        "Законченный рассказ",
        40,
        75,
        "40–75 секунд. История с завязкой и итогом; длинный контекст оправдан содержанием.",
        "Стиль: законченный рассказ. Выбирай конкретную историю с понятной завязкой, развитием и итогом. "
        "Сохраняй необходимые объяснения, исключай рекламу, повторное обсуждение и затянутое вступление. "
        "Не обрезай итог ради эмоциональной фразы посередине.",
    ),
}

EXPORT_PROFILES = {
    "gaming": "Игровые мемы · камера 35% / игра 65%",
    "dialogue": "Разговор / IRL · приглушённый размытый фон",
    "story": "Рассказ · видео сверху, нижняя зона",
}


def search_profile(settings, code):
    if not isinstance(code, str) or code not in SEARCH_PROFILES:
        raise ValueError("Неизвестный профиль поиска")
    profile = SEARCH_PROFILES[code]
    result = replace(
        settings, min_seconds=profile.minimum, max_seconds=profile.maximum, profile=code
    )
    result.validate()
    return result


def export_profile(settings, code, media, has_transcript):
    if not isinstance(code, str) or code not in EXPORT_PROFILES:
        raise ValueError("Неизвестный профиль оформления")
    fps = min((24, 25, 30, 50, 60), key=lambda rate: abs(rate - media.fps))
    layout, subtitle_y = {
        "gaming": ("gaming_split", 0.28),
        "dialogue": ("blur_center", 0.58),
        "story": ("top_black", 0.31),
    }[code]
    # Keep the user's source regions and all per-clip choices. Region selection
    # remains a separate explicit action and is required before gaming export.
    result = replace(
        settings,
        preset=code,
        aspect="9:16",
        resolution=1080,
        fps=fps,
        quality=18,
        layout=layout,
        gaming_fraction=0.35 if code == "gaming" else settings.gaming_fraction,
        top_fraction=0.4,
        video_y=0.0,
        subtitles="burn" if has_transcript else "off",
        subtitle_y=subtitle_y,
        color="off",
        normalize_audio=True,
        readable_subtitles=True,
        background_dim=0.3,
    )
    result.validate()
    return result
