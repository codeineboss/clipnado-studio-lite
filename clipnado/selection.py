"""Count-aware interval selection; scores are never raised to satisfy a quota."""
from __future__ import annotations

import bisect
import re

from .models import Candidate, MediaInfo, SearchSettings
from .process import Cancellation

STOP = set("и в во не на что это как так вот но а мы вы он она они я ты с со у к за по из для или уже еще всё все его ее их мне нам вас нас есть будет был было тут там только потом потому чтобы если когда просто очень да нет ну то бы же ли до про при".split())


def content_words(text):
    return {word for word in re.findall(r"[а-яёa-z]+", text.lower()) if len(word) > 2 and word not in STOP}


def choose_candidates(candidates: list[Candidate], settings: SearchSettings, token: Cancellation) -> list[Candidate]:
    """Choose disjoint episodes; mixed mode prioritizes stronger complete clips.

    Other modes prefer reaching the target, then total score.

    Retaining alternative boundaries is essential: a greedy high-scoring long
    interval can otherwise block two equally valid shorter episodes. Extra
    alternatives remain available through Show More after the target is met.
    """
    unique = {}
    for c in candidates:
        token.check()
        if c.score < settings.min_score:
            continue
        key = c.start_ms, c.end_ms
        if key not in unique or unique[key].score < c.score:
            unique[key] = c
    pool = sorted(unique.values(), key=lambda c: (c.end_ms, c.start_ms, -c.score))
    if not pool:
        return []
    if settings.profile in ("mixed", "highlights"):
        # Preserve a stronger complete story when weaker excerpts overlap it.
        # A target count must not force an otherwise good episode into pieces.
        chosen = sorted(pool, key=lambda c: (-c.score, 0 if 20000 <= c.end_ms-c.start_ms <= 40000 else 1, c.end_ms-c.start_ms, c.start_ms)) if settings.profile == "highlights" else sorted(pool, key=lambda c: (-c.score, c.start_ms, c.end_ms))
    else:
        ends = [c.end_ms for c in pool]
        previous = [bisect.bisect_right(ends, c.start_ms, hi=i) for i, c in enumerate(pool)]
        target = min(settings.target_count, len(pool))
        # scores[k][i] is the maximum total score for exactly k intervals in pool[:i].
        scores = [[0] * (len(pool) + 1)]
        for k in range(1, target + 1):
            token.check()
            row = [-1] * (len(pool) + 1)
            for i, c in enumerate(pool, 1):
                if i % 1024 == 0:
                    token.check()
                before = scores[k - 1][previous[i - 1]]
                take = before + c.score if before >= 0 else -1
                row[i] = max(row[i - 1], take)
            scores.append(row)
            if row[-1] < 0:
                break
        k = max(i for i, row in enumerate(scores) if row[-1] >= 0)
        chosen = []
        i = len(pool)
        while k and i:
            if scores[k][i] == scores[k][i - 1]:
                i -= 1
            else:
                chosen.append(pool[i - 1])
                i = previous[i - 1]
                k -= 1

    # Human-visible textual repeats cannot fill the target. Keep the strongest
    # occurrence, then try additional distinct episodes from the entire pool.
    selected, texts = [], []
    exact_texts = set()
    def append(c):
        if any(c.start_ms < other.end_ms and c.end_ms > other.start_ms for other in selected):
            return
        words = content_words(c.transcript_excerpt)
        exact = " ".join(c.transcript_excerpt.lower().split())
        if c.provider.startswith("local-qwen:"):
            if len(exact) > 50 and exact in exact_texts:
                return
        elif any(
            (len(words) >= 6 and words == other)
            or (len(words) >= 12 and len(other) >= 12
                and (len(words & other) / max(1, len(words | other)) >= .8
                     or len(words & other) / min(len(words), len(other)) >= .92))
            for other in texts
        ):
            return
        selected.append(c)
        texts.append(words)
        exact_texts.add(exact)
    for c in (chosen if settings.profile == "highlights" else sorted(chosen, key=lambda c: (-c.score, c.start_ms))):
        token.check()
        if len(selected) >= settings.count * 3:
            break
        append(c)
    for c in sorted(pool, key=lambda c: (-c.score, c.start_ms, c.end_ms)):
        token.check()
        if len(selected) >= settings.count * 3:
            break
        append(c)
    return sorted(selected, key=lambda c: c.start_ms)


def shortfall_message(media: MediaInfo, settings: SearchSettings, candidates: list[Candidate], has_text: bool,
                      audiovisual: bool = False) -> str:
    count = sum(c.score >= settings.min_score for c in candidates)
    if count >= settings.target_count:
        return ""
    message = f"Найдено {count} из целевых {settings.target_count} клипов с оценкой от {settings.min_score}. "
    capacity = media.duration_ms // (settings.min_seconds * 1000)
    if media.duration_ms < settings.min_seconds * 1000:
        message += f"Исходник короче заданной длины клипа {settings.min_seconds} с. "
    elif capacity < settings.target_count:
        message += (f"При минимальной длине {settings.min_seconds} с исходник вмещает не более {capacity} "
                    "непересекающихся клипов. ")
    if not has_text and not audiovisual:
        message += "Нет расшифровки для смысловой оценки; аудиопики сами по себе не подтверждают интересный момент. "
    else:
        message += "Другие интервалы не прошли фильтры оценки, границ или повторов. "
    return message + "Рейтинг не повышался ради количества."
