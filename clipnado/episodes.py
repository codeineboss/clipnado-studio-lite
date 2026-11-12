"""Offline event/context heuristics. These cues do not establish human interest."""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

from .models import Candidate, MediaInfo, SearchSettings, Segment
from .process import Cancellation
from .selection import choose_candidates, content_words


def pattern(expression: str):
    return re.compile(expression, re.IGNORECASE)


LAUGH = pattern(r"\b(?:смех|сме[её]тся|сме[её]мся|смеял\w*|рассмеял\w*|хохот\w*|ржу|ору|ахах\w*|ха[ -]?ха(?:[ -]?ха)*)\b")
HUMOR = pattern(r"\b(?:смешн\w*|шутк\w*|прикол\w*|пранк\w*|подкол\w*)\b")
REACTION = pattern(r"\b(?:офиг\w*|обалд\w*|охрен\w*|невероятно|серь[её]зно|прекрати\w*)\b|\b(?:не верю|не может быть|вот это да|ты издеваешься|что (?:ты )?творишь|что (?:ты )?нес[её]шь)\b")
TWIST = pattern(r"\b(?:вдруг|внезапно|неожиданно|неожиданный|не ожидал\w*)\b|\b(?:но тут|не поверите)\b")
HOOK = pattern(r"\b(?:однажды|помню|представьте)\b|\b(?:расскажу|рассказывать)\b.{0,35}\b(?:истори\w*|случа\w*)\b")
ENDING = pattern(r"\b(?:оказалось|выяснилось|закончилось|кончилось|получилось)\b|\b(?:в итоге|в результате|вот так)\b")
QUESTION = pattern(r"\?|\b(?:почему|зачем|кто это|как вас зовут|что (?:ты|вы)|ты ч[её]|вы ч[её])\b")
PROMO = pattern(r"\b(?:промокод\w*|реклам\w*|подписывайтесь|подпишитесь)\b|\b(?:ставьте лайк|ссылка в описании)\b")
NEGATED_HUMOR = pattern(r"\b(?:не смешно|не смешной|не шутка|без шуток)\b")


@dataclass
class Cue:
    laugh: bool
    humor: bool
    reaction: bool
    twist: bool
    hook: bool
    ending: bool
    question: bool
    promo: bool

    @classmethod
    def read(cls, text):
        return cls(bool(LAUGH.search(text)), bool(HUMOR.search(text) and not NEGATED_HUMOR.search(text)),
                   bool(REACTION.search(text)), bool(TWIST.search(text)), bool(HOOK.search(text)),
                   bool(ENDING.search(text)), bool(QUESTION.search(text)), bool(PROMO.search(text)))

    def anchor(self, categories):
        return (("Юмор" in categories and self.laugh) or ("Реакция" in categories and self.reaction)
                or ("Неожиданность" in categories and self.twist)
                or ("История" in categories and (self.hook or self.twist)))


def select_episodes(media: MediaInfo, segments: list[Segment], settings: SearchSettings,
                    energy, token: Cancellation) -> list[Candidate]:
    segments = sorted(segments, key=lambda s: (s.start_ms, s.end_ms))
    cues = [Cue.read(s.text) for s in segments]
    starts = [s.start_ms for s in segments]
    ends = [s.end_ms for s in segments]
    ordered_energy = sorted(energy)
    energy_times = [at for at, _ in ordered_energy]
    lo = min(settings.min_seconds * 1000, media.duration_ms)
    hi = min(settings.max_seconds * 1000, media.duration_ms)
    threshold = max(settings.min_score, 68 - settings.sensitivity * .25)
    evaluated = {}

    def evaluate(left, right):
        key = (left, right)
        if key in evaluated:
            return evaluated[key]
        start, end = starts[left], ends[right]
        items = segments[left:right + 1]
        flags = cues[left:right + 1]
        text = " ".join(s.text for s in items)
        words = text.split()
        if len(words) < 12:
            evaluated[key] = None
            return None
        laughs = [i for i, c in enumerate(flags) if c.laugh]
        reactions = [i for i, c in enumerate(flags) if c.reaction]
        twists = [i for i, c in enumerate(flags) if c.twist]
        hooks = [i for i, c in enumerate(flags) if c.hook]
        endings = [i for i, c in enumerate(flags) if c.ending]
        questions = [i for i, c in enumerate(flags) if c.question]
        strong = sorted(set(laughs + reactions + twists))
        if not strong:
            evaluated[key] = None
            return None
        # A generic connector or isolated shout is insufficient. Require surrounding evidence.
        first_event = strong[0]
        same_phrase = items[first_event].text
        first_cue = min(match.start() for regex in (LAUGH, REACTION, TWIST)
                        for match in [regex.search(same_phrase)] if match)
        question = QUESTION.search(same_phrase)
        conversation = any(q < first_event for q in questions) or bool(question and question.end() < first_cue)
        story = any(h.start() < t.start() < e.start() for h in HOOK.finditer(text)
                    for t in TWIST.finditer(text) for e in ENDING.finditer(text))
        response = any(a < b for a in reactions + twists for b in laughs)
        if not (conversation or story or response or len(strong) >= 2):
            evaluated[key] = None
            return None
        category_values = {
            "Юмор": 16 * bool(laughs) + 4 * any(c.humor for c in flags) + 5 * response,
            "Реакция": 16 * bool(reactions) + 4 * (conversation and bool(reactions)),
            "Неожиданность": 12 * bool(twists) + 4 * (bool(twists) and (conversation or response)),
            "История": 4 * bool(hooks) + 12 * story + 6 * (story and bool(endings)),
        }
        category = max(settings.categories, key=lambda c: category_values.get(c, 0))
        if not category_values.get(category, 0):
            evaluated[key] = None
            return None
        # Count kinds of evidence, rather than occurrences of the same word.
        score = (22 + 16 * bool(laughs) + 16 * bool(reactions) + 10 * bool(twists)
                 + 8 * conversation + 12 * story + 8 * response + min(4, max(0, len(strong) - 1) * 2))
        coverage = sum(max(0, min(s.end_ms, end) - max(s.start_ms, start)) for s in items) / (end - start)
        score += min(5, coverage * 5)
        if re.search(r"[.!?…][\]\)]?$", items[-1].text.strip()):
            score += 3
        last_event = max(strong + endings)
        # Keep a subsequent response; discourage extending into an unrelated quiet tail.
        score -= max(0, end - items[last_event].end_ms - 6000) / 6000
        # Prefer the end of the response/outcome over the first eligible 60-second boundary.
        if flags[-1].laugh or flags[-1].reaction or flags[-1].ending:
            score += 4
        score -= min(30, sum(c.promo for c in flags) * 15)
        unique = content_words(text)
        if len(unique) < 6:
            score -= 20
        elif len(unique) / len(words) < .2:
            score -= 12
        # Sound is supporting evidence only and cannot create a speech candidate by itself.
        levels = [value for _, value in ordered_energy[bisect.bisect_left(energy_times, start):
                                                      bisect.bisect_left(energy_times, end)]]
        if levels:
            ordered = sorted(levels)
            median = ordered[len(ordered) // 2]
            score += min(4, max(0, max(levels) / max(.01, median) - 1))
        score = round(max(0, min(96, score)))
        if score < threshold:
            evaluated[key] = None
            return None
        evidence = []
        if conversation:
            evidence.append("вопрос перед реакцией")
        if story:
            evidence.append("признаки завязки, поворота и завершения")
        if laughs:
            evidence.append("смех в расшифровке")
        if response:
            evidence.append("реакция после события")
        if reactions:
            evidence.append("эмоциональная реплика")
        excerpt_left = max(0, strong[0] - 1)
        excerpt = " ".join(s.text for s in items[excerpt_left:min(len(items), strong[-1] + 2)])
        c = Candidate(media.source_file_id, start, end, score, category,
                      "; ".join(evidence).capitalize() + ". Границы по репликам. Локальная эвристика: проверьте смысл.",
                      excerpt[:1000], provider="local-context", version=2)
        c.validate(media)
        evaluated[key] = c
        return c

    def retrieve(start_budget, end_budget):
        for seed, cue in enumerate(cues):
            token.check()
            if not cue.anchor(settings.categories):
                continue
            # Search phrase boundaries across the entire allowed duration.
            left_min = bisect.bisect_left(starts, max(0, ends[seed] - hi))
            possible_starts = list(range(left_min, seed + 1))
            def start_priority(index):
                pause = starts[index] - ends[index - 1] if index else 2000
                setup = cues[index].question or cues[index].hook
                return (-(2 * setup + (pause >= 1000)), abs(starts[seed] - starts[index] - lo // 3))
            possible_starts.sort(key=start_priority)
            for left in possible_starts[:start_budget]:
                token.check()
                end_min = max(ends[seed], starts[left] + lo)
                end_max = min(media.duration_ms, starts[left] + hi)
                right_stop = bisect.bisect_right(starts, end_max)
                eligible = [i for i in range(seed, right_stop) if end_min <= ends[i] <= end_max]
                eligible.sort(key=lambda i: (-(cues[i].laugh + cues[i].reaction + cues[i].ending), ends[i]))
                for right in eligible[:end_budget]:
                    evaluate(left, right)
        return choose_candidates([c for c in evaluated.values() if c], settings, token)

    selected = retrieve(16, 20)
    if len(selected) < settings.target_count:
        # Widen retrieval, never the score floor, before reporting a shortfall.
        selected = retrieve(64, 40)
    return selected
