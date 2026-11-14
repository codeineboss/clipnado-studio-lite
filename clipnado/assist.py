"""Optional local boundary suggestions. Never mutates the caller's clip."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .local_llm import (ClipProposal, LocalAnalysisResult, LocalModelProvider, PromptTooLong,
                        build_request, clip_proposals, decode_choices, model_identity, stage_model, validate_tools)
from .models import Candidate, MediaInfo, SearchSettings, Segment, timecode
from .process import Cancellation, Progress
from .projects import atomic_json

RADIUS_MS = 30000


def neighboring_proposals(media, segments, anchor, settings, left, right):
    proposals = clip_proposals(media, segments, settings, left, right)
    # Include exact old boundaries as a reference when they meet the length
    # setting. This gives the model an explicit way to recommend no change.
    if settings.min_seconds * 1000 <= anchor.end_ms - anchor.start_ms <= settings.max_seconds * 1000:
        included = [i for i in range(left, right) if segments[i].end_ms > anchor.start_ms and segments[i].start_ms < anchor.end_ms]
        if included:
            proposals.insert(0, ClipProposal(anchor.start_ms, anchor.end_ms, included[0], included[-1],
                any(s.start_ms < t < s.end_ms for s in segments[left:right] for t in (anchor.start_ms, anchor.end_ms))))
    result, seen = [], set()
    for p in proposals:
        overlap = min(anchor.end_ms, p.end_ms) - max(anchor.start_ms, p.start_ms)
        if (abs(p.start_ms - anchor.start_ms) <= RADIUS_MS and abs(p.end_ms - anchor.end_ms) <= RADIUS_MS
            and overlap >= .5 * min(anchor.end_ms - anchor.start_ms, p.end_ms - p.start_ms)
            and (p.start_ms, p.end_ms) not in seen):
            result.append(p)
            seen.add((p.start_ms, p.end_ms))
    return result


class LocalClipAssistant(LocalModelProvider):
    def refine(self, media: MediaInfo, segments: list[Segment], anchor: Candidate, settings: SearchSettings,
               cache: Path, token: Cancellation, progress: Progress):
        token.check()
        anchor = copy.deepcopy(anchor)
        settings = copy.deepcopy(settings)
        settings.validate()
        anchor.validate(media)
        if not segments:
            return LocalAnalysisResult([], ["Для подбора границ нужна расшифровка речи."], {"complete": False})
        validate_tools(self.tools)
        segments = sorted(segments, key=lambda s: (s.start_ms, s.end_ms))
        nearby = [i for i, s in enumerate(segments) if s.end_ms > max(0, anchor.start_ms - RADIUS_MS) and s.start_ms < anchor.end_ms + RADIUS_MS]
        if not nearby:
            return LocalAnalysisResult([], ["Рядом с клипом нет реплик расшифровки."], {"complete": False})
        left, right = nearby[0], nearby[-1] + 1
        proposals = neighboring_proposals(media, segments, anchor, settings, left, right)
        if not proposals:
            return LocalAnalysisResult([], ["Рядом нет интервала заданной длины. Проверьте настройки длины клипов."], {"complete": True})
        model_id = model_identity(Path(self.tools.semantic_model), token)
        payload = {"source": media.source_file_id, "model": model_id, "anchor": [anchor.start_ms, anchor.end_ms],
                   "segments": [asdict(s) for s in segments[left:right]], "settings": asdict(settings), "version": 1}
        identity = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        folder = cache / "local-refine" / identity
        saved = folder / "result.json"
        allowed = {(p.start_ms, p.end_ms) for p in proposals}
        try:
            data = json.loads(saved.read_text(encoding="utf-8"))
            candidates = [Candidate(**c) for c in data["candidates"]]
            for c in candidates:
                c.validate(media)
                if (c.start_ms, c.end_ms) not in allowed:
                    raise ValueError("Invalid cached alternative")
            return LocalAnalysisResult(candidates, [], {"complete": True, "current_recommended": bool(data["current_recommended"]), "cached": True})
        except (OSError, KeyError, ValueError, TypeError):
            pass
        staged = stage_model(Path(self.tools.semantic_model), cache, model_id, token)
        instruction = (f"Это отдельная задача уточнения ОДНОГО клипа {timecode(anchor.start_ms)}–{timecode(anchor.end_ms)}. "
            "Общая цель количества клипов сейчас не применяется. Сравни текущий интервал с предложенными соседними. "
            "Выбирай до трёх вариантов того же события с более понятным началом или сохранённой развязкой. "
            "Не выбирай другой эпизод. В reason объясни пользу границ. Если текущие границы лучше, выбери их; "
            "если полезных вариантов нет, пустой список допустим. Альтернативы могут перекрываться.")
        stats = {"max_prompt_tokens": 0, "warnings": []}
        for cap in (40, 20, 8):
            request = build_request(media, segments, settings, left, right, 1, cap,
                                    proposals_override=proposals, instruction=instruction, limit_override=3)
            try:
                candidates, _, outputs, error = self._window(request, staged, token, progress, stats,
                    lambda items: decode_choices(items, media, segments, settings, request.proposals, model_id))
            except PromptTooLong:
                continue
            if candidates is None:
                folder.mkdir(parents=True, exist_ok=True)
                atomic_json(folder / "failed.json", {"error": error, "outputs": outputs})
                return LocalAnalysisResult([], ["Qwen не смогла проверить варианты границ. Текущий клип сохранён."], {"complete": False})
            current = any((c.start_ms, c.end_ms) == (anchor.start_ms, anchor.end_ms) for c in candidates)
            alternatives = [c for c in candidates if (c.start_ms, c.end_ms) != (anchor.start_ms, anchor.end_ms) and c.score >= settings.min_score]
            alternatives = sorted(alternatives, key=lambda c: -c.score)[:3]
            atomic_json(saved, {"candidates": [asdict(c) for c in alternatives], "current_recommended": current})
            return LocalAnalysisResult(alternatives, stats["warnings"], {"complete": True, "current_recommended": current, "cached": False})
        return LocalAnalysisResult([], ["Контекст этого клипа слишком плотный для сравнения границ. Текущие границы сохранены; их можно изменить в таблице."], {"complete": False})
