"""Import externally prepared transcripts for local semantic search."""

from __future__ import annotations

import re

from .models import Segment, parse_time


def from_srt(text: str, duration: int) -> list[Segment]:
    result = []
    for block in re.split(r"\n\s*\n", text.replace("\r", "").lstrip("\ufeff").strip()):
        lines = block.splitlines()
        index = next((i for i, line in enumerate(lines) if " --> " in line), None)
        if index is None or index + 1 == len(lines):
            raise ValueError("Некорректный SRT")
        left, right = lines[index].split(" --> ", 1)
        segment = Segment(
            parse_time(left), parse_time(right.split()[0]), " ".join(lines[index + 1 :])
        )
        segment.validate(duration)
        result.append(segment)
    return sorted(result, key=lambda s: s.start_ms)
