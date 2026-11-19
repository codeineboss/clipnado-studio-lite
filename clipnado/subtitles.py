from __future__ import annotations

import html
import math
import re

from .models import Segment, parse_time, timecode


def clip_segments(segments: list[Segment], start: int, end: int) -> list[Segment]:
    return sorted([Segment(max(0, s.start_ms - start), min(end, s.end_ms) - start, s.text)
                   for s in segments if s.start_ms < end and s.end_ms > start and s.text.strip()], key=lambda s: s.start_ms)


def to_srt(segments: list[Segment]) -> str:
    return "\n\n".join(f"{i}\n{timecode(s.start_ms, True)} --> {timecode(s.end_ms, True)}\n{html.unescape(re.sub('<[^>]*>', '', s.text)).strip()}"
                       for i, s in enumerate(segments, 1)) + "\n"


def from_srt(text: str, duration: int) -> list[Segment]:
    result = []
    for block in re.split(r"\n\s*\n", text.replace("\r", "").lstrip("\ufeff").strip()):
        lines = block.splitlines()
        index = next((i for i, line in enumerate(lines) if " --> " in line), None)
        if index is None or index + 1 == len(lines):
            raise ValueError("Некорректный SRT")
        left, right = lines[index].split(" --> ", 1)
        segment = Segment(parse_time(left), parse_time(right.split()[0]), " ".join(lines[index + 1:]))
        segment.validate(duration)
        result.append(segment)
    return sorted(result, key=lambda s: s.start_ms)


def readable_segments(segments: list[Segment], characters: int = 32) -> list[Segment]:
    """Two short lines per card. Timing within a segment is estimated, not word alignment."""
    result = []
    for segment in segments:
        words = html.unescape(re.sub('<[^>]*>', '', segment.text)).split()
        lines, line = [], ""
        for word in words:
            if line and len(line) + len(word) + 1 > characters:
                lines.append(line)
                line = word
            else:
                line = (line + " " + word).strip()
        if line:
            lines.append(line)
        pages = ["\n".join(lines[i:i+2]) for i in range(0, len(lines), 2)]
        # Preserve every word even for dense imported captions. A millisecond is
        # the minimum legal event; fewer cards then use a smaller font instead.
        if len(pages) > segment.end_ms - segment.start_ms:
            pages = [" ".join(words)]
        total = sum(len(p) for p in pages)
        elapsed = 0
        for i, page in enumerate(pages):
            left = segment.start_ms + round((segment.end_ms-segment.start_ms) * elapsed / total)
            elapsed += len(page)
            right = segment.end_ms if i == len(pages)-1 else segment.start_ms + round((segment.end_ms-segment.start_ms) * elapsed / total)
            if right > left:
                result.append(Segment(left, right, page))
    return result


def to_ass(segments: list[Segment], width: int, height: int, y: float, video_y: float = 0, zone_fraction: float = .4, readable: bool = False) -> str:
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: {2 if readable else 0}
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{max(20, round(width * .035))},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,8,30,30,10,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    def stamp(ms):
        return timecode(ms)[:-1].lstrip("0") if ms >= 3600000 else "0" + timecode(ms)[2:-1]
    rows = []
    for s in segments:
        text = s.text.replace("\\", " ").replace("{", "(").replace("}", ")").replace("\n", "\\N").replace("\r", "")
        font = max(20, round(width * (.047 if readable else .035)))
        available_height = max(10, height * (video_y + zone_fraction - y) - 6)
        while font > 12 and ((readable and max(len(line) for line in text.split("\\N")) * font * .6 > width - 60) or max(math.ceil(len(line) * font * .6 / max(1, width - 60)) for line in text.split("\\N")) * len(text.split("\\N")) * font * 1.3 > available_height):
            font -= 1
        clip = f"\\clip(0,{round(height * video_y)},{width},{int(height * (video_y + zone_fraction)) // 2 * 2})"
        rows.append(f"Dialogue: 0,{stamp(s.start_ms)},{stamp(s.end_ms)},Default,,0,0,0,,{{\\pos({width // 2},{round(height * y)})\\fs{font}{clip}}}{text}")
    return header + "\n".join(rows) + "\n"
