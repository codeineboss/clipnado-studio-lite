import json
import math
import sys
import threading
import time

import pytest

from clipnado.analysis import LocalFallbackProvider, deduplicate, recognize
from clipnado.media import Tools, probe
from clipnado.models import Candidate, ExportSettings, MediaInfo, Project, SearchSettings, Segment, parse_time, timecode
from clipnado.process import Cancellation, Cancelled, run
from clipnado.projects import load_project, save_project
from clipnado.subtitles import clip_segments, from_srt, to_ass, to_srt


@pytest.mark.parametrize("ms", [0, 1, 1234, 3661123, 18000000])
def test_time_roundtrip(ms):
    assert parse_time(timecode(ms)) == ms
    assert parse_time(timecode(ms, True)) == ms


@pytest.mark.parametrize("text", ["00:60:00", "-1:00:00", "00:00:nan", "0", "00:00:60"])
def test_bad_time(text):
    with pytest.raises(ValueError):
        parse_time(text)


def test_project_atomic_roundtrip(media, tmp_path):
    p = Project(media, [Segment(0, 1000, "Привет, мир!")], [Candidate(media.source_file_id, 0, 1000, 70, "История", "Тест")])
    path = tmp_path / "проект.json"
    save_project(p, path)
    assert load_project(path).to_dict() == p.to_dict()
    assert "API" not in path.read_text()
    p.candidates[0].end_ms = media.duration_ms + 1
    with pytest.raises(ValueError):
        save_project(p, path)
    assert load_project(path).candidates[0].end_ms == 1000
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("start,end,score", [(-1, 100, 1), (0, 0, 5), (0, 999999, 5), (0, 100, 101), (float("nan"), 100, 50), (0, 100, 50.5)])
def test_invalid_candidate(media, start, end, score):
    with pytest.raises(ValueError):
        Candidate(media.source_file_id, start, end, score, "Тест", "Тест").validate(media)


def test_corrupt_project(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version":999}')
    with pytest.raises(ValueError):
        load_project(path)


def test_cancel_running_process():
    token = Cancellation()
    timer = threading.Timer(.3, token.cancel)
    timer.start()
    start = time.monotonic()
    try:
        with pytest.raises(Cancelled):
            run([sys.executable, "-c", "import time; time.sleep(15)"], token)
    finally:
        timer.join()
    assert time.monotonic() - start < 5


def test_bad_file(tmp_path, tools):
    path = tmp_path / "bad.mp4"
    path.write_bytes(b"garbage")
    with pytest.raises(RuntimeError):
        probe(path, tools)


def test_srt_clipping():
    original = [Segment(900, 2100, "Привет, мир!"), Segment(2500, 4500, "Конец.")]
    clipped = clip_segments(original, 1000, 4000)
    assert clipped == [Segment(0, 1100, "Привет, мир!"), Segment(1500, 3000, "Конец.")]
    assert clip_segments(list(reversed(original)), 1000, 4000) == clipped
    assert from_srt(to_srt(clipped), 3000) == clipped
    ass = to_ass(clipped, 720, 1280, .31)
    assert "0:00:00.00" in ass and "Привет" in ass


def test_deduplicate():
    clips = [Candidate("x", 0, 60000, 90, "", ""), Candidate("x", 1000, 61000, 80, "", ""), Candidate("x", 65000, 125000, 70, "", "")]
    assert len(deduplicate(clips, 15)) == 2


def test_annotated_transcript_search():
    media = MediaInfo("test.mp4", "id", 360000, 1920, 1080, 30, True)
    text = []
    for i in range(36):
        value = "Спокойно обсуждаем технические детали настройки проекта и проверяем таблицы."
        if 8 <= i <= 15:
            value = "Однажды случилась невероятная история: вдруг всё изменилось, а потом оказалось, что это шутка. В итоге получилось смешно!"
        text.append(Segment(i * 10000, (i + 1) * 10000, value))
    candidates = LocalFallbackProvider().analyze(media, text, SearchSettings(), [], Cancellation())
    assert 1 <= len(candidates) < 15
    assert any(c.start_ms < 160000 and c.end_ms > 80000 for c in candidates)
    assert all(c.start_ms < 160000 and c.end_ms > 80000 for c in candidates)
    assert all(60000 <= c.end_ms - c.start_ms <= 90000 for c in candidates)
    assert candidates == sorted(candidates, key=lambda c: c.start_ms)
    assert len(deduplicate(candidates, 100)) == len(candidates)


def test_neutral_transcript_does_not_fill_quota():
    media = MediaInfo("test.mp4", "id", 120000, 1920, 1080, 30, True)
    segments = [Segment(i, i + 10000, "Продолжаем проверять технические параметры и редактируем рабочие таблицы.") for i in range(0, 120000, 10000)]
    assert LocalFallbackProvider().analyze(media, segments, SearchSettings(), [], Cancellation()) == []


def test_no_sound_and_reuse(media, tools, tmp_path):
    media.has_audio = False
    segments, energy, warnings = recognize(media, tools, tmp_path, Cancellation(), lambda *_: None)
    assert not segments and not energy and warnings
    saved = [Segment(0, 1000, "Текст")]
    assert recognize(media, tools, tmp_path, Cancellation(), lambda *_: None, saved)[0] == saved


@pytest.mark.parametrize("duration", [900, 1800, 18000])
def test_long_import(duration, tmp_path, tools):
    import subprocess
    path = tmp_path / f"long-{duration}.mp4"
    subprocess.run([tools.ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "color=s=160x90:r=1:c=blue",
                    "-t", str(duration), "-c:v", "libx264", "-preset", "ultrafast", "-threads", "2", str(path)], check=True)
    info = probe(path, tools)
    assert abs(info.duration_ms - duration * 1000) < 50
    assert not info.has_audio
    assert info.width == 160 and info.height == 90
