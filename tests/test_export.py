import array
import subprocess
from pathlib import Path

import pytest

from clipnado.export import export_clip
from clipnado.media import probe
from clipnado.models import Candidate, ExportSettings, Segment
from clipnado.process import Cancellation, Cancelled
from clipnado.subtitles import from_srt


def frame(path, tools, at=0, vf=None):
    args = [tools.ffmpeg, "-v", "error", "-ss", str(at), "-i", str(path), "-frames:v", "1"]
    if vf:
        args += ["-vf", vf]
    return subprocess.check_output(args + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])


@pytest.mark.parametrize("aspect,subs", [("9:16", "off"), ("16:9", "srt"), ("1:1", "burn"), ("9:16", "burn"), ("16:9", "burn")])
def test_real_export(media, tools, tmp_path, aspect, subs):
    c = Candidate(media.source_file_id, 1000, 4000, 80, "История", "Тест")
    settings = ExportSettings(aspect=aspect, resolution=720, subtitles=subs)
    out = export_clip(media, c, [Segment(500, 5000, "Привет, мир! Это тест кириллицы.")], settings,
                      tmp_path / "Экспорт с пробелом ' тест", tools, Cancellation(), lambda *_: None)
    assert out.exists()
    width, height = settings.dimensions()
    image = frame(out, tools, .2)
    # Entire lower zone remains black, including when captions are burned into the upper zone.
    bottom = image[width * int(height * .50) * 3:]
    assert max(bottom) <= 5
    # Red source's visible rectangle has 16:9 proportions within the upper zone.
    middle_x = width // 2
    rows = [y for y in range(int(height * .4)) if image[(y * width + middle_x) * 3] > 150]
    expected_height = min(width / (16 / 9), int(height * .4) // 2 * 2)
    assert abs(len(rows) - expected_height) < 8
    assert abs(probe(out, tools).duration_ms - 3000) <= 100
    if subs == "srt":
        segments = from_srt(out.with_suffix(".srt").read_text(encoding="utf-8-sig"), 3000)
        assert segments[0].start_ms == 0 and segments[0].end_ms == 3000
        assert "Привет" in segments[0].text
    else:
        assert not out.with_suffix(".srt").exists()
    if subs == "burn":
        # White Cyrillic glyph pixels in an otherwise red/black frame prove actual subtitle rendering.
        zone = image[:width * int(height * .4) * 3]
        whites = sum(1 for r, g, b in zip(zone[::3], zone[1::3], zone[2::3]) if min(r, g, b) > 180)
        assert whites > 100
    # The tone and white flash still begin at exported t≈1 s (source t=2 s).
    raw = subprocess.check_output([tools.ffmpeg, "-v", "error", "-i", str(out), "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"])
    samples = array.array("h", raw)
    first = next(i for i, x in enumerate(samples) if abs(x) > 1000) / 8000
    assert .9 < first < 1.1
    before = frame(out, tools, .7)
    during = frame(out, tools, 1.2)
    pixel = (int(height * .2) * width + width // 2) * 3
    assert before[pixel + 1] < 30 and during[pixel + 1] > 200


def test_cancel_and_disk_full(media, tools, tmp_path, monkeypatch):
    candidate = Candidate(media.source_file_id, 0, 5000, 0, "", "")
    token = Cancellation()
    token.cancel()
    with pytest.raises(Cancelled):
        export_clip(media, candidate, [], ExportSettings(), tmp_path, tools, token, lambda *_: None)
    def fail(*_):
        raise OSError("Недостаточно свободного места")
    monkeypatch.setattr("clipnado.export.require_space", fail)
    with pytest.raises(OSError, match="места"):
        export_clip(media, candidate, [], ExportSettings(), tmp_path, tools, Cancellation(), lambda *_: None)
    assert not list(tmp_path.glob("*.mp4"))


def test_export_without_audio_and_reexport(source, tools, tmp_path):
    silent = tmp_path / "silent.mp4"
    subprocess.run([tools.ffmpeg, "-v", "error", "-i", str(source), "-an", "-c:v", "copy", str(silent)], check=True)
    media = probe(silent, tools)
    c = Candidate(media.source_file_id, 0, 1000, 0, "Вручную", "")
    settings = ExportSettings(resolution=720)
    out = export_clip(media, c, [], settings, tmp_path, tools, Cancellation(), lambda *_: None)
    assert not probe(out, tools).has_audio
    c.end_ms = 1200
    second = export_clip(media, c, [], settings, tmp_path, tools, Cancellation(), lambda *_: None)
    assert second != out and out.exists()


def test_missing_subtitles_fail_before_export(media, tools, tmp_path):
    with pytest.raises(ValueError, match="нет текста"):
        export_clip(media, Candidate(media.source_file_id, 0, 1000, 0, "", ""), [], ExportSettings(subtitles="srt"),
                    tmp_path, tools, Cancellation(), lambda *_: None)


def test_default_1080_and_shifted_zone(media, tools, tmp_path):
    settings = ExportSettings(video_y=.1)
    out = export_clip(media, Candidate(media.source_file_id, 0, 1000, 0, "", ""), [], settings, tmp_path, tools, Cancellation(), lambda *_: None)
    info = probe(out, tools)
    assert (info.width, info.height, round(info.fps)) == (1080, 1920, 30)
    image = frame(out, tools)
    assert max(image[:1080 * 180 * 3]) < 5
    assert max(image[1080 * 970 * 3:]) < 5


def test_anamorphic_aspect_is_preserved(tools, tmp_path):
    path = tmp_path / "anamorphic.mp4"
    subprocess.run([tools.ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=320x180:r=25:d=1",
                    "-vf", "setsar=2", "-c:v", "libx264", "-threads", "2", str(path)], check=True)
    media = probe(path, tools)
    out = export_clip(media, Candidate(media.source_file_id, 0, 1000, 0, "", ""), [], ExportSettings(resolution=720),
                      tmp_path, tools, Cancellation(), lambda *_: None)
    image = frame(out, tools)
    rows = [y for y in range(512) if image[(y * 720 + 360) * 3] > 150]
    assert abs(len(rows) - 720 / (32 / 9)) < 6
