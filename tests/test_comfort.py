import array
import json
import subprocess

import pytest

from clipnado.export import composition_frame, export_clip
from clipnado.media import probe
from clipnado.models import Candidate, ExportSettings, Segment
from clipnado.process import Cancellation
from clipnado.subtitles import readable_segments, to_ass
from test_export import frame


def candidate(media):
    return Candidate(
        media.source_file_id,
        1000,
        4000,
        80,
        "Юмор",
        "Неожиданный ответ",
        "Смешная история с неожиданным итогом",
        selected=True,
    )


def test_readable_captions_preserve_text_two_lines_and_bounds():
    text = " ".join(
        [
            "Эта реплика содержит много слов, чтобы проверить разбиение длинных субтитров без потери содержания."
        ]
        * 5
    )
    cards = readable_segments([Segment(50, 6050, text)])
    assert len(cards) > 2 and cards[0].start_ms == 50 and cards[-1].end_ms == 6050
    assert all(a.end_ms == b.start_ms for a, b in zip(cards, cards[1:]))
    assert all(
        0 < c.end_ms - c.start_ms and len(c.text.splitlines()) <= 2 for c in cards
    )
    assert " ".join(" ".join(c.text.split()) for c in cards) == text
    ass = to_ass(cards, 1080, 1920, 0.31, readable=True)
    assert "\\N" in ass and "WrapStyle: 2" in ass


def test_real_comfort_export_preview_sync_and_empty_lower_zone(media, tools, tmp_path):
    from PySide6.QtGui import QImage

    c = candidate(media)
    settings = ExportSettings(
        resolution=720,
        color="soft",
        normalize_audio=True,
        subtitles="burn",
    )
    segments = [
        Segment(
            0,
            8000,
            "Проверяем читаемые субтитры на русском языке и свободную нижнюю часть.",
        )
    ]
    out = export_clip(
        media,
        c,
        segments,
        settings,
        tmp_path / "Экспорт ' тест",
        tools,
        Cancellation(),
        lambda *_: None,
    )
    width, height = settings.dimensions()
    image = frame(out, tools, 0.3)
    assert image[(height // 5 * width + width // 2) * 3] > 180
    lower = image[width * int(height * 0.45) * 3 :]
    assert max(lower) <= 5
    later = frame(out, tools, 1.7)
    assert max(later[width * int(height * 0.45) * 3 :]) <= 5
    raw = subprocess.check_output(
        [
            tools.ffmpeg,
            "-v",
            "error",
            "-i",
            str(out),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "8000",
            "-f",
            "s16le",
            "-",
        ]
    )
    samples = array.array("h", raw)
    first = next(i for i, x in enumerate(samples) if abs(x) > 1000) / 8000
    assert 0.9 < first < 1.1
    preview = QImage.fromData(
        composition_frame(media, c, settings, tools, Cancellation(), segments)
    )
    assert (preview.width(), preview.height()) == (360, 640)
    assert preview.pixelColor(180, 500).value() <= 5


@pytest.mark.parametrize("layout", ["blur_center", "gaming_split"])
def test_background_dimming_preserves_foreground(media, tools, tmp_path, layout):
    settings = ExportSettings(
        resolution=720,
        layout=layout,
        background_dim=0.4,
        color="bright",
        gaming_camera=[0.7, 0, 0.3, 0.3],
    )
    c = candidate(media)
    output = export_clip(
        media,
        c,
        [],
        settings,
        tmp_path,
        tools,
        Cancellation(),
        lambda *_: None,
    )
    image = frame(output, tools, 0.2)
    width, height = settings.dimensions()
    if layout == "blur_center":
        bg = (height // 10 * width + width // 2) * 3
        fg = (height // 2 * width + width // 2) * 3
        assert 130 < image[bg] < 175 and image[fg] > 240


def test_loudness_real_output_target_silence_and_peak(tools, tmp_path):
    from clipnado.comfort import audio_filter

    src = tmp_path / "quiet.mp4"
    subprocess.run(
        [
            tools.ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x90:r=24:d=4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=4",
            "-af",
            "volume=0.8",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(src),
        ],
        check=True,
    )
    info = probe(src, tools)
    c = Candidate(info.source_file_id, 0, 4000, 80, "", "")
    settings = ExportSettings(resolution=720, normalize_audio=True, color="bright")
    output = export_clip(
        info,
        c,
        [],
        settings,
        tmp_path / "output",
        tools,
        Cancellation(),
        lambda *_: None,
    )
    report = subprocess.run(
        [
            tools.ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(output),
            "-af",
            "loudnorm=I=-16:TP=-1.5:print_format=json",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    import re

    data = json.loads(re.findall(r'\{\s*"input_i".*?\}', report.stderr, re.S)[-1])
    assert abs(float(data["input_i"]) + 16) < 1 and float(data["input_tp"]) < -0.9
    assert max(frame(output, tools, 1)) <= 5  # Gamma does not turn black into grey.
    silent = tmp_path / "silent.mp4"
    subprocess.run(
        [
            tools.ffmpeg,
            "-v",
            "error",
            "-i",
            str(src),
            "-af",
            "volume=0",
            "-c:v",
            "copy",
            str(silent),
        ],
        check=True,
    )
    info = probe(silent, tools)
    c.source_file_id = info.source_file_id
    assert (
        audio_filter(info, c, settings, tools, Cancellation(), lambda *_: None)
        == "aresample=async=1:first_pts=0"
    )
