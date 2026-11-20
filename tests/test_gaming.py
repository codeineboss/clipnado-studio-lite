import array
import json
from pathlib import Path
import subprocess

import pytest

from clipnado.export import composition_frame, export_clip
from clipnado.media import probe
from clipnado.models import Candidate, ExportSettings, Project, Segment, crop_region
from clipnado.process import Cancellation, Cancelled
from clipnado.projects import load_project, save_project
from scripts.gaming_smoke import CAMERA, GAME


@pytest.mark.parametrize("fps", [30, 60])
def test_gaming_real_export_camera_game_audio_and_clean_video(
    gaming_source, tools, tmp_path, fps
):
    from test_export import frame

    media = probe(gaming_source, tools)
    clip = Candidate(
        media.source_file_id,
        1000,
        4000,
        80,
        "Реакция",
        "Игровой момент",
        export_layout="gaming_split",
    )
    settings = ExportSettings(
        aspect="1:1",
        resolution=720,
        fps=fps,
        gaming_camera=CAMERA,
        gaming_game=GAME,
    )
    out = export_clip(
        media,
        clip,
        [Segment(500, 4500, "Качественная игровая нарезка")],
        settings,
        tmp_path / "Экспорт с пробелом",
        tools,
        Cancellation(),
        lambda *_: None,
    )
    info = probe(out, tools)
    assert (info.width, info.height, round(info.fps), info.has_audio) == (
        720,
        1280,
        fps,
        True,
    )
    assert abs(info.duration_ms - 3000) < 100
    before = frame(out, tools, 0.4)
    flash = frame(out, tools, 1.3)
    after = frame(out, tools, 2.6)

    def rgb(pixels, x, y):
        return tuple(pixels[(y * 720 + x) * 3 :][:3])

    # Exact two source regions: no duplicate camera in game, no reserved dark zone.
    for x, y in ((10, 10), (710, 500), (360, 250)):
        assert rgb(before, x, y)[0] > 180 and rgb(before, x, y)[2] < 40
        assert min(rgb(flash, x, y)) > 180
    for x, y in ((10, 520), (710, 1270), (360, 900)):
        assert rgb(before, x, y)[2] > 180 and rgb(before, x, y)[0] < 40
        assert rgb(flash, x, y)[1] > 180 and rgb(flash, x, y)[0] < 40
        assert rgb(after, x, y)[2] > 180
    whites = sum(
        min(r, g, b) > 180 for r, g, b in zip(before[::3], before[1::3], before[2::3])
    )
    assert whites == 0
    assert not out.with_suffix(".srt").exists()
    samples = array.array(
        "h",
        subprocess.check_output(
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
        ),
    )
    first = next(i for i, x in enumerate(samples) if abs(x) > 1000) / 8000
    assert 0.9 < first < 1.1
    from PySide6.QtGui import QImage

    image = QImage.fromData(
        composition_frame(media, clip, settings, tools, Cancellation())
    )
    assert (image.width(), image.height()) == (360, 640)
    assert (
        image.pixelColor(100, 80).red() > 180
        and image.pixelColor(100, 500).green() > 180
    )


@pytest.mark.parametrize("sar", [1, 2])
def test_gaming_preserves_source_display_proportions_and_detail(tools, tmp_path, sar):
    from test_export import frame

    source = tmp_path / "форма камеры.mp4"
    pattern = (
        "color=c=blue:s=640x360:r=25:d=1,drawbox=x=480:y=0:w=160:h=120:color=red:t=fill,"
        "drawbox=x=550:y=50:w=20:h=20:color=white:t=fill," + f"setsar={sar}"
    )
    subprocess.run(
        [
            tools.ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    media = probe(source, tools)
    settings = ExportSettings(
        layout="gaming_split", resolution=720, gaming_camera=CAMERA, gaming_game=GAME
    )
    out = export_clip(
        media,
        Candidate(media.source_file_id, 0, 1000, 80, "", ""),
        [],
        settings,
        tmp_path,
        tools,
        Cancellation(),
        lambda *_: None,
    )
    pixels = frame(out, tools, 0.2)

    def white(x, y):
        return min(pixels[(y * 720 + x) * 3 :][:3]) > 200

    xs = [x for x in range(720) if white(x, 256)]
    ys = [y for y in range(512) if white(360, y)]
    assert (
        xs and ys and abs(len(xs) / len(ys) - sar) < 0.12
    ), "Camera image was stretched"
    assert (
        pixels[(150 * 720 + 360) * 3] > 180 and pixels[(150 * 720 + 360) * 3 + 1] < 30
    )


@pytest.mark.parametrize("fraction", [0.25, 0.4, 0.6])
def test_gaming_1080_geometry_and_camera_fraction(
    gaming_source, tools, tmp_path, fraction
):
    from test_export import frame

    media = probe(gaming_source, tools)
    settings = ExportSettings(
        layout="gaming_split",
        gaming_camera=CAMERA,
        gaming_game=GAME,
        gaming_fraction=fraction,
    )
    out = export_clip(
        media,
        Candidate(media.source_file_id, 0, 500, 80, "", ""),
        [],
        settings,
        tmp_path,
        tools,
        Cancellation(),
        lambda *_: None,
    )
    info = probe(out, tools)
    assert (info.width, info.height) == (1080, 1920)
    pixels = frame(out, tools, 0.1)
    top = int(1920 * fraction) // 2 * 2
    assert pixels[((top - 4) * 1080 + 540) * 3] > 180
    assert pixels[((top + 4) * 1080 + 540) * 3 + 2] > 180


@pytest.mark.parametrize(
    "rect",
    [
        [-1, 0, 0.2, 0.2],
        [0.9, 0, 0.2, 0.2],
        [0, 0, 0, 0.2],
        [0, 0, 1, 2],
        [0, 0, float("nan"), 0.2],
        ["x", 0, 0.2, 0.2],
        [True, 0, 0.2, 0.2],
        [0, 0, 0.2],
        None,
    ],
)
def test_invalid_gaming_regions_are_rejected(rect):
    with pytest.raises(ValueError):
        crop_region(rect)


def test_gaming_requires_camera_before_creating_files(media, tools, tmp_path):
    folder = tmp_path / "missing-camera"
    with pytest.raises(ValueError, match="Камера и игра"):
        export_clip(
            media,
            Candidate(media.source_file_id, 0, 1000, 80, "", ""),
            [],
            ExportSettings(layout="gaming_split"),
            folder,
            tools,
            Cancellation(),
            lambda *_: None,
        )
    assert not folder.exists()
    token = Cancellation()
    token.cancel()
    with pytest.raises(Cancelled):
        export_clip(
            media,
            Candidate(media.source_file_id, 0, 1000, 80, "", ""),
            [],
            ExportSettings(layout="gaming_split", gaming_camera=CAMERA),
            tmp_path,
            tools,
            token,
            lambda *_: None,
        )


def test_gaming_settings_roundtrip_and_legacy_tiktok_migration(media, tmp_path):
    project = Project(
        media,
        candidates=[
            Candidate(
                media.source_file_id,
                0,
                1000,
                80,
                "",
                "",
                selected=True,
                export_layout="gaming_split",
            )
        ],
        export=ExportSettings(
            layout="gaming_split",
            gaming_camera=CAMERA,
            gaming_game=GAME,
            gaming_fraction=0.35,
        ),
    )
    path = tmp_path / "игровой проект.json"
    save_project(project, path)
    loaded = load_project(path)
    assert loaded.export.gaming_camera == CAMERA and loaded.export.gaming_game == GAME
    assert (
        loaded.export.gaming_fraction == 0.35
        and loaded.candidates[0].export_layout == "gaming_split"
    )
    data = project.to_dict()
    data["export"]["layout"] = "tt_fill"
    data["export"]["tiktok_tracking"] = "speaker"
    for key in ("gaming_camera", "gaming_game", "gaming_fraction"):
        data["export"].pop(key)
    data["candidates"][0]["export_layout"] = "tt_fill"
    data["query_results"] = [dict(data["candidates"][0])]
    path.write_text(json.dumps(data), encoding="utf-8")
    legacy = load_project(path)
    assert (
        legacy.export.layout == "gaming_split" and legacy.export.gaming_camera is None
    )
    assert legacy.query_results[0].export_layout == "gaming_split" and legacy.warnings


def test_black_bar_suggestion_handles_corner_camera_and_black_intro():
    from PySide6.QtGui import QColor, QImage, QPainter
    from PySide6.QtCore import Qt
    from clipnado.gaming import trim_black_bars

    image = QImage(640, 360, QImage.Format_RGB32)
    image.fill(Qt.black)
    painter = QPainter(image)
    painter.fillRect(0, 20, 640, 320, QColor("blue"))
    painter.fillRect(480, 0, 160, 120, QColor("red"))
    painter.end()
    result = trim_black_bars(image)
    assert result == pytest.approx([0, 20 / 360, 1, 320 / 360])
    image.fill(Qt.black)
    assert trim_black_bars(image) == [0, 0, 1, 1]
