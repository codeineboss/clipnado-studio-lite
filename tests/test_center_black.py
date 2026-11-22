"""Real centered exports: same foreground size, darker blur, saved preferences."""

from dataclasses import replace
import json

import pytest
from PySide6.QtGui import QImage

from clipnado.export import composition_frame, export_clip
from clipnado.media import probe
from clipnado.models import Candidate, ExportSettings, Project
from clipnado.process import Cancellation
from clipnado.projects import load_project, save_project
from test_export import frame


@pytest.mark.parametrize("aspect", ["9:16", "1:1", "16:9"])
@pytest.mark.parametrize("resolution", [720, 1080])
def test_center_black_preserves_frame_size_audio_and_dark_blur(media, tools, tmp_path, aspect, resolution):
    clip = Candidate(media.source_file_id, 1000, 4000, 80, "Юмор", "Кадр")
    # Upper-zone controls must not resize/reposition the centered layouts.
    settings = ExportSettings(layout="center_black", aspect=aspect, resolution=resolution,
                              top_fraction=.2, video_y=.8)
    out = export_clip(media, clip, [], settings, tmp_path, tools, Cancellation(), lambda *_: None)
    info = probe(out, tools)
    width, height = settings.dimensions()
    assert (info.width, info.height, info.has_audio) == (width, height, True)
    red = frame(out, tools, .2)
    white = frame(out, tools, 1.5)  # Source flash/tone at t=2..3, clip starts at 1.
    middle = (height//2 * width + width//2)*3
    assert red[middle] > 220 and max(red[middle+1:middle+3]) < 30
    assert min(white[middle:middle+3]) > 230

    images = [QImage.fromData(composition_frame(media, clip, s, tools, Cancellation()))
              for s in (settings, replace(settings, layout="blur_center"),
                        replace(settings, layout="blur_center", background_dim=.25))]
    black, darker, old = images
    assert black.size() == darker.size() == old.size()
    x = black.width()//2
    bounds = []
    for image in images:
        ys = [y for y in range(image.height()) if image.pixelColor(x, y).value() > 225]
        assert ys
        bounds.append((min(ys), max(ys)))
    for start, end in bounds:
        assert abs(start-bounds[0][0]) <= 1 and abs(end-bounds[0][1]) <= 1
    assert abs(bounds[0][0] - (black.height()-1-bounds[0][1])) <= 2
    middle_y = black.height()//2
    assert black.pixelColor(x, middle_y) == darker.pixelColor(x, middle_y) == old.pixelColor(x, middle_y)
    if aspect != "16:9":
        assert black.pixelColor(x, 2).value() <= 5
        assert black.pixelColor(x, black.height()-3).value() <= 5
        assert darker.pixelColor(x, 2).value() < old.pixelColor(x, 2).value()-20
        assert max(red[:width*3]) <= 5 and max(red[-width*3:]) <= 5
    assert settings.background_dim == .4
    assert not list(tmp_path.glob('*.srt'))


def test_center_black_legacy_brightness_preferences(media, tmp_path):
    clip = Candidate(media.source_file_id, 0, 1000, 80, "Юмор", "Кадр",
                     selected=True, export_layout="center_black")
    path = tmp_path/'project.json'
    for brightness in (.25, .4):
        project = Project(media, candidates=[clip],
                          export=ExportSettings(layout="center_black", background_dim=brightness))
        save_project(project, path)
        loaded = load_project(path)
        assert loaded.export.layout == "center_black" and loaded.export.background_dim == brightness
        assert loaded.candidates == [clip]
    legacy = json.loads(path.read_text())
    legacy['export'].pop('background_dim')
    path.write_text(json.dumps(legacy), encoding='utf-8')
    assert load_project(path).export.background_dim == 0
