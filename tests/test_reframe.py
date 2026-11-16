"""Camera semantics, actual local models, cancellation, and real encoded pixels."""

import json
import math
from pathlib import Path
import subprocess

import pytest

from clipnado.models import Candidate, ExportSettings, Project, Segment
from clipnado.process import Cancellation, Cancelled
from clipnado.reframe import CropPlan, direct_camera, face_position, prepare_plan


def test_director_switches_between_speakers_without_centering_between_faces():
    faces = {0: (60, 70, 80, 100), 1: (500, 70, 80, 100)}
    states = [{"scene": 0, "faces": faces} for _ in range(100)]
    scores = {
        0: {i: 0.95 if i < 40 else 0.1 for i in range(100)},
        1: {i: 0.1 if i < 40 else 0.95 for i in range(100)},
    }
    plan = direct_camera(states, scores, 640, 360, 4000)
    assert plan.at(1500)[0] < 0.05 and plan.at(2100)[0] > 0.95
    assert plan.info["switches"] == 1
    assert all(
        x < 0.05 or x > 0.95 for _, x, _ in plan.points
    ), "Never point the camera at empty space between speakers"
    # Alternating one-frame spikes must not make a chattering camera.
    noisy = {
        0: {i: 0.95 if i % 2 else 0.1 for i in range(100)},
        1: {i: 0.1 if i % 2 else 0.95 for i in range(100)},
    }
    assert direct_camera(states, noisy, 640, 360, 4000).info["switches"] == 0


def test_director_tracks_motion_and_resets_on_new_shots():
    states = [{"scene": 0, "faces": {0: (50 + i * 2, 70, 65, 90)}} for i in range(75)]
    states += [{"scene": 1, "faces": {1: (540, 70, 65, 90)}} for _ in range(25)]
    plan = direct_camera(states, {}, 640, 360, 4000)
    assert plan.at(2000)[0] > plan.at(0)[0]
    assert (
        plan.at(3000)[0] > 0.95
    ), "A shot cut must not pan across an unrelated empty scene"
    assert all(0 <= x <= 1 and 0 <= y <= 1 for _, x, y in plan.points)
    nofaces = direct_camera([{"scene": 0, "faces": {}}] * 25, {}, 640, 360, 1000)
    assert nofaces.at(500) == (0.5, 0.5) and nofaces.warnings
    # If speech is uncertain, choose an actual face and hold it, never average.
    two = direct_camera(
        [{"scene": 0, "faces": {0: (0, 60, 80, 100), 1: (550, 60, 80, 100)}}] * 50,
        {},
        640,
        360,
        2000,
    )
    assert two.at(1500)[0] in (0, 1) and two.warnings


@pytest.mark.parametrize(
    "width,height,box",
    [
        (640, 360, (0, 0, 80, 100)),
        (640, 360, (600, 100, 40, 80)),
        (360, 1000, (100, 0, 50, 80)),
        (360, 1000, (100, 900, 50, 80)),
    ],
)
def test_face_crop_is_in_bounds_and_includes_face(width, height, box):
    x, y = face_position(box, width, height)
    cw, ch = min(width, height * 9 / 16), min(height, width * 16 / 9)
    assert 0 <= x <= 1 and 0 <= y <= 1
    assert (
        x * (width - cw) <= box[0] + 1 and x * (width - cw) + cw >= box[0] + box[2] - 1
    )
    assert (
        y * (height - ch) <= box[1] + 1
        and y * (height - ch) + ch >= box[1] + box[3] - 1
    )


@pytest.mark.parametrize(
    "points",
    [
        [[0, float("nan"), 0.5]],
        [[0, 2, 0.5]],
        [[1, 0.5, 0.5]],
        [[0, 0.5, 0.5], [0, 0.4, 0.4]],
        [[0, 0.5, 0.5], [1000, 0.4, 0.4]],
    ],
)
def test_invalid_cached_crop_is_rejected(points):
    with pytest.raises(ValueError):
        CropPlan(1000, points, {}, []).validate(1000)


def test_actual_face_and_speaker_models_in_isolated_worker_and_cache(
    tools, tmp_path, monkeypatch
):
    from scripts.vision_smoke import make_fixture
    from clipnado.media import probe

    path = make_fixture(tmp_path / "два лица с пробелом.mp4", tools.ffmpeg)
    media = probe(path, tools)
    candidate = Candidate(media.source_file_id, 0, 3000, 80, "", "")
    cache = tmp_path / "кэш с пробелом"
    progress = []
    plan = prepare_plan(
        media, candidate, tools, cache, Cancellation(), lambda *p: progress.append(p)
    )
    assert plan.info["max_faces"] == 2 and plan.info["face_coverage"] > 0.95
    assert plan.info["speaker_model_used"] and plan.at(1000)[0] < 0.2
    assert any("движению рта" in p[1] for p in progress)
    assert not list(cache.rglob("*.jpg")) and not list(cache.rglob("*.wav"))

    def no_run(*_a, **_kw):
        pytest.fail("Cached plan must not start an analysis process")

    monkeypatch.setattr("clipnado.reframe.run", no_run)
    second = prepare_plan(
        media, candidate, tools, cache, Cancellation(), lambda *_: None
    )
    assert second.to_dict() == plan.to_dict()
    token = Cancellation()
    token.cancel()
    with pytest.raises(Cancelled):
        prepare_plan(media, candidate, tools, cache, token, lambda *_: None)


@pytest.mark.parametrize("subtitles", ["off", "srt", "burn"])
def test_dynamic_crop_real_export_and_preview_offsets(
    tools, tmp_path, monkeypatch, subtitles
):
    from clipnado.export import export_clip, composition_frame
    from clipnado.media import probe
    from PySide6.QtGui import QImage
    from test_export import frame

    source = tmp_path / "две стороны.mp4"
    subprocess.run(
        [
            tools.ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=lime:s=640x360:r=25:d=5,drawbox=x=0:y=0:w=220:h=360:color=red:t=fill,drawbox=x=420:y=0:w=220:h=360:color=blue:t=fill",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:duration=5",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ],
        check=True,
    )
    media = probe(source, tools)
    clip = Candidate(media.source_file_id, 1000, 4000, 80, "", "")
    plan = CropPlan(
        3000,
        [[0, 0, 0.5], [800, 1, 0.5], [2000, 0, 0.5]],
        {"max_faces": 2, "switches": 2},
        [],
    )
    # Synthetic speaking scores/plan: tests renderer correctness, not NN accuracy.
    monkeypatch.setattr("clipnado.export.prepare_plan", lambda *_: plan)
    settings = ExportSettings(
        resolution=720, layout="tt_fill", subtitles=subtitles, subtitle_y=0.85
    )
    out = export_clip(
        media,
        clip,
        [Segment(0, 5000, "Два собеседника")],
        settings,
        tmp_path / "результат",
        tools,
        Cancellation(),
        lambda *_: None,
    )
    for at, channel in ((0.3, 0), (1.3, 2), (2.5, 0)):
        pixels = frame(out, tools, at)
        # Center of output is on the chosen person, not the green middle.
        rgb = pixels[(500 * 720 + 360) * 3 :][:3]
        assert rgb[channel] > 180 and rgb[1] < 50
    info = probe(out, tools)
    assert info.has_audio and abs(info.duration_ms - 3000) < 100
    assert out.with_suffix(".srt").exists() == (subtitles == "srt")
    image = QImage.fromData(
        composition_frame(media, clip, settings, tools, Cancellation())
    )
    assert (
        image.pixelColor(180, 250).blue() > 180
    ), "Preview at source start+1s must use crop at clip-relative 1s"


def test_center_setting_skips_vision_and_legacy_project_uses_auto(
    media, tools, tmp_path, monkeypatch
):
    from clipnado.export import export_clip
    from clipnado.projects import load_project

    data = Project(media).to_dict()
    data["export"].pop("tiktok_tracking")
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_project(path).export.tiktok_tracking == "speaker"

    def no_vision(*_a, **_kw):
        pytest.fail("Fixed center mode must not run vision models")

    monkeypatch.setattr("clipnado.export.prepare_plan", no_vision)
    settings = ExportSettings(
        resolution=720, layout="tt_fill", tiktok_tracking="center"
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
    assert out.is_file()
    with pytest.raises(ValueError):
        ExportSettings(tiktok_tracking="network").validate()


def test_mfcc_matches_independent_frontend_golden():
    import numpy as np
    from clipnado.reframe_worker import mfcc

    golden = np.load(
        Path(__file__).resolve().parent.parent / "qa/fixtures/mfcc-reference.npz",
        allow_pickle=False,
    )
    actual = mfcc(golden["signal"])
    np.testing.assert_allclose(actual, golden["features"], rtol=2e-6, atol=2e-5)
    assert np.isfinite(mfcc(np.zeros(400, dtype=np.int16))).all()


def test_cancel_local_vision_cleans_frames_and_does_not_cache(tools, tmp_path):
    from scripts.vision_smoke import make_fixture
    from clipnado.media import probe

    media = probe(make_fixture(tmp_path / "faces.mp4", tools.ffmpeg), tools)
    token = Cancellation()
    cache = tmp_path / "cache"

    def cancel(percent, text):
        if "поиск лиц" in text:
            token.cancel()

    with pytest.raises(Cancelled):
        prepare_plan(
            media,
            Candidate(media.source_file_id, 0, 3000, 80, "", ""),
            tools,
            cache,
            token,
            cancel,
        )
    assert not list(cache.rglob("*.jpg")) and not list(cache.rglob("plan.json"))


def test_subprocess_retains_selected_python_environment():
    import sys
    from clipnado.process import run

    prefix = run(
        [sys.executable, "-c", "import sys;print(sys.prefix)"], Cancellation()
    ).strip()
    assert Path(prefix) == Path(
        sys.prefix
    ), "Resolving a venv interpreter symlink silently selects the base Python"
