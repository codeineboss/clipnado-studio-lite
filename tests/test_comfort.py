import array
import copy
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import pytest

from clipnado.animations import (
    LOOP_SECONDS,
    SCENES,
    default_plan,
    plan_key,
    prepare_animations,
    render_frame,
    render_loop,
    selected_plan,
    validate_plan,
)
from clipnado.export import composition_frame, export_clip
from clipnado.media import Tools, probe
from clipnado.models import Candidate, ExportSettings, Project, Segment
from clipnado.process import Cancellation, Cancelled
from clipnado.projects import load_project, save_project
from clipnado.subtitles import readable_segments, from_srt, to_ass
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


def test_animation_plans_stable_distinct_and_invalidated(media, tmp_path):
    c = candidate(media)
    first = default_plan(c, "varied")
    assert first == default_plan(c, "varied")
    other = candidate(media)
    assert first["seed"] != default_plan(other, "varied")["seed"]
    c.animation_plan = first
    project = Project(
        media,
        candidates=[c],
        export=ExportSettings(animation="varied", color="soft", normalize_audio=True),
    )
    save_project(project, tmp_path / "проект.json")
    restored = load_project(tmp_path / "проект.json")
    assert restored.candidates[0].animation_plan == first
    assert restored.export == project.export
    c.start_ms += 100
    assert selected_plan(c, "varied")["key"] != first["key"]
    assert selected_plan(c, "waves")["scene"] == "waves"
    data = project.to_dict()
    data["export"].pop("background_dim")
    data["export"].pop("readable_subtitles")
    (tmp_path / "old.json").write_text(json.dumps(data), encoding="utf-8")
    old = load_project(tmp_path / "old.json")
    assert old.export.background_dim == 0 and not old.export.readable_subtitles


@pytest.mark.parametrize(
    "field,value",
    [
        ("scene", "exec('bad')"),
        ("scene", []),
        ("palette", {}),
        ("tempo", True),
        ("seed", -1),
        ("key", "../../file"),
    ],
)
def test_animation_plan_rejects_untrusted_data(media, field, value):
    plan = default_plan(candidate(media), "varied")
    plan[field] = value
    with pytest.raises(ValueError):
        validate_plan(plan)


@pytest.mark.parametrize("scene", list(SCENES))
def test_procedural_scenes_move_and_loop_without_flashes(media, scene):
    plan = default_plan(candidate(media), scene)

    def pixels(t):
        image = render_frame(plan, 360, 384, t)
        return bytes(image.constBits())

    first = pixels(0)
    assert first != pixels(1.7)
    # Periodic geometry closes its loop; anti-alias rounding can change by 1.
    end = pixels(LOOP_SECONDS)
    assert sum(abs(a - b) for a, b in zip(first, end)) / len(first) < 0.02
    assert max(first) <= 212
    assert sum(first) / len(first) < 60


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


def test_real_animation_export_preview_cache_and_sync(media, tools, tmp_path):
    from PySide6.QtGui import QImage

    c = candidate(media)
    c.animation_plan = default_plan(c, "waves")
    settings = ExportSettings(
        resolution=720,
        animation="waves",
        color="soft",
        normalize_audio=True,
        subtitles="burn",
    )
    segments = [
        Segment(
            0,
            8000,
            "Проверяем читаемые субтитры на русском языке и движение в нижней части.",
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
        cache=tmp_path / "cache",
    )
    width, height = settings.dimensions()
    image = frame(out, tools, 0.3)
    assert image[(height // 5 * width + width // 2) * 3] > 180
    lower = image[width * int(height * 0.45) * 3 :]
    assert max(lower) > 80 and sum(lower) / len(lower) < 60
    later = frame(out, tools, 1.7)
    assert lower != later[width * int(height * 0.45) * 3 :]
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
    assert preview.pixelColor(180, 500).value() > 8
    files = list((tmp_path / "cache" / "animation-video").glob("*.mp4"))
    assert len(files) == 1
    stamp = files[0].stat().st_mtime_ns
    assert (
        render_loop(
            c.animation_plan,
            settings,
            tmp_path / "cache",
            tools,
            Cancellation(),
            lambda *_: None,
        )
        == files[0]
    )
    assert files[0].stat().st_mtime_ns == stamp
    assert not list((tmp_path / "cache" / "animation-video").glob("render-*"))


@pytest.mark.parametrize("layout", ["blur_center", "gaming_split"])
def test_other_layouts_ignore_animation_and_dim_only_background(
    media, tools, tmp_path, layout
):
    settings = ExportSettings(
        resolution=720,
        layout=layout,
        animation="varied",
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
        cache=tmp_path / "cache",
    )
    assert not (tmp_path / "cache").exists()
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


def test_qwen_plan_failure_is_honest_and_cancellable(media, tools, tmp_path):
    c = candidate(media)
    warnings = prepare_animations(
        [c],
        ExportSettings(animation="qwen"),
        tools,
        tmp_path,
        Cancellation(),
        lambda *_: None,
    )
    assert warnings and "локальный автоматический выбор" in warnings[0]
    assert c.animation_plan["origin"] == "procedural"
    token = Cancellation()
    token.cancel()
    with pytest.raises(Cancelled):
        render_loop(
            c.animation_plan,
            ExportSettings(resolution=720),
            tmp_path,
            tools,
            token,
            lambda *_: None,
        )
    assert not list(tmp_path.rglob("*.mp4"))


@pytest.mark.skipif(
    not os.environ.get("CLIPNADO_TEST_LLAMA")
    or not os.environ.get("CLIPNADO_TEST_QWEN"),
    reason="Set actual local Qwen paths",
)
def test_actual_qwen_animation_planning_and_cache(media, tmp_path, monkeypatch):
    tools = Tools(
        llama=os.environ["CLIPNADO_TEST_LLAMA"],
        semantic_model=os.environ["CLIPNADO_TEST_QWEN"],
        threads=2,
        acceleration="cpu",
    )
    settings = ExportSettings(animation="qwen")
    clips = [candidate(media), candidate(media)]
    clips[1].category = "История"
    clips[1].transcript_excerpt = "Спокойный рассказ о прогулке по ночному городу."
    warnings = prepare_animations(
        clips, settings, tools, tmp_path, Cancellation(), lambda *_: None
    )
    assert not warnings, warnings
    assert all(c.animation_plan["origin"] == "qwen" for c in clips)
    plans = copy.deepcopy([c.animation_plan for c in clips])
    for c in clips:
        c.animation_plan = None

    def repeated(*_, **__):
        pytest.fail("Repeated Qwen inference")

    monkeypatch.setattr("clipnado.local_llm.run", repeated)
    assert not prepare_animations(
        clips, settings, tools, tmp_path, Cancellation(), lambda *_: None
    )
    assert [c.animation_plan for c in clips] == plans


def test_animation_cancel_during_render_removes_partial_files(media, tools, tmp_path):
    c = candidate(media)
    token = Cancellation()

    def progress(_value, text):
        # Heartbeats occur every half-second; let the child start rendering.
        if "Подготовка анимации" in text and "0 с" not in text:
            token.cancel()

    # Slow enough that an encoder is alive when cancellation arrives.
    settings = ExportSettings(resolution=1080, fps=60, animation="gears")
    import threading

    timer = threading.Timer(0.8, token.cancel)
    timer.start()
    try:
        with pytest.raises(Cancelled):
            render_loop(
                default_plan(c, "gears"), settings, tmp_path, tools, token, progress
            )
    finally:
        timer.cancel()
    assert not list(tmp_path.rglob("*.mp4")) and not list(tmp_path.rglob("render-*"))
