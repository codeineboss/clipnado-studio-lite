"""Editorial preset behavior and genuine local model coverage."""

import copy
import json
import os
from pathlib import Path

import pytest

from clipnado.local_llm import LocalModelProvider, build_request
from clipnado.media import Tools
from clipnado.models import (
    Candidate,
    ExportSettings,
    MediaInfo,
    Project,
    SearchSettings,
    Segment,
)
from clipnado.process import Cancellation
from clipnado.profiles import (
    EXPORT_PROFILES,
    SEARCH_PROFILES,
    export_profile,
    search_profile,
)
from clipnado.projects import load_project, save_project


@pytest.mark.parametrize("code", list(SEARCH_PROFILES))
def test_search_profiles_preserve_goals_and_offer_context(code):
    before = SearchSettings(count=25, target_count=12, min_score=65, query="шутка")
    result = search_profile(before, code)
    p = SEARCH_PROFILES[code]
    assert (result.min_seconds, result.max_seconds, result.profile) == (
        p.minimum,
        p.maximum,
        code,
    )
    assert (result.count, result.target_count, result.min_score, result.query) == (
        25,
        12,
        65,
        "шутка",
    )
    assert before.profile == "custom" and before.min_seconds == 60
    media = MediaInfo("fixture.mp4", "profiles", 120000, 1920, 1080, 30, True)
    segments = [
        Segment(i * 2000, (i + 1) * 2000, "Событие и реакция: реплика " + str(i))
        for i in range(60)
    ]
    request = build_request(media, segments, result, 0, len(segments), 1)
    assert p.instruction in request.system
    assert request.proposals
    assert all(
        p.minimum * 1000 <= x.end_ms - x.start_ms <= p.maximum * 1000
        for x in request.proposals
    )
    assert "Это поиск по содержанию" in request.system


@pytest.mark.parametrize("code", list(EXPORT_PROFILES))
@pytest.mark.parametrize("fps,expected", [(29.97, 30), (59.94, 60), (24, 24), (50, 50)])
def test_export_profiles_preserve_regions_and_choose_quality(code, fps, expected):
    source = MediaInfo("fixture", "source", 100000, 1920, 1080, fps, True)
    before = ExportSettings(
        gaming_camera=[0.75, 0, 0.25, 0.3],
        gaming_game=[0, 0.3, 1, 0.7],
        video_y=0.6,
        top_fraction=0.4,
    )
    result = export_profile(before, code, source, True)
    result.require_gaming_regions()
    assert (result.fps, result.resolution, result.quality, result.aspect) == (
        expected,
        1080,
        18,
        "9:16",
    )
    assert (
        result.subtitles == "burn"
        and result.readable_subtitles
        and result.normalize_audio
    )
    assert result.preset == code
    assert (
        result.gaming_camera == before.gaming_camera
        and result.gaming_game == before.gaming_game
    )
    assert before.video_y == 0.6
    if code == "gaming":
        assert result.gaming_fraction == 0.35
    elif code == "dialogue":
        assert result.layout == "blur_center" and result.background_dim == 0.3
    else:
        assert result.layout == "top_black" and result.video_y == 0
    assert export_profile(before, code, source, False).subtitles == "off"


def test_profile_roundtrip_legacy_and_invalid_names(media, tmp_path):
    p = Project(
        media,
        search=search_profile(SearchSettings(), "memes"),
        export=export_profile(ExportSettings(), "dialogue", media, True),
    )
    path = tmp_path / "профили.json"
    save_project(p, path)
    again = load_project(path)
    assert again.search == p.search and again.export == p.export
    data = p.to_dict()
    data["search"].pop("profile")
    data["export"].pop("preset")
    data["search"].update(min_seconds=60, max_seconds=90)
    path.write_text(json.dumps(data), encoding="utf-8")
    old = load_project(path)
    assert (
        old.search.profile == "custom"
        and old.search.min_seconds == 60
        and old.export.preset == "custom"
    )
    for value in ([], "viral-guaranteed", False):
        with pytest.raises(ValueError):
            SearchSettings(profile=value).validate()
        with pytest.raises(ValueError):
            ExportSettings(preset=value).validate()


def test_profile_cache_is_separate_and_reused(tmp_path, monkeypatch):
    import sys

    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF-local-test")
    tools = Tools(llama=sys.executable, semantic_model=str(model), acceleration="cpu")
    media = MediaInfo("fixture", "profiles", 40000, 1920, 1080, 30, True)
    segments = [
        Segment(i * 2000, (i + 1) * 2000, "Реплика смешного диалога номер " + str(i))
        for i in range(20)
    ]
    seen = []

    def window(self, request, staged, token, progress, stats, decode):
        seen.append(request.system)
        return decode([]), 1000, [], None

    monkeypatch.setattr(LocalModelProvider, "_window", window)
    provider = LocalModelProvider(tools)
    settings = search_profile(SearchSettings(), "quote")
    provider.analyze(
        media, segments, settings, tmp_path / "cache", Cancellation(), lambda *_: None
    )
    first = len(seen)
    assert first > 0
    provider.analyze(
        media, segments, settings, tmp_path / "cache", Cancellation(), lambda *_: None
    )
    assert len(seen) == first
    # Same duration/rows but another editorial goal must not reuse the selection.
    other = copy.deepcopy(settings)
    other.profile = "memes"
    provider.analyze(
        media, segments, other, tmp_path / "cache", Cancellation(), lambda *_: None
    )
    assert len(seen) > first


@pytest.mark.skipif(
    not os.environ.get("CLIPNADO_TEST_LLAMA")
    or not os.environ.get("CLIPNADO_TEST_QWEN"),
    reason="Set actual local Qwen paths",
)
def test_actual_qwen_short_meme_profile_and_cache(tmp_path, monkeypatch):
    tools = Tools(
        llama=os.environ["CLIPNADO_TEST_LLAMA"],
        semantic_model=os.environ["CLIPNADO_TEST_QWEN"],
        threads=2,
        acceleration="cpu",
    )
    dialogue = [
        (0, 4000, "Всем привет, подождём остальных зрителей."),
        (4000, 7000, "Я сейчас с первого раза перепрыгну эту яму, учитесь у мастера!"),
        (7000, 10000, "Смотрите, разбегаюсь и прыгаю!"),
        (10000, 13000, "Ой! Я опять упал в ту же яму!"),
        (13000, 17000, "Ты хотя бы разбег взял в другую сторону!"),
        (17000, 21000, "Ахаха, ладно, больше не буду учить вас прыгать."),
        (
            21000,
            26000,
            "А теперь реклама магазина. Ссылка в описании, скидка по промокоду.",
        ),
        (26000, 32000, "Подождите минуту, пока настраиваю микрофон."),
    ]
    segments = [Segment(*row) for row in dialogue]
    media = MediaInfo(
        "authored-meme.mp4", "actual-meme-profile", 32000, 1920, 1080, 30, True
    )
    settings = search_profile(SearchSettings(count=6, target_count=1), "quote")
    provider = LocalModelProvider(tools)
    result = provider.analyze(
        media, segments, settings, tmp_path / "cache", Cancellation(), lambda *_: None
    )
    assert result.info["complete"] and result.candidates
    assert all(
        8000 <= c.end_ms - c.start_ms <= 20000 and c.provider.startswith("local-qwen:")
        for c in result.candidates
    )
    # The promise and the streamer's own spoken reaction end at 13 s.
    # The other participant's later joke is optional, not a required boundary.
    assert any(
        4000 <= c.start_ms <= 7000 and 13000 <= c.end_ms <= 21000
        for c in result.candidates
    ), [(c.start_ms, c.end_ms) for c in result.candidates]

    def repeated(*_a, **_k):
        pytest.fail("Repeated actual Qwen inference")

    monkeypatch.setattr("clipnado.local_llm.run", repeated)
    again = provider.analyze(
        media, segments, settings, tmp_path / "cache", Cancellation(), lambda *_: None
    )
    assert [c.id for c in again.candidates] == [c.id for c in result.candidates]


def test_retired_animations_migrate_without_changing_clips_and_export_black(
    media, tools, tmp_path
):
    from clipnado.export import export_clip
    from test_export import frame

    c = Candidate(
        media.source_file_id, 1000, 4000, 80, "Юмор", "Сохранённый мем", selected=True
    )
    data = Project(media, candidates=[c]).to_dict()
    data["export"]["animation"] = "qwen"
    data["export"]["resolution"] = 720
    data["candidates"][0]["animation_plan"] = {"scene": "legacy-data-no-longer-used"}
    data["query_results"] = [dict(data["candidates"][0])]
    path = tmp_path / "old-animation.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    restored = load_project(path)
    assert (
        restored.candidates[0].id,
        restored.candidates[0].start_ms,
        restored.candidates[0].end_ms,
    ) == (c.id, 1000, 4000)
    assert any("Встроенные анимации удалены" in w for w in restored.warnings)
    assert not hasattr(restored.export, "animation") and not hasattr(
        restored.candidates[0], "animation_plan"
    )
    save_project(restored, tmp_path / "new.json")
    saved = json.loads((tmp_path / "new.json").read_text(encoding="utf-8"))
    assert (
        "animation" not in saved["export"]
        and "animation_plan" not in saved["query_results"][0]
    )
    result = export_clip(
        media,
        restored.candidates[0],
        [],
        restored.export,
        tmp_path / "out",
        tools,
        Cancellation(),
        lambda *_: None,
    )
    pixels = frame(result, tools, 0.5)
    assert max(pixels[720 * 640 * 3 :]) <= 5


def test_legacy_qwen_cache_ignores_retired_animation_plan(tmp_path):
    from clipnado.local_llm import read_candidates
    from dataclasses import asdict

    source = MediaInfo("source", "legacy-cache", 100000, 1920, 1080, 30, True)
    c = Candidate(
        source.source_file_id,
        0,
        60000,
        80,
        "Юмор",
        "Сохранённый эпизод",
        provider="local-qwen:legacy-model",
        version=5,
    )
    data = asdict(c)
    data["animation_plan"] = {"scene": "retired"}
    file = tmp_path / "window.json"
    file.write_text(json.dumps([data]), encoding="utf-8")
    result = read_candidates(
        file,
        source,
        [Segment(0, 100000, "Речь")],
        SearchSettings(),
        0,
        1,
        "legacy-model",
    )
    assert result[0].id == c.id and not hasattr(result[0], "animation_plan")
