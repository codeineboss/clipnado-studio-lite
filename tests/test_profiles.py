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
        60,
        90 if code in ("story", "mixed", "highlights") else 60,
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
    assert (p.instruction in request.system) == (code != 'highlights')
    assert request.proposals
    assert all(
        result.min_seconds * 1000 <= x.end_ms - x.start_ms <= 90000
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
    assert result.normalize_audio
    assert result.preset == code
    assert (
        result.gaming_camera == before.gaming_camera
        and result.gaming_game == before.gaming_game
    )
    assert before.video_y == 0.6
    if code == "gaming":
        assert result.gaming_fraction == 0.35
    elif code == "dialogue":
        assert result.layout == "blur_center" and result.background_dim == 0.4
    else:
        assert result.layout == "top_black" and result.video_y == 0
    assert export_profile(before, code, source, False) == result


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
    settings = search_profile(SearchSettings(min_seconds=30, max_seconds=90), "quote")
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
def test_actual_qwen_mixed_short_meme_and_long_story_and_cache(tmp_path, monkeypatch):
    tools = Tools(
        llama=os.environ["CLIPNADO_TEST_LLAMA"],
        semantic_model=os.environ["CLIPNADO_TEST_QWEN"],
        threads=2,
        acceleration="cpu",
    )
    dialogue = [
        (0, 6000, "Всем привет, начинаем стрим. Пока подождём зрителей."),
        (
            6000,
            12000,
            "Сейчас покажу идеальный прыжок. Я профессионал, здесь невозможно упасть!",
        ),
        (
            12000,
            18000,
            "Разбегаюсь, прыгаю... Ааа, я снова упал в яму! Только что всех учил и сам всё провалил!",
        ),
        (
            18000,
            24000,
            "Ахаха, спасибо, профессионал! Теперь мы точно знаем, как делать не надо. Всё, больше никого не учу!",
        ),
        (
            24000,
            30000,
            "Рекламная пауза: магазин, ссылка в описании и скидка по промокоду.",
        ),
        (
            30000,
            36000,
            "Вспомнил одну историю. Как я случайно стал ведущим чужой свадьбы. Без начала тут вообще непонятно.",
        ),
        (
            36000,
            42000,
            "Друг попросил забрать его костюм из химчистки и довезти до ресторана. Я согласился, адрес он прислал.",
        ),
        (
            42000,
            48000,
            "На входе стояла администратор. Увидела мужчину с костюмом и сразу сказала: наконец-то, мы вас ждали.",
        ),
        (
            48000,
            54000,
            "Я решил, что это про доставку. Она провела меня в зал, дала микрофон и убежала встречать гостей.",
        ),
        (
            54000,
            60000,
            "Я хотел объяснить, но в этот момент заиграла музыка и весь зал начал хлопать. Передо мной стояли молодожёны.",
        ),
        (
            60000,
            66000,
            "Жених сказал: наш ведущий опоздал, хорошо что прислали замену. А я даже не знал имён жениха и невесты.",
        ),
        (
            66000,
            72000,
            "Гости ждали тост. Я растерялся и сказал: главное в браке вовремя возвращать чужие вещи. Все засмеялись.",
        ),
        (
            72000,
            78000,
            "Меня попросили провести конкурс. Я объявил соревнование: кто быстрее найдёт настоящий адрес ресторана.",
        ),
        (
            78000,
            84000,
            "Тут позвонил мой друг: ты где? Оказалось, я перепутал корпус, его ресторан был в соседнем здании.",
        ),
        (
            84000,
            90000,
            "Именно тогда в зал зашёл настоящий ведущий. Я думал, теперь меня выгонят, и приготовился объясняться.",
        ),
        (
            90000,
            96000,
            "Но невеста сказала: этого оставляем, он смешнее. Мне заплатили за вечер, а костюм я всё-таки доставил.",
        ),
        (
            96000,
            102000,
            "Самое смешное: через год эта пара позвала меня на годовщину. Так доставка костюма превратилась в подработку.",
        ),
        (
            102000,
            108000,
            "История закончилась. А теперь снова реклама магазина, скидки и промокоды, ссылка в описании.",
        ),
    ]
    segments = [Segment(*row) for row in dialogue]
    media = MediaInfo(
        "authored-mixed.mp4", "actual-mixed-profile", 108000, 1920, 1080, 30, True
    )
    settings = search_profile(
        SearchSettings(count=6, target_count=2, min_seconds=15, max_seconds=90),
        "mixed",
    )
    provider = LocalModelProvider(tools)
    result = provider.analyze(
        media, segments, settings, tmp_path / "cache", Cancellation(), lambda *_: None
    )
    assert result.info["complete"] and result.candidates
    assert (
        result.info["max_prompt_tokens"] <= 3200
    ), "Repeated timecodes overloaded the mixed request"
    assert all(
        15000 <= c.end_ms - c.start_ms <= 90000 and c.provider.startswith("local-qwen:")
        for c in result.candidates
    )
    intervals = [(c.start_ms, c.end_ms, c.category) for c in result.candidates]
    assert any(
        6000 <= c.start_ms <= 12000 and 18000 <= c.end_ms <= 24000
        for c in result.candidates
    ), intervals
    # A funny self-contained story may also be labelled humor/reaction at <=60s.
    # Its opening situation and resolution are the objective content requirement.
    # Starting with the mistaken greeting at 42s is also a coherent hook; the
    # initial delivery explanation may be revealed later in the story.
    assert any(
        30000 <= c.start_ms <= 42000 and 96000 <= c.end_ms <= 102000
        for c in result.candidates
    ), intervals

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


def test_story_profile_keeps_context_above_regular_limit():
    from clipnado.local_llm import decode_choices

    source = MediaInfo("story", "story", 96000, 1920, 1080, 30, True)
    segments = [
        Segment(i * 6000, (i + 1) * 6000, "Продолжение одной истории номер " + str(i))
        for i in range(16)
    ]
    regular = search_profile(SearchSettings(min_seconds=15), "memes")
    story = search_profile(SearchSettings(min_seconds=15), "story")
    a = build_request(source, segments, regular, 0, len(segments), 1)
    b = build_request(source, segments, story, 0, len(segments), 1)
    assert all(15000 <= p.end_ms - p.start_ms <= 60000 for p in a.proposals)
    long = next(i for i, p in enumerate(b.proposals) if p.end_ms - p.start_ms > 60000)
    item = dict(
        clip_id=long,
        hook=20,
        event=20,
        emotion=20,
        completeness=20,
        category="История",
        reason="Для итога нужен контекст",
    )
    chosen = decode_choices([item], source, segments, story, b.proposals, "model")
    assert 60000 < chosen[0].end_ms - chosen[0].start_ms <= 90000
    with pytest.raises(ValueError):
        decode_choices([item], source, segments, regular, b.proposals, "model")


@pytest.mark.parametrize("mode", ["srt", "burn"])
def test_retired_subtitles_migrate_and_export_clean_video(media, tools, tmp_path, mode):
    from clipnado.export import export_clip
    from test_export import frame

    c = Candidate(
        media.source_file_id, 1000, 4000, 85, "Юмор", "Сохранённый клип", selected=True
    )
    segments = [Segment(0, 8000, "Этот текст нужен только для смыслового поиска.")]
    data = Project(
        media, segments=segments, candidates=[c], export=ExportSettings(resolution=720)
    ).to_dict()
    data["export"].update(subtitles=mode, subtitle_y=0.31, readable_subtitles=True)
    path = tmp_path / "old-captions.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    restored = load_project(path)
    assert restored.segments == segments and restored.candidates == [c]
    save_project(restored, tmp_path / "new.json")
    saved = json.loads((tmp_path / "new.json").read_text(encoding="utf-8"))
    assert (
        not {"subtitles", "subtitle_y", "readable_subtitles"} & saved["export"].keys()
    )
    out = export_clip(
        media,
        c,
        restored.segments,
        restored.export,
        tmp_path / "out",
        tools,
        Cancellation(),
        lambda *_: None,
    )
    assert not list(out.parent.glob("*.srt")) and not list(out.parent.glob("*.ass"))
    image = frame(out, tools, 0.3)
    assert (
        sum(min(r, g, b) > 180 for r, g, b in zip(image[::3], image[1::3], image[2::3]))
        == 0
    )


def test_mixed_request_offers_short_events_and_long_stories_together():
    from clipnado.local_llm import decode_choices

    source = MediaInfo("mixed", "mixed", 180000, 1920, 1080, 30, True)
    segments = [
        Segment(i * 6000, (i + 1) * 6000, "Реплика номер " + str(i)) for i in range(30)
    ]
    settings = search_profile(SearchSettings(min_seconds=15), "mixed")
    request = build_request(source, segments, settings, 0, len(segments), 1)
    short = next(
        i
        for i, p in enumerate(request.proposals)
        if 15000 <= p.end_ms - p.start_ms <= 30000
    )
    long = next(
        i
        for i, p in enumerate(request.proposals)
        if 60000 < p.end_ms - p.start_ms <= 90000
    )
    item = dict(hook=20, event=20, emotion=20, completeness=20, reason="Цельный эпизод")
    # The category shown in the existing results table also enforces the
    # exception for longer stories. A joke cannot silently become a 90s clip.
    assert not decode_choices(
        [dict(item, clip_id=long, category="Юмор")],
        source,
        segments,
        settings,
        request.proposals,
        "model",
    )
    accepted = decode_choices(
        [
            dict(item, clip_id=short, category="Юмор"),
            dict(item, clip_id=long, category="История"),
        ],
        source,
        segments,
        settings,
        request.proposals,
        "model",
    )
    assert [c.category for c in accepted] == ["Юмор", "История"]
    assert "длительность" in request.prompt
    assert all(s.text in request.prompt for s in segments)
    item_schema = request.schema["properties"]["clips"]["items"]
    branches = item_schema.get("anyOf", [item_schema])
    assert set.union(
        *(set(b["properties"]["category"]["enum"]) for b in branches)
    ) == set(settings.categories)


def test_mixed_selection_keeps_stronger_story_instead_of_filling_quota():
    from clipnado.selection import choose_candidates

    settings = search_profile(SearchSettings(min_seconds=15, target_count=3), "mixed")
    story = Candidate(
        "mixed", 0, 78000, 95, "История", "Завязка и неожиданный итог", "Полная история"
    )
    shorts = [
        Candidate("mixed", a, a + 24000, 65, "Реакция", "Часть истории", str(a))
        for a in (0, 30000, 54000)
    ]
    meme = Candidate(
        "mixed", 90000, 114000, 85, "Юмор", "Отдельная шутка", "Другой момент"
    )
    result = choose_candidates([*shorts, story, meme], settings, Cancellation())
    assert [c.id for c in result] == [story.id, meme.id]


def test_mixed_neighbor_request_keeps_absolute_anchor_reference(tmp_path, monkeypatch):
    import sys
    from clipnado.assist import LocalClipAssistant

    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF-test")
    tools = Tools(llama=sys.executable, semantic_model=str(model), acceleration="cpu")
    source = MediaInfo("source", "mixed-neighbor", 120000, 1920, 1080, 30, True)
    segments = [
        Segment(i * 6000, (i + 1) * 6000, "Часть истории номер " + str(i))
        for i in range(20)
    ]
    anchor = Candidate(
        source.source_file_id, 24000, 90000, 85, "История", "Текущий рассказ"
    )
    settings = search_profile(SearchSettings(min_seconds=15), "mixed")

    def window(self, request, staged, token, progress, stats, decode):
        assert (
            "00:00:24.000" in request.prompt and "00:01:30.000" in request.prompt
        ), "The anchor in the instruction cannot be mapped to the proposal list"
        current = next(
            i
            for i, p in enumerate(request.proposals)
            if (p.start_ms, p.end_ms) == (anchor.start_ms, anchor.end_ms)
        )
        item = dict(
            clip_id=current,
            hook=20,
            event=20,
            emotion=20,
            completeness=20,
            category="История",
            reason="Сохранить текущие границы",
        )
        return decode([item]), 1000, [], None

    monkeypatch.setattr(LocalClipAssistant, "_window", window)
    result = LocalClipAssistant(tools).refine(
        source,
        segments,
        anchor,
        settings,
        tmp_path / "cache",
        Cancellation(),
        lambda *_: None,
    )
    assert result.info["complete"] and result.info["current_recommended"]
    assert not result.candidates and (anchor.start_ms, anchor.end_ms) == (24000, 90000)


def test_mixed_proposals_do_not_join_events_through_an_ad_break():
    from clipnado.local_llm import clip_proposals, advertising_break

    source = MediaInfo("stream", "events", 120000, 1920, 1080, 30, True)
    segments = [
        Segment(i * 6000, (i + 1) * 6000, "Событие номер " + str(i)) for i in range(20)
    ]
    segments[4].text = (
        "Рекламная пауза: магазин, ссылка в описании, скидка по промокоду."
    )
    settings = search_profile(SearchSettings(min_seconds=15), "mixed")
    proposals = clip_proposals(source, segments, settings, 0, len(segments))
    assert proposals and all(
        p.end_ms <= 24000 or p.start_ms >= 30000 for p in proposals
    )
    assert any(p.start_ms == 6000 and p.end_ms == 24000 for p in proposals)
    assert any(p.start_ms >= 30000 and p.end_ms - p.start_ms > 60000 for p in proposals)
    assert not advertising_break(
        "Вчера мы обсуждали смешную рекламу: скидка по промокоду, ссылка в описании."
    )
    assert not advertising_break(
        "А теперь расскажу историю: я купил костюм со скидкой по промокоду."
    )
    assert not advertising_break(
        "Рекламная пауза — так мы назвали эту шутку про скидки."
    )
    # Coarse imports do not locate the end of an ad precisely enough to block
    # the entire source segment; keep those flagged intervals for model review.
    coarse = [Segment(0, 120000, segments[4].text + " Далее целая история.")]
    assert clip_proposals(source, coarse, settings, 0, 1)
    # Preserve explicit content searches, including requests about advertising.
    settings.query = "рекламная пауза"
    assert any(
        p.start_ms < 24000 and p.end_ms > 30000
        for p in clip_proposals(source, segments, settings, 0, len(segments))
    )


def test_mixed_windows_cover_independent_events_without_waiting_intro_or_ads():
    from clipnado.local_llm import transcript_windows, waiting_intro

    segments = [
        Segment(i * 6000, (i + 1) * 6000, "Содержание эпизода номер " + str(i))
        for i in range(18)
    ]
    segments[0].text = "Всем привет, начинаем стрим. Пока подождём зрителей."
    for i in (4, 17):
        segments[i].text = "Рекламная пауза: ссылка в описании, скидка по промокоду."
    settings = search_profile(SearchSettings(min_seconds=15), "mixed")
    windows = transcript_windows(segments, settings)
    assert windows == [(1, 4), (5, 17)]
    assert {i for left, right in windows for i in range(left, right)} == set(
        range(18)
    ) - {0, 4, 17}
    assert not waiting_intro(
        Segment(
            0,
            6000,
            "Всем привет! Сейчас расскажу историю о том, как вчера ждали зрителей.",
        )
    )
    assert not waiting_intro(
        Segment(0, 6000, "Привет всем! Сейчас покажу смешной момент.")
    )
    assert not waiting_intro(
        Segment(0, 180000, segments[0].text + " Далее длинный рассказ.")
    )
    settings.query = "ожидание зрителей"
    assert transcript_windows(segments, settings) == [(0, 18)]


def test_mixed_shortfall_checks_unselected_speech_once_and_reuses_cache(
    tmp_path, monkeypatch
):
    import sys

    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF-test")
    tools = Tools(llama=sys.executable, semantic_model=str(model), acceleration="cpu")
    media = MediaInfo("source", "missing-event", 150000, 1920, 1080, 30, True)
    segments = [
        Segment(
            i * 6000,
            (i + 1) * 6000,
            ("Независимая шутка " if i < 9 else "Другая полная история ") + str(i),
        )
        for i in range(25)
    ]
    settings = search_profile(SearchSettings(min_seconds=15, target_count=2), "mixed")
    calls = []

    def window(self, request, staged, token, progress, stats, decode):
        calls.append(request)
        if len(calls) == 1:
            index = next(
                i
                for i, p in enumerate(request.proposals)
                if p.start_ms >= 54000 and p.end_ms >= 138000
            )
            category = "История"
        else:
            assert len(calls) == 2, "Recovery search must not recurse"
            assert all(
                p.end_ms <= 60000 for p in request.proposals
            ), "Previously selected story was rescanned"
            index = next(
                i
                for i, p in enumerate(request.proposals)
                if p.end_ms - p.start_ms <= 30000
            )
            category = "Юмор"
        item = dict(
            clip_id=index,
            hook=20,
            event=20,
            emotion=20,
            completeness=20,
            category=category,
            reason="Цельный самостоятельный эпизод",
        )
        return decode([item]), 1000, [], None

    monkeypatch.setattr(LocalModelProvider, "_window", window)
    provider = LocalModelProvider(tools)
    result = provider.analyze(
        media, segments, settings, tmp_path / "cache", Cancellation(), lambda *_: None
    )
    assert len(calls) == 2 and len(result.candidates) == 2 and result.info["complete"]
    assert result.info["recovery_windows"] == 1 and result.info["checked_windows"] == 2
    assert {c.category for c in result.candidates} == {"Юмор", "История"}
    assert all(c.score == 80 for c in result.candidates)
    again = provider.analyze(
        media, segments, settings, tmp_path / "cache", Cancellation(), lambda *_: None
    )
    assert len(calls) == 2 and again.info["cached_windows"] == 2
    assert [c.id for c in again.candidates] == [c.id for c in result.candidates]


def test_mixed_span_reply_resolves_global_rows_and_rejects_unknown_pairs():
    from clipnado.local_llm import decode_choices

    media = MediaInfo("source", "span", 120000, 1920, 1080, 30, True)
    segments = [
        Segment(i * 6000, (i + 1) * 6000, "Реплика номер " + str(i)) for i in range(20)
    ]
    settings = search_profile(SearchSettings(min_seconds=15), "mixed")
    request = build_request(media, segments, settings, 5, 20, 1)
    p = request.proposals[0]
    base = dict(
        hook=20,
        event=20,
        emotion=20,
        completeness=20,
        category="Юмор",
        reason="Законченное событие",
    )
    reply = dict(base, span=f"R{p.first}-R{p.last}")
    result = decode_choices(
        [reply], media, segments, settings, request.proposals, "model"
    )
    assert (result[0].start_ms, result[0].end_ms) == (p.start_ms, p.end_ms)
    assert "R5:" in request.prompt and "clip_id" not in request.system
    assert "span" in request.retry_hint
    with pytest.raises(ValueError, match="Пара реплик"):
        decode_choices(
            [dict(base, span="R999-R1000")],
            media,
            segments,
            settings,
            request.proposals,
            "model",
        )
    item_schema = request.schema["properties"]["clips"]["items"]
    branches = item_schema.get("anyOf", [item_schema])
    for branch in branches:
        pairs = branch["properties"]["span"]["enum"]
        long = any(
            p.end_ms - p.start_ms > 60000 and f"R{p.first}-R{p.last}" in pairs
            for p in request.proposals
        )
        if long:
            assert branch["properties"]["category"]["enum"] == ["История"]
