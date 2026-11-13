"""Local protocol/grounding regressions. The real GGUF test is separate."""
import json
import socket
import sys
from pathlib import Path

import pytest

from clipnado.local_llm import (LocalModelProvider, clip_proposals, decode_choices, prepare_backend,
                                response_json, stage_model, transcript_windows, validate_tools)
from clipnado.media import Tools
from clipnado.models import MediaInfo, SearchSettings, Segment, Candidate
from clipnado.process import Cancellation, Cancelled
from clipnado.selection import choose_candidates


@pytest.fixture
def neural_tools(tmp_path, monkeypatch):
    monkeypatch.setattr("clipnado.local_llm.prompt_tokens", lambda *_a, **_k: 1000)
    model = tmp_path / "модель тест.gguf"
    model.write_bytes(b"GGUF-test-model")
    return Tools(llama=sys.executable, semantic_model=str(model), acceleration="cpu")


@pytest.fixture
def dialogue():
    return [Segment(i * 15000, (i + 1) * 15000, "Реплика обсуждения номер " + str(i)) for i in range(8)]


def item(clip_id=0):
    return {"clip_id": clip_id, "hook": 20, "event": 22, "emotion": 18,
            "completeness": 20, "category": "Юмор", "reason": "Вопрос и неожиданный ответ учителя."}


def test_invalid_model_rejected_before_recognition(neural_tools, tmp_path):
    neural_tools.semantic_model = str(tmp_path / "missing.gguf")
    with pytest.raises(ValueError, match="не найдена"):
        validate_tools(neural_tools)
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"not a model")
    neural_tools.semantic_model = str(bad)
    with pytest.raises(ValueError, match="GGUF"):
        validate_tools(neural_tools)


@pytest.mark.parametrize("free,layers", [(5500, 99), (3500, 24), (2200, 12), (1200, 0)])
def test_vulkan_memory_budget_and_speech_backend(neural_tools, tmp_path, monkeypatch, free, layers):
    for field in ("llama_vulkan", "whisper_vulkan"):
        path = tmp_path / field
        path.write_bytes(b"engine")
        setattr(neural_tools, field, str(path))
    neural_tools.acceleration = "auto"
    monkeypatch.setattr("clipnado.local_llm.run", lambda *_a, **_k: f"Available devices:\n  Vulkan0: NVIDIA GTX 1660 (6144 MiB, {free} MiB free)")
    active, warnings = prepare_backend(neural_tools, Cancellation(), lambda *_: None)
    assert active.llm_gpu_layers == layers
    assert (active.llama == neural_tools.llama_vulkan) == bool(layers)
    assert active.gpu and active.whisper == neural_tools.whisper_vulkan


def test_vulkan_failure_uses_cpu(neural_tools, tmp_path, monkeypatch):
    file = tmp_path / "vulkan.exe"
    file.write_bytes(b"engine")
    neural_tools.acceleration, neural_tools.llama_vulkan = "auto", str(file)
    def broken(*_a, **_kw):
        raise RuntimeError("missing Vulkan loader")
    monkeypatch.setattr("clipnado.local_llm.run", broken)
    active, warnings = prepare_backend(neural_tools, Cancellation(), lambda *_: None)
    assert active.llama == neural_tools.llama and not active.gpu and warnings


def test_windows_cover_entire_long_transcript():
    segments = [Segment(i, i + 10000, "Обычная реплика для проверки покрытия всей временной шкалы.") for i in range(0, 1800000, 10000)]
    windows = transcript_windows(segments, SearchSettings())
    assert windows[0][0] == 0 and windows[-1][1] == len(segments)
    for a, b in zip(windows, windows[1:]):
        assert b[0] < a[1]
        assert segments[a[1] - 1].end_ms - segments[b[0]].start_ms >= 90000


def test_decoder_uses_generated_json_after_log_or_prompt_echo():
    assert response_json('example: {"clips":[{}]}\nresult: {"clips":[]}\nperformance stats') == []
    with pytest.raises(ValueError, match="JSON"):
        response_json('{"clips":[{"first":')


def test_model_choice_uses_only_offered_bounds_and_preserves_scores(dialogue):
    media = MediaInfo("file.mp4", "id", 120000, 1920, 1080, 30, True)
    proposals = clip_proposals(media, dialogue, SearchSettings(), 0, len(dialogue))
    clips = decode_choices([item(), item()], media, dialogue, SearchSettings(), proposals, "model-id")
    assert len(clips) == 1  # repeated IDs never fill the clip count
    assert clips[0].score == 80
    assert (clips[0].start_ms, clips[0].end_ms) == (proposals[0].start_ms, proposals[0].end_ms)
    assert 60000 <= clips[0].end_ms - clips[0].start_ms <= 90000
    with pytest.raises(ValueError, match="Номер"):
        decode_choices([item(999)], media, dialogue, SearchSettings(), proposals, "model-id")


@pytest.mark.parametrize("segments", [
    [Segment(0, 50000, "Первая длинная реплика"), Segment(50000, 100000, "Вторая длинная реплика")],
    [Segment(0, 180000, "Длинная реплика из импортированного SRT")],
    [Segment(0, 45000, "Вопрос"), Segment(65000, 110000, "Ответ после паузы")],
])
def test_coarse_segments_and_pauses_offer_valid_clips_instead_of_rejecting_everything(segments):
    # No whole-utterance combination in these inputs is 60–90 seconds.
    media = MediaInfo("source", "id", segments[-1].end_ms, 1920, 1080, 30, True)
    settings = SearchSettings()
    proposals = clip_proposals(media, segments, settings, 0, len(segments))
    assert proposals and len(proposals) <= 80
    clips = decode_choices([item(i) for i in range(len(proposals))], media, segments, settings, proposals, "model")
    assert all(0 <= c.start_ms < c.end_ms <= media.duration_ms and 60000 <= c.end_ms - c.start_ms <= 90000 for c in clips)
    assert any("внутри реплики" in c.reason for c in clips)


def test_late_window_uses_local_choices_and_real_absolute_source_times():
    segments = [Segment(i * 10000, (i + 1) * 10000, f"Реплика {i}") for i in range(240)]
    media = MediaInfo("source", "id", 2400000, 1920, 1080, 30, True)
    spans = clip_proposals(media, segments, SearchSettings(), 160, 208)
    clips = decode_choices([item(0)], media, segments, SearchSettings(), spans, "model")
    assert len(spans) <= 80 and clips[0].start_ms >= 1600000 and clips[0].end_ms <= 2080000
    assert "Реплика 160" in clips[0].transcript_excerpt


def test_model_staging_cross_volume_copy_and_cancel(neural_tools, tmp_path, monkeypatch):
    def different_volume(*_):
        raise OSError("different volume")
    monkeypatch.setattr("clipnado.local_llm.os.link", different_volume)
    source = Path(neural_tools.semantic_model)
    staged = stage_model(source, tmp_path / "Кэш", "model-id", Cancellation())
    assert staged.name == "model.gguf" and staged.read_bytes() == source.read_bytes()
    assert stage_model(source, tmp_path / "Кэш", "model-id", Cancellation()) == staged
    token = Cancellation()
    token.cancel()
    with pytest.raises(Cancelled):
        stage_model(source, tmp_path / "other", "another-id", token)


def test_full_local_analysis_cache_and_no_network(neural_tools, dialogue, tmp_path, monkeypatch):
    calls = []
    def network(*_a, **_kw):
        pytest.fail("Local analysis attempted a network connection")
    monkeypatch.setattr(socket.socket, "connect", network)
    monkeypatch.setattr(socket, "create_connection", network)
    def engine(args, token, cwd, **kwargs):
        calls.append(args)
        assert "--offline" in args and "--single-turn" in args
        assert not any("http" in value for value in args)
        model = Path(cwd) / args[args.index("-m") + 1]
        assert model.read_bytes().startswith(b"GGUF")
        return json.dumps({"clips": [item()]}, ensure_ascii=False)
    monkeypatch.setattr("clipnado.local_llm.run", engine)
    media = MediaInfo("source.mp4", "id", 120000, 1920, 1080, 30, True)
    p = LocalModelProvider(neural_tools)
    first = p.analyze(media, dialogue, SearchSettings(), tmp_path / "cache", Cancellation(), lambda *_: None)
    second = p.analyze(media, dialogue, SearchSettings(), tmp_path / "cache", Cancellation(), lambda *_: None)
    assert len(calls) == 1
    assert first.info["complete"] and len(first.candidates) == 1
    assert first.candidates[0].id == second.candidates[0].id


def test_neural_dedup_keeps_different_events_with_similar_vocabulary():
    common = "Учитель ученик урок ответ вопрос тетрадь задание школа объяснение реплика пример обсуждение учебник класс доска. "
    clips = [Candidate("id", i * 60000, (i + 1) * 60000, 80, "Юмор", "Событие", common + end,
                       provider="local-qwen:fixture") for i, end in enumerate(["Лягушка выпрыгнула на стол.", "Кот включил микрофон конференции."])]
    assert len(choose_candidates(clips, SearchSettings(), Cancellation())) == 2
    clips[1].transcript_excerpt = clips[0].transcript_excerpt
    assert len(choose_candidates(clips, SearchSettings(), Cancellation())) == 1


def test_cancellation_does_not_mark_window_complete(neural_tools, dialogue, tmp_path, monkeypatch):
    def cancel(*_a, **_kw):
        raise Cancelled("cancelled")
    monkeypatch.setattr("clipnado.local_llm.run", cancel)
    with pytest.raises(Cancelled):
        LocalModelProvider(neural_tools).analyze(MediaInfo("source", "id", 120000, 1920, 1080, 30, True), dialogue,
                                               SearchSettings(), tmp_path / "cache", Cancellation(), lambda *_: None)
    assert not list((tmp_path / "cache").rglob("window-*.json"))


def test_invalid_reply_is_retried_with_constrained_ids(neural_tools, dialogue, tmp_path, monkeypatch):
    calls = []
    def engine(args, token, cwd, **kwargs):
        schema = json.loads((Path(cwd) / "schema.json").read_text(encoding="utf-8"))
        fields = schema["properties"]["clips"]["items"]["properties"]
        assert fields["clip_id"]["enum"] and "first" not in fields and "last" not in fields
        calls.append((Path(cwd) / "prompt.txt").read_text(encoding="utf-8"))
        if len(calls) == 1:
            return json.dumps({"clips": [item(9999)]})
        assert "Предыдущий ответ отклонён" in calls[-1]
        return json.dumps({"clips": [item()]})
    monkeypatch.setattr("clipnado.local_llm.run", engine)
    result = LocalModelProvider(neural_tools).analyze(MediaInfo("source", "id", 120000, 1920, 1080, 30, True), dialogue,
                                                    SearchSettings(), tmp_path / "cache", Cancellation(), lambda *_: None)
    assert len(calls) == 2 and result.info["complete"] and result.candidates


def test_bad_window_preserves_other_results_and_resumes_only_missing_window(neural_tools, tmp_path, monkeypatch):
    segments = [Segment(i, i + 10000, "Речь из длинного видео " + str(i)) for i in range(0, 900000, 10000)]
    calls = []
    def engine(args, token, cwd, **kwargs):
        calls.append(args)
        return '{"clips":[' if len(calls) <= 2 else json.dumps({"clips": [item()]})
    monkeypatch.setattr("clipnado.local_llm.run", engine)
    media = MediaInfo("source", "id", 900000, 1920, 1080, 30, True)
    cache = tmp_path / "cache"
    result = LocalModelProvider(neural_tools).analyze(media, segments, SearchSettings(), cache, Cancellation(), lambda *_: None)
    assert not result.info["complete"] and result.info["failed_windows"] == 1
    assert result.info["checked_windows"] == result.info["total_windows"] - 1
    assert result.candidates and result.warnings
    assert not [p for p in cache.rglob("window-0-*.json") if not p.name.endswith(".failed.json")] and list(cache.rglob("window-0-*.failed.json"))
    saved_ids = {c.id for c in result.candidates}
    before = len(calls)
    resumed = LocalModelProvider(neural_tools).analyze(media, segments, SearchSettings(), cache, Cancellation(), lambda *_: None)
    assert len(calls) == before + 1 and resumed.info["complete"]
    assert saved_ids <= {c.id for c in resumed.candidates}
    assert not list(cache.rglob("*.failed.json"))


@pytest.mark.parametrize("answer", [[], [{**item(), "hook": 5, "event": 5, "emotion": 5, "completeness": 5}]])
def test_empty_or_low_score_choices_are_not_retried_or_inflated(neural_tools, dialogue, tmp_path, monkeypatch, answer):
    calls = []
    def engine(*_a, **_kw):
        calls.append(1)
        return json.dumps({"clips": answer})
    monkeypatch.setattr("clipnado.local_llm.run", engine)
    p = LocalModelProvider(neural_tools)
    args = (MediaInfo("source", "id", 120000, 1920, 1080, 30, True), dialogue, SearchSettings(), tmp_path / "cache", Cancellation(), lambda *_: None)
    for _ in range(2):
        result = p.analyze(*args)
        assert result.info["complete"] and result.candidates == []
    assert len(calls) == 1


def test_no_feasible_length_is_reported_without_calling_model(neural_tools, tmp_path, monkeypatch):
    monkeypatch.setattr("clipnado.local_llm.run", lambda *_a, **_k: pytest.fail("Impossible clip sent to model"))
    result = LocalModelProvider(neural_tools).analyze(MediaInfo("source", "id", 10000, 1920, 1080, 30, True),
        [Segment(0, 10000, "Короткая речь")], SearchSettings(), tmp_path / "cache", Cancellation(), lambda *_: None)
    assert not result.info["complete"] and not result.candidates and result.warnings
