"""Local protocol/grounding regressions. The real GGUF test is separate."""
import json
import socket
import sys
from pathlib import Path

import pytest

from clipnado.local_llm import (LocalModelProvider, decode_clips, expand_span, prepare_backend,
                                response_json, stage_model, transcript_windows, validate_tools)
from clipnado.media import Tools
from clipnado.models import MediaInfo, SearchSettings, Segment, Candidate
from clipnado.process import Cancellation, Cancelled
from clipnado.selection import choose_candidates


@pytest.fixture
def neural_tools(tmp_path):
    model = tmp_path / "модель тест.gguf"
    model.write_bytes(b"GGUF-test-model")
    return Tools(llama=sys.executable, semantic_model=str(model), acceleration="cpu")


@pytest.fixture
def dialogue():
    return [Segment(i * 15000, (i + 1) * 15000, "Реплика обсуждения номер " + str(i)) for i in range(8)]


def item(first=1, last=4):
    return {"first": first, "last": last, "hook": 20, "event": 22, "emotion": 18,
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


def test_model_ids_ground_to_real_source_and_add_continuous_context(dialogue):
    media = MediaInfo("file.mp4", "id", 120000, 1920, 1080, 30, True)
    clips = decode_clips([item(3, 3)], media, dialogue, SearchSettings(), 0, len(dialogue), "model-id")
    assert clips[0].score == 80
    assert clips[0].start_ms <= 45000 and clips[0].end_ms >= 60000
    assert 60000 <= clips[0].end_ms - clips[0].start_ms <= 90000
    assert clips[0].provider.startswith("local-qwen:")
    with pytest.raises(ValueError, match="границы"):
        decode_clips([item(-1, 100)], media, dialogue, SearchSettings(), 0, len(dialogue), "model-id")


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
