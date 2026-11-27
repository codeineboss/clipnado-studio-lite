"""Exact native-token budget, adaptive coverage, and compatible-cache regressions."""
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

import clipnado.local_llm as llm
from clipnado.media import Tools
from clipnado.models import MediaInfo, SearchSettings, Segment
from clipnado.process import Cancellation


def choice():
    return {"clip_id": 0, "hook": 20, "event": 20, "emotion": 20, "completeness": 20,
            "category": "Юмор", "reason": "Вопрос и ответ"}


@pytest.fixture
def tools(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF-test")
    return Tools(llama=sys.executable, semantic_model=str(model), acceleration="cpu")


def media_for(segments):
    return MediaInfo("source", "id", segments[-1].end_ms, 1920, 1080, 30, True)


def rows(work):
    prompt = (Path(work) / "prompt.txt").read_text(encoding="utf-8")
    return prompt.split("\n\nДОПУСТИМЫЕ КЛИПЫ")[0].splitlines()[1:]


@pytest.mark.parametrize("overflow", [False, True])
def test_native_probe_is_no_generation_and_uses_full_chat_request(tmp_path, monkeypatch, overflow):
    def engine(args, token, cwd, **kwargs):
        assert args[0] == "cpu"
        assert args[args.index("-n") + 1] == "0"
        assert args[args.index("-ngl") + 1] == "0"
        assert "--system-prompt-file" in args and "--single-turn" in args and "--no-warmup" in args
        assert "--verbose-prompt" in args and "--offline" in args
        if overflow:
            raise RuntimeError("llama-cli.exe: main: prompt is too long (8471 tokens, max 8188)")
        return "main: prompt: 'example number of tokens in prompt = 1'\nmain: number of tokens in prompt = 6900\nno generation"
    monkeypatch.setattr(llm, "run", engine)
    args = ["vulkan", "-n", "1536", "-ngl", "99", "--system-prompt-file", "system.txt", "--single-turn", "--no-warmup", "--offline"]
    assert llm.prompt_tokens(args, "cpu", Cancellation(), tmp_path) == (8471 if overflow else 6900)
    assert args[0] == "vulkan" and args[args.index("-n") + 1] == "1536"


def test_adaptive_windows_cover_all_text_and_resume_without_probes(tools, tmp_path, monkeypatch):
    segments = [Segment(i * 10000, (i + 1) * 10000, f"MARKER{i}") for i in range(120)]
    covered, generated = set(), []
    def measure(args, cpu, token, work):
        assert (Path(work) / "system.txt").stat().st_size > 0
        assert "ДОПУСТИМЫЕ КЛИПЫ" in (Path(work) / "prompt.txt").read_text(encoding="utf-8")
        return 8471 if len(rows(work)) > 30 else 2000
    def engine(args, token, cwd, **kwargs):
        assert args[args.index("-c") + 1] == "8192" and "--no-context-shift" in args
        assert len(rows(cwd)) <= 30
        text = (Path(cwd) / "prompt.txt").read_text(encoding="utf-8")
        import re
        covered.update(int(n) for n in re.findall(r"MARKER(\d+)", text))
        generated.append(1)
        return '{"clips":[]}'
    monkeypatch.setattr(llm, "prompt_tokens", measure)
    monkeypatch.setattr(llm, "run", engine)
    args = (media_for(segments), segments, SearchSettings(), tmp_path / "cache", Cancellation(), lambda *_: None)
    result = llm.LocalModelProvider(tools).analyze(*args)
    assert result.info["complete"] and result.info["adaptive_splits"] > 0
    assert covered == set(range(120))
    assert result.info["checked_windows"] == result.info["total_windows"] == len(generated)
    assert result.info["max_prompt_tokens"] == 2000
    def fail(*_a, **_kw):
        pytest.fail("Completed windows were recomputed")
    monkeypatch.setattr(llm, "prompt_tokens", fail)
    monkeypatch.setattr(llm, "run", fail)
    cached = llm.LocalModelProvider(tools).analyze(*args)
    assert cached.info["complete"] and cached.info["cached_windows"] == len(generated)


def test_runtime_overflow_splits_without_marking_vulkan_broken(tools, tmp_path, monkeypatch):
    cpu = Tools(llama=sys.executable, semantic_model=tools.semantic_model)
    gpu = tmp_path / "vulkan.exe"
    gpu.write_bytes(b"engine")
    tools.llama = str(gpu)
    segments = [Segment(i * 15000, (i + 1) * 15000, f"Реплика {i}") for i in range(8)]
    monkeypatch.setattr(llm, "prompt_tokens", lambda *_a: 1000)
    calls = []
    def engine(args, token, cwd, **kwargs):
        calls.append(args[0])
        assert args[0] == str(gpu), "Context overflow must not retry the same input on CPU"
        if len(calls) == 1:
            raise RuntimeError("main: prompt is too long (8471 tokens, max 8188)")
        return json.dumps({"clips": [choice()]})
    monkeypatch.setattr(llm, "run", engine)
    result = llm.LocalModelProvider(tools, cpu).analyze(media_for(segments), segments, SearchSettings(), tmp_path / "cache", Cancellation(), lambda *_: None)
    assert result.info["complete"] and result.info["adaptive_splits"] == 1
    assert result.info["llm_backend"] == "Vulkan" and not result.warnings


def test_reply_and_corrective_message_are_reserved_before_generation(tools, tmp_path, monkeypatch):
    segments = [Segment(i * 10000, (i + 1) * 10000, str(i)) for i in range(48)]
    # 7900 fits the CLI's raw 8188 input limit but leaves no room for its reply.
    monkeypatch.setattr(llm, "prompt_tokens", lambda args, cpu, token, work: 7900 if len(rows(work)) > 30 else 1500)
    def engine(*_a, **_k):
        return '{"clips":[]}'
    monkeypatch.setattr(llm, "run", engine)
    result = llm.LocalModelProvider(tools).analyze(media_for(segments), segments, SearchSettings(), tmp_path / "cache", Cancellation(), lambda *_: None)
    assert result.info["complete"] and result.info["adaptive_splits"] > 0
    assert result.info["max_prompt_tokens"] == 1500


def test_one_large_utterance_reduces_table_without_truncating_text(tools, tmp_path, monkeypatch):
    text = "Полный исходный текст " * 40
    segments = [Segment(0, 300000, text)]
    def measure(args, cpu, token, work):
        assert text in (Path(work) / "prompt.txt").read_text(encoding="utf-8")
        schema = json.loads((Path(work) / "schema.json").read_text(encoding="utf-8"))
        n = len(schema["properties"]["clips"]["items"]["properties"]["clip_id"]["enum"])
        return 8471 if n > 8 else 1000
    monkeypatch.setattr(llm, "prompt_tokens", measure)
    monkeypatch.setattr(llm, "run", lambda *_a, **_k: json.dumps({"clips": [choice()]}))
    result = llm.LocalModelProvider(tools).analyze(media_for(segments), segments, SearchSettings(), tmp_path / "cache", Cancellation(), lambda *_: None)
    assert result.info["complete"] and result.candidates and result.info["adaptive_splits"] == 0


def test_unfittable_utterance_does_not_abort_other_windows(tools, tmp_path, monkeypatch):
    segments = [Segment(0, 180000, "Ж?" * 10000), Segment(180000, 300000, "Обычный диалог")]
    monkeypatch.setattr(llm, "prompt_tokens", lambda args, cpu, token, work: 20000 if "Ж?" in (Path(work) / "prompt.txt").read_text(encoding="utf-8") else 1000)
    monkeypatch.setattr(llm, "run", lambda *_a, **_k: json.dumps({"clips": [choice()]}))
    result = llm.LocalModelProvider(tools).analyze(media_for(segments), segments, SearchSettings(), tmp_path / "cache", Cancellation(), lambda *_: None)
    assert not result.info["complete"] and result.info["failed_windows"] == 1
    assert result.candidates and all(c.start_ms >= 180000 for c in result.candidates)
    assert result.info["checked_windows"] + result.info["failed_windows"] == result.info["total_windows"]


@pytest.mark.parametrize("empty", [False, True])
def test_legacy_selection_cache_recomputed_for_hook_criteria(tools, tmp_path, monkeypatch, empty):
    segments = [Segment(i * 10000, (i + 1) * 10000, str(i)) for i in range(12)]
    media, settings, token = media_for(segments), SearchSettings(), Cancellation()
    model_id = llm.model_identity(Path(tools.semantic_model), token)
    common = {"source": media.source_file_id, "model": model_id, "segments": [asdict(s) for s in segments], "settings": {k: v for k, v in asdict(settings).items() if k not in ("query", "profile")}, "version": 3}
    identity = hashlib.sha256(json.dumps(common, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    folder = tmp_path / "cache" / "local-qwen" / identity
    folder.mkdir(parents=True)
    request = llm.build_request(media, segments, settings, 0, len(segments), 1)
    candidates = [] if empty else llm.decode_choices([choice()], media, segments, settings, request.proposals, model_id)
    (folder / "window-0.json").write_text(json.dumps([asdict(c) for c in candidates]), encoding="utf-8")
    original = (folder / "window-0.json").read_bytes()
    calls = []
    monkeypatch.setattr(llm, "prompt_tokens", lambda *_a, **_k: 1000)
    def generate(*_a, **_k):
        calls.append(True)
        return json.dumps({'clips':[choice()]})
    monkeypatch.setattr(llm, "run", generate)
    result = llm.LocalModelProvider(tools).analyze(media, segments, settings, tmp_path / "cache", token, lambda *_: None)
    assert result.info["complete"] and result.info["cached_windows"] == 0 and len(calls)==1
    assert result.candidates and (folder / 'window-0.json').read_bytes()==original
    again = llm.LocalModelProvider(tools).analyze(media, segments, settings, tmp_path / "cache", token, lambda *_: None)
    assert again.candidates==result.candidates and len(calls)==1


def test_split_ranges_terminate_and_cover_every_source_row():
    for segments in ([Segment(i * 10000, (i + 1) * 10000, str(i)) for i in range(48)],
                     [Segment(0, 300000, str(i)) for i in range(32)]):
        todo, seen, leaves = [(0, len(segments))], set(), []
        while todo:
            left, right = todo.pop()
            if (left, right) in seen:
                continue
            seen.add((left, right))
            if right - left <= 4:
                leaves.append((left, right))
                continue
            children = llm.split_window(segments, SearchSettings(), left, right)
            assert children[0][0] == left and children[1][1] == right
            assert children[1][0] <= children[0][1]
            assert all(0 < b - a < right - left for a, b in children)
            todo.extend(children)
        assert {i for a, b in leaves for i in range(a, b)} == set(range(len(segments)))
