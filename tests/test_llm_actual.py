"""Actual GGUF inference; enable with local llama-cli and Qwen model paths.

This checks one authored example, not precision/recall on entertainment videos.
"""
import json
import os
from pathlib import Path

import pytest

from clipnado.local_llm import LocalModelProvider
from clipnado.media import Tools
from clipnado.models import MediaInfo, SearchSettings, Segment
from clipnado.process import Cancellation


def actual_selection(tmp_path, monkeypatch, coarse=False):
    fixture = Path(__file__).resolve().parent.parent / "qa" / "qwen-dialogue-fixture.json"
    data = json.loads(fixture.read_text(encoding="utf-8"))
    segments = [Segment(**s) for s in data["segments"]]
    if coarse:
        segments = [Segment(0, 50000, " ".join(s.text for s in segments[:3])),
                    Segment(50000, 100000, " ".join(s.text for s in segments[3:]))]
    media = MediaInfo("authored-dialogue.mp4", "actual-qwen-test", data["duration_ms"], 1920, 1080, 30, True)
    tools = Tools(llama=os.environ["CLIPNADO_TEST_LLAMA"], semantic_model=os.environ["CLIPNADO_TEST_QWEN"],
                  threads=2, acceleration="cpu")
    provider = LocalModelProvider(tools)
    cache = tmp_path / "Кэш с пробелами"
    result = provider.analyze(media, segments, SearchSettings(), cache, Cancellation(), lambda *_: None)
    assert result.info["complete"]
    assert result.candidates, "Actual model failed to select the authored humorous dialogue"
    assert all(60000 <= c.end_ms - c.start_ms <= 90000 for c in result.candidates)
    if coarse:
        assert any("внутри реплики" in c.reason for c in result.candidates)
    else:
        assert any(c.start_ms <= 20000 and c.end_ms >= 80000 for c in result.candidates)
    assert all(c.provider.startswith("local-qwen:") and c.score >= 60 for c in result.candidates)
    def repeated_inference(*_args, **_kwargs):
        pytest.fail("Repeated actual inference")
    monkeypatch.setattr("clipnado.local_llm.run", repeated_inference)
    again = provider.analyze(media, segments, SearchSettings(), cache, Cancellation(), lambda *_: None)
    assert [c.id for c in again.candidates] == [c.id for c in result.candidates]


@pytest.mark.skipif(not os.environ.get("CLIPNADO_TEST_LLAMA") or not os.environ.get("CLIPNADO_TEST_QWEN"),
                    reason="Set actual local llama-cli and Qwen GGUF paths")
def test_actual_qwen_semantic_selection_and_cache(tmp_path, monkeypatch):
    actual_selection(tmp_path, monkeypatch)


@pytest.mark.skipif(not os.environ.get("CLIPNADO_TEST_LLAMA") or not os.environ.get("CLIPNADO_TEST_QWEN"),
                    reason="Set actual local llama-cli and Qwen GGUF paths")
def test_actual_qwen_coarse_segments_and_cache(tmp_path, monkeypatch):
    actual_selection(tmp_path, monkeypatch, coarse=True)


@pytest.mark.skipif(not os.environ.get("CLIPNADO_TEST_LLAMA") or not os.environ.get("CLIPNADO_TEST_QWEN"),
                    reason="Set actual local llama-cli and Qwen GGUF paths")
def test_actual_qwen_context_budget_without_generation(tmp_path, monkeypatch):
    """Real tokenizer/ChatML, oversized complete requests, no dense-text inference.

    Long input is synthetic symbol-rich text. Only the no-generation probes run
    natively; successful generation is a fixture so this test isolates budgeting.
    Actual semantic generation is exercised by the two dialogue tests above.
    """
    import re
    import clipnado.local_llm as llm
    segments = [Segment(i * 10000, (i + 1) * 10000, ("Ж7!q9/Ё5?z2=Б3;w8:" * 7) + f" MARKER{i}") for i in range(80)]
    media = MediaInfo("dense-authored-transcript.mp4", "actual-context-test", 800000, 1920, 1080, 30, True)
    tools = Tools(llama=os.environ["CLIPNADO_TEST_LLAMA"], semantic_model=os.environ["CLIPNADO_TEST_QWEN"], threads=2, acceleration="cpu")
    real_run = llm.run
    observed, accepted, covered, retry_checked = [], [], set(), []
    def tokenize_only(args, token, cwd, **kwargs):
        if args[args.index("-n") + 1] == "0":
            try:
                output = real_run(args, token, cwd=cwd, **kwargs)
            except RuntimeError as exc:
                count = llm.context_overflow(exc)
                if count:
                    observed.append(count[0])
                raise
            count = int(re.findall(r"^main:\s*number of tokens in prompt\s*=\s*(\d+)\s*$", output, re.M)[-1])
            observed.append(count)
            assert re.search(r"prompt eval time\s*=\s*0\.00 ms", output), "Probe evaluated the dense prompt"
            return output
        assert args[args.index("-c") + 1] == "8192" and "--no-context-shift" in args
        assert observed[-1] + int(args[args.index("-n") + 1]) + llm.RETRY_RESERVE <= llm.CONTEXT_TOKENS
        accepted.append(observed[-1])
        prompt_file = Path(cwd) / "prompt.txt"
        text = prompt_file.read_text(encoding="utf-8")
        if not retry_checked:
            # Verify the reserved corrective message with the actual tokenizer,
            # not an estimated characters-to-tokens ratio.
            probe = list(args)
            probe[probe.index("-n") + 1] = "0"
            probe[probe.index("-ngl") + 1] = "0"
            probe.append("--verbose-prompt")
            try:
                prompt_file.write_text(text + llm.RETRY_HINT, encoding="utf-8")
                output = real_run(probe, token, cwd=cwd, **kwargs)
                retry_count = int(re.findall(r"^main:\s*number of tokens in prompt\s*=\s*(\d+)\s*$", output, re.M)[-1])
                assert retry_count + int(args[args.index("-n") + 1]) + 68 <= llm.CONTEXT_TOKENS
                retry_checked.append(retry_count)
            finally:
                prompt_file.write_text(text, encoding="utf-8")
        covered.update(int(n) for n in re.findall(r"MARKER(\d+)", text))
        return '{"clips":[]}'  # generation fixture, NOT model-quality evidence
    monkeypatch.setattr(llm, "run", tokenize_only)
    result = llm.LocalModelProvider(tools).analyze(media, segments, SearchSettings(), tmp_path / "Кэш плотного текста", Cancellation(), lambda *_: None)
    assert max(observed) > 8188, "Fixture no longer reproduces a raw input overflow"
    assert accepted and retry_checked and result.info["adaptive_splits"] > 0 and result.info["complete"]
    assert covered == set(range(len(segments))), "Adaptive splitting lost transcript rows"
    assert result.info["context_tokens"] == 8192
