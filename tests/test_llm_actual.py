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
