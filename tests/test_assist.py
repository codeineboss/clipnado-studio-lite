"""Optional semantic search and nearby-boundary proposals; existing clips stay intact."""
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

import clipnado.local_llm as llm
from clipnado.assist import LocalClipAssistant, neighboring_proposals
from clipnado.media import Tools
from clipnado.models import Candidate, MediaInfo, Project, SearchSettings, Segment, ExportSettings
from clipnado.process import Cancellation, Cancelled
from clipnado.projects import load_project, save_project


@pytest.fixture
def data(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF-test")
    tools = Tools(llama=sys.executable, semantic_model=str(model), acceleration="cpu")
    segments = [Segment(i * 15000, (i + 1) * 15000, f"Реплика {i}") for i in range(12)]
    media = MediaInfo("source", "id", 180000, 1920, 1080, 30, True)
    monkeypatch.setattr(llm, "prompt_tokens", lambda *_: 1000)
    return media, segments, tools


def item(i=0, relevance=90):
    return {"clip_id": i, "hook": 20, "event": 20, "emotion": 20, "completeness": 20,
            "category": "Юмор", "reason": "Соответствующий запросу диалог", "relevance": relevance}


def test_query_is_semantic_condition_and_has_separate_relevance_threshold(data):
    media, segments, _ = data
    plain = SearchSettings()
    query = SearchSettings(query="где не согласны с собеседником")
    a = llm.build_request(media, segments, plain, 0, len(segments), 1)
    b = llm.build_request(media, segments, query, 0, len(segments), 1)
    assert "relevance" not in a.schema["properties"]["clips"]["items"]["required"]
    assert query.query in b.system and "буквальное совпадение слов не обязательно" in b.system
    assert "relevance" in b.schema["properties"]["clips"]["items"]["required"]
    assert llm.decode_choices([item(relevance=69)], media, segments, query, b.proposals, "model") == []
    clips = llm.decode_choices([item(relevance=85)], media, segments, query, b.proposals, "model")
    assert clips[0].score == 80 and clips[0].match_score == 85 and query.query in clips[0].reason


def test_queries_have_independent_cache_and_empty_query_preserves_normal_mode(data, tmp_path, monkeypatch):
    media, segments, tools = data
    calls = []
    def engine(args, token, cwd, **_):
        system = (Path(cwd) / "system.txt").read_text(encoding="utf-8")
        calls.append(system)
        return json.dumps({"clips": [item()]})
    monkeypatch.setattr(llm, "run", engine)
    p = llm.LocalModelProvider(tools)
    for query in ("спор", "спор", "история про школу", ""):
        result = p.analyze(media, segments, SearchSettings(query=query), tmp_path / "cache", Cancellation(), lambda *_: None)
        assert result.info["complete"]
        assert all((c.match_score is not None) == bool(query) for c in result.candidates)
    assert len(calls) == 3


@pytest.mark.parametrize("query", ["x" * 501, None, 12])
def test_invalid_queries_are_rejected(query):
    with pytest.raises(ValueError, match="запрос"):
        SearchSettings(query=query).validate()


def test_neighbors_preserve_event_overlap_source_and_duration(data):
    media, segments, _ = data
    anchor = Candidate("id", 30000, 105000, 80, "Юмор", "Исходник")
    proposals = neighboring_proposals(media, segments, anchor, SearchSettings(), 0, len(segments))
    assert proposals and (proposals[0].start_ms, proposals[0].end_ms) == (30000, 105000)
    for p in proposals:
        assert abs(p.start_ms - anchor.start_ms) <= 30000 and abs(p.end_ms - anchor.end_ms) <= 30000
        assert 60000 <= p.end_ms - p.start_ms <= 90000
        assert min(p.end_ms, anchor.end_ms) - max(p.start_ms, anchor.start_ms) >= .5 * min(p.end_ms-p.start_ms,75000)


def test_refinement_proposes_variants_without_mutating_clip_and_reuses_cache(data, tmp_path, monkeypatch):
    media, segments, tools = data
    anchor = Candidate("id", 30000, 105000, 80, "Юмор", "Исходник", selected=True)
    before = asdict(anchor)
    calls = []
    def engine(args, token, cwd, **_):
        calls.append(1)
        assert "ОДНОГО клипа" in (Path(cwd) / "system.txt").read_text(encoding="utf-8")
        # Current ID=0; a genuine alternative has another ID.
        return json.dumps({"clips": [item(0), item(1)]})
    monkeypatch.setattr(llm, "run", engine)
    p = LocalClipAssistant(tools)
    a = p.refine(media, segments, anchor, SearchSettings(), tmp_path / "cache", Cancellation(), lambda *_: None)
    assert a.info["current_recommended"] and a.candidates and asdict(anchor) == before
    assert all((c.start_ms,c.end_ms)!=(anchor.start_ms,anchor.end_ms) for c in a.candidates)
    b = p.refine(media, segments, anchor, SearchSettings(), tmp_path / "cache", Cancellation(), lambda *_: None)
    assert b.info["cached"] and len(calls) == 1 and [c.id for c in a.candidates] == [c.id for c in b.candidates]


def test_current_only_and_context_overflow_keep_original(data, tmp_path, monkeypatch):
    media, segments, tools = data
    anchor = Candidate("id", 30000, 105000, 80, "Юмор", "Исходник")
    before = copy.deepcopy(anchor)
    monkeypatch.setattr(llm, "run", lambda *_a, **_k: json.dumps({"clips": [item(0)]}))
    result = LocalClipAssistant(tools).refine(media, segments, anchor, SearchSettings(), tmp_path / "a", Cancellation(), lambda *_: None)
    assert not result.candidates and result.info["current_recommended"] and anchor == before
    monkeypatch.setattr(llm, "prompt_tokens", lambda *_: 20000)
    result = LocalClipAssistant(tools).refine(media, segments, anchor, SearchSettings(), tmp_path / "b", Cancellation(), lambda *_: None)
    assert not result.info["complete"] and result.warnings and anchor == before


def test_refinement_cancel_does_not_change_original(data, tmp_path):
    media, segments, tools = data
    anchor = Candidate("id", 30000, 105000, 80, "Юмор", "Исходник")
    token = Cancellation(); token.cancel()
    with pytest.raises(Cancelled):
        LocalClipAssistant(tools).refine(media, segments, anchor, SearchSettings(), tmp_path / "a", token, lambda *_: None)
    assert anchor.start_ms == 30000 and anchor.end_ms == 105000


def test_project_saves_query_results_and_new_export_mode(data, tmp_path):
    media, segments, _ = data
    c = Candidate("id", 0, 60000, 80, "Юмор", "По запросу", match_score=90)
    project = Project(media, segments=segments, query_results=[c], last_query="спор", query_info={"complete": True}, export=ExportSettings(layout="blur_center"))
    path = tmp_path / "project.json"
    save_project(project, path)
    loaded = load_project(path)
    assert loaded.last_query == "спор" and loaded.query_results[0].match_score == 90
    assert not loaded.candidates and loaded.export.layout == "blur_center"
    # Old project files open in the original export/search mode.
    data = project.to_dict()
    for key in ("query_results", "last_query", "query_info"): data.pop(key)
    data["export"].pop("layout"); data["search"].pop("query")
    path.write_text(json.dumps(data), encoding="utf-8")
    old = load_project(path)
    assert old.last_query == "" and old.query_results == [] and old.export.layout == "top_black" and old.search.query == ""
