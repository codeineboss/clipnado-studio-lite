"""Mock API protocol tests and real FFmpeg preparation; no live model benchmark."""
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from clipnado.groq_analysis import GroqFreeProvider, speech_chunk, text_windows
from clipnado.models import MediaInfo, SearchSettings
from clipnado.process import Cancellation, Cancelled
from clipnado.providers import ProviderError
from clipnado.video_analysis import GeminiVideoProvider, prepare_proxy, video_windows


def moment(start=0, end=60000):
    return {"start_ms": start, "end_ms": end, "evidence_ms": start + 30000,
            "hook": 20, "event": 20, "emotion": 20, "completeness": 20,
            "category": "Реакция", "reason": "Вопрос, неожиданный ответ и реакция."}


def gemini_body(items):
    return {"candidates": [{"content": {"parts": [{"text": json.dumps({"clips": items})}]}}]}


@pytest.fixture
def online_media():
    return MediaInfo("PRIVATE_SOURCE.mp4", "source-id", 420000, 1920, 1080, 60, True)


@pytest.fixture
def fake_proxy(tmp_path, monkeypatch):
    proxy = tmp_path / "proxy.mp4"
    proxy.write_bytes(b"test-video-payload")
    monkeypatch.setattr("clipnado.video_analysis.prepare_proxy", lambda *_: proxy)
    # Protocol waits are not model latency measurements.
    now = [0]
    def monotonic():
        now[0] += 60
        return now[0]
    monkeypatch.setattr("clipnado.video_analysis.time", SimpleNamespace(monotonic=monotonic))
    monkeypatch.setattr(Cancellation, "wait", lambda self, _: self.check())
    return proxy


def test_video_windows_cover_thirty_minutes_and_boundary_events():
    windows = video_windows(1800000, 90)
    assert windows[0][0] == 0 and windows[-1][1] == 1800000
    assert all(b[0] <= a[1] - 90000 for a, b in zip(windows, windows[1:]))
    for start in range(0, 1800000 - 90000, 1000):
        assert any(a <= start and start + 90000 <= b for a, b in windows)


def test_real_video_proxy_preserves_duration_audio_and_cache(media, tools, tmp_path, monkeypatch):
    original = Path(media.path).read_bytes()
    path = prepare_proxy(media, tools, tmp_path, Cancellation(), lambda *_: None)
    assert path.exists() and path.read_bytes() != original
    assert Path(media.path).read_bytes() == original
    actual = __import__("clipnado.video_analysis", fromlist=["run"]).run
    def cached_only(args, *a, **kw):
        assert args[0] == tools.ffprobe, "Cached video was encoded again"
        return actual(args, *a, **kw)
    monkeypatch.setattr("clipnado.video_analysis.run", cached_only)
    assert prepare_proxy(media, tools, tmp_path, Cancellation(), lambda *_: None) == path


@pytest.mark.parametrize("video,free", [(False, True), (True, False)])
def test_video_never_prepares_or_sends_without_both_consents(online_media, tools, tmp_path, video, free):
    def handler(_):
        pytest.fail("Network without consent")
    p = GeminiVideoProvider("SECRET", True, free, video_consent=video, transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError, match="согласие"):
        p.analyze_video(online_media, SearchSettings(), tools, tmp_path, Cancellation(), lambda *_: None)
    assert not list(tmp_path.iterdir())


def video_handler(log, *, quota_second=False, malicious_upload=False, bad_moments=False, cancel_on_upload=None):
    def handler(request):
        log.append((request.method, request.url.path))
        assert request.url.host == "generativelanguage.googleapis.com"
        assert request.headers["x-goog-api-key"] == "SECRET"
        assert "SECRET" not in str(request.url)
        if request.url.path.startswith("/v1beta/models/") and request.method == "GET":
            return httpx.Response(200, json={"name": "models/gemini-3.8-flash"})
        if request.url.path == "/upload/v1beta/files":
            assert b"PRIVATE_SOURCE" not in request.content
            url = "https://evil.example/collect" if malicious_upload else "https://generativelanguage.googleapis.com/upload/session"
            return httpx.Response(200, headers={"x-goog-upload-url": url})
        if request.url.path == "/upload/session":
            assert request.content == b"test-video-payload"
            assert int(request.headers["content-length"]) == len(request.content)
            assert "transfer-encoding" not in request.headers
            if cancel_on_upload:
                cancel_on_upload.cancel()
            return httpx.Response(200, json={"file": {"name": "files/test", "state": "ACTIVE",
                "uri": "https://generativelanguage.googleapis.com/v1beta/files/test"}})
        if request.method == "DELETE":
            return httpx.Response(200, json={})
        data = json.loads(request.content)
        part = data["contents"][0]["parts"][0]
        assert part["videoMetadata"]["fps"] == 2 and "fileData" in part
        assert b"PRIVATE_SOURCE" not in request.content
        start = round(float(part["videoMetadata"]["startOffset"][:-1]) * 1000)
        if quota_second and start:
            return httpx.Response(429, text="SECRET must not be exposed")
        return httpx.Response(200, json=gemini_body([moment(-1, 1)] if bad_moments else [moment(start, start + 60000)]))
    return handler


def test_video_full_protocol_cache_resume_and_quota(online_media, tools, tmp_path, fake_proxy):
    cache = tmp_path / "cache"
    log = []
    p = GeminiVideoProvider("SECRET", True, True, video_consent=True,
        transport=httpx.MockTransport(video_handler(log, quota_second=True)))
    partial = p.analyze_video(online_media, SearchSettings(), tools, cache, Cancellation(), lambda *_: None)
    assert not partial.info["complete"] and partial.info["checked_windows"] == 1
    assert partial.candidates and all("SECRET" not in w for w in partial.warnings)
    assert log[-1] == ("DELETE", "/v1beta/files/test")
    for path in cache.rglob("*.json"):
        assert "SECRET" not in path.read_text() and "fileUri" not in path.read_text()
    log.clear()
    p.transport = httpx.MockTransport(video_handler(log))
    result = p.analyze_video(online_media, SearchSettings(), tools, cache, Cancellation(), lambda *_: None)
    assert result.info["complete"] and result.info["local_asr_skipped"]
    assert result.info["checked_windows"] == result.info["total_windows"] == 2
    assert len(result.candidates) == 2
    assert sum(path.endswith(":generateContent") for _, path in log) == 1, "Finished window was sent twice"
    def offline(_):
        pytest.fail("Complete cached analysis performed network request")
    p.transport = httpx.MockTransport(offline)
    again = p.analyze_video(online_media, SearchSettings(), tools, cache, Cancellation(), lambda *_: None)
    assert again.info["from_cache"] and [c.id for c in again.candidates] == [c.id for c in result.candidates]


def test_video_rejects_external_upload_destination(online_media, tools, tmp_path, fake_proxy):
    log = []
    p = GeminiVideoProvider("SECRET", True, True, video_consent=True,
        transport=httpx.MockTransport(video_handler(log, malicious_upload=True)))
    with pytest.raises(ProviderError, match="адрес"):
        p.analyze_video(online_media, SearchSettings(), tools, tmp_path, Cancellation(), lambda *_: None)
    assert len(log) == 2


def test_cancel_after_upload_still_deletes_remote_copy(online_media, tools, tmp_path, fake_proxy):
    log, token = [], Cancellation()
    p = GeminiVideoProvider("SECRET", True, True, video_consent=True,
        transport=httpx.MockTransport(video_handler(log, cancel_on_upload=token)))
    with pytest.raises(Cancelled):
        p.analyze_video(online_media, SearchSettings(), tools, tmp_path, token, lambda *_: None)
    assert log[-1] == ("DELETE", "/v1beta/files/test")


def test_bad_model_timestamps_are_not_saved_as_checked(online_media, tools, tmp_path, fake_proxy):
    log = []
    p = GeminiVideoProvider("SECRET", True, True, video_consent=True,
        transport=httpx.MockTransport(video_handler(log, bad_moments=True)))
    with pytest.raises(ProviderError, match="таймкоды"):
        p.analyze_video(online_media, SearchSettings(), tools, tmp_path, Cancellation(), lambda *_: None)
    assert not list(tmp_path.rglob("window-*.json"))
    assert log[-1][0] == "DELETE"


def test_video_waits_for_file_processing_before_inference(online_media, tools, tmp_path, fake_proxy):
    log = []
    actual = video_handler(log)
    def handler(request):
        if request.url.path == "/v1beta/files/test" and request.method == "GET":
            log.append(("GET", request.url.path))
            return httpx.Response(200, json={"name": "files/test", "state": "ACTIVE",
                "uri": "https://generativelanguage.googleapis.com/v1beta/files/test"})
        response = actual(request)
        if request.url.path == "/upload/session":
            data = json.loads(response.content)
            data["file"]["state"] = "PROCESSING"
            return httpx.Response(200, json=data)
        return response
    p = GeminiVideoProvider("SECRET", True, True, video_consent=True, transport=httpx.MockTransport(handler))
    assert p.analyze_video(online_media, SearchSettings(), tools, tmp_path, Cancellation(), lambda *_: None).info["complete"]
    assert ("GET", "/v1beta/files/test") in log


def test_real_compact_speech_preparation(media, tools, tmp_path):
    path = tmp_path / "audio.mp3"
    speech_chunk(media, 0, media.duration_ms, path, tools, Cancellation(), lambda *_: None)
    assert 1000 < path.stat().st_size < 100000


def test_groq_server_asr_selection_and_cache(media, tools, tmp_path, monkeypatch):
    media.duration_ms = 120000
    calls = []
    def speech(*args):
        args[3].write_bytes(b"audio-payload")
    monkeypatch.setattr("clipnado.groq_analysis.speech_chunk", speech)
    def handler(request):
        calls.append(request.url.path)
        assert request.url.host == "api.groq.com" and request.headers["Authorization"] == "Bearer SECRET"
        assert "SECRET" not in str(request.url)
        if request.url.path.endswith("/audio/transcriptions"):
            assert b"audio-payload" in request.content and b"whisper-large-v3" in request.content
            assert str(media.path).encode() not in request.content
            return httpx.Response(200, json={"segments": [{"start": 0, "end": 60, "text": "Вопрос и ответ."},
                                                         {"start": 60, "end": 120, "text": "Завершение истории."}]})
        body = json.loads(request.content)
        assert body["model"] == "openai/gpt-oss-120b" and "service_tier" not in body
        assert "Вопрос" in body["messages"][1]["content"]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"clips": [moment()]})}}]})
    p = GroqFreeProvider("SECRET", True, True, transport=httpx.MockTransport(handler))
    result = p.analyze_video(media, SearchSettings(), tools, tmp_path, Cancellation(), lambda *_: None)
    assert result.info["complete"] and result.info["local_asr_skipped"]
    assert len(result.candidates) == 1 and result.candidates[0].score == 80 and len(result.segments) == 2
    calls.clear()
    p.transport = httpx.MockTransport(lambda _: pytest.fail("Groq cache repeated API request"))
    assert p.analyze_video(media, SearchSettings(), tools, tmp_path, Cancellation(), lambda *_: None).info["complete"]
    assert not calls


@pytest.mark.parametrize("consent,free", [(False, True), (True, False)])
def test_groq_no_network_without_free_consent(online_media, tools, tmp_path, consent, free):
    p = GroqFreeProvider("SECRET", consent, free, transport=httpx.MockTransport(lambda _: pytest.fail("Network without consent")))
    with pytest.raises(ProviderError, match="согласие"):
        p.analyze_video(online_media, SearchSettings(), tools, tmp_path, Cancellation(), lambda *_: None)


def test_groq_quota_is_not_retried_or_leaked():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(429, text="SECRET invalid")
    p = GroqFreeProvider("SECRET", True, True, transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError) as error:
        p._request("GET", "/models", Cancellation())
    assert "SECRET" not in str(error.value) and "429" in str(error.value) and len(calls) == 1


def test_groq_dense_text_windows_preserve_coverage_and_budget():
    from clipnado.models import Segment
    segments = [Segment(i, i + 5000, "Русская речь и подробное описание события. " * 4) for i in range(0, 600000, 5000)]
    windows = text_windows(segments, 600000, 90)
    assert windows[0][0] == 0 and windows[-1][1] == 600000
    for start, end in windows:
        assert sum(len(s.text) + 70 for s in segments if s.start_ms < end and s.end_ms > start) <= 6000
    for start in range(0, 600000 - 90000, 1000):
        assert any(a <= start and start + 90000 <= b for a, b in windows)


def test_groq_partial_speech_resumes_without_becoming_a_full_transcript(online_media, tools, tmp_path, monkeypatch):
    monkeypatch.setattr("clipnado.groq_analysis.speech_chunk", lambda *args: args[3].write_bytes(b"audio"))
    now = [0]
    def monotonic():
        now[0] += 120
        return now[0]
    monkeypatch.setattr("clipnado.groq_analysis.time", SimpleNamespace(monotonic=monotonic))
    calls, quota = [], [True]
    def handler(request):
        calls.append(request.url.path)
        if request.url.path.endswith("/audio/transcriptions"):
            if quota[0] and len(calls) == 2:
                return httpx.Response(429)
            return httpx.Response(200, json={"segments": [{"start": 0, "end": 60, "text": "Вопрос и ответ в этом фрагменте."}]})
        transcript = json.loads(json.loads(request.content)["messages"][1]["content"])
        start = transcript[0]["start_ms"]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"clips": [moment(start, start + 60000)]})}}]})
    p = GroqFreeProvider("SECRET", True, True, transport=httpx.MockTransport(handler))
    partial = p.analyze_video(online_media, SearchSettings(), tools, tmp_path, Cancellation(), lambda *_: None)
    assert not partial.info["complete"] and partial.info["checked_speech_blocks"] == 1
    assert partial.segments is None and not partial.candidates
    quota[0] = False
    calls.clear()
    result = p.analyze_video(online_media, SearchSettings(), tools, tmp_path, Cancellation(), lambda *_: None)
    assert result.info["complete"] and len(result.segments) == 2
    assert sum(path.endswith("/audio/transcriptions") for path in calls) == 1
    assert result.segments[1].start_ms == 300000
