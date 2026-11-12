import json

import httpx
import pytest

from clipnado.models import MediaInfo, SearchSettings, Segment
from clipnado.process import Cancellation
from clipnado.providers import GeminiFreeProvider, ProviderError

MEDIA = MediaInfo("PRIVATE_FILE.mp4", "SECRET_SOURCE_ID", 120000, 1920, 1080, 30, True)
TEXT = [Segment(0, 60000, "Однажды началась интересная история."), Segment(60000, 120000, "В итоге всё получилось.")]


def provider(handler, consent=True, free=True):
    return GeminiFreeProvider("TEST_SECRET_KEY", consent, free, transport=httpx.MockTransport(handler))


def test_no_key():
    with pytest.raises(ProviderError, match="ключ"):
        GeminiFreeProvider("")


@pytest.mark.parametrize("consent,free", [(False, True), (True, False), (False, False)])
def test_no_network_without_consent(consent, free):
    def handler(_):
        pytest.fail("Network was used without consent")
    with pytest.raises(ProviderError):
        provider(handler, consent, free).analyze(MEDIA, TEXT, SearchSettings(), [], Cancellation())


@pytest.mark.parametrize("code", [400, 401, 403, 404, 429])
def test_api_errors_do_not_leak_key_or_retry_quota(code):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(code, text="TEST_SECRET_KEY is invalid")
    with pytest.raises(ProviderError) as error:
        provider(handler).analyze(MEDIA, TEXT, SearchSettings(), [], Cancellation())
    assert str(code) in str(error.value)
    assert "TEST_SECRET_KEY" not in str(error.value)
    assert len(calls) == 1
    assert "key=" not in str(calls[0].url)


def test_only_text_and_validated_candidates_sent():
    def handler(request):
        body = request.content.decode()
        assert "PRIVATE_FILE" not in body and "SECRET_SOURCE_ID" not in body and "TEST_SECRET_KEY" not in body
        assert "inlineData" not in body and "fileData" not in body
        assert "Однажды" in body
        clips = [{"start_ms": 0, "end_ms": 60000, "score": 85, "category": "История", "reason": "Цельная история"},
                 {"start_ms": -10, "end_ms": 60000, "score": 99, "category": "История", "reason": "Плохой диапазон"},
                 {"start_ms": 0, "end_ms": 60000, "score": 999, "category": "История", "reason": "Неверный рейтинг"}]
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": json.dumps({"clips": clips})}]}}]})
    clips = provider(handler).analyze(MEDIA, TEXT, SearchSettings(), [], Cancellation())
    assert len(clips) == 1 and clips[0].score == 85


def test_offline_limited_retry(monkeypatch):
    calls = []
    def handler(request):
        calls.append(1)
        raise httpx.ConnectError("TEST_SECRET_KEY", request=request)
    token = Cancellation()
    monkeypatch.setattr(token, "wait", lambda _: None)
    with pytest.raises(ProviderError, match="соединения") as error:
        provider(handler).analyze(MEDIA, TEXT, SearchSettings(), [], token)
    assert "TEST_SECRET_KEY" not in str(error.value)
    assert len(calls) == 2


def test_connection_check_does_not_generate():
    def handler(request):
        assert request.method == "GET" and "generateContent" not in str(request.url)
        return httpx.Response(200, json={"name": "models/gemini-2.5-flash"})
    assert "Бесплатность" in provider(handler).check_connection(Cancellation())


def test_target_is_requested_but_low_scores_are_never_promoted():
    def handler(request):
        body = json.loads(request.content)
        instruction = body["systemInstruction"]["parts"][0]["text"]
        assert "минимум 10" in instruction and "оценкой от 60" in instruction
        assert "не завышай" in instruction
        clips = [{"start_ms": 0, "end_ms": 60000, "score": 59, "category": "История", "reason": "Недостаточно признаков"}]
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": json.dumps({"clips": clips})}]}}]})
    assert provider(handler).analyze(MEDIA, TEXT, SearchSettings(sensitivity=100), [], Cancellation()) == []
