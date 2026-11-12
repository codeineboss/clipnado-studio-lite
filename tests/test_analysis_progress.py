import json
import sys
import wave
from pathlib import Path

import pytest

from clipnado.analysis import CHUNK_MS, SpeechProgress, recognize
from clipnado.media import Tools
from clipnado.models import MediaInfo
from clipnado.process import Cancellation, Cancelled, run


def test_child_output_and_heartbeat_arrive_before_completion():
    events = []
    script = "import time;time.sleep(.8);print('progress = 50%',flush=True);time.sleep(.7);print('[00:00:00.000 --> 00:00:45.900]',flush=True)"
    result = run([sys.executable, "-c", script], on_output=lambda text, elapsed: events.append((text, elapsed)))
    assert any(not text and elapsed < .8 for text, elapsed in events)
    assert any("50%" in text and elapsed < 1.5 for text, elapsed in events)
    assert "00:00:45.900" in result


def test_heartbeat_can_cancel_a_silent_child():
    token = Cancellation()
    def tick(text, elapsed):
        if elapsed >= .4:
            token.cancel()
    with pytest.raises(Cancelled):
        run([sys.executable, "-c", "import time;time.sleep(20)"], token, on_output=tick)


def test_whisper_progress_handles_split_lines_and_never_goes_back():
    parser = SpeechProgress(60000)
    assert parser.feed("progress = 5") == 0
    assert parser.feed("0%\n") == 30000
    assert parser.feed("[00:00:30.000 --> 00:00:45.900] речь\n") == 45900
    assert parser.feed("progress = 10%\n") == 45900
    assert parser.feed("progress = 100%\n") == 60000


def test_nineteen_minute_analysis_has_minute_chunks_live_progress_and_resume(tmp_path, monkeypatch):
    model = tmp_path / "base.bin"
    model.write_bytes(b"test-model")
    tools = Tools(whisper=sys.executable, model=str(model))
    media = MediaInfo("nineteen-minutes.mp4", "source-id", 1152894, 1920, 1080, 60, True)
    cache = tmp_path / "Кэш"
    extracted = []
    completed = []
    events = []
    active_token = Cancellation()
    should_cancel = [True]

    def extract(media, start, duration, path, tools, token, progress=None):
        assert duration <= CHUNK_MS + 1500
        extracted.append((start, duration))
        with wave.open(str(path), "wb") as output:
            output.setparams((1, 2, 16000, 16000, "NONE", "not compressed"))
            output.writeframes(b"\x01\x00" * 16000)

    def engine(args, token, cwd=None, on_output=None, **kwargs):
        assert "-pp" in args
        assert on_output is not None
        on_output("", .5)
        on_output("progress = 50%\n", 1.0)
        if len(completed) == 2 and should_cancel[0]:
            token.cancel()
            token.check()
        duration = extracted[-1][1]
        (Path(cwd) / "speech.json").write_text(json.dumps({"transcription": [
            {"offsets": {"from": 0, "to": duration}, "text": "Однажды случилась интересная история, и потом всё получилось."}
        ]}, ensure_ascii=False), encoding="utf-8")
        completed.append(extracted[-1])
        return ""

    monkeypatch.setattr("clipnado.analysis.extract_audio", extract)
    monkeypatch.setattr("clipnado.analysis.run", engine)
    with pytest.raises(Cancelled):
        recognize(media, tools, cache, active_token, lambda p, message: events.append((p, message)))
    saved = json.loads(next(cache.glob("*.checkpoint.json")).read_text(encoding="utf-8"))
    assert saved["next_start_ms"] == 120000
    assert any(0 < p < 80 and "Whisper" in message for p, message in events)
    assert any("ждём первый результат" in message for _, message in events)
    should_cancel[0] = False
    # Update extracted to a different folder: identical model content must resume.
    relocated = tmp_path / "обновление" / "base.bin"
    relocated.parent.mkdir()
    relocated.write_bytes(model.read_bytes())
    tools.model = str(relocated)
    before = len(extracted)
    segments, energy, warnings = recognize(media, tools, cache, Cancellation(), lambda p, message: events.append((p, message)))
    assert extracted[before][0] == 118500
    assert len(completed) == 20
    assert len(segments) == 20
    assert not list(cache.glob("*.checkpoint.json"))
    assert any("Продолжение с 00:02:00" in message for _, message in events)
    for segment in segments:
        segment.validate(media.duration_ms)
    assert all(a.end_ms <= b.start_ms for a, b in zip(segments, segments[1:]))
    assert segments[-1].end_ms == media.duration_ms
    assert recognize(media, tools, cache, Cancellation(), lambda *_: pytest.fail("Cache was recomputed"))[0] == segments
