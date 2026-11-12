"""Optional real-engine test: set CLIPNADO_TEST_WHISPER and CLIPNADO_TEST_MODEL.

The audio fixture is public FLEURS ru_ru/test, CC-BY-4.0; see THIRD_PARTY.md.
No network requests or paid APIs are used by this test.
"""
import os
import subprocess
from pathlib import Path

import pytest

from clipnado.analysis import recognize
from clipnado.media import Tools, probe
from clipnado.process import Cancellation


@pytest.mark.skipif(not os.environ.get("CLIPNADO_TEST_WHISPER") or not os.environ.get("CLIPNADO_TEST_MODEL"), reason="Set local whisper/model paths to run actual Russian ASR")
def test_actual_russian_recognition_and_cache(tmp_path, monkeypatch):
    tools = Tools(whisper=os.environ["CLIPNADO_TEST_WHISPER"], model=os.environ["CLIPNADO_TEST_MODEL"], threads=2)
    audio = Path(__file__).resolve().parent.parent / "qa" / "russian-human.wav"
    video = tmp_path / "русская речь.mp4"
    subprocess.run([tools.ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=s=160x90:r=1",
                    "-stream_loop", "-1", "-i", str(audio), "-t", "65",
                    "-c:v", "libx264", "-threads", "2", "-c:a", "aac", str(video)], check=True)
    media = probe(video, tools)
    # Real copy fallback (e.g. cache and model on different volumes), with a Unicode working directory.
    def different_volume(*_):
        raise OSError("Cross-device link")
    monkeypatch.setattr(os, "link", different_volume)
    cache = tmp_path / "Кэш с пробелом ' тест"
    events = []
    segments, energy, warnings = recognize(media, tools, cache, Cancellation(), lambda p, text: events.append((p, text)))
    text = " ".join(s.text for s in segments).lower()
    assert "китае" in text and "времени" in text and "власти" in text
    assert energy
    assert any(0 < p < 80 and "Whisper" in text for p, text in events)
    assert any("блок 2/2" in text for _, text in events)
    for s in segments:
        s.validate(media.duration_ms)
    cached = recognize(media, tools, cache, Cancellation(), lambda *_: pytest.fail("Recomputed cached ASR"))
    assert cached[0] == segments
