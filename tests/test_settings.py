from pathlib import Path

import pytest

from clipnado.media import Tools
from clipnado.settings import restore_tools


@pytest.mark.parametrize("custom_model", [False, True])
def test_update_repairs_removed_old_bundle_paths_and_preserves_custom_models(tmp_path, monkeypatch, custom_model):
    new = tmp_path / "new-version"
    new.mkdir()
    for filename in ("ffmpeg.exe", "ffprobe.exe", "whisper-cli.exe", "ggml-base.bin"):
        (new / filename).write_bytes(b"tool")
    defaults = Tools(str(new / "ffmpeg.exe"), str(new / "ffprobe.exe"), str(new / "whisper-cli.exe"), str(new / "ggml-base.bin"))
    monkeypatch.setattr("clipnado.settings.default_tools", lambda: defaults)
    old = tmp_path / "deleted-version"
    saved = {key: str(old / Path(getattr(defaults, key)).name) for key in ("ffmpeg", "ffprobe", "whisper", "model")}
    if custom_model:
        saved["model"] = str(old / "ggml-small.bin")
    result = restore_tools(saved)
    assert result.ffmpeg == defaults.ffmpeg and result.whisper == defaults.whisper
    assert result.model == (saved["model"] if custom_model else defaults.model)
