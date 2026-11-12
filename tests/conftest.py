import os
import subprocess
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def tools():
    from clipnado.media import Tools
    return Tools(threads=2)


@pytest.fixture(scope="session")
def source(tmp_path_factory):
    folder = tmp_path_factory.mktemp("медиа с пробелом ' тест")
    path = folder / "исходное видео.mp4"
    # Red 16:9 image, synchronous white flash and tone at t=2..3 sec; non-integer FPS.
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=red:s=320x180:r=30000/1001:d=8,drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='between(t,2,3)'",
                    "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000:duration=8",
                    "-af", "volume='if(between(t,2,3),1,0)':eval=frame", "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True)
    return path


@pytest.fixture
def media(source, tools):
    from clipnado.media import probe
    return probe(source, tools)
