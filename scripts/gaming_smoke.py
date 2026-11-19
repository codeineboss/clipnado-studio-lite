"""Our synthetic stream: camera top-right, game below, synchronized flash/tone."""

from pathlib import Path
import subprocess

CAMERA = [0.75, 0.0, 0.25, 1 / 3]
GAME = [0.0, 1 / 3, 1.0, 2 / 3]


def make_fixture(path, ffmpeg="ffmpeg"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    video = (
        "color=c=blue:s=640x360:r=30000/1001:d=5,"
        "drawbox=x=480:y=0:w=160:h=120:color=red:t=fill,"
        "drawbox=x=480:y=0:w=160:h=120:color=white:t=fill:enable='between(t,2,3)',"
        "drawbox=x=0:y=120:w=640:h=240:color=lime:t=fill:enable='between(t,2,3)'"
    )
    subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            video,
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:sample_rate=48000:duration=5",
            "-af",
            "volume='if(between(t,2,3),1,0)':eval=frame",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-threads",
            "2",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
    )
    return path
