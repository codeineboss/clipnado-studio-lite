"""Build a two-face public-domain synthetic fixture; no downloads at runtime."""

from pathlib import Path
import subprocess


def make_fixture(path, ffmpeg="ffmpeg"):
    import cv2
    from clipnado.reframe_worker import read_frame

    portrait = read_frame(
        Path(__file__).resolve().parent.parent / "qa/fixtures/nasa-astronaut.png"
    )
    # Large separation reproduces the user's rigid-center cropping failure.
    image = cv2.hconcat(
        [cv2.resize(portrait, (320, 320)), cv2.resize(portrait, (320, 320))]
    )
    still = Path(path).with_suffix(".jpg")
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Cannot create vision QA fixture")
    encoded.tofile(still)
    subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-y",
            "-loop",
            "1",
            "-i",
            str(still),
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            "-t",
            "3",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )
    still.unlink()
    return Path(path)
