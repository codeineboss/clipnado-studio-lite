"""Isolated geometry renderer; cancellation kills both this process and encoder."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile

from .animations import LOOP_SECONDS, render_frame, validate_plan


def main():
    plan_file, output, ffmpeg, width, height, fps, threads = sys.argv[1:]
    width, height, fps, threads = map(int, (width, height, fps, threads))
    with open(plan_file, encoding="utf-8") as stream:
        plan = validate_plan(json.load(stream))

    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with tempfile.TemporaryFile() as log:
        child = subprocess.Popen(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "12",
                "-threads",
                str(threads),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                output,
            ],
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=log,
            creationflags=flags,
        )
        try:
            for i in range(LOOP_SECONDS * fps):
                image = render_frame(plan, width, height, i / fps)
                pixels = image.constBits()
                if image.bytesPerLine() == width * 3:
                    child.stdin.write(pixels)
                else:
                    for y in range(height):
                        child.stdin.write(
                            pixels[
                                y * image.bytesPerLine() : y * image.bytesPerLine()
                                + width * 3
                            ]
                        )
                if i % fps == 0:
                    print(f"out_time_us={i * 1000000 // fps}", flush=True)
            child.stdin.close()
            code = child.wait(timeout=120)
            if code:
                log.seek(0)
                raise RuntimeError(log.read().decode("utf-8", "replace")[-2000:])
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


if __name__ == "__main__":
    main()
