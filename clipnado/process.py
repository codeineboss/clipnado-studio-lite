from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable

Progress = Callable[[int, str], None]


class Cancelled(Exception):
    pass


class Cancellation:
    def __init__(self):
        self.event = threading.Event()

    def cancel(self):
        self.event.set()

    def check(self):
        if self.event.is_set():
            raise Cancelled("Задача отменена")

    def wait(self, seconds: float):
        if self.event.wait(seconds):
            self.check()


def run(args: list[str], token: Cancellation | None = None, cwd: Path | None = None,
        progress: Progress | None = None, duration_ms: int = 0, timeout: float = 86400,
        on_output: Callable[[str, float], None] | None = None) -> str:
    token = token or Cancellation()
    token.check()
    started = time.monotonic()
    output_offset = 0
    next_output = started
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with tempfile.TemporaryDirectory(prefix="clipnado-process-") as temp_dir, open(Path(temp_dir) / "output.log", "w+b") as output:
        try:
            executable = Path(args[0])
            command = [str(executable.absolute()) if executable.is_file() else str(args[0]), *[str(x) for x in args[1:]]]
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                     stdout=output, stderr=subprocess.STDOUT, cwd=cwd, creationflags=flags)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Не найден инструмент: {Path(args[0]).name}. Проверьте настройки.") from exc
        try:
            while child.poll() is None:
                token.check()
                if time.monotonic() - started > timeout:
                    raise RuntimeError("Превышено время ожидания внешнего инструмента")
                if on_output and time.monotonic() >= next_output:
                    with (Path(temp_dir) / "output.log").open("rb") as reader:
                        reader.seek(output_offset)
                        chunk = reader.read(65536)
                        output_offset = reader.tell()
                    on_output(chunk.decode("utf-8", "replace"), time.monotonic() - started)
                    next_output = time.monotonic() + .5
                if progress and duration_ms:
                    # A separate file descriptor avoids racing the subprocess's output position.
                    log_path = Path(temp_dir) / "output.log"
                    with log_path.open("rb") as reader:
                        reader.seek(max(0, log_path.stat().st_size - 2048))
                        data = reader.read().decode("utf-8", "replace")
                    import re
                    times = re.findall(r"out_time_us=(\d+)", data)
                    if times:
                        progress(min(99, int(int(times[-1]) / duration_ms / 10)), "Кодирование…")
                token.wait(0.1)
            token.check()
        finally:
            if child.poll() is None:
                if os.name == "nt":
                    # A Windows venv python.exe can be a redirector that owns a
                    # second Python process. Killing just the redirector leaves
                    # that process alive with our log/video files still open.
                    try:
                        subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       creationflags=flags, timeout=5, check=False)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
                if child.poll() is None:
                    child.terminate()
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        output.seek(0)
        raw = output.read()
        result = raw.decode("utf-8", "replace")
        if on_output:
            on_output(raw[output_offset:].decode("utf-8", "replace"), time.monotonic() - started)
        if child.returncode:
            raise RuntimeError(f"{Path(args[0]).name}: операция не выполнена.\n{result[-2200:]}")
        return result
