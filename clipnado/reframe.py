"""Local active-speaker crop planning and a conservative camera director.

No vision libraries are imported in the UI process. The isolated worker uses
bundled MIT models; only crop coordinates are cached, not faces or identities.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

from .process import Cancellation, Cancelled, Progress, run
from .projects import atomic_json
from .media import extract_audio, require_space

VERSION = 2
ANALYSIS_FPS = 25
MODEL_FILES = {
    "yunet-2023mar.onnx": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    "light-asd.onnx": "a6df8167180cc9c8c039f4f244b4bcd393c97a20560ef766c745c99f5c84ce14",
}


def model_bytes(name):
    path = Path(__file__).parent / "vision_models" / name
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(
            "Модель автокадра не найдена. Полностью распакуйте новый ZIP программы."
        ) from exc
    if hashlib.sha256(data).hexdigest() != MODEL_FILES[name]:
        raise RuntimeError(
            "Неверная контрольная сумма модели автокадра. Распакуйте её из проверенного ZIP заново."
        )
    return data


@dataclass
class CropPlan:
    duration_ms: int
    points: list[list[float]]
    info: dict
    warnings: list[str]

    def validate(self, duration_ms):
        if (
            self.duration_ms != duration_ms
            or not isinstance(self.points, list)
            or not self.points
        ):
            raise ValueError("Неверная длительность плана кадрирования")
        previous = -1
        for item in self.points:
            if not isinstance(item, list) or len(item) != 3:
                raise ValueError("Неверные точки кадрирования")
            at, x, y = item
            if (
                not all(type(v) in (int, float) and math.isfinite(v) for v in item)
                or not previous < at < duration_ms
                or not 0 <= x <= 1
                or not 0 <= y <= 1
            ):
                raise ValueError("Кадрирование выходит за границы видео")
            previous = at
        if (
            self.points[0][0] != 0
            or not isinstance(self.info, dict)
            or not isinstance(self.warnings, list)
            or not all(isinstance(w, str) for w in self.warnings)
        ):
            raise ValueError("Неверные данные кадрирования")
        return self

    def at(self, ms):
        import bisect

        i = max(0, bisect.bisect_right([p[0] for p in self.points], ms) - 1)
        return self.points[i][1], self.points[i][2]

    def to_dict(self):
        return {
            "duration_ms": self.duration_ms,
            "points": self.points,
            "info": self.info,
            "warnings": self.warnings,
        }


def clamp(value, lo=0.0, hi=1.0):
    return min(hi, max(lo, value))


def face_position(box, width, height):
    """Crop offset fractions, including headroom for tall source video."""
    x, y, w, h = box
    cw, ch = min(width, height * 9 / 16), min(height, width * 16 / 9)
    left = clamp(x + w / 2 - cw / 2, 0, max(0, width - cw))
    top = clamp(y + h / 2 - ch * 0.38, 0, max(0, height - ch))
    return (
        left / (width - cw) if width > cw else 0.5,
        top / (height - ch) if height > ch else 0.5,
    )


def keep_face(position, box, width, height):
    cw, ch = min(width, height * 9 / 16), min(height, width * 16 / 9)
    result = list(position)
    for axis, (total, view, low, size) in enumerate(
        ((width, cw, box[0], box[2]), (height, ch, box[1], box[3]))
    ):
        if total <= view or size > view * 0.94:
            continue
        padding = min(size * 0.12, (view - size) * 0.35)
        start = result[axis] * (total - view)
        start = clamp(
            start, max(0, low + size + padding - view), min(total - view, low - padding)
        )
        result[axis] = clamp(start / (total - view))
    return tuple(result)


def direct_camera(states, scores, width, height, duration_ms):
    """Choose a visible speaking track, never the midpoint between two faces.

    states: [{scene:int, faces:{track_id:[x,y,w,h]}}] at 25 FPS.
    scores: {track_id: {frame_index:speaking_probability}}, produced by Light-ASD.
    IDs are temporary tracks, not person recognition.
    """
    current, scene, pending, pending_at, last_change = None, None, None, 0, -10000
    position = (0.5, 0.5)
    points = []
    switches = 0
    visible = 0
    ambiguous = 0
    for i, state in enumerate(states):
        ms = i * 1000 // ANALYSIS_FPS
        if ms >= duration_ms:
            break
        faces = state["faces"]
        new_scene = state["scene"] != scene
        if new_scene:
            current, pending, scene = None, None, state["scene"]
            position = (0.5, 0.5)
        if faces:
            visible += 1
            probabilities = {k: float(scores.get(k, {}).get(i, 0)) for k in faces}
            best = max(faces, key=lambda k: probabilities[k])
            if probabilities[best] < 0.6:
                ambiguous += len(faces) > 1
                best = (
                    current
                    if current in faces
                    else max(faces, key=lambda k: faces[k][2] * faces[k][3])
                )
            if current not in faces:
                chosen = best
            elif (
                best != current
                and probabilities[best] >= 0.6
                and probabilities[best] - probabilities[current] >= 0.12
            ):
                if pending != best:
                    pending, pending_at = best, ms
                chosen = (
                    best
                    if ms - pending_at >= 280 and ms - last_change >= 800
                    else current
                )
            else:
                pending = None
                chosen = current
            if chosen != current:
                switches += current is not None
                current, pending, last_change = chosen, None, ms
                position = face_position(faces[current], width, height)
            else:
                desired = face_position(faces[current], width, height)
                position = tuple(
                    old if abs(new - old) < 0.012 else old + (new - old) * 0.18
                    for old, new in zip(position, desired)
                )
                position = keep_face(position, faces[current], width, height)
        # Brief detection loss holds the last view; a new scene starts centered.
        if (
            not points
            or max(abs(position[0] - points[-1][1]), abs(position[1] - points[-1][2]))
            > 0.0004
        ):
            points.append([ms, round(position[0], 6), round(position[1], 6)])
    info = {
        "max_faces": max((len(s["faces"]) for s in states), default=0),
        "switches": switches,
        "face_coverage": round(visible / max(1, len(states)), 3),
        "uncertain_frames": ambiguous,
        "speaker_model_used": bool(scores),
        "frames": len(states),
    }
    warnings = []
    if not visible:
        warnings.append(
            "Лица не найдены: использовано центральное кадрирование. Для такого эпизода можно выбрать размытый фон."
        )
    elif ambiguous > max(25, len(states) * 0.4):
        warnings.append(
            "На части эпизода говорящий определён неуверенно; кадр удерживает видимое лицо. Проверьте результат при просмотре."
        )
    return CropPlan(duration_ms, points or [[0, 0.5, 0.5]], info, warnings).validate(
        duration_ms
    )


def crop_commands(plan, path):
    lines = []
    for ms, x, y in plan.points:
        lines.extend(
            (
                f"{ms/1000:.3f} crop@focus x (iw-ow)*{x:.6f};",
                f"{ms/1000:.3f} crop@focus y (ih-oh)*{y:.6f};",
            )
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def prepare_plan(
    media, candidate, tools, cache, token: Cancellation, progress: Progress
):
    duration = candidate.end_ms - candidate.start_ms
    for name in MODEL_FILES:
        model_bytes(name)
    identity = {
        "source": media.source_file_id,
        "start": candidate.start_ms,
        "end": candidate.end_ms,
        "models": MODEL_FILES,
        "version": VERSION,
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(cache)
    folder = cache / "reframe" / key
    saved = folder / "plan.json"
    try:
        if saved.stat().st_size > 20_000_000:
            raise ValueError("Large crop cache")
        plan = CropPlan(**json.loads(saved.read_text(encoding="utf-8"))).validate(
            duration
        )
        token.check()
        progress(
            100,
            f'Автокадр: использован готовый план · лиц в кадре до {plan.info.get("max_faces",0)}',
        )
        return plan
    except (OSError, ValueError, TypeError, KeyError):
        pass
    folder.mkdir(parents=True, exist_ok=True)
    progress(1, "Автокадр: подготовка локального анализа лиц и речи…")
    with tempfile.TemporaryDirectory(prefix="job-", dir=folder) as tmp:
        work = Path(tmp)
        job = work / "job.json"
        result = work / "result.json"
        frames = work / "frames"
        frames.mkdir()
        require_space(
            work, max(100_000_000, int(duration / 1000 * ANALYSIS_FPS * 70_000))
        )
        run(
            [
                tools.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{candidate.start_ms/1000:.3f}",
                "-i",
                media.path,
                "-t",
                f"{duration/1000:.3f}",
                "-an",
                "-sn",
                "-dn",
                "-map",
                "0:v:0",
                "-filter_threads",
                str(tools.threads),
                "-vf",
                "fps=25,scale=trunc(iw*sar/2)*2:ih,setsar=1,scale=640:640:force_original_aspect_ratio=decrease:force_divisible_by=2",
                "-threads",
                str(tools.threads),
                "-q:v",
                "2",
                "-progress",
                "pipe:1",
                str(frames / "%06d.jpg"),
            ],
            token,
            progress=lambda p, _: progress(
                max(1, p // 5), "Автокадр: подготовка кадров выбранного клипа…"
            ),
            duration_ms=duration,
        )
        audio = work / "audio.wav"
        if media.has_audio:
            extract_audio(media, candidate.start_ms, duration, audio, tools, token)
        atomic_json(
            job,
            {
                "media": media.__dict__,
                "candidate": candidate.__dict__,
                "tools": tools.__dict__,
                "frames": str(frames),
                "audio": str(audio) if audio.is_file() else None,
                "work": str(work),
                "result": str(result),
            },
        )
        executable = Path(sys.executable)
        console = executable.with_name("python.exe")
        if console.is_file():
            executable = console
        pending = ""

        def heartbeat(text, elapsed):
            nonlocal pending
            pending += text
            lines = pending.split("\n")
            pending = lines.pop()
            for line in lines:
                try:
                    data = json.loads(line)
                    if "percent" in data:
                        progress(
                            max(20, min(99, 20 + int(data["percent"]) * 79 // 100)),
                            str(data["text"]),
                        )
                except (ValueError, TypeError, KeyError):
                    pass

        try:
            run(
                [str(executable), "-m", "clipnado.reframe_worker", str(job)],
                token,
                timeout=max(180, duration / 1000 * 10),
                on_output=heartbeat,
            )
        except RuntimeError as exc:
            if result.is_file():
                try:
                    message = json.loads(result.read_text(encoding="utf-8")).get(
                        "error"
                    )
                except (OSError, ValueError):
                    message = None
                if message:
                    raise RuntimeError("Автокадр: " + message) from exc
            raise RuntimeError(
                "Локальный модуль автокадра не запустился. Можно отключить автокадр TikTok в экспорте и использовать прежний центр. "
                + str(exc)
            ) from exc
        plan = CropPlan(**json.loads(result.read_text(encoding="utf-8"))).validate(
            duration
        )
        token.check()
        atomic_json(saved, plan.to_dict())
        progress(
            100,
            f'Автокадр готов · лиц в кадре до {plan.info["max_faces"]} · переключений {plan.info["switches"]}',
        )
        return plan
