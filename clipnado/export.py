from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

from .media import Tools, require_space
from .models import Candidate, ExportSettings, MediaInfo, Segment
from .process import Cancellation, Progress, run
from .subtitles import clip_segments, readable_segments, to_ass, to_srt
from .comfort import audio_filter, color_filter
from .animations import animation_region, enabled, render_frame, render_loop, selected_plan


def clip_export_settings(settings: ExportSettings, candidate: Candidate) -> ExportSettings:
    """Resolve a clip's saved choice without mutating the shared settings."""
    effective = replace(settings, layout=settings.layout if candidate.export_layout is None else candidate.export_layout)
    effective.validate()
    return effective


def video_filter(settings: ExportSettings) -> str:
    settings.require_gaming_regions()
    width, height = settings.dimensions()
    if settings.layout == "gaming_split":
        top = int(height * settings.gaming_fraction) // 2 * 2
        bottom = height - top
        def crop(rect):
            x, y, w, h = rect
            return (f"crop=w='max(2,trunc(iw*{w:.9f}/2)*2)':h='max(2,trunc(ih*{h:.9f}/2)*2)':"
                    f"x='trunc(iw*{x:.9f}/2)*2':y='trunc(ih*{y:.9f}/2)*2'")
        return (f"{color_filter(settings)}fps={settings.fps},scale=trunc(iw*sar/2)*2:ih,setsar=1,split=2[camsrc][gamesrc];"
                f"[camsrc]{crop(settings.gaming_camera)},scale={width}:{top}:force_original_aspect_ratio=increase:force_divisible_by=2:flags=lanczos,"
                f"crop={width}:{top},setsar=1[cam];"
                f"[gamesrc]{crop(settings.gaming_game)},scale={width}:{bottom}:force_original_aspect_ratio=increase:force_divisible_by=2:flags=lanczos,"
                f"crop={width}:{bottom},setsar=1[game];"
                "[cam][game]vstack=inputs=2,setsar=1,format=yuv420p")
    if settings.layout == "blur_center":
        # Downscale only the background before Gaussian blur. Both branches
        # originate from the same frame/PTS; the foreground remains uncropped.
        bw, bh = max(2, width // 8 * 2), max(2, height // 8 * 2)
        return (f"{color_filter(settings)}fps={settings.fps},scale=trunc(iw*sar/2)*2:ih,setsar=1,split=2[bgsrc][fgsrc];"
                f"[bgsrc]scale={bw}:{bh}:force_original_aspect_ratio=increase:force_divisible_by=2,"
                f"crop={bw}:{bh},gblur=sigma=12:steps=2,colorchannelmixer=rr={1-settings.background_dim:.3f}:gg={1-settings.background_dim:.3f}:bb={1-settings.background_dim:.3f},scale={width}:{height},setsar=1[bg];"
                f"[fgsrc]scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1[fg];"
                "[bg][fg]overlay=(W-w)/2:(H-h)/2:shortest=1,setsar=1,format=yuv420p")
    zone = int(height * settings.top_fraction) // 2 * 2
    y = int(height * settings.video_y) // 2 * 2
    return (f"{color_filter(settings)}fps={settings.fps},scale=trunc(iw*sar/2)*2:ih,setsar=1,"
            f"scale={width}:{zone}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
            f"setsar=1,pad={width}:{zone}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"pad={width}:{height}:0:{y}:color=black,format=yuv420p")


def validate_export(path: Path, tools: Tools, settings: ExportSettings, duration_ms: int,
                    has_audio: bool, token: Cancellation):
    data = json.loads(run([tools.ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)], token, timeout=60))
    streams = data["streams"]
    video = next(s for s in streams if s["codec_type"] == "video")
    width, height = settings.dimensions()
    if (video["width"], video["height"]) != (width, height) or video["codec_name"] != "h264":
        raise RuntimeError("Проверка экспорта: неверная геометрия или кодек")
    duration = float(data["format"]["duration"]) * 1000
    if abs(duration - duration_ms) > max(200, 2000 / settings.fps):
        raise RuntimeError("Проверка экспорта: неверная длительность")
    if has_audio:
        audio = next((s for s in streams if s["codec_type"] == "audio"), None)
        if not audio or audio["codec_name"] != "aac":
            raise RuntimeError("Проверка экспорта: отсутствует звук AAC")
        if abs(float(audio.get("start_time", 0)) - float(video.get("start_time", 0))) > .1:
            raise RuntimeError("Проверка экспорта: аудио и видео имеют разные стартовые метки")
    run([tools.ffmpeg, "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"], token)


def export_clip(media: MediaInfo, candidate: Candidate, segments: list[Segment], settings: ExportSettings,
                folder: Path, tools: Tools, token: Cancellation, progress: Progress, cache: Path | None = None) -> Path:
    candidate.validate(media)
    settings = clip_export_settings(settings, candidate)
    settings.require_gaming_regions()
    token.check()
    subtitles = clip_segments(segments, candidate.start_ms, candidate.end_ms)
    if settings.subtitles != "off" and not subtitles:
        raise ValueError("В выбранном интервале нет текста для субтитров. Импортируйте SRT или выберите «Выключены».")
    if settings.readable_subtitles:
        subtitles = readable_segments(subtitles)
    folder = Path(folder).resolve()
    duration = candidate.end_ms - candidate.start_ms
    # Conservative high-quality H.264 estimate; FFmpeg's actual disk errors are also surfaced.
    require_space(folder, max(100_000_000, duration * 12000))
    base = f"clip_{candidate.start_ms:010d}_{candidate.end_ms:010d}"
    number = 0
    lock = None
    while True:
        target = folder / f"{base}{'_' + str(number) if number else ''}.mp4"
        lock = target.with_suffix(".lock")
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            if target.exists() or target.with_suffix(".srt").exists():
                lock.unlink()
                number += 1
                continue
            break
        except FileExistsError:
            number += 1
    try:
        with tempfile.TemporaryDirectory(prefix=".clipnado-", dir=folder) as tmp:
            stage = Path(tmp)
            vf = video_filter(settings)
            animation = None
            if enabled(settings):
                if cache is None:
                    from .settings import data_root
                    cache = data_root() / "cache"
                animation = render_loop(selected_plan(candidate, settings.animation), settings, cache, tools, token, progress)
                _, _, y = animation_region(settings)
                vf += f"[base];[1:v]setpts=PTS-STARTPTS[anim];[base][anim]overlay=0:{y}:shortest=1"
            if settings.subtitles == "burn":
                width, height = settings.dimensions()
                (stage / "subs.ass").write_text(to_ass(subtitles, width, height, settings.subtitle_y,
                                                        settings.video_y if settings.layout == "top_black" else 0,
                                                        settings.top_fraction if settings.layout == "top_black" else 1, settings.readable_subtitles), encoding="utf-8-sig")
                # Relative ASCII filter paths work with Windows drives, Cyrillic folders and apostrophes.
                vf += ",ass=filename=subs.ass"
            af = audio_filter(media, candidate, settings, tools, token, progress)
            args = [tools.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{candidate.start_ms / 1000:.3f}",
                    "-i", media.path]
            if animation:
                args += ["-stream_loop", "-1", "-i", str(animation)]
                args += ["-filter_complex_threads", str(tools.threads), "-filter_complex", vf + "[out]", "-map", "[out]"]
            else:
                args += ["-filter_threads", str(tools.threads), "-vf", vf, "-map", "0:v:0"]
            args += ["-t", f"{duration / 1000:.3f}", "-map", "0:a:0?", "-sn", "-dn", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(settings.quality), "-threads", str(tools.threads),
                    "-c:a", "aac", "-b:a", "192k", "-af", af, "-movflags", "+faststart",
                    "-map_metadata", "-1", "-progress", "pipe:1", str(stage / "clip.mp4")]
            run(args, token, cwd=stage, progress=progress, duration_ms=duration)
            progress(99, "Проверка MP4…")
            validate_export(stage / "clip.mp4", tools, settings, duration, media.has_audio, token)
            token.check()
            # Publish only validated files. Roll back sidecar if the MP4 commit fails.
            sidecar = target.with_suffix(".srt")
            if settings.subtitles == "srt":
                (stage / "clip.srt").write_text(to_srt(subtitles), encoding="utf-8-sig")
                os.replace(stage / "clip.srt", sidecar)
            try:
                os.replace(stage / "clip.mp4", target)
            except Exception:
                if settings.subtitles == "srt":
                    sidecar.unlink(missing_ok=True)
                raise
            return target
    finally:
        if lock:
            lock.unlink(missing_ok=True)


def composition_frame(media: MediaInfo, candidate: Candidate, settings: ExportSettings,
                      tools: Tools, token: Cancellation, segments: list[Segment] | None = None) -> bytes:
    """Render the export geometry, animation and optional burned captions."""
    candidate.validate(media)
    settings = clip_export_settings(settings, candidate)
    at = min(candidate.start_ms + 1000, (candidate.start_ms + candidate.end_ms) // 2)
    with tempfile.TemporaryDirectory(prefix="clipnado-composition-") as tmp:
        path = Path(tmp) / "frame.png"
        # Accurate seek can land after the final encoded frame of a short or
        # low-FPS input. Try the preceding visible frame instead of failing on
        # a missing PNG at EOF.
        previous = max(0, at - max(100, 1000 / media.fps if media.fps else 1000))
        for position in (at, previous):
            vf = video_filter(settings)
            args = [tools.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{position / 1000:.3f}", "-i", media.path]
            if enabled(settings):
                width, height, y = animation_region(settings)
                image = render_frame(selected_plan(candidate, settings.animation), width, height, max(0, position-candidate.start_ms)/1000)
                if not image.save(str(Path(tmp) / "animation.png")):
                    raise OSError("Не удалось сохранить кадр анимации")
                args += ["-loop", "1", "-i", str(Path(tmp) / "animation.png")]
                vf += f"[base];[base][1:v]overlay=0:{y}:shortest=1"
            if settings.subtitles == "burn" and segments is not None:
                captions = clip_segments(segments, candidate.start_ms, candidate.end_ms)
                if settings.readable_subtitles:
                    captions = readable_segments(captions)
                width, height = settings.dimensions()
                (Path(tmp) / "subs.ass").write_text(to_ass(captions, width, height, settings.subtitle_y,
                    settings.video_y if settings.layout == "top_black" else 0,
                    settings.top_fraction if settings.layout == "top_black" else 1, settings.readable_subtitles), encoding="utf-8-sig")
                vf += f",setpts=PTS+{max(0, position-candidate.start_ms)/1000:.3f}/TB,ass=filename=subs.ass,setpts=PTS-STARTPTS"
            vf += ",scale=360:640:force_original_aspect_ratio=decrease"
            if enabled(settings):
                args += ["-filter_complex_threads", str(tools.threads), "-filter_complex", vf + "[out]", "-map", "[out]"]
            else:
                args += ["-filter_threads", str(tools.threads), "-vf", vf, "-map", "0:v:0"]
            run(args + ["-an", "-sn", "-dn", "-frames:v", "1", "-threads", str(tools.threads), str(path)], token, cwd=Path(tmp), timeout=60)
            if path.is_file():
                return path.read_bytes()
        raise ValueError("В этом месте видео нет доступного кадра для предпросмотра.")
