import array
import subprocess
from pathlib import Path

import pytest

from clipnado.export import export_clip
from clipnado.media import probe
from clipnado.models import Candidate, ExportSettings, Segment
from clipnado.process import Cancellation, Cancelled
from clipnado.subtitles import from_srt


def frame(path, tools, at=0, vf=None):
    args = [tools.ffmpeg, "-v", "error", "-ss", str(at), "-i", str(path), "-frames:v", "1"]
    if vf:
        args += ["-vf", vf]
    return subprocess.check_output(args + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])


@pytest.mark.parametrize("aspect,subs", [("9:16", "off"), ("16:9", "srt"), ("1:1", "burn"), ("9:16", "burn"), ("16:9", "burn")])
def test_real_export(media, tools, tmp_path, aspect, subs):
    c = Candidate(media.source_file_id, 1000, 4000, 80, "История", "Тест")
    settings = ExportSettings(aspect=aspect, resolution=720, subtitles=subs)
    out = export_clip(media, c, [Segment(500, 5000, "Привет, мир! Это тест кириллицы.")], settings,
                      tmp_path / "Экспорт с пробелом ' тест", tools, Cancellation(), lambda *_: None)
    assert out.exists()
    width, height = settings.dimensions()
    image = frame(out, tools, .2)
    # Entire lower zone remains black, including when captions are burned into the upper zone.
    bottom = image[width * int(height * .50) * 3:]
    assert max(bottom) <= 5
    # Red source's visible rectangle has 16:9 proportions within the upper zone.
    middle_x = width // 2
    rows = [y for y in range(int(height * .4)) if image[(y * width + middle_x) * 3] > 150]
    expected_height = min(width / (16 / 9), int(height * .4) // 2 * 2)
    assert abs(len(rows) - expected_height) < 8
    assert abs(probe(out, tools).duration_ms - 3000) <= 100
    if subs == "srt":
        segments = from_srt(out.with_suffix(".srt").read_text(encoding="utf-8-sig"), 3000)
        assert segments[0].start_ms == 0 and segments[0].end_ms == 3000
        assert "Привет" in segments[0].text
    else:
        assert not out.with_suffix(".srt").exists()
    if subs == "burn":
        # White Cyrillic glyph pixels in an otherwise red/black frame prove actual subtitle rendering.
        zone = image[:width * int(height * .4) * 3]
        whites = sum(1 for r, g, b in zip(zone[::3], zone[1::3], zone[2::3]) if min(r, g, b) > 180)
        assert whites > 100
    # The tone and white flash still begin at exported t≈1 s (source t=2 s).
    raw = subprocess.check_output([tools.ffmpeg, "-v", "error", "-i", str(out), "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"])
    samples = array.array("h", raw)
    first = next(i for i, x in enumerate(samples) if abs(x) > 1000) / 8000
    assert .9 < first < 1.1
    before = frame(out, tools, .7)
    during = frame(out, tools, 1.2)
    pixel = (int(height * .2) * width + width // 2) * 3
    assert before[pixel + 1] < 30 and during[pixel + 1] > 200


def test_cancel_and_disk_full(media, tools, tmp_path, monkeypatch):
    candidate = Candidate(media.source_file_id, 0, 5000, 0, "", "")
    token = Cancellation()
    token.cancel()
    with pytest.raises(Cancelled):
        export_clip(media, candidate, [], ExportSettings(), tmp_path, tools, token, lambda *_: None)
    def fail(*_):
        raise OSError("Недостаточно свободного места")
    monkeypatch.setattr("clipnado.export.require_space", fail)
    with pytest.raises(OSError, match="места"):
        export_clip(media, candidate, [], ExportSettings(), tmp_path, tools, Cancellation(), lambda *_: None)
    assert not list(tmp_path.glob("*.mp4"))


def test_export_without_audio_and_reexport(source, tools, tmp_path):
    silent = tmp_path / "silent.mp4"
    subprocess.run([tools.ffmpeg, "-v", "error", "-i", str(source), "-an", "-c:v", "copy", str(silent)], check=True)
    media = probe(silent, tools)
    c = Candidate(media.source_file_id, 0, 1000, 0, "Вручную", "")
    settings = ExportSettings(resolution=720)
    out = export_clip(media, c, [], settings, tmp_path, tools, Cancellation(), lambda *_: None)
    assert not probe(out, tools).has_audio
    c.end_ms = 1200
    second = export_clip(media, c, [], settings, tmp_path, tools, Cancellation(), lambda *_: None)
    assert second != out and out.exists()


def test_missing_subtitles_fail_before_export(media, tools, tmp_path):
    with pytest.raises(ValueError, match="нет текста"):
        export_clip(media, Candidate(media.source_file_id, 0, 1000, 0, "", ""), [], ExportSettings(subtitles="srt"),
                    tmp_path, tools, Cancellation(), lambda *_: None)


def test_default_1080_and_shifted_zone(media, tools, tmp_path):
    settings = ExportSettings(video_y=.1)
    out = export_clip(media, Candidate(media.source_file_id, 0, 1000, 0, "", ""), [], settings, tmp_path, tools, Cancellation(), lambda *_: None)
    info = probe(out, tools)
    assert (info.width, info.height, round(info.fps)) == (1080, 1920, 30)
    image = frame(out, tools)
    assert max(image[:1080 * 180 * 3]) < 5
    assert max(image[1080 * 970 * 3:]) < 5


def test_anamorphic_aspect_is_preserved(tools, tmp_path):
    path = tmp_path / "anamorphic.mp4"
    subprocess.run([tools.ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=320x180:r=25:d=1",
                    "-vf", "setsar=2", "-c:v", "libx264", "-threads", "2", str(path)], check=True)
    media = probe(path, tools)
    out = export_clip(media, Candidate(media.source_file_id, 0, 1000, 0, "", ""), [], ExportSettings(resolution=720),
                      tmp_path, tools, Cancellation(), lambda *_: None)
    image = frame(out, tools)
    rows = [y for y in range(512) if image[(y * 720 + 360) * 3] > 150]
    assert abs(len(rows) - 720 / (32 / 9)) < 6


@pytest.mark.parametrize("aspect,subs", [("9:16", "off"), ("1:1", "burn"), ("16:9", "srt")])
def test_blurred_center_export_geometry_background_and_subtitles(media, tools, tmp_path, aspect, subs):
    settings = ExportSettings(aspect=aspect, resolution=720, layout="blur_center", subtitles=subs, subtitle_y=.7)
    c = Candidate(media.source_file_id, 1000, 4000, 80, "История", "Тест")
    out = export_clip(media, c, [Segment(500, 5000, "Кириллические субтитры по центру.")], settings,
                      tmp_path / "Размытый фон ' тест", tools, Cancellation(), lambda *_: None)
    info = probe(out, tools)
    width, height = settings.dimensions()
    assert (info.width, info.height, round(info.fps)) == (width, height, 30)
    image = frame(out, tools, .2)
    # The red source also fills the surrounding background; no reserved black zone.
    for y in (int(height*.1),height//2,int(height*.9)):
        pixel = (y*width+width//2)*3
        assert image[pixel] > 150 and image[pixel+1] < 50
    if subs == "burn":
        assert sum(1 for r,g,b in zip(image[::3],image[1::3],image[2::3]) if min(r,g,b)>180) > 100
    if subs == "srt": assert out.with_suffix(".srt").exists()
    # Same-source flash remains synchronized in foreground and background.
    during = frame(out, tools, 1.2)
    for y in (int(height*.1), height//2):
        pixel=(y*width+width//2)*3
        assert during[pixel+1]>200


def test_blur_keeps_center_sharp_and_smooths_background(tools, tmp_path):
    path = tmp_path / "checker.mp4"
    subprocess.run([tools.ffmpeg,"-v","error","-f","lavfi","-i",
        "nullsrc=s=320x180:r=30:d=1,geq=r='255*mod(floor(X/8)+floor(Y/8),2)':g='255*mod(floor(X/8)+floor(Y/8),2)':b='255*mod(floor(X/8)+floor(Y/8),2)'",
        "-c:v","libx264","-threads","2","-pix_fmt","yuv420p",str(path)],check=True)
    media = probe(path,tools)
    out=export_clip(media,Candidate(media.source_file_id,0,1000,80,"Юмор","Тест"),[],ExportSettings(resolution=720,layout="blur_center"),tmp_path,tools,Cancellation(),lambda *_:None)
    image=frame(out,tools,.2)
    def contrast(y):
        values=[image[(y*720+x)*3] for x in range(40,680)]
        return sum(abs(values[i]-values[i-1]) for i in range(1,len(values))) / (len(values)-1)
    assert contrast(640)>contrast(100)*4, "Foreground lost sharpness or background was not blurred"


@pytest.mark.parametrize('width,height,sar', [(320, 180, 1), (180, 320, 1), (240, 240, 1), (320, 180, 2)])
def test_tiktok_fill_crops_center_without_stretching_or_borders(tools, tmp_path, width, height, sar):
    path = tmp_path / 'кадр для TikTok.mp4'
    pattern = (f'color=c=lime:s={width}x{height}:r=25:d=1,'
               'drawbox=x=0:y=0:w=iw/5:h=ih:color=red:t=fill,'
               'drawbox=x=iw*4/5:y=0:w=iw/5:h=ih:color=blue:t=fill,'
               'drawbox=x=iw/2-15:y=ih/2-15:w=30:h=30:color=white:t=fill,'
               f'setsar={sar}')
    subprocess.run([tools.ffmpeg, '-v', 'error', '-f', 'lavfi', '-i', pattern,
                    '-c:v', 'libx264', '-threads', '2', '-pix_fmt', 'yuv420p', str(path)], check=True)
    media = probe(path, tools)
    # TikTok is always vertical, including in a queue whose other clips are square.
    settings = ExportSettings(aspect='1:1', resolution=720, layout='tt_fill')
    output = export_clip(media, Candidate(media.source_file_id, 0, 1000, 80, 'Юмор', 'Тест'), [], settings,
                         tmp_path, tools, Cancellation(), lambda *_: None)
    info = probe(output, tools)
    assert (info.width, info.height) == (720, 1280)
    pixels = frame(output, tools, .2)
    def rgb(x, y):
        i = (y * 720 + x) * 3
        return pixels[i:i+3]
    for x, y in ((5, 5), (715, 5), (5, 1275), (715, 1275)):
        assert max(rgb(x, y)) > 150, 'Unexpected added black border'
    xs = [x for x in range(720) if min(rgb(x, 640)) > 200]
    ys = [y for y in range(1280) if min(rgb(360, y)) > 200]
    assert xs and ys and abs(len(xs) / len(ys) - sar) < .1, 'Source display proportions were stretched'
    if width >= height:
        for x in (5, 715):
            assert rgb(x, 100)[1] > 180, 'Center crop included an unrelated side panel'
    else:
        assert rgb(5, 100)[0] > 180 and rgb(715, 100)[2] > 180


@pytest.mark.parametrize('subs', ['off', 'srt', 'burn'])
def test_tiktok_keeps_audio_timing_and_subtitles(media, tools, tmp_path, subs):
    from clipnado.export import composition_frame
    from PySide6.QtGui import QImage
    candidate = Candidate(media.source_file_id, 1000, 4000, 80, 'Юмор', 'Момент', export_layout='tt_fill')
    settings = ExportSettings(aspect='16:9', resolution=720, subtitles=subs, subtitle_y=.8, layout='blur_center')
    out = export_clip(media, candidate, [Segment(500, 5000, 'Качественный клип для TikTok')], settings,
                      tmp_path, tools, Cancellation(), lambda *_: None)
    info = probe(out, tools)
    assert (info.width, info.height, round(info.fps), info.has_audio) == (720, 1280, 30, True)
    pixels = frame(out, tools, .2)
    assert pixels[1] < 30 and pixels[-2] < 30
    whites = sum(min(r, g, b) > 180 for r, g, b in zip(pixels[::3], pixels[1::3], pixels[2::3]))
    assert whites > 100 if subs == 'burn' else whites == 0
    if subs == 'srt':
        text = from_srt(out.with_suffix('.srt').read_text(encoding='utf-8-sig'), 3000)
        assert text[0].start_ms == 0 and text[0].end_ms == 3000
    raw = subprocess.check_output([tools.ffmpeg, '-v', 'error', '-i', str(out), '-vn', '-ac', '1', '-ar', '8000', '-f', 's16le', '-'])
    first = next(i for i, x in enumerate(array.array('h', raw)) if abs(x) > 1000) / 8000
    assert .9 < first < 1.1
    assert frame(out, tools, .7)[1] < 30 and frame(out, tools, 1.2)[1] > 200
    # The export preview uses the same crop geometry, at a reduced display size.
    image = QImage.fromData(composition_frame(media, candidate, settings, tools, Cancellation()))
    assert (image.width(), image.height()) == (360, 640)
    assert image.pixelColor(10, 10).green() > 200  # source flash at start+1 second
    assert settings.layout == 'blur_center' and settings.aspect == '16:9'


@pytest.mark.parametrize('layout', ['unknown-layout', [], 5])
def test_invalid_individual_layout_is_rejected(media, layout):
    candidate=Candidate(media.source_file_id,0,1000,80,'Юмор','Момент',export_layout=layout)
    with pytest.raises(ValueError,match='оформление'):
        candidate.validate(media)


def test_legacy_clips_inherit_saved_common_layout(media, tmp_path):
    import json
    from clipnado.models import Project
    from clipnado.projects import load_project
    from clipnado.export import clip_export_settings
    project=Project(media,candidates=[Candidate(media.source_file_id,0,1000,80,'Юмор','Момент')],export=ExportSettings(layout='blur_center'))
    data=project.to_dict()
    data['candidates'][0].pop('export_layout')
    path=tmp_path/'project017.json';path.write_text(json.dumps(data),encoding='utf-8')
    old=load_project(path)
    assert old.candidates[0].export_layout is None
    assert clip_export_settings(old.export,old.candidates[0]).layout=='blur_center'


@pytest.mark.parametrize('fps', [1, 30])
def test_composition_preview_handles_short_clip_at_source_end(tools, tmp_path, fps):
    from PySide6.QtGui import QImage
    from clipnado.export import composition_frame
    source=tmp_path/'last-frame.mp4'
    subprocess.run([tools.ffmpeg,'-v','error','-f','lavfi','-i',f'color=c=red:s=320x180:r={fps}:d=1',
                    '-c:v','libx264','-threads','2','-pix_fmt','yuv420p',str(source)],check=True)
    media=probe(source,tools)
    candidate=Candidate(media.source_file_id,0,1000,80,'Юмор','Короткий клип',export_layout='tt_fill')
    image=QImage.fromData(composition_frame(media,candidate,ExportSettings(resolution=720),tools,Cancellation()))
    assert (image.width(),image.height())==(360,640)
    assert image.pixelColor(10,10).red()>180
