"""Verify the actual embedded Windows runtime; stream progress and kill timed-out trees."""

import argparse
import os
from pathlib import Path
import sys


def main():
    # The CI host can default to cp1252 even though child output is UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--semantic-model", required=True, type=Path)
    args = parser.parse_args()
    root = args.folder.resolve()
    from finalize_windows_release import SELECTION_BASELINE_SHA256, digest
    for name, expected in SELECTION_BASELINE_SHA256.items():
        if digest(root / 'clipnado' / name) != expected:
            raise RuntimeError('Packaged selection differs from 0.1.15: ' + name)
    python = root / "runtime/python.exe"
    # This process needs only the standard-library runner, not source-environment Qt.
    sys.path.insert(0, str(root))
    from clipnado.process import run
    import tempfile

    with tempfile.TemporaryDirectory(prefix="clipnado-portable-smoke-") as tmp:
        os.environ.update(
            QT_QPA_PLATFORM="offscreen",
            QT_FFMPEG_DECODING_HW_DEVICE_TYPES=",",
            PYTHONUTF8="1",
            CLIPNADO_DATA_DIR=str(Path(tmp) / "settings"),
            CLIPNADO_QWEN_MODEL=str(args.semantic_model.resolve()),
        )

        def execute(code, timeout, extra=()):
            return run(
                [str(python), "-u", "-c", code, *map(str, extra)],
                cwd=root,
                timeout=timeout,
                on_output=lambda chunk, elapsed: print(chunk, end="", flush=True),
            )

        diagnostics = run(
            [str(python), "-u", str(root / "scripts/diagnostics.py")],
            cwd=root,
            timeout=90,
            on_output=lambda chunk, elapsed: print(chunk, end="", flush=True),
        )
        if "ERROR" in diagnostics or "MISSING" in diagnostics:
            raise RuntimeError("Packaged diagnostics failed")
        # A separate process separates native-model startup from the GUI deadline.
        execute(
            r"""
import os
from pathlib import Path
from clipnado.process import run
print('PACKAGED_QWEN_TOKEN_START',flush=True)
model=os.environ['CLIPNADO_QWEN_MODEL']
assert Path(model).is_file()
result=run([str(Path('tools/llama-cli.exe').resolve()),'-m',model,'-p','Привет','-n','1','-c','256',
            '-b','32','-t','2','-ngl','0','--conversation','--single-turn','--simple-io',
            '--no-display-prompt','--no-warmup','--offline'],timeout=180)
assert 'eval time' in result
print('PACKAGED_ACTUAL_QWEN_TOKEN_OK',flush=True)
""",
            210,
        )
        code = r"""
import json,sys,socket,traceback,faulthandler
from pathlib import Path
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication,QMessageBox,QCheckBox
from clipnado.app import MainWindow,configure_application
from clipnado.media import probe
from clipnado.models import Candidate,Segment
from clipnado.process import Cancellation,run
from clipnado.gaming import GamingRegionsDialog,source_frame
from scripts.gaming_smoke import CAMERA,GAME,make_fixture
import clipnado.local_llm as llm
from clipnado.whisper_output import read_whisper_output
utf8_folder=Path(sys.argv[1])/'utf8-check';utf8_folder.mkdir(parents=True,exist_ok=True)
utf8_raw=json.dumps({'transcription':[{'offsets':{'from':0,'to':60000},'text':'До BAD_UTF8 после'}]},ensure_ascii=False).encode('utf-8').replace(b'BAD_UTF8',b'\xb3')
utf8_path=utf8_folder/'speech.json';utf8_path.write_bytes(utf8_raw)
utf8_diag=utf8_folder/'original.raw'
utf8_data,utf8_warnings=read_whisper_output(utf8_path,utf8_diag,60000,120000)
assert utf8_data['transcription'][0]['text']=='До � после' and utf8_warnings
assert utf8_diag.read_bytes()==utf8_raw
print('PACKAGED_WHISPER_UTF8_RECOVERY_OK',flush=True)
faulthandler.enable()
faulthandler.dump_traceback_later(90,repeat=True)
print('PACKAGED_GUI_IMPORTS_OK',flush=True)
app=QApplication([])
configure_application(app)
print('PACKAGED_UI_CONFIGURED_OK',flush=True)
def uncaught(kind,value,tb):
    traceback.print_exception(kind,value,tb)
    app.exit(2)
sys.excepthook=uncaught
QMessageBox.information=lambda *_:QMessageBox.Ok
QMessageBox.question=lambda *_:QMessageBox.Yes
print('PACKAGED_WINDOW_CREATE_START',flush=True)
w=MainWindow()
w.tools.acceleration='cpu'
w.show_error=lambda text:(print('PACKAGED_ERROR',text,flush=True),app.exit(2))
assert not Path('clipnado/animations.py').exists() and not Path('clipnado/animation_render.py').exists()
assert not Path('clipnado/highlights.py').exists()
assert not hasattr(w,'key') and not hasattr(w,'tiktok_tracking') and not hasattr(w,'animation_mode')
assert Path(w.tools.llama_vulkan).is_file() and Path(w.tools.whisper_vulkan).is_file()
print('PACKAGED_WINDOW_CREATED',flush=True)
def no_network(*_a,**_kw):raise RuntimeError('Unexpected network connection')
socket.socket.connect=no_network;socket.create_connection=no_network
w.media_loaded(probe(Path('demo/demo.mp4'),w.tools));w.show();w.play_candidate()
assert w.project.search.profile=='mixed' and w.minimum.value()==15 and w.search_settings().max_seconds==90
assert w.background_dim.value()==40
print('PACKAGED_VIDEO_PLAY_START',flush=True)
stage=[0];original_clip_ids=[];playback_frames=[0];timer=QTimer()
def protocol_fixture(args,*_a,**kw):
    if args[args.index('-n')+1]=='0':return 'main: number of tokens in prompt = 1000'
    item={'clip_id':0,'hook':20,'event':20,'emotion':20,'completeness':20,'category':'Юмор','reason':'Protocol fixture'}
    if 'Это поиск по содержанию' in (Path(kw['cwd'])/'system.txt').read_text(encoding='utf-8'):item['relevance']=90
    return json.dumps({'clips':[item]})
def check():
    if stage[0]==0:
        if w.video.frame_count<1 or w.player.position()<100 or w.video.image.isNull():return
        print('PACKAGED_VIDEO_PIXELS_OK',flush=True)
        print('PACKAGED_PAUSE_START',flush=True)
        w.player.pause();print('PACKAGED_PAUSE_OK',flush=True)
        w.player.setPosition(0);print('PACKAGED_SEEK_OK',flush=True)
        playback_frames[0]=w.video.frame_count;stage[0]=-1;w.player.play()
    elif stage[0]==-1:
        if w.player.position()<100 or w.video.frame_count<=playback_frames[0]:return
        print('PACKAGED_RESUME_OK',flush=True)
        w.player.pause()
        w.minimum.setValue(2)
        w.project.candidates=[Candidate(w.project.media.source_file_id,0,2000,80,'Юмор','Portable export',selected=True) for _ in range(4)]
        w.refresh_candidates();w.export_clips.cellWidget(1,3).setCurrentIndex(1);w.export_clips.cellWidget(2,3).setCurrentIndex(2)
        w.export_clips.cellWidget(3,3).setCurrentIndex(3)
        w.gaming_camera=CAMERA.copy();w.gaming_game=GAME.copy()
        w.resolution.setCurrentText('720');w.aspect.setCurrentText('1:1');w.export_folder.setText(sys.argv[1])
        stage[0]=1;w.export_selected()
    elif stage[0]==1 and w.task_thread is None:
        files=sorted(Path(sys.argv[1]).glob('*.mp4'));assert len(files)==4
        infos=[probe(p,w.tools) for p in files]
        assert [(m.width,m.height) for m in infos]==[(720,720),(720,720),(720,1280),(720,720)]
        assert [c.export_layout for c in w.project.candidates]==[None,'blur_center','gaming_split','center_black']
        image=QImage.fromData(source_frame(infos[3],400,w.tools,Cancellation()))
        assert image.pixelColor(image.width()//2,0).value()<=5
        assert image.pixelColor(image.width()//2,image.height()-1).value()<=5
        print('PACKAGED_CENTER_BLACK_EXPORT_OK',flush=True)
        print('PACKAGED_MIXED_GAMING_EXPORT_OK',flush=True)
        long_source=Path(sys.argv[1])/'long-demo.mp4'
        run([w.tools.ffmpeg,'-v','error','-y','-stream_loop','-1','-i',str(Path('demo/demo.mp4').resolve()),'-t','60','-c','copy',str(long_source)])
        w.media_loaded(probe(long_source,w.tools))
        w.gaming_camera=CAMERA.copy();w.gaming_game=GAME.copy()
        w.project.segments=[Segment(0,30000,'Вопрос'),Segment(30000,60000,'Ответ')]
        llm.run=protocol_fixture;stage[0]=2;w.analyze()
    elif stage[0]==2 and w.task_thread is None:
        assert w.project.analysis_info['network_used'] is False
        assert w.project.candidates and w.project.candidates[0].provider.startswith('local-qwen:')
        assert len(w.project.candidates)==2 and w.project.analysis_info['recovery_windows']==1
        assert w.project.search.profile=='mixed' and w.project.analysis_info['mode']=='local-qwen'
        print('PACKAGED_015_SELECTION_OK',flush=True)
        original_clip_ids[:]=[c.id for c in w.project.candidates]
        print('PACKAGED_LOCAL_WORKFLOW_OK',flush=True)
        stage[0]=3;w.content_query.setText('необычный ответ');w.find_by_content()
    elif stage[0]==3 and w.task_thread is None:
        assert w.project.last_query=='необычный ответ' and w.project.query_results[0].match_score==90
        assert [c.id for c in w.project.candidates]==original_clip_ids and w.assistant_dialog
        w.assistant_dialog.apply_selected();assert [c.id for c in w.project.candidates]==original_clip_ids
        print('PACKAGED_CONTENT_QUERY_PROTOCOL_OK',flush=True)
        w.table.selectRow(0);stage[0]=4;w.refine_selected()
    elif stage[0]==4 and w.task_thread is None:
        assert w.assistant_dialog and [c.id for c in w.project.candidates]==original_clip_ids
        assert (w.project.candidates[0].start_ms,w.project.candidates[0].end_ms)==(0,30000)
        w.close_assistant_dialog();print('PACKAGED_BOUNDARY_REVIEW_PROTOCOL_OK',flush=True)
        w.select_all(True);w.export_clips.cellWidget(0,3).setCurrentIndex(2)
        stage[0]=5;w.preview_export_clip()
    elif stage[0]==5 and w.task_thread is None:
        from clipnado.preview import VideoPreview
        assert w.export_preview_dialog
        preview=w.export_preview_dialog.findChild(VideoPreview)
        image=preview.image.copy()
        assert (image.width(),image.height())==(360,640)
        guides=w.export_preview_dialog.findChild(QCheckBox,'tiktok_guides')
        assert guides.isChecked() and preview.guides_enabled
        with_guides=preview.pixmap().toImage()
        guides.setChecked(False)
        assert not preview.guides_enabled and preview.pixmap().toImage()!=with_guides
        guides.setChecked(True)
        assert preview.guides_enabled and preview.image==image
        print('PACKAGED_TIKTOK_GUIDES_OK',flush=True)
        w.close_export_preview();print('PACKAGED_EXPORT_PREVIEW_OK',flush=True)
        path=make_fixture(Path(sys.argv[1])/'game-stream.mp4',w.tools.ffmpeg)
        w.media_loaded(probe(path,w.tools))
        w.project.candidates=[Candidate(w.project.media.source_file_id,1000,4000,80,'Реакция','Игровой момент',selected=True,export_layout='gaming_split')]
        w.refresh_candidates();w.resolution.setCurrentText('720');w.export_folder.setText(str(Path(sys.argv[1])/'gaming'))
        timer.stop()
        editor_timer=QTimer(w)
        def configure():
            dialogs=[d for d in w.findChildren(GamingRegionsDialog) if not d.closed]
            if not dialogs or dialogs[-1].worker is not None or dialogs[-1].canvas.image.isNull():return
            d=dialogs[-1];d.canvas.camera=CAMERA.copy();d.canvas.game=GAME.copy();d.fraction.setValue(40)
            editor_timer.stop();d.accept()
        editor_timer.timeout.connect(configure);editor_timer.start(30)
        w.configure_gaming();editor_timer.deleteLater()
        assert w.project.export.gaming_camera==CAMERA
        print('PACKAGED_GAMING_EDITOR_OK',flush=True)
        stage[0]=6;w.export_selected();timer.start(50)
    elif stage[0]==6 and w.task_thread is None:
        files=list((Path(sys.argv[1])/'gaming').glob('*.mp4'));assert len(files)==1
        info=probe(files[0],w.tools);assert (info.width,info.height,info.has_audio)==(720,1280,True)
        image=QImage.fromData(source_frame(info,400,w.tools,Cancellation()))
        top=image.pixelColor(image.width()//2,image.height()//5)
        bottom=image.pixelColor(image.width()//2,image.height()*4//5)
        assert top.red()>180 and top.blue()<40 and bottom.blue()>180 and bottom.red()<40
        print('PACKAGED_GAMING_PIXELS_OK',flush=True)
        w.project.candidates[0].export_layout='top_black'
        w.project.segments=[Segment(0,w.project.media.duration_ms,'Проверяем крупные субтитры на русском языке и свободную нижнюю часть.')]
        w.refresh_candidates();w.color_mode.setCurrentIndex(1);w.normalize_audio.setChecked(True)
        assert not hasattr(w, "subtitle_mode") and not hasattr(w, "readable_subtitles")
        w.export_folder.setText(str(Path(sys.argv[1])/'comfort'))
        stage[0]=7;w.export_selected()
    elif stage[0]==7 and w.task_thread is None:
        files=list((Path(sys.argv[1])/'comfort').glob('*.mp4'));assert len(files)==1
        info=probe(files[0],w.tools);assert (info.width,info.height,info.has_audio)==(720,1280,True)
        image=QImage.fromData(source_frame(info,400,w.tools,Cancellation()))
        values=[image.pixelColor(x,y).value() for x in range(0,image.width(),5) for y in range(image.height()//2,image.height(),5)]
        assert max(values)<=5
        white = [image.pixelColor(x,y) for x in range(5,image.width(),10) for y in range(5,image.height()//3,10)]
        assert not any(min(p.red(),p.green(),p.blue())>180 for p in white)
        streams=json.loads(run([w.tools.ffprobe,'-v','error','-show_streams','-of','json',str(files[0])]))['streams']
        assert not any(s['codec_type']=='subtitle' for s in streams)
        from clipnado.projects import save_project,load_project
        save_project(w.project,Path(sys.argv[1])/'comfort.json')
        saved=load_project(Path(sys.argv[1])/'comfort.json')
        assert saved.export.normalize_audio and saved.export.color=='soft'
        print('PACKAGED_COMFORT_EXPORT_OK',flush=True)
        stage[0]=8;w.preview_export_clip()
    elif stage[0]==8 and w.task_thread is None:
        assert w.export_preview_dialog
        print('PACKAGED_CLEAN_PREVIEW_OK',flush=True)
        w.close_export_preview()
        identities=[c.id for c in w.project.candidates]
        w.search_profile.setCurrentIndex(w.search_profile.findData('quote'))
        assert w.minimum.value()==15 and w.search_settings().max_seconds==60 and w.project.search.profile=='quote'
        assert [c.id for c in w.project.candidates]==identities
        w.export_preset.setCurrentIndex(w.export_preset.findData('dialogue'));w.apply_export_preset()
        assert w.project.candidates[0].export_layout=='top_black'
        assert w.project.export.preset=='dialogue' and w.project.export.normalize_audio
        assert w.background_dim.value()==40 and w.project.export.background_dim==.4
        w.apply_layout_to_selected()
        w.export_folder.setText(str(Path(sys.argv[1])/'profiles'))
        stage[0]=9;w.export_selected()
    elif stage[0]==9 and w.task_thread is None:
        files=list((Path(sys.argv[1])/'profiles').glob('*.mp4'));assert len(files)==1
        info=probe(files[0],w.tools);assert (info.width,info.height,round(info.fps))==(1080,1920,30)
        from clipnado.projects import save_project,load_project
        save_project(w.project,Path(sys.argv[1])/'profiles.json')
        saved=load_project(Path(sys.argv[1])/'profiles.json')
        assert saved.search.profile=='quote' and saved.export.preset=='dialogue'
        assert 'subtitles' not in saved.to_dict()['export']
        assert not list((Path(sys.argv[1])/'comfort').glob('*.srt'))
        w.resolution.setCurrentText('720');assert w.export_settings().preset=='custom'
        print('PACKAGED_MEME_PROFILES_EXPORT_OK',flush=True)
        old=saved.to_dict();old['search'].update(profile='highlights',min_seconds=10,max_seconds=90)
        path=Path(sys.argv[1])/'project-016.json';path.write_text(json.dumps(old),encoding='utf-8')
        migrated=load_project(path)
        assert migrated.search.profile=='mixed' and migrated.search.min_seconds==15
        assert migrated.candidates==saved.candidates and migrated.segments==saved.segments
        assert migrated.export==saved.export
        print('PACKAGED_016_PROJECT_MIGRATION_OK',flush=True)
        timer.stop();w.dirty=False;w.close();app.quit()
def deadline():
    print('PACKAGED_GUI_DEADLINE',stage[0],w.player.position(),w.video.frame_count,w.player.errorString(),flush=True)
    if w.token:w.token.cancel()
    for worker,token in w.poster_jobs:token.cancel()
    app.exit(3)
timer.timeout.connect(check);timer.start(50)
QTimer.singleShot(150000,deadline)
result=app.exec();faulthandler.cancel_dump_traceback_later();sys.exit(result)
"""
        output = execute(code, 210, [Path(tmp) / "export"])
        required = (
            "PACKAGED_WHISPER_UTF8_RECOVERY_OK",
            "PACKAGED_VIDEO_PIXELS_OK",
            "PACKAGED_PAUSE_OK",
            "PACKAGED_SEEK_OK",
            "PACKAGED_RESUME_OK",
            "PACKAGED_UI_CONFIGURED_OK",
            "PACKAGED_TIKTOK_GUIDES_OK",
            "PACKAGED_CENTER_BLACK_EXPORT_OK",
            "PACKAGED_MIXED_GAMING_EXPORT_OK",
            "PACKAGED_LOCAL_WORKFLOW_OK",
            "PACKAGED_015_SELECTION_OK",
            "PACKAGED_016_PROJECT_MIGRATION_OK",
            "PACKAGED_CONTENT_QUERY_PROTOCOL_OK",
            "PACKAGED_BOUNDARY_REVIEW_PROTOCOL_OK",
            "PACKAGED_EXPORT_PREVIEW_OK",
            "PACKAGED_GAMING_EDITOR_OK",
            "PACKAGED_GAMING_PIXELS_OK",
            "PACKAGED_COMFORT_EXPORT_OK",
            "PACKAGED_CLEAN_PREVIEW_OK",
            "PACKAGED_MEME_PROFILES_EXPORT_OK",
        )
        if not all(tag in output for tag in required):
            raise RuntimeError(
                "Portable GUI verification did not reach all required stages"
            )
        print("PORTABLE_WINDOWS_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
