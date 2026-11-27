import copy
from pathlib import Path

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QDialog

from clipnado.clip_editor import ClipEditorDialog, MANUAL_NOTE, edited_candidate
from clipnado.media import probe
from clipnado.models import Candidate, MediaInfo, Segment
from clipnado.projects import load_project, save_project
from test_gui import app, window, wait_task


def test_manual_bounds_accept_example_outside_search_limits_and_reject_invalid():
    media = MediaInfo('source','manual',3600000,1920,1080,30,True)
    original = Candidate(media.source_file_id,504000,536000,88,'Юмор','Хук',selected=True,export_layout='gaming_split')
    old = copy.deepcopy(original)
    segments = [Segment(500000,504000,'Контекст'),Segment(504000,536000,'Событие'),Segment(536000,539000,'Реакция')]
    revised = edited_candidate(media,segments,original,500000,539000)
    assert original == old and revised.id==original.id and revised.selected and revised.export_layout=='gaming_split'
    assert revised.transcript_excerpt=='Контекст Событие Реакция' and MANUAL_NOTE in revised.reason
    # Manual choices need only fit the source, even longer than automatic max90.
    assert edited_candidate(media,segments,original,0,200000).end_ms==200000
    for start,end in ((-1,1000),(1000,1000),(2000,1000),(0,3600001)):
        with pytest.raises(ValueError): edited_candidate(media,segments,original,start,end)
    assert original==old


def test_editor_cancel_reset_and_invalid_draft_never_change_candidate(app,window,media):
    window.media_loaded(media)
    c = Candidate(media.source_file_id,2000,4000,80,'Юмор','Выбор',selected=True)
    old=copy.deepcopy(c)
    d=ClipEditorDialog(window,media,c,window.player)
    d.start.setText('00:01');d.end.setText('00:07')
    assert d.bounds()==(1000,7000)
    d.end.setText('00:00');d.accept()
    assert not d.apply_button.isEnabled() and d.result()!=QDialog.Accepted
    d.reset();assert d.bounds()==(2000,4000)
    d.reject();d.deleteLater()
    assert c==old


def test_editor_real_playback_apply_export_roundtrip_and_undo(app,window,media,tmp_path):
    window.media_loaded(media)
    c=Candidate(media.source_file_id,2000,4000,80,'Юмор','Автовыбор',selected=True,export_layout='center_black')
    original=copy.deepcopy(c)
    window.project.candidates=[c]
    window.project.segments=[Segment(1000,7000,'Контекст, событие и реакция.')]
    window.refresh_candidates();window.table.selectRow(0)
    state={'pixels':False,'applied':False,'ticks':0}
    timer=QTimer(window)
    def edit():
        d=next((d for d in window.findChildren(ClipEditorDialog) if d.isVisible()),None)
        if d is None:return
        state['ticks']+=1
        if state['ticks']==1:d.play_context()
        if d.video.frame_count and not d.video.image.isNull() and d.player.position()>200:
            state['pixels']=True
            d.player.pause();d.seek(1000)
            d.start.setText('00:01');d.end.setText('00:07')
            assert d.bounds()==(1000,7000)
            state['applied']=True;timer.stop();d.accept()
        elif state['ticks']>300:
            timer.stop();d.reject()
    timer.timeout.connect(edit);timer.start(30)
    window.edit_selected_clip();timer.stop();timer.deleteLater()
    assert state['pixels'] and state['applied'],state
    assert (c.start_ms,c.end_ms)==(1000,7000) and c.id==original.id
    assert c.selected and c.export_layout=='center_black' and window.boundary_undo
    saved=tmp_path/'edited.json';save_project(window.project,saved)
    assert load_project(saved).candidates[0]==c
    window.resolution.setCurrentText('720');window.export_folder.setText(str(tmp_path/'export'))
    window.export_selected();wait_task(app,window)
    output=next((tmp_path/'export').glob('*.mp4'))
    assert abs(probe(output,window.tools).duration_ms-6000)<200
    window.undo_boundaries()
    assert c==original and not window.boundary_undo


def test_editor_repeated_play_apply_cancel_reuses_source_player(app,window,media):
    window.media_loaded(media)
    c=Candidate(media.source_file_id,2000,4000,80,'Юмор','Автовыбор',selected=True)
    window.project.candidates=[c];window.refresh_candidates();window.table.selectRow(0)
    old=copy.deepcopy(c)
    for iteration in range(5):
        ticks=[0];pixels=[False];timer=QTimer(window)
        def close_editor():
            d=next((d for d in window.findChildren(ClipEditorDialog) if d.isVisible()),None)
            if d is None:return
            assert d.player is window.player and d.sink is window.video_sink
            ticks[0]+=1
            if d.video.frame_count and d.player.position()>200:
                pixels[0]=True;timer.stop()
                d.accept() if iteration%2 else d.reject()
            elif ticks[0]>400:
                timer.stop();d.reject()
        timer.timeout.connect(close_editor);timer.start(25)
        window.edit_selected_clip();timer.stop();timer.deleteLater()
        assert pixels[0],(iteration,ticks[0])
        assert c==old and not window.boundary_undo
