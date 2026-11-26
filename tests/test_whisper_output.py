"""Native JSON corruption must not erase valid text or completed ASR blocks."""
import json
from pathlib import Path
import sys

import pytest

from clipnado.analysis import recognize
from clipnado.media import Tools
from clipnado.models import MediaInfo
from clipnado.process import Cancellation, Cancelled
from clipnado.whisper_output import read_whisper_output


def response(text, duration=60000):
    return {'transcription':[{'offsets':{'from':0,'to':duration},'text':text}]}


@pytest.mark.parametrize('bom',[b'',b'\xef\xbb\xbf'],ids=['plain','bom'])
def test_valid_whisper_utf8_and_bom_are_preserved(tmp_path,bom):
    expected=response('Русская речь: смех 😂 и злость.')
    path=tmp_path/'speech.json';path.write_bytes(bom+json.dumps(expected,ensure_ascii=False).encode('utf-8'))
    diagnostic=tmp_path/'raw.dat'
    actual,warnings=read_whisper_output(path,diagnostic,0,60000)
    assert actual==expected and not warnings and not diagnostic.exists()


def test_invalid_byte_b3_at_3748_keeps_valid_russian_and_reports_interval(tmp_path):
    body={'padding':'',**response('До сбоя BAD_UTF8 после сбоя.')}
    first=json.dumps(body,ensure_ascii=False).encode('utf-8')
    body['padding']='x'*(3748-first.index(b'BAD_UTF8'))
    raw=json.dumps(body,ensure_ascii=False).encode('utf-8').replace(b'BAD_UTF8',b'\xb3')
    with pytest.raises(UnicodeDecodeError) as original:
        raw.decode('utf-8')
    assert original.value.start==3748 and raw[3748]==0xb3
    path=tmp_path/'speech.json';path.write_bytes(raw)
    diagnostic=tmp_path/'диагностика'/'raw.dat'
    data,warnings=read_whisper_output(path,diagnostic,60000,120000)
    assert data['transcription'][0]['text']=='До сбоя � после сбоя.'
    assert len(warnings)==1 and '00:01:00.000–00:02:00.000' in warnings[0]
    assert str(diagnostic) in warnings[0] and diagnostic.read_bytes()==raw


@pytest.mark.parametrize('raw',[b'{"transcription": [',b'{"transcription": \xb3}'],ids=['truncated','invalid-json-token'])
def test_malformed_whisper_json_is_not_silently_accepted(tmp_path,raw):
    path=tmp_path/'speech.json';path.write_bytes(raw)
    diagnostic=tmp_path/'raw.dat'
    with pytest.raises(RuntimeError,match='Whisper:') as error:
        read_whisper_output(path,diagnostic,60000,120000)
    assert '00:01:00.000–00:02:00.000' in str(error.value)
    assert 'кэше' in str(error.value) and diagnostic.read_bytes()==raw


def test_utf8_recovery_and_warnings_survive_cancel_resume_and_cache(tmp_path,monkeypatch):
    model=tmp_path/'base.bin';model.write_bytes(b'fixture')
    tools=Tools(whisper=sys.executable,model=str(model))
    media=MediaInfo('long.mp4','utf8-checkpoint',180000,1920,1080,24,True)
    cache=tmp_path/'Кэш с пробелом';extracted=[]
    def extract(_media,start,duration,path,*_args,**_kwargs):
        extracted.append((start,duration));Path(path).touch()
    monkeypatch.setattr('clipnado.analysis.extract_audio',extract)
    monkeypatch.setattr('clipnado.analysis.audio_energy',lambda _path,offset,_token:[(offset+1500,.1)])
    def engine(_args,_token,cwd,**_kwargs):
        start,duration=extracted[-1]
        text='Второй блок BAD_UTF8 после ошибки.' if start==58500 else 'Цельный текст блока '+str(start)
        raw=json.dumps(response(text,duration),ensure_ascii=False).encode('utf-8').replace(b'BAD_UTF8',b'\xb3')
        (Path(cwd)/'speech.json').write_bytes(raw)
        return ''
    monkeypatch.setattr('clipnado.analysis.run',engine)
    token=Cancellation()
    def progress(_value,text):
        if text.startswith('Сохранено 00:02:00'):
            token.cancel()
    with pytest.raises(Cancelled):
        recognize(media,tools,cache,token,progress)
    saved=json.loads(next(cache.glob('*.checkpoint.json')).read_text(encoding='utf-8'))
    assert saved['next_start_ms']==120000 and len(saved['warnings'])==1
    assert saved['segments'][1]['text']=='Второй блок � после ошибки.'
    assert [s for s,_ in extracted]==[0,58500]
    result=recognize(media,tools,cache,Cancellation(),lambda *_:None)
    assert [s for s,_ in extracted]==[0,58500,118500]
    assert len(result[0])==3 and result[0][-1].end_ms==180000
    assert result[2]==saved['warnings'] and not list(cache.glob('*.checkpoint.json'))
    cached=recognize(media,tools,cache,Cancellation(),lambda *_:pytest.fail('Cached ASR recomputed'))
    assert cached[0]==result[0] and cached[2]==result[2]


def test_broken_json_does_not_advance_checkpoint_and_only_failed_block_repeats(tmp_path,monkeypatch):
    model=tmp_path/'base.bin';model.write_bytes(b'fixture')
    tools=Tools(whisper=sys.executable,model=str(model))
    media=MediaInfo('long.mp4','broken-json-checkpoint',120000,1920,1080,24,True)
    cache=tmp_path/'cache';extracted=[];broken=[True]
    def extract(_media,start,duration,path,*_args,**_kwargs):
        extracted.append((start,duration));Path(path).touch()
    monkeypatch.setattr('clipnado.analysis.extract_audio',extract)
    monkeypatch.setattr('clipnado.analysis.audio_energy',lambda _path,offset,_token:[(offset+1500,.1)])
    def engine(_args,_token,cwd,**_kwargs):
        start,duration=extracted[-1]
        raw=(b'{"transcription": [' if start and broken[0]
             else json.dumps(response('Сохранённая речь '+str(start),duration),ensure_ascii=False).encode('utf-8'))
        (Path(cwd)/'speech.json').write_bytes(raw)
        return ''
    monkeypatch.setattr('clipnado.analysis.run',engine)
    with pytest.raises(RuntimeError,match='Whisper:'):
        recognize(media,tools,cache,Cancellation(),lambda *_:None)
    saved=json.loads(next(cache.glob('*.checkpoint.json')).read_text(encoding='utf-8'))
    assert saved['next_start_ms']==60000 and len(saved['segments'])==1
    assert len(list(cache.glob('*.json')))==1, 'Only the completed-block checkpoint may exist'
    broken[0]=False
    result=recognize(media,tools,cache,Cancellation(),lambda *_:None)
    assert [start for start,_ in extracted]==[0,58500,58500]
    assert len(result[0])==2 and result[0][-1].end_ms==120000 and not result[2]
