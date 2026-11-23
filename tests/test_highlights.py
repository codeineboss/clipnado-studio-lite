"""Editorial invariants, evidence validation and resumable two-pass orchestration."""
from dataclasses import asdict, replace
import sys

import pytest

from clipnado.highlights import (Event, audio_hints, discovery_request, opening_text,
                                model_event, quote_reviews, review_request, reviewed_candidates, validate_event)
from clipnado.local_llm import LocalModelProvider, PromptTooLong
from clipnado.media import Tools
from clipnado.models import MediaInfo, SearchSettings, Segment
from clipnado.process import Cancellation, Cancelled
from clipnado.profiles import search_profile


@pytest.fixture
def speech():
    return [Segment(0,4000,'Сейчас покажу идеальный прыжок, я профессионал.'),
            Segment(4000,8000,'Блин, я опять упал в эту яму!'),
            Segment(8000,12000,'Ахаха, ну ты и чёртов профессионал!'),
            Segment(12000,18000,'Ладно, теперь обсудим настройки.'),
            Segment(18000,26000,'Никаких происшествий, просто обсуждаем обычные дела.')]


@pytest.fixture
def settings():
    return search_profile(SearchSettings(min_seconds=10,max_seconds=90,count=6,target_count=2), 'highlights')


@pytest.fixture
def event():
    return Event(0,1,3,'Сейчас покажу идеальный прыжок','я опять упал в эту яму',
                 'теперь обсудим настройки','Юмор',95)


def verdict(event, **changes):
    value = dict(id=0, first=event.first, last=2, opening=True, meaningful=True,
                 clean_end=True, standalone=True, needs_visual=False, long_context=False,
                 score=85, hook_quote=event.hook_quote, end_quote='чёртов профессионал',
                 reason='Обещание сразу, провал и сильная реакция. Хвост удалён.')
    value.update(changes)
    return value


def media_for(speech):
    return MediaInfo('authored.mp4','authored-hook-tests',speech[-1].end_ms,1920,1080,30,True)


@pytest.mark.parametrize('flag,value', [('opening',False),('meaningful',False),('clean_end',False),
                                      ('standalone',False),('needs_visual',True)])
def test_high_score_cannot_compensate_editorial_failure(speech, settings, event, flag, value):
    good, bad = reviewed_candidates([verdict(event, score=100, **{flag:value})], [event], media_for(speech), speech, settings, 'model')
    assert not good and flag in bad[0]['failed']


def test_event_can_end_at_peak_without_additional_outcome(settings):
    speech = [Segment(0,10000,'Сейчас покажу вам самый смешной способ проиграть!'),
              Segment(10000,20000,'Блядь, я проиграл самому себе!'),
              Segment(20000,30000,'А теперь обычные настройки и лишняя болтовня.')]
    event = Event(0,1,2,'Сейчас покажу','я проиграл самому себе','лишняя болтовня','Реакция',95)
    good, bad = reviewed_candidates([verdict(event,last=1,hook_quote='Сейчас покажу',
                                           end_quote='я проиграл самому себе')],
                                     [event],media_for(speech),speech,settings,'model')
    assert not bad and len(good)==1
    assert (good[0].start_ms,good[0].end_ms)==(0,20000)
    assert 'Блядь' in good[0].transcript_excerpt


def test_review_trims_tail_preserves_source_quotes_and_short_meme(speech,settings,event):
    good,bad=reviewed_candidates([verdict(event)],[event],media_for(speech),speech,settings,'model')
    assert not bad and len(good)==1
    assert good[0].end_ms==12000 and good[0].category=='Юмор'
    assert 'настройки' not in good[0].transcript_excerpt
    assert 'чёртов профессионал' in good[0].reason


@pytest.mark.parametrize('long_context', [False,True])
def test_long_gaming_reaction_requires_context_but_not_story_category(settings,long_context):
    speech=[Segment(0,4000,'Сейчас останусь один против пятерых, смотри!'),
            Segment(4000,62000,'Патроны кончаются, ещё один противник, осталось добраться до выхода.'),
            Segment(62000,70000,'Да блядь, я всё-таки победил!')]
    event=Event(0,2,2,'Сейчас останусь один против пятерых','я всё-таки победил','я всё-таки победил','Реакция',95)
    good,bad=reviewed_candidates([verdict(event,last=2,hook_quote=event.hook_quote,
                                         end_quote=event.end_quote,long_context=long_context)],
                                  [event],media_for(speech),speech,settings,'model')
    assert bool(good)==long_context
    if good:assert good[0].end_ms==70000 and good[0].category=='Реакция'
    else:assert 'long_context' in bad[0]['failed']


@pytest.mark.parametrize('change', [dict(hook_quote='я профессионал'),dict(event_quote='случилось невероятное'),
                                  dict(end_quote='теперь обсудим'),dict(first=2),dict(peak=99)])
def test_discovery_rejects_invented_quotes_late_hooks_and_wrong_rows(speech,settings,event,change):
    with pytest.raises(ValueError):
        validate_event({**asdict(event),**change},speech,settings,0,len(speech))


def test_opening_estimate_never_uses_later_words():
    speech=[Segment(0,10000,'один два три четыре пять шесть семь восемь девять десять')]
    text,approximate=opening_text(speech,0,0)
    assert approximate and text=='один два три четыре'


def test_critic_is_blind_to_discovery_rank(speech,settings,event):
    a=review_request(speech,settings,[replace(event,rank=1)])
    b=review_request(speech,settings,[replace(event,rank=100)])
    assert a.system==b.system and a.prompt==b.prompt and a.schema==b.schema
    assert 'rank' not in a.prompt


def test_discovery_can_start_at_any_source_row_and_considers_audio_peaks(speech,settings):
    request=discovery_request(media_for(speech),speech,settings,0,len(speech),[(0,.01),(1000,.01),(5000,.3),(6000,.01)],3)
    fields=request.schema['properties']['events']['items']['properties']
    assert fields['first']['enum']==[f'R{i}' for i in range(len(speech))]
    assert request.response_field=='events' and not request.proposals
    assert audio_hints(speech,[(0,.01),(1000,.01),(5000,.3),(6000,.01)],0,len(speech))
    assert 'Всплески громкости' in request.prompt and 'НЕ доказательство' in request.prompt
    assert 'Мат' in request.system and 'оскорбления' in request.system


def test_no_verdict_is_not_ready_for_export(speech,settings,event):
    good,bad=reviewed_candidates([], [event],media_for(speech),speech,settings,'model')
    assert not good and bad[0]['failed']==['no_verdict']


def test_two_pass_cache_resume_and_stable_clip_ids(speech,settings,event,tmp_path,monkeypatch):
    model=tmp_path/'model.gguf';model.write_bytes(b'GGUF-test')
    provider=LocalModelProvider(Tools(llama=sys.executable,semantic_model=str(model),acceleration='cpu'))
    calls=[]
    def window(request,staged,token,progress,stats,decode):
        calls.append(request.response_field)
        rows=[asdict(event)] if request.response_field=='events' else [verdict(event)]
        return decode(rows),1000,[],None
    monkeypatch.setattr(provider,'_window',window)
    first=provider.analyze(media_for(speech),speech,settings,tmp_path/'cache',Cancellation(),lambda *_:None)
    assert first.info['complete'] and calls==['events','reviews']
    assert len(first.candidates)==1 and first.candidates[0].end_ms==12000
    monkeypatch.setattr(provider,'_window',lambda *_:pytest.fail('Repeated model inference'))
    second=provider.analyze(media_for(speech),speech,settings,tmp_path/'cache',Cancellation(),lambda *_:None)
    assert second.candidates==first.candidates and second.info['cached_windows']==2


def test_failed_critic_never_exposes_unreviewed_clip_and_can_resume(speech,settings,event,tmp_path,monkeypatch):
    model=tmp_path/'model.gguf';model.write_bytes(b'GGUF-test')
    provider=LocalModelProvider(Tools(llama=sys.executable,semantic_model=str(model),acceleration='cpu'))
    def window(request,staged,token,progress,stats,decode):
        if request.response_field=='events':return decode([asdict(event)]),1000,[],None
        return None,1000,[],'Incomplete reviewer response'
    monkeypatch.setattr(provider,'_window',window)
    result=provider.analyze(media_for(speech),speech,settings,tmp_path/'cache',Cancellation(),lambda *_:None)
    assert not result.info['complete'] and not result.candidates and result.info['review_failed_windows']==1
    def resumed(request,staged,token,progress,stats,decode):
        assert request.response_field=='reviews'
        return decode([verdict(event)]),1000,[],None
    monkeypatch.setattr(provider,'_window',resumed)
    result=provider.analyze(media_for(speech),speech,settings,tmp_path/'cache',Cancellation(),lambda *_:None)
    assert result.info['complete'] and len(result.candidates)==1


def test_old_score_sums_do_not_bypass_new_review(speech,settings,event):
    weak=verdict(event,score=100,meaningful=False,reason='Связный разговор, но события нет.')
    good,bad=reviewed_candidates([weak],[event],media_for(speech),speech,settings,'model')
    assert not good and bad[0]['reason']=='Связный разговор, но события нет.'


def test_discovery_context_splits_cover_source_and_resume_without_retrying_parent(settings,tmp_path,monkeypatch):
    model=tmp_path/'model.gguf';model.write_bytes(b'GGUF-test')
    provider=LocalModelProvider(Tools(llama=sys.executable,semantic_model=str(model),acceleration='cpu'))
    speech=[Segment(i*10000,(i+1)*10000,f'Реплика номер {i}') for i in range(12)]
    calls=[]
    def window(request,staged,token,progress,stats,decode):
        rows=request.schema['properties']['events']['items']['properties']['first']['enum']
        calls.append(rows)
        if len(rows)>6:raise PromptTooLong(9000,7000)
        return decode([]),1000,[],None
    monkeypatch.setattr(provider,'_window',window)
    result=provider.analyze(media_for(speech),speech,settings,tmp_path/'cache',Cancellation(),lambda *_:None)
    assert result.info['complete'] and result.info['adaptive_splits']>0
    assert {i for rows in calls if len(rows)<=6 for i in rows}=={f'R{i}' for i in range(len(speech))}
    monkeypatch.setattr(provider,'_window',lambda *_:pytest.fail('Cached split recomputed'))
    result=provider.analyze(media_for(speech),speech,settings,tmp_path/'cache',Cancellation(),lambda *_:None)
    assert result.info['complete'] and not result.candidates


def test_oversized_review_batches_split_and_keep_all_verdicts(settings,tmp_path,monkeypatch):
    model=tmp_path/'model.gguf';model.write_bytes(b'GGUF-test')
    provider=LocalModelProvider(Tools(llama=sys.executable,semantic_model=str(model),acceleration='cpu'))
    speech=[Segment(i*12000,(i+1)*12000,f'Неожиданная шутка номер {i}') for i in range(3)]
    events=[Event(i,i,i,'Неожиданная',s.text,s.text,'Юмор',90-i) for i,s in enumerate(speech)]
    calls=[]
    def window(request,staged,token,progress,stats,decode):
        if request.response_field=='events':return decode([asdict(e) for e in events]),1000,[],None
        rows=request.schema['properties']['reviews']['items']['properties']['first']['enum']
        calls.append(rows)
        if len(rows)>1:raise PromptTooLong(9000,7000)
        i=int(rows[0][1:]);text=speech[i].text
        return decode([verdict(events[i],first=i,last=i,hook_quote='Неожиданная',end_quote=text)]),1000,[],None
    monkeypatch.setattr(provider,'_window',window)
    result=provider.analyze(media_for(speech),speech,settings,tmp_path/'cache',Cancellation(),lambda *_:None)
    assert result.info['complete'] and len(result.candidates)==3
    assert [c.start_ms for c in result.candidates]==[0,12000,24000]
    assert result.info['reviewed_batches']==3


def test_cancel_between_passes_then_resume_only_editorial_review(speech,settings,event,tmp_path,monkeypatch):
    model=tmp_path/'model.gguf';model.write_bytes(b'GGUF-test')
    provider=LocalModelProvider(Tools(llama=sys.executable,semantic_model=str(model),acceleration='cpu'))
    token=Cancellation()
    def window(request,staged,token,progress,stats,decode):
        assert request.response_field=='events'
        token.cancel()
        return decode([asdict(event)]),1000,[],None
    monkeypatch.setattr(provider,'_window',window)
    with pytest.raises(Cancelled):
        provider.analyze(media_for(speech),speech,settings,tmp_path/'cache',token,lambda *_:None)
    def resume(request,staged,token,progress,stats,decode):
        assert request.response_field=='reviews'
        return decode([verdict(event)]),1000,[],None
    monkeypatch.setattr(provider,'_window',resume)
    result=provider.analyze(media_for(speech),speech,settings,tmp_path/'cache',Cancellation(),lambda *_:None)
    assert result.info['complete'] and len(result.candidates)==1


def test_modified_evidence_cache_cannot_bypass_review(speech,settings,event,tmp_path,monkeypatch):
    import json
    model=tmp_path/'model.gguf';model.write_bytes(b'GGUF-test')
    provider=LocalModelProvider(Tools(llama=sys.executable,semantic_model=str(model),acceleration='cpu'))
    calls=[]
    def window(request,staged,token,progress,stats,decode):
        calls.append(request.response_field)
        return decode([asdict(event)] if request.response_field=='events' else [verdict(event)]),1000,[],None
    monkeypatch.setattr(provider,'_window',window)
    provider.analyze(media_for(speech),speech,settings,tmp_path/'cache',Cancellation(),lambda *_:None)
    path=next((tmp_path/'cache'/'local-qwen-hooks').rglob('review-*.json'))
    saved=json.loads(path.read_text(encoding='utf-8'));saved[0]['hook_quote']='Выдуманный зацеп'
    path.write_text(json.dumps(saved),encoding='utf-8')
    calls.clear()
    result=provider.analyze(media_for(speech),speech,settings,tmp_path/'cache',Cancellation(),lambda *_:None)
    assert result.info['complete'] and len(result.candidates)==1 and calls==['reviews']


def test_model_selects_rows_and_source_quotes_are_inserted_without_generation(speech,settings):
    event=model_event(dict(first='R0',peak='R1',last='R2',rank=90,category='Юмор'),speech,settings,0,len(speech))
    assert event.event_quote==speech[1].text and event.end_quote==speech[2].text
    raw=verdict(event,first='R0',last='R2')
    raw.pop('hook_quote');raw.pop('end_quote')
    reviews=quote_reviews([raw],[event],speech)
    good,bad=reviewed_candidates(reviews,[event],media_for(speech),speech,settings,'model')
    assert len(good)==1 and not bad and speech[1].text in good[0].reason
    request=discovery_request(media_for(speech),speech,settings,0,len(speech),[],3)
    assert 'hook_quote' not in request.schema['properties']['events']['items']['properties']
    assert 'end_quote' not in review_request(speech,settings,[event]).schema['properties']['reviews']['items']['properties']
