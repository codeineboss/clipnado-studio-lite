"""Real Qwen on positive/negative authored examples, not reach prediction."""
import json
import os
from pathlib import Path

import pytest

from clipnado.highlights import Event, quote_reviews, review_request, reviewed_candidates
from clipnado.local_llm import LocalModelProvider, model_identity, stage_model
from clipnado.media import Tools
from clipnado.models import MediaInfo, SearchSettings, Segment
from clipnado.process import Cancellation
from clipnado.profiles import search_profile

pytestmark=pytest.mark.skipif(not os.environ.get('CLIPNADO_TEST_LLAMA') or not os.environ.get('CLIPNADO_TEST_QWEN'),
                             reason='Set actual local Qwen paths')


def inputs():
    data=json.loads((Path(__file__).resolve().parent.parent/'qa/highlights-dialogue-fixture.json').read_text(encoding='utf-8'))
    speech=[Segment(**row) for row in data['segments']]
    media=MediaInfo('authored-hook-dialogue.mp4','actual-hook-dialogue',52000,1920,1080,30,True)
    settings=search_profile(SearchSettings(min_seconds=10,max_seconds=90,count=6,target_count=2),'highlights')
    tools=Tools(llama=os.environ['CLIPNADO_TEST_LLAMA'],semantic_model=os.environ['CLIPNADO_TEST_QWEN'],threads=2,acceleration='cpu')
    return speech,media,settings,tools


def test_actual_qwen_hooks_find_event_reject_chatter_and_cache(tmp_path,monkeypatch):
    speech,media,settings,tools=inputs()
    provider=LocalModelProvider(tools)
    result=provider.analyze(media,speech,settings,tmp_path/'cache',Cancellation(),lambda *_:None)
    assert result.info['complete'],result.warnings
    assert result.candidates,'Real event discovery/review returned no authored strong moment'
    assert result.info['reviewed_batches'] and result.info['max_prompt_tokens']<=3500
    assert any((c.start_ms<=4000 and 12000<=c.end_ms<=20000)
               or (40000<=c.start_ms<=44000 and 48000<=c.end_ms<=52000) for c in result.candidates)
    assert all(not (20000<=c.start_ms and c.end_ms<=40000) for c in result.candidates),'Ordinary keyboard discussion was accepted'
    assert all(c.end_ms<=20000 or c.start_ms>=40000 for c in result.candidates),'An event was padded with the ordinary conversation'
    assert all(c.version==6 and c.provider.startswith('local-qwen:') and 'редакторская проверка' in c.reason for c in result.candidates)
    monkeypatch.setattr(provider,'_window',lambda *_:pytest.fail('Repeated actual inference'))
    cached=provider.analyze(media,speech,settings,tmp_path/'cache',Cancellation(),lambda *_:None)
    assert cached.candidates==result.candidates and cached.info['cached_windows']>=2


def test_actual_qwen_critic_rejects_high_rank_ordinary_dialogue(tmp_path):
    speech,media,settings,tools=inputs()
    # Deliberately rank the mundane dialogue higher before a blind REAL review.
    events=[Event(0,2,3,'Сейчас покажу идеальный прыжок','я снова упал в ту же яму','падению в ямы','Реакция',1),
            Event(5,7,9,'Теперь поговорим','чёрные кнопки и обычный провод','больше ничего не случилось','История',100)]
    provider=LocalModelProvider(tools);token=Cancellation()
    identity=model_identity(Path(tools.semantic_model),token)
    staged=stage_model(Path(tools.semantic_model),tmp_path/'cache',identity,token)
    request=review_request(speech,settings,events)
    stats={'max_prompt_tokens':0,'warnings':[]}
    result,_,_,error=provider._window(request,staged,token,lambda *_:None,stats,
        lambda rows: reviewed_candidates(quote_reviews(rows,events,speech),events,media,speech,settings,identity))
    assert result is not None,error
    good,bad=result
    assert good and all(c.start_ms<20000 and c.end_ms<=20000 for c in good)
    assert any(item['first']>=5 for item in bad),'Critic accepted the deliberately high-ranked ordinary conversation'
    assert any('Блядь' in c.transcript_excerpt for c in good),'Profane strong reaction was excluded'
