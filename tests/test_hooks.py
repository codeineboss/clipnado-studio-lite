"""Opening priority without a second inference pass or a hard rejection gate."""
import os
import pytest

from clipnado.local_llm import LocalModelProvider, build_request, decode_choices, editorial_score
from clipnado.media import Tools
from clipnado.models import MediaInfo, SearchSettings, Segment
from clipnado.process import Cancellation
from clipnado.profiles import search_profile


def test_hook_weight_prioritizes_opening_without_rejecting_lower_scores():
    strong = [24, 20, 17, 20]
    delayed = [10, 25, 25, 25]
    assert sum(delayed) > sum(strong)
    assert editorial_score(strong) > editorial_score(delayed)
    assert editorial_score([0, 25, 25, 25]) == 60  # no hidden hook gate
    assert editorial_score([25]*4) == 100 and editorial_score([0]*4) == 0
    assert editorial_score(delayed, query=True) == sum(delayed)


def test_request_scopes_hook_to_first_seconds_and_keeps_validated_intervals():
    media = MediaInfo('file', 'hook', 60000, 1920, 1080, 30, True)
    segments = [Segment(i, i+6000, 'Реплика '+str(i)) for i in range(0,60000,6000)]
    settings = search_profile(SearchSettings(min_seconds=15), 'mixed')
    request = build_request(media, segments, settings, 0, len(segments), 1)
    assert 'В ПЕРВЫХ 2–6 СЕКУНДАХ' in request.system
    assert 'через 20 секунд' in request.system and 'не придумывай хук' in request.system
    assert 'R0 (6 с)' in request.prompt
    assert all(15000 <= p.end_ms-p.start_ms <= 90000 for p in request.proposals)
    p = request.proposals[0]
    item = dict(first_row=p.first,last_row=p.last,ending_quote=segments[p.last].text,hook=24,event=20,emotion=17,completeness=20,
                category='Юмор',reason='Сразу цепляющее обещание и его результат')
    result = decode_choices([item],media,segments,settings,request.proposals,'model')
    assert result[0].score == editorial_score([24,20,17,20])
    assert (result[0].start_ms,result[0].end_ms)==(p.start_ms,p.end_ms)
    with pytest.raises(ValueError,match='Цитата финала'):
        decode_choices([{**item,'ending_quote':'Придуманный финал'}],media,segments,settings,request.proposals,'model')


@pytest.mark.skipif(not os.environ.get('CLIPNADO_TEST_LLAMA') or not os.environ.get('CLIPNADO_TEST_QWEN'),
                    reason='Set actual local Qwen paths')
def test_actual_qwen_hook_skips_empty_lead_in_and_keeps_reaction(tmp_path, monkeypatch):
    # Authored transcript, not evidence of precision on users' real streams.
    lines = [
        'Сейчас, подождите, я тут настройки открою. Пока ничего не происходит.',
        'Так, где у меня была эта кнопка, секунду. Проверяю настройки.',
        'Почти готово, ещё подождите. Потом расскажу что-нибудь.',
        'Спорим, пройду этот прыжок с закрытыми глазами? Да я тут профессионал!',
        'Закрываю глаза, прыгаю... Блядь, я упал в самую первую яму!',
        'Ахахаха! Профессионал! Ты даже глаза закрыть не успел, а уже проиграл!',
        'Всё, понял: профессионально умею только падать. Больше никого не учу!',
        'Ладно, этот момент закончился, вернёмся к скучным настройкам.',
        'Рекламная пауза: скидка по промокоду, ссылка в описании магазина.',
    ]
    segments = [Segment(i*6000,(i+1)*6000,t) for i,t in enumerate(lines)]
    media = MediaInfo('authored-hook.mp4','actual-hook',54000,1920,1080,30,True)
    settings = search_profile(SearchSettings(count=3,target_count=1,min_seconds=15),'mixed')
    tools = Tools(llama=os.environ['CLIPNADO_TEST_LLAMA'],semantic_model=os.environ['CLIPNADO_TEST_QWEN'],threads=2)
    provider = LocalModelProvider(tools)
    result = provider.analyze(media,segments,settings,tmp_path/'cache',Cancellation(),lambda *_:None)
    assert result.info['complete'] and result.candidates
    best = max(result.candidates,key=lambda c:c.score)
    # The bet and the blind attempt are both immediate, understandable hooks.
    # The observed bad 24–48s result must still fail the ending check below.
    assert best.start_ms in (18000,24000), [(c.start_ms,c.end_ms,c.reason) for c in result.candidates]
    assert 36000 <= best.end_ms <= 42000, (best.start_ms,best.end_ms,best.reason)
    assert 'профессионал' in best.transcript_excerpt.lower() and best.score >= 60
    monkeypatch.setattr('clipnado.local_llm.run',lambda *_a,**_k:pytest.fail('Repeated actual inference'))
    again = provider.analyze(media,segments,settings,tmp_path/'cache',Cancellation(),lambda *_:None)
    assert again.candidates==result.candidates
