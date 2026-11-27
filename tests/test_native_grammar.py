"""Quoted speech must remain exact data instead of breaking native grammar."""
import json
import os
from pathlib import Path
import sys

import pytest

from clipnado.local_llm import LocalModelProvider, build_request, prompt_tokens, response_json
from clipnado.media import Tools
from clipnado.models import MediaInfo, SearchSettings, Segment
from clipnado.native_grammar import compile_schema, literal
from clipnado.process import Cancellation, ProcessExecutionError, run
from clipnado.profiles import search_profile


def quote_schema(value):
    return {'type':'object','properties':{'quotes':{'type':'array','minItems':1,'maxItems':1,
        'items':{'type':'object','properties':{'value':{'type':'string','enum':[value]}},
                 'required':['value'],'additionalProperties':False}}},
        'required':['quotes'],'additionalProperties':False}


def test_enum_grammar_quotes_json_serialization_twice_without_changing_source():
    value='Он сказал: "Да\\нет". '+chr(1)+' 😂'
    encoded=json.dumps(value,ensure_ascii=False)
    grammar=compile_schema(quote_schema(value))
    assert literal(encoded) in grammar
    assert json.loads(json.loads(literal(encoded)))==value
    assert chr(1) not in grammar
    assert 'jchar ::=' in grammar


@pytest.mark.parametrize('schema',[
    {'type':'object','properties':{},'required':[],'additionalProperties':True},
    {'type':'object','properties':{'x':{'type':'string','maxLength':2}},'required':[],'additionalProperties':False},
    {'type':'array','items':{'type':'integer','minimum':0,'maximum':2}},
    {'type':'integer'}, {'type':'string'}, {'type':'integer','enum':[True]},
    {'anyOf':[]}, {'type':'string','enum':['x'],'pattern':'^x$'},
])
def test_unsupported_schema_constraints_are_never_silently_dropped(schema):
    with pytest.raises(ValueError):compile_schema(schema)


def test_native_process_error_retains_cause_before_a_long_verbose_tail():
    marker='parse: error parsing grammar: quoted literal'
    code='import sys;print('+repr(marker)+');print("19 -> 4\\n"*1500);sys.exit(7)'
    with pytest.raises(ProcessExecutionError) as error:
        run([sys.executable,'-c',code])
    assert error.value.returncode==7 and error.value.output.startswith(marker)
    assert len(error.value.output)>2200 and marker not in str(error.value)


def test_context_probe_uses_full_error_output_for_hidden_overflow(tmp_path,monkeypatch):
    def fail(*_args,**_kwargs):
        raise ProcessExecutionError('token dump tail', 'main: prompt is too long (8471 tokens, max 8188)\n'+'19 -> 4\n'*1000, 1)
    monkeypatch.setattr('clipnado.local_llm.run',fail)
    assert prompt_tokens(['native','-n','500','-ngl','99'],'cpu',Cancellation(),tmp_path)==8471


@pytest.mark.parametrize('phase',['probe','generation'])
def test_native_window_failure_preserves_results_full_diagnostic_and_resume(tmp_path,monkeypatch,phase):
    model=tmp_path/'fixture.gguf';model.write_bytes(b'GGUF-test')
    tools=Tools(llama=sys.executable,semantic_model=str(model),acceleration='cpu')
    segments=[Segment(i,i+10000,'Обычная речь '+str(i)) for i in range(0,900000,10000)]
    media=MediaInfo('source','native-resume',900000,1920,1080,30,True)
    cache=tmp_path/'cache';calls=[];failed=[False]
    full='parse: error parsing grammar: quoted source\n'+'19 -> 4\n'*1000
    def failure():
        failed[0]=True
        raise ProcessExecutionError(full[-2200:],full,1)
    def probe(*_args,**_kwargs):
        calls.append('probe')
        if phase=='probe' and not failed[0]:failure()
        return 1000
    def generate(*_args,**_kwargs):
        calls.append('generate')
        if phase=='generation' and not failed[0]:failure()
        return json.dumps({'clips':[dict(clip_id=0,hook=20,event=20,emotion=20,completeness=20,category='Юмор',reason='Цельный момент')]})
    monkeypatch.setattr('clipnado.local_llm.prompt_tokens',probe)
    monkeypatch.setattr('clipnado.local_llm.run',generate)
    provider=LocalModelProvider(tools)
    result=provider.analyze(media,segments,SearchSettings(),cache,Cancellation(),lambda *_:None)
    assert result.candidates and result.info['failed_windows']==1 and not result.info['complete']
    path=next(cache.rglob('*.failed.json'));diagnostic=json.loads(path.read_text(encoding='utf-8'))
    assert full in diagnostic['outputs'] and 'код 1' in diagnostic['error']
    assert len(result.warnings[0])<700 and '19 -> 4' not in result.warnings[0]
    ids={c.id for c in result.candidates};before=len(calls)
    resumed=provider.analyze(media,segments,SearchSettings(),cache,Cancellation(),lambda *_:None)
    assert resumed.info['complete'] and len(calls)==before+2
    assert ids<={c.id for c in resumed.candidates} and not list(cache.rglob('*.failed.json'))


def native_paths():
    return os.environ['CLIPNADO_TEST_LLAMA'],os.environ['CLIPNADO_TEST_QWEN']


actual=pytest.mark.skipif(not os.environ.get('CLIPNADO_TEST_LLAMA') or not os.environ.get('CLIPNADO_TEST_QWEN'),reason='Set actual local Qwen paths')


@actual
def test_actual_qwen_grammar_quoted_transcript_no_generation(tmp_path):
    engine,model=native_paths()
    texts=['Он сказал: "Я всё понял". '+str(i)+' '+chr(1)+' 😂 \\ видео' for i in range(120)]
    segments=[Segment(i*3500,(i+1)*3500,text) for i,text in enumerate(texts)]
    media=MediaInfo('authored','quoted',420000,1920,1080,30,True)
    request=build_request(media,segments,search_profile(SearchSettings(min_seconds=15),'mixed'),0,len(segments),1)
    (tmp_path/'prompt.txt').write_text(request.prompt,encoding='utf-8')
    (tmp_path/'system.txt').write_text(request.system,encoding='utf-8')
    (tmp_path/'schema.gbnf').write_text(compile_schema(request.schema),encoding='utf-8')
    args=[engine,'-m',model,'-f','prompt.txt','--system-prompt-file','system.txt','--grammar-file','schema.gbnf',
          '--conversation','--single-turn','--simple-io','--no-display-prompt','--offline','--no-warmup','-c','8192','-n','0','-ngl','0','-t','2']
    output=run([*args,'--verbose-prompt'],cwd=tmp_path,timeout=180)
    assert 'failed to initialize' not in output and 'n_predict = 0' in output
    assert 'prompt eval time =    0.00 ms' in output or __import__('re').search(r'prompt eval time\s*=\s*0\.00 ms',output)


@actual
def test_actual_qwen_generates_exact_quoted_json_with_controls(tmp_path):
    engine,model=native_paths()
    value='Он сказал: "Да\\нет". '+chr(1)+' 😂'
    (tmp_path/'schema.gbnf').write_text(compile_schema(quote_schema(value)),encoding='utf-8')
    output=run([engine,'-m',model,'-p','Верни JSON по заданной грамматике.','--grammar-file','schema.gbnf',
                '--conversation','--single-turn','--simple-io','--no-display-prompt','--offline','--no-warmup',
                '-c','512','-n','96','-ngl','0','-t','2'],cwd=tmp_path,timeout=480)
    assert response_json(output,'quotes')==[{'value':value}]
