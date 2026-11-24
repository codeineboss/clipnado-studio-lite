"""Restore the preferred selector without changing saved editing decisions."""
import hashlib
import json
from pathlib import Path
import sys

from clipnado.analysis import recognize
from clipnado.media import Tools, fingerprint
from clipnado.models import Candidate, ExportSettings, Project, SearchSettings, Segment
from clipnado.process import Cancellation
from clipnado.projects import load_project, save_project


def project_fixture(media):
    return Project(media, segments=[Segment(0,1500,'Блядь, вот это реакция!'),Segment(1500,4000,'Контекст и продолжение.')],
        candidates=[Candidate(media.source_file_id,500,3000,85,'Реакция','Сохранённый момент',
                              selected=True,export_layout='center_black',version=6),
                    Candidate(media.source_file_id,4000,7000,75,'История','Сохранённый контекст',
                              selected=False,export_layout='blur_center',version=6)],
        search=SearchSettings(min_seconds=15,max_seconds=90,profile='mixed'),
        export=ExportSettings(layout='center_black',background_dim=.35),
        audio_energy=[[0,.1],[2000,.4]],analysis_info={'mode':'local-qwen-hooks','complete':True})


def test_016_project_migration_preserves_clips_transcript_and_export_choices(media,tmp_path):
    original=project_fixture(media);data=original.to_dict()
    data['search'].update(profile='highlights',min_seconds=10,max_seconds=90)
    path=tmp_path/'проект 0.1.16.json';path.write_text(json.dumps(data),encoding='utf-8')
    restored=load_project(path)
    assert (restored.search.profile,restored.search.min_seconds,restored.search.max_seconds)==('mixed',15,90)
    for field in ('segments','candidates','export','audio_energy','analysis_info'):
        assert getattr(restored,field)==getattr(original,field)
    assert any('метод 0.1.15' in message for message in restored.warnings)
    save_project(restored,path);again=load_project(path)
    assert again.to_dict()==restored.to_dict(), 'Migration must not repeat or change editing decisions'


def test_015_project_keeps_original_search_and_boundaries(media,tmp_path):
    project=project_fixture(media);project.analysis_info={'mode':'local-qwen','complete':True}
    path=tmp_path/'проект 0.1.15.json';save_project(project,path)
    restored=load_project(path)
    assert restored.to_dict()==project.to_dict()


def test_016_fine_cache_audio_is_recovered_without_replacing_saved_transcript(media,tmp_path,monkeypatch):
    model=tmp_path/'model.bin';model.write_bytes(b'fixture')
    tools=Tools(whisper=sys.executable,model=str(model))
    segments=[Segment(0,1000,'Ранее отредактированная пользователем реплика.')]
    key=hashlib.sha256((media.source_file_id+fingerprint(model,Cancellation())+':ru:v5-fine80').encode()).hexdigest()
    (tmp_path/f'{key}.json').write_text(json.dumps({'energy':[[0,.1],[2000,.4]],'segments':[]}),encoding='utf-8')
    monkeypatch.setattr('clipnado.analysis.run',lambda *_a,**_k: (_ for _ in ()).throw(AssertionError('ASR must not restart')))
    result=recognize(media,tools,tmp_path,Cancellation(),lambda *_:None,existing=segments)
    assert result[0]==segments and result[1]==[[0,.1],[2000,.4]]
