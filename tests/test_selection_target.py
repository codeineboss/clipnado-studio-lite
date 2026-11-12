"""Quantity regressions: reaching a target must not manufacture scores or repeats."""
import json

import pytest

from clipnado.models import Candidate, MediaInfo, Project, SearchSettings
from clipnado.process import Cancellation, Cancelled
from clipnado.projects import load_project, save_project
from clipnado.selection import choose_candidates, shortfall_message


def candidate(start, end, score):
    return Candidate("source", start * 1000, end * 1000, score, "Реакция", "Fixture")


def test_alternative_boundaries_reach_ten_instead_of_five_greedy_clips():
    short = [candidate(i * 60, (i + 1) * 60, 80) for i in range(10)]
    blockers = [candidate(i * 120, i * 120 + 90, 95) for i in range(5)]
    result = choose_candidates(blockers + short, SearchSettings(), Cancellation())
    assert len(result) == 10
    assert [(c.start_ms, c.end_ms, c.score) for c in result] == [(c.start_ms, c.end_ms, 80) for c in short]
    assert all(a.end_ms <= b.start_ms for a, b in zip(result, result[1:]))


def test_score_floor_is_absolute_even_when_target_cannot_be_met():
    source = [candidate(i * 60, (i + 1) * 60, 59 if i % 2 else 75) for i in range(10)]
    before = [c.score for c in source]
    settings = SearchSettings(sensitivity=100)
    result = choose_candidates(source, settings, Cancellation())
    assert len(result) == 5 and all(c.score == 75 for c in result)
    assert [c.score for c in source] == before
    media = MediaInfo("fixture.mp4", "source", 600000, 1920, 1080, 30, True)
    assert "5 из целевых 10" in shortfall_message(media, settings, result, True)


def test_empty_or_short_material_reports_shortfall():
    settings = SearchSettings()
    media = MediaInfo("fixture.mp4", "source", 180000, 1920, 1080, 30, False)
    assert choose_candidates([], settings, Cancellation()) == []
    message = shortfall_message(media, settings, [], False)
    assert "0 из целевых 10" in message and "не более 3" in message and "Нет расшифровки" in message
    media.duration_ms = 600000
    assert shortfall_message(media, settings, [candidate(i * 60, (i + 1) * 60, 80) for i in range(10)], True) == ""


def test_goal_settings_and_old_project_migration(media, tmp_path):
    settings = SearchSettings(count=20, target_count=12, min_score=65)
    path = tmp_path / "settings.json"
    save_project(Project(media, search=settings), path)
    assert load_project(path).search == settings
    data = json.loads(path.read_text(encoding="utf-8"))
    data["search"].pop("target_count")
    data["search"].pop("min_score")
    data["search"]["count"] = 5
    path.write_text(json.dumps(data), encoding="utf-8")
    old = load_project(path)
    assert old.search.target_count == 5 and old.search.min_score == 60


@pytest.mark.parametrize("settings", [SearchSettings(target_count=16), SearchSettings(min_score=101), SearchSettings(target_count=0)])
def test_invalid_target_settings(settings):
    with pytest.raises(ValueError):
        settings.validate()


def test_selection_can_be_cancelled():
    token = Cancellation()
    token.cancel()
    with pytest.raises(Cancelled):
        choose_candidates([candidate(0, 60, 80)], SearchSettings(), token)
