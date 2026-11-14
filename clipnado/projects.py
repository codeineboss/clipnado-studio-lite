from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .models import Candidate, ExportSettings, MediaInfo, Project, SearchSettings, Segment


def atomic_json(path: Path, value: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as stream:
            name = stream.name
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def save_project(project: Project, path: Path):
    atomic_json(path, project.to_dict())


def load_project(path: Path) -> Project:
    path = Path(path)
    if path.stat().st_size > 100_000_000:
        raise ValueError("Файл проекта слишком большой")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        search = dict(data.get("search", {}))
        # An old project may intentionally show fewer than ten candidates.
        search.setdefault("target_count", min(10, search.get("count", 15)))
        project = Project(media=MediaInfo(**data["media"]),
                          segments=[Segment(**s) for s in data.get("segments", [])],
                          candidates=[Candidate(**c) for c in data.get("candidates", [])],
                          search=SearchSettings(**search),
                          export=ExportSettings(**data.get("export", {})),
                          schema_version=data["schema_version"], warnings=data.get("warnings", []),
                          audio_energy=data.get("audio_energy", []), analysis_info=data.get("analysis_info", {}),
                          query_results=[Candidate(**c) for c in data.get("query_results", [])],
                          last_query=data.get("last_query", ""), query_info=data.get("query_info", {}))
        project.validate()
        return project
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Повреждённый или несовместимый файл проекта: " + str(exc)) from exc
