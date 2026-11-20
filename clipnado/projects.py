from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .models import candidate_from_dict, Candidate, ExportSettings, MediaInfo, Project, SearchSettings, Segment


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
        export = dict(data.get("export", {}))
        retired_animation = export.pop("animation", "off")
        # Old saved projects keep their original appearance.
        export.setdefault("readable_subtitles", False)
        export.setdefault("background_dim", 0.)
        export.pop("tiktok_tracking", None)
        migrated = export.get("layout") == "tt_fill"
        if migrated:export["layout"] = "gaming_split"
        for group in ("candidates", "query_results"):
            for item in data.get(group, []):
                if item.get("export_layout") == "tt_fill":
                    item["export_layout"] = "gaming_split"
                    migrated = True
        warnings = list(data.get("warnings", []))
        if retired_animation not in (None, "off"):
            warning = "Встроенные анимации удалены. Нижняя зона остаётся свободной для вашего монтажа."
            if warning not in warnings:
                warnings.append(warning)
        if migrated:
            warnings.append("Полный кадр TikTok заменён игровой нарезкой. Перед экспортом задайте области камеры и игры.")
        project = Project(media=MediaInfo(**data["media"]),
                          segments=[Segment(**s) for s in data.get("segments", [])],
                          candidates=[candidate_from_dict(c) for c in data.get("candidates", [])],
                          search=SearchSettings(**search),
                          export=ExportSettings(**export),
                          schema_version=data["schema_version"], warnings=warnings,
                          audio_energy=data.get("audio_energy", []), analysis_info=data.get("analysis_info", {}),
                          query_results=[candidate_from_dict(c) for c in data.get("query_results", [])],
                          last_query=data.get("last_query", ""), query_info=data.get("query_info", {}))
        project.validate()
        if project.search.min_seconds < 30:
            project.search.min_seconds = 30
            if project.search.max_seconds is not None:
                project.search.max_seconds = max(30, project.search.max_seconds)
            warning = "Минимальная длина автоматического поиска поднята до 30 секунд. Существующие клипы сохранены."
            if warning not in project.warnings:
                project.warnings.append(warning)
        return project
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Повреждённый или несовместимый файл проекта: " + str(exc)) from exc
