"""Publish only after the actual portable runtime and complete native suite pass."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

MODEL_SHA = "35295e9a0a42eff5ebc592412c754991c0b6cc36bf4f282b2a491b211476752d"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def validate_test_report(path):
    cases = ET.parse(path).findall(".//testcase")
    names = {c.attrib["name"] for c in cases}
    required = {
        "test_real_comfort_export_preview_sync_and_empty_lower_zone",
        "test_retired_animations_migrate_without_changing_clips_and_export_black",
        "test_legacy_qwen_cache_ignores_retired_animation_plan",
        "test_actual_qwen_mixed_short_meme_and_long_story_and_cache",
        "test_profile_cache_is_separate_and_reused",
        "test_search_profiles_new_defaults_change_without_analysis_and_restore_legacy",
        "test_export_profiles_apply_preserve_individual_choices_and_roundtrip",
        "test_export_profile_real_mp4_clean_video_and_missing_camera_preflight",
        "test_loudness_real_output_target_silence_and_peak",
        "test_mixed_request_offers_short_events_and_long_stories_together",
        "test_mixed_neighbor_request_keeps_absolute_anchor_reference",
        "test_mixed_proposals_do_not_join_events_through_an_ad_break",
        "test_mixed_windows_cover_independent_events_without_waiting_intro_or_ads",
        "test_mixed_shortfall_checks_unselected_speech_once_and_reuses_cache",
        "test_mixed_span_reply_resolves_global_rows_and_rejects_unknown_pairs",
        "test_video_frames_are_deferred_until_gui_event_loop",
        "test_tiktok_guides_toggle_and_real_export_stays_clean[top_black]",
        "test_tiktok_guides_toggle_and_real_export_stays_clean[blur_center]",
        "test_tiktok_guides_toggle_and_real_export_stays_clean[gaming_split]",
        "test_mixed_selection_keeps_stronger_story_instead_of_filling_quota",
        "test_story_profile_keeps_context_above_regular_limit",
        "test_search_story_switch_updates_upper_limit_without_losing_clips",
        "test_retired_subtitles_migrate_and_export_clean_video[srt]",
        "test_retired_subtitles_migrate_and_export_clean_video[burn]",
        "test_comfort_controls_preview_export_and_project_roundtrip",
        "test_actual_russian_recognition_and_cache",
        "test_actual_qwen_content_query_and_cache",
        "test_actual_qwen_neighbor_review_and_cache",
        "test_actual_qwen_context_budget_without_generation",
        "test_actual_qwen_semantic_selection_and_cache",
        "test_actual_qwen_coarse_segments_and_cache",
        "test_windows_cancel_stops_descendant_process_and_releases_logs",
        "test_mixed_export_uses_individual_layouts_and_saves_choices",
        "test_export_choices_follow_clip_identity_across_selection_and_sort",
        "test_export_framing_preview_uses_chosen_clip_and_mode",
        "test_gaming_region_editor_drag_apply_save_and_cancel",
        "test_gaming_mixed_queue_requires_regions_before_any_export",
        "test_gaming_settings_roundtrip_and_legacy_tiktok_migration",
        "test_gaming_requires_camera_before_creating_files",
        "test_black_bar_suggestion_handles_corner_camera_and_black_intro",
    }
    if (
        len(cases) < 209
        or not required <= names
        or any(
            c.find(tag) is not None
            for c in cases
            for tag in ("failure", "error", "skipped")
        )
    ):
        raise RuntimeError(
            "Complete Windows suite with actual Whisper/Qwen and gaming export did not pass"
        )
    if not any(
        n.startswith("test_gaming_real_export_camera_game_audio_and_clean_video[")
        for n in names
    ):
        raise RuntimeError("Real gaming MP4 verification is missing")
    return cases


def main():
    portable, output, model = (Path(p).resolve() for p in sys.argv[1:4])
    root = Path(__file__).resolve().parent.parent
    xml = root / "qa/test-results-windows.xml"
    cases = validate_test_report(xml)
    if digest(model) != MODEL_SHA or model.stat().st_size >= 2**31:
        raise RuntimeError("Invalid separate Qwen asset")
    run_url = f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    validation_url = run_url
    origin = None
    reused = root / "qa/reused-validation.json"
    if reused.exists():
        origin = json.loads(reused.read_text(encoding="utf-8"))
        if (
            not origin.get("application_and_tests_identical")
            or origin["package_commit"] != os.environ["GITHUB_SHA"]
            or origin["report_sha256"] != digest(xml)
            or origin["tests_passed"] != len(cases)
        ):
            raise RuntimeError("Invalid reused validation provenance")
        validation_url = origin["validation_url"]
    report = f"""# Проверка Clipnado Studio Lite 0.1.14

Windows Server 2022 x64. **Пройдено {len(cases)} тестов, без пропусков и ошибок.**
Набор тестов: {validation_url}
Готовый переносимый пакет: {run_url}

- Поиск по умолчанию: смешанная подборка мемов, реакций, событий и рассказов за один запуск. Минимум 15 секунд, обычные моменты до 60, истории с нужным контекстом до 90. Настоящая Qwen выбрала отдельную шутку и рассказ из одного авторского диалога; повторный поиск использовал кэш. Проверены длинные допустимые интервалы и отклонение длинной шутки, сохранение более сильной цельной истории вместо слабых отрывков ради количества, изоляция кэша стилей.
- Смешанный режим оценивает независимые речевые блоки отдельно. Явные коммерческие паузы и короткое ожидание зрителей в начале не склеивают разные события; исходная расшифровка сохраняется полностью, поиск по содержанию включает и исключённые из обычного отбора реплики. При недоборе допускается ограниченная дополнительная проверка невыбранной речи с кэшированием, без повышения оценок и повторного просмотра уже выбранных событий.
- Для естественных границ модель выбирает первую/последнюю реплики напрямую из ограниченного грамматикой набора допустимых пар. Время и длина проверяются локально; длинные интервалы допускают категорию «История». Неизвестные пары отклоняются. Для неточных крупных сегментов сохраняется прежний выбор подготовленных интервалов с предупреждением о границе внутри реплики. Контрольный пример ограничен 3200 входными токенами на запрос; это проверка объёма, не замер ускорения на физическом ПК пользователя. При уточнении соседних границ сохранены абсолютные ссылки на исходный интервал.
- Разметка интерфейса TikTok включается галочкой только в предпросмотре: верхняя панель, правые кнопки и область описания. Проверены все три оформления и реальный MP4 с включённой разметкой: текст и значки в экспорт не попадают. Исходный кадр и настройки экспорта не меняются. Проверены вертикальные, квадратные и горизонтальные кадры, изменение размера окна, выключение/повторное включение и отсутствие рисования за пределами примерного экрана 9:16. Расположение элементов приблизительное и не гарантирует безопасные границы на всех устройствах.
- Реальный запуск, GUI-тесты и проверка переносимого пакета используют одну функцию настройки имени/организации и стиля Fusion со штатной темой.
- Кадры предварительного просмотра передаются в очередь главного потока через зарегистрированный Qt-слот. Тест проверяет, что синхронная передача кадра не запускает перерисовку внутри обновления виджетов; пиксели отображаются после обработки событий.
- Отдельные стили поиска сохранены. GUI проверил смену верхнего предела 60/90, повышение минимума, сохранение существующих клипов и старых проектов без запуска модели. Профили экспорта: игровая камера 35% / игра 65%, разговор с размытым фоном, рассказ с верхним видео. 1080p, CRF 18, FPS по источнику, выравнивание звука; области камеры/игры и индивидуальные оформления сохраняются.
- Субтитры удалены из экспорта и интерфейса: нет ASS-рендера и SRT-файлов. Старые проекты с SRT/встраиванием открываются, их прежние настройки игнорируются; расшифровка и клипы сохранены. Реальные MP4 проверены по отсутствию белых букв на синтетическом цветном источнике и отсутствию дорожек/файлов субтитров. Импорт SRT сохранён только для загрузки расшифровки в смысловой поиск. Текст, уже присутствующий на исходном изображении, не удаляется.
- Цветокоррекция, осветление без подъёма чёрного, затемнение только размытого фона, двухпроходное измерение громкости (цель −16 LUFS, ограничение усиления тихих записей). Реальные MP4 проверены по пикселям, громкости, пикам, длительности и синхронизации.
- Встроенные анимации и их планировщик Qwen удалены. Старые поля проекта/кэша игнорируются, клипы и расшифровка сохраняются. Переносимый пакет экспортировал клип с обработкой цвета/звука; пиксели свободной нижней зоны проверены на чёрный цвет.
- Игровое оформление: один источник делится на две области — веб-камера сверху, игра снизу. Настоящий FFmpeg проверен для 720×1280 и 1080×1920, 30/60 FPS, доли камеры 25/40/60%, отсутствия добавленных полей и растяжения. Анаморфный SAR проверен по форме контрольного квадрата.
- Источник — авторский синтетический стрим с красной камерой справа сверху и синей игрой. Синхронные вспышки обеих областей и тон проверили время при ненулевом начале клипа; проверены чистое изображение без добавленных букв и пиксели предпросмотра. Это проверка рендера, не качества поиска на пользовательских игровых стримах.
- GUI: выделение рамки мышью с учётом полей виджета, изменение доли камеры, сохранение/открытие, отмена, новая независимая разметка для нового источника. Смешанная очередь проверяет области до создания первого файла. Индивидуальные варианты следуют за клипами при сортировке/отметках и сохраняются в проекте.
- Прежние варианты верхней тёмной зоны и размытого фона сохранены. Старый полный кадр TikTok заменён игровой компоновкой; старый проект открывается, но требует настройки областей перед игровым экспортом. Модели слежения за говорящим и их библиотеки удалены.
- Настоящий Whisper распознал русскую речь. Настоящая Qwen выбрала эпизоды, обработала смысловой запрос и соседние границы на составленном диалоге; проверены кэш и бюджет контекста 8192. Эти примеры подтверждают выполнение, не измеряют точность на длинных стримах.
- Переносимый пакет запущен именно через вложенный CPython 3.13.9: диагностика Qt/CLI/моделей, реальная генерация одного токена Qwen, декодированные кадры и воспроизведение, три оформления одной очередью, полный игровой MP4 и пиксели камеры/игры, выбор областей в GUI и предпросмотр. GUI-проводка LLM дополнительно проверена с протокольной фикстурой при заблокированных Python-соединениях; фактическая LLM проверена отдельными тестами выше.
- Нативные CPU/Vulkan-движки собраны MSVC; AVX включён, AVX2/FMA/F16C/BMI2/AVX512 отключены. Готовые MP4 проверены декодированием, ZIP — CRC. GGUF поставляется отдельно и проверен по SHA-256.

"""
    if origin:
        report += f"Набор тестов восстановлен из ревизии {origin['source_commit']}; подтверждено совпадение приложения, тестов, зависимостей и сборочных входов. Изменены только разрешённые сценарии упаковки. В текущем запуске переносимый пакет собран и проверен заново.\n\n"
    report += """Физические Xeon E5-2670/GTX 1660/Windows 10 пользователя и его длинные исходные видео не проверялись. Качество маленькой веб-камеры ограничено исходником. Qwen анализирует текст речи, поэтому чисто визуальные события игры может пропустить. Количество и интересность десяти клипов для любого видео не гарантируются. Это предварительный выпуск.
"""
    (portable / "QA_REPORT.md").write_text(report, encoding="utf-8")
    shutil.copy2(xml, portable / "qa" / xml.name)
    readme = portable / "README.md"
    content = readme.read_text(encoding="utf-8")
    verified = f"**Windows: {len(cases)} тестов прошли без пропусков.** [Полный набор]({validation_url}), [готовый пакет]({run_url}). Настоящие Whisper/Qwen, игровой MP4, выбор областей и встроенный Python проверены. Подробности: [QA_REPORT.md](QA_REPORT.md)."
    content = re.sub(
        r"(?s)(<!-- WINDOWS_VALIDATION_START -->).*?(<!-- WINDOWS_VALIDATION_END -->)",
        lambda m: m[1] + "\n" + verified + "\n" + m[2],
        content,
    )
    readme.write_text(content, encoding="utf-8")
    for cache in list(portable.rglob("__pycache__")):
        if cache.is_dir():
            shutil.rmtree(cache)
    manifest_path = portable / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["windows_execution_verified"] = True
    manifest["windows_validation"] = {
        "os": "Windows Server 2022 x64",
        "tests_passed": len(cases),
        "workflow_url": validation_url,
        "package_workflow_url": run_url,
        "validation_provenance": origin,
        "source_commit": os.environ["GITHUB_SHA"],
        "retired_animations_removed": True,
        "empty_lower_zone_export": True,
        "legacy_animation_projects_migrated": True,
        "actual_qwen_meme_profile": True,
        "search_and_export_profiles": True,
        "automatic_minimum_seconds": 15,
        "regular_maximum_seconds": 60,
        "story_maximum_seconds": 90,
        "mixed_search": True,
        "compact_mixed_request": True,
        "constrained_utterance_spans": True,
        "independent_speech_blocks": True,
        "bounded_unselected_speech_recovery": True,
        "deferred_video_frames": True,
        "shared_production_gui_configuration": True,
        "tiktok_preview_guides": True,
        "guides_excluded_from_export": True,
        "separate_events_at_explicit_ad_breaks": True,
        "mixed_fixture_prompt_limit": 3200,
        "portable_profile_export": True,
        "loudness_and_color": True,
        "export_subtitles_removed": True,
        "legacy_subtitle_projects_migrated": True,
        "transcript_search_preserved": True,
        "actual_whisper": True,
        "actual_qwen_dialogue": True,
        "portable_qwen_token": True,
        "portable_gui_export": True,
        "gaming_split_export": True,
        "gaming_regions_editor": True,
        "individual_clip_layouts": True,
        "blurred_export": True,
        "export_framing_preview": True,
        "target_hardware_verified": False,
        "physical_vulkan_verified": False,
        "compiler": "MSVC",
    }
    manifest["files"] = {
        p.relative_to(portable).as_posix(): digest(p)
        for p in sorted(portable.rglob("*"))
        if p.is_file() and p != manifest_path
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "ClipnadoStudioLite-Windows-x64.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(portable.rglob("*")):
            if p.is_file():
                z.write(p, Path("ClipnadoStudioLite") / p.relative_to(portable))
    with zipfile.ZipFile(archive) as z:
        if z.testzip():
            raise RuntimeError("Windows ZIP CRC verification failed")
    subprocess.run(
        [
            "git",
            "archive",
            "--format=zip",
            "--prefix=ClipnadoStudioLite-source/",
            "-o",
            str(output / "ClipnadoStudioLite-source.zip"),
            "HEAD",
        ],
        cwd=root,
        check=True,
    )
    shutil.copy2(readme, output / "START-HERE.md")
    shutil.copy2(portable / "QA_REPORT.md", output / "QA_REPORT.md")
    checks = [
        f"{digest(p)}  {p.name}"
        for p in sorted(output.iterdir())
        if p.is_file() and p.name not in ("SHA256SUMS.txt", "release-notes.md")
    ]
    checks.append(f"{MODEL_SHA}  {model.name}")
    (output / "SHA256SUMS.txt").write_text("\n".join(checks) + "\n", encoding="utf-8")
    notes = f"""Clipnado Studio Lite 0.1.14 — **разметка TikTok в предпросмотре**.

Во вкладке «Экспорт» выберите клип и нажмите «Предпросмотр кадра». Галочка «Показать интерфейс TikTok · примерно» показывает верхнюю панель, правые кнопки, имя канала и область описания. Снимите галочку для чистого кадра. Расположение приблизительное: интерфейс зависит от устройства, режима приложения и длины описания.

Разметка видна только в окне просмотра и **не попадает в MP4**. Исходный кадр и настройки экспорта сохраняются. Поддержаны все три оформления; квадратные и горизонтальные кадры показываются внутри примерного экрана 9:16 с сохранением пропорций. Смешанный поиск, длительности, качество экспорта и локальные модели сохранены. API и новые модели не нужны.

**{len(cases)} Windows-тестов без пропусков и ошибок**, включая настоящий отбор Whisper/Qwen, переключение разметки, реальные MP4 трёх оформлений и проверку переносимого Python. [Проверка]({validation_url}) · [Готовый пакет]({run_url}). SHA256SUMS.txt и QA_REPORT.md прилагаются.

Распакуйте ZIP в новую папку. Прежний `{model.name}` подходит — положите его в `models` или укажите существующий путь. Запуск: `ClipnadoStudioLite.exe`.

Физический ПК пользователя и его длинные стримы не проверены. Разметка не является точной спецификацией интерфейса TikTok. Qwen анализирует речь; просмотры и количество качественных моментов не гарантируются.
"""
    (output / "release-notes.md").write_text(notes, encoding="utf-8")
    print(
        f"Windows release verified: {len(cases)} tests; ZIP {archive.stat().st_size} bytes",
        flush=True,
    )


if __name__ == "__main__":
    main()
