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
        "test_actual_qwen_animation_planning_and_cache",
        "test_real_animation_export_preview_cache_and_sync",
        "test_loudness_real_output_target_silence_and_peak",
        "test_animation_cancel_during_render_removes_partial_files",
        "test_readable_captions_preserve_text_two_lines_and_bounds",
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
        len(cases) < 187
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
        n.startswith("test_gaming_real_export_camera_game_audio_and_subtitles[")
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
    report = f"""# Проверка Clipnado Studio Lite 0.1.11

Windows Server 2022 x64. **Пройдено {len(cases)} тестов, без пропусков и ошибок.**
Набор тестов: {validation_url}
Готовый переносимый пакет: {run_url}

- Новые настройки: мягкая цветокоррекция/осветление без подъёма чёрного, отдельное затемнение размытого фона, двухпроходное измерение громкости (цель −16 LUFS, ограничение усиления тихих записей), читаемые карточки субтитров до двух строк. Реальные MP4 проверены по пикселям, громкости, пикам, длительности и синхронизации.
- Семь собственных процедурных 2D-сцен: движение, замыкание цикла, ограниченная яркость. Qwen действительно выбрала сцены/цвет/темп; кэш исключил повторный вызов модели. Анимация проверена в готовом MP4 и предпросмотре, не перекрывает верхнее видео; другие форматы её не получают. Отмена удаляет незавершённый рендер.
- Переносимый пакет дополнительно экспортировал клип с анимацией, обработкой цвета/звука и ASS через вложенный Python; проверены пиксели нижней зоны, предпросмотр и сохранение настроек.
- Игровое оформление: один источник делится на две области — веб-камера сверху, игра снизу. Настоящий FFmpeg проверен для 720×1280 и 1080×1920, 30/60 FPS, доли камеры 25/40/60%, отсутствия добавленных полей и растяжения. Анаморфный SAR проверен по форме контрольного квадрата.
- Источник — авторский синтетический стрим с красной камерой справа сверху и синей игрой. Синхронные вспышки обеих областей и тон проверили время при ненулевом начале клипа; проверены off/SRT/ASS и пиксели предпросмотра. Это проверка рендера, не качества поиска на пользовательских игровых стримах.
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
        "actual_qwen_animation_planning": True,
        "procedural_animation_export": True,
        "portable_animation_export": True,
        "loudness_and_color": True,
        "readable_subtitles": True,
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
    notes = f"""Clipnado Studio Lite 0.1.11 — **локальные анимации и комфорт просмотра**.

В «Экспорт» появился блок **«Изображение, звук и нижняя зона»**:
- Исходные цвета, мягкая коррекция или небольшое осветление — без растяжения.
- Выравнивание всего звука клипа с измерением громкости; речь и игра не разделяются.
- Более крупные субтитры, до двух коротких строк на карточке. Время внутри реплики приблизительное.
- Затемнение только размытого фона.
- Для верхнего видео — семь собственных спокойных 2D-анимаций: волны, орбиты, маятники, лабиринт, шестерёнки, домино, частицы. Разные варианты автоматически, выбор конкретной сцены или **Qwen · подобрать по содержанию**. Qwen выбирает сцену/палитру/темп, программа рисует геометрию. Дополнительные модели и API не требуются. Подготовка занимает время, результат кэшируется. Сцены циклические, не генерация произвольного 3D-видео.

Анимация выключена по умолчанию, доступна только в свободной нижней зоне. «Камера и игра» и размытый фон сохранены. Старые проекты сохраняют прежнее оформление; новый проект немного затемняет размытый фон. Предпросмотр теперь показывает также анимацию и встроенные субтитры.

**Обновление:** распакуйте ZIP в новую папку. Прежний Qwen GGUF подходит, повторно скачивать его не нужно. Для первой установки скачайте также `{model.name}` и положите в `models`. Запуск: `ClipnadoStudioLite.exe`.

**{len(cases)} тестов Windows без пропусков**, настоящие Whisper/Qwen, измерение звука и готовые MP4. Вложенный Python, интерфейс, анимационный рендер и экспорт переносимого пакета проверены отдельно. [Тесты]({validation_url}) · [Пакет]({run_url}). SHA256SUMS.txt и QA_REPORT.md прилагаются.

Физический ПК пользователя и его длинные стримы не проверены. Поиск Qwen использует текст речи; цветокоррекция не восстанавливает отсутствующие в исходнике детали, а анимации не гарантируют удержание зрителей.
"""
    (output / "release-notes.md").write_text(notes, encoding="utf-8")
    print(
        f"Windows release verified: {len(cases)} tests; ZIP {archive.stat().st_size} bytes",
        flush=True,
    )


if __name__ == "__main__":
    main()
