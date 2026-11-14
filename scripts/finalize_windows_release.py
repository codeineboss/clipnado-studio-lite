"""Require native inference tests before publishing the offline Windows package."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

MODEL_SHA = '35295e9a0a42eff5ebc592412c754991c0b6cc36bf4f282b2a491b211476752d'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    portable, output, model = (Path(p).resolve() for p in sys.argv[1:4])
    root = Path(__file__).resolve().parent.parent
    test_xml = root / 'qa' / 'test-results-windows.xml'
    cases = ET.parse(test_xml).findall('.//testcase')
    names = {c.attrib['name'] for c in cases}
    required = {'test_actual_qwen_content_query_and_cache', 'test_actual_qwen_neighbor_review_and_cache', 'test_actual_qwen_context_budget_without_generation', 'test_actual_qwen_semantic_selection_and_cache', 'test_actual_qwen_coarse_segments_and_cache',
                'test_windows_cancel_stops_descendant_process_and_releases_logs'}
    if len(cases) < 127 or not required <= names or any(
        c.find(tag) is not None for c in cases for tag in ('failure', 'error', 'skipped')
    ):
        raise RuntimeError('Complete Windows regression suite with actual local models did not pass')
    if digest(model) != MODEL_SHA or model.stat().st_size >= 2**31:
        raise RuntimeError('Separate Qwen release asset is invalid')
    run_url = f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    report = f'''# Проверка Clipnado Studio Lite 0.1.7

Windows Server 2022 x64 / GitHub Actions: {run_url}

- Пройдено тестов: {len(cases)}, без пропусков и ошибок. Проверены GUI, настоящий FFmpeg, экспорт H.264/AAC и субтитров, отмена дочерних процессов, кэш и сохранение проекта.
- Новое оформление: настоящий FFmpeg проверен для 9:16, 16:9 и 1:1 с размытым фоном, резким центром, исходным звуком и режимами off/SRT/ASS. Проверка пикселей сравнила резкость центра и фона; контрольная вспышка проверила синхронность. Выбор оформления сохраняется и восстанавливается в проекте.
- Поиск по содержанию и подбор соседних границ проверены настоящей Qwen на составленном русском диалоге: смысловой запрос, допустимые интервалы и повторное использование кэша. Модель может сохранить исходные границы. GUI дополнительно проверен с фикстурой: основные клипы сохраняются до добавления результатов, границы меняются только по нажатию, отмена возвращает исходник, устаревшие предложения после изменения текста или клипа отклоняются. Это проверка одного примера, не измерение точности на стримах.
- Сохранённая регрессия 0.1.6: полный запрос измеряется настоящим токенизатором/шаблоном чата llama.cpp без генерации (-n 0). Плотная искусственная расшифровка воспроизвела запрос свыше 8188 токенов. Автоматическое разбиение сохранило все реплики и оставило запас для ответа/повтора. Генерация длинного искусственного текста в этом тесте заменена фикстурой; тест подтверждает бюджет и покрытие, не интересность текста.
- Проверены счётчик 8471/8188 из сообщения пользователя, резерв ответа при 7900 входных токенах, переполнение после предварительной проверки, отсутствие ошибочного переключения GPU→CPU, кэш разделённых окон, перенос завершённых результатов 0.1.5 и одиночная слишком длинная реплика. Контекст остаётся 8192 токена; увеличения расхода VRAM нет.
- Реальный Whisper распознал человеческую русскую речь; реальная Qwen3-4B выбрала эпизод из составленного русского диалога. Второй запуск использовал кэш. Дополнительно настоящая Qwen обработала вариант с двумя крупными репликами по 50 секунд, где старый алгоритм не мог сформировать целиком по репликам клип 60–90 секунд. Это проверка выполнения моделей и двух вариантов одного примера, а не точности на пользовательских стримах.
- Переносимый пакет отдельно запущен через вложенный Python: Qt, видеокадры, воспроизведение, настоящий экспорт MP4 с размытым фоном, загрузка GGUF и генерация одного токена настоящей Qwen. Полный диалог проверен тестом выше; GUI-проводка обычного анализа, поиска по содержанию и обзора границ дополнительно проверена с подменённым ответом CLI при заблокированных Python-соединениях.
- CPU и Vulkan-движки Whisper/llama.cpp собраны MSVC. AVX включён, AVX2/BMI2/FMA/F16C/AVX512 выключены. Vulkan использует установленный драйвер; CUDA и сервер/API не нужны.
- GGUF поставляется отдельным файлом, его SHA-256 проверен. ZIP проверен на целостность; суммы всех файлов выпуска — в SHA256SUMS.txt.

Физические Xeon E5-2670, GTX 1660/Vulkan и Windows 10 пользователя не проверены. Скорость на этом ПК и точность на реальных длинных видео не измерены. Qwen анализирует расшифровку, не изображение: чисто визуальные моменты может пропустить. Цель 10 клипов с порогом 60 не означает гарантии для любого видео. Это предварительный выпуск.
'''
    (portable / 'QA_REPORT.md').write_text(report, encoding='utf-8')
    shutil.copy2(test_xml, portable / 'qa' / test_xml.name)
    for cache in list(portable.rglob('__pycache__')):
        if cache.is_dir():
            shutil.rmtree(cache)
    manifest_path = portable / 'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['windows_execution_verified'] = True
    manifest['windows_validation'] = {
        'os': 'Windows Server 2022 x64', 'tests_passed': len(cases), 'workflow_url': run_url,
        'actual_whisper': True, 'actual_qwen_dialogue': True, 'portable_qwen_token': True,
        'portable_gui_export': True, 'compiler': 'MSVC', 'target_hardware_verified': False,
        'physical_vulkan_verified': False, 'actual_content_query': True, 'actual_boundary_review': True,
        'blurred_export': True,
    }
    manifest['files'] = {
        str(p.relative_to(portable)).replace('\\', '/'): digest(p)
        for p in sorted(portable.rglob('*')) if p.is_file() and p != manifest_path
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    output.mkdir(parents=True, exist_ok=True)
    archive = output / 'ClipnadoStudioLite-Windows-x64.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as package:
        for path in sorted(portable.rglob('*')):
            if path.is_file():
                package.write(path, Path('ClipnadoStudioLite') / path.relative_to(portable))
    with zipfile.ZipFile(archive) as package:
        if package.testzip():
            raise RuntimeError('Windows ZIP verification failed')
    subprocess.run(['git', 'archive', '--format=zip', '--prefix=ClipnadoStudioLite-source/', '-o',
                    str(output / 'ClipnadoStudioLite-source.zip'), 'HEAD'], cwd=root, check=True)
    shutil.copy2(portable / 'README.md', output / 'START-HERE.md')
    shutil.copy2(portable / 'QA_REPORT.md', output / 'QA_REPORT.md')
    checks = [f'{digest(p)}  {p.name}' for p in sorted(output.iterdir())
              if p.is_file() and p.name not in ('SHA256SUMS.txt', 'release-notes.md')]
    checks.append(f'{MODEL_SHA}  {model.name}')
    (output / 'SHA256SUMS.txt').write_text('\n'.join(checks) + '\n', encoding='utf-8')
    notes = f'''Clipnado Studio Lite 0.1.7 — полностью локальные Whisper + Qwen3-4B.

Добавлены три возможности:

- **Поиск по содержанию:** введите, какие моменты найти. Qwen проверит расшифровку по смыслу, покажет отдельные результаты с соответствием запросу и предпросмотром. Добавляйте выбранные в основной список кнопкой «Добавить отмеченные».
- **Соседние границы:** выберите клип → «Подобрать границы». Просмотрите варианты со сдвигом до 30 секунд и примените подходящий. До применения клип сохраняется; есть отмена. Модель может рекомендовать оставить текущий интервал.
- **Видео по центру с размытым фоном:** вкладка «Экспорт» → «Оформление». Резкое видео целиком по центру, его увеличенная размытая копия заполняет фон синхронно. Старое оформление с тёмной зоной остаётся по умолчанию. Поддерживаются все форматы и субтитры, выбор сохраняется в проекте.

Обычный поиск, его совместимый кэш и защита от переполнения контекста сохранены. Для нового поискового запроса нужен новый проход Qwen по готовому тексту; подбор границ проверяет только соседний контекст.

**Обновление:** достаточно нового ZIP и прежнего GGUF-файла. Полностью распакуйте ZIP в отдельную папку и укажите прежнюю модель либо положите её в `models`. Старые проекты поддерживаются; повторно скачивать 2 ГБ не требуется.

**Для установки скачайте два файла из Assets:**
1. Полностью распакуйте **ClipnadoStudioLite-Windows-x64.zip**.
2. Скачайте **{model.name}** (2,08 ГБ) и положите в папку **models** внутри распакованной программы. Либо выберите файл кнопкой «Указать модель Qwen (.gguf)».
3. Запустите **ClipnadoStudioLite.exe**, выберите видео и нажмите «Найти эпизоды».

Python, FFmpeg, Whisper base и CPU/Vulkan-движки включены. После скачивания интернет не нужен. По умолчанию программа проверяет видеокарту через Vulkan и при необходимости переключается на CPU. Под Xeon E5-2670 исключены AVX2/FMA/F16C/BMI2. CUDA Toolkit устанавливать не нужно. При первом запуске возможна подготовка модели на диске; CPU-анализ может быть долгим.

**Windows Server 2022: пройдено {len(cases)} тестов без пропусков**, включая реальный Whisper и реальную Qwen: обычный отбор, смысловой запрос и подбор границ на составленном диалоге. Готовый пакет дополнительно проверен: настоящее выполнение Qwen, GUI, видео и экспорт. [Результаты]({run_url}).

Физические GTX 1660 / Xeon и скорость на пользовательском видео не проверены. Модель анализирует текст речи, а не изображение; точность и минимум 10 интересных клипов для любого видео не гарантируются. Подробности в QA_REPORT.md; контрольные суммы — SHA256SUMS.txt.
'''
    (output / 'release-notes.md').write_text(notes, encoding='utf-8')
    print(f'Windows release verified: {len(cases)} tests; ZIP {archive.stat().st_size} bytes', flush=True)


if __name__ == '__main__':
    main()
