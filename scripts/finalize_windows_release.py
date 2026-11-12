"""Record native Windows validation and produce downloadable release assets."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    portable, output = (Path(p).resolve() for p in sys.argv[1:3])
    root = Path(__file__).resolve().parent.parent
    test_xml = root / "qa" / "test-results-windows.xml"
    cases = ET.parse(test_xml).findall(".//testcase")
    if len(cases) < 81 or any(c.find("failure") is not None or c.find("error") is not None or c.find("skipped") is not None for c in cases):
        raise RuntimeError("Complete Windows regression suite did not pass")
    run_url = f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    report = f"""# Проверка Windows — Clipnado Studio Lite 0.1.2

Проверка выполнена на GitHub Actions **Windows Server 2022 x64**: {run_url}

- {len(cases)} теста прошли, включая GUI, экспорт H.264/AAC, субтитры, реальные вызовы FFmpeg, распознавание русской речи и кэш.
- Регрессии 0.1.2: цель 10 клипов с порогом 60, сравнение альтернативных границ для набора, отсутствие повышения оценок и заполнения повторами, объяснение недобора, сохранение новых настроек и аудиопризнаков. На искусственном примере выбор интервалов даёт 10 клипов вместо пяти при жадном отборе; на отдельной составленной расшифровке находится не менее 10 разных эпизодов. Это технические примеры, не оценка интересности реальных видео.
- Сам переносимый пакет дополнительно проверен через вложенный Windows Embedded Python: импорты Qt, запуск FFmpeg/ffprobe/Whisper, GUI, ручной интервал и настоящий экспорт MP4.
- whisper.cpp и EXE-запускатель собраны нативно MSVC. Whisper: AVX включён, AVX2/BMI2/FMA/F16C/AVX512/CUDA отключены.
- ZIP проверен на целостность, контрольные суммы находятся в SHA256SUMS.txt.

Не проверены физический Xeon E5-2670, Windows 10 пользователя, GTX 1660/CUDA и человеческое качество отбора нескольких размеченных стримов. Это предварительный выпуск. Автоматические тесты не гарантируют точность отбора.

Сведения об исходной Linux-проверке сохранены ниже как исторический отчёт; её утверждения о ещё не выполненной Windows-проверке относятся к состоянию до этого запуска.

---

""" + (root / "QA_REPORT.md").read_text(encoding="utf-8")
    (portable / "QA_REPORT.md").write_text(report, encoding="utf-8")
    original_readme = (root / "README.md").read_text(encoding="utf-8")
    readme = f"""# Скачать и запустить Clipnado Studio Lite

Этот пакет собран и проверен на Windows Server 2022 x64: {run_url}

Полностью распакуйте ZIP и запустите **ClipnadoStudioLite.exe**. Python, FFmpeg и модель включены; для локального режима интернет не требуется. При проблеме запуска откройте diagnostics.cmd.

Комплектные whisper-cli.exe и запускатель собраны MSVC; прежние сведения о кросс-сборке LLVM-MinGW ниже описывают предыдущую Linux-сборку. Текущие результаты находятся в QA_REPORT.md. Совместимость с конкретным ПК пользователя и качество отбора реальных стримов требуют отдельной проверки.

---

""" + original_readme
    (portable / "README.md").write_text(readme, encoding="utf-8")
    shutil.copy2(test_xml, portable / "qa" / test_xml.name)
    for cache in list(portable.rglob("__pycache__")):
        if cache.is_dir():
            shutil.rmtree(cache)
    manifest_path = portable / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["windows_execution_verified"] = True
    manifest["windows_validation"] = {"os": "Windows Server 2022 x64", "tests_passed": len(cases), "workflow_url": run_url,
                                      "portable_gui_export": True, "compiler": "MSVC", "target_hardware_verified": False}
    manifest["files"] = {str(p.relative_to(portable)).replace("\\", "/"): digest(p) for p in sorted(portable.rglob("*"))
                         if p.is_file() and p != manifest_path}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "ClipnadoStudioLite-Windows-x64.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as package:
        for path in sorted(portable.rglob("*")):
            if path.is_file():
                package.write(path, Path("ClipnadoStudioLite") / path.relative_to(portable))
    with zipfile.ZipFile(archive) as package:
        if package.testzip():
            raise RuntimeError("Windows ZIP verification failed")
    subprocess.run(["git", "archive", "--format=zip", "--prefix=ClipnadoStudioLite-source/", "-o",
                    str(output / "ClipnadoStudioLite-source.zip"), "HEAD"], cwd=root, check=True)
    shutil.copy2(portable / "README.md", output / "START-HERE.md")
    shutil.copy2(portable / "QA_REPORT.md", output / "QA_REPORT.md")
    checks = [f"{digest(p)}  {p.name}" for p in sorted(output.iterdir()) if p.is_file() and p.name != "SHA256SUMS.txt"]
    (output / "SHA256SUMS.txt").write_text("\n".join(checks) + "\n", encoding="utf-8")
    notes = f"""Предварительный выпуск Clipnado Studio Lite 0.1.2.

Добавлена цель **10 разных клипов с оценкой от 60**. Отбор сравнивает варианты границ для всего набора, сохраняет контекст и завершение эпизода, исключает текстовые повторы. Оценки ради количества не повышаются; если подходящих эпизодов меньше, программа сообщает недобор. Десять интересных клипов из любого видео не гарантируются: результат зависит от длительности, событий и расшифровки.

Для повторного поиска откройте сохранённый проект, проверьте «Минимум (цель)» = 10 и «Оценка от» = 60, нажмите «Найти эпизоды». Сохранённая расшифровка используется без повторного распознавания. API для локального режима не требуется.

Скачайте **ClipnadoStudioLite-Windows-x64.zip** из Assets, полностью распакуйте и запустите **ClipnadoStudioLite.exe**. Python, FFmpeg и модель Whisper base включены.

**Проверено на Windows Server 2022 x64: {len(cases)} теста прошли.** Сам переносимый пакет проверен отдельно: вложенный Python/Qt, GUI и реальный экспорт MP4. [Результаты Windows-проверки]({run_url}).

Не проверены физический Xeon E5-2670, Windows 10 пользователя, GTX 1660/CUDA и качество отбора нескольких реальных размеченных стримов. Точность ИИ не гарантируется. Комплектный Whisper собран для AVX без AVX2/BMI2/FMA/F16C/AVX512, через CPU.

В Assets также находятся исходники, инструкция, подробный отчёт и SHA256SUMS.txt.
"""
    (output / "release-notes.md").write_text(notes, encoding="utf-8")
    print(f"Windows release verified: {len(cases)} tests; ZIP {archive.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    main()
