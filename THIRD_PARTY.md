# Компоненты и лицензии

Код Clipnado Studio Lite 0.1.0 распространяется по GPL-3.0-or-later; полный текст в LICENSE. Все исходники приложения включены в поставку. Нет платных SDK, подписок или обязательных онлайн-служб.

| Компонент | Лицензия | Исходники / сведения |
|---|---|---|
| CPython 3.13.9, embedded x64 | PSF License | https://www.python.org/downloads/release/python-3139/ ; https://github.com/python/cpython/tree/v3.13.9 ; `runtime/LICENSE.txt` |
| PySide6 / Shiboken6 / Qt 6.8.3 | LGPL-3.0 / GPL, в зависимости от модуля | https://code.qt.io/cgit/pyside/pyside-setup.git/tree/?h=v6.8.3 ; https://download.qt.io/archive/qt/6.8/6.8.3/submodules/ ; динамические DLL и metadata wheel сохранены |
| FFmpeg 8.1.2 essentials, Gyan | GPL v3 (этой сборки) | https://www.gyan.dev/ffmpeg/builds/ ; исходный commit FFmpeg `38b88335f9` ; полная конфигурация в `tools/FFmpeg-README.txt`, лицензия в `tools/FFmpeg-LICENSE.txt`; исходники внешних библиотек — https://www.gyan.dev/ffmpeg/resources/ |
| whisper.cpp 1.8.2 / GGML | MIT | https://github.com/ggml-org/whisper.cpp/tree/4979e04f5dcaccb36057e059bbaed8a2f5288315 ; исходники и лицензия в `licenses/` |
| OpenAI Whisper base weights (GGML conversion) | MIT | https://github.com/openai/whisper ; https://huggingface.co/ggerganov/whisper.cpp ; SHA-256 `60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe` |
| HTTPX / HTTPCore / h11 / AnyIO / IDNA / typing_extensions | BSD / MIT | METADATA и лицензии соответствующих wheel в `runtime/Lib/site-packages/*dist-info` |
| certifi | MPL-2.0 (CA bundle) | https://github.com/certifi/python-certifi ; лицензия wheel сохранена |
| keyring / jaraco.* / more-itertools / pywin32-ctypes | MIT | лицензии wheel и https://github.com/jaraco/keyring ; https://github.com/enthought/pywin32-ctypes |
| LLVM-MinGW (только инструмент сборки) | LLVM / MIT и лицензии mingw runtime | https://github.com/mstorsjo/llvm-mingw/releases/tag/20250613 ; Windows whisper статически линкует C/C++ runtime; собственные уведомления runtime приложены |
| eSpeak NG (только генерация тестового звука) | GPL-3.0-or-later | https://github.com/espeak-ng/espeak-ng ; бинарники eSpeak в поставку не входят |
| FLEURS, тестовый русский образец (только QA) | CC-BY-4.0 | https://huggingface.co/datasets/google/fleurs ; Conneau et al., *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech* (2022); файл `10000611681338527501.wav`, ru_ru/test; публичный материал использован для проверки распознавания, не для обучения |

Используемые модули Qt — Core, Gui, Widgets, Multimedia, MultimediaWidgets. Замена DLL и отладка модифицированной библиотеки не ограничиваются. Поставляются исходники приложения и сценарий сборки. При дальнейшем распространении сохраняйте уведомления, тексты лицензий и доступ к соответствующим исходникам каждого компонента, включая библиотеки FFmpeg/Qt. API Google регулируется отдельными условиями сервиса и не требуется для локальной работы.
