# Компоненты и лицензии

Код Clipnado Studio Lite 0.1.7 распространяется по GPL-3.0-or-later; полный текст в LICENSE. Все исходники приложения включены в поставку. Нет платных SDK, подписок или обязательных онлайн-служб.

| Компонент | Лицензия | Исходники / сведения |
|---|---|---|
| CPython 3.13.9, embedded x64 | PSF License | https://www.python.org/downloads/release/python-3139/ ; https://github.com/python/cpython/tree/v3.13.9 ; `runtime/LICENSE.txt` |
| PySide6 / Shiboken6 / Qt 6.8.3 | LGPL-3.0 / GPL, в зависимости от модуля | https://code.qt.io/cgit/pyside/pyside-setup.git/tree/?h=v6.8.3 ; https://download.qt.io/archive/qt/6.8/6.8.3/submodules/ ; динамические DLL и metadata wheel сохранены |
| FFmpeg 8.1.2 essentials, Gyan | GPL v3 (этой сборки) | https://www.gyan.dev/ffmpeg/builds/ ; исходный commit FFmpeg `38b88335f9` ; полная конфигурация в `tools/FFmpeg-README.txt`, лицензия в `tools/FFmpeg-LICENSE.txt`; исходники внешних библиотек — https://www.gyan.dev/ffmpeg/resources/ |
| whisper.cpp 1.8.2 / GGML | MIT | https://github.com/ggml-org/whisper.cpp/tree/4979e04f5dcaccb36057e059bbaed8a2f5288315 ; исходники и лицензия в `licenses/` |
| OpenAI Whisper base weights (GGML conversion) | MIT | https://github.com/openai/whisper ; https://huggingface.co/ggerganov/whisper.cpp ; SHA-256 `60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe` |
| eSpeak NG (только генерация тестового звука) | GPL-3.0-or-later | https://github.com/espeak-ng/espeak-ng ; бинарники eSpeak в поставку не входят |
| FLEURS, тестовый русский образец (только QA) | CC-BY-4.0 | https://huggingface.co/datasets/google/fleurs ; Conneau et al., *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech* (2022); файл `10000611681338527501.wav`, ru_ru/test; публичный материал использован для проверки распознавания, не для обучения |

Используемые модули Qt — Core, Gui, Widgets, Multimedia, MultimediaWidgets. Замена DLL и отладка модифицированной библиотеки не ограничиваются. Поставляются исходники приложения и сценарий сборки. При дальнейшем распространении сохраняйте уведомления, тексты лицензий и доступ к соответствующим исходникам каждого компонента, включая библиотеки FFmpeg/Qt. Вызовов внешних API в версии 0.1.7 нет.

| Локальный компонент | Лицензия | Источник |
|---|---|---|
| llama.cpp b7099 / GGML | MIT | https://github.com/ggml-org/llama.cpp/tree/10e9780154365b191fb43ca4830659ef12def80f ; `licenses/llama-cpp-LICENSE` |
| Qwen3-4B-Instruct-2507 | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554 ; `licenses/Qwen3-LICENSE` |
| GGUF-квантизация Q3_K_M | Apache-2.0, исходная модель Qwen | https://huggingface.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF/tree/ae44f08e1392f39c0e474af10c3ff8355c8b6688 ; SHA-256 `35295e9a0a42eff5ebc592412c754991c0b6cc36bf4f282b2a491b211476752d` |
| Vulkan SDK 1.4.313.2 (только CI-сборка) | Лицензии Khronos/LunarG и инструментов SDK | https://vulkan.lunarg.com/sdk/home#windows ; SDK пользователю не поставляется, приложение использует системный Vulkan loader драйвера |
