# Компоненты и лицензии

Код Clipnado Studio Lite 0.1.9 распространяется по GPL-3.0-or-later; полный текст в LICENSE. Все исходники приложения включены в поставку. Нет платных SDK, подписок или обязательных онлайн-служб.

| Компонент | Лицензия | Исходники / сведения |
|---|---|---|
| CPython 3.13.9, embedded x64 | PSF License | https://www.python.org/downloads/release/python-3139/ ; https://github.com/python/cpython/tree/v3.13.9 ; `runtime/LICENSE.txt` |
| PySide6 / Shiboken6 / Qt 6.8.3 | LGPL-3.0 / GPL, в зависимости от модуля | https://code.qt.io/cgit/pyside/pyside-setup.git/tree/?h=v6.8.3 ; https://download.qt.io/archive/qt/6.8/6.8.3/submodules/ ; динамические DLL и metadata wheel сохранены |
| FFmpeg 8.1.2 essentials, Gyan | GPL v3 (этой сборки) | https://www.gyan.dev/ffmpeg/builds/ ; исходный commit FFmpeg `38b88335f9` ; полная конфигурация в `tools/FFmpeg-README.txt`, лицензия в `tools/FFmpeg-LICENSE.txt`; исходники внешних библиотек — https://www.gyan.dev/ffmpeg/resources/ |
| whisper.cpp 1.8.2 / GGML | MIT | https://github.com/ggml-org/whisper.cpp/tree/4979e04f5dcaccb36057e059bbaed8a2f5288315 ; исходники и лицензия в `licenses/` |
| OpenAI Whisper base weights (GGML conversion) | MIT | https://github.com/openai/whisper ; https://huggingface.co/ggerganov/whisper.cpp ; SHA-256 `60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe` |
| eSpeak NG (только генерация тестового звука) | GPL-3.0-or-later | https://github.com/espeak-ng/espeak-ng ; бинарники eSpeak в поставку не входят |
| FLEURS, тестовый русский образец (только QA) | CC-BY-4.0 | https://huggingface.co/datasets/google/fleurs ; Conneau et al., *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech* (2022); файл `10000611681338527501.wav`, ru_ru/test; публичный материал использован для проверки распознавания, не для обучения |

Используемые модули Qt — Core, Gui, Widgets, Multimedia, MultimediaWidgets. Замена DLL и отладка модифицированной библиотеки не ограничиваются. Поставляются исходники приложения и сценарий сборки. При дальнейшем распространении сохраняйте уведомления, тексты лицензий и доступ к соответствующим исходникам каждого компонента, включая библиотеки FFmpeg/Qt. Вызовов внешних API в версии 0.1.9 нет.

| Локальный компонент | Лицензия | Источник |
|---|---|---|
| llama.cpp b7099 / GGML | MIT | https://github.com/ggml-org/llama.cpp/tree/10e9780154365b191fb43ca4830659ef12def80f ; `licenses/llama-cpp-LICENSE` |
| Qwen3-4B-Instruct-2507 | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554 ; `licenses/Qwen3-LICENSE` |
| GGUF-квантизация Q3_K_M | Apache-2.0, исходная модель Qwen | https://huggingface.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF/tree/ae44f08e1392f39c0e474af10c3ff8355c8b6688 ; SHA-256 `35295e9a0a42eff5ebc592412c754991c0b6cc36bf4f282b2a491b211476752d` |
| Vulkan SDK 1.4.313.2 (только CI-сборка) | Лицензии Khronos/LunarG и инструментов SDK | https://vulkan.lunarg.com/sdk/home#windows ; SDK пользователю не поставляется, приложение использует системный Vulkan loader драйвера |


| Автокадр / локальная библиотека | Лицензия | Источник |
|---|---|---|
| YuNet 2023mar, pretrained ONNX | MIT | https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet ; `clipnado/vision_models/LICENSE-YuNet.txt` |
| Light-ASD (CVPR 2023), pretrained AVA | MIT, © 2023 Liao Junhua | https://github.com/Junhua-Liao/Light-ASD/tree/ed38c232de5efe0261dbd68627c0ade7cdfe14eb ; `clipnado/vision_models/LICENSE-Light-ASD.txt`. Собственный экспорт стандартного ONNX-графа: `scripts/export_active_speaker.py`; источники, SHA и сравнение результатов: `clipnado/vision_models/provenance.json`. Прикладной код чужой программы не включён. |
| ONNX Runtime 1.22.1 CPU | MIT | https://github.com/microsoft/onnxruntime/tree/v1.22.1 ; license/notices внутри wheel и `onnxruntime` |
| OpenCV Python headless 4.11.0.86 | Apache-2.0 / MIT wrapper; сторонние компоненты см. wheel | https://github.com/opencv/opencv-python/tree/4.11.0.86 ; license/notices в `cv2` и `.dist-info` |
| NumPy 2.2.6 | BSD-3-Clause; bundled libraries см. notices | https://github.com/numpy/numpy/tree/v2.2.6 ; license/notices внутри wheel |
| coloredlogs / humanfriendly / pyreadline3 | MIT / MIT / BSD | Точные версии и SHA в `requirements-windows.lock`; тексты лицензий сохранены в metadata wheel |
| FlatBuffers / protobuf | Apache-2.0 / BSD-3-Clause | https://github.com/google/flatbuffers ; https://github.com/protocolbuffers/protobuf ; notices в wheel |
| SymPy / mpmath / packaging | BSD-3-Clause / BSD-3-Clause / Apache-2.0 или BSD | https://github.com/sympy/sympy ; https://github.com/mpmath/mpmath ; https://github.com/pypa/packaging ; лицензии в wheel |
| NASA, портрет Eileen Collins (только QA) | Public domain, NASA | `qa/fixtures/README.md`: используется для синтетического теста лиц; не является проверкой точности на разговорах |

PyTorch, ONNX-пакет построения графа, SciPy и python_speech_features использованы только при разработке/сравнении; не входят в runtime. Логика слежения, переключений, MFCC, кэша и экспорта написана самостоятельно. Приложение не загружает модели из интернета при запуске.
