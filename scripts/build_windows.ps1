param([switch]$Cuda)
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root
function Checked([scriptblock]$Action) {
    & $Action
    if ($LASTEXITCODE -ne 0) { throw "External command failed: $LASTEXITCODE" }
}
# Requires Python 3.10–3.13 x64, Git, CMake, and Visual Studio Build Tools (C++).
# Run from Developer PowerShell for VS. Nothing enables billing or sends a transcript.
Checked { python -m venv .venv }
Checked { .\.venv\Scripts\python.exe -m pip install -e '.[dev]' }
if (-not (Test-Path 'build/whisper-src')) {
    Checked { git clone https://github.com/ggml-org/whisper.cpp.git build/whisper-src }
}
Checked { git -C build/whisper-src checkout 4979e04f5dcaccb36057e059bbaed8a2f5288315 }
$Gpu = if ($Cuda) { 'ON' } else { 'OFF' }
$Options = @('-DGGML_NATIVE=OFF','-DGGML_AVX=ON','-DGGML_AVX2=OFF','-DGGML_BMI2=OFF',
    '-DGGML_FMA=OFF','-DGGML_F16C=OFF','-DGGML_AVX512=OFF','-DGGML_AVX_VNNI=OFF',
    '-DGGML_OPENMP=OFF',"-DGGML_CUDA=$Gpu",'-DWHISPER_CURL=OFF','-DWHISPER_BUILD_TESTS=OFF',
    '-DWHISPER_BUILD_EXAMPLES=ON','-DBUILD_SHARED_LIBS=OFF')
if ($Cuda) { $Options += '-DCMAKE_CUDA_ARCHITECTURES=75' }
Checked { cmake -S build/whisper-src -B build/whisper-build -A x64 @Options }
Checked { cmake --build build/whisper-build --config Release --target whisper-cli -j 4 }
New-Item -ItemType Directory -Force tools,models | Out-Null
Get-ChildItem build/whisper-build -Filter whisper-cli.exe -Recurse | Select-Object -First 1 | Copy-Item -Destination tools/whisper-cli.exe
if ($Cuda) {
    Write-Host 'CUDA build needs matching CUDA runtime DLLs in tools or system PATH. CPU mode remains default in GUI.'
}
if (-not (Test-Path models/ggml-base.bin)) {
    Invoke-WebRequest 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin' -OutFile models/ggml-base.bin.part
    $Hash = (Get-FileHash models/ggml-base.bin.part -Algorithm SHA256).Hash.ToLower()
    if ($Hash -ne '60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe') { throw 'Whisper model SHA256 mismatch' }
    Move-Item models/ggml-base.bin.part models/ggml-base.bin
}
Write-Host 'Put ffmpeg.exe and ffprobe.exe (with libx264 and libass) in tools, or add FFmpeg to PATH.'
Checked { .\.venv\Scripts\python.exe -m pytest -q }
Checked { .\.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --windowed --name ClipnadoStudioLite --collect-submodules keyring.backends --collect-submodules win32ctypes --hidden-import PySide6.QtMultimediaWidgets run.py }
Copy-Item tools dist/ClipnadoStudioLite/tools -Recurse -Force
Copy-Item models dist/ClipnadoStudioLite/models -Recurse -Force
Copy-Item README.md,LICENSE,THIRD_PARTY.md dist/ClipnadoStudioLite/
Write-Host 'Build ready in dist/ClipnadoStudioLite. Verify it on the actual Windows PC before release.'
