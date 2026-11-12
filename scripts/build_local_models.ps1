param([switch]$SkipVulkan)
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root
function Checked([scriptblock]$Action) {
    & $Action
    if ($LASTEXITCODE -ne 0) { throw "Local engine build failed: $LASTEXITCODE" }
}
$Commit = '10e9780154365b191fb43ca4830659ef12def80f'
if (-not (Test-Path build/llama-src)) {
    Checked { git clone --depth 1 --branch b7099 https://github.com/ggml-org/llama.cpp.git build/llama-src }
}
Checked { git -C build/llama-src checkout $Commit }
$Options = @('-DGGML_NATIVE=OFF','-DGGML_AVX=ON','-DGGML_AVX2=OFF','-DGGML_BMI2=OFF',
    '-DGGML_FMA=OFF','-DGGML_F16C=OFF','-DGGML_AVX512=OFF','-DGGML_AVX_VNNI=OFF',
    '-DGGML_OPENMP=OFF','-DGGML_RPC=OFF','-DBUILD_SHARED_LIBS=OFF',
    '-DCMAKE_POLICY_VERSION_MINIMUM=3.10','-DCMAKE_POLICY_DEFAULT_CMP0091=NEW',
    '-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded')
Checked { cmake -S build/llama-src -B build/llama-cpu -G 'Visual Studio 17 2022' -A x64 @Options -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF }
Checked { cmake --build build/llama-cpu --config Release --target llama-cli -j 4 }
New-Item -ItemType Directory -Force tools,models | Out-Null
Get-ChildItem build/llama-cpu -Filter llama-cli.exe -Recurse | Select-Object -First 1 | Copy-Item -Destination tools/llama-cli.exe

$Model = 'Qwen_Qwen3-4B-Instruct-2507-Q3_K_M.gguf'
if (-not (Test-Path "models/$Model")) {
    Checked { curl.exe --fail --location --retry 2 --output "models/$Model.part" "https://huggingface.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF/resolve/ae44f08e1392f39c0e474af10c3ff8355c8b6688/$Model" }
    Move-Item "models/$Model.part" "models/$Model"
}
if ((Get-FileHash "models/$Model" -Algorithm SHA256).Hash.ToLower() -ne '35295e9a0a42eff5ebc592412c754991c0b6cc36bf4f282b2a491b211476752d') { throw 'Qwen checksum mismatch' }

if ($SkipVulkan) { return }
$Version = '1.4.313.2'
$TempFolder = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { Join-Path $Root 'build' }
$Installer = Join-Path $TempFolder 'VulkanSDK-1.4.313.2.exe'
if (-not (Test-Path "C:/VulkanSDK/$Version/Bin/glslc.exe")) {
    Checked { curl.exe --fail --location --retry 2 --output $Installer "https://sdk.lunarg.com/sdk/download/$Version/windows/vulkan-sdk.exe" }
    if ((Get-FileHash $Installer -Algorithm SHA256).Hash.ToLower() -ne '34a921d951858274ca8e470e9d0a3b7624db41216b3908ccea9f73c8a1b7500e') { throw 'Vulkan SDK checksum mismatch' }
    $Process = Start-Process -FilePath $Installer -ArgumentList '--accept-licenses','--default-answer','--confirm-command','install' -Wait -PassThru
    if ($Process.ExitCode -ne 0) { throw 'Vulkan SDK install failed' }
    Remove-Item $Installer
}
$env:VULKAN_SDK = "C:/VulkanSDK/$Version"
$env:PATH = "$env:VULKAN_SDK/Bin;$env:PATH"
Checked { cmake -S build/llama-src -B build/llama-vulkan -G 'Visual Studio 17 2022' -A x64 @Options -DGGML_VULKAN=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF }
Checked { cmake --build build/llama-vulkan --config Release --target llama-cli -j 4 }
New-Item -ItemType Directory -Force tools/vulkan | Out-Null
Get-ChildItem build/llama-vulkan -Filter llama-cli.exe -Recurse | Select-Object -First 1 | Copy-Item -Destination tools/vulkan/llama-cli.exe
Checked { cmake -S build/whisper-src -B build/whisper-vulkan -G 'Visual Studio 17 2022' -A x64 @Options -DGGML_VULKAN=ON -DWHISPER_CURL=OFF -DWHISPER_BUILD_TESTS=OFF }
Checked { cmake --build build/whisper-vulkan --config Release --target whisper-cli -j 4 }
Get-ChildItem build/whisper-vulkan -Filter whisper-cli.exe -Recurse | Select-Object -First 1 | Copy-Item -Destination tools/vulkan/whisper-cli.exe
