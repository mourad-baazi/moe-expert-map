# Builds llama.cpp (CPU) plus the moe-trace tool with a portable toolchain in ./toolchain.
# Usage (from the repo root):  powershell -ExecutionPolicy Bypass -File scripts\build.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$t = Join-Path $root "toolchain"
$env:PATH = "$t\w64devkit\bin;$((Get-ChildItem "$t\cmake-*\bin").FullName);$env:PATH"
$env:TEMP = Join-Path $root "build-tmp"; $env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null

$lc = Join-Path $root "llama.cpp"
if (-not (Test-Path $lc)) { git clone --depth 1 https://github.com/ggml-org/llama.cpp $lc }

# plug moe-trace into the llama.cpp build (once)
$hook = Join-Path $lc "examples\CMakeLists.txt"
$line = "add_subdirectory(`"$($root -replace '\\','/')/tools/moe-trace`" moe-trace)"
if (-not (Select-String -Path $hook -SimpleMatch "tools/moe-trace" -Quiet)) { Add-Content $hook "`n$line" }

cmake -S $lc -B "$lc\build" -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build "$lc\build" -j 12 --target llama-moe-trace llama-cli llama-bench
if ($LASTEXITCODE -ne 0) { throw "Build failed" }
Write-Host "Built: $lc\build\bin\llama-moe-trace.exe"
