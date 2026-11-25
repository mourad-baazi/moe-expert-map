# Traces every domain in data\corpus with the source-built CPU tracer (llama-moe-trace), which records valid
# router outputs for all 48 layer passes. (The C-API tracer on the prebuilt CUDA DLLs records NaN routing for
# layer passes 18+ on build b12622, so it is not used for results.)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force results\traces_v2 | Out-Null
foreach ($d in "chat", "code", "math", "arabic") {
  & "$root\llama.cpp\build\bin\llama-moe-trace.exe" -m models\Inkling-2-Mini-UD-Q2_K_XL-mainline.gguf `
      -f "data\corpus\$d.txt" --trace-out "results\traces_v2\$d.csv" -c 20480 -b 512 -t 6 2>&1 |
      Select-String "wrote|error|failed|keeping"
}
