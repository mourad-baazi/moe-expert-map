# Reproduces the placement benchmark: default llama.cpp expert offload vs loop-aware placement.
# Needs the official llama.cpp CUDA build unpacked in toolchain\llama-cuda (release b12622 or later)
# and the model in models\. Usage:  powershell -ExecutionPolicy Bypass -File scripts\bench.ps1
$root = Split-Path -Parent $PSScriptRoot
$m = Join-Path $root "models\Inkling-2-Mini-UD-Q2_K_XL-mainline.gguf"
$b = Join-Path $root "toolchain\llama-cuda\llama-bench.exe"
$common = @("-m", $m, "-t", "6", "-ngl", "99", "-p", "0", "-n", "64", "-r", "3")

Write-Host "1) Default: experts of the last 10 layers on GPU (--n-cpu-moe 14)"
& $b @common -ncmoe 14

Write-Host "2) Loop-aware: experts of looped layers 8-17 on GPU (same VRAM)"
& $b @common -ot "blk\.([8-9]|1[0-7])\.ffn_.*_exps=CUDA0;exps=CPU"
