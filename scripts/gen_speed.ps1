# Measures real generation speed (tokens/s) for several placement/memory configs at a given context size.
# Usage: powershell -File scripts\gen_speed.ps1 [-Ctx 8192]
param([int]$Ctx = 8192)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$exe = "$root\toolchain\llama-cuda\llama-completion.exe"
function ot($layers) { "blk\.($layers)\.ffn_.*_exps=CUDA0,exps=CPU" }
$configs = [ordered]@{
  "A  9 looped layers, batch 512 (current chat.sh)" = @("-ot", (ot "9|1[0-7]"))
  "B 11 looped layers, batch 128"                   = @("-ot", (ot "[7-9]|1[0-7]"), "-ub", "128", "-b", "128")
  "C 12 looped layers, batch 128"                   = @("-ot", (ot "[6-9]|1[0-7]"), "-ub", "128", "-b", "128")
  "D 12 looped, batch 128, 8-bit context memory"    = @("-ot", (ot "[6-9]|1[0-7]"), "-ub", "128", "-b", "128", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0")
  "E 12 looped + layer 18, batch 128, 8-bit ctx"    = @("-ot", (ot "[6-9]|1[0-8]"), "-ub", "128", "-b", "128", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0")
}
$prompt = "Write a short paragraph explaining how a refrigerator keeps food cold."
foreach ($name in $configs.Keys) {
  $args = @("-m", "models\Inkling-2-Mini-UD-Q2_K_XL-mainline.gguf", "-ngl", "99", "-t", "6", "-c", "$Ctx", "-n", "128",
            "--temp", "0", "-no-cnv", "-p", $prompt) + $configs[$name]
  $out = & $exe @args 2>&1
  $speeds = ($out | Select-String "^\S+ I common_perf_print:\s+eval time" | ForEach-Object { if ($_ -match '([\d.]+) tokens per second') { $matches[1] } })
  $err = ($out | Select-String "out of memory|failed to allocate|error" | Select-Object -First 1)
  $vram = (nvidia-smi --query-gpu=memory.used --format=csv,noheader)
  Write-Host ("{0,-50} {1,8} tok/s  {2}" -f $name, ($speeds -join ","), ($(if ($err) { "ERROR: " + $err.Line.Substring(0, [Math]::Min(80, $err.Line.Length)) } else { "" })))
}
