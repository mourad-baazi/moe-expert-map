# Starts llama-server with chat.sh's settings plus extra args, asks 4 fixed questions, prints generation speed.
# Usage: powershell -File scripts\speed_run.ps1 -Label x [-Layers '[8-9]|1[0-7]'] [-Extra '--moe-cache-mib','1500'] [-Build llama-cuda]
param([string]$Label, [string]$Layers = '[8-9]|1[0-7]', [string]$Extra = "", [string]$Build = "llama-cuda",
      [string]$Model = "models\Inkling-2-Mini-UD-Q2_K_XL-mainline.gguf", [int]$Tokens = 300)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 2
$exe = if ($Build -eq "vulkan") { "$root\llama.cpp\build-vk\bin\llama-server.exe" } else { "$root\toolchain\$Build\llama-server.exe" }
if ($Build -eq "vulkan") { $env:PATH = "$root\toolchain\vulkan\mingw64\bin;$root\toolchain\w64devkit\bin;$env:PATH" }
$a = @("-m", $Model, "-ngl", "99", "-ot", "blk\.($Layers)\.ffn_.*_exps=CUDA0,exps=CPU", "-t", "12", "-c", "8192", "-np", "1",
       "-ub", "128", "-b", "128", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--host", "127.0.0.1", "--port", "8080") + @($Extra -split " " | Where-Object { $_ })
if ($Build -eq "vulkan") { $a = $a -replace '=CUDA0', '=Vulkan0' }
$p = Start-Process $exe -ArgumentList $a -PassThru -WindowStyle Hidden `
     -RedirectStandardError "$root\build-tmp\speed.err" -RedirectStandardOutput "$root\build-tmp\speed.out"
$qs = "Explain in 4 sentences how a refrigerator keeps food cold.",
      "Write a Python function that checks if a number is prime, with a short explanation.",
      "Explain in 4 sentences why the sky is blue.", "Give 3 tips for learning a new language, one sentence each."
try {
    $ok = $false
    for ($i = 0; $i -lt 240; $i++) {
        if ($p.HasExited) { break }
        try { if ((Invoke-RestMethod http://127.0.0.1:8080/health -TimeoutSec 2).status -eq "ok") { $ok = $true; break } } catch {}
        Start-Sleep 1
    }
    if (-not $ok) { "== $Label FAILED TO START"; Get-Content "$root\build-tmp\speed.err" -Tail 8; return }
    $speeds = foreach ($q in $qs) {
        $body = @{ prompt = $q; n_predict = $Tokens; temperature = 0; cache_prompt = $false } | ConvertTo-Json
        $r = Invoke-RestMethod http://127.0.0.1:8080/completion -Method Post -Body $body -ContentType "application/json" -TimeoutSec 600
        [math]::Round($r.timings.predicted_per_second, 1)
    }
    $vram = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    "== $Label : " + ($speeds -join ", ") + " tok/s | avg of last 3: " + [math]::Round((($speeds | Select-Object -Skip 1) | Measure-Object -Average).Average, 1) + " | VRAM used $vram MiB"
} finally {
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep 2
}
