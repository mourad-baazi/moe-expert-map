# Starts llama-server with a model, runs quality_eval.py against it, stops the server.
# Usage: powershell -File scripts\quality_run.ps1 -Model models\x.gguf -Save results\quality\x.json [-Ref results\quality\full.json]
param([string]$Model, [string]$Save, [string]$Ref = "", [string[]]$Extra = @())
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$a = @("-m", $Model, "-ngl", "99", "-ncmoe", "99", "-t", "6", "-c", "2048", "-np", "1", "--host", "127.0.0.1", "--port", "8080") + $Extra
$p = Start-Process "$root\toolchain\llama-cuda\llama-server.exe" -ArgumentList $a -PassThru -WindowStyle Hidden `
     -RedirectStandardError "$root\build-tmp\q-srv.err" -RedirectStandardOutput "$root\build-tmp\q-srv.out"
try {
    for ($i = 0; $i -lt 150; $i++) {
        try { if ((Invoke-RestMethod http://127.0.0.1:8080/health -TimeoutSec 2).status -eq "ok") { break } } catch {}
        Start-Sleep 1
    }
    $args = @("-I", "$root\scripts\quality_eval.py", "--save", $Save)
    if ($Ref) { $args += @("--ref", $Ref) }
    py @args
} finally {
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep 2
}
