# Starts llama-server with given extra args, runs a request sequence through server_probe.py, then stops it.
# Usage: powershell -File scripts\server_test.ps1 -Label "x" -Sequence "plain,tools,plain" -Extra "-ncmoe","99"
param([string]$Label, [string]$Sequence, [string[]]$Extra, [string]$Build = "llama-cuda",
      [string]$Model = "models\Inkling-2-Mini-UD-Q2_K_XL-mainline.gguf")
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$a = @("-m", $Model, "-ngl", "99", "-t", "6", "-c", "8192", "-np", "1",
       "--jinja", "--chat-template-file", "templates\inkling2-chat-fixed.jinja", "--host", "127.0.0.1", "--port", "8080") + $Extra
$p = Start-Process "$root\toolchain\$Build\llama-server.exe" -ArgumentList $a -PassThru -WindowStyle Hidden `
     -RedirectStandardError "$root\build-tmp\srv.err" -RedirectStandardOutput "$root\build-tmp\srv.out"
try {
    for ($i = 0; $i -lt 150; $i++) {
        try { if ((Invoke-RestMethod http://127.0.0.1:8080/health -TimeoutSec 2).status -eq "ok") { break } } catch {}
        Start-Sleep 1
    }
    Write-Host "== $Label"
    py -I "$root\scripts\server_probe.py" $Sequence
} finally {
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep 2
}
