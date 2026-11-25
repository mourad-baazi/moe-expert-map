# Starts llama-server (loop-aware setup), generates the chat, math and Arabic corpora, then stops the server.
# Usage: powershell -File scripts\run_corpus.ps1 [-Target 12000]
param([int]$Target = 12000)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force "$root\build-tmp" | Out-Null
$a = @("-m", "models\Inkling-2-Mini-UD-Q2_K_XL-mainline.gguf", "-ngl", "99",
       "-ot", "blk\.([8-9]|1[0-7])\.ffn_.*_exps=CUDA0,exps=CPU", "-t", "6", "-c", "8192", "-np", "1",
       "-ub", "128", "-b", "128", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
       "--jinja", "--chat-template-file", "templates\inkling2-chat-fixed.jinja", "--host", "127.0.0.1", "--port", "8080")
$p = Start-Process "$root\toolchain\llama-cuda\llama-server.exe" -ArgumentList $a -PassThru -WindowStyle Hidden `
     -RedirectStandardError "$root\build-tmp\corpus-srv.err" -RedirectStandardOutput "$root\build-tmp\corpus-srv.out"
try {
    for ($i = 0; $i -lt 150; $i++) {
        try { if ((Invoke-RestMethod http://127.0.0.1:8080/health -TimeoutSec 2).status -eq "ok") { break } } catch {}
        Start-Sleep 1
    }
    foreach ($d in "chat", "math", "arabic") { py -I "$root\scripts\gen_corpus.py" --domain $d --target $Target }
} finally {
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
}
