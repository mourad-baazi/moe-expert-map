# Greedy fact-completion accuracy, to compare compute paths (e.g. GPU vs CPU) of the same model file.
# Usage: powershell -File scripts\fact_check.ps1 -Label gpu -Extra "-ngl","99","-ncmoe","99"
param([string]$Label, [string[]]$Extra, [string]$Build = "llama-cuda",
      [string]$Model = "models\Inkling-2-Mini-UD-Q2_K_XL-mainline.gguf")
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$facts = @(
  @("The capital of Japan is", "Tokyo"), @("The capital of Italy is", "Rome"), @("The capital of Egypt is", "Cairo"),
  @("The capital of Canada is", "Ottawa"), @("The capital of Australia is", "Canberra"), @("The capital of Morocco is", "Rabat"),
  @("The chemical symbol for gold is", "Au"), @("The chemical symbol for sodium is", "Na"),
  @("The largest planet in our solar system is", "Jupiter"), @("The planet closest to the Sun is", "Mercury"),
  @("Water boils at sea level at a temperature of 100 degrees", "Celsius"), @("The author of Romeo and Juliet is William", "Shakespeare"),
  @("12 multiplied by 12 equals", "144"), @("The square root of 81 is", "9"), @("7 times 8 equals", "56"),
  @("The first man to walk on the Moon was Neil", "Armstrong"), @("The longest river in Africa is the", "Nile"),
  @("The programming language created by Guido van Rossum is", "Python"), @("The freezing point of water in Fahrenheit is", "32"),
  @("The smallest prime number is", "2")
)
$exe = "$root\toolchain\$Build\llama-completion.exe"
$ok = 0
foreach ($f in $facts) {
  $out = & $exe -m $Model -p $f[0] -n 8 --temp 0 -no-cnv -c 256 -t 6 @Extra 2>$null | Out-String
  $cont = $out.Substring([Math]::Min($out.Length, $out.IndexOf($f[0]) + $f[0].Length)).Trim() -replace "\s+", " "
  $hit = $cont -match [regex]::Escape($f[1])
  if ($hit) { $ok++ }
  Write-Host ("  {0} {1,-55} -> {2}" -f ($(if ($hit) { "OK " } else { "MISS" })), $f[0], $cont.Substring(0, [Math]::Min(40, $cont.Length)))
}
Write-Host "== $Label : $ok / $($facts.Count) correct"
