$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONPYCACHEPREFIX = Join-Path $PSScriptRoot '.cache\pycache'
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot '.cache') | Out-Null
if (-not $env:OLLAMA_MODELS) {
    if (Test-Path -LiteralPath 'D:\models\manifests') {
        $env:OLLAMA_MODELS = 'D:\models'
    } else {
        $env:OLLAMA_MODELS = Join-Path $PSScriptRoot '.cache\ollama-models'
    }
}
try {
    $researchLoopExisting = Invoke-RestMethod -Uri 'http://127.0.0.1:8767/api/state' -TimeoutSec 1
    if ($researchLoopExisting.settings -and $researchLoopExisting.version) {
        Write-Host 'ResearchLoop is already running: http://127.0.0.1:8767'
        Start-Process 'http://127.0.0.1:8767'
        exit 0
    }
} catch { }
try {
    $null = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 2
} catch {
    $researchLoopOllama = Get-Command ollama -ErrorAction SilentlyContinue
    if ($researchLoopOllama) {
        Start-Process -FilePath $researchLoopOllama.Source -ArgumentList 'serve' -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $PSScriptRoot '.cache\ollama-start-stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot '.cache\ollama-start-stderr.log')
    }
}
Write-Host 'Open ResearchLoop: http://127.0.0.1:8767'
$researchLoopPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $researchLoopPython) {
    & $researchLoopPython (Join-Path $PSScriptRoot 'run.py')
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    py -3.12 (Join-Path $PSScriptRoot 'run.py')
} else {
    python (Join-Path $PSScriptRoot 'run.py')
}
