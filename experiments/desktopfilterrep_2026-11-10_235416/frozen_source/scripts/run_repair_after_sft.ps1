param()
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'desktop_env.ps1')
Set-Location -LiteralPath $ProjectRoot
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$PythonExe=Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Prefix='desktoprepair_20261106_152712'
$Deadline=[DateTimeOffset]::Parse('2026-11-06T13:12:12+00:00')
while ($true) {
    if ([DateTimeOffset]::UtcNow -ge $Deadline) { throw 'Six-hour cleanup boundary reached while waiting for SFT' }
    $StatusPath="environment/$Prefix-sft-status.json"
    if (Test-Path -LiteralPath $StatusPath) {
        $State=Get-Content -Raw -LiteralPath $StatusPath | ConvertFrom-Json
        if ($State.status -eq 'failed') { throw 'SFT phase failed; preserve all evidence and inspect before proceeding' }
        if ($State.status -eq 'completed_sft_diagnostics') { break }
    }
    Start-Sleep -Seconds 5
}
& $PythonExe -u scripts/calibrate_repair_sampling.py *> "logs/$Prefix-calibration-runner.log"
if ($LASTEXITCODE -ne 0) { throw 'Calibration failed; main study not started' }
& $PythonExe -u scripts/freeze_repair_sampling.py *> "logs/$Prefix-sampling-freeze.log"
if ($LASTEXITCODE -ne 0) { throw 'Sampling freeze failed; main study not started' }
& $PythonExe -u scripts/run_repair_controlled.py --stage all *> "logs/$Prefix-controlled-runner.log"
if ($LASTEXITCODE -ne 0) { throw 'Controlled study stopped; preserve exact completed scope' }
