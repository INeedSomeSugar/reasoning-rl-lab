$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot '../scripts/desktop_env.ps1')
Set-Location -LiteralPath $ProjectRoot
$PythonExe=Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Prefix='desktoprepair_20261106_152712'
while ($true) {
    if ([DateTimeOffset]::UtcNow -ge [DateTimeOffset]::Parse('2026-11-06T13:12:12Z')) { throw 'Handoff reserve boundary reached' }
    $State=Get-Content -Raw -LiteralPath "environment/$Prefix-confirmation-status.json" | ConvertFrom-Json
    if ($State.status -eq 'completed_one_stage_confirmation') { break }
    if ($State.status -ne 'running') { throw "Confirmation stopped: $($State.status)" }
    Start-Sleep -Seconds 5
}
$Tasks=@(
    @{name='sft-forensics-final';file='analysis/repair_sft_forensics.py'},
    @{name='summary-final';file='analysis/summarize_repair_study.py'},
    @{name='audit-final';file='analysis/audit_repair_study.py'},
    @{name='grpo-attribution';file='analysis/repair_grpo_attribution.py'},
    @{name='documents-final';file='analysis/finalize_repair_documents.py'}
)
foreach ($Task in $Tasks) {
    $LogPath="logs/$Prefix-$($Task.name).log"
    if (Test-Path -LiteralPath $LogPath) { throw "Preserve existing final analysis log: $LogPath" }
    & $PythonExe -u $Task.file *> $LogPath
    if ($LASTEXITCODE -ne 0) { throw "Final analysis failed: $($Task.name)" }
}
