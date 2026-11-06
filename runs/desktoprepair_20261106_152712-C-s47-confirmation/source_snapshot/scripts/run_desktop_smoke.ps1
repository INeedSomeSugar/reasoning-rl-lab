param([string]$Prefix = 'desktoprocm10_20261105', [string]$Config = 'configs/desktop_rocm10_smoke.yaml')
$ErrorActionPreference = 'Stop'
if ($Prefix -notmatch '^[A-Za-z0-9_-]+$') { throw 'Prefix must be a simple run name.' }
. (Join-Path $PSScriptRoot 'desktop_env.ps1')
Set-Location -LiteralPath $ProjectRoot
$PythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$StatusPath = Join-Path $ProjectRoot "environment/$Prefix-status.json"
if (Test-Path -LiteralPath $StatusPath) { throw 'Preserve existing runner status; use a new prefix.' }
$State = [ordered]@{ status='running'; started_utc=[DateTime]::UtcNow.ToString('o'); prefix=$Prefix; config=$Config; completed=@(); holdout_evaluated=$false }
function Save-RunnerState { $State | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $StatusPath -Encoding utf8 }
function Invoke-Phase([string]$Phase, [string[]]$Arguments) {
    $State['phase'] = $Phase
    Save-RunnerState
    $LogPath = Join-Path $ProjectRoot "logs/$Prefix-$Phase.log"
    if (Test-Path -LiteralPath $LogPath) { throw "Existing log: $LogPath" }
    & $PythonExe @Arguments 2>&1 | Tee-Object -FilePath $LogPath
    if ($LASTEXITCODE -ne 0) {
        $State['status'] = 'failed'
        $State['exit_code'] = $LASTEXITCODE
        Save-RunnerState
        throw "Phase $Phase failed; later phases were not started."
    }
    $State['completed'] += $Phase
    Save-RunnerState
}
Save-RunnerState
try {
    Invoke-Phase 'gpu-preflight' @('-c', 'import torch; assert torch.version.hip and torch.cuda.is_available(), "ROCm GPU unavailable"; assert torch.cuda.device_count()==1, "Expose only the RX 9070 XT for this single-GPU protocol"; assert "9070 XT" in torch.cuda.get_device_name(0), "Device 0 must be RX 9070 XT"; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))')
    Invoke-Phase 'hardware' @('-m','reasoning_rl','hardware','--output',"environment/$Prefix-hardware.json")
    Invoke-Phase 'data-integrity' @('-m','reasoning_rl','prepare-data')
    Invoke-Phase 'checkpoint-integrity' @('scripts/verify_checkpoints.py')
    Invoke-Phase 'tests' @('-m','pytest','-q')
    Invoke-Phase 'capability' @('-m','reasoning_rl','check-model','--hardware','rocm','--name',"$Prefix-capability",'--config',$Config)
    Invoke-Phase 'A-val' @('-m','reasoning_rl','evaluate','--hardware','rocm','--name',"$Prefix-A-val",'--config',$Config)
    Invoke-Phase 'B' @('-m','reasoning_rl','sft','--hardware','rocm','--name',"$Prefix-B",'--config',$Config)
    $SharedSft = "checkpoints/$Prefix-B"
    Invoke-Phase 'B-val' @('-m','reasoning_rl','evaluate','--hardware','rocm','--name',"$Prefix-B-val",'--config',$Config,'--adapter',$SharedSft)
    foreach ($Arm in @(@{group='C'; strategy='uniform'},@{group='D'; strategy='dynamic'},@{group='E'; strategy='static'})) {
        $RunName = "$Prefix-$($Arm.group)-s13"
        Invoke-Phase "$($Arm.group)-s13" @('-m','reasoning_rl','grpo','--hardware','rocm','--name',$RunName,'--config',$Config,'--sft-checkpoint',$SharedSft,'--strategy',$Arm.strategy)
        Invoke-Phase "$($Arm.group)-reloadval" @('-m','reasoning_rl','evaluate','--hardware','rocm','--name',"$RunName-reloadval",'--config',$Config,'--adapter',"checkpoints/$RunName")
    }
    Invoke-Phase 'compare' @('-m','reasoning_rl','compare','--runs',"$Prefix-C-s13","$Prefix-D-s13","$Prefix-E-s13",'--name',"$Prefix-smoke-s13")
    Invoke-Phase 'evidence-audit' @('scripts/audit_evidence.py')
    $State['status'] = 'completed_smoke_only'
    $State['finished_utc'] = [DateTime]::UtcNow.ToString('o')
    Save-RunnerState
} catch {
    if ($State['status'] -eq 'running') { $State['status'] = 'interrupted_or_failed'; Save-RunnerState }
    throw
}
