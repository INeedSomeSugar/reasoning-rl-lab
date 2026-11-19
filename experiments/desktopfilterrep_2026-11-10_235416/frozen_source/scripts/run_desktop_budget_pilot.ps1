param([string]$Prefix='desktoprocm10pilot_20261105', [string]$Config='configs/desktop_rocm10_bf16_pilot.yaml')
$ErrorActionPreference='Stop'
if ($Prefix -notmatch '^[A-Za-z0-9_-]+$') { throw 'Use a simple unique prefix' }
. (Join-Path $PSScriptRoot 'desktop_env.ps1')
Set-Location -LiteralPath $ProjectRoot
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$PythonExe=Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$StatusPath=Join-Path $ProjectRoot "environment/$Prefix-status.json"
$FrozenPlan="experiments/$Prefix-frozen-development.json"
if (Test-Path -LiteralPath $StatusPath) { throw 'Preserve existing status; investigate before resuming with a new family' }
$State=[ordered]@{status='running';prefix=$Prefix;config=$Config;started_utc=[DateTime]::UtcNow.ToString('o');completed=@();holdout_evaluated=$false}
function Save-State { $State | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $StatusPath -Encoding utf8 }
function Invoke-Phase([string]$Phase,[string[]]$Arguments) {
    $State['phase']=$Phase; Save-State
    $LogPath=Join-Path $ProjectRoot "logs/$Prefix-$Phase.log"
    if (Test-Path -LiteralPath $LogPath) { throw "Preserve existing log: $LogPath" }
    & $PythonExe -u @Arguments 2>&1 | Tee-Object -FilePath $LogPath
    if ($LASTEXITCODE -ne 0) { $State['exit_code']=$LASTEXITCODE; throw "Phase $Phase failed" }
    $State['completed']+=@($Phase); Save-State
}
try {
    Save-State
    Invoke-Phase 'preflight' @('-c','import torch; assert torch.version.hip and torch.cuda.device_count()==1 and "9070 XT" in torch.cuda.get_device_name(0) and torch.cuda.is_bf16_supported(); print("Single RX 9070 XT BF16 preflight passed")')
    Invoke-Phase 'A-val' @('-m','reasoning_rl','evaluate','--hardware','rocm','--name',"$Prefix-A-val",'--config',$Config)
    Invoke-Phase 'B' @('-m','reasoning_rl','sft','--hardware','rocm','--name',"$Prefix-B",'--config',$Config)
    $Sft="checkpoints/$Prefix-B"
    Invoke-Phase 'B-val' @('-m','reasoning_rl','evaluate','--hardware','rocm','--name',"$Prefix-B-val",'--config',$Config,'--adapter',$Sft)
    Invoke-Phase 'freeze-development' @('scripts/freeze_local_development.py','--prefix',$Prefix,'--config',$Config,'--development-plan','experiments/desktop_bf16_development_plan_20261105.json','--baseline',"$Prefix-A-val",'--sft-evaluation',"$Prefix-B-val",'--sft-checkpoint',$Sft,'--output',$FrozenPlan)
    $FrozenRecord=Get-Content -Raw -LiteralPath $FrozenPlan | ConvertFrom-Json
    $Design=@(@{seed=13;group='C';strategy='uniform'},@{seed=13;group='D';strategy='dynamic'},@{seed=13;group='E';strategy='static'},@{seed=29;group='D';strategy='dynamic'},@{seed=29;group='C';strategy='uniform'},@{seed=47;group='C';strategy='uniform'},@{seed=47;group='D';strategy='dynamic'})
    $Trained=@()
    foreach ($Arm in $Design) {
        $ActualHash=& $PythonExe -c 'from reasoning_rl.runtime import code_hash; print(code_hash())'
        if ($LASTEXITCODE -ne 0 -or $ActualHash -ne $FrozenRecord.code_sha256) { throw 'Source/config changed after development freeze' }
        $RunName="$Prefix-$($Arm.group)-s$($Arm.seed)"
        Invoke-Phase "$($Arm.group)-s$($Arm.seed)" @('-m','reasoning_rl','grpo','--hardware','rocm','--name',$RunName,'--config',$Config,'--seed',"$($Arm.seed)",'--sft-checkpoint',$Sft,'--strategy',$Arm.strategy)
        $Trained+=@($RunName)
        Invoke-Phase "$($Arm.group)-s$($Arm.seed)-reloadval" @('-m','reasoning_rl','evaluate','--hardware','rocm','--name',"$RunName-reloadval",'--config',$Config,'--seed',"$($Arm.seed)",'--adapter',"checkpoints/$RunName")
        if ($Arm.group -eq 'E') {
            $First=@($Trained | ForEach-Object { Get-Content -Raw -LiteralPath "runs/$_/training_result.json" | ConvertFrom-Json })
            if (@($First | Where-Object { $_.adapter_updated }).Count -eq 0) { $State['status']='stopped_no_learning_signal'; throw 'All seed-13 arms had zero adapter updates; preserve negative pilot before replication' }
        }
    }
    Invoke-Phase 'compare' (@('-m','reasoning_rl','compare','--runs')+$Trained+@('--name',$Prefix))
    Invoke-Phase 'summarize' @('scripts/summarize_budget_pilot.py','--prefix',$Prefix,'--baseline',"$Prefix-A-val",'--sft-evaluation',"$Prefix-B-val")
    Invoke-Phase 'evidence-audit' @('scripts/audit_evidence.py')
    $State['status']='completed_development_pilot'; $State['finished_utc']=[DateTime]::UtcNow.ToString('o'); Save-State
} catch {
    if ($State['status'] -eq 'running') { $State['status']='failed' }
    $State['error']=$_.Exception.Message; $State['finished_utc']=[DateTime]::UtcNow.ToString('o'); Save-State
    throw
}
