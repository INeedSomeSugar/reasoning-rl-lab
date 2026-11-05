param([string]$Prefix='localbudget_20261105', [string]$SftCheckpoint='checkpoints/localpilot2_20261105_B', [switch]$OneSeed)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$python=Join-Path $PWD '.venv\Scripts\python.exe'
$statusPath=Join-Path $PWD "environment/$Prefix-status.json"
$status=[ordered]@{state='running';prefix=$Prefix;pid=$PID;started_utc=[DateTime]::UtcNow.ToString('o');completed=@();current=$null;error=$null}
$frozenPlan=Get-Content -Raw -LiteralPath 'experiments/local_budget_frozen_development.json' | ConvertFrom-Json
function Save-Status { $status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding utf8 }
function Invoke-Stage([string]$Name,[string]$DoneFile,[string[]]$Arguments) {
    $actualCodeHash=& $python -c 'from reasoning_rl.runtime import code_hash; print(code_hash())'
    if ($LASTEXITCODE -ne 0 -or $actualCodeHash -ne $frozenPlan.code_sha256) { throw 'Source/config changed after development-plan freeze. Preserve existing runs and investigate.' }
    $status.current=$Name
    Save-Status
    if (Test-Path -LiteralPath $DoneFile) { $status.completed+=@($Name);Save-Status;return }
    & $python -u -m reasoning_rl @Arguments 2>&1 | Tee-Object -FilePath "logs/$Name.log"
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $Name (exit $LASTEXITCODE). Preserve partial evidence and inspect before selecting a new run name." }
    if (-not (Test-Path -LiteralPath $DoneFile)) { throw "Stage returned without completion evidence: $Name" }
    $status.completed+=@($Name)
    Save-Status
}
try {
    Save-Status
    $design=@(@{seed=13;arms=@('uniform','dynamic','static')})
    if (-not $OneSeed) { $design+=@(@{seed=29;arms=@('dynamic','uniform')},@{seed=47;arms=@('uniform','dynamic')}) }
    $labels=@{uniform='C';dynamic='D';static='E'}
    $trained=@()
    foreach ($pair in $design) {
        foreach ($strategy in $pair.arms) {
            $name="$Prefix-$($labels[$strategy])-s$($pair.seed)"
            Invoke-Stage $name "runs/$name/training_result.json" @('grpo','--name',$name,'--config','configs/local_pilot.yaml','--seed',"$($pair.seed)",'--sft-checkpoint',$SftCheckpoint,'--strategy',$strategy)
            $trained+=@($name)
        }
        if ($pair.seed -eq 13) {
            $firstResults=@($trained | ForEach-Object {Get-Content -Raw -LiteralPath "runs/$_/training_result.json" | ConvertFrom-Json})
            if (@($firstResults | Where-Object {$_.adapter_updated}).Count -eq 0) {
                $status.state='stopped_no_learning_signal'
                $status.error='All three seed-13 arms completed with zero adapter update; preserve the negative pilot and inspect the train pool before more seeds.'
                return
            }
        }
    }
    foreach ($name in $trained) {
        $evaluationName="$name-reloadval"
        Invoke-Stage $evaluationName "runs/$evaluationName/validation/metrics.json" @('evaluate','--name',$evaluationName,'--config','configs/local_pilot.yaml','--adapter',"checkpoints/$name")
    }
    & $python -m reasoning_rl compare --runs @trained --name $Prefix
    if ($LASTEXITCODE -ne 0) {throw 'Comparison contract verification failed'}
    & $python -m reasoning_rl report
    if ($LASTEXITCODE -ne 0) {throw 'Report generation failed'}
    $status.state='completed'
    $status.current=$null
} catch {
    $status.state='failed'
    $status.error=$_.Exception.Message
    throw
} finally {
    $status['finished_utc']=[DateTime]::UtcNow.ToString('o')
    Save-Status
}
