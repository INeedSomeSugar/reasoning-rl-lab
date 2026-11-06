param([string]$Prefix='localbudget_20261105', [string]$SftCheckpoint='checkpoints/localpilot2_20261105_B', [string]$Config='configs/local_pilot.yaml', [string]$FrozenPlan='experiments/local_budget_frozen_development.json', [switch]$OneSeed)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$python=Join-Path $PWD '.venv\Scripts\python.exe'
$statusPath=Join-Path $PWD "environment/$Prefix-status.json"
$status=[ordered]@{state='running';prefix=$Prefix;config=$Config;frozen_plan=$FrozenPlan;pid=$PID;started_utc=[DateTime]::UtcNow.ToString('o');completed=@();reused_completed_stages=@();current=$null;error=$null}
$frozenPlanRecord=Get-Content -Raw -LiteralPath $FrozenPlan | ConvertFrom-Json
function Save-Status { $status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding utf8 }
function Invoke-Stage([string]$Name,[string]$DoneFile,[string[]]$Arguments) {
    $actualCodeHash=& $python -c 'from reasoning_rl.runtime import code_hash; print(code_hash())'
    if ($LASTEXITCODE -ne 0 -or $actualCodeHash -ne $frozenPlanRecord.code_sha256) { throw 'Source/config changed after development-plan freeze. Preserve existing runs and investigate.' }
    $status.current=$Name
    Save-Status
    if (Test-Path -LiteralPath $DoneFile) {
        if ($Arguments[0] -eq 'grpo') {
            & $python -c 'import json,sys; from pathlib import Path; from reasoning_rl.runtime import ROOT,checkpoint_hash; result=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig")); assert checkpoint_hash(ROOT/"checkpoints"/sys.argv[2])==result["checkpoint_sha256"], "Completed metadata requires the matching real adapter weights"' $DoneFile $Name
            if ($LASTEXITCODE -ne 0) {throw "Missing or mismatched completed checkpoint: $Name"}
        }
        $status.completed+=@($Name);$status.reused_completed_stages+=@($Name);Save-Status;return
    }
    & $python -u -m reasoning_rl @Arguments 2>&1 | Tee-Object -FilePath "logs/$Name.log"
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $Name (exit $LASTEXITCODE). Preserve partial evidence and inspect before selecting a new run name." }
    if (-not (Test-Path -LiteralPath $DoneFile)) { throw "Stage returned without completion evidence: $Name" }
    $status.completed+=@($Name)
    Save-Status
}
try {
    Save-Status
    & $python -c 'import json,sys,yaml; from pathlib import Path; from reasoning_rl.runtime import ROOT,code_hash,sha256_file,checkpoint_hash; p=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig")); c=yaml.safe_load(Path(sys.argv[2]).read_text(encoding="utf-8-sig")); assert code_hash()==p["code_sha256"], "Frozen source hash mismatch"; assert c==p["config"], "Frozen configuration mismatch"; assert sha256_file(ROOT/"data/manifest.json")==p["data_manifest_sha256"], "Frozen data manifest mismatch"; assert checkpoint_hash(sys.argv[3])==p["sft_checkpoint_sha256"], "Frozen SFT checkpoint mismatch"; print("Frozen development protocol and real SFT weights verified")' $FrozenPlan $Config $SftCheckpoint
    if ($LASTEXITCODE -ne 0) {throw 'Frozen development preflight failed before GPU work'}
    $design=@(@{seed=13;arms=@('uniform','dynamic','static')})
    if (-not $OneSeed) { $design+=@(@{seed=29;arms=@('dynamic','uniform')},@{seed=47;arms=@('uniform','dynamic')}) }
    $labels=@{uniform='C';dynamic='D';static='E'}
    $trained=@()
    foreach ($pair in $design) {
        foreach ($strategy in $pair.arms) {
            $name="$Prefix-$($labels[$strategy])-s$($pair.seed)"
            Invoke-Stage $name "runs/$name/training_result.json" @('grpo','--name',$name,'--config',$Config,'--seed',"$($pair.seed)",'--sft-checkpoint',$SftCheckpoint,'--strategy',$strategy)
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
        Invoke-Stage $evaluationName "runs/$evaluationName/validation/metrics.json" @('evaluate','--name',$evaluationName,'--config',$Config,'--adapter',"checkpoints/$name")
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
