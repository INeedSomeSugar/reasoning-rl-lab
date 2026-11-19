param([string]$SftCheckpoint, [string]$Config='configs/main.yaml', [string]$Prefix='main', [ValidateSet('cuda','rocm')][string]$Hardware='cuda')
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$python='.venv\Scripts\python.exe'
if (-not $SftCheckpoint) { throw 'Supply the one fixed SFT checkpoint for all GRPO arms.' }
$orders=@(@('uniform','dynamic','static'), @('dynamic','static','uniform'), @('static','uniform','dynamic'))
$seeds=@(13,29,47)
for ($index=0; $index -lt $seeds.Count; $index++) {
    $seed=$seeds[$index]
    foreach ($strategy in $orders[$index]) {
        & $python -m reasoning_rl grpo --name "$Prefix-$strategy-s$seed" --config $Config --seed $seed --hardware $Hardware --sft-checkpoint $SftCheckpoint --strategy $strategy
        if ($LASTEXITCODE -ne 0) { throw "Failed: $strategy seed $seed" }
    }
}
# No automatic final-test execution: freeze a concrete plan after the predeclared validation rule.
