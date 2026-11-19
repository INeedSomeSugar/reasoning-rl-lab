param(
    [string]$Prefix = ('local17_' + (Get-Date -Format 'yyyyMMdd_HHmmss')),
    [string]$Config = 'configs/smoke.yaml'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:UV_CACHE_DIR = Join-Path $projectRoot '.cache\uv-tools'
$env:TEMP = Join-Path $projectRoot '.cache\tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path 'logs','environment',$env:TEMP | Out-Null
$history = [System.Collections.Generic.List[object]]::new()
function Invoke-Stage {
    param([string]$Stage, [string[]]$Arguments)
    $started = Get-Date
    $entry = [ordered]@{stage=$Stage; status='running'; started=$started.ToString('o'); arguments=$Arguments}
    $history.Add($entry)
    $history | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath "environment/$Prefix-status.json" -Encoding utf8
    & $python -m reasoning_rl @Arguments 2>&1 | Tee-Object -FilePath "logs/$Prefix-$Stage.log"
    $code = $LASTEXITCODE
    $entry.status = if ($code -eq 0) {'completed'} else {'failed'}
    $entry.finished = (Get-Date).ToString('o')
    $entry.wall_seconds = ((Get-Date)-$started).TotalSeconds
    $history | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath "environment/$Prefix-status.json" -Encoding utf8
    if ($code -ne 0) { throw "Stage $Stage failed with exit code $code; preserve log and use a new run name after repair." }
}
Invoke-Stage -Stage 'capability' -Arguments @('check-model','--name',"$Prefix-capability",'--config',$Config)
Invoke-Stage -Stage 'A' -Arguments @('evaluate','--name',"$Prefix-A",'--config',$Config)
Invoke-Stage -Stage 'B' -Arguments @('sft','--name',"$Prefix-B",'--config',$Config)
Invoke-Stage -Stage 'B-val' -Arguments @('evaluate','--name',"$Prefix-B-val",'--config',$Config,'--adapter',"checkpoints/$Prefix-B")
foreach ($arm in @(@('C','uniform'),@('D','dynamic'),@('E','static'))) {
    Invoke-Stage -Stage $arm[0] -Arguments @('grpo','--name',"$Prefix-$($arm[0])-s13",'--config',$Config,'--sft-checkpoint',"checkpoints/$Prefix-B",'--strategy',$arm[1])
}
Invoke-Stage -Stage 'compare' -Arguments @('compare','--runs',"$Prefix-C-s13","$Prefix-D-s13","$Prefix-E-s13",'--name',$Prefix)
Invoke-Stage -Stage 'report' -Arguments @('report')
