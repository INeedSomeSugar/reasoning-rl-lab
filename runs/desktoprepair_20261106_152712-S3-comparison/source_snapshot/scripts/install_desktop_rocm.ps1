param([string]$Attempt = 'local-wheels-01')
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'desktop_env.ps1')
Set-Location -LiteralPath $ProjectRoot
foreach ($LogPrefix in @('desktop-rocm','desktop-common','desktop-project','desktop-pip-check')) {
    if (Test-Path -LiteralPath "logs/$LogPrefix-$Attempt.log") { throw 'Preserve existing logs; use a new -Attempt value.' }
}
$UvExe = Join-Path $ProjectRoot '.cache\bootstrap\uv-bin\uv.exe'
$PythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$env:UV_HTTP_TIMEOUT = '30'
$env:UV_HTTP_RETRIES = '3'
$env:UV_CONCURRENT_DOWNLOADS = '4'
$Receipt = Get-Content -Raw -LiteralPath 'environment/desktop_amd_wheels.json' | ConvertFrom-Json
$WheelPaths = @()
foreach ($Wheel in $Receipt.wheels) {
    if (-not $Wheel.zip_crc_verified -or -not $Wheel.local_sha256) { throw "Wheel not yet verified: $($Wheel.filename)" }
    $WheelPath = Join-Path $ProjectRoot ".cache/wheels/$($Wheel.filename)"
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $WheelPath).Hash.ToLower() -ne $Wheel.local_sha256) {
        throw "Local wheel changed: $($Wheel.filename)"
    }
    $WheelPaths += $WheelPath
}
$InstallArgs = @('pip','install','--python',$PythonExe,'--index-url','https://stable.repo.amd.com/rocm/whl-next/') + $WheelPaths + @('torch[device-gfx1201]==2.13.0+rocm10.0.0','rocm==10.0.0')
& $UvExe @InstallArgs 2>&1 | Tee-Object -FilePath "logs/desktop-rocm-$Attempt.log"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$CommonArgs = @('pip','install','--python',$PythonExe,'-r','requirements/common.txt')
if (Test-Path -LiteralPath 'environment/desktop_common_wheels.json') {
    $CommonReceipt = Get-Content -Raw -LiteralPath 'environment/desktop_common_wheels.json' | ConvertFrom-Json
    foreach ($Wheel in $CommonReceipt.wheels) {
        $WheelPath = Join-Path $ProjectRoot ".cache/wheels/$($Wheel.filename)"
        if (-not $Wheel.verified -or (Get-FileHash -Algorithm SHA256 -LiteralPath $WheelPath).Hash.ToLower() -ne $Wheel.sha256) {
            throw "Common dependency is not verified: $($Wheel.filename)"
        }
    }
    $CommonArgs += @('--offline','--no-index','--find-links',(Join-Path $ProjectRoot '.cache/wheels'))
}
& $UvExe @CommonArgs 2>&1 | Tee-Object -FilePath "logs/desktop-common-$Attempt.log"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $UvExe pip install --python $PythonExe --no-deps -e . 2>&1 | Tee-Object -FilePath "logs/desktop-project-$Attempt.log"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $UvExe pip check --python $PythonExe 2>&1 | Tee-Object -FilePath "logs/desktop-pip-check-$Attempt.log"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $UvExe pip freeze --python $PythonExe | Set-Content -LiteralPath 'environment/requirements-desktop-rocm10.lock.txt' -Encoding utf8
& $UvExe pip list --python $PythonExe --format json | Set-Content -LiteralPath 'environment/packages-desktop-rocm10.json' -Encoding utf8
Write-Output 'Project dependencies installed and checked. GPU execution still requires independent validation.'
