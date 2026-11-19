$ErrorActionPreference = 'Stop'
$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$reg = [Microsoft.Win32.Registry]::LocalMachine
$windowsKey = $reg.OpenSubKey('SOFTWARE\Microsoft\Windows NT\CurrentVersion')
$windowsVersion = @{}
foreach ($field in @('ProductName','DisplayVersion','CurrentBuild','UBR')) {
    $windowsVersion[$field] = $windowsKey.GetValue($field)
}
$windowsKey.Close()
$displayKeys = $reg.OpenSubKey('SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}')
$displayMemory = @()
foreach ($subkey in $displayKeys.GetSubKeyNames()) {
    if ($subkey -notmatch '^\d{4}$') { continue }
    $key = $displayKeys.OpenSubKey($subkey)
    $memory = $key.GetValue('HardwareInformation.qwMemorySize')
    if ($memory -is [byte[]]) { $memory = [BitConverter]::ToUInt64($memory, 0) }
    $displayMemory += @{ description=$key.GetValue('DriverDesc'); memory_bytes=$memory; driver_version=$key.GetValue('DriverVersion') }
    $key.Close()
}
$displayKeys.Close()
$amdKey = $reg.OpenSubKey('SOFTWARE\AMD\CN')
$amdVersion = @{}
if ($null -ne $amdKey) {
    foreach ($field in @('DriverVersion','ReleaseVersion','BuildNumber')) { $amdVersion[$field] = $amdKey.GetValue($field) }
    $amdKey.Close()
}
$record = [ordered]@{
    collected_utc = [DateTime]::UtcNow.ToString('o')
    source_commit = (git -C $ProjectRoot rev-parse HEAD)
    os = Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,OSArchitecture,TotalVisibleMemorySize,FreePhysicalMemory
    cpu = Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors
    memory_modules = @(Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity,Speed,PartNumber)
    gpus = @(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,DriverDate,Status,ConfigManagerErrorCode)
    display_memory_registry = $displayMemory
    disks = @(Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,FileSystem,Size,FreeSpace)
    windows_registry = $windowsVersion
    amd_registry = $amdVersion
    wsl_distribution_registry_present = Test-Path -LiteralPath 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss'
    global_python_installed_by_this_task = $false
    system_driver_modified_by_this_task = $false
}
$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $ProjectRoot 'environment\desktop_host_20261105.json') -Encoding utf8
$record | ConvertTo-Json -Depth 8
