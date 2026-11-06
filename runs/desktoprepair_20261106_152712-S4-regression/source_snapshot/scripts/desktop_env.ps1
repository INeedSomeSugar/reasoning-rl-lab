# Dot-source from a project-local PowerShell process before installs and runs.
$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$env:UV_CACHE_DIR = Join-Path $ProjectRoot '.cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $ProjectRoot '.cache\python'
$env:UV_PYTHON_BIN_DIR = Join-Path $ProjectRoot '.cache\python-bin'
$env:UV_TOOL_DIR = Join-Path $ProjectRoot '.cache\uv-tools'
$env:UV_TOOL_BIN_DIR = Join-Path $ProjectRoot '.cache\uv-tool-bin'
$env:PIP_CACHE_DIR = Join-Path $ProjectRoot '.cache\pip'
$env:HF_HOME = Join-Path $ProjectRoot '.cache\huggingface'
$env:HF_DATASETS_CACHE = Join-Path $ProjectRoot '.cache\datasets'
$env:TORCH_HOME = Join-Path $ProjectRoot '.cache\torch'
$env:TRITON_CACHE_DIR = Join-Path $ProjectRoot '.cache\triton'
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $ProjectRoot '.cache\inductor'
$env:MIOPEN_CUSTOM_CACHE_DIR = Join-Path $ProjectRoot '.cache\miopen'
$env:MIOPEN_USER_DB_PATH = Join-Path $ProjectRoot '.cache\miopen-db'
$env:AMD_COMGR_CACHE_DIR = Join-Path $ProjectRoot '.cache\comgr'
$env:XDG_CACHE_HOME = Join-Path $ProjectRoot '.cache\xdg'
$env:MPLCONFIGDIR = Join-Path $ProjectRoot '.cache\matplotlib'
$env:TEMP = Join-Path $ProjectRoot '.cache\tmp'
$env:TMP = $env:TEMP
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
# Device 0 was verified as the RX 9070 XT. Hide the iGPU from this process only;
# otherwise Transformers Trainer automatically attempts DataParallel on both GPUs.
$env:HIP_VISIBLE_DEVICES = '0'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
New-Item -ItemType Directory -Force -Path $env:TEMP,$env:MIOPEN_CUSTOM_CACHE_DIR,$env:MIOPEN_USER_DB_PATH,$env:AMD_COMGR_CACHE_DIR | Out-Null
