param(
    [ValidateSet("cu126", "cu130", "cu132")]
    [string]$CudaVariant = "cu126"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

function Invoke-Checked {
    param([string]$Executable, [string[]]$CommandArguments)
    & $Executable @CommandArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE: $Executable $CommandArguments"
    }
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating .venv ..." -ForegroundColor Cyan
    Invoke-Checked "python" @("-m", "venv", (Join-Path $ProjectRoot ".venv"))
}

Write-Host "Installing CUDA-enabled PyTorch ($CudaVariant) ..." -ForegroundColor Cyan
Invoke-Checked $VenvPython @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked $VenvPython @("-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio")
Invoke-Checked $VenvPython @(
    "-m", "pip", "install", "torch==2.12.1",
    "--index-url", "https://download.pytorch.org/whl/$CudaVariant"
)
Invoke-Checked $VenvPython @("-m", "pip", "install", "PySide6>=6.7", "numpy>=1.26")
Invoke-Checked $VenvPython @("-m", "pip", "install", "-e", $ProjectRoot, "--no-deps")

Write-Host "Verifying CUDA ..." -ForegroundColor Cyan
Invoke-Checked $VenvPython @(
    (Join-Path $ProjectRoot "scripts\diagnose_accelerator.py"),
    "--require", "cuda"
)

Write-Host "CUDA is ready. Start with: .venv\Scripts\python -m deep_learning_studio" -ForegroundColor Green
