# ==============================================================================
# Eiko Smart Windows Environment Installer
# ==============================================================================
$ErrorActionPreference = "Stop"

# 1. Ensure the script is running with Administrator privileges
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "[*] Requesting Administrator privileges for system checks..." -ForegroundColor Cyan
    Start-Process powershell.exe -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs
    Exit
}

Write-Host "====================================================" -ForegroundColor Green
Write-Host " Starting Eiko Smart Environment Setup " -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Green

# Helper function to refresh PATH
function Refresh-EnvPath {
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}

# ---------------------------------------------------------
# Step 1: Check and Install CUDA (Requires >= 12.6)
# ---------------------------------------------------------
Write-Host "`n[1/5] Checking CUDA installation..." -ForegroundColor Cyan
$installCuda = $true
$cudaCheck = Get-Command nvcc -ErrorAction SilentlyContinue

if ($cudaCheck) {
    $nvccOutput = nvcc --version | Out-String
    if ($nvccOutput -match "release (\d+\.\d+)") {
        $cudaVer = [version]$matches[1]
        if ($cudaVer -ge [version]"12.6") {
            Write-Host "-> Found CUDA $cudaVer. Skipping installation." -ForegroundColor Yellow
            $installCuda = $false
        } else {
            Write-Host "-> Found CUDA $cudaVer, but 12.6+ is required. Upgrading..." -ForegroundColor Magenta
        }
    }
}

if ($installCuda) {
    winget install --id Nvidia.CUDA -v 12.6.0 -e --accept-package-agreements --accept-source-agreements
    Refresh-EnvPath
}

# ---------------------------------------------------------
# Step 2: Check and Install MSVC Build Tools
# ---------------------------------------------------------
Write-Host "`n[2/5] Checking MSVC C++ Build Tools..." -ForegroundColor Cyan
# Winget's list command is the safest way to check for the headless build tools
$msvcCheck = winget list --id Microsoft.VisualStudio.2022.BuildTools --accept-source-agreements | Out-String

if ($msvcCheck -match "Microsoft.VisualStudio.2022.BuildTools") {
    Write-Host "-> MSVC Build Tools already installed. Skipping." -ForegroundColor Yellow
} else {
    Write-Host "-> Installing MSVC Build Tools..." -ForegroundColor Magenta
    winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" --accept-package-agreements
    Refresh-EnvPath
}

# ---------------------------------------------------------
# Step 3: Check and Install Python (Requires >= 3.9)
# ---------------------------------------------------------
Write-Host "`n[3/5] Checking Python installation..." -ForegroundColor Cyan
$installPython = $true
$pyCheck = Get-Command python -ErrorAction SilentlyContinue

if ($pyCheck) {
    $pyOutput = python -c "import platform; print(platform.python_version())" 2>$null
    if ($pyOutput) {
        $pyVer = [version]$pyOutput
        if ($pyVer -ge [version]"3.9") {
            Write-Host "-> Found Python $pyVer. Skipping installation." -ForegroundColor Yellow
            $installPython = $false
        } else {
            Write-Host "-> Found Python $pyVer, but Eiko requires 3.9+. Upgrading to 3.12..." -ForegroundColor Magenta
        }
    }
}

if ($installPython) {
    winget install -e --id Python.Python.3.12 --scope machine --override "/passive Precompile=1 Include_debug=1 Include_symbols=1" --accept-package-agreements
    Refresh-EnvPath
}

# ---------------------------------------------------------
# Step 4: Virtual Environment Setup
# ---------------------------------------------------------
Write-Host "`n[4/5] Preparing Virtual Environment..." -ForegroundColor Cyan
cd $env:USERPROFILE
$venvPath = "$env:USERPROFILE\eiko"

# Prefer a globally installed 3.12 if the script just installed it, otherwise fallback to system python
$pythonExe = "C:\Program Files\Python312\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python" 
}

if (-not (Test-Path "$venvPath\Scripts\activate")) {
    Write-Host "-> Creating new virtual environment at $venvPath..." -ForegroundColor Magenta
    & $pythonExe -m venv eiko
} else {
    Write-Host "-> Virtual environment already exists. Using existing environment." -ForegroundColor Yellow
}

# ---------------------------------------------------------
# Step 5: Install Python Libraries (Pip handles idempotency)
# ---------------------------------------------------------
Write-Host "`n[5/5] Checking and Installing Eiko ML stack..." -ForegroundColor Cyan
Write-Host "-> Note: Pip will automatically skip packages that are already installed." -ForegroundColor DarkGray

& "$venvPath\Scripts\pip.exe" install --upgrade pip
& "$venvPath\Scripts\pip.exe" install torch torchvision --index-url https://download.pytorch.org/whl/cu126
& "$venvPath\Scripts\pip.exe" install eiko[jax]

# ---------------------------------------------------------
# Verification
# ---------------------------------------------------------
Write-Host "`n====================================================" -ForegroundColor Green
Write-Host " Verification Running... " -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Green
& "$venvPath\Scripts\python.exe" -c "import eiko.eiko_torch; import eiko.eiko_jax; print('-> Success: Eiko is fully operational!')"

Write-Host "`n[*] Installation Complete! You can close this window now." -ForegroundColor Green
Read-Host "Press Enter to exit"
