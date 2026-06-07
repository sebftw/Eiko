# ==============================================================================
# Eiko Smart Windows Environment Installer
# ==============================================================================
# Capture the active virtual environment before elevation context is lost
param (
    [string]$InheritedVenv = $env:VIRTUAL_ENV,
    [string]$InvokerProfile = $env:USERPROFILE,
    [string]$InvokerName = $env:USERNAME,
    [string]$InvokerDomain = $env:USERDOMAIN
)

$ErrorActionPreference = "Stop"

# 1. Ensure the script is running with Administrator privileges
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "[*] Requesting Administrator privileges for system checks..." -ForegroundColor Cyan
    # Pass the current VIRTUAL_ENV to the elevated process so it isn't forgotten
    Start-Process powershell.exe -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -InheritedVenv `"$InheritedVenv`" -InvokerProfile `"$InvokerProfile`" -InvokerName `"$InvokerName`" -InvokerDomain `"$InvokerDomain`"" -Verb RunAs
    Exit
}

Write-Host "====================================================" -ForegroundColor Green
Write-Host " Starting Eiko Smart Environment Setup " -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Green

function Refresh-EnvPath {
    $machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
    
    # Rebuild the path, but ensure the current process paths (like venv) are kept
    $newPath = "$machinePath;$userPath"
    if ($env:VIRTUAL_ENV) {
        $venvScripts = "$env:VIRTUAL_ENV\Scripts"
        if ($newPath -notmatch [regex]::Escape($venvScripts)) {
            $newPath = "$venvScripts;$newPath"
        }
    }
    $env:Path = $newPath

    $systemCudaPath = [System.Environment]::GetEnvironmentVariable("CUDA_PATH", "Machine")
    if (-not [string]::IsNullOrWhiteSpace($systemCudaPath)) {
        $env:CUDA_PATH = $systemCudaPath
        $env:CUDA_HOME = $systemCudaPath
    }
}

# ---------------------------------------------------------
# Step 0: Check NVIDIA Display Drivers
# ---------------------------------------------------------
Write-Host "`n[0/5] Checking NVIDIA Display Drivers..." -ForegroundColor Cyan

$minDriver = 550
$driverValid = $false
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue

if ($nvidiaSmi) {
    # Extract the major version number (e.g., "551.23" becomes 551)
    $smiOutput = & $nvidiaSmi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1 | Out-String
    if ($smiOutput -match "(?m)^(\d+)") {
        $driverVer = [int]$matches[1]
        Write-Host "-> Found active NVIDIA driver: v$driverVer" -ForegroundColor DarkGray

        if ($driverVer -lt $minDriver) {
            Write-Host "-> Driver is too old for CUDA 12.6 (Requires v$minDriver+)." -ForegroundColor Yellow
            $driverValid = $false
        } else {
            $driverValid = $true
        }
    }
} else {
    Write-Host "-> No active NVIDIA display driver detected (nvidia-smi not found)." -ForegroundColor Yellow
    $driverValid = $false
}

if (-not $driverValid) {
    Write-Host "`n====================================================" -ForegroundColor Red
    Write-Host " CRITICAL: NVIDIA DRIVER UPDATE REQUIRED" -ForegroundColor Red
    Write-Host "====================================================" -ForegroundColor Red
    Write-Host "Your system requires NVIDIA display driver version $minDriver or higher to run." -ForegroundColor Yellow
    Write-Host "Because Windows hardware matching is highly specific, this cannot be safely automated." -ForegroundColor Magenta
    Write-Host "`nPlease update your drivers manually:" -ForegroundColor White
    Write-Host "  1. Open the 'NVIDIA App' or 'GeForce Experience' on your PC." -ForegroundColor White
    Write-Host "  2. Navigate to the 'Drivers' tab and install the latest update." -ForegroundColor White
    Write-Host "  3. Reboot your computer and run this script again." -ForegroundColor White
    Read-Host "`nPress Enter to exit"
    Exit
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
        }
    }
}

if ($installCuda) {
    Write-Host "-> Installing/Upgrading to CUDA 12.6..." -ForegroundColor Magenta
    winget install --id Nvidia.CUDA -v 12.6.0 -e --accept-package-agreements --accept-source-agreements
    Refresh-EnvPath
}

# ---------------------------------------------------------
# Step 2: Check and Install MSVC Build Tools
# ---------------------------------------------------------
Write-Host "`n[2/5] Checking MSVC C++ Build Tools..." -ForegroundColor Cyan
$msvcCheck = winget list --id Microsoft.VisualStudio.2022.BuildTools --accept-source-agreements | Out-String

if ($msvcCheck -match "Microsoft.VisualStudio.2022.BuildTools") {
    Write-Host "-> MSVC Build Tools already installed. Skipping." -ForegroundColor Yellow
} else {
    Write-Host "-> Installing MSVC Build Tools..." -ForegroundColor Magenta
    winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" --accept-source-agreements --accept-package-agreements
    Refresh-EnvPath
}

# ---------------------------------------------------------
# Step 3: Check and Install Python (Requires >= 3.9)
# ---------------------------------------------------------
Write-Host "`n[3/5] Checking Python installation..." -ForegroundColor Cyan
$installPython = $true
$pyCheck = Get-Command python -ErrorAction SilentlyContinue

if ($pyCheck -and (Get-Item $pyCheck.Source).Length -gt 0) {
    $pyOutput = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>$null
    if ($pyOutput) {
    if ($pyOutput -match "(\d+\.\d+\.\d+)") {
        $pyVer = [version]$matches[1]
        if ($pyVer -ge [version]"3.9") {
            Write-Host "-> Found Python $pyVer. Skipping installation." -ForegroundColor Yellow
            $installPython = $false
        }
    }
}
}

if ($installPython) {
    Write-Host "-> Installing Python 3.12..." -ForegroundColor Magenta
    winget install -e --id Python.Python.3.12 --accept-source-agreements --scope machine --override "/passive Precompile=1 Include_debug=1 Include_symbols=1 PrependPath=1" --accept-package-agreements
    Refresh-EnvPath
}

# ---------------------------------------------------------
# Step 4: Virtual Environment Setup & Detection
# ---------------------------------------------------------
Write-Host "`n[4/5] Preparing Virtual Environment..." -ForegroundColor Cyan
$venvPath = "$InvokerProfile\eiko"

if (-not [string]::IsNullOrWhiteSpace($InheritedVenv)) {
    # Scenario A: User already activated a venv before running the script
    $venvPath = $InheritedVenv
    Write-Host "-> Detected active virtual environment at: $venvPath" -ForegroundColor Green
    Write-Host "-> Skipping environment creation. Integrating directly." -ForegroundColor Yellow
} else {
    # Scenario B & C: No active venv, check for existing or create new
    if (Test-Path "$venvPath\Scripts\activate") {
        Write-Host "-> Found existing 'eiko' environment at $venvPath." -ForegroundColor Yellow
    } else {
        Write-Host "-> Creating new virtual environment at $venvPath..." -ForegroundColor Magenta
        # Ensure we only grab the first result if multiple python exes exist
        $pythonExe = (Get-Command python -ErrorAction SilentlyContinue | Select-Object -First 1).Source
        if (-not $pythonExe) { 
            Write-Host "[!] Python executable not found in PATH." -ForegroundColor Red
            Exit 
        }
        
        & $pythonExe -m venv $venvPath

        # Grant the standard user full control over the new venv
        $acl = Get-Acl $venvPath
        $rule = New-Object System.Security.AccessControl.FileSystemAccessRule("$InvokerDomain\$InvokerName", "FullControl", "ContainerInherit,ObjectInherit", "None", "Allow")
        $acl.AddAccessRule($rule)
        Set-Acl -Path $venvPath -AclObject $acl
    }
}

# ---------------------------------------------------------
# Step 5: Install Python Libraries (with Smart Bypass)
# ---------------------------------------------------------
Write-Host "`n[5/5] Checking existing Eiko ML Stack..." -ForegroundColor Cyan

function Run-PipCommand ([string[]]$PipArgs) {
    & "$venvPath\Scripts\pip.exe" $PipArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[!] Network or Package Error occurred during pip installation." -ForegroundColor Red
        Write-Host "Command failed: pip $($PipArgs -join ' ')" -ForegroundColor DarkGray
        Read-Host "Press Enter to exit"
        Exit
    }
}

Run-PipCommand "install", "--upgrade", "pip", "--quiet"

# Inline Python check for existing, valid PyTorch
$pythonCheck = @"
import sys
try:
    import torch
    v = torch.__version__.split('+')[0].split('.')
    if torch.version.cuda and (int(v[0]) > 2 or (int(v[0]) == 2 and int(v[1]) >= 4)):
        sys.exit(0)
except ImportError:
    pass
sys.exit(1)
"@

$pythonCheck | & "$venvPath\Scripts\python.exe"

if ($LASTEXITCODE -eq 0) {
    Write-Host "-> Found valid PyTorch (CUDA enabled, v2.4+). Skipping wheel downloads." -ForegroundColor Green
} else {
    Write-Host "-> Installing default ML target stack..." -ForegroundColor Magenta
    Run-PipCommand "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu126"
}

# Install Eiko
Write-Host "-> Installing Eiko..." -ForegroundColor Magenta
Run-PipCommand "install", "eiko"

# ---------------------------------------------------------
# Verification
# ---------------------------------------------------------
Write-Host "`n====================================================" -ForegroundColor Green
Write-Host " Verification Running... " -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Green

& "$venvPath\Scripts\python.exe" -c "import eiko.eiko_torch; print('-> Success: Eiko and PyTorch CUDA layers are fully operational!')"

Write-Host "`n[*] Installation Complete! You can close this window now." -ForegroundColor Green
Read-Host "Press Enter to exit"
