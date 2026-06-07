# ==============================================================================
# Eiko Smart Windows Environment Installer
# ==============================================================================
param (
    [string]$InheritedVenv = $env:VIRTUAL_ENV,
    [string]$InvokerProfile = $env:USERPROFILE,
    [string]$InvokerName = $env:USERNAME,
    [string]$InvokerDomain = $env:USERDOMAIN,
    [switch]$ElevatedSession  # Internal flag to track elevation loops
)

$ErrorActionPreference = "Stop"

# 1. Ensure the script is running with Administrator privileges
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "[*] Requesting Administrator privileges for system checks..." -ForegroundColor Cyan
    
    # Run the administrative steps in a hidden/background window, then come back
    $argList = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -InheritedVenv `"$InheritedVenv`" -InvokerProfile `"$InvokerProfile`" -InvokerName `"$InvokerName`" -InvokerDomain `"$InvokerDomain`" -ElevatedSession"
    Start-Process powershell.exe -ArgumentList $argList -Verb RunAs -Wait
    
    # --- Post-Elevation Hand-off to User Session ---
    $venvPath = "$InvokerProfile\eiko"
    if (-not [string]::IsNullOrWhiteSpace($InheritedVenv)) { $venvPath = $InheritedVenv }
    
    if (Test-Path "$venvPath\Scripts\Activate.ps1") {
        Write-Host "`n[*] Activating Eiko Environment..." -ForegroundColor Green
        & "$venvPath\Scripts\Activate.ps1"
    }
    
    # CHANGED: Use 'return' instead of 'Exit' so the terminal stays open
    return 
}

Write-Host "====================================================" -ForegroundColor Green
Write-Host " Starting Eiko Smart Environment Setup " -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Green

function Refresh-EnvPath {
    $machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
    
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
    if (-not $ElevatedSession) { Read-Host "`nPress Enter to exit" }
    return # CHANGED
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
	if ($LASTEXITCODE -ne 0) {
        Write-Host "[!] Critical Error: CUDA installation failed with exit code $LASTEXITCODE." -ForegroundColor Red
        return
    }
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
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[!] Critical Error: MSVC Build Tools installation failed with exit code $LASTEXITCODE." -ForegroundColor Red
        return
    }
	Refresh-EnvPath
}

# ---------------------------------------------------------
# Step 3: Check and Install Python (Requires >= 3.9)
# ---------------------------------------------------------
Write-Host "`n[3/5] Checking Python installation..." -ForegroundColor Cyan
$installPython = $true
$pyCheck = Get-Command python -ErrorAction SilentlyContinue

if ($pyCheck -and $pyCheck.Source -notmatch "WindowsApps") {
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
    if ($LASTEXITCODE -ne 0) {
		Write-Host "[!] Critical Error: Python installation failed with exit code $LASTEXITCODE." -ForegroundColor Red
		return
    }
	Refresh-EnvPath
}

# ---------------------------------------------------------
# Step 4: Virtual Environment Setup & Detection
# ---------------------------------------------------------
Write-Host "`n[4/5] Preparing Virtual Environment..." -ForegroundColor Cyan
$venvPath = "$InvokerProfile\eiko"

if (-not [string]::IsNullOrWhiteSpace($InheritedVenv)) {
    $venvPath = $InheritedVenv
    Write-Host "-> Detected active virtual environment at: $venvPath" -ForegroundColor Green
} else {
    if (Test-Path "$venvPath\Scripts\activate") {
        Write-Host "-> Found existing 'eiko' environment at $venvPath." -ForegroundColor Yellow
    } else {
        Write-Host "-> Creating new virtual environment at $venvPath..." -ForegroundColor Magenta
        $pythonExe = (Get-Command python -ErrorAction SilentlyContinue | Select-Object -First 1).Source
        if (-not $pythonExe) { 
            Write-Host "[!] Python executable not found in PATH." -ForegroundColor Red
            return 
        }
        
        & $pythonExe -m venv $venvPath

        $acl = Get-Acl $venvPath
        $rule = New-Object System.Security.AccessControl.FileSystemAccessRule("$InvokerDomain\$InvokerName", "FullControl", "ContainerInherit,ObjectInherit", "None", "Allow")
        $acl.AddAccessRule($rule)
        $acl.AddAccessRule($rule)
        Set-Acl -Path $venvPath -AclObject $acl
    }
}

if (Test-Path "$venvPath\Scripts\Activate.ps1") {
    Write-Host "-> Activating Eiko environment..." -ForegroundColor Green
    . "$venvPath\Scripts\Activate.ps1"
}

# ---------------------------------------------------------
# Step 5: Install Python Libraries
# ---------------------------------------------------------
Write-Host "`n[5/5] Checking existing Eiko ML Stack..." -ForegroundColor Cyan

function Run-PipCommand ([string[]]$PipArgs) {
    & "$venvPath\Scripts\pip.exe" $PipArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[!] Network or Package Error occurred during pip installation." -ForegroundColor Red
        if (-not $ElevatedSession) { Read-Host "Press Enter to exit" }
        throw "Pip command failed. Script halted." # CHANGED
    }
}

Run-PipCommand "install", "--upgrade", "pip", "--quiet"

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

Write-Host "-> Installing Eiko..." -ForegroundColor Magenta
Run-PipCommand "install", "eiko"

# ---------------------------------------------------------
# Verification
# ---------------------------------------------------------
Write-Host "`n====================================================" -ForegroundColor Green
Write-Host " Verification Running... " -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Green

& "$venvPath\Scripts\python.exe" -c "import eiko.eiko_torch; print('-> Success: Eiko and PyTorch CUDA layers are fully operational!')"

Write-Host "`n[*] Installation Complete!" -ForegroundColor Green

# Only pause if this is not part of the background elevated automation pipeline
if (-not $ElevatedSession) {
    Write-Host "Done. You are now inside the Eiko environment." -ForegroundColor Cyan
}