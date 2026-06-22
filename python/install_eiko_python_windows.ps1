# ==============================================================================
# Eiko Smart Windows Environment Installer (CUDA 13.0 / Blackwell Target)
# ==============================================================================
param (
    [string]$InheritedVenv = $env:VIRTUAL_ENV,
    [string]$InvokerProfile = $env:USERPROFILE,
    [string]$InvokerName = $env:USERNAME,
    [string]$InvokerDomain = $env:USERDOMAIN,
    [switch]$ElevatedSession  # Internal flag to track elevation loops
)

$ErrorActionPreference = "Continue"

# Unified exit handler
function Exit-Script {
    Write-Host "`nPress any key to finish..." -ForegroundColor Cyan
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    
    # Only exit if we are in the elevated background worker
    if ($ElevatedSession) {
        exit
    }
}

# ---------------------------------------------------------
# Step 0: Explanation and User Confirmation (Runs Once)
# ---------------------------------------------------------
if (-not $ElevatedSession) {
    # Resolve the target path early for the display message
    $targetVenvPath = "$InvokerProfile\eiko"
    if (-not [string]::IsNullOrWhiteSpace($InheritedVenv)) { $targetVenvPath = $InheritedVenv }
    
    Write-Host "====================================================" -ForegroundColor Cyan
    Write-Host " Eiko Smart Environment Setup (Windows)             " -ForegroundColor Cyan
    Write-Host "====================================================" -ForegroundColor Cyan

    Write-Host "`nThis script will configure your system for the Eiko environment by performing the following actions:" -ForegroundColor Gray
    Write-Host "  1. Validate NVIDIA display driver compatibility (v575+)." -ForegroundColor Gray
    Write-Host "  2. Install or verify the NVIDIA CUDA Toolkit (>= v13.0)." -ForegroundColor Gray
    Write-Host "  3. Install Microsoft Visual C++ Build Tools." -ForegroundColor Gray
    Write-Host "  4. Install Python 3.12 (if not already present)." -ForegroundColor Gray
    Write-Host "  5. Configure a Python eiko virtual environment at:" -ForegroundColor Gray
    Write-Host "     -> $targetVenvPath" -ForegroundColor Cyan
    Write-Host "  6. Install the Eiko ML stack (PyTorch CU130, etc.)." -ForegroundColor Gray

    Write-Host "`n[!] DISCLAIMER: This script requires administrative privileges and modifies system variables." -ForegroundColor Yellow
    Write-Host "    It is provided 'as-is' without any express or implied warranties. Run at your own risk." -ForegroundColor Yellow

    $choices = [System.Management.Automation.Host.ChoiceDescription[]] @(
        (New-Object System.Management.Automation.Host.ChoiceDescription("&Yes", "Accept terms, grant admin rights, and begin installation.")),
        (New-Object System.Management.Automation.Host.ChoiceDescription("&No", "Cancel the setup safely."))
    )

    $decision = $Host.UI.PromptForChoice("Confirmation", "Do you agree to these terms and want to proceed?", $choices, 1)
    if ($decision -eq 1) {
        Write-Host "`n[*] Setup cancelled by user." -ForegroundColor Yellow
        Exit-Script
    }
}

# ---------------------------------------------------------
# Admin Check & Privilege Escalation
# ---------------------------------------------------------
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "`n[INFO] Requesting Administrator privileges for system installation..." -ForegroundColor Magenta
    
    # Run the administrative steps in an elevated window, then wait for it to finish
    $argList = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -InheritedVenv `"$InheritedVenv`" -InvokerProfile `"$InvokerProfile`" -InvokerName `"$InvokerName`" -InvokerDomain `"$InvokerDomain`" -ElevatedSession"
    $proc = Start-Process powershell.exe -ArgumentList $argList -Verb RunAs -Wait
    
    # When the elevated window closes, gracefully activate the environment in the parent shell (if it's still open)
    $venvPath = "$InvokerProfile\eiko"
    if (-not [string]::IsNullOrWhiteSpace($InheritedVenv)) { $venvPath = $InheritedVenv }
    
    if (Test-Path "$venvPath\Scripts\Activate.ps1") {
        . "$venvPath\Scripts\Activate.ps1"
    }
    
    # Silently close the background installer process
    exit
}

# (The banner prints again inside the elevated window for clarity)
if ($ElevatedSession) {
    Write-Host "====================================================" -ForegroundColor Cyan
    Write-Host " Eiko Smart Environment Setup (Elevated Worker)     " -ForegroundColor Cyan
    Write-Host "====================================================" -ForegroundColor Cyan
}

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
# Step 1: Check NVIDIA Display Drivers
# ---------------------------------------------------------
Write-Host "`n[1/6] Checking NVIDIA Display Drivers..." -ForegroundColor Cyan

$minDriver = 575
$driverValid = $false
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue

if ($nvidiaSmi) {
    $smiOutput = & $nvidiaSmi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1 | Out-String
    if ($smiOutput -match "(?m)^(\d+)") {
        $driverVer = [int]$matches[1]

        if ($driverVer -lt $minDriver) {
            # SOFT FAIL CASE
            Write-Host "  -> [!] NVIDIA driver v$driverVer is below the recommended v$minDriver for CUDA 13.0." -ForegroundColor Yellow
            $driverValid = $false
        } else {
            # SUCCESS CASE
            Write-Host "  -> [OK] NVIDIA driver v$driverVer detected (Meets v$minDriver+ requirement for CUDA 13.0)." -ForegroundColor Green
            Write-Host "  -> GPU environment is ready for Eiko installation."
            $driverValid = $true
        }
    }
} else {
    # MISSING DRIVER CASE
    Write-Host "  -> [!] No active NVIDIA display driver detected (nvidia-smi not found)." -ForegroundColor Yellow
    $driverValid = $false
}

if (-not $driverValid) {
    Write-Host "`n----------------------------------------------------" -ForegroundColor Yellow
    Write-Host " NOTICE: NVIDIA Driver Update Required              " -ForegroundColor Yellow
    Write-Host "----------------------------------------------------" -ForegroundColor Yellow
    Write-Host "To support the required CUDA environment, please update your Windows host driver:"
    Write-Host "  1. Open 'NVIDIA App' or 'GeForce Experience'."
    Write-Host "  2. Install the latest Game Ready or Studio driver (v$minDriver+)."
    Write-Host "  3. Rerun this script once the update is complete."
    Exit-Script
}

# ---------------------------------------------------------
# Step 2: Check and Install CUDA (Requires >= 13.0)
# ---------------------------------------------------------
Write-Host "`n[2/6] Checking CUDA installation (Target: 13.0)..." -ForegroundColor Cyan

$installCuda = $true
$cudaCheck = Get-Command nvcc -ErrorAction SilentlyContinue

if ($cudaCheck) {
    $nvccOutput = nvcc --version | Out-String
    if ($nvccOutput -match "release (\d+\.\d+)") {
        $cudaVer = [version]$matches[1]
        if ($cudaVer -ge [version]"13.0") {
            Write-Host "  -> Found CUDA $cudaVer. Skipping installation." -ForegroundColor Yellow
            $installCuda = $false
        }
    }
}

if ($installCuda) {
    Write-Host "  -> Installing/Upgrading to CUDA 13.0..." -ForegroundColor Magenta
    winget install --id Nvidia.CUDA -v 13.0.0 -e --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[!] Critical Error: CUDA installation failed with exit code $LASTEXITCODE." -ForegroundColor Red
        Exit-Script
    }
    Refresh-EnvPath
}

# ---------------------------------------------------------
# Step 3: Check and Install MSVC Build Tools
# ---------------------------------------------------------
Write-Host "`n[3/6] Checking MSVC C++ Build Tools..." -ForegroundColor Cyan

$msvcCheck = winget list --id Microsoft.VisualStudio.2022.BuildTools --accept-source-agreements | Out-String

if ($msvcCheck -match "Microsoft.VisualStudio.2022.BuildTools") {
    Write-Host "  -> MSVC Build Tools already installed. Skipping." -ForegroundColor Yellow
} else {
    Write-Host "  -> Installing MSVC Build Tools..." -ForegroundColor Magenta
    winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[!] Critical Error: MSVC Build Tools installation failed with exit code $LASTEXITCODE." -ForegroundColor Red
        Exit-Script
    }
    Refresh-EnvPath
}

# AUTOMATICALLY IMPORT MSVC ENVs FOR COMPILING
function Invoke-MsvcEnvironment {
    Write-Host "  -> Integrating MSVC Developer paths into current shell..." -ForegroundColor Magenta
    $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vsWhere) {
        $installPath = & $vsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($installPath -and (Test-Path "$installPath\Common7\Tools\Launch-VsDevShell.ps1")) {
            Import-Module "$installPath\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
            Enter-VsDevShell -VsInstallPath $installPath -SkipAutomaticLocation -DevCmdArguments "-arch=amd64 -host_arch=amd64" | Out-Null
            Write-Host "  -> Success: MSVC environment activated." -ForegroundColor Green
        } else {
            Write-Host "  -> Warning: Could not find Launch-VsDevShell.ps1. High-level ML libraries may fail to compile." -ForegroundColor Yellow
        }
    } else {
        Write-Host "  -> Warning: vswhere.exe not found. Skipping path adjustments." -ForegroundColor Yellow
    }
}
Invoke-MsvcEnvironment

# ---------------------------------------------------------
# Step 4: Check and Install Python (Requires >= 3.9)
# ---------------------------------------------------------
Write-Host "`n[4/6] Checking Python installation..." -ForegroundColor Cyan

$installPython = $true
$pyCheck = Get-Command python -ErrorAction SilentlyContinue

if ($pyCheck -and $pyCheck.Source -notmatch "WindowsApps") {
    $pyOutput = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>$null
    if ($pyOutput) {
        if ($pyOutput -match "(\d+\.\d+\.\d+)") {
            $pyVer = [version]$matches[1]
            if ($pyVer -ge [version]"3.9") {
                Write-Host "  -> Found Python $pyVer. Skipping installation." -ForegroundColor Yellow
                $installPython = $false
            }
        }
    }
}

if ($installPython) {
    Write-Host "  -> Installing Python 3.12..." -ForegroundColor Magenta
    winget install -e --id Python.Python.3.12 --accept-source-agreements --scope machine --override "/passive Precompile=1 Include_debug=1 Include_symbols=1 PrependPath=1" --accept-package-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[!] Critical Error: Python installation failed with exit code $LASTEXITCODE." -ForegroundColor Red
        Exit-Script
    }
    Refresh-EnvPath
}

# ---------------------------------------------------------
# Step 5: Virtual Environment Setup & Detection
# ---------------------------------------------------------
Write-Host "`n[5/6] Preparing Virtual Environment..." -ForegroundColor Cyan

$venvPath = "$InvokerProfile\eiko"

if (-not [string]::IsNullOrWhiteSpace($InheritedVenv)) {
    $venvPath = $InheritedVenv
    Write-Host "  -> Detected active virtual environment at: $venvPath" -ForegroundColor Green
} else {
    if (Test-Path "$venvPath\Scripts\activate") {
        Write-Host "  -> Found existing 'eiko' environment at $venvPath." -ForegroundColor Yellow
    } else {
        Write-Host "  -> Creating new virtual environment at $venvPath..." -ForegroundColor Magenta
        $pythonExe = (Get-Command python -ErrorAction SilentlyContinue | Select-Object -First 1).Source
        if (-not $pythonExe) { 
            Write-Host "`n[!] Python executable not found in PATH." -ForegroundColor Red
            Exit-Script
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
    Write-Host "  -> Activating Eiko environment internally..." -ForegroundColor Magenta
    . "$venvPath\Scripts\Activate.ps1"
}

# ---------------------------------------------------------
# Step 6: Install Python Libraries
# ---------------------------------------------------------
Write-Host "`n[6/6] Checking existing Eiko ML Stack..." -ForegroundColor Cyan

function Run-PipCommand ($PipArgs) {
    & "$venvPath\Scripts\python.exe" -m pip $PipArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[!] Network or Package Error occurred during pip installation." -ForegroundColor Red
        Exit-Script
    }
}

Run-PipCommand @("install", "--upgrade", "pip", "--quiet")

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
    Write-Host "  -> Found valid PyTorch (CUDA enabled, v2.4+). Skipping wheel downloads." -ForegroundColor Yellow
} else {
    Write-Host "  -> Installing default ML target stack (CUDA 13.0)..." -ForegroundColor Magenta
    Run-PipCommand "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu130"
}

Write-Host "  -> Installing Eiko..." -ForegroundColor Magenta
Run-PipCommand "install", "eiko"

# ---------------------------------------------------------
# Verification & Handoff
# ---------------------------------------------------------
Write-Host "`n====================================================" -ForegroundColor Green
Write-Host " Verification Running... " -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Green

& "$venvPath\Scripts\python.exe" -c "import eiko.eiko_torch; print('  -> Success: Eiko and PyTorch CUDA layers are fully operational!')"

# Create the quick-start launcher in the same directory as the installer
$launcherPath = "$PSScriptRoot\start_eiko.bat"
$batContent = "@echo off`ntitle Eiko Environment`ncall `"$venvPath\Scripts\activate.bat`"`ncmd /k"
Set-Content -Path $launcherPath -Value $batContent

Write-Host "`n====================================================" -ForegroundColor Green
Write-Host " EIKO ENVIRONMENT INSTALLED SUCCESSFULLY" -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Green

# 1. Address the immediate present: Get them ready to work right now
if (Test-Path "$venvPath\Scripts\Activate.ps1") {
    . "$venvPath\Scripts\Activate.ps1"
}

# 2. Address the future: Give them the simplest possible rule to remember
Write-Host ""
Write-Host "[!] How to use Eiko:" -ForegroundColor Yellow
Write-Host "    Simply double-click " -NoNewline
Write-Host "start_eiko.bat" -ForegroundColor Cyan -NoNewline
Write-Host "    (It will automatically open a ready-to-use terminal)`n" -ForegroundColor DarkGray

# Ensure the active window stays open until the user acknowledges
Exit-Script
