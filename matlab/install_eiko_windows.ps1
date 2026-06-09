# ==============================================================================
# Eiko Smart Windows Environment Installer (MATLAB Edition)
# ==============================================================================
param (
    [switch]$ElevatedSession  # Internal flag to track elevation loops
)

$ErrorActionPreference = "Inquire"

# 1. Ensure the script is running with Administrator privileges
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "[*] Requesting Administrator privileges for system checks..." -ForegroundColor Cyan
    
    # Run the administrative steps in a hidden/background window, then come back
    $argList = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -ElevatedSession"
    Start-Process powershell.exe -ArgumentList $argList -Verb RunAs -Wait
    
    return 
}

Write-Host "====================================================" -ForegroundColor Green
Write-Host " Starting Eiko MATLAB Environment Setup (Windows)   " -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Green

function Refresh-EnvPath {
    $machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
    
    $newPath = "$machinePath;$userPath"
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
Write-Host "`n[0/4] Checking NVIDIA Display Drivers..." -ForegroundColor Cyan

# CUDA 12.4 requires driver >= 550.52
$minDriver = 550
$driverValid = $false
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue

if ($nvidiaSmi) {
    $smiOutput = & $nvidiaSmi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1 | Out-String
    if ($smiOutput -match "(?m)^(\d+)") {
        $driverVer = [int]$matches[1]
        Write-Host "-> Found active NVIDIA driver: v$driverVer" -ForegroundColor DarkGray

        if ($driverVer -lt $minDriver) {
            Write-Host "-> Driver is too old for CUDA 12.4 (Requires v$minDriver+)." -ForegroundColor Yellow
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
    return
}

# ---------------------------------------------------------
# Step 1: Check and Install Targeted CUDA (Pinned to 12.4.1)
# ---------------------------------------------------------
Write-Host "`n[1/4] Checking CUDA installation (Pinned: 12.4.1)..." -ForegroundColor Cyan
$installCuda = $true
$targetCuda = "12.4.1"
$cudaCheck = Get-Command nvcc -ErrorAction SilentlyContinue

if ($cudaCheck) {
    $nvccOutput = nvcc --version | Out-String
    if ($nvccOutput -match "release (\d+\.\d+)") {
        $cudaVer = [version]$matches[1]
        if ($cudaVer -ge [version]"12.4") {
            Write-Host "-> Found CUDA $cudaVer. Skipping installation." -ForegroundColor Yellow
            $installCuda = $false
        }
    }
}

if ($installCuda) {
    Write-Host "-> Installing strict known-good matrix: CUDA $targetCuda..." -ForegroundColor Magenta
    winget install --id Nvidia.CUDA -v $targetCuda -e --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[!] Critical Error: CUDA installation failed with exit code $LASTEXITCODE." -ForegroundColor Red
        return
    }
    Refresh-EnvPath
}

# ---------------------------------------------------------
# Step 2: Locate MATLAB & Probe Required MSVC Version
# ---------------------------------------------------------
Write-Host "`n[2/4] Locating MATLAB and Probing Compiler Requirements..." -ForegroundColor Cyan

$matlabExe = (Get-Command matlab -ErrorAction SilentlyContinue).Source

if (-not $matlabExe) {
    Write-Host "[!] 'matlab' command not found in your system PATH." -ForegroundColor Red
    Write-Host "Please ensure MATLAB is installed and correctly added to your environment variables." -ForegroundColor Yellow
    if (-not $ElevatedSession) { Read-Host "`nPress Enter to exit" }
    return
}

Write-Host "-> MATLAB executable found at: $matlabExe" -ForegroundColor DarkGray
Write-Host "-> Booting MATLAB headlessly to query supported C++ compilers... (This may take 10-20 seconds)" -ForegroundColor Magenta

# One-liner to fetch the supported compilers, regex the year, and dump it to stdout.
$probeScript = "cc=mex.getCompilerConfigurations('C++','Supported'); for i=1:length(cc), m=regexp(cc(i).Name,'2022|2019|2017','match','once'); if ~isempty(m), fprintf('MSVC_YEAR:%s\n',m); return; end; end; fprintf('MSVC_YEAR:UNKNOWN\n');"

# Run the probe and capture output
$probeOutput = & $matlabExe -batch $probeScript

$msvcId = "Microsoft.VisualStudio.2022.BuildTools"
$msvcYear = "2022"

if ($probeOutput -match "MSVC_YEAR:(\d{4})") {
    $msvcYear = $matches[1]
    $msvcId = "Microsoft.VisualStudio.$msvcYear.BuildTools"
    Write-Host "-> MATLAB explicitly requested MSVC $msvcYear." -ForegroundColor Green
} else {
    Write-Host "-> MATLAB compiler query failed or returned unrecognized output. Defaulting to MSVC 2022." -ForegroundColor Yellow
}

# ---------------------------------------------------------
# Step 3: Check and Install Targeted MSVC Build Tools
# ---------------------------------------------------------
Write-Host "`n[3/4] Checking MSVC $msvcYear Build Tools..." -ForegroundColor Cyan
$msvcCheck = winget list --id $msvcId --accept-source-agreements | Out-String

if ($msvcCheck -match $msvcId) {
    Write-Host "-> MSVC $msvcYear Build Tools already installed. Skipping." -ForegroundColor Yellow
} else {
    Write-Host "-> Installing MSVC $msvcYear Build Tools..." -ForegroundColor Magenta
    winget install --id $msvcId -e --override "--passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[!] Critical Error: MSVC Build Tools installation failed with exit code $LASTEXITCODE." -ForegroundColor Red
        return
    }
    Refresh-EnvPath
}

# AUTOMATICALLY IMPORT MSVC ENVs FOR COMPILING
function Invoke-MsvcEnvironment {
    Write-Host "-> Integrating MSVC Developer paths into current shell..." -ForegroundColor DarkGray
    $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vsWhere) {
        # vswhere dynamically locates the highest available installed version that matches our workload
        $installPath = & $vsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($installPath -and (Test-Path "$installPath\Common7\Tools\Launch-VsDevShell.ps1")) {
            Import-Module "$installPath\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
            Enter-VsDevShell -VsInstallPath $installPath -SkipAutomaticLocation -DevCmdArguments "-arch=amd64 -host_arch=amd64" | Out-Null
            Write-Host "-> Success: MSVC environment activated!" -ForegroundColor Green
        } else {
            Write-Host "-> Warning: Could not find Launch-VsDevShell.ps1. MEX compilation may fail." -ForegroundColor Yellow
        }
    } else {
        Write-Host "-> Warning: vswhere.exe not found. Skipping path adjustments." -ForegroundColor Yellow
    }
}
Invoke-MsvcEnvironment

# ---------------------------------------------------------
# Step 4: Execute setup.m
# ---------------------------------------------------------
Write-Host "`n[4/4] Verifying Eiko via setup.m..." -ForegroundColor Cyan

$setupPath = Join-Path $PSScriptRoot "setup.m"

if (-not (Test-Path $setupPath)) {
    Write-Host "[!] Could not find 'setup.m' in the script directory." -ForegroundColor Red
    if (-not $ElevatedSession) { Read-Host "`nPress Enter to exit" }
    return
}

Write-Host "-> Booting MATLAB to run verification..." -ForegroundColor Magenta

# Construct the MATLAB command string cleanly
$matlabSafePath = $PSScriptRoot -replace '\\', '/'
$matlabCmd = "addpath('$matlabSafePath'); setup;"

# Execute statelessly (Wrapped in single quotes to protect the internal double quotes)
$matlabProc = Start-Process matlab -ArgumentList '-batch', "`"$matlabCmd`"" -Wait -PassThru -NoNewWindow

if ($matlabProc.ExitCode -eq 0) {
    Write-Host "`n[*] Verification Complete: Eiko for MATLAB is operational!" -ForegroundColor Green
} else {
    Write-Host "`n[!] Verification failed. MATLAB encountered an error while running setup.m." -ForegroundColor Red
}

if (-not $ElevatedSession) {
    Read-Host "`nPress Enter to exit"
}
