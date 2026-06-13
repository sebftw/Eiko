# ==============================================================================
# Eiko Smart Windows Environment Installer (MATLAB Edition)
# ==============================================================================
# DISCLAIMER:
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
param (
    [switch]$ElevatedSession  # Internal flag to track elevation loops
)

$ErrorActionPreference = "Continue"

# Unified exit handler
function Exit-Script {
    # if (-not $ElevatedSession) {
        Write-Host "`nPress any key to exit..." -ForegroundColor Cyan
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    # }
    exit
}

# 1. Ensure the script is running with Administrator privileges
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "[INFO] Requesting Administrator privileges for system checks..." -ForegroundColor Magenta
    
    # Run the steps as administrator.
    $argList = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -ElevatedSession"
    Start-Process powershell.exe -ArgumentList $argList -Verb RunAs -Wait
    
    return
}

Write-Host "====================================================" -ForegroundColor Cyan
Write-Host " Eiko MATLAB Installer (Windows)            " -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan

Write-Host "`nThis script will install all the requirements for Eiko by performing the following actions:" -ForegroundColor Gray
Write-Host "  1. Check if your NVIDIA graphics drivers are up to date." -ForegroundColor Gray
Write-Host "  2. Set up the correct NVIDIA CUDA tools (v12.4.1) for GPU computing." -ForegroundColor Gray
Write-Host "  3. Detect which C++ compiler your specific MATLAB version needs." -ForegroundColor Gray
Write-Host "  4. Download and install the necessary Microsoft C++ Build Tools." -ForegroundColor Gray
Write-Host "  5. Run a final test in MATLAB to ensure everything is working." -ForegroundColor Gray

# Disclaimer Notice
Write-Host "`n[!] DISCLAIMER: This script requires administrative privileges and modifies your system." -ForegroundColor Yellow
Write-Host "    It is provided 'as-is' without any express or implied warranties. Run at your own risk." -ForegroundColor Yellow


$choices = [System.Management.Automation.Host.ChoiceDescription[]] @(
    # Format: New-Object ChoiceDescription("Label", "Help Text")
    (New-Object System.Management.Automation.Host.ChoiceDescription("&Yes", "Accept the terms, grant admin rights, and begin installation.")),
    (New-Object System.Management.Automation.Host.ChoiceDescription("&No", "Cancel the setup immediately without installing anything."))
)

# 1 sets the default safe choice to "No"
$decision = $Host.UI.PromptForChoice("Confirmation", "Do you agree to these terms and want to proceed?", $choices, 1)

# $decision 0 is Yes, 1 is No
if ($decision -eq 1) {
    Write-Host "`n[*] Setup cancelled by user." -ForegroundColor Yellow
    Exit-Script
}

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
Write-Host "`n[1/5] Checking NVIDIA Display Drivers..." -ForegroundColor Cyan

# CUDA 12.4 requires driver >= 550.52
$minDriver = 550
$driverValid = $false
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue

if ($nvidiaSmi) {
    $smiOutput = & $nvidiaSmi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1 | Out-String
    if ($smiOutput -match "(?m)^(\d+)") {
        $driverVer = [int]$matches[1]
        Write-Host "  -> Found active NVIDIA driver: v$driverVer" -ForegroundColor DarkGray
        
        if ($driverVer -lt $minDriver) {
            Write-Host "  -> Driver is too old for CUDA 12.4 (Requires v$minDriver+)." -ForegroundColor Yellow
            $driverValid = $false
        } else {
            $driverValid = $true
        }
    }
} else {
    Write-Host "  -> No active NVIDIA display driver detected (nvidia-smi not found)." -ForegroundColor Yellow
    $driverValid = $false
}

if (-not $driverValid) {
    Write-Host "`n====================================================" -ForegroundColor Red
    Write-Host " CRITICAL ERROR: NVIDIA DRIVER UPDATE REQUIRED" -ForegroundColor Red
    Write-Host "====================================================" -ForegroundColor Red
    Exit-Script
}

# ---------------------------------------------------------
# Step 1: Check and Install Targeted CUDA (Pinned to 12.4.1)
# ---------------------------------------------------------
Write-Host "`n[2/5] Checking CUDA installation (Target: 12.4.1)..." -ForegroundColor Cyan

$installCuda = $true
$targetCuda = "12.4.1"
$cudaCheck = Get-Command nvcc -ErrorAction SilentlyContinue

if ($cudaCheck) {
    $nvccOutput = nvcc --version | Out-String
    if ($nvccOutput -match "release (\d+\.\d+)") {
        $cudaVer = [version]$matches[1]
        if ($cudaVer -ge [version]"12.4") {
            Write-Host "  -> Found CUDA $cudaVer. Skipping installation." -ForegroundColor Yellow
            $installCuda = $false
        }
    }
}

if ($installCuda) {
    Write-Host "  -> Installing target CUDA version ($targetCuda)..." -ForegroundColor Magenta
    winget install --id Nvidia.CUDA -v $targetCuda -e --accept-package-agreements --accept-source-agreements
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[!] Critical Error: CUDA installation failed with exit code $LASTEXITCODE." -ForegroundColor Red
        Exit-Script
    }
    Refresh-EnvPath
}

# ---------------------------------------------------------
# Step 2: Locate MATLAB & Probe Required MSVC Version
# ---------------------------------------------------------
Write-Host "`n[3/5] Locating MATLAB and Probing Compiler Requirements..." -ForegroundColor Cyan

$matlabExe = (Get-Command matlab -ErrorAction SilentlyContinue).Source
if (-not $matlabExe) {
    Write-Host "`n[!] 'matlab' command not found in your system PATH." -ForegroundColor Red
    Write-Host "Please ensure MATLAB is installed and correctly added to your environment variables." -ForegroundColor Yellow
    Exit-Script
}

Write-Host "  -> MATLAB executable found at: $matlabExe" -ForegroundColor DarkGray
Write-Host "  -> Querying MATLAB for supported C++ compilers (headless mode, ~10-20 seconds)..." -ForegroundColor Magenta

# One-liner to fetch the supported compilers, regex the year, and dump the latest one to stdout.
$probeScript = "cc=mex.getCompilerConfigurations('C++','Supported'); max_y=0; for i=1:length(cc), m=regexp(cc(i).Name,'(?<=Microsoft Visual C\+\+\s)\d{4}','match','once'); if ~isempty(m), max_y=max(max_y,str2double(m)); end; end; if max_y>0, fprintf('MSVC_YEAR:%d\n',max_y); else, fprintf('MSVC_YEAR:UNKNOWN\n'); end;"

# Run the probe and capture output
$probeOutput = & $matlabExe -batch $probeScript
$requestedYear = "2022" # Safe default if probe fails entirely

if ($probeOutput -match "MSVC_YEAR:(\d{4})") {
    $requestedYear = $matches[1]
    Write-Host "  -> MATLAB requested MSVC $requestedYear." -ForegroundColor Green
} else {
    Write-Host "  -> MATLAB compiler query failed or returned unrecognized output. Defaulting to MSVC 2022." -ForegroundColor Yellow
}

$msvcId = "Microsoft.VisualStudio.$requestedYear.BuildTools"
$legacyComponent = ""

# ---------------------------------------------------------
# Step 3: Validate and Install MSVC Build Tools
# ---------------------------------------------------------
Write-Host "`n[4/5] Validating MSVC $requestedYear availability via WinGet..." -ForegroundColor Cyan

# Initialize component tracking
$componentIdToCheck = "Microsoft.VisualStudio.Workload.VCTools" # Base default
$isLegacy = $false

# Check if WinGet actually hosts this exact year natively
$null = winget show --id $msvcId --accept-source-agreements 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  -> WinGet does not host a native package for MSVC $requestedYear." -ForegroundColor Yellow
    Write-Host "  -> Falling back to the 2022 Build Tools host with legacy toolset." -ForegroundColor DarkGray
    
    $isLegacy = $true
    # Map legacy years to their specific internal component IDs
    if ($requestedYear -eq "2019") {
        $componentIdToCheck = "Microsoft.VisualStudio.Component.VC.v142.x86.x64"
    } elseif ($requestedYear -eq "2017") {
        $componentIdToCheck = "Microsoft.VisualStudio.Component.VC.v141.x86.x64"
    } elseif ($requestedYear -eq "2015") {
        $componentIdToCheck = "Microsoft.VisualStudio.Component.VC.v140.x86.x64"
    }
    
    # Override the target package to the modern 2022 host
    $msvcId = "Microsoft.VisualStudio.2022.BuildTools"
}

# Check if the component/workload is already installed
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$isInstalled = $false

if (Test-Path $vsWhere) {
    # vswhere will return a path if the requested component/workload is present
    $existingPath = & $vsWhere -latest -products * -requires $componentIdToCheck -property installationPath
    if ($existingPath) { 
        $isInstalled = $true
    }
}

if ($isInstalled) {
    Write-Host "  -> MSVC toolset ($componentIdToCheck) is already installed. Skipping." -ForegroundColor Yellow
} else {
    Write-Host "  -> Installing/Modifying $msvcId to include required toolset..." -ForegroundColor Magenta
    
    # Construct arguments cleanly based on whether it's legacy or native
    if ($isLegacy) {
        $overrideArgs = "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --add $componentIdToCheck --includeRecommended"
    } else {
        $overrideArgs = "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
    }
    
    winget install --id $msvcId -e --override $overrideArgs --accept-source-agreements --accept-package-agreements
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[!] Critical Error: MSVC Build Tools installation failed with exit code $LASTEXITCODE." -ForegroundColor Red
        Exit-Script
    }
}

Refresh-EnvPath

# AUTOMATICALLY IMPORT MSVC ENVs FOR COMPILING
function Invoke-MsvcEnvironment {
    Write-Host "  -> Integrating MSVC Developer paths into the current shell..." -ForegroundColor Magenta
    $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    
    if (Test-Path $vsWhere) {
        # vswhere dynamically locates the highest available installed version that matches our workload
        $installPath = & $vsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($installPath -and (Test-Path "$installPath\Common7\Tools\Launch-VsDevShell.ps1")) {
            Import-Module "$installPath\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
            Enter-VsDevShell -VsInstallPath $installPath -SkipAutomaticLocation -DevCmdArguments "-arch=amd64 -host_arch=amd64" | Out-Null
            Write-Host "  -> Success: MSVC environment activated." -ForegroundColor Green
        } else {
            Write-Host "  -> Warning: Could not find Launch-VsDevShell.ps1. MEX compilation may fail." -ForegroundColor Yellow
        }
    } else {
        Write-Host "  -> Warning: vswhere.exe not found. Skipping path adjustments." -ForegroundColor Yellow
    }
}

Invoke-MsvcEnvironment

# ---------------------------------------------------------
# Step 4: Execute setup.m
# ---------------------------------------------------------
Write-Host "`n[5/5] Verifying Eiko via setup.m..." -ForegroundColor Cyan

$eikoPath  = Join-Path $PSScriptRoot "eiko"
$setupPath = Join-Path $eikoPath "+eiko_lib\setup.m"

if (-not (Test-Path -LiteralPath $setupPath)) {
    Write-Host "`n[!] Could not find 'setup.m' in the script directory ($setupPath)." -ForegroundColor Red
    Exit-Script
}

Write-Host "  -> Starting MATLAB to run system verification..." -ForegroundColor Magenta

# Construct the MATLAB command string cleanly
$matlabSafePath = $eikoPath -replace '\\', '/'
$matlabCmd = "addpath('$matlabSafePath'); eiko_lib.setup;"

# Execute statelessly (Wrapped in single quotes to protect the internal double quotes)
$matlabProc = Start-Process matlab -ArgumentList '-batch', "`"$matlabCmd`"" -Wait -PassThru -NoNewWindow

if ($matlabProc.ExitCode -eq 0) {
    Write-Host "`n[*] Verification Complete: Eiko for MATLAB is operational!" -ForegroundColor Green
} else {
    Write-Host "`n[!] Verification failed. MATLAB encountered an error while running setup.m." -ForegroundColor Red
}

Exit-Script
