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
    Write-Host "`nPress any key to exit..." -ForegroundColor Cyan
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit
}

Write-Host "====================================================" -ForegroundColor Cyan
Write-Host " Eiko MATLAB Installer (Windows)            " -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan

Write-Host "`nThis script will install all the requirements for Eiko by performing the following actions:" -ForegroundColor Gray
Write-Host "  1. Detect MATLAB and probe for compiler and CUDA requirements." -ForegroundColor Gray
Write-Host "  2. Check if your NVIDIA graphics drivers are up to date." -ForegroundColor Gray
Write-Host "  3. Set up the correct NVIDIA CUDA tools." -ForegroundColor Gray
Write-Host "  4. Download and install the necessary Microsoft C++ Build Tools." -ForegroundColor Gray
Write-Host "  5. Run a final test in MATLAB to ensure everything is working." -ForegroundColor Gray

# ---------------------------------------------------------
# Phase 1: User Confirmation (Runs in current privilege level)
# ---------------------------------------------------------

# If we are NOT in the auto-relaunched elevated session, show the intro and prompt.
if (-not $ElevatedSession) {
	
    # Disclaimer Notice
    Write-Host "`n[!] DISCLAIMER: This script requires administrative privileges and modifies your system." -ForegroundColor Yellow
    Write-Host "    It is provided 'as-is' without any express or implied warranties. Run at your own risk." -ForegroundColor Yellow 

    $choices = [System.Management.Automation.Host.ChoiceDescription[]] @(
        (New-Object System.Management.Automation.Host.ChoiceDescription("&Yes", "Accept the terms, grant admin rights, and begin installation.")),
        (New-Object System.Management.Automation.Host.ChoiceDescription("&No", "Cancel the setup immediately without installing anything."))
    )

    $decision = $Host.UI.PromptForChoice("Confirmation", "Do you agree to these terms and want to proceed?", $choices, 1)

    if ($decision -eq 1) {
        Write-Host "`n[*] Setup cancelled by user." -ForegroundColor Yellow
        Exit-Script
    }

    # ---------------------------------------------------------
    # Phase 2: Elevation Trigger
    # ---------------------------------------------------------
    # Now that they agreed, check if we actually have admin rights
    $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

    if (-not $isAdmin) {
        Write-Host "`n[INFO] Requesting Administrator privileges to begin installation..." -ForegroundColor Magenta
        
        # Relaunch the script as admin, passing the flag to skip the menu we just did
        $argList = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -ElevatedSession"
        Start-Process powershell.exe -ArgumentList $argList -Verb RunAs -Wait
        
        # Close the non-admin window since the new elevated window is taking over
        exit
    }
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
# Step 1: Locate MATLAB & Probe System Requirements
# ---------------------------------------------------------
Write-Host "`n[1/5] Locating MATLAB and Probing System Requirements..." -ForegroundColor Cyan

$matlabExe = (Get-Command matlab -ErrorAction SilentlyContinue).Source
if (-not $matlabExe) {
    Write-Host "`n[!] 'matlab' command not found in your system PATH." -ForegroundColor Red
    Write-Host "Please ensure MATLAB is installed and correctly added to your environment variables." -ForegroundColor Yellow
    Exit-Script
}

Write-Host "  -> MATLAB executable found at: $matlabExe" -ForegroundColor DarkGray
Write-Host "  -> Querying MATLAB for version and C++ compiler requirements (~10-20 seconds)..." -ForegroundColor Magenta

# Fetch supported compilers AND the MATLAB release version in one headless pass
$probeScript = "cc=mex.getCompilerConfigurations('C++','Supported'); max_y=0; for i=1:length(cc), m=regexp(cc(i).Name,'(?<=Microsoft Visual C\+\+\s)\d{4}','match','once'); if ~isempty(m), max_y=max(max_y,str2double(m)); end; end; fprintf('MSVC_YEAR:%d\n',max_y); fprintf('MATLAB_RELEASE:%s\n', version('-release'));"

# FIX: Pipe to Out-String so PowerShell treats the multi-line output as a single string, allowing $matches to populate correctly.
$probeOutput = & $matlabExe -batch $probeScript | Out-String

$requestedYear = "2022" # Default fallback
$matlabRelease = "UNKNOWN"
$matlabYear = 0

if ($probeOutput -match "MATLAB_RELEASE:(\d{4}[ab])") {
    $matlabRelease = $matches[1]
    $matlabYear = [int]($matlabRelease.Substring(0,4))
    Write-Host "  -> MATLAB Release R$matlabRelease detected." -ForegroundColor Green
}

if ($probeOutput -match "MSVC_YEAR:(\d{4})") {
    $requestedYear = $matches[1]
    Write-Host "  -> MATLAB requested MSVC $requestedYear." -ForegroundColor Green
} else {
    Write-Host "  -> MATLAB compiler query failed. Defaulting to MSVC 2022." -ForegroundColor Yellow
}

$msvcId = "Microsoft.VisualStudio.$requestedYear.BuildTools"

# ---------------------------------------------------------
# Step 2: Check NVIDIA Display Drivers
# ---------------------------------------------------------
Write-Host "`n[2/5] Checking NVIDIA Display Drivers..." -ForegroundColor Cyan

# CUDA 12.8 requires driver >= 570.00 (Even if MATLAB ships with CUDA, the host driver is still required)
$minDriver = 570
$driverValid = $false
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue

if ($nvidiaSmi) {
    $smiOutput = & $nvidiaSmi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1 | Out-String
    if ($smiOutput -match "(?m)^(\d+)") {
        $driverVer = [int]$matches[1]
        
        if ($driverVer -lt $minDriver) {
            Write-Host "  -> [!] NVIDIA driver v$driverVer is below the recommended v$minDriver for CUDA 12.8." -ForegroundColor Yellow
            $driverValid = $false
        } else {
            Write-Host "  -> [OK] NVIDIA driver v$driverVer detected (Meets v$minDriver+ requirement)." -ForegroundColor Green
            $driverValid = $true
        }
    }
} else {
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
# Step 3: Check and Install Targeted CUDA
# ---------------------------------------------------------
Write-Host "`n[3/5] Checking system CUDA requirements..." -ForegroundColor Cyan

$installCuda = $true
$targetCuda = "12.8.2"

if ($matlabYear -ge 2026) {
    Write-Host "  -> Built-in CUDA detected (MATLAB R$matlabRelease). Skipping system CUDA installation." -ForegroundColor Green
    $installCuda = $false
} else {
    $cudaCheck = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($cudaCheck) {
        $nvccOutput = nvcc --version | Out-String
        if ($nvccOutput -match "release (\d+\.\d+)") {
            $cudaVer = [version]$matches[1]
            if ($cudaVer -ge [version]"12.8") {
                Write-Host "  -> Found system CUDA $cudaVer. Skipping installation." -ForegroundColor Yellow
                $installCuda = $false
            }
        }
    }
}

if ($installCuda) {
    Write-Host "  -> Installing target system CUDA version ($targetCuda)..." -ForegroundColor Magenta
    winget install --id Nvidia.CUDA -v $targetCuda -e --accept-package-agreements --accept-source-agreements
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[!] Critical Error: CUDA installation failed with exit code $LASTEXITCODE." -ForegroundColor Red
        Exit-Script
    }
    Refresh-EnvPath
}

# ---------------------------------------------------------
# Step 4: Validate and Install MSVC Build Tools
# ---------------------------------------------------------
Write-Host "`n[4/5] Validating MSVC $requestedYear availability via WinGet..." -ForegroundColor Cyan

$componentIdToCheck = "Microsoft.VisualStudio.Workload.VCTools"
$isLegacy = $false

$null = winget show --id $msvcId --accept-source-agreements 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  -> WinGet does not host a native package for MSVC $requestedYear." -ForegroundColor Yellow
    Write-Host "  -> Falling back to the 2022 Build Tools host with legacy toolset." -ForegroundColor DarkGray
    
    $isLegacy = $true
    if ($requestedYear -eq "2019") {
        $componentIdToCheck = "Microsoft.VisualStudio.Component.VC.v142.x86.x64"
    } elseif ($requestedYear -eq "2017") {
        $componentIdToCheck = "Microsoft.VisualStudio.Component.VC.v141.x86.x64"
    } elseif ($requestedYear -eq "2015") {
        $componentIdToCheck = "Microsoft.VisualStudio.Component.VC.v140.x86.x64"
    }
    $msvcId = "Microsoft.VisualStudio.2022.BuildTools"
}

$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$isInstalled = $false

if (Test-Path $vsWhere) {
    $existingPath = & $vsWhere -latest -products * -requires $componentIdToCheck -property installationPath
    if ($existingPath) { 
        $isInstalled = $true
    }
}

if ($isInstalled) {
    Write-Host "  -> MSVC toolset ($componentIdToCheck) is already installed. Skipping." -ForegroundColor Yellow
} else {
    Write-Host "  -> Installing/Modifying $msvcId to include required toolset..." -ForegroundColor Magenta
    
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

function Invoke-MsvcEnvironment {
    Write-Host "  -> Integrating MSVC Developer paths into the current shell..." -ForegroundColor Magenta
    $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    
    if (Test-Path $vsWhere) {
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
# Step 5: Execute setup.m & Generate Init Script
# ---------------------------------------------------------
Write-Host "`n[5/5] Verifying Eiko via setup.m..." -ForegroundColor Cyan

$eikoPath  = Join-Path $PSScriptRoot "eiko"
$setupPath = Join-Path $eikoPath "+eiko_lib\setup.m"

if (-not (Test-Path -LiteralPath $setupPath)) {
    Write-Host "`n[!] Could not find 'setup.m' in the script directory ($setupPath)." -ForegroundColor Red
    Exit-Script
}

Write-Host "  -> Starting MATLAB to run system verification..." -ForegroundColor Magenta

$matlabSafePath = $eikoPath -replace '\\', '/'
$matlabCmd = "addpath('$matlabSafePath'); eiko_lib.setup; exit;"
$matlabProc = Start-Process matlab -ArgumentList '-batch', "`"$matlabCmd`"" -Wait -PassThru -NoNewWindow

if ($matlabProc.ExitCode -eq 0) {
    Write-Host "`n[*] Verification Complete: Eiko for MATLAB is operational!" -ForegroundColor Green
} else {
    Write-Host "`n[!] Verification failed. MATLAB encountered an error while running setup.m." -ForegroundColor Red
    Exit-Script
}

$launcherPath = Join-Path $PSScriptRoot "start_eiko.m"
# Using a literal Here-String (@' ... '@)
$launcherContent = @'
% Eiko Initialization Script
[scriptDir, ~, ~] = fileparts(mfilename('fullpath'));
eikoPath = fullfile(scriptDir, 'eiko');
addpath(eikoPath);
disp('[*] Eiko environment active. Ready to compute!');
'@

$launcherContent | Out-File -FilePath $launcherPath -Encoding utf8

Write-Host "`n====================================================" -ForegroundColor Green
Write-Host " EIKO ENVIRONMENT INSTALLED SUCCESSFULLY" -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Green

Write-Host "`n[!] How to use Eiko:" -ForegroundColor Yellow
Write-Host "    Inside MATLAB, run: " -NoNewline -ForegroundColor Gray
Write-Host "start_eiko`n" -ForegroundColor Cyan
Write-Host "    (This will add Eiko to your path for the current session)`n" -ForegroundColor DarkGray

# Explicitly hold the window open for the instructions
Write-Host "Installation process complete. Press any key to close this setup window..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

exit
