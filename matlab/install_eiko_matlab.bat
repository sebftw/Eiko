:; # ==============================================================================
:; # Eiko Cross-Platform Unified Installer Polyglot Script
:; # Works on Linux (via "bash install_eiko_matlab.bat") and Windows (via double-click)
:; # ==============================================================================
:; # Linux/Bash Section
:; if [ ! -f "install_eiko_matlab_ubuntu.sh" ]; then
:;    echo "Downloading latest Eiko release..."
:;    URL="https://github.com/sebftw/Eiko/releases/latest/download/eiko_matlab.zip"
:;    if command -v curl >/dev/null 2>&1; then curl -sL "$URL" -o eiko_matlab.zip
:;    elif command -v wget >/dev/null 2>&1; then wget -qO eiko_matlab.zip "$URL"
:;    elif command -v python3 >/dev/null 2>&1; then python3 -c "import urllib.request; urllib.request.urlretrieve('$URL', 'eiko_matlab.zip')"
:;    else echo "Error: Neither curl, wget, nor python3 were found." && exit 1; fi
:;    
:;    echo "Extracting files..."
:;    rm -rf Eiko 2>/dev/null
:;    
:;    if command -v unzip >/dev/null 2>&1; then unzip -q eiko_matlab.zip -d Eiko
:;    elif command -v python3 >/dev/null 2>&1; then python3 -m zipfile -e eiko_matlab.zip Eiko
:;    else echo "Error: unzip or python3 is required to extract the release." && exit 1; fi
:;    
:;    rm -f eiko_matlab.zip
:;    cd Eiko || { echo "Failed to find directory"; exit 1; }
:; else
:;    echo "Local install_eiko_matlab_ubuntu.sh found. Skipping download."
:; fi
:; chmod +x install_eiko_matlab_ubuntu.sh
:; bash ./install_eiko_matlab_ubuntu.sh
:; read -n 1 -s -r -p "Installation finished. Press any key to exit..."
:; exit $?
@echo off
REM Windows Command Prompt / Batch Section
powershell -NoExit -ExecutionPolicy Bypass -Command "$ProgressPreference = 'SilentlyContinue'; if (-not (Test-Path '.\install_eiko_matlab_windows.ps1')) { Write-Host 'Downloading latest Eiko release...'; Invoke-WebRequest -Uri 'https://github.com/sebftw/Eiko/releases/latest/download/eiko_matlab.zip' -OutFile 'eiko_matlab.zip'; Write-Host 'Extracting files...'; if (Test-Path 'Eiko') { Remove-Item 'Eiko' -Recurse -Force }; Expand-Archive -Path 'eiko_matlab.zip' -DestinationPath '.\Eiko' -Force; Remove-Item -Path 'eiko_matlab.zip' -Force; Set-Location -Path 'Eiko'; } else { Write-Host 'Local installer found. Skipping download.'; } & '.\install_eiko_matlab_windows.ps1'"
if %errorlevel% neq 0 (
    echo [!] Windows Installation failed.
    pause
    exit /b %errorlevel%
)
