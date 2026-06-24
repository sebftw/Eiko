#!/bin/sh 2>nul
:; # ==============================================================================
:; # Eiko Cross-Platform Unified Installer Polyglot Script
:; # Works on Linux (via "bash install_eiko_matlab.bat") and Windows (via double-click)
:; # Logic: 
:; # 1. If local installer is in current dir, run it.
:; # 2. If Eiko dir exists, cd into it and run the installer.
:; # 3. Otherwise, download Eiko, cd into it, and run the installer.
:; # ==============================================================================
:; # Linux/Bash Section
:; if [ -f "install_eiko_matlab_ubuntu.sh" ]; then
:;    echo "Local installer found in current directory."
:; elif [ -d "Eiko" ]; then
:;    echo "Eiko directory found. Entering..."
:;    cd Eiko || { echo "Failed to enter directory"; exit 1; }
:;    if [ ! -f "install_eiko_matlab_ubuntu.sh" ]; then
:;        echo "Error: Eiko directory exists, but it is missing install_eiko_matlab_ubuntu.sh."
:;        echo "Please move the Eiko directory (or this installer) elsewhere, and try again."
:;        exit 1
:;    fi
:; else
:;    echo "Downloading latest Eiko release..."
:;    URL="https://github.com/sebftw/Eiko/releases/latest/download/eiko_matlab.zip"
:;    if command -v curl >/dev/null 2>&1; then curl -sL "$URL" -o eiko_matlab.zip
:;    elif command -v wget >/dev/null 2>&1; then wget -qO eiko_matlab.zip "$URL"
:;    elif command -v python3 >/dev/null 2>&1; then python3 -c "import urllib.request; urllib.request.urlretrieve('$URL', 'eiko_matlab.zip')"
:;    else echo "Error: Neither curl, wget, nor python3 were found." && exit 1; fi
:;    
:;    echo "Extracting files..."
:;    if command -v unzip >/dev/null 2>&1; then unzip -q eiko_matlab.zip -d Eiko
:;    elif command -v python3 >/dev/null 2>&1; then python3 -m zipfile -e eiko_matlab.zip Eiko
:;    else echo "Error: unzip or python3 is required to extract the release." && rm -f eiko_matlab.zip && exit 1; fi
:;    
:;    rm -f eiko_matlab.zip
:;    cd Eiko || { echo "Failed to find directory"; exit 1; }
:; fi
:; 
:; chmod +x install_eiko_matlab_ubuntu.sh
:; bash ./install_eiko_matlab_ubuntu.sh
:; read -n 1 -s -r -p "Installation finished. Press any key to exit..."
:; exit $?
@echo off
REM Windows Command Prompt / Batch Section
powershell -NoExit -ExecutionPolicy Bypass -Command "$ProgressPreference = 'SilentlyContinue'; if (Test-Path '.\install_eiko_matlab_windows.ps1') { Write-Host 'Local installer found in current directory.' } elseif (Test-Path 'Eiko') { Write-Host 'Eiko directory found. Entering...'; Set-Location -Path 'Eiko'; if (-not (Test-Path '.\install_eiko_matlab_windows.ps1')) { Write-Host '[!] Error: Eiko directory exists but it is missing the installer script. Please move the Eiko directory (or this installer), and try again.' -ForegroundColor Red; exit 1 } } else { Write-Host 'Downloading latest Eiko release...'; Invoke-WebRequest -Uri 'https://github.com/sebftw/Eiko/releases/latest/download/eiko_matlab.zip' -OutFile 'eiko_matlab.zip'; Write-Host 'Extracting files...'; Expand-Archive -Path 'eiko_matlab.zip' -DestinationPath '.\Eiko' -Force; Remove-Item -Path 'eiko_matlab.zip' -Force; Set-Location -Path 'Eiko'; } if (Test-Path '.\install_eiko_matlab_windows.ps1') { & '.\install_eiko_matlab_windows.ps1' }"
if %errorlevel% neq 0 (
    echo [!] Windows Installation failed or was aborted.
    pause
    exit /b %errorlevel%
)
