:; # ==============================================================================
:; # Eiko Cross-Platform Unified Installer Polyglot Script
:; # Works on Linux Bash (via source <(curl...)) and Windows (via double-click or cmd)
:; # ==============================================================================
:; # Linux/Bash Section
:; if [ ! -f "install_eiko_ubuntu.sh" ]; then #
:;   echo "Downloading Eiko repository..." #
:;   curl -sL https://github.com/sebftw/Eiko/archive/refs/heads/main.zip -o Eiko-main.zip || wget -qO Eiko-main.zip https://github.com/sebftw/Eiko/archive/refs/heads/main.zip #
:;   echo "Extracting repository..." #
:;   unzip -qo Eiko-main.zip #
:;   rm Eiko-main.zip #
:;   rm -rf Eiko #
:;   mv Eiko-main Eiko #
:;   cd Eiko/matlab || { echo "Failed to find directory"; exit 1; } #
:; else #
:;   echo "Local install_eiko_ubuntu.sh found. Skipping download." #
:; fi #
:; chmod +x install_eiko_ubuntu.sh #
:; bash ./install_eiko_ubuntu.sh #
:; read -n 1 -s -r -p "Installation finished. Press any key to exit..."
:; exit $? #

@echo off
REM Windows Command Prompt / Batch Section
powershell -NoExit -ExecutionPolicy Bypass -Command "if (Test-Path '.\install_eiko_windows.ps1') { Write-Host 'Local installer found. Skipping download.' } else { $ProgressPreference = 'SilentlyContinue'; Write-Host 'Downloading Eiko repository...'; Invoke-WebRequest -Uri 'https://github.com/sebftw/Eiko/archive/refs/heads/main.zip' -OutFile 'Eiko-main.zip'; Write-Host 'Extracting repository...'; Expand-Archive -Path 'Eiko-main.zip' -DestinationPath '.' -Force; Remove-Item -Path 'Eiko-main.zip' -Force; if (Test-Path 'Eiko') { Remove-Item 'Eiko' -Recurse -Force }; Rename-Item -Path 'Eiko-main' -NewName 'Eiko'; Set-Location -Path 'Eiko\matlab' } .\install_eiko_windows.ps1"
if %errorlevel% neq 0 (
    echo [!] Windows Installation failed.
    pause
    exit /b %errorlevel%
)
