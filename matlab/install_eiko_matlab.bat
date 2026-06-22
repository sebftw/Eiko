:; # ==============================================================================
:; # Eiko Cross-Platform Unified Installer Polyglot Script
:; # Works on Linux Bash (via wget -qO- ... | bash) and Windows (via double-click)
:; # ==============================================================================
:; # Linux/Bash Section
:; if [ ! -f "install_eiko_matlab_ubuntu.sh" ]; then
:;   echo "Downloading Eiko repository..."
:;   URL="https://github.com/sebftw/Eiko/archive/refs/heads/main.tar.gz"
:;   if command -v curl >/dev/null 2>&1; then curl -sL "$URL" -o Eiko-src.tar.gz
:;   elif command -v wget >/dev/null 2>&1; then wget -qO Eiko-src.tar.gz "$URL"
:;   elif command -v python3 >/dev/null 2>&1; then python3 -c "import urllib.request; urllib.request.urlretrieve('$URL', 'Eiko-src.tar.gz')"
:;   else echo "Error: Neither curl, wget, nor python3 were found." && exit 1; fi
:;   echo "Extracting repository..."
:;   tar -xzf Eiko-src.tar.gz
:;   rm Eiko-src.tar.gz
:;   rm -rf Eiko 2>/dev/null
:;   # Dynamically find the extracted directory (handles 'Eiko-main', 'Eiko-0.8.5', etc.)
:;   EXTRACTED_DIR=$(find . -maxdepth 1 -type d -name "Eiko-*" | head -n 1)
:;   mv "$EXTRACTED_DIR" Eiko
:;   cd Eiko/matlab || { echo "Failed to find directory"; exit 1; }
:; else
:;   echo "Local install_eiko_matlab_ubuntu.sh found. Skipping download."
:; fi
:; chmod +x install_eiko_matlab_ubuntu.sh
:; bash ./install_eiko_matlab_ubuntu.sh
:; read -n 1 -s -r -p "Installation finished. Press any key to exit..."
:; exit $?
@echo off
REM Windows Command Prompt / Batch Section
powershell -NoExit -ExecutionPolicy Bypass -Command "if (Test-Path '.\install_eiko_matlab_windows.ps1') { Write-Host 'Local installer found. Skipping download.' } else { $ProgressPreference = 'SilentlyContinue'; Write-Host 'Downloading Eiko repository...'; Invoke-WebRequest -Uri 'https://github.com/sebftw/Eiko/archive/refs/heads/main.zip' -OutFile 'Eiko-src.zip'; Write-Host 'Extracting repository...'; Expand-Archive -Path 'Eiko-src.zip' -DestinationPath '.' -Force; Remove-Item -Path 'Eiko-src.zip' -Force; if (Test-Path 'Eiko') { Remove-Item 'Eiko' -Recurse -Force }; $extracted = Get-ChildItem -Directory | Where-Object Name -like 'Eiko-*' | Select-Object -First 1; Rename-Item -Path $extracted.FullName -NewName 'Eiko'; Set-Location -Path 'Eiko\matlab' } .\install_eiko_matlab_windows.ps1"
if %errorlevel% neq 0 (
    echo [!] Windows Installation failed.
    pause
    exit /b %errorlevel%
)
