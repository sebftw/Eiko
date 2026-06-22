:; # ==============================================================================
:; # Eiko Cross-Platform Unified Installer Polyglot Script
:; # Works on Linux Bash (via wget -qO- ... | bash) and Windows (via double-click)
:; # ==============================================================================
:; # Linux/Bash Section
:; if [ ! -f "install_eiko_matlab_ubuntu.sh" ]; then
:;    echo "Downloading Eiko repository..."
:;    URL="https://github.com/sebftw/Eiko/archive/refs/heads/main.tar.gz"
:;    if command -v curl >/dev/null 2>&1; then curl -sL "$URL" -o Eiko-src.tar.gz
:;    elif command -v wget >/dev/null 2>&1; then wget -qO Eiko-src.tar.gz "$URL"
:;    elif command -v python3 >/dev/null 2>&1; then python3 -c "import urllib.request; urllib.request.urlretrieve('$URL', 'Eiko-src.tar.gz')"
:;    else echo "Error: Neither curl, wget, nor python3 were found." && exit 1; fi
:;    
:;    echo "Extracting MATLAB components..."
:;    rm -rf .eiko_tmp Eiko 2>/dev/null
:;    mkdir .eiko_tmp
:;    tar -xzf Eiko-src.tar.gz -C .eiko_tmp
:;    
:;    # Dynamically find the extracted directory and move ONLY the matlab folder
:;    EXTRACTED_DIR=$(find .eiko_tmp -maxdepth 1 -type d -name "Eiko-*" | head -n 1)
:;    mv "$EXTRACTED_DIR/matlab" ./Eiko
:;    
:;    # Clean up the archive and the remaining repo contents
:;    rm -rf .eiko_tmp Eiko-src.tar.gz
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
powershell -NoExit -ExecutionPolicy Bypass -Command "if (Test-Path '.\install_eiko_matlab_windows.ps1') { Write-Host 'Local installer found. Skipping download.' } else { $ProgressPreference = 'SilentlyContinue'; Write-Host 'Downloading Eiko repository...'; Invoke-WebRequest -Uri 'https://github.com/sebftw/Eiko/archive/refs/heads/main.zip' -OutFile 'Eiko-src.zip'; Write-Host 'Extracting MATLAB components...'; if (Test-Path 'Eiko') { Remove-Item 'Eiko' -Recurse -Force }; New-Item -ItemType Directory -Force -Path '.\EikoTmp' | Out-Null; Expand-Archive -Path 'Eiko-src.zip' -DestinationPath '.\EikoTmp' -Force; Remove-Item -Path 'Eiko-src.zip' -Force; $extracted = Get-ChildItem -Path '.\EikoTmp' -Directory | Where-Object Name -like 'Eiko-*' | Select-Object -First 1; $target = Join-Path $extracted.FullName 'matlab'; Move-Item -Path $target -Destination '.\Eiko' -Force; Remove-Item -Path '.\EikoTmp' -Recurse -Force; Set-Location -Path 'Eiko' } .\install_eiko_matlab_windows.ps1"
if %errorlevel% neq 0 (
    echo [!] Windows Installation failed.
    pause
    exit /b %errorlevel%
)
