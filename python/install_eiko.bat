:; # ==============================================================================
:; # Eiko Cross-Platform Unified Installer Polyglot Script
:; # Works on Linux Bash (via source <(curl...)) and Windows (via double-click or cmd)
:; # ==============================================================================
:; # Linux/Bash Section
:; source <(wget -qO- https://raw.githubusercontent.com/sebftw/Eiko/main/python/install_eiko_ubuntu.sh) #
:; exit $? #

@echo off
REM Windows Command Prompt / Batch Section
powershell -NoExit -ExecutionPolicy Bypass -Command "$script = (Invoke-WebRequest -UseBasicParsing -Uri 'https://raw.githubusercontent.com/sebftw/Eiko/main/python/install_eiko_windows.ps1').Content; . ([ScriptBlock]::Create($script))"
if %errorlevel% neq 0 (
    echo [!] Windows Installation failed.
    pause
    exit /b %errorlevel%
)