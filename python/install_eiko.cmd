#!/bin/bash
:<<'BATCH'
@GOTO WINDOWS_START
BATCH

# ==============================================================================
# LINUX RUNNER (Bash)
# ==============================================================================
REMOTE_INSTALLER_URL="https://raw.githubusercontent.com/sebftw/Eiko/main/python/install_eiko_ubuntu.sh"

# 1. Check if eiko is already available in the current environment
if command -v python3 >/dev/null 2>&1 && python3 -c "import eiko" >/dev/null 2>&1; then
    exec python3 "$@"
fi

# 2. Fallback to dedicated Eiko sandbox
VENV_PATH="$HOME/eiko"

# Check if venv exists and eiko module is installed inside it
if [ ! -f "$VENV_PATH/bin/activate" ] || ! "$VENV_PATH/bin/python" -c "import eiko" >/dev/null 2>&1; then
    echo -e "\033[1;33m-> Eiko not found. Downloading and launching installer...\033[0m"
    
    TEMP_SH=$(mktemp)
    if curl -sSf "$REMOTE_INSTALLER_URL" -o "$TEMP_SH"; then
        bash "$TEMP_SH"
        SH_EXIT=$?
        rm -f "$TEMP_SH"
        if [ $SH_EXIT -ne 0 ]; then
            echo -e "\033[0;31m-> Installation failed. Exiting.\033[0m"
            exit 1
        fi
    else
        echo -e "\033[0;31m-> Failed to download the Linux installer script. Check connection.\033[0m"
        rm -f "$TEMP_SH"
        exit 1
    fi
fi

# Execute the user's script if arguments are provided, 
# otherwise launch the interactive Python shell safely.
if [ $# -gt 0 ]; then
    exec "$VENV_PATH/bin/python" "$@"
else
    exec "$VENV_PATH/bin/python"
fi

# Stop Bash from reading into the Windows section
exit 0

:WINDOWS_START
@ECHO OFF
REM ==============================================================================
REM WINDOWS RUNNER (Batch)
REM ==============================================================================
SET "REMOTE_INSTALLER_URL=https://raw.githubusercontent.com/sebftw/Eiko/main/python/install_eiko_windows.ps1"

REM 1. Check if eiko is already available in the current environment
WHERE python >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    python -c "import eiko" >nul 2>&1
    IF %ERRORLEVEL% EQU 0 (
        python %*
        EXIT /B %ERRORLEVEL%
    )
)

REM 2. Fallback to dedicated Eiko sandbox
SET "VENV_PATH=%USERPROFILE%\eiko"

IF NOT EXIST "%VENV_PATH%\Scripts\python.exe" GOTO RUN_INSTALLER

"%VENV_PATH%\Scripts\python.exe" -c "import eiko" >nul 2>&1
IF %ERRORLEVEL% NEQ 0 GOTO RUN_INSTALLER

GOTO RUN_EIKO

:RUN_INSTALLER
ECHO -^> Eiko not found. Downloading and launching installer...

PowerShell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $script = Invoke-WebRequest -Uri '%REMOTE_INSTALLER_URL%' -UseBasicParsing; Invoke-Expression $script.Content"

IF %ERRORLEVEL% NEQ 0 (
    ECHO -^> Installation failed. Exiting.
    EXIT /B %ERRORLEVEL%
)

:: Re-verify after install to ensure it actually exists now
IF NOT EXIST "%VENV_PATH%\Scripts\python.exe" (
    ECHO -^> Installation script completed, but Python executable was not found.
    EXIT /B 1
)

:RUN_EIKO
"%VENV_PATH%\Scripts\python.exe" %*
EXIT /B %ERRORLEVEL%