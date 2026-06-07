:<<- 'BATCH'
@ECHO OFF
GOTO WINDOWS_START
BATCH
#!/bin/bash
# ==============================================================================
# LINUX RUNNER (Bash)
# ==============================================================================
REMOTE_INSTALLER_URL="https://raw.githubusercontent.com/sebftw/Eiko/main/python/install_eiko_ubuntu.sh"

# 1. Check if eiko is already available globally
if command -v python3 >/dev/null 2>&1 && python3 -c "import eiko.eiko_torch" >/dev/null 2>&1; then
    echo -e "\033[1;32m-> Eiko is available globally. No virtual environment needed.\033[0m"
    exit 0
fi

# 2. Fallback to dedicated Eiko sandbox
VENV_PATH="$HOME/eiko"

# Check if venv exists and eiko module is installed inside it
if [ ! -f "$VENV_PATH/bin/activate" ] || ! "$VENV_PATH/bin/python" -c "import eiko.eiko_torch" >/dev/null 2>&1; then
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

# 3. Activate the environment
echo -e "\033[1;32m-> Activating Eiko virtual environment...\033[0m"
source "$VENV_PATH/bin/activate"
exec bash


:WINDOWS_START
REM ==============================================================================
REM WINDOWS RUNNER (Batch)
REM ==============================================================================
SET "REMOTE_INSTALLER_URL=https://raw.githubusercontent.com/sebftw/Eiko/main/python/install_eiko_windows.ps1"

REM 1. Check if python exists at all
WHERE python >nul 2>&1
IF %ERRORLEVEL% NEQ 0 GOTO CHECK_VENV

REM 2. It exists, now test if eiko can be imported
python -c "import eiko.eiko_torch" >nul 2>&1
IF %ERRORLEVEL% EQU 0 GOTO RUN_GLOBAL_PYTHON

:CHECK_VENV
REM 3. Fallback to dedicated Eiko sandbox...

:: If the virtual environment activation script doesn't exist, install.
IF NOT EXIST "%VENV_PATH%\Scripts\activate.bat" GOTO RUN_INSTALLER

:: If the environment exists, test if eiko can actually be imported inside it
"%VENV_PATH%\Scripts\python.exe" -c "import eiko.eiko_torch" >nul 2>&1
IF %ERRORLEVEL% NEQ 0 GOTO RUN_INSTALLER

GOTO ACTIVATE_VENV

:RUN_INSTALLER
ECHO -^> Eiko not found. Downloading and launching installer...

PowerShell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $script = Invoke-WebRequest -Uri '%REMOTE_INSTALLER_URL%' -UseBasicParsing; Invoke-Expression $script.Content"

IF %ERRORLEVEL% NEQ 0 (
    ECHO -^> Installation failed. Exiting.
    EXIT /B %ERRORLEVEL%
)

:: Re-verify after install
IF NOT EXIST "%VENV_PATH%\Scripts\activate.bat" (
    ECHO -^> Installation script completed, but activate.bat was not found.
    EXIT /B 1
)

:ACTIVATE_VENV
ECHO -^> Activating Eiko virtual environment...
CALL "%VENV_PATH%\Scripts\activate.bat"
EXIT /B 0