:<<"::WINDOWS_WRAPPER"
@ECHO OFF
GOTO :WINDOWS_START
::WINDOWS_WRAPPER

# ==============================================================================
# LINUX RUNNER (Bash)
# ==============================================================================

# 1. Check if eiko is already available in the current environment
if command -v python3 >/dev/null 2>&1 && python3 -c "import eiko" >/dev/null 2>&1; then
    exec python3 "$@"
fi

# 2. Fallback to dedicated Eiko sandbox
VENV_PATH="$HOME/eiko"

# Check if venv exists and eiko module is installed inside it
if [ ! -f "$VENV_PATH/bin/activate" ] || ! "$VENV_PATH/bin/python" -c "import eiko" >/dev/null 2>&1; then
    echo -e "\033[1;33m-> Eiko not found in current env or sandbox. Launching installer...\033[0m"
    bash ./install.sh
    if [ $? -ne 0 ]; then
        echo -e "\033[0;31m-> Installation failed. Exiting.\033[0m"
        exit 1
    fi
fi

# Execute the user's script (or open a REPL) using the venv Python
exec "$VENV_PATH/bin/python" "$@"

:WINDOWS_START
REM ==============================================================================
REM WINDOWS RUNNER (Batch)
REM ==============================================================================

REM 1. Check if eiko is already available in the current environment
python -c "import eiko" >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    python %*
    EXIT /B %ERRORLEVEL%
)

REM 2. Fallback to dedicated Eiko sandbox
SET VENV_PATH=%USERPROFILE%\eiko

REM Check if Python exists in venv
IF NOT EXIST "%VENV_PATH%\Scripts\python.exe" GOTO RUN_INSTALLER

REM Check if eiko module is installed inside it
"%VENV_PATH%\Scripts\python.exe" -c "import eiko" >nul 2>&1
IF %ERRORLEVEL% NEQ 0 GOTO RUN_INSTALLER

GOTO RUN_EIKO

:RUN_INSTALLER
ECHO -^> Eiko not found in current env or sandbox. Launching installer...
PowerShell -NoProfile -ExecutionPolicy Bypass -File ".\install.ps1"
IF %ERRORLEVEL% NEQ 0 (
    ECHO -^> Installation failed. Exiting.
    EXIT /B %ERRORLEVEL%
)

:RUN_EIKO
"%VENV_PATH%\Scripts\python.exe" %*
EXIT /B %ERRORLEVEL%
