:<<"::WINDOWS_WRAPPER"
@ECHO OFF
GOTO :WINDOWS_START
::WINDOWS_WRAPPER

# ==============================================================================
# LINUX RUNNER (Bash)
# ==============================================================================
VENV_PATH="$HOME/eiko"

# Check if venv exists and eiko module is installed
if [ ! -f "$VENV_PATH/bin/activate" ] || ! "$VENV_PATH/bin/python" -c "import eiko" >/dev/null 2>&1; then
    echo -e "\033[1;33m-> Eiko environment missing or incomplete. Launching installer...\033[0m"
    bash ./install.sh
    if [ $? -ne 0 ]; then
        echo -e "\033[0;31m-> Installation failed. Exiting.\033[0m"
        exit 1
    fi
fi

# Execute Eiko directly using the venv Python (replaces current shell process)
exec "$VENV_PATH/bin/python" -m eiko "$@"

:WINDOWS_START
REM ==============================================================================
REM WINDOWS RUNNER (Batch)
REM ==============================================================================
SET VENV_PATH=%USERPROFILE%\eiko

REM Check if Python exists in venv
IF NOT EXIST "%VENV_PATH%\Scripts\python.exe" GOTO RUN_INSTALLER

REM Check if eiko module is installed
"%VENV_PATH%\Scripts\python.exe" -c "import eiko" >nul 2>&1
IF %ERRORLEVEL% NEQ 0 GOTO RUN_INSTALLER

GOTO RUN_EIKO

:RUN_INSTALLER
ECHO -^> Eiko environment missing or incomplete. Launching installer...
PowerShell -NoProfile -ExecutionPolicy Bypass -File ".\install.ps1"
IF %ERRORLEVEL% NEQ 0 (
    ECHO -^> Installation failed. Exiting.
    EXIT /B %ERRORLEVEL%
)

:RUN_EIKO
"%VENV_PATH%\Scripts\python.exe" -m eiko %*
EXIT /B %ERRORLEVEL%
