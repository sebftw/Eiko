#!/bin/sh
:; # ==============================================================================
:; # Eiko Cross-Platform Unified Installer
:; # Works on Linux (via "bash install_eiko_matlab.bat") and Windows (via double-click)
:; # Logic: If local installer exists, use it. Otherwise, download and run.
:; # ==============================================================================
:; # DISCLAIMER:
:; # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
:; # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
:; # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
:; # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
:; # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
:; # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
:; # SOFTWARE.
:; # ==============================================================================
:; if [ ! -f "install_eiko_python_ubuntu.sh" ]; then
:;   echo "Installer not found locally. Downloading..."
:;   URL="https://raw.githubusercontent.com/sebftw/Eiko/main/python/install_eiko_python_ubuntu.sh"
:;   if command -v curl >/dev/null 2>&1; then curl -sL "$URL" -o install_eiko_python_ubuntu.sh
:;   elif command -v wget >/dev/null 2>&1; then wget -qO install_eiko_python_ubuntu.sh "$URL"
:;   else echo "Error: Neither curl nor wget found." && exit 1; fi
:; fi
:; chmod +x install_eiko_python_ubuntu.sh
:; bash ./install_eiko_python_ubuntu.sh
:; read -n 1 -s -r -p "Installation finished. Press any key to exit..."
:; exit $?

@echo off
REM Windows Command Prompt Section
set "SCRIPT_NAME=install_eiko_python_windows.ps1"
if not exist "%SCRIPT_NAME%" (
    echo Installer not found locally. Downloading...
    powershell -ExecutionPolicy Bypass -Command "Invoke-WebRequest -UseBasicParsing -Uri 'https://raw.githubusercontent.com/sebftw/Eiko/main/python/install_eiko_python_windows.ps1' -OutFile '%SCRIPT_NAME%'"
)
powershell -ExecutionPolicy Bypass -Command ". .\%SCRIPT_NAME%"
if %errorlevel% neq 0 (
    echo [!] Windows Installation failed.
    pause
    exit /b %errorlevel%
)
