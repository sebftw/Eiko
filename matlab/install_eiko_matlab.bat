:; # ==============================================================================
:; # Eiko Cross-Platform Unified Installer Polyglot Script
:; # Works on Linux Bash (via wget -qO- ... | bash) and Windows (via double-click)
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
:; # Linux/Bash Section
:; if [ ! -f "install_eiko_ubuntu.sh" ]; then #
:;   echo "Downloading Eiko repository..." #
:;   URL="https://github.com/sebftw/Eiko/archive/refs/heads/main.tar.gz" #
:;   if command -v curl >/dev/null 2>&1; then #
:;     curl -sL "$URL" -o Eiko-main.tar.gz #
:;   elif command -v wget >/dev/null 2>&1; then #
:;     wget -qO Eiko-main.tar.gz "$URL" #
:;   elif command -v python3 >/dev/null 2>&1; then #
:;     python3 -c "import urllib.request; urllib.request.urlretrieve('$URL', 'Eiko-main.tar.gz')" #
:;   else #
:;     echo "Error: Neither curl, wget, nor python3 were found. Please install one." #
:;     exit 1 #
:;   fi #
:;   echo "Extracting repository..." #
:;   tar -xzf Eiko-main.tar.gz #
:;   rm Eiko-main.tar.gz #
:;   rm -rf Eiko #
:;   mv Eiko-main Eiko #
:;   cd Eiko/matlab || { echo "Failed to find directory"; exit 1; } #
:; else #
:;   echo "Local install_eiko_ubuntu.sh found. Skipping download." #
:; fi #
:; chmod +x install_eiko_ubuntu.sh #
:; bash ./install_eiko_ubuntu.sh #
:; read -n 1 -s -r -p "Installation finished. Press any key to exit..." #
:; exit $? #
@echo off
REM Windows Command Prompt / Batch Section
powershell -NoExit -ExecutionPolicy Bypass -Command "if (Test-Path '.\install_eiko_windows.ps1') { Write-Host 'Local installer found. Skipping download.' } else { $ProgressPreference = 'SilentlyContinue'; Write-Host 'Downloading Eiko repository...'; Invoke-WebRequest -Uri 'https://github.com/sebftw/Eiko/archive/refs/heads/main.zip' -OutFile 'Eiko-main.zip'; Write-Host 'Extracting repository...'; Expand-Archive -Path 'Eiko-main.zip' -DestinationPath '.' -Force; Remove-Item -Path 'Eiko-main.zip' -Force; if (Test-Path 'Eiko') { Remove-Item 'Eiko' -Recurse -Force }; Rename-Item -Path 'Eiko-main' -NewName 'Eiko'; Set-Location -Path 'Eiko\matlab' } .\install_eiko_windows.ps1"
if %errorlevel% neq 0 (
    echo [!] Windows Installation failed.
    pause
    exit /b %errorlevel%
)
