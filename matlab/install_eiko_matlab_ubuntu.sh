#!/bin/bash
# ==============================================================================
# Eiko Smart Linux Environment Installer (MATLAB Edition)
# ==============================================================================
# DISCLAIMER:
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

# Colors for terminal output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
MAGENTA='\033[1;35m'
RED='\033[0;31m'
NC='\033[0m' # No Color (Standard Terminal Text)

# Helper function to exit cleanly whether sourced or executed directly
safe_exit() {
    local code=${1:-0}
    # Check if the script was sourced or executed directly
    if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
        return "$code"
    else
        echo -e "\n${CYAN}Press Enter to exit...${NC}"
        read -r
        exit "$code"
    fi
}

echo -e "${CYAN}====================================================${NC}"
echo -e "${CYAN} Eiko MATLAB Environment Setup (Linux)              ${NC}"
echo -e "${CYAN}====================================================${NC}"

# ---------------------------------------------------------
# Step 0: Explanation and User Confirmation
# ---------------------------------------------------------
echo -e "\nThis script will configure your system for Eiko by performing the following actions:"
echo -e "  1. Validate your NVIDIA display driver compatibility."
echo -e "  2. Identify MATLAB's native C++ compiler (GCC) requirements."
echo -e "  3. Install the appropriate GNU C++ Build Tools."
echo -e "  4. Deploy or verify the NVIDIA CUDA Toolkit (v12.8)."
echo -e "  5. Execute system verification to confirm successful environment integration."

echo -e "\n${YELLOW}[!] DISCLAIMER: This script requires sudo privileges and modifies system packages."
echo -e "    It is provided 'as-is' without any express or implied warranties. Run at your own risk.${NC}"

read -p $'\nDo you agree to these terms and want to proceed? (y/N): ' confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo -e "\n${YELLOW}[*] Setup cancelled by user.${NC}"
    safe_exit 0
fi

# Request sudo privileges upfront for system checks now that the user has agreed
echo -e "\n${MAGENTA}[INFO] Requesting sudo privileges for system installation...${NC}"
sudo -v || safe_exit 1

# Keep sudo alive while the script runs
while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &

# Ensure system package lists are updated
sudo apt-get update -qq

# ---------------------------------------------------------
# Step 1: Check NVIDIA Driver Version
# ---------------------------------------------------------
echo -e "\n${CYAN}[1/5] Checking NVIDIA Driver Compatibility...${NC}"

MIN_DRIVER=570 
UPDATE_DRIVER=false
IS_WSL=false

# Detect WSL
if grep -qi microsoft /proc/version || [ -n "$WSL_DISTRO_NAME" ]; then
    IS_WSL=true
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | cut -d'.' -f1)
    
    if [ "$DRIVER_VER" -lt "$MIN_DRIVER" ]; then
        # SOFT FAIL CASE: Informative warning instead of fatalistic screaming
        echo -e "${YELLOW}  -> [!] NVIDIA driver v${DRIVER_VER} is below the recommended v${MIN_DRIVER} for CUDA 12.8.${NC}"
        
        if [ "$IS_WSL" = true ]; then
            echo -e "\n${YELLOW}----------------------------------------------------${NC}"
            echo -e "${YELLOW} NOTICE: NVIDIA Driver Update Required              ${NC}"
            echo -e "\n${YELLOW}----------------------------------------------------${NC}"
            echo -e "To support the required CUDA environment, please update your Windows host driver:"
            echo -e "  1. Open 'NVIDIA App' or 'GeForce Experience' on Windows."
            echo -e "  2. Install the latest Game Ready or Studio driver (v${MIN_DRIVER}+)."
            echo -e "  3. Rerun this script once the update is complete."
            safe_exit 1
        else
            # For native Linux, we can try to handle it gracefully down-script
            UPDATE_DRIVER=true
        fi
    else
        # SUCCESS CASE
        echo -e "${GREEN}  -> [OK] NVIDIA driver v${DRIVER_VER} detected (Meets v${MIN_DRIVER}+ requirement for CUDA 12.8).${NC}"
        echo -e "  -> GPU environment is ready for Eiko installation."
    fi
else
    echo -e "${YELLOW}  -> No active NVIDIA display driver detected.${NC}"
    if [ "$IS_WSL" = true ]; then
        echo -e "\n${YELLOW}----------------------------------------------------${NC}"
        echo -e "${YELLOW} NOTICE: NVIDIA Driver Update Required              ${NC}"
        echo -e "\n${YELLOW}----------------------------------------------------${NC}"
        echo -e "To support the required CUDA environment, please update your Windows host driver:"
        echo -e "  1. Open 'NVIDIA App' or 'GeForce Experience' on Windows."
        echo -e "  2. Install the latest Game Ready or Studio driver (v${MIN_DRIVER}+)."
        echo -e "  3. Rerun this script once the update is complete."
        safe_exit 1
    else
        UPDATE_DRIVER=true
    fi
fi

# Only proceed with the auto-update logic if NOT in WSL
if [ "$UPDATE_DRIVER" = true ]; then
    echo -e "${YELLOW}  -> Note: Installing Linux display drivers will cause a brief screen flicker and requires a reboot.${NC}"
    read -p "     Would you like to automatically install the recommended NVIDIA driver now? (y/N): " driver_confirm
    
    if [[ "$driver_confirm" =~ ^[Yy]$ ]]; then
        echo -e "${CYAN}  -> Fetching and installing the recommended proprietary driver...${NC}"
        sudo ubuntu-drivers autoinstall
        
        echo -e "\n${GREEN}----------------------------------------------------${NC}"
        echo -e "${GREEN} [OK] Driver Installed — Restart Required            ${NC}"
        echo -e "${GREEN}----------------------------------------------------${NC}"
        echo -e "The NVIDIA driver was updated successfully. The Linux kernel must reload"
        echo -e "before the CUDA toolkit can communicate with the GPU."
        echo -e "\nPlease reboot your computer, then rerun this script to finish the setup."
        safe_exit 0
    else
        echo -e "\n${YELLOW}  -> [!] Setup paused: A compatible NVIDIA driver is required to continue.${NC}"
        echo -e "     Please update your drivers manually, reboot, and rerun this script."
        safe_exit 1
    fi
fi

# ---------------------------------------------------------
# Step 2: Locating MATLAB & Probing GCC Requirements
# ---------------------------------------------------------
echo -e "\n${CYAN}[2/5] Locating MATLAB and Probing Compiler Requirements...${NC}"

if ! command -v matlab >/dev/null 2>&1; then
    echo -e "${RED}[!] 'matlab' command not found in your system PATH.${NC}"
    echo -e "${YELLOW}Please ensure MATLAB is installed and correctly added to your environment variables.${NC}"
    safe_exit 1
fi

echo -e "  -> MATLAB found. Starting MATLAB in headless mode to query supported C++ compilers..."

# Headless probe script: iterates over supported C++ compilers, looks for GNU, and extracts the major version number.
PROBE_SCRIPT="cc=mex.getCompilerConfigurations('C++','Supported'); for i=1:length(cc), if ~isempty(strfind(lower(cc(i).Manufacturer), 'gnu')), m=regexp(cc(i).Version, '^(\d+)', 'tokens', 'once'); if ~isempty(m), fprintf('GCC_MAJOR:%s\n', m{1}); exit; end; end; end; fprintf('GCC_MAJOR:UNKNOWN\n'); exit;"
MATLAB_OUT=$(matlab -batch "$PROBE_SCRIPT" 2>&1)

if [[ "$MATLAB_OUT" =~ GCC_MAJOR:([0-9]+) ]]; then
    TARGET_GCC_VER="${BASH_REMATCH[1]}"
    echo -e "${GREEN}  -> MATLAB requested GCC ${TARGET_GCC_VER}.${NC}"
else
    TARGET_GCC_VER="10" # Safe default for older LTS systems
    echo -e "${YELLOW}  -> MATLAB compiler query failed or returned unrecognized output. Defaulting to GCC 10.${NC}"
fi

# ---------------------------------------------------------
# Step 3: Check and Install C++ Build Tools
# ---------------------------------------------------------
echo -e "\n${CYAN}[3/5] Checking Required GCC Build Tools...${NC}"

if dpkg -s build-essential >/dev/null 2>&1; then
    echo -e "${YELLOW}  -> build-essential is already installed. Skipping.${NC}"
else
    echo -e "${MAGENTA}  -> Installing standard build utilities (build-essential)...${NC}"
    sudo apt-get install -y build-essential
fi

# Install the dynamically targeted GCC version
if dpkg -s "g++-${TARGET_GCC_VER}" >/dev/null 2>&1; then
    echo -e "${YELLOW}  -> GCC ${TARGET_GCC_VER} toolchain is already available. Skipping.${NC}"
else
    echo -e "${MAGENTA}  -> Installing GCC ${TARGET_GCC_VER} toolchain to match MATLAB requirements...${NC}"
    sudo apt-get install -y "gcc-${TARGET_GCC_VER}" "g++-${TARGET_GCC_VER}" || {
        echo -e "${YELLOW}[!] Warning: gcc-${TARGET_GCC_VER} could not be installed (may require a PPA update on this OS).${NC}"
    }
fi

# Always ensure GCC 10 is available as an explicit fallback/C++20 requirement
if [ "$TARGET_GCC_VER" != "10" ]; then
    if dpkg -s g++-10 >/dev/null 2>&1; then
        echo -e "${YELLOW}  -> GCC 10 toolchain is already available. Skipping.${NC}"
    else
        echo -e "${MAGENTA}  -> Installing GCC 10 toolchain (secondary requirement)...${NC}"
        sudo apt-get install -y gcc-10 g++-10
    fi
fi

# ---------------------------------------------------------
# Step 4: Check and Install Pinned CUDA Toolkit (12.8)
# ---------------------------------------------------------
TARGET_CUDA_PKG="cuda-toolkit-12-8"
TARGET_CUDA_VER="12.8"
source /etc/os-release

echo -e "\n${CYAN}[4/5] Checking CUDA installation (Target: ${TARGET_CUDA_VER})...${NC}"

INSTALL_CUDA=true
NVCC_CMD=""

if command -v nvcc >/dev/null 2>&1; then
    NVCC_CMD="nvcc"
else
    FALLBACK_PATH=$(ls -1d /usr/local/cuda*/bin/nvcc 2>/dev/null | sort -V -r | head -n 1 || true)
    if [ -n "$FALLBACK_PATH" ] && [ -x "$FALLBACK_PATH" ]; then
        NVCC_CMD="$FALLBACK_PATH"
    fi
fi

if [ -n "$NVCC_CMD" ]; then
    CUDA_VER=$("$NVCC_CMD" --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    if [ "$(printf '%s\n' "$TARGET_CUDA_VER" "$CUDA_VER" | sort -V | head -n1)" = "$TARGET_CUDA_VER" ]; then
        echo -e "${YELLOW}  -> Found CUDA $CUDA_VER. Skipping installation.${NC}"
        INSTALL_CUDA=false
    fi
fi

if [ "$INSTALL_CUDA" = true ]; then
    echo -e "${MAGENTA}  -> Installing target CUDA version (${TARGET_CUDA_PKG})...${NC}"
    ARCH=$(uname -m)
    if [ "$ARCH" = "aarch64" ]; then ARCH="sbsa"; fi
    
    # Simple explicit error check for wget payload
    if wget -qO cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/${ARCH}/cuda-keyring_1.1-1_all.deb"; then
        sudo dpkg -i cuda-keyring.deb
        rm cuda-keyring.deb
        sudo apt-get update -qq
        sudo apt-get install -y $TARGET_CUDA_PKG
    else
        echo -e "${RED}[!] Failed to download CUDA GPG keys.${NC}"
        safe_exit 1
    fi
fi

export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda

# ---------------------------------------------------------
# Step 5: Execute setup.m
# ---------------------------------------------------------
echo -e "\n${CYAN}[5/5] Verifying Eiko via setup.m...${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$SCRIPT_DIR/eiko/+eiko_lib/setup.m" ]; then
    echo -e "${RED}[!] Could not find 'setup.m' in 'eiko/+eiko_lib/'.${NC}"
    safe_exit 1
fi

echo -e "${MAGENTA}  -> Starting MATLAB to run system verification...${NC}"

if matlab -batch "addpath('$SCRIPT_DIR/eiko'); eiko_lib.setup;"; then
    echo -e "\n${GREEN}[*] Verification Complete: Eiko for MATLAB is operational!${NC}"
else
    echo -e "\n${RED}[!] Verification failed. MATLAB encountered an error while running setup.m.${NC}"
    safe_exit 1
fi

# ---------------------------------------------------------
# Step 6: Generate MATLAB Initialization Script
# ---------------------------------------------------------
LAUNCHER_PATH="$SCRIPT_DIR/start_eiko.m"
cat << EOF > "$LAUNCHER_PATH"
% Eiko Initialization Script
addpath('$SCRIPT_DIR/eiko');
disp('[*] Eiko is activated. Ready to compute!');
EOF

echo -e "\n${GREEN}====================================================${NC}"
echo -e "${GREEN} EIKO ENVIRONMENT INSTALLED SUCCESSFULLY${NC}"
echo -e "${GREEN}====================================================${NC}"

GRAY='\033[0;37m'
DARK_GRAY='\033[1;30m'

echo -e "\n${YELLOW}[!] How to use Eiko:${NC}"
echo -e -n "${GRAY}    Inside MATLAB, navigate to this folder and run: ${NC}"
echo -e -n "${CYAN}start_eiko.m${NC}\n"
echo -e "${DARK_GRAY}    (This will add Eiko to your path for the current session)\n${NC}"

safe_exit 0
