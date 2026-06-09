#!/bin/bash
# ==============================================================================
# Eiko Smart Linux Environment Installer (MATLAB Edition)
# ==============================================================================
# We remove global 'set -e' because it forces violent drops when sourced.
# Instead, we will catch errors surgically.

# Colors for terminal output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
MAGENTA='\033[1;35m'
RED='\033[0;31m'
DARK_GRAY='\033[1;30m'
NC='\033[0m' # No Color

# Helper function to exit cleanly whether sourced or executed directly
safe_exit() {
    local code=${1:-0}
    # Check if the script was sourced or executed directly
    if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
        return "$code"
    else
        exit "$code"
    fi
}

echo -e "${GREEN}====================================================${NC}"
echo -e "${GREEN} Starting Eiko MATLAB Environment Setup (Linux)     ${NC}"
echo -e "${GREEN}====================================================${NC}"

# Request sudo privileges upfront for system checks
echo -e "[*] Requesting sudo privileges for system checks..."
sudo -v || safe_exit 1
# Keep sudo alive while the script runs
while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &

# Ensure system is updated
sudo apt-get update -qq

# ---------------------------------------------------------
# Step 0: Check NVIDIA Driver Version
# ---------------------------------------------------------
echo -e "\n${CYAN}[0/5] Checking NVIDIA Driver Compatibility...${NC}"

MIN_DRIVER=550 
UPDATE_DRIVER=false
IS_WSL=false

# Detect WSL
if grep -qi microsoft /proc/version || [ -n "$WSL_DISTRO_NAME" ]; then
    IS_WSL=true
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | cut -d'.' -f1)
    echo -e "${DARK_GRAY}-> Found active NVIDIA driver: v${DRIVER_VER}${NC}"

    if [ "$DRIVER_VER" -lt "$MIN_DRIVER" ]; then
        echo -e "${YELLOW}-> Driver is too old for CUDA 12.4 (Requires v${MIN_DRIVER}+).${NC}"
        
        if [ "$IS_WSL" = true ]; then
            echo -e "\n${RED}====================================================${NC}"
            echo -e "${RED} CRITICAL: NVIDIA DRIVER UPDATE REQUIRED             ${NC}"
            echo -e "${RED}====================================================${NC}"
            echo -e "${YELLOW}Your system requires NVIDIA display driver version ${MIN_DRIVER} or higher to run.${NC}"
            echo -e "${MAGENTA}Because Windows hardware matching is highly specific, this cannot be safely automated.${NC}"
            echo -e "\nPlease update your drivers manually:"
            echo -e "  1. Open the 'NVIDIA App' or 'GeForce Experience' on your Windows PC."
            echo -e "  2. Navigate to the 'Drivers' tab and install the latest update."
            echo -e "  3. Reboot your computer and run this script again."
            read -p $'\nPress Enter to exit'
            safe_exit 1
        else
            UPDATE_DRIVER=true
        fi
    fi
else
    echo -e "${YELLOW}-> No active NVIDIA display driver detected.${NC}"
    if [ "$IS_WSL" = true ]; then
        echo -e "\n${RED}====================================================${NC}"
        echo -e "${RED} CRITICAL: NVIDIA DRIVER UPDATE REQUIRED             ${NC}"
        echo -e "${RED}====================================================${NC}"
        echo -e "${YELLOW}Your system requires NVIDIA display driver version ${MIN_DRIVER} or higher to run.${NC}"
        echo -e "${MAGENTA}Because Windows hardware matching is highly specific, this cannot be safely automated.${NC}"
        echo -e "\nPlease update your drivers manually:"
        echo -e "  1. Open the 'NVIDIA App' or 'GeForce Experience' on your Windows PC."
        echo -e "  2. Navigate to the 'Drivers' tab and install the latest update."
        echo -e "  3. Reboot your computer and run this script again."
        read -p $'\nPress Enter to exit'
        safe_exit 1
    else
        UPDATE_DRIVER=true
    fi
fi

# Only proceed with the auto-update logic if NOT in WSL
if [ "$UPDATE_DRIVER" = true ]; then
    echo -e "${MAGENTA}WARNING: Updating Linux display drivers may cause a screen flicker and requires a reboot.${NC}"
    read -p "Would you like to automatically install the recommended NVIDIA driver now? (y/N): " confirm
    
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        echo -e "${MAGENTA}-> Fetching and installing the recommended proprietary driver...${NC}"
        sudo ubuntu-drivers autoinstall
        
        echo -e "\n${RED}====================================================${NC}"
        echo -e "${RED} CRITICAL: SYSTEM REBOOT REQUIRED                    ${NC}"
        echo -e "${RED}====================================================${NC}"
        echo -e "${YELLOW}The NVIDIA display driver has been updated. The CUDA toolkit cannot map to the GPU until the kernel reloads.${NC}"
        echo -e "Please reboot your computer, then run this script again to finish the Eiko installation."
        safe_exit 0
    else
        echo -e "\n${RED}[!] Cannot proceed without a compatible NVIDIA driver.${NC}"
        echo -e "Update your drivers manually, reboot, and rerun this script."
        safe_exit 1
    fi
fi

# ---------------------------------------------------------
# Step 1: Locating MATLAB & Probing GCC Requirements
# ---------------------------------------------------------
echo -e "\n${CYAN}[1/5] Locating MATLAB and Probing Compiler Requirements...${NC}"

if ! command -v matlab >/dev/null 2>&1; then
    echo -e "${RED}[!] 'matlab' command not found in your system PATH.${NC}"
    echo -e "${YELLOW}Please ensure MATLAB is installed and correctly added to your environment variables.${NC}"
    safe_exit 1
fi

echo -e "${DARK_GRAY}-> MATLAB found. Booting headlessly to query supported C++ compilers...${NC}"

# Headless probe script: iterates over supported C++ compilers, looks for GNU, and extracts the major version number.
PROBE_SCRIPT="cc=mex.getCompilerConfigurations('C++','Supported'); for i=1:length(cc), if ~isempty(strfind(lower(cc(i).Manufacturer), 'gnu')), m=regexp(cc(i).Version, '^(\d+)', 'tokens', 'once'); if ~isempty(m), fprintf('GCC_MAJOR:%s\n', m{1}); exit; end; end; end; fprintf('GCC_MAJOR:UNKNOWN\n'); exit;"

MATLAB_OUT=$(matlab -batch "$PROBE_SCRIPT" 2>&1)

if [[ "$MATLAB_OUT" =~ GCC_MAJOR:([0-9]+) ]]; then
    TARGET_GCC_VER="${BASH_REMATCH[1]}"
    echo -e "${GREEN}-> MATLAB explicitly requested GCC ${TARGET_GCC_VER}.${NC}"
else
    TARGET_GCC_VER="10" # Safe default for older LTS systems
    echo -e "${YELLOW}-> MATLAB compiler query failed or returned unrecognized output. Defaulting to GCC 10.${NC}"
fi

# ---------------------------------------------------------
# Step 2: Check and Install C++ Build Tools
# ---------------------------------------------------------
echo -e "\n${CYAN}[2/5] Checking Required GCC Build Tools...${NC}"

if dpkg -s build-essential >/dev/null 2>&1; then
    echo -e "${DARK_GRAY}-> build-essential already installed.${NC}"
else
    echo -e "${MAGENTA}-> Installing standard build utilities (build-essential)...${NC}"
    sudo apt-get install -y build-essential
fi

# Install the dynamically targeted GCC version
if dpkg -s "g++-${TARGET_GCC_VER}" >/dev/null 2>&1; then
    echo -e "${YELLOW}-> GCC ${TARGET_GCC_VER} toolchain already available.${NC}"
else
    echo -e "${MAGENTA}-> Installing GCC ${TARGET_GCC_VER} toolchain to match MATLAB requirements...${NC}"
    sudo apt-get install -y "gcc-${TARGET_GCC_VER}" "g++-${TARGET_GCC_VER}" || {
        echo -e "${YELLOW}[!] Warning: gcc-${TARGET_GCC_VER} could not be installed (may require a PPA update on this OS).${NC}"
    }
fi

# Always ensure GCC 10 is available as an explicit fallback/C++20 requirement
if [ "$TARGET_GCC_VER" != "10" ]; then
    if dpkg -s g++-10 >/dev/null 2>&1; then
        echo -e "${YELLOW}-> GCC 10 toolchain already available.${NC}"
    else
        echo -e "${MAGENTA}-> Installing GCC 10 toolchain (secondary requirement)...${NC}"
        sudo apt-get install -y gcc-10 g++-10
    fi
fi

# ---------------------------------------------------------
# Step 3: Check and Install Pinned CUDA Toolkit (12.4)
# ---------------------------------------------------------
TARGET_CUDA_PKG="cuda-toolkit-12-4"
TARGET_CUDA_VER="12.4"
source /etc/os-release

echo -e "\n${CYAN}[3/5] Checking CUDA installation (Pinned: ${TARGET_CUDA_VER})...${NC}"
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
        echo -e "${YELLOW}-> Found CUDA $CUDA_VER. Skipping installation.${NC}"
        INSTALL_CUDA=false
    fi
fi

if [ "$INSTALL_CUDA" = true ]; then
    echo -e "${MAGENTA}-> Installing strict known-good matrix: ${TARGET_CUDA_PKG}...${NC}"
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
# Step 4: Execute setup.m
# ---------------------------------------------------------
echo -e "\n${CYAN}[4/5] Verifying Eiko via setup.m...${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$SCRIPT_DIR/eiko/+eiko_lib/setup.m" ]; then
    echo -e "${RED}[!] Could not find 'setup.m' in 'eiko/+eiko_lib/'.${NC}"
    safe_exit 1
fi

echo -e "${MAGENTA}-> Booting MATLAB to run verification...${NC}"
# -batch runs MATLAB headlessly without splash/desktop and exits with an error code if the script fails.
if matlab -batch "addpath('$SCRIPT_DIR/eiko'); eiko_lib.setup;"; then
    echo -e "\n${GREEN}[*] Verification Complete: Eiko for MATLAB is operational!${NC}"
else
    echo -e "\n${RED}[!] Verification failed. MATLAB encountered an error while running setup.m.${NC}"
    safe_exit 1
fi
