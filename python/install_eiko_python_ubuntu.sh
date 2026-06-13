#!/bin/bash
# ==============================================================================
# Eiko Smart Linux Environment Installer (Ubuntu/Debian)
# ==============================================================================
# We remove global 'set -e' because it forces violent drops when sourced.
# Instead, we will catch errors surgically.

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
echo -e "${CYAN} Eiko Smart Environment Setup (Linux)               ${NC}"
echo -e "${CYAN}====================================================${NC}"

# ---------------------------------------------------------
# Step 0: Explanation and User Confirmation
# ---------------------------------------------------------
echo -e "\nThis script will configure your system for the Eiko environment by performing the following actions:"
echo -e "  1. Validate NVIDIA display driver compatibility."
echo -e "  2. Install the appropriate C++ Build Tools."
echo -e "  3. Install or verify the NVIDIA CUDA Toolkit based on your Ubuntu version."
echo -e "  4. Install Python 3.12 (if necessary) and configure an isolated virtual environment."
echo -e "  5. Install PyTorch, JAX, and Eiko."
echo -e "  6. Verify that Eiko was installed correctly."

echo -e "\n${YELLOW}[!] DISCLAIMER: This script requires sudo privileges and modifies system packages."
echo -e "    It is provided 'as-is' without any express or implied warranties. Run at your own risk.${NC}"

read -p $'\nDo you agree to these terms and want to proceed? (y/N): ' confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo -e "\n${YELLOW}[*] Setup cancelled by user.${NC}"
    safe_exit 0
fi

# Request sudo privileges upfront for system checks now that the user agreed
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

# Source OS release early to set driver versions dynamically
if [ -f /etc/os-release ]; then
    source /etc/os-release
fi

# Dynamically assign minimum driver based on targeted CUDA version
if [ "$VERSION_ID" == "20.04" ]; then
    MIN_DRIVER=550 # Requirement for CUDA 12.4
else
    MIN_DRIVER=560 # Requirement for CUDA 12.6
fi

UPDATE_DRIVER=false
IS_WSL=false

# Detect WSL
if grep -qi microsoft /proc/version || [ -n "$WSL_DISTRO_NAME" ]; then
    IS_WSL=true
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | cut -d'.' -f1)
    echo -e "  -> Found active NVIDIA driver: v${DRIVER_VER}"

    if [ "$DRIVER_VER" -lt "$MIN_DRIVER" ]; then
        echo -e "${YELLOW}  -> Driver is too old (Requires v${MIN_DRIVER}+).${NC}"
        
        if [ "$IS_WSL" = true ]; then
            echo -e "\n${RED}====================================================${NC}"
            echo -e "${RED} CRITICAL ERROR: NVIDIA DRIVER UPDATE REQUIRED       ${NC}"
            echo -e "${RED}====================================================${NC}"
            echo -e "${YELLOW}Your system requires NVIDIA display driver version ${MIN_DRIVER} or higher to run.${NC}"
            echo -e "${MAGENTA}Because Windows hardware matching is highly specific, this cannot be safely automated.${NC}"
            echo -e "\nPlease update your drivers manually:"
            echo -e "  1. Open the 'NVIDIA App' or 'GeForce Experience' on your Windows PC."
            echo -e "  2. Navigate to the 'Drivers' tab and install the latest update."
            echo -e "  3. Reboot your computer and run this script again."
            safe_exit 1
        else
            UPDATE_DRIVER=true
        fi
    fi
else
    echo -e "${YELLOW}  -> No active NVIDIA display driver detected.${NC}"
    if [ "$IS_WSL" = true ]; then
        echo -e "\n${RED}====================================================${NC}"
        echo -e "${RED} CRITICAL ERROR: NVIDIA DRIVER UPDATE REQUIRED       ${NC}"
        echo -e "${RED}====================================================${NC}"
        echo -e "${YELLOW}Your system requires NVIDIA display driver version ${MIN_DRIVER} or higher to run.${NC}"
        echo -e "${MAGENTA}Because Windows hardware matching is highly specific, this cannot be safely automated.${NC}"
        echo -e "\nPlease update your drivers manually:"
        echo -e "  1. Open the 'NVIDIA App' or 'GeForce Experience' on your Windows PC."
        echo -e "  2. Navigate to the 'Drivers' tab and install the latest update."
        echo -e "  3. Reboot your computer and run this script again."
        safe_exit 1
    else
        UPDATE_DRIVER=true
    fi
fi

# Only proceed with the auto-update logic if NOT in WSL
if [ "$UPDATE_DRIVER" = true ]; then
    echo -e "${MAGENTA}WARNING: Updating Linux display drivers may cause a screen flicker and requires a reboot.${NC}"
    read -p "Would you like to automatically install the recommended NVIDIA driver now? (y/N): " driver_confirm
    
    if [[ "$driver_confirm" =~ ^[Yy]$ ]]; then
        echo -e "${MAGENTA}  -> Fetching and installing the recommended proprietary driver...${NC}"
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
# Step 2: Check and Install C++ Build Tools
# ---------------------------------------------------------
echo -e "\n${CYAN}[2/5] Checking GCC Build Tools...${NC}"
if dpkg -s build-essential >/dev/null 2>&1; then
    echo -e "${YELLOW}  -> build-essential is already installed. Skipping.${NC}"
else
    echo -e "${MAGENTA}  -> Installing GCC and standard build utilities...${NC}"
    sudo apt-get install -y build-essential
fi

# ---------------------------------------------------------
# Determine Target Matrix based on OS Version
# ---------------------------------------------------------
if [ "$VERSION_ID" == "20.04" ]; then
    echo -e "${YELLOW}  -> Detected Ubuntu 20.04. Engaging Legacy Compatibility Mode...${NC}"
    TARGET_CUDA_PKG="cuda-toolkit-12-4"
    TARGET_CUDA_VER="12.4"
    TARGET_WHEEL_URL="https://download.pytorch.org/whl/cu124"
    TARGET_TORCH_VER="torch==2.4.1 torchvision"
    TARGET_JAX_VER="jax==0.4.13 jaxlib==0.4.13+cuda12.cudnn89"
    MIN_PYTHON_VER=3.8
    
    # Install GCC 10 toolchain (leaves default gcc/g++ untouched)
    if dpkg -s g++-10 >/dev/null 2>&1; then
        echo -e "${YELLOW}  -> GCC 10 toolchain already available.${NC}"
    else
        echo -e "${MAGENTA}  -> Installing GCC 10 toolchain for isolated C++20 compilation...${NC}"
        sudo apt-get install -y gcc-10 g++-10
    fi
else
    echo -e "  -> Detected Modern Ubuntu. Engaging Next-Gen Mode..."
    TARGET_CUDA_PKG="cuda-toolkit-12-6"
    TARGET_CUDA_VER="12.6"
    TARGET_WHEEL_URL="https://download.pytorch.org/whl/cu126"
    TARGET_TORCH_VER="torch torchvision"
    TARGET_JAX_VER="jax[cuda12]>=0.4.30"
    MIN_PYTHON_VER=3.9
fi

# ---------------------------------------------------------
# Step 3: Check and Install Target CUDA
# ---------------------------------------------------------
echo -e "\n${CYAN}[3/5] Checking CUDA installation (Requires >= ${TARGET_CUDA_VER})...${NC}"
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
    echo -e "${MAGENTA}  -> Installing ${TARGET_CUDA_PKG}...${NC}"
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
# Step 4: Check and Install Python & Virtual Environment
# ---------------------------------------------------------
echo -e "\n${CYAN}[4/5] Checking Python and Virtual Environment...${NC}"
INSTALL_PYTHON=true
PYTHON_EXE="python3"

if command -v python3 >/dev/null 2>&1; then
    PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [ "$(printf '%s\n' "$MIN_PYTHON_VER" "$PY_VER" | sort -V | head -n1)" = "$MIN_PYTHON_VER" ]; then
        echo -e "${YELLOW}  -> Found Python $PY_VER. Skipping system installation.${NC}"
        INSTALL_PYTHON=false
    fi
fi

if [ "$INSTALL_PYTHON" = true ]; then
    echo -e "${MAGENTA}  -> System Python is below v${MIN_PYTHON_VER}. Upgrading to 3.12 via Deadsnakes...${NC}"
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update -qq
    sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
    PYTHON_EXE="python3.12"
fi

if [ "$INSTALL_PYTHON" = false ]; then
    sudo apt-get install -y python${PY_VER}-venv python${PY_VER}-dev >/dev/null 2>&1 || true
fi

VENV_PATH="$HOME/eiko"
if [ -n "$VIRTUAL_ENV" ]; then
    VENV_PATH="$VIRTUAL_ENV"
    echo -e "${GREEN}  -> Detected active virtual environment at: $VENV_PATH${NC}"
else
    if [ ! -f "$VENV_PATH/bin/activate" ]; then
        echo -e "${MAGENTA}  -> Creating new virtual environment at $VENV_PATH...${NC}"
        $PYTHON_EXE -m venv "$VENV_PATH"
    fi
fi

# Helper function to append a line to the activate file ONLY if it doesn't exist
append_if_missing() {
    local line="$1"
    local file="$2"
    # Escapes backslashes and special chars to avoid grep pattern breakage
    if ! grep -Fq "$line" "$file" 2>/dev/null; then
        echo "$line" >> "$file"
    fi
}

ACTIVATE_SCRIPT="$VENV_PATH/bin/activate"

# Inject core CUDA paths safely (Idempotent)
append_if_missing 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' "$ACTIVATE_SCRIPT"
append_if_missing 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' "$ACTIVATE_SCRIPT"
append_if_missing 'export LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so:${LD_PRELOAD}' "$ACTIVATE_SCRIPT"

# Inject C++20 isolated compiler overrides if on Ubuntu 20.04 (Idempotent)
if [ "$VERSION_ID" == "20.04" ]; then
    append_if_missing 'if command -v g++-10 >/dev/null 2>&1; then export CC=gcc-10; export CXX=g++-10; fi' "$ACTIVATE_SCRIPT"
fi

echo -e "  -> Activating environment session..."
source "$ACTIVATE_SCRIPT"

# ---------------------------------------------------------
# Step 5: Install Python Libraries & Verify
# ---------------------------------------------------------
echo -e "\n${CYAN}[5/5] Checking existing Eiko ML stack...${NC}"

run_pip_command() {
    local pip_exe="pip"
    
    set +e
    # 1. 2>&1 merges stderr into stdout so grep can filter both
    # 2. --line-buffered prevents the terminal from appearing "frozen" during long steps
    "$pip_exe" --no-input "$@" 2>&1 | grep --line-buffered -E -v "Requirement already satisfied|does not provide the extra|Looking in links"
    
    local exit_code=${PIPESTATUS[0]}
    set -e
    
    if [ $exit_code -ne 0 ]; then
        echo -e "\n${RED}[!] Error occurred during pip execution.${NC}"
        return 1
    fi
    return 0
}

run_pip_command install --upgrade pip --no-input || safe_exit 1

TORCH_VALID=false
if python -c "
import sys
try:
    import torch
    v = torch.__version__.split('+')[0].split('.')
    if torch.version.cuda and (int(v[0]) > 2 or (int(v[0]) == 2 and int(v[1]) >= 4)):
        sys.exit(0)
except ImportError:
    pass
sys.exit(1)
" 2>/dev/null; then
    echo -e "${YELLOW}  -> Found valid PyTorch (CUDA enabled, v2.4+). Skipping installation.${NC}"
    TORCH_VALID=true
fi

if [ "$TORCH_VALID" = false ]; then
    echo -e "${MAGENTA}  -> Installing target PyTorch stack for CUDA ${TARGET_CUDA_VER}...${NC}"
    run_pip_command install $TARGET_TORCH_VER --index-url $TARGET_WHEEL_URL --no-input || safe_exit 1
fi

echo -e "${MAGENTA}  -> Installing Eiko dependencies...${NC}"
run_pip_command install $TARGET_JAX_VER --no-input -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html || safe_exit 1
run_pip_command install "eiko[jax]" --no-input || safe_exit 1

# ---------------------------------------------------------
# Final Verification
# ---------------------------------------------------------
echo -e "\n${MAGENTA}  -> Verifying Eiko installation...${NC}"
export TORCH_CUDA_ARCH_LIST=$(python3 -c "import torch; print('.'.join(map(str, torch.cuda.get_device_capability())))")

if python -c "import eiko.eiko_torch; import eiko.eiko_jax; print('  -> Success: Eiko, PyTorch, and JAX CUDA layers are fully operational!')"; then
    echo -e "\n${GREEN}[*] Installation Complete!${NC}"
    if [ -z "$VIRTUAL_ENV" ]; then
        echo -e "Your environment is now active in this shell! (Run 'source $ACTIVATE_SCRIPT' in new terminals)"
    fi
else
    echo -e "\n${RED}[!] Verification failed. Runtime environment setup is broken.${NC}"
    safe_exit 1
fi

safe_exit 0
