#!/bin/bash
# ==============================================================================
# Eiko Smart Linux Environment Installer (Ubuntu/Debian)
# ==============================================================================
set -e

# Colors for terminal output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
MAGENTA='\033[1;35m'
RED='\033[0;31m'
DARK_GRAY='\033[1;30m'
NC='\033[0m' # No Color

echo -e "${GREEN}====================================================${NC}"
echo -e "${GREEN} Starting Eiko Smart Environment Setup (Linux)      ${NC}"
echo -e "${GREEN}====================================================${NC}"

# Request sudo privileges upfront for system checks
echo -e "[*] Requesting sudo privileges for system checks..."
sudo -v
# Keep sudo alive while the script runs
while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &

# Ensure system is updated
sudo apt-get update -qq

# ---------------------------------------------------------
# Step 0: Check and Update NVIDIA Display Drivers
# ---------------------------------------------------------
echo -e "\n${CYAN}[0/5] Checking NVIDIA Display Drivers...${NC}"

# CUDA 12.6 requires driver version 550 or higher
MIN_DRIVER=550 
UPDATE_DRIVER=false

if command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | cut -d'.' -f1)
    echo -e "${DARK_GRAY}-> Found active NVIDIA driver: v${DRIVER_VER}${NC}"

    if [ "$DRIVER_VER" -lt "$MIN_DRIVER" ]; then
        echo -e "${YELLOW}-> Driver is too old for CUDA 12.6 (Requires v${MIN_DRIVER}+).${NC}"
        UPDATE_DRIVER=true
    fi
else
    echo -e "${YELLOW}-> No active NVIDIA display driver detected.${NC}"
    UPDATE_DRIVER=true
fi

if [ "$UPDATE_DRIVER" = true ]; then
    echo -e "${MAGENTA}WARNING: Updating Linux display drivers may cause a screen flicker and requires a reboot.${NC}"
    read -p "Would you like to automatically install the recommended NVIDIA driver now? (y/N): " confirm
    
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        echo -e "${MAGENTA}-> Fetching and installing the recommended proprietary driver...${NC}"
        sudo ubuntu-drivers autoinstall
        
        echo -e "\n${RED}====================================================${NC}"
        echo -e "${RED} CRITICAL: SYSTEM REBOOT REQUIRED                   ${NC}"
        echo -e "${RED}====================================================${NC}"
        echo -e "${YELLOW}The NVIDIA display driver has been updated. The CUDA toolkit cannot map to the GPU until the kernel reloads.${NC}"
        echo -e "Please reboot your computer, then run this script again to finish the Eiko installation."
        exit 0
    else
        echo -e "\n${RED}[!] Cannot proceed without a compatible NVIDIA driver.${NC}"
        echo -e "Update your drivers manually, reboot, and rerun this script."
        exit 1
    fi
fi

# ---------------------------------------------------------
# Step 1: Check and Install C++ Build Tools
# ---------------------------------------------------------
echo -e "\n${CYAN}[1/5] Checking GCC Build Tools...${NC}"
if dpkg -s build-essential >/dev/null 2>&1; then
    echo -e "${YELLOW}-> build-essential already installed. Skipping.${NC}"
else
    echo -e "${MAGENTA}-> Installing GCC and standard build utilities...${NC}"
    sudo apt-get install -y build-essential
fi

source /etc/os-release

# ---------------------------------------------------------
# Determine Target Matrix based on OS Version
# ---------------------------------------------------------
if [ "$VERSION_ID" == "20.04" ]; then
    echo -e "${YELLOW}-> Detected Ubuntu 20.04. Engaging Legacy Compatibility Mode...${NC}"
    TARGET_CUDA_PKG="cuda-toolkit-12-1"
    TARGET_CUDA_VER="12.1"
    TARGET_WHEEL_URL="https://download.pytorch.org/whl/cu121"
    TARGET_TORCH_VER="torch==2.4.0 torchvision"
    MIN_PYTHON_VER=3.8
else
    echo -e "${CYAN}-> Detected Modern Ubuntu. Engaging Next-Gen Mode...${NC}"
    TARGET_CUDA_PKG="cuda-toolkit-12-6"
    TARGET_CUDA_VER="12.6"
    TARGET_WHEEL_URL="https://download.pytorch.org/whl/cu126"
    TARGET_TORCH_VER="torch torchvision"
    MIN_PYTHON_VER=3.9
fi

# ---------------------------------------------------------
# Step 2: Check and Install Target CUDA
# ---------------------------------------------------------
echo -e "\n${CYAN}[2/5] Checking CUDA installation (Requires >= ${TARGET_CUDA_VER})...${NC}"
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
    echo -e "${MAGENTA}-> Installing ${TARGET_CUDA_PKG}...${NC}"
    OS_ID=${VERSION_ID//./}
    ARCH=$(uname -m)
    if [ "$ARCH" = "aarch64" ]; then ARCH="sbsa"; fi # NVIDIA uses 'sbsa' for ARM64 repos
    
    wget -qO cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/${OS_REPO_STR}/${ARCH}/cuda-keyring_1.1-1_all.deb"
    sudo dpkg -i cuda-keyring.deb
    rm cuda-keyring.deb
    
    sudo apt-get update -qq
    sudo apt-get install -y $TARGET_CUDA_PKG
fi

export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda

# ---------------------------------------------------------
# Step 3: Check and Install Python
# ---------------------------------------------------------
echo -e "\n${CYAN}[3/5] Checking Python installation...${NC}"
INSTALL_PYTHON=true
PYTHON_EXE="python3"

if command -v python3 >/dev/null 2>&1; then
    PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [ "$(printf '%s\n' "$MIN_PYTHON_VER" "$PY_VER" | sort -V | head -n1)" = "$MIN_PYTHON_VER" ]; then
        echo -e "${YELLOW}-> Found Python $PY_VER. Skipping installation.${NC}"
        INSTALL_PYTHON=false
    fi
fi

if [ "$INSTALL_PYTHON" = true ]; then
    echo -e "${MAGENTA}-> System Python is below v${MIN_PYTHON_VER}. Upgrading to 3.12 via Deadsnakes...${NC}"
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update -qq
    sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
    PYTHON_EXE="python3.12"
fi

# Ensure the native venv module is installed if we are using the system Python
if [ "$INSTALL_PYTHON" = false ]; then
    sudo apt-get install -y python${PY_VER}-venv python${PY_VER}-dev >/dev/null 2>&1 || true
fi

# ---------------------------------------------------------
# Step 4: Virtual Environment Setup
# ---------------------------------------------------------
# (Environment detection remains exactly the same)
VENV_PATH="$HOME/eiko"
if [ -n "$VIRTUAL_ENV" ]; then
    VENV_PATH="$VIRTUAL_ENV"
    echo -e "${GREEN}-> Detected active virtual environment at: $VENV_PATH${NC}"
else
    if [ ! -f "$VENV_PATH/bin/activate" ]; then
        echo -e "${MAGENTA}-> Creating new virtual environment...${NC}"
        $PYTHON_EXE -m venv "$VENV_PATH"
        
        # Inject CUDA paths into the activate script
        echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> "$VENV_PATH/bin/activate"
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> "$VENV_PATH/bin/activate"
    fi
fi


# ---------------------------------------------------------
# Step 5: Install Python Libraries (with Smart Bypass)
# ---------------------------------------------------------
echo -e "\n${CYAN}[5/5] Checking existing Eiko ML stack...${NC}"

run_pip_command() {
    local pip_exe="$VENV_PATH/bin/pip"
    set +e 
    "$pip_exe" "$@"
    local exit_code=$?
    set -e
    if [ $exit_code -ne 0 ]; then
        echo -e "\n${RED}[!] Error occurred during pip installation.${NC}"
        exit 1
    fi
}

run_pip_command install --upgrade pip --quiet

# Inline Python check for existing, valid PyTorch
TORCH_VALID=false
if "$VENV_PATH/bin/python" -c "
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
    echo -e "${GREEN}-> Found valid PyTorch (CUDA enabled, v2.4+). Skipping wheel downloads.${NC}"
    TORCH_VALID=true
fi

if [ "$TORCH_VALID" = false ]; then
    echo -e "${MAGENTA}-> Installing target PyTorch stack for CUDA ${TARGET_CUDA_VER}...${NC}"
    run_pip_command install $TARGET_TORCH_VER --index-url $TARGET_WHEEL_URL
fi

# Install Eiko (pip will natively accept the existing torch if TORCH_VALID was true)
echo -e "${MAGENTA}-> Installing Eiko...${NC}"
run_pip_command install eiko[jax] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# ---------------------------------------------------------
# Verification
# ---------------------------------------------------------
echo -e "\n${GREEN}====================================================${NC}"
echo -e "${GREEN} Verification Running...                            ${NC}"
echo -e "${GREEN}====================================================${NC}"

"$VENV_PATH/bin/python" -c "import eiko.eiko_torch; import eiko.eiko_jax; print('-> Success: Eiko, PyTorch, and JAX CUDA layers are fully operational!')"

echo -e "\n${GREEN}[*] Installation Complete!${NC}"
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "To activate your environment in the future, run: \n${CYAN}source $VENV_PATH/bin/activate${NC}"
fi
