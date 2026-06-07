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

# Ask for the administrator password upfront for system checks
echo -e "[*] Requesting sudo privileges for system checks..."
sudo -v
# Keep sudo alive while the script runs
while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &

# Ensure system is updated
sudo apt-get update -qq

# ---------------------------------------------------------
# Step 1: Check and Install C++ Build Tools
# ---------------------------------------------------------
echo -e "\n${CYAN}[1/5] Checking MSVC/GCC Build Tools...${NC}"
if dpkg -s build-essential >/dev/null 2>&1; then
    echo -e "${YELLOW}-> build-essential already installed. Skipping.${NC}"
else
    echo -e "${MAGENTA}-> Installing GCC and standard build utilities...${NC}"
    sudo apt-get install -y build-essential
fi

# ---------------------------------------------------------
# Step 2: Check and Install CUDA (Requires >= 12.6)
# ---------------------------------------------------------
echo -e "\n${CYAN}[2/5] Checking CUDA installation...${NC}"
INSTALL_CUDA=true

if command -v nvcc >/dev/null 2>&1; then
    CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    if awk "BEGIN {exit !($CUDA_VER >= 12.6)}"; then
        echo -e "${YELLOW}-> Found CUDA $CUDA_VER. Skipping installation.${NC}"
        INSTALL_CUDA=false
    fi
fi

if [ "$INSTALL_CUDA" = true ]; then
    echo -e "${MAGENTA}-> Installing/Upgrading to CUDA 12.6...${NC}"
    source /etc/os-release
    OS_ID=${VERSION_ID//./}
    
    wget -qO cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${OS_ID}/x86_64/cuda-keyring_1.1-1_all.deb"
    sudo dpkg -i cuda-keyring.deb
    rm cuda-keyring.deb
    
    sudo apt-get update -qq
    sudo apt-get install -y cuda-toolkit-12-6
    
    # Export paths to current session
    export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi

# ---------------------------------------------------------
# Step 3: Check and Install Python (Requires >= 3.9)
# ---------------------------------------------------------
echo -e "\n${CYAN}[3/5] Checking Python installation...${NC}"
INSTALL_PYTHON=true
PYTHON_EXE="python3"

if command -v python3 >/dev/null 2>&1; then
    PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if awk "BEGIN {exit !($PY_VER >= 3.9)}"; then
        echo -e "${YELLOW}-> Found Python $PY_VER. Skipping installation.${NC}"
        INSTALL_PYTHON=false
    fi
fi

if [ "$INSTALL_PYTHON" = true ]; then
    echo -e "${MAGENTA}-> Installing Python 3.12...${NC}"
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update -qq
    sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
    PYTHON_EXE="python3.12"
fi

# Ensure python3-venv is installed for the system python just in case
if [ "$INSTALL_PYTHON" = false ]; then
    sudo apt-get install -y python${PY_VER}-venv python${PY_VER}-dev >/dev/null 2>&1 || true
fi

# ---------------------------------------------------------
# Step 4: Virtual Environment Setup & Detection
# ---------------------------------------------------------
echo -e "\n${CYAN}[4/5] Preparing Virtual Environment...${NC}"
VENV_PATH=""

if [ -n "$VIRTUAL_ENV" ]; then
    # Scenario A: User already activated a venv before running the script
    VENV_PATH="$VIRTUAL_ENV"
    echo -e "${GREEN}-> Detected active virtual environment at: $VENV_PATH${NC}"
    echo -e "${YELLOW}-> Skipping environment creation. Integrating directly.${NC}"
else
    # Scenario B & C: No active venv, check for existing or create new
    VENV_PATH="$HOME/eiko"
    
    if [ -f "$VENV_PATH/bin/activate" ]; then
        echo -e "${YELLOW}-> Found existing 'eiko' environment at $VENV_PATH.${NC}"
    else
        echo -e "${MAGENTA}-> Creating new virtual environment at $VENV_PATH...${NC}"
        
        if ! command -v $PYTHON_EXE >/dev/null 2>&1; then
            echo -e "${RED}[!] Python executable not found in PATH.${NC}"
            exit 1
        fi
        
        $PYTHON_EXE -m venv "$VENV_PATH"
    fi
fi

# ---------------------------------------------------------
# Step 5: Install Python Libraries (with Network Resilience)
# ---------------------------------------------------------
echo -e "\n${CYAN}[5/5] Checking and Installing Eiko ML stack...${NC}"

# Safe function targeting real executable to trap errors cleanly
run_pip_command() {
    local pip_exe="$VENV_PATH/bin/pip"
    
    # Disable exit-on-error temporarily to trap the specific failure
    set +e 
    "$pip_exe" "$@"
    local exit_code=$?
    set -e
    
    if [ $exit_code -ne 0 ]; then
        echo -e "\n${RED}[!] Error occurred during pip installation.${NC}"
        echo -e "${DARK_GRAY}Command failed: pip $*${NC}"
        echo -e "${YELLOW}Please check your network or package variations and try again.${NC}"
        exit 1
    fi
}

# Array elements cleanly map to command line arguments
run_pip_command install --upgrade pip quiet
run_pip_command install torch torchvision --index-url https://download.pytorch.org/whl/cu126
run_pip_command install jax jaxlib pybind11 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
run_pip_command install eiko

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
