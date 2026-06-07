#!/bin/bash
# ==============================================================================
# Eiko Smart Linux Environment Installer (Ubuntu/Debian)
# ==============================================================================
set -e # Exit immediately if a command exits with a non-zero status

# Colors for terminal output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
MAGENTA='\033[1;35m'
NC='\033[0m' # No Color

echo -e "${GREEN}====================================================${NC}"
echo -e "${GREEN} Starting Eiko Smart Environment Setup (Linux)      ${NC}"
echo -e "${GREEN}====================================================${NC}"

# Ask for the administrator password upfront
echo -e "[*] Requesting sudo privileges for system checks..."
sudo -v
# Keep sudo alive while the script runs
while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &

# Ensure system is updated
sudo apt-get update -qq

# ---------------------------------------------------------
# Step 1: Check and Install C++ Build Tools
# ---------------------------------------------------------
echo -e "\n${CYAN}[1/4] Checking MSVC/GCC Build Tools...${NC}"
if dpkg -s build-essential >/dev/null 2>&1; then
    echo -e "${YELLOW}-> build-essential already installed. Skipping.${NC}"
else
    echo -e "${MAGENTA}-> Installing GCC and standard build utilities...${NC}"
    sudo apt-get install -y build-essential
fi

# ---------------------------------------------------------
# Step 2: Check and Install CUDA (Requires >= 12.6)
# ---------------------------------------------------------
echo -e "\n${CYAN}[2/4] Checking CUDA installation...${NC}"
INSTALL_CUDA=true

if command -v nvcc >/dev/null 2>&1; then
    CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    if awk "BEGIN {exit !($CUDA_VER >= 12.6)}"; then
        echo -e "${YELLOW}-> Found CUDA $CUDA_VER. Skipping installation.${NC}"
        INSTALL_CUDA=false
    else
        echo -e "${MAGENTA}-> Found CUDA $CUDA_VER, but 12.6+ is required. Upgrading...${NC}"
    fi
fi

if [ "$INSTALL_CUDA" = true ]; then
    # Detect Ubuntu Version (e.g., "22.04" -> "2204")
    source /etc/os-release
    OS_ID=${VERSION_ID//./}
    
    echo -e "${MAGENTA}-> Fetching NVIDIA developer keys for Ubuntu $VERSION_ID...${NC}"
    wget -qO cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${OS_ID}/x86_64/cuda-keyring_1.1-1_all.deb"
    sudo dpkg -i cuda-keyring.deb
    rm cuda-keyring.deb
    
    sudo apt-get update -qq
    # Explicitly install the toolkit ONLY to protect display drivers
    sudo apt-get install -y cuda-toolkit-12-6
    
    # Export paths to current session
    export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi

# ---------------------------------------------------------
# Step 3: Check and Install Python (Requires >= 3.9)
# ---------------------------------------------------------
echo -e "\n${CYAN}[3/4] Checking Python installation...${NC}"
INSTALL_PYTHON=true
PYTHON_EXE="python3"

if command -v python3 >/dev/null 2>&1; then
    PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if awk "BEGIN {exit !($PY_VER >= 3.9)}"; then
        echo -e "${YELLOW}-> Found Python $PY_VER. Skipping installation.${NC}"
        INSTALL_PYTHON=false
    else
        echo -e "${MAGENTA}-> Found Python $PY_VER, but Eiko requires 3.9+. Upgrading to 3.12...${NC}"
    fi
fi

if [ "$INSTALL_PYTHON" = true ]; then
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update -qq
    sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
    PYTHON_EXE="python3.12"
fi

# Ensure python3-venv is installed for the system python just in case
if [ "$INSTALL_PYTHON" = false ]; then
    sudo apt-get install -y python${PY_VER}-venv python${PY_VER}-dev || true
fi

# ---------------------------------------------------------
# Step 4: Virtual Environment & Eiko Installation
# ---------------------------------------------------------
echo -e "\n${CYAN}[4/4] Preparing Virtual Environment and ML Stack...${NC}"
cd "$HOME"
VENV_PATH="$HOME/eiko"

if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo -e "${MAGENTA}-> Creating new virtual environment at $VENV_PATH...${NC}"
    $PYTHON_EXE -m venv "$VENV_PATH"
else
    echo -e "${YELLOW}-> Virtual environment already exists. Using existing environment.${NC}"
fi

# Use the virtual environment's pip directly
PIP_EXE="$VENV_PATH/bin/pip"

echo -e "${MAGENTA}-> Installing packages (existing dependencies will be natively skipped)...${NC}"
$PIP_EXE install --upgrade pip quiet
$PIP_EXE install torch torchvision --index-url https://download.pytorch.org/whl/cu126
$PIP_EXE install eiko[jax] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


# ---------------------------------------------------------
# Verification
# ---------------------------------------------------------
echo -e "\n${GREEN}====================================================${NC}"
echo -e "${GREEN} Verification Running...                            ${NC}"
echo -e "${GREEN}====================================================${NC}"
"$VENV_PATH/bin/python" -c "import eiko.eiko_torch; import eiko.eiko_jax; print('-> Success: Eiko and PyTorch CUDA layers are fully operational!')"

echo -e "\n${GREEN}[*] Installation Complete!${NC}"
echo -e "To activate your environment in the future, run: \n${CYAN}source ~/eiko/bin/activate${NC}"
