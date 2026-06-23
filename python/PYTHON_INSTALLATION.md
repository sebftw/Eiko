# Installing Eiko for Python
- [Automated Installer (Recommended)](#automated-installer)
- [Manual Installation](#manual-installation)
- [Installation from Source](#installation-from-source)
- [Installing from Scratch](#installing-from-scratch)
  * [Windows](#windows)
  * [Linux](#linux)
- [Google Colab (no GPU required)](#google-colab)

##  Automated Installer
If you are new to Python or want to set up your environment instantly, use the automated installation scripts. This script will validate your NVIDIA drivers, install CUDA and C++ Build Tools, set up Python, create a virtual environment, and install the complete Eiko ML stack.

* **Windows:** Download [`install_eiko_python.bat`](./install_eiko_python.bat) and double-click the file.
* **Linux:** Download [`install_eiko_python.bat`](./install_eiko_python.bat), open a terminal, and run `bash install_eiko_python.bat`.

💡 Tip: The installer will generate a `start_eiko` script. Run this script at any time to open a terminal with your Eiko environment activated and ready to go!

## Manual Installation
If you already have a functional environment, you can install Eiko manually. We strongly recommend installing inside a virtual environment (`venv` or `conda`).
```bash
pip install eiko
```
Eiko requires PyTorch or JAX as a backend, so you must also run one of the following if you don't already have Torch or JAX installed:
* **PyTorch:** `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130`
* or **JAX:** `pip install jax[cuda13]`

(See the [PyTorch Get Started guide](https://pytorch.org/get-started/) and [JAX Installation guide](https://docs.jax.dev/en/latest/installation.html) for installation options)


## Installation from Source
To download Eiko's source from GitHub and install locally, run:
```bash
git clone https://github.com/sebftw/Eiko.git
cd Eiko
pip install -e .
```
This includes the example code, MATLAB support, and more.
If you intend to run the example code, you should also run the following to get the required dependencies
```bash
pip install -e ".[examples]"
```

## Installing from Scratch
If you prefer to configure everything manually, this guide will walk you through setting up a compiler, CUDA, and Python from scratch.

### Windows

#### Display Drivers & CUDA
* Press the **Windows key**, search for the **NVIDIA App** (or GeForce Experience), and navigate to the **Drivers** tab and install the latest driver.
* Once updated, download and install the CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).

#### Compiler & Python
After installing CUDA, you need a compatible C++ compiler.
* Navigate to [Visual Studio Downloads](https://visualstudio.microsoft.com/downloads/) and download the Visual Studio Installer.
* Install **Visual Studio Community**. During installation, select the **Desktop development with C++ workload**.
* Download and install Python from [python.org](https://www.python.org/downloads/windows/). Ensure you check the "Add python.exe to PATH" box during installation.

#### Environment Setup & Installation
To ensure the compiler and CUDA paths are correctly exposed to Python during installation, do not use standard PowerShell. Instead:
* Press the **Windows key** and search for **x64 Native Tools Command Prompt for VS** (or VS 2022). Open it.
* Navigate to your desired installation directory and create/activate a new virtual environment: <br>
```bat
python -m venv eiko
call eiko\Scripts\activate.bat
```
💡 Tip: The virtual environment folder `eiko` is portable and self-contained. You can move it anywhere, and it will still work.

* Install PyTorch and Eiko inside this virtual environment: <br>
```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install eiko
```
(See [PyTorch Get Started](https://pytorch.org/get-started/) if you want to adjust the installation commands for a different PyTorch version).
* Compile Eiko for your hardware and verify your installation by running:
```bat
python -c "import eiko.eiko_torch; print('Success!')"
```

(Note: Every time you open a new Command Prompt, you must run `eiko\Scripts\activate.bat` before using Eiko).

### Linux
#### Display Drivers & CUDA
First, update your display drivers to ensure compatibility with recent CUDA versions:
```bash
sudo ubuntu-drivers install
sudo reboot
```
You can verify the driver installation by running `nvidia-smi`.

Next, navigate to the [NVIDIA CUDA Downloads page](https://developer.nvidia.com/cuda-downloads). Run `uname -m && cat /etc/os-release` in your terminal to determine your OS version, select `deb (network)`, and follow the provided instructions to install the toolkit. Verify the installation by running `nvcc --version`.

#### Compiler
Next, you need a C++ compiler. Open your terminal and run:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install build-essential -y
```
You can verify that the compiler is installed successfully by running `gcc --version`.

#### Python & Environment Setup
Finally, install Python and its virtual environment modules:
```bash
sudo apt install python3 python3-venv python3-pip python3-dev -y
```
Close your terminal and open a new one to ensure all system paths are updated. Then, create and activate a new environment for Eiko:
```bash
python3 -m venv eiko
source eiko/bin/activate
```
Now, install PyTorch, JAX, and Eiko:
```bash
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python3 -m pip install jax[cuda13]
python3 -m pip install eiko
```
(See [PyTorch Get Started guide](https://pytorch.org/get-started/) or [JAX Installation guide](https://docs.jax.dev/en/latest/installation.html) if you want to adjust these installation commands).

Verify your PyTorch and Eiko installation:
```bash
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python3 -c "import eiko.eiko_torch; import eiko.eiko_jax; print('Eiko loaded successfully!')"
```

(Note: Every time you open a new terminal, you must run `source eiko/bin/activate` before using Eiko).


## Google Colab
If the installation process is too involved or you do not have a dedicated NVIDIA GPU, you can run Eiko entirely in the cloud.

You can start using Eiko immediately by opening our interactive notebook: [Eiko in Colab](https://colab.research.google.com/github/sebftw/Eiko/blob/main/examples/python/eiko_in_colab.ipynb)







