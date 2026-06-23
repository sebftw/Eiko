# Installing Eiko for Python
- [Installing Eiko for Python](#installing-eiko-for-python)
  * [Standard Installation](#standard-installation)
    + [Optional Dependencies](#optional-dependencies)
  * [Installation from Source](#installation-from-source)
  * [Installing from Scratch](#installing-from-scratch)
    + [Linux](#linux)
      - [CUDA](#cuda)
      - [Compiler](#compiler)
      - [Python](#python)
    + [Windows](#windows)
      - [CUDA](#cuda-1)
      - [Compiler](#compiler-1)
      - [Python](#python-1)
  * [Zero-Installation Setups](#zero-installation-setups)
    + [Google Colab (no GPU required)](#google-colab--no-gpu-required-)


##  Automated Installer (Recommended)
If you are new to Python or want to set up your environment instantly, use the automated installation scripts. This script will validate your NVIDIA drivers, install CUDA and C++ Build Tools, set up Python, create a virtual environment, and install the complete Eiko ML stack.

* **Windows:** Download [`install_eiko_python.bat`](./install_eiko_python.bat) and double-click the file.
* **Linux:** Download [`install_eiko_python.bat`](./install_eiko_python.bat), open a terminal, and run `bash install_eiko_python.bat`.

💡 Tip: The installer will generate a `start_eiko` script in the same folder. You can double-click this file at any time to instantly open a terminal with your Eiko environment activated and ready to go!

## Manual Installation
If you already have a functional environment, you can install Eiko manually. We strongly recommend installing inside a virtual environment (`venv` or `conda`).
```bash
pip install eiko
```
Eiko requires PyTorch or JAX as a backend, so you must also run one of the following if you don't already have Torch or JAX installed:
* **PyTorch:** `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130`
* or **JAX:** `pip install jax[cuda13]`

(See the [PyTorch Get Started guide](https://pytorch.org/get-started/) and [JAX Installation guide](https://docs.jax.dev/en/latest/installation.html) for other possible configurations)


## Installation from Source
To download Eiko's source from GitHub and install locally, run:
```
git clone https://github.com/sebftw/Eiko.git
cd Eiko
pip install -e .
```
This includes the example code, MATLAB support, and more.
If you intend to run the example code, you should also run the following to get the required dependencies
```
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
* Install PyTorch and Eiko inside this compiler-ready terminal: <br>
```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install eiko
```
(See [PyTorch Get Started](https://pytorch.org/get-started/) if you want to adjust the installation commands for a different PyTorch version).
* Compile Eiko for your hardware and verify your installation by running:
```bat
python -c "import eiko.eiko_torch; print('Success!')"
```

### Linux
#### Display Drivers & CUDA
First, update your display drivers to ensure compatibility with recent CUDA versions:
```
sudo ubuntu-drivers install
sudo reboot
```
You can verify the driver installation by running `nvidia-smi`.

Next, navigate to the [NVIDIA CUDA Downloads page](https://developer.nvidia.com/cuda-downloads). Run `uname -m && cat /etc/os-release` in your terminal to determine your OS version, select `deb (network)`, and follow the provided instructions to install the toolkit. Verify the installation by running `nvcc --version`.


#### CUDA
First, update your display drivers to ensure compatibility with the latest version of CUDA.
```
sudo ubuntu-drivers install
```
After installing the new driver, you want to restart your system (`sudo reboot`).
You can verify that the driver was installed by runnning `nvidia-smi`.


Then, navigate to [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) to download CUDA.
Run `uname -mr && cat /etc/os-release` to find out which version you should download. Select `deb (network)`, and follow the provided instructions.

To verify that CUDA was installed, run `nvcc --version`.

#### Compiler
Next, you want a C++ compiler. Open the terminal and run
```
sudo apt update && sudo apt upgrade -y
sudo apt install build-essential -y
```
You can verify that the compiler is installed successfully by running `gcc --version`.

#### Python
Finally, you need to install Python with Eiko.
```
sudo apt install python3 python3-venv python3-pip python3-distutils python3-dev -y
```
This installs Python along with `venv` (virtual environment) and more, which is highly recommended to avoid clashing package installations.

Close your terminal and open a new one to ensure all system paths are updated.
Then, create and activate a new environment for Eiko:
```
python3 -m venv eiko
source eiko/bin/activate
```
After activation, your command line prompt should begin with `(eiko)`.

Now you can install PyTorch. Navigate to [PyTorch Get Started](https://pytorch.org/get-started/) and select **Stable**, **Linux**, **Pip**, and **Python**. For **Compute Platform**, select the CUDA version that matches the output of `nvcc --version`.
The webpage will then tell you which commands to run to get PyTorch.

> If you get an error "`Command 'pip3' not found`" you must replace the line "`pip3 ...`" with "`python3 -m pip ...`".

Verify your PyTorch installation with:
```python
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Finally, install and compile Eiko:
```
python3 -m pip install eiko
python3 -c "import eiko.eiko_torch"
```
You are now ready to use Eiko.

**Beware**: Every time you open a new terminal, you must run `source eiko/bin/activate` to reactivate the eiko environment.

### Windows Installation

You can set up your entire environment (CUDA, C++ Compilers, Python 3.12, PyTorch, and Eiko) completely automatically.

1. Download the **`install_eiko.ps1`** script to your computer.
2. Right-click the file and select **Run with PowerShell**.
3. Grant Administrator access if prompted, and let the installer configure your machine.

Once the window closes, your setup is complete, and you can use Eiko!

**Beware:** Whenever you open a fresh terminal window to work with Eiko again, you must activate the environment with:
```powershell
& "$env:USERPROFILE\eiko\Scripts\Activate.ps1"
```
if using PowerShell, or
```dos
%USERPROFILE%\eiko\Scripts\activate.bat
```
if using Command Prompt (cmd).

## Zero-Installation Setups
Since the installation process is quite involved, this section provides a an alternative way to running Eiko.

### Google Colab (no GPU required)
This option is ideal if you do not own a NVIDIA GPU.
Google Colab provides free GPU access, so you can run Eiko there through the following notebook: [Eiko in Colab](https://colab.research.google.com/github/sebftw/Eiko/blob/main/examples/python/eiko_in_colab.ipynb)







