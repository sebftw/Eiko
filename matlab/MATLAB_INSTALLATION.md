# Installing Eiko for MATLAB
- [Prerequisites](#prerequisites)
- [Manual Installation (Recommended)](#manual-installation)
- [Automated Installer](#automated-installer)
- [Installing from Scratch](#installing-from-scratch)
  * [Windows](#windows)
  * [Linux](#linux)

## Prerequisites
Before continuing, ensure you have MATLAB installed with the following toolboxes:
* Parallel Computing Toolbox
* Deep Learning Toolbox (Optional, only required for differentiable operations)

## Manual Installation
This is the recommended approach for MATLAB:
1. Download [`eiko_matlab.zip`](https://github.com/sebftw/Eiko/releases/latest/download/eiko_matlab.zip) and extract the "eiko" folder.
2. Inside MATLAB, right-click the `eiko` folder and select **Add to Path** → **Selected Folder(s)**.
3. Eiko is now installed.

💡 **Tip:** To avoid doing this every time you launch MATLAB, you can add `addpath('/path/to/eiko')` to your `startup.m` file.

## Automated Installer
Another way to configure your system for Eiko is to use our automated installation script. It will check your NVIDIA drivers, download the correct C++ build tools, and the CUDA toolkit. This allows it to compile Eiko specifically for your system.

* **Windows:** Download [`install_eiko_matlab.bat`](./install_eiko_matlab.bat) and double-click the file.
* **Linux:** Download [`install_eiko_matlab.bat`](./install_eiko_matlab.bat), open a terminal, and run `bash install_eiko_matlab.bat`.

💡 Tip: The installer will generate a `start_eiko.m` script. You can run this script from MATLAB to activate Eiko!

## Installing from Scratch
If you prefer to set up your environment entirely by hand, follow these steps to install the required dependencies before running Eiko.
Download the latest version of Eiko [here](https://github.com/sebftw/Eiko/archive/refs/heads/main.zip) and follow the [Windows](#windows) or [Linux](#linux) instructions below.

### Windows

#### Display Drivers & CUDA
* Press the **Windows key**, search for the **NVIDIA App** (or GeForce Experience), and navigate to the **Drivers** tab and install the latest driver.
* Once updated, download and install the CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads). (Note: Recent MATLAB versions ship with a built-in CUDA compiler, so this step can often be skipped).

#### Compiler
After installing CUDA, you need a compatible C++ compiler.
* Navigate to [Visual Studio Downloads](https://visualstudio.microsoft.com/downloads/) and download the Visual Studio Installer.
* Install **Visual Studio Community**.
* During installation, select the **Desktop development with C++ workload**. On the right-hand "Installation details" panel, ensure that **MSVC v143 - VS 2022 C++ x64/x86 build tools** is checked.

#### Finalizing Setup
Press the **Windows key** and search for and open **x64 Native Tools Command Prompt for VS 2022**. Type:
```
matlab
```
Opening MATLAB from this specific terminal (not standard PowerShell) ensures the correct compiler paths are loaded. Once MATLAB opens, you can right-click the "eiko" folder and select **Add to Path &rarr; Selected Folder(s)** and run `eiko_lib.setup` to compile Eiko for your hardware.

💡 Tip: To avoid re-adding Eiko to your path every time you launch MATLAB, run the following command in the MATLAB Command Window after adding the eiko folder to path: `savepath`.

### Linux

#### Display Drivers & CUDA
First, update your drivers to ensure they're compatible with the latest version of CUDA:
```bash
sudo ubuntu-drivers install
sudo reboot
```
You can verify that the driver was installed successfully by running `nvidia-smi`.

Next, navigate to the [NVIDIA CUDA Downloads page](https://developer.nvidia.com/cuda-downloads). Run `uname -m && cat /etc/os-release` in your terminal to determine your OS version, select `deb (network)`, and follow the provided instructions to install the toolkit. Verify the installation by running `nvcc --version`.

#### Compiler
You will need a C++ compiler to build the MATLAB MEX files. Open your terminal and run:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install build-essential -y
```
Verify the compiler is installed by running `gcc --version`.

#### Finalizing Setup
With both the compiler and CUDA installed, open MATLAB, navigate the "eiko" folder, and run `eiko_lib.setup` to compile Eiko for your hardware and add the `eiko` folder to your search path.

💡 Tip: To avoid re-adding Eiko to your path every time you launch MATLAB, run the following command in the MATLAB Command Window after adding the eiko folder to path: `savepath`.
