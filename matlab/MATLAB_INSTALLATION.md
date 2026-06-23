# Installing Eiko for MATLAB
- [Prerequisites](#prerequisites)
- [Automated Installer (Recommended)](#automated-installer--recommended-)
- [Manual Installation](#manual-installation)
- [Installing from Scratch](#installing-from-scratch)
  * [Windows](#windows)
  * [Linux](#linux)

## Prerequisites
Before continuing, ensure you have MATLAB installed along with the following toolboxes:
* Parallel Computing Toolbox (Required for GPU acceleration and MEX compilation)
* Deep Learning Toolbox (Required for differentiable operations)

## Automated Installer (Recommended)
The easiest way to configure your system for Eiko is to use our automated installation script. It will check your NVIDIA drivers, download the correct C++ build tools, and CUDA toolkit.

* **Windows:** Download [`install_eiko_matlab.bat`](./install_eiko_matlab.bat) and double-click the file.
* **Linux:** Download [`install_eiko_matlab.bat`](./install_eiko_matlab.bat), open a terminal, and run `bash install_eiko_matlab.bat`.

💡 Tip: The installer will generate a `start_eiko.m` script. You can run this script from MATLAB to activate Eiko!

## Manual Installation
If you already have a suitable C++ compiler and CUDA Toolkit installed, Eiko is easily installed by:
1. Downloading [Eiko](https://github.com/sebftw/Eiko/archive/refs/heads/main.zip) if it is not already downloaded.
2. Opening MATLAB, navigating to the `matlab` folder (the same folder this document is in).
3. Right-clicking the "eiko" folder and selecting **Add to Path &rarr; Selected Folder(s)**.

💡 **Tip:** To avoid doing this every time you launch MATLAB, you can add `addpath('/path/to/eiko')` to your `startup.m` file.

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
With both the compiler and CUDA installed, open MATLAB, navigate to the directory this guide is in, and run `eiko_lib.setup` to compile Eiko for your hardware and add the `eiko` folder to your search path.

💡 Tip: To avoid re-adding Eiko to your path every time you launch MATLAB, run the following command in the MATLAB Command Window after adding the eiko folder to path: `savepath`.
