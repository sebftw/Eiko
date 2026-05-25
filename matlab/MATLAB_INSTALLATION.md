# Installing Eiko for MATLAB
- [Installing Eiko for MATLAB](#installing-eiko-for-matlab)
  * [Standard Installation](#standard-installation)
  * [Installing from Scratch](#installing-from-scratch)
    + [Linux](#linux)
      - [CUDA](#cuda)
      - [Compiler](#compiler)
    + [Windows](#windows)
      - [CUDA](#cuda-1)
      - [Compiler](#compiler-1)

## Standard Installation
Run `setup.m` to install Eiko for MATLAB.

If you encounter any issues, try following the instructions in the [Installing from Scratch](#installing-from-scratch) section.

## Installing from Scratch
If you do not have a compiler or CUDA yet, this guide is for you!

Download the latest version of Eiko [here](https://github.com/sebftw/Eiko/archive/refs/heads/main.zip) and follow the [Windows](#windows) or [Linux](#linux) instructions below.

### Linux
#### CUDA
First, you want to update your drivers to ensure they're compatible with the latest version of CUDA.
```
sudo ubuntu-drivers install
```
Then, you must reboot your system (`sudo reboot`). You can verify that the driver was installed by runnning `nvidia-smi`.


Then, navigate to [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) to download CUDA. Run `uname -mr` and `cat /etc/os-release` to find out which version you should download, select `deb (network)`, and follow the instructions.
To verify that CUDA was installed, run `nvcc --version`.

#### Compiler
Next, you want a C++ compiler. Open the terminal and run
```
sudo apt update && sudo apt upgrade -y
sudo apt install build-essential -y
```
The first line updates your system, and the second line installs a C++ compiler.

You can verify that the compiler is installed sucessfully by running `gcc --version`.

Now that you have both a compiler and CUDA, you should be able to install Eiko for MATLAB by running `setup.m` from the Eiko folder.

The setup script will append the required files to your MATLAB path.

### Windows

#### CUDA
To install CUDA, you first want to ensure your drivers are up to date, so they are compatible with the latest CUDA version.
Therefore, press the **Windows key**, type **NVIDIA App** (or GeForce Experience), and open the update application. Navigate to **Drivers** and install the latest version.

Then, CUDA can be downloaded from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).

#### Compiler
After installing CUDA, you need a valid compiler. Navigate to [Visual Studio Downloads](https://visualstudio.microsoft.com/downloads/) to download the Visual Studio Installer.
* You want to install **Visual Studio Community 2022**. If you can't find it, you can just select the latest version and later choose 2022 inside the Visual Studio Installer program.
* Inside the Visual Studio Installer, select the **Desktop development with C++** workload.
* Look for the "Installation details" panel on the right of the installer, and make sure **MSVC v143 - VS 2022 C++ x64/x86 build tools** is checked.

After installing VS 2022, press the Windows key and search for **x64 Native Tools Command Prompt for VS 2022**. You *must* use this specific command prompt, not standard PowerShell, as it loads the necessary C++ compiler paths.
Then open MATLAB from within this command line by running 
```
matlab
```
MATLAB should open up, and you can run `setup.m` from the Eiko folder to install Eiko.

The setup script will append the required files to your MATLAB path.