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

## Standard Installation
Run
```
pip install eiko
```
You now have Eiko installed and ready for use.

If you encounter any issues, try following the instructions in the [Installing from Scratch](#installing-from-scratch) section.

### Optional Dependencies
You can install additional dependencies for Eiko.
* To run visualization examples: `pip install eiko[examples]`
* To install with JAX: `pip install eiko[jax]`

The example scripts can be found in [Eiko examples](../examples/README.md).

## Installation from Source
To download Eiko's source from Github and install locally, run:
```
git clone https://github.com/sebftw/Eiko.git
cd Eiko
pip install -e .
```
This includes the examples code, MATLAB support, and more.
If you intend on running the example code, you should also run the following to get the required dependencies
```
pip install -e ".[examples]"
```

## Installing from Scratch
If you do not have a compiler or CUDA yet, this guide is for you!
### Linux
#### CUDA
First, update your display drivers to ensure compatibility with the latest version of CUDA.
```
sudo ubuntu-drivers install
```
After installing the new driver, you want to restart your system (`sudo reboot`).
You can verify that the driver was installed by runnning `nvidia-smi`.


Then, navigate to [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) to download CUDA.
Run `uname -mr` and `cat /etc/os-release` to find out which version you should download. Select `deb (network)`, and follow the provided instructions.

To verify that CUDA was installed, run `nvcc --version`.

#### Compiler
Next, you want a C++ compiler. Open the terminal and run
```
sudo apt update && sudo apt upgrade -y
sudo apt install build-essential -y
```
You can verify that the compiler is installed sucessfully by running `gcc --version`.

#### Python
Finally, you need to install Python with Eiko.
```
sudo apt install python3-full python3-pip python3-dev -y
```
This installs Python along with `venv` (virtual environment) and more, which is highly recommended to avoid clashing package installations.

Close your terminal and open a new one to ensure all system paths are updated.
Then, create and activate a new environment for Eiko:
```
python3 -m venv eiko
source eiko/bin/activate
```
After activation, your command line prompt should begin with `(eiko)`.

Now you can install PyTorch.

Now you can install Pytorch. Navigate to [PyTorch Get Started](https://pytorch.org/get-started/) and select **Stable**, **Linux**, **Pip**, and **Python**. For **Compute Platform**, select the CUDA version that matches the output of `nvcc --version`.
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

#### Python

Download Python from [Python.org](https://www.python.org/downloads/windows/) to download and install Python. Here is our recommended installation:
* Check the box that says "**Add python.exe to PATH**".
* Select "Customize Installation" and make sure that **pip** is selected.
* When you get the option (next page), check "**Install Python for all users**" to install Python in "C:\Program Files\". Also select **Precompile standard library**, **Download debugging symbols**, and **Download debug binaries**.

Next, you want to install Eiko inside a virtual environment.
Press the Windows key and search for **x64 Native Tools Command Prompt for VS 2022**. You *must* use this specific command prompt, not standard PowerShell, as it loads the necessary C++ compiler paths.

To verify the Python installation, run the following
```
python --version
pip --version
```

Next, set up and activate the virtual environment inside `C:\Users\YOURUSERNAME\eiko` by running:
```
cd %USERPROFILE%
python -m venv eiko
%USERPROFILE%\eiko\Scripts\activate
```
Your command line should now start with `(eiko)`.

You want to install Pytorch within this environment. Navigate to [PyTorch Get Started](https://pytorch.org/get-started/), select **Stable**, **Windows**, **Pip**, and **Python**.
For the **Compute Platform**, select the CUDA version that matches the output of `nvcc --version`.

After getting PyTorch, you can install and verify Eiko with
```
pip install eiko
python -c "import eiko.eiko_torch"
```
You are now ready to use Eiko!

**Beware**: Every time you open a new command prompt, you must run `%USERPROFILE%\eiko\Scripts\activate` to reactivate the eiko environment.

## Zero-Installation Setups
Since the installation process is quite involved, this section provides a an alternative way to running Eiko.

### Google Colab (no GPU required)
This option is ideal if you do not own a NVIDIA GPU.
Google Colab provides free GPU access, so you can run Eiko there through the following notebook: [Eiko in Colab](https://colab.research.google.com/github/sebftw/Eiko/blob/main/examples/python/eiko_in_colab.ipynb)







