![Eiko Logo](https://raw.githubusercontent.com/sebftw/Eiko/main/images/eiko_logo.png "Eiko")<br>
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/eiko)
[![PyPI version](https://img.shields.io/pypi/v/eiko)](https://pypi.org/project/eiko/)
![GitHub License](https://img.shields.io/github/license/sebftw/Eiko)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebftw/Eiko/blob/main/examples/python/eiko_in_colab.ipynb)
<!-- [![GitHub Downloads](https://img.shields.io/github/downloads/sebftw/Eiko/total?color=blue&label=GitHub%20Downloads&style=flat-square&logo=github)](https://github.com/sebftw/Eiko/releases) -->

**Eiko** is a GPU-accelerated Eikonal equation solver, enabling fast
computation of the shortest time-of-flight through an arbitrary 2D or 3D medium.

<!-- START_MATLAB_ONLY -->
- [Why use Eiko?](#why-use-eiko)
- [Installation](#installation)
  * [Requirements](#requirements)
  * [Installing for MATLAB](#installing-for-matlab)
  * [Installing for Python](#installing-for-python)
- [Quick Start](#quick-start)
  * [MATLAB example](#matlab-example)
  * [Python example](#python-example)
  * [Visual examples](#visual-examples)
- [Project Layout](#project-layout)
- [Contributing](#contributing)
- [Citing](#citing)
<!-- END_MATLAB_ONLY -->

Eiko takes a sound-speed map and initial time-of-flight values at a few points and returns the full time-of-flight map.
```
time_of_flight = eiko(initial_delays, slowness_map)
```
For example, if the initial delays describe a plane-wave:

<p align="center">
<img width="480" height="410" alt="Eikonal Plane Wave Aberration Animation" src="https://github.com/user-attachments/assets/0944ba3f-ffc9-47e7-8a42-ef26cf41e6d2" style="background: transparent !important; border: none !important; box-shadow: none !important;" />
</p>

## Why use Eiko?
A few reasons to use Eiko:
1. Eiko is **fast** - up to 100x faster than comparable libraries.
2. Eiko is **differentiable**, allowing it to be used with PyTorch or JAX.
3. Eiko supports **advection**, allowing it to compute apodizations through a lens.

Eiko also supports batch processing, allowing many time-of-flight maps to be computed efficiently in parallel.
![Performance Comparison](https://raw.githubusercontent.com/sebftw/Eiko/main/examples/python/comparison/fps_comparison.png "Comparison between Eiko and other solvers.")


## Installation
This section describes how to install Eiko for MATLAB or Python.

### Requirements
Because Eiko compiles custom CUDA kernels during installation, your system must have the following:
* **OS:** Windows or Linux
* **Hardware:** A NVIDIA GPU
* **Compiler:** 
  * A C++ compiler (e.g., MSVC for Windows, GCC for Linux)
  * The CUDA Toolkit (provides `nvcc`)

If you don't own a GPU, you can run Eiko for Python from Google Colab with zero installation required [here](https://colab.research.google.com/github/sebftw/Eiko/blob/main/examples/python/eiko_in_colab.ipynb).

<!-- START_MATLAB_ONLY -->
### Installing for MATLAB
Run `setup.m` to install Eiko in MATLAB.

See also [the Eiko MATLAB installation guide](/matlab/MATLAB_INSTALLATION.md).
<!-- END_MATLAB_ONLY -->

### Installing for Python
Run the following command to install Eiko for Python.
```
pip install eiko
```

<!-- START_MATLAB_ONLY -->
See also [the Eiko Python installation guide](/python/PYTHON_INSTALLATION.md).
<!-- END_MATLAB_ONLY -->

## Quick Start
An example of how to use Eiko is shown below.
<br>
See [EXAMPLES](/examples/README.md) for many more examples.

<!-- START_MATLAB_ONLY -->
### MATLAB example
```matlab
% 1. Setup grid parameters.
N = 101;            % Number of grid points (NxN grid)
dx = 0.001;         % Grid spacing in meters (1 mm, for example)
c = 1540;           % Speed of sound in m/s (uniform medium)

% 2. Create the slowness map (1/velocity).
f = ones(N, N, 'single', 'gpuArray') / c;

% 3. Initialize the time-of-flight field.
u_init = inf(N, N, 'single', 'gpuArray'); % Unknown points set to infinity.

% Set a point source at the center of the grid to time = 0.
center_idx = ceil(N/2);
u_init(center_idx, center_idx) = 0;

% 4. Compute the numerical solution using eiko.
u = eiko(u_init, f, dx);

% 5. Visualize the result.
% Create physical coordinate axes in millimeters using dx
axis_mm = ((1:N) - center_idx) * dx * 1000;

% Set up plot.
figure;
imagesc(axis_mm, axis_mm, u * 1e6);
axis image;

% Format axes and text size.
title('Time-of-Flight', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('x (mm)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('y (mm)', 'FontSize', 12, 'FontWeight', 'bold');

% Format colorbar.
cb = colorbar;
cb.Label.String = 'Time (\mus)';
cb.Label.FontSize = 12;
```
For 3D inputs, use `eiko3d`.
<!-- END_MATLAB_ONLY -->
### Python example

```python
from eiko import eiko

# 1. Setup device and grid parameters.
device = torch.device("cuda")
N = 101
dx = 0.001  # Grid spacing in meters (1 mm)
c = 1540.0  # Speed of sound in m/s (uniform medium)

# 2. Create the slowness map (1/velocity) on the device.
f = torch.full((N, N), 1.0 / c, dtype=torch.float32, device=device)

# 3. Initialize the time-of-flight field.
# Unknown points are set to infinity
u_init = torch.full((N, N), float('inf'), dtype=torch.float32, device=device)

# Set a point source at the center of the grid to time = 0.
center_idx = N // 2
u_init[center_idx, center_idx] = 0.0

# 4. Compute the numerical solution.
u = eiko(u_init, f, dx=dx)

# 5. Visualize the result.
import matplotlib.pyplot as plt
import torch

# Create physical coordinate axes in millimeters using dx
axis_mm = (torch.arange(N) - center_idx) * dx * 1000

# Set up plot (equivalent to 'figure')
plt.figure(figsize=(6, 6))

# 'imagesc' equivalent with physical extents. 
# 'extent' maps the data coordinates to the axes.
extent = [axis_mm[0], axis_mm[-1], axis_mm[0], axis_mm[-1]]
plt.imshow(u.cpu() * 1e6, extent=extent, origin='lower', cmap='viridis')

# 'axis image' equivalent (forces equal pixel aspect ratio)
plt.gca().set_aspect('equal', adjustable='box')

# Format axes and text size
plt.title('Time-of-Flight', fontsize=14, fontweight='bold')
plt.xlabel('x (mm)', fontsize=12, fontweight='bold')
plt.ylabel('y (mm)', fontsize=12, fontweight='bold')

# Format colorbar
cb = plt.colorbar()
cb.set_label(r'Time ($\mu$s)', fontsize=12)

# Display the plot
plt.show()
```
For 3D inputs, use `from eiko import eiko3d`.

The result should look something like this:

![Image of example](https://raw.githubusercontent.com/sebftw/Eiko/main/images/eiko_example.png "Result of the example code.")

### Visual examples
Some visual examples of Eiko in action:
<details>
 <summary>Click to view visual examples</summary>


 
| | | |
| :---: | :---: | :---: |
| <img loading="lazy" alt="plane_wave1" src="https://github.com/user-attachments/assets/989bce3d-b3d7-4b4b-a735-e58d7e58524c" width="100%" /> | <img loading="lazy" alt="aberrated_plane_wave" src="https://github.com/user-attachments/assets/0b0bec62-b80b-47a8-acee-268b71fab730" width="100%" /> | <img loading="lazy" alt="plane_wave3" src="https://github.com/user-attachments/assets/796f7cc5-4363-48d6-964d-76cb251377f9" width="100%" /> |
| <img loading="lazy" alt="plane_wave3D" src="https://github.com/user-attachments/assets/e94ac0cd-5e3e-4e90-a5d9-80d783a8ca00" width="100%" /> | <img loading="lazy" alt="plane_wave3D_aberrated" src="https://github.com/user-attachments/assets/37a75203-b599-4ec6-9598-0dc62346692b" width="100%" /> | <img loading="lazy" alt="aberration_correction" src="https://github.com/user-attachments/assets/e92c6f26-8bf7-4a38-ba0d-7978ad017d34" width="100%" /> |
| <img loading="lazy" alt="maze" src="https://github.com/user-attachments/assets/a2b65417-1edc-4a10-9cef-3bac17b2e106" width="100%" /> | <img loading="lazy" alt="tomography_inversion" src="https://github.com/user-attachments/assets/f9b2edce-409c-4fc4-97c1-3f2f804fd172" width="100%" /> | <img loading="lazy" alt="lens_design2_hd" src="https://github.com/user-attachments/assets/c0dd15a9-eb56-4a8b-b27c-02332e7dbf0d" width="100%" /> |
</details>


## Project Layout
The files are as follows:
```
Eiko/
├── examples/       # Example scripts (tomography, lens design, etc.).
│   ├── matlab/     #   MATLAB example scripts 
│   └── python/     #   Python example scripts
├── matlab/         # MATLAB Eiko library
├── python/         # Python Eiko library
├── src/            # Core CUDA C++ Eiko implementation and interface
├── images/         # Various image assets
├── pyproject.toml  # Python package configuration (for pip install)
├── setup.m         # MATLAB compilation and setup script
├── THEORY.md       # Mathematical background and Eikonal algorithm details
└── README.md       # This file
```
To learn more about how Eiko works, see [THEORY](THEORY.md).

## Contributing
Feel free to contribute to the project. Bug reports and feature requests may be submitted on the "Issues" page.

## Citing
You can cite Eiko as
```
@misc{eiko2026,
  author = {Pr{\ae}sius, Sebastian},
  title = {Eiko: the GPU-accelerated Eikonal equation solver},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/sebftw/Eiko}}
}
```

## Trademarks
This project is an independent open-source software project and is not affiliated with, endorsed by, or sponsored by any of the companies mentioned. 
* **MATLAB** is a registered trademark of The MathWorks, Inc.
* **NVIDIA** and **CUDA** are registered trademarks of NVIDIA Corporation.
* **PyTorch** is a trademark of the Linux Foundation.
* **Ubuntu** is a registered trademark of Canonical Ltd.
* **Windows** is a registered trademark of Microsoft Corporation.

All other trademarks, service marks, and company names are the property of their respective owners.
