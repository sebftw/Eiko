# Eiko
**Eiko** is a GPU-accelerated Eikonal equation solver, enabling fast
computation of the shortest time-of-flight through an arbitrary 2D or 3D medium.

A few reasons to use Eiko:
1. Eiko is **fast** - up to 10x faster than comparable libraries.
2. Eiko is **differentiable**, allowing it to be used with PyTorch or JAX.
3. Eiko supports **advection**, allowing it to compute apodizations through a lens.

![Image of example](/examples/python/comparison/fps_comparison.png "Comparison between Eiko and other solvers.")

## Installation
### Requirements
Because Eiko compiles custom CUDA kernels during installation, your system must have the following:
* **OS:** Linux or Windows
* **Hardware:** NVIDIA GPU
* **Compiler:** 
  * A C++ compiler (e.g., GCC for Linux, MSVC for Windows)
  * The CUDA Toolkit (provides `nvcc`)

### Installing for MATLAB
Run `setup.m` to install Eiko for MATLAB.

### Installing for Python
Run
```
pip install eiko
```
You now have Eiko installed and ready for use.

See also [the Eiko Python installation guide](/python/PYTHON_INSTALLATION.md).

## Quick Start
An example of how to use Eiko is shown below.
<br>
See [EXAMPLES](/examples/README.md) for many more examples.

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
u_numerical = eiko(u_init, f, dx);

% 5. Visualize the result.
% Create physical coordinate axes in millimeters using dx
axis_mm = ((1:N) - center_idx) * dx * 1000;

% Set up plot.
figure;
imagesc(axis_mm, axis_mm, u_numerical * 1e6);
axis image;

% Format axes and text size.
title('Time-of-Flight Field', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('x (mm)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('y (mm)', 'FontSize', 12, 'FontWeight', 'bold');

% Format colorbar.
cb = colorbar;
cb.Label.String = 'Time (\mus)';
cb.Label.FontSize = 12;
```
For 3D inputs, call `eiko3d`.

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
u_numerical = eiko(u_init, f, dx=dx)
```
For 3D inputs, use `from eiko import eiko3d`.


The result should look something like this:

![Image of example](/images/eiko_example.png "Result of the example code.")

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
├── pyproject.toml  # Python package configuration (for pip install)
├── setup.m         # MATLAB compilation and setup script
├── THEORY.md       # Mathematical background and Eikonal algorithm details
└── README.md       # This file
```
To learn more about how Eiko works, see [THEORY](THEORY.md).

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