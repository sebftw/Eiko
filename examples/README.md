# Examples
This directory contains a suite of examples demonstrating the capabilities of Eiko, the eikonal equation solver. They cover various applications, ranging from simple visualization and validation to advanced tasks like acoustic lens design and differentiable travel-time tomography.

## Prerequisites
To run the Python examples, you will need additional visualization libraries such as matplotlib. You can install Eiko with these dependencies (or add them to your current installation) by running:
```
pip install eiko[examples]
```

## Example Scripts
The scripts are located in the following subfolders.
* `matlab/`: MATLAB **.m** example scripts.
* `python/`: Python **.py** example scripts.

A brief description of each script:

| Script | Description |
| :--- | :--- |
| [`time_of_flight.m`](matlab/time_of_flight.m)<br>[`time_of_flight.py`](python/time_of_flight.py) | Accuracy testing against an analytical solution. |
| [`interactive_eiko.m`](matlab/interactive_eiko.m)<br>[`interactive_eiko.py`](python/interactive_eiko.py) | Real-time mouse-controlled point source solver. |
| [`logo.m`](matlab/logo.m)<br>[`logo.py`](python/logo.py) | Animated Eiko logo generation. |
| [`plane_wave.m`](matlab/plane_wave.m)<br>[`plane_wave.py`](python/plane_wave.py) | Steered plane wave with time-reversal aberration correction. |
| [`plane_wave_3D.m`](matlab/plane_wave_3D.m)<br>[`plane_wave_3D.py`](python/plane_wave_3D.py) | Volumetric (3D) steered plane waves with aberration correction. |
| [`aberration_correction.m`](matlab/aberration_correction.m)<br>[`aberration_correction.py`](python/aberration_correction.py) | Spherical wavefront aberration correction using time-reversal. |
| [`lens_design.m`](matlab/lens_design.m)<br>[`lens_design.py`](python/lens_design.py) | Plano-convex acoustic lens design (transforms a plane wave into a spherical one). |
| [`tomography.m`](matlab/tomography.m)<br>[`tomography_torch.py`](python/tomography_torch.py)<br>[`tomography_jax.py`](python/tomography_jax.py) | Differentiable optimization for sound speed map recovery.  Includes an example using PyTorch, JAX, and the MATLAB Deep Learning Toolbox. |


## How to Run
**MATLAB**: Navigate to the [`matlab/`](matlab/) directory and run the script directly.<br>
**Python**: Navigate to the [`python/`](python/) directory and execute the script directly. For example:
```
python time_of_flight.py
```

