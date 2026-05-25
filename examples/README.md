# Examples
This directory contains a suite of examples demonstrating the capabilities of Eiko, the eikonal equation solver. They cover a range of applications, from simple visualization and validation to advanced tasks such as acoustic lens design and differentiable travel-time tomography.

| | | |
| :---: | :---: | :---: |
| <img alt="plane_wave1" src="https://github.com/user-attachments/assets/989bce3d-b3d7-4b4b-a735-e58d7e58524c" width="100%" /> | <img alt="aberrated_plane_wave" src="https://github.com/user-attachments/assets/0b0bec62-b80b-47a8-acee-268b71fab730" width="100%" /> | <img alt="plane_wave3" src="https://github.com/user-attachments/assets/796f7cc5-4363-48d6-964d-76cb251377f9" width="100%" /> |
| <img alt="plane_wave3D" src="https://github.com/user-attachments/assets/e94ac0cd-5e3e-4e90-a5d9-80d783a8ca00" width="100%" /> | <img alt="plane_wave3D_aberrated" src="https://github.com/user-attachments/assets/37a75203-b599-4ec6-9598-0dc62346692b" width="100%" /> | <img alt="aberation_correction" src="https://github.com/user-attachments/assets/e92c6f26-8bf7-4a38-ba0d-7978ad017d34" width="100%" /> |
| <img alt="maze" src="https://github.com/user-attachments/assets/a2b65417-1edc-4a10-9cef-3bac17b2e106" width="100%" /> | <img alt="tomography_inversion" src="https://github.com/user-attachments/assets/f9b2edce-409c-4fc4-97c1-3f2f804fd172" width="100%" /> | <img alt="lens_design2_hd" src="https://github.com/user-attachments/assets/c0dd15a9-eb56-4a8b-b27c-02332e7dbf0d" width="100%" /> |

## Prerequisites
To run the Python examples, you will need additional visualization libraries such as `matplotlib`. You can install Eiko with these dependencies (or add them to your current installation) by running:
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

