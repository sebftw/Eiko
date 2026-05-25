# To run this script you must install eiko with the optional example dependencies.
# This is done using "pip install eiko[examples]".

import sys

# Check if the optional dependency is available
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("\n[Eiko] Error: This example script requires 'matplotlib' for visualization.", file=sys.stderr)
    print("Please install it by running: pip install \"eiko[examples]\"\n", file=sys.stderr)
    sys.exit(1)

import torch
import numpy as np

# Import Eiko
from eiko import eiko

# =========================================================================
# 1. Setup Domain and Medium Properties
# =========================================================================
# Prefer execution on the GPU to interface with the custom CUDA solver
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N = 101
dx = 0.001          # Grid spacing in meters (1 mm)
c = 1540.0          # Speed of sound in m/s (uniform medium)
msfm = True;        # Whether to use multi-stencil fast marching for improved accuracy.

# Create slowness map (1/c) everywhere natively on the device
f = torch.full((N, N), 1.0 / c, dtype=torch.float32, device=device)

# Create spatial coordinate grids centered at 0
offset = (N // 2) * dx
x_coords = torch.arange(N, dtype=torch.float32, device=device) * dx - offset
y_coords = torch.arange(N, dtype=torch.float32, device=device) * dx - offset

# PyTorch meshgrid requires 'ij' indexing to match MATLAB's default behavior
Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')

# =========================================================================
# 2. Calculate the Analytical Solution
# =========================================================================
# The closed-form time-of-flight is distance / speed
R = torch.sqrt(X**2 + Y**2)
u_analytical = R / c

# =========================================================================
# 3. Compute Numerical Solution using 'eiko'
# =========================================================================
# Initialize u_init with infinity at unknown points
u_init = torch.full((N, N), float('inf'), dtype=torch.float32, device=device)

# Set the point source at the center of the grid to time = 0
center_idx = N // 2
u_init[center_idx, center_idx] = 0.0

# Call the eiko solver.
print(f"Running EIKO solver on {device}...")
with torch.no_grad():
    u_numerical = eiko(u_init, f, dx=dx, msfm=msfm)

# =========================================================================
# 4. Calculate Error and Check Tolerance
# =========================================================================
# Compare numerical and analytical fields
error_map = torch.abs(u_numerical - u_analytical)
max_error = torch.max(error_map).item()

# Define an acceptable tolerance. 
# We allow a maximum error roughly equivalent to traveling 1.5 grid cells.
tolerance = 1.5 * (dx / c)

# =========================================================================
# 5. Visualization
# =========================================================================
print('Rendering visualization...')
# Bring tensors back to CPU for matplotlib
u_num_cpu = u_numerical.cpu().numpy()
u_ana_cpu = u_analytical.cpu().numpy()
err_cpu = error_map.cpu().numpy()

# Set extent for real-world axis scales
extent = [x_coords[0].item()*1000, x_coords[-1].item()*1000, y_coords[0].item()*1000, y_coords[-1].item()*1000]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')
fig.suptitle('EIKO Solver Validation - Constant Speed of Sound', fontsize=14, fontweight='bold')

# Plot Numerical
im0 = axes[0].imshow(u_num_cpu, extent=extent, cmap='viridis')
axes[0].set_title('Numerical (eiko)')
axes[0].set_xlabel('x (mm)')
axes[0].set_ylabel('y (mm)')
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# Plot Analytical
im1 = axes[1].imshow(u_ana_cpu, extent=extent, cmap='viridis')
axes[1].set_title('Analytical')
axes[1].set_xlabel('x (mm)')
axes[1].set_ylabel('y (mm)')
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# Plot Error
im2 = axes[2].imshow(err_cpu, extent=extent, cmap='hot')
axes[2].set_title(f'Abs Error (Max: {max_error:.2e} s)')
axes[2].set_xlabel('x (mm)')
axes[2].set_ylabel('y (mm)')
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.draw()

# =========================================================================
# 6. Assertion
# =========================================================================
if max_error > tolerance:
    # We display the plot before raising the exception so you can still inspect it
    plt.show(block=False) 
    raise AssertionError(
        f"Validation failed! Maximum error ({max_error:.4e} s) exceeds the tolerance ({tolerance:.4e} s)."
    )
else:
    print(f"EIKO validation passed! Maximum error is {max_error:.4e} s (Tolerance: {tolerance:.4e} s).")
    plt.show()