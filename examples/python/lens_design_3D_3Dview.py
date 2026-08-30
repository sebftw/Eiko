# To run this script you must install eiko with the optional example dependencies.
# This is done using "pip install eiko[examples]".

import sys
import torch
import numpy as np

# Check if the optional dependencies are available
try:
    import matplotlib
    matplotlib.use('QtAgg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
except ImportError:
    print("\n[Eiko] Error: This script requires 'matplotlib' for visualization.", file=sys.stderr)
    sys.exit(1)

try:
    from scipy.signal.windows import tukey
except ImportError:
    print("\n[Eiko] Error: This script requires 'scipy' to generate the Tukey window.", file=sys.stderr)
    sys.exit(1)

try:
    from eiko import eiko3d, animate_eikonal
except ImportError:
    print("\n[Eiko] Error: This script requires 'eiko' and its visualization dependencies.", file=sys.stderr)
    sys.exit(1)

# =========================================================================
# CONFIGURATION
# =========================================================================
# Toggle apodization window along the lateral (Y) axis (True = Tukey, False = Rectangular / all ones)
apod_tukey_lateral = True

# =========================================================================
# 1. Lens & Setup Parameters
# =========================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

c1 = 3000.0  
c2 = 1500.0  
f  = 0.031   
d  = 0.04    
D  = 0.08    
elevation_steering_angle_deg = 15.0

# Isotropic 0.5mm grid
dx_phys = 0.5e-3 
x_domain = 0.1
y_domain = 0.12
z_domain = 0.12

Nx = int(round(x_domain / dx_phys))
Ny = int(round(y_domain / dx_phys))
Nz = int(round(z_domain / dx_phys))

x_vec = np.arange(Nx) * dx_phys - 0.02 
y_vec = np.arange(Ny) * dx_phys - y_domain / 2
z_vec = np.arange(Nz) * dx_phys - z_domain / 2
X, Y, Z = np.meshgrid(x_vec, y_vec, z_vec, indexing='ij')

# Physical coordinate extents in millimeters [xmin, xmax, ymin, ymax, zmin, zmax]
extent_mm = [x_vec[0]*1e3, x_vec[-1]*1e3, y_vec[0]*1e3, y_vec[-1]*1e3, z_vec[0]*1e3, z_vec[-1]*1e3]

# =========================================================================
# 2. Mathematical Surface Evaluation (Cylindrical Extrusion)
# =========================================================================
n_ratio = c2 / c1            
L = d * (1 - n_ratio) + f     
A = 1 - n_ratio**2
B = 2 * (f - n_ratio * L)
C0 = f**2 - L**2        

h_grid = np.zeros((Nx, Ny, Nz))
valid_Y = np.abs(Y) <= D / 2

C_valid = Y[valid_Y]**2 + C0
Delta = B**2 - 4 * A * C_valid
h_grid[valid_Y] = (-B + np.sqrt(Delta)) / (2 * A)

# Lens extends along elevation (Z) to allow steered beams to exit cleanly
lens_mask = valid_Y & (X >= 0) & (X <= h_grid)

# =========================================================================
# 3. Discretize for animate_eikonal (Relative Slowness)
# =========================================================================
# animate_eikonal operates best when background slowness is 1.0 (1 pixel / unit time)
# The lens slowness is scaled relative to the background
f_map = np.ones((Nx, Ny, Nz), dtype=np.float32)
f_map[lens_mask] = c2 / c1

source_x = 0.0
source_idx_x = np.where(x_vec <= source_x / 1000.0)[0][-1] - 1
start_x_idx = np.argmin(np.abs(x_vec - 0.0))

valid_y_mask = np.abs(y_vec) <= D / 2
valid_z_mask = np.abs(z_vec) <= D / 2
valid_aperture = np.outer(valid_y_mask, valid_z_mask)

y_in = y_vec[valid_y_mask]
z_in = z_vec[valid_z_mask]

# Calculate target phase delays to steer the flat elevation axis (Z)
theta_target = np.radians(elevation_steering_angle_deg)
theta_lens = np.arcsin((c1 / c2) * np.sin(theta_target))

# Convert physical arrival times into grid-pixel units for animate_eikonal
z_delays_phys = z_in * np.sin(theta_lens) / c1
z_delays_pixels = z_delays_phys * (c2 / dx_phys)
z_delays_pixels -= np.min(z_delays_pixels)
z_delays_2d = np.tile(z_delays_pixels, (len(y_in), 1))

u_init = np.full((Nx, Ny, Nz), np.inf, dtype=np.float32)
u_init_slice = np.full((Ny, Nz), np.inf, dtype=np.float32)
u_init_slice[np.ix_(valid_y_mask, valid_z_mask)] = z_delays_2d
u_init[source_idx_x, :, :] = u_init_slice

# Apply lateral baffles (prevent backward and off-aperture expansion)
f_map[:source_idx_x, :, :] = np.inf
f_map[source_idx_x:start_x_idx, ~valid_aperture] = np.inf

# =========================================================================
# 4. Apodization window
# =========================================================================
# Generate 2D separable apodization with toggle for lateral (Y) axis
if apod_tukey_lateral:
    tw_y = tukey(len(y_in), alpha=0.15)
else:
    tw_y = np.ones(len(y_in))

tw_z = tukey(len(z_in), alpha=0.15)
tw_2d = np.outer(tw_y, tw_z)

v_lens_2d = np.zeros((Ny, Nz), dtype=np.float32)
v_lens_2d[np.ix_(valid_y_mask, valid_z_mask)] = tw_2d

v_init = np.zeros_like(f_map)
v_init[source_idx_x, :, :] = v_lens_2d
v_init[source_idx_x + 1, :, :] = v_lens_2d

# =========================================================================
# 5. Compute & Animate 3D Field
# =========================================================================
u_init_t = torch.tensor(u_init, device=device)
f_map_t = torch.tensor(f_map, device=device)
v_init_t = torch.tensor(v_init, device=device)
lens_mask_t = torch.tensor(lens_mask, device=device)

print(f"Computing 3D steered plane wave through cylindrical lens... (Elevation {elevation_steering_angle_deg}°)")
with torch.no_grad():
    u_lensed, v_lensed = eiko3d(u_init_t, f_map_t, v_init=v_init_t, msfm=True)

print("Rendering 3D volume animation...")
animate_eikonal(
    u_lensed, 
    v_lensed, 
    render_mode='slice', 
    outline=lens_mask_t, 
    extent=extent_mm,
    title=f'Lensed Cylindrical Wave ({elevation_steering_angle_deg}° Elevation Steered)'
)