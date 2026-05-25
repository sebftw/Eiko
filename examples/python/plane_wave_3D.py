# To run this script you must install eiko with the optional example dependencies.
# This is done using "pip install eiko[examples]".

import torch
import sys
import torch
import torch.nn.functional as F

# Check if the optional dependencies are available
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Polygon
except ImportError:
    print("\n[Eiko] Error: This script requires 'matplotlib' for visualization.", file=sys.stderr)
    sys.exit(1)

try:
    from scipy.signal.windows import tukey
except ImportError:
    print("\n[Eiko] Error: This script requires 'scipy' to generate the Tukey window.", file=sys.stderr)
    sys.exit(1)

try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("[Eiko] Warning: 'scikit-image' not found. Contours and overlays will be disabled.", file=sys.stderr)

import numpy as np
from scipy.signal.windows import tukey
from eiko import eiko, animate_eikonal

# =========================================================================
# 1. Setup 3D Domain and Transducer Aperture
# =========================================================================
canvas_height = int(1080 / 4)
canvas_width  = int(1920 / 4)
canvas_depth  = canvas_width
msfm = False 
steering_angle_deg = 20
theta = np.deg2rad(steering_angle_deg)

device = 'cuda'

aperture_y = 0
aperture_x = slice(int(canvas_width * 0.2), int(canvas_width * 0.8))
aperture_z = slice(int(canvas_depth * 0.2), int(canvas_depth * 0.8))

# Apodization windows
tw_x = tukey(len(range(*aperture_x.indices(canvas_width))))
tw_z = tukey(len(range(*aperture_z.indices(canvas_depth))))
apod_2d = torch.tensor(np.outer(tw_z, tw_x), device=device, dtype=torch.float32)

# Firing delays
delays_x = torch.arange(canvas_width, device=device).float() * np.sin(theta)
delays_x -= delays_x.min()

u_init = torch.full((canvas_height, canvas_width, canvas_depth), float('inf'), device=device)
v_init = torch.zeros((canvas_height, canvas_width, canvas_depth), device=device)

# Assign to aperture
u_init[aperture_y, aperture_x, aperture_z] = delays_x[aperture_x]
v_init[aperture_y, aperture_x, aperture_z] = apod_2d.T 

# =========================================================================
# 2. 3D Plane Wave in a Homogeneous Medium
# =========================================================================
f_homo = torch.ones((canvas_height, canvas_width, canvas_depth), device=device)
u_homo, v_homo = eiko(u_init, f_homo, v_init=v_init, msfm=msfm)
animate_eikonal(u_homo, v_homo, render_mode='slice', title=f'Plane Wave ({steering_angle_deg}° Steered)')


print(f"U min/max: {u_homo.min():.2f} / {u_homo.max():.2f}")
print(f"V min/max: {v_homo.min():.2f} / {v_homo.max():.2f}")

# Check if U is still mostly Inf
print(f"Fraction of valid U: {torch.isfinite(u_homo).float().mean():.2%}")

# =========================================================================
# 3. 3D Plane Wave in an Inhomogeneous Medium (Aberrated)
# =========================================================================
z_grid, y_grid, x_grid = torch.meshgrid(
    torch.arange(canvas_depth, device=device), 
    torch.arange(canvas_height, device=device), 
    torch.arange(canvas_width, device=device), indexing='ij'
)

center_x, center_y, center_z = canvas_width/2, canvas_height*0.35, canvas_depth/2
radius = canvas_width * 0.12
sphere_mask = ((x_grid - center_x)**2 + (y_grid - center_y)**2 + (z_grid - center_z)**2) <= radius**2

f_inhomo = torch.ones_like(x_grid, dtype=torch.float32)
f_inhomo[sphere_mask] = 0.8

u_uncorrected, v_uncorrected = eiko(u_init, f_inhomo, v_init=v_init, msfm=msfm)
animate_eikonal(u_uncorrected, v_uncorrected, render_mode='slice', outline=sphere_mask, 
                title=f'Aberrated Plane Wave ({steering_angle_deg}° Steered)')

# =========================================================================
# 4. 3D Aberration Correction via Time-Reversal
# =========================================================================
# 4a. Back-propagation
u_init_bw = torch.full((canvas_height, canvas_width, canvas_depth), float('inf'), device=device)
u_init_bw[-1, :, :] = torch.arange(canvas_width, device=device).float() * np.sin(theta)

u_bw = eiko(torch.flip(u_init_bw, [0]), torch.flip(f_inhomo, [0]), msfm=msfm)
u_bw = torch.flip(u_bw, [0])

# 4b. Extract delays
arrival_times = u_bw[aperture_y, aperture_x, aperture_z]
delays_corrected = arrival_times.max() - arrival_times

# 4c. Re-initialize
u_init_corrected = torch.full((canvas_height, canvas_width, canvas_depth), float('inf'), device=device)
u_init_corrected[aperture_y, aperture_x, aperture_z] = delays_corrected

# 4d. Forward pass
u_corrected, v_corrected = eiko(u_init_corrected, f_inhomo, v_init=v_init, msfm=msfm)

animate_eikonal(u_corrected, v_corrected, render_mode='slice', outline=sphere_mask, 
                title=f'3D Aberration Corrected ({steering_angle_deg}° Steered)')