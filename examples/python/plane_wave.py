# To run this script you must install eiko with the optional example dependencies.
# This is done using "pip install eiko[examples]".

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

# Helper function.
import scipy.ndimage as ndimage
def generate_island(width=200, height=200, threshold=0.3):
    """Generates an organic 'island' aberration mask."""
    # 1. Random noise
    raw_noise = np.random.rand(height, width)
    
    # 2. Gaussian smoothing (fspecial 'gaussian' equivalent)
    sigma = 8
    # gaussian_filter performs the smoothing
    smooth_noise = ndimage.gaussian_filter(raw_noise, sigma=sigma, mode='reflect')
    smooth_noise = (smooth_noise - smooth_noise.min()) / (smooth_noise.max() - smooth_noise.min())
    
    # 3. Geographic falloff (meshgrid equivalent)
    # Y, X = np.meshgrid(np.arange(width), np.arange(height))
    Y, X = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    center_dist = np.sqrt((X - width/2)**2 + (Y - height/2)**2)
    falloff_map = np.maximum(0, 1 - (center_dist / (min(width, height) * 0.6)))
    
    # 4. Mask and binary morphology
    island_mask = (smooth_noise * falloff_map) > threshold
    
    # imfill(island_mask, 'holes') equivalent
    island_mask = ndimage.binary_fill_holes(island_mask)
    
    # 5. Allow small islands (bwareaopen equivalent)
    min_island_size = int(80 / (200 * 200) * width * height)
    # Label connected components and filter by size
    labeled_array, num_features = ndimage.label(island_mask)
    sizes = ndimage.sum(island_mask, labeled_array, range(1, num_features + 1))
    # Create mask of components to keep
    keep_mask = sizes >= min_island_size
    # Map those back to the labeled array
    island_mask = np.isin(labeled_array, np.where(keep_mask)[0] + 1)
    
    # 6. Final Gaussian filter (imgaussfilt equivalent)
    island_mask = ndimage.gaussian_filter(island_mask.astype(float), sigma=2.0)
    
    return island_mask

# =========================================================================
# Setup
# =========================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
canvas_height = 1080 // 4
canvas_width = 1920 // 4

# Aperture settings
aperture_y = 1
aperture_x = torch.arange(int(canvas_width * 0.2), int(canvas_width * 0.8), device=device)
aperture_width = len(aperture_x)

# Steering parameters
steering_angle_deg = 15
theta = np.deg2rad(steering_angle_deg)

# Solver settings
msfm = True
gated_x = 0

# Apodization (v_init)
v_init = torch.zeros((canvas_height, canvas_width), dtype=torch.float32, device=device)
tw = torch.tensor(tukey(aperture_width + 2, alpha=0.5)[1:-1], dtype=torch.float32, device=device)
v_init[aperture_y, aperture_x] = tw

# =========================================================================
# 1. Homogeneous Medium
# =========================================================================
print("Simulating homogeneous medium...")
f_homo = torch.ones((canvas_height, canvas_width), dtype=torch.float32, device=device)

# Delays
delays_ideal = aperture_x.float() * np.sin(theta)
delays_ideal -= delays_ideal.min()

u_init = torch.full((canvas_height, canvas_width), float('inf'), device=device)
u_init[aperture_y, aperture_x] = delays_ideal

u_homo, v_homo = eiko(u_init, f_homo, v_init=v_init, msfm=msfm, gated=gated_x)
animate_eikonal(u_homo, v_homo, style='real', title=f'Plane wave ({steering_angle_deg}° Steered)')

# =========================================================================
# 2. Inhomogeneous Medium (Aberration)
# =========================================================================
print("Simulating inhomogeneous medium...")
# Create island mask (simulating tissue)
island_np = generate_island(canvas_width, canvas_height, 0.25)
islands = torch.tensor(island_np, dtype=torch.float32, device=device)
f_inhomo = torch.ones_like(f_homo) - (islands * 0.2)

u_uncorrected, v_uncorrected = eiko(u_init, f_inhomo, v_init=v_init, msfm=msfm, gated=gated_x)
animate_eikonal(u_uncorrected, v_uncorrected, outline=islands, style='real', 
                title=f'Aberrated Plane Wave ({steering_angle_deg}° Steered)')


# =========================================================================
# 3. Time-Reversal Correction
# =========================================================================
print("Performing time-reversal correction...")

# Back-propagation: plane wave from bottom
u_init_bw = torch.full((canvas_height, canvas_width), float('inf'), device=device)
u_init_bw[-1, :] = -torch.arange(canvas_width, device=device).float() * np.sin(theta)

u_bw = eiko(torch.flip(u_init_bw, [0]), torch.flip(f_inhomo, [0]), 
               msfm=msfm, gated=gated_x)
u_bw = torch.flip(u_bw, [0])

# Extract and reverse arrival times
arrival_times = u_bw[aperture_y, aperture_x]
delays_corrected = arrival_times.max() - arrival_times

# Run forward pass with corrected delays
u_init_corrected = torch.full((canvas_height, canvas_width), float('inf'), device=device)
u_init_corrected[aperture_y, aperture_x] = delays_corrected

u_corrected, v_corrected = eiko(u_init_corrected, f_inhomo, v_init=v_init, 
                                msfm=msfm, gated=gated_x)

animate_eikonal(u_corrected, v_corrected, outline=islands, style='real', 
                title=f'Aberration Corrected Plane Wave ({steering_angle_deg}° Steered)')