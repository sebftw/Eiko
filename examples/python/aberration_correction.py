import sys

# Check if optional dependencies are available
try:
    import matplotlib.pyplot as plt
    import scipy.ndimage as ndimage
    from scipy.signal.windows import tukey
except ImportError:
    print("\n[Eiko] Error: This example script requires 'matplotlib' and 'scipy'.", file=sys.stderr)
    print("Please install them by running: pip install matplotlib scipy\n", file=sys.stderr)
    sys.exit(1)

import torch
import numpy as np
import torch.nn.functional as F

# Import Eiko
from eiko import eiko

# =========================================================================
# Helper Functions
# =========================================================================

def generate_island(width, height, threshold=0.3):
    """Generates an organic 'island' aberration mask using smoothed noise."""
    np.random.seed(42)  # Fixed seed for reproducible shapes
    raw_noise = np.random.rand(height, width)
    
    # Smooth the noise to create organic shapes
    sigma = 8 
    smooth_noise = ndimage.gaussian_filter(raw_noise, sigma=sigma, mode='nearest')
    smooth_noise = (smooth_noise - smooth_noise.min()) / (smooth_noise.max() - smooth_noise.min())
    
    # Geographic falloff to prevent clipping at borders
    Y, X = np.ogrid[:height, :width]
    center_dist = np.sqrt((X - width/2)**2 + (Y - height/2)**2)
    falloff_map = np.maximum(0, 1 - (center_dist / (min(width, height) * 0.6)))
    
    # Mask and fill holes
    island_mask = (smooth_noise * falloff_map) > threshold
    island_mask = ndimage.binary_fill_holes(island_mask)
    
    # Allow small islands (size threshold filter)
    min_size = int(80 / (200 * 200) * width * height)
    labeled, num_features = ndimage.label(island_mask)
    if num_features > 0:
        sizes = ndimage.sum(island_mask, labeled, range(1, num_features + 1))
        mask_sizes = sizes < min_size
        remove_pixel = mask_sizes[labeled - 1]
        island_mask[remove_pixel] = 0
        
    island_mask = ndimage.gaussian_filter(island_mask.astype(float), sigma=2.0)
    return torch.tensor(island_mask, dtype=torch.float32)


def flipud_at_y(tensor, y_center):
    """Flips a 2D tensor vertically around a specific y-index."""
    H, W = tensor.shape
    y_indices = torch.arange(H, device=tensor.device)
    
    # Calculate mirrored coordinates
    mirrored_y = 2 * y_center - y_indices
    valid = (mirrored_y >= 0) & (mirrored_y < H)
    
    out = torch.zeros_like(tensor)
    out[valid, :] = tensor[mirrored_y[valid], :]
    return out

# =========================================================================
# 1. Setup Domain and Medium Properties
# =========================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}...")

canvasHeight = int(np.ceil(1080 / 4))
canvasWidth = int(np.ceil(1920 / 4)) + 1
f = torch.ones((canvasHeight, canvasWidth), dtype=torch.float32, device=device)

# Simulate inhomogeneous medium (islands)
iheight = int(np.ceil(canvasHeight * 0.5))
islands_raw = generate_island(canvasWidth, iheight, 0.25).to(device)

# Pad the islands to sit within the canvas (10px top, remainder bottom)
pad_bottom = canvasHeight - iheight - 10
islands = F.pad(islands_raw, (0, 0, 10, pad_bottom), mode='constant', value=0)
f = f + islands * 0.3

# =========================================================================
# 2. Transducer Aperture & Initial Setup
# =========================================================================
steering_angle_deg = 0.0
theta = np.deg2rad(steering_angle_deg)

aperture_y = 0
apert_start, apert_end = int(canvasWidth * 0.2), int(canvasWidth * 0.8)
aperture_x = torch.arange(apert_start, apert_end, device=device)

# Uncorrected Firing Delays & Amplitudes
delays_init = aperture_x.float() * np.sin(theta)
delays_init = delays_init - delays_init.min()

v_init = torch.zeros((canvasHeight, canvasWidth), dtype=torch.float32, device=device)
tw_len = len(aperture_x)
tw = torch.tensor(tukey(tw_len + 2, alpha=0.5)[1:-1], dtype=torch.float32, device=device)
v_init[aperture_y, aperture_x] = tw

# =========================================================================
# 3. Dynamic Target Depth
# =========================================================================
island_rows = torch.where(islands.amax(dim=1) > 0)[0]

if len(island_rows) == 0:
    target_y = canvasHeight - 20
    print(f"No island detected. Evaluating wavefront at row {target_y}.")
else:
    last_island_row = island_rows.max().item()
    target_y = int(np.ceil(canvasHeight / 2))
    print(f"Island ends at row {last_island_row}. Evaluating wavefront at row {target_y}.")

# =========================================================================
# 4. Focused Aberration Correction (Time Reversal / Backward Pass)
# =========================================================================
focus_x = canvasWidth // 2
focus_y = target_y
print(f"Tracing virtual source from (x={focus_x}, y={focus_y})...")

aperture_width = len(aperture_x)
focal_distance = focus_y - aperture_y
dy = 21

u_virtual = torch.full((canvasHeight, canvasWidth), float('inf'), dtype=torch.float32, device=device)
Y, X = torch.meshgrid(torch.arange(canvasHeight, device=device), 
                      torch.arange(canvasWidth, device=device), indexing='ij')

dist_from_focus = torch.sqrt((X - focus_x)**2 + (Y - focus_y)**2)

# F-number geometry matching
template_width = max(1, round(aperture_width * (dy / focal_distance)))
R = int(np.ceil(np.sqrt(dy**2 + (template_width/2)**2))) + 2

# Analytical initialization
init_mask = dist_from_focus <= R
local_slowness = f[focus_y, focus_x]
u_virtual[init_mask] = dist_from_focus[init_mask] * local_slowness

v_backwards = torch.zeros((canvasHeight, canvasWidth), dtype=torch.float32, device=device)
for current_dy in range(1, dy + 1):
    current_width = max(1, round(aperture_width * (current_dy / focal_distance)))
    virt_x_start = focus_x - current_width // 2
    virt_x_end = virt_x_start + current_width
    
    # Apodization window
    tw_virt = tukey(current_width + 2, alpha=0.5)[1:-1]
    
    for i, vx in enumerate(range(virt_x_start, virt_x_end)):
        if 0 <= vx < canvasWidth:
            v_backwards[focus_y + current_dy, vx] = tw_virt[i]
            v_backwards[focus_y - current_dy, vx] = tw_virt[i]

# Clear TOF outside analytical bounds
u_virtual[focus_y + dy + 1:, :] = float('inf')
u_virtual[:focus_y - dy, :] = float('inf')

print("Propagating backward pass...")
with torch.no_grad():
    u_backwards, v_backwards = eiko(u_virtual, f, v_init=v_backwards, dx=1.0, msfm=True)

arrival_times = u_backwards[aperture_y, aperture_x]
delays_corrected = arrival_times.max() - arrival_times

# =========================================================================
# 5. Forward Pass (Aberration Corrected)
# =========================================================================
u_init_forward = torch.full((canvasHeight, canvasWidth), float('inf'), dtype=torch.float32, device=device)
u_init_forward[aperture_y, aperture_x] = delays_corrected

print("Propagating corrected forward pass...")
with torch.no_grad():
    u_focused, v_focused = eiko(u_init_forward, f, v_init=v_init, dx=1.0, msfm=True)

# =========================================================================
# 6. Stitch Post-Focal Expansion
# =========================================================================
t_focal = u_focused[focus_y, focus_x]
u_after = u_focused.clone()
u_after[focus_y:, :] = u_backwards[focus_y:, :] + t_focal

# Stitch and mirror amplitudes
v_after = torch.maximum(v_backwards, flipud_at_y(v_backwards, focus_y))

# Smooth horizontal amplitudes (PyTorch Conv2D)
kernel = torch.ones((1, 1, 1, 11), dtype=torch.float32, device=device)
v_after_padded = F.pad(v_after.unsqueeze(0).unsqueeze(0), (5, 5, 0, 0), mode='reflect')
v_after_smoothed = F.conv2d(v_after_padded, kernel).squeeze() / 11.0

# =========================================================================
# 7. Visualization & Animation
# =========================================================================
print('Rendering static verification plots...')

u_back_cpu = u_backwards.cpu().numpy()
u_foc_cpu = u_focused.cpu().numpy()
u_after_cpu = u_after.cpu().numpy()
islands_cpu = islands.cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='white')
fig.suptitle('Aberration Correction - Time Reversal via Eikonal', fontsize=14, fontweight='bold')

for ax in axes:
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    ax.invert_yaxis()  # Match standard image/matrix coordinates

# 1. Backward Pass
im0 = axes[0].imshow(u_back_cpu, cmap='viridis')
axes[0].plot(focus_x, focus_y, 'r*', markersize=10)
axes[0].set_title('Backward Pass (Virtual Source)')
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# 2. Forward Pass
im1 = axes[1].imshow(u_foc_cpu, cmap='viridis')
axes[1].plot(focus_x, focus_y, 'r*', markersize=10)
axes[1].contour(islands_cpu > 0, levels=[0.5], colors='k', linewidths=2)
axes[1].set_title('Forward Pass (First Arrival)')
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# 3. Stitched Full Pass
im2 = axes[2].imshow(u_after_cpu, cmap='viridis')
axes[2].plot(focus_x, focus_y, 'r*', markersize=10)
axes[2].contour(islands_cpu > 0, levels=[0.5], colors='k', linewidths=2)
axes[2].set_title('Full Pass (Stitched)')
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.draw()  # Draw the static comparison figure without blocking

# --- TRIGGER THE EIKONAL ANIMATION WORKFLOW ---
print('Launching dynamic wavefront animation...')

# Import the utility function provided in animate_eikonal.py
from eiko import animate_eikonal

# Run the animation matching your exact original MATLAB configuration parameter design
anim = animate_eikonal(
    u=u_after, 
    v=v_after_smoothed, 
    outline=islands, 
    style='real', 
    title='Aberration Correction',
    pulse_width=15.0,  # Scaled down from 80.0 to match the smaller canvas size cleanly
    speed=0.4
)