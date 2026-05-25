# To run this script you must install eiko with the optional example dependencies.
# This is done using "pip install eiko[examples]".

import sys

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

import torch
import numpy as np

# Import the Eiko solver
from eiko import eiko

# =========================================================================
# ACOUSTIC LENS GENERATOR
# =========================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Lens and Medium Parameters
c1 = 3000.0  
c2 = 1500.0  
f  = 0.031   
d  = 0.04    
D  = 0.08    

# 2. Grid Parameters
dx = 0.25e-3 
dy = 0.25e-3 
x_domain = 0.1  
y_domain = 0.12
msfm = False

Nx = int(round(x_domain / dx))
Ny = int(round(y_domain / dy))

x_vec = np.arange(Nx) * dx - 0.02 
y_vec = np.arange(Ny) * dy - y_domain / 2
X, Y = np.meshgrid(x_vec, y_vec, indexing='ij')

# 3. Mathematical Surface Evaluation
n = c2 / c1           
L = d * (1 - n) + f     
A = 1 - n**2
B = 2 * (f - n * L)
C0 = f**2 - L**2        

h_grid = np.zeros((Nx, Ny))
valid_y = np.abs(Y) <= D / 2

C_valid = Y[valid_y]**2 + C0
Delta = B**2 - 4 * A * C_valid
h_grid[valid_y] = (-B + np.sqrt(Delta)) / (2 * A)

# 4. Discretization 
lens_mask = (np.abs(Y) <= D / 2) & (X >= 0) & (X <= h_grid)
sound_speed_map = c2 * np.ones((Nx, Ny))
sound_speed_map[lens_mask] = c1

# 5. Compute Plane Wave (u)
source_x = -10.0 # mm
valid_y_idx = np.abs(y_vec) <= D / 2
source_idx_x = np.where(x_vec <= source_x / 1000.0)[0][-1] - 1

slowness = (1.0 / sound_speed_map).astype(np.float32)
u_init = np.full((Nx, Ny), np.inf, dtype=np.float32)
u_init[source_idx_x, valid_y_idx] = 0.0

slowness[:source_idx_x, valid_y_idx] = np.inf
first_lens_idx = np.where(lens_mask[:, int(Ny/2)])[0][0]
slowness[:first_lens_idx, ~valid_y_idx] = np.inf

# Add apodization (v).
v_init = np.zeros_like(slowness)
num_valid_y = np.sum(valid_y_idx)
tw = tukey(num_valid_y + 2)
v_init[source_idx_x, valid_y_idx] = tw[1:-1]
v_init[source_idx_x + 1, valid_y_idx] = tw[1:-1]
# ^ A tukey window initialized at the source.

print("Computing plane wave passing through lens...")
with torch.no_grad():
    u_t, _ = eiko(torch.tensor(u_init, device=device), 
                  torch.tensor(slowness, device=device), 
                  v_init=torch.tensor(v_init, device=device), 
                  dx=dx, msfm=msfm)
u = u_t.cpu().numpy()

aperture_mask = np.zeros_like(slowness)
aperture_mask[:source_idx_x, valid_y_idx] = 1.0

# 6. Compute Virtual Focus (u_virt)
slowness_virt = (1.0 / c2) * np.ones((Nx, Ny), dtype=np.float32)
u_init_virt = np.full((Nx, Ny), np.inf, dtype=np.float32)

dist_to_vf = np.sqrt((source_x/1000.0 - (-f))**2 + y_vec[valid_y_idx]**2)
time_delays = (dist_to_vf - np.min(dist_to_vf)) / c2

u_init_virt[source_idx_x, valid_y_idx] = time_delays
slowness_virt[:source_idx_x, valid_y_idx] = np.inf

v_init_virt = np.zeros_like(slowness_virt)
v_init_virt[source_idx_x, valid_y_idx] = tw[1:-1]

print("Computing virtual focus synthesis...")
with torch.no_grad():
    u_virt_t, _ = eiko(torch.tensor(u_init_virt, device=device), 
                       torch.tensor(slowness_virt, device=device), 
                       v_init=torch.tensor(v_init_virt, device=device), 
                       dx=dx, msfm=msfm)
u_virt = u_virt_t.cpu().numpy()

aperture_mask_virt = np.zeros_like(slowness_virt)
aperture_mask_virt[:source_idx_x, valid_y_idx] = 1.0

# =========================================================================
# 7. Synchronize the Wavefronts
# =========================================================================
# Track when the physical plane wave hits the front center vertex of the lens (X = 0, Y = 0)
exit_x_idx = np.argmin(np.abs(x_vec - d))
center_y_idx = np.argmin(np.abs(y_vec))

# Extract the absolute arrival time (in seconds) at that exact coordinate.
t_lens_exit = u[exit_x_idx, center_y_idx]
t_virt_exit = u_virt[exit_x_idx, center_y_idx]

# Shift both time fields so that t=0 exactly represents the moment they leave the lens.
u_shifted = u.copy()
u_shifted[np.isfinite(u_shifted)] -= t_lens_exit

u_virt_shifted = u_virt.copy()
u_virt_shifted[np.isfinite(u_virt_shifted)] -= t_virt_exit

# Scale time fields (seconds) to pixel indices for the animation rendering
# (Multiplied by c2 to convert time back to distance, divided by dx for grid units)
u_scaled = u_shifted * c2 / dx
u_virt_scaled = u_virt_shifted * c2 / dx

# =========================================================================
# 8. Render Plots & Synchronized Animation
# =========================================================================
print("Rendering Ray Tracing Plot...")
fig1, ax = plt.subplots(figsize=(10, 6), facecolor='w')
ax.set_title(f'Acoustic Lens: Plano-Convex Ellipse\n($c_{{lens}}$ = {int(c1)} m/s, $c_{{medium}}$ = {int(c2)} m/s)', fontsize=14)
ax.set_xlabel('Propagation Axis x (m)')
ax.set_ylabel('Transverse Axis y (m)')
ax.grid(True)
ax.axis('equal')

y = np.linspace(-D/2, D/2, 500)
h_trace = (-B + np.sqrt(B**2 - 4 * A * (y**2 + C0))) / (2 * A) 
lens_x = np.concatenate(([0], h_trace, [0]))
lens_y = np.concatenate(([y[0]], y, [y[-1]]))
poly = Polygon(np.column_stack([lens_x, lens_y]), facecolor=[0.8, 0.9, 1.0], edgecolor='b', linewidth=1.5, alpha=0.5, label='Acoustic Lens')
ax.add_patch(poly)

ax.plot(-f, 0, 'p', markersize=12, markerfacecolor='r', markeredgecolor='k', label='Virtual Source')

num_rays = 11
for i, yi in enumerate(np.linspace(-D/2 * 0.9, D/2 * 0.9, num_rays)):
    hi = (-B + np.sqrt(B**2 - 4 * A * (yi**2 + C0))) / (2 * A)
    ax.plot([-f, hi], [yi, yi], 'b-', linewidth=1.5, label='Incident/Internal Rays' if i==0 else "") 
    
    phi_n = np.arctan2(-(-2 * yi / (2 * A * hi + B)), 1)
    phi_t = phi_n + np.arcsin((c2 / c1) * np.sin(-phi_n))
    
    x_end, y_end = hi + 0.06 * np.cos(phi_t), yi + 0.06 * np.sin(phi_t)
    ax.plot([hi, x_end], [yi, y_end], 'r-', linewidth=1.5, label='Transmitted Rays' if i==0 else "")
    ax.plot([-f, hi], [0, yi], 'r--', linewidth=1, label='Back-traced Rays' if i==0 else "")

ax.set_xlim([-f - 0.02, d + 0.06 + 0.01])
ax.set_ylim([-D/2 - 0.02, D/2 + 0.02])
ax.legend(loc='best')
plt.tight_layout()
plt.show(block=False)

# --- Side-by-Side Animation Function ---
def animate_side_by_side(u1, u2, mask1_overlay, mask1_outline, mask2_overlay):
    pulse_width, speed, freq = 80.0, 0.5, 6 * np.pi
    u1_T, u2_T = u1.T.astype(np.float32), u2.T.astype(np.float32)
    
    valid_u1 = u1_T[np.isfinite(u1_T)]
    valid_u2 = u2_T[np.isfinite(u2_T)]
    
    # Establish dynamic bounds to account for the negative shifted times
    min_t = min(np.min(valid_u1), np.min(valid_u2))
    max_t = max(np.max(valid_u1), np.max(valid_u2))
    
    u1_safe, u2_safe = np.copy(u1_T), np.copy(u2_T)
    u1_safe[np.isinf(u1_safe)] = max_t + pulse_width * 2
    u2_safe[np.isinf(u2_safe)] = max_t + pulse_width * 2

    fig2, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor='white')
    fig2.suptitle('Lens Design', fontsize=16, fontweight='bold')
    
    # Setup Left Panel
    axes[0].set_title('Plane Wave into Spherical', fontsize=14)
    axes[0].set_xlabel('X (Lateral)'); axes[0].set_ylabel('Z (Depth)')
    im1 = axes[0].imshow(np.zeros_like(u1_T), vmin=-1, vmax=1, cmap='gray', aspect='equal')
    
    # Setup Right Panel
    axes[1].set_title('Standard Spherical Wave (Virtual Focus)', fontsize=14)
    axes[1].set_xlabel('X (Lateral)'); axes[1].set_ylabel('Z (Depth)')
    im2 = axes[1].imshow(np.zeros_like(u2_T), vmin=-1, vmax=1, cmap='gray', aspect='equal')

    if HAS_SKIMAGE:
        if mask1_outline is not None:
            axes[0].contour(mask1_outline.T, levels=[0.5], colors='k', linewidths=2)
        for ax_obj, mask in zip(axes, [mask1_overlay, mask2_overlay]):
            if mask is not None:
                for contour in measure.find_contours(mask.T, 0.5):
                    ax_obj.fill(contour[:, 1], contour[:, 0], color='red', alpha=0.3, edgecolor='r', linewidth=2.5)

    def compute_frame(u_s, t):
        diff = u_s - t
        valid = np.abs(diff) <= (pulse_width / 2.0)
        img = np.zeros_like(u_s, dtype=np.float32)
        if np.any(valid):
            d_val = diff[valid]
            img[valid] = 0.5 * (1.0 + np.cos(2 * np.pi * d_val / pulse_width)) * np.cos(freq * d_val / pulse_width)
        return img

    # Start the timeline early enough to catch the waves originating
    time_steps = np.arange(min_t - pulse_width/2, max_t + pulse_width, speed)
    
    def update(frame_idx):
        t = time_steps[frame_idx]
        im1.set_data(compute_frame(u1_safe, t))
        im2.set_data(compute_frame(u2_safe, t))
        return [im1, im2]
        
    print("Animating...")
    anim = animation.FuncAnimation(fig2, update, frames=len(time_steps), interval=1000/60, blit=True)
    plt.tight_layout()
    plt.show()
    return anim

# Execute the combined animation
anim = animate_side_by_side(u_scaled, u_virt_scaled, aperture_mask, lens_mask, aperture_mask_virt)