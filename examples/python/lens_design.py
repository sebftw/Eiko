# To run this script you must install eiko with the optional example dependencies.
# This is done using "pip install eiko[examples]".

import sys

# Check if the optional dependencies are available
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Polygon
    from matplotlib.lines import Line2D
except ImportError:
    print("\n[Eiko] Error: This script requires 'matplotlib' for visualization.", file=sys.stderr)
    sys.exit(1)

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
msfm = True

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

# -------------------------------------------------------------------------
# TUKEY APODIZATION HELPER FUNCTION (Physical Domain)
# -------------------------------------------------------------------------
def tukey_window(x, alpha=0.15):
    # x is normalized physical coordinate [-1, 1]
    r = np.abs(x)
    w = np.zeros_like(r)
    flat = r <= (1.0 - alpha)
    taper = (r > (1.0 - alpha)) & (r <= 1.0)
    w[flat] = 1.0
    w[taper] = 0.5 * (1.0 + np.cos(np.pi * (r[taper] - (1.0 - alpha)) / alpha))
    return w

# Maximum sine angle subtended at the lens rim (x=d, y=D/2) from virtual focus (-f, 0)
sin_theta_max = (D / 2.0) / np.sqrt((d + f)**2 + (D / 2.0)**2)

# 5. Compute Plane Wave (u)
source_x = -0.0 # mm
valid_y_idx = np.abs(y_vec) <= D / 2
source_idx_x = np.where(x_vec <= source_x / 1000.0)[0][-1] - 1
start_x_idx = np.argmin(np.abs(x_vec - 0.0)) # Grid index at the lens entrance

slowness = (1.0 / sound_speed_map).astype(np.float32)
u_init = np.full((Nx, Ny), np.inf, dtype=np.float32)
u_init[source_idx_x, valid_y_idx] = 0.0

# Apply lateral baffles
slowness[:source_idx_x, valid_y_idx] = np.inf
slowness[:start_x_idx, ~valid_y_idx] = np.inf

# Extract the active y-coordinates
y_in = y_vec[valid_y_idx]

# Define the physical apodization on the plane wave array (using 85% of aperture)
tw_lens = tukey_window(y_in / (D / 2.0), alpha=0.15)

# Evaluate exactly which angle each ray will take after exiting the curved lens
h_in = (-B + np.sqrt(B**2 - 4 * A * (y_in**2 + C0))) / (2 * A)
sin_theta_lens = y_in / np.sqrt((h_in + f)**2 + y_in**2)

# Evaluate matching apodization
v_init = np.zeros_like(slowness)
v_init[source_idx_x, valid_y_idx] = tw_lens
v_init[source_idx_x + 1, valid_y_idx] = tw_lens

print("Computing plane wave passing through lens...")
with torch.no_grad():
    u_t, v_t = eiko(torch.tensor(u_init, device=device), 
                  torch.tensor(slowness, device=device), 
                  v_init=torch.tensor(v_init, device=device), 
                  dx=dx, msfm=msfm)
u = u_t.cpu().numpy()
v = v_t.cpu().numpy()

aperture_mask = np.zeros_like(slowness)
aperture_mask[:source_idx_x, valid_y_idx] = 1.0
aperture_mask[:start_x_idx, ~valid_y_idx] = 1.0

# 6. Compute Virtual Focus (u_virt)
slowness_virt = (1.0 / c2) * np.ones((Nx, Ny), dtype=np.float32)
u_init_virt = np.full((Nx, Ny), np.inf, dtype=np.float32)

# Absolute acoustic distance from the virtual point source (-f, 0)
x_src = x_vec[source_idx_x]
dist_to_vf = np.sqrt((x_src - (-f))**2 + y_vec[valid_y_idx]**2)
u_init_virt[source_idx_x, valid_y_idx] = dist_to_vf / c2
slowness_virt[:source_idx_x, valid_y_idx] = np.inf

# Apply lateral baffles
slowness_virt[:start_x_idx, ~valid_y_idx] = np.inf

# Evaluate the emission angle for the virtual source at the grid injection plane
sin_theta_virt = y_in / np.sqrt((x_src + f)**2 + y_in**2)

# Interpolate the physical array's exact angular energy profile onto the virtual source
tw_virt = np.interp(sin_theta_virt, sin_theta_lens, tw_lens, left=0.0, right=0.0)

v_init_virt = np.zeros_like(slowness_virt)
v_init_virt[source_idx_x, valid_y_idx] = tw_virt
v_init_virt[source_idx_x + 1, valid_y_idx] = tw_virt

print("Computing virtual focus synthesis...")
with torch.no_grad():
    u_virt_t, v_virt_t = eiko(torch.tensor(u_init_virt, device=device), 
                       torch.tensor(slowness_virt, device=device), 
                       v_init=torch.tensor(v_init_virt, device=device), 
                       dx=dx, msfm=msfm)
u_virt = u_virt_t.cpu().numpy()
v_virt = v_virt_t.cpu().numpy()

aperture_mask_virt = np.zeros_like(slowness_virt)
aperture_mask_virt[:source_idx_x, valid_y_idx] = 1.0
aperture_mask_virt[:start_x_idx, ~valid_y_idx] = 1.0

# =========================================================================
# 7. Synchronize the Wavefronts
# =========================================================================
# Sync waves 1cm after the lens exit
target_depth_idx = np.argmin(np.abs(x_vec - 0.05))
center_y_idx = np.argmin(np.abs(y_vec))

# 1. Arrival times at the target depth along the acoustic axis (y = 0)
t_target_lens = u[target_depth_idx, center_y_idx]
t_target_virt = u_virt[target_depth_idx, center_y_idx]

# 2. Shift both fields so t = 0 occurs exactly when the pulse reaches target_depth_idx
u_shifted = u.copy()
u_shifted[np.isfinite(u_shifted)] -= t_target_lens

u_virt_shifted = u_virt.copy()
u_virt_shifted[np.isfinite(u_virt_shifted)] -= t_target_virt

# 3. Convert time fields to spatial index units for the rendering pulse width
u_scaled = u_shifted * c2 / dx
u_virt_scaled = u_virt_shifted * c2 / dx

# Normalize energy fields (v) for display
max_v1 = np.nanmax(v[np.isfinite(v)]) if np.any(np.isfinite(v)) else 1.0
max_v2 = np.nanmax(v_virt[np.isfinite(v_virt)]) if np.any(np.isfinite(v_virt)) else 1.0
v_norm = np.nan_to_num(v / (max_v1 if max_v1 > 0 else 1.0), nan=0.0)
v_virt_norm = np.nan_to_num(v_virt / (max_v2 if max_v2 > 0 else 1.0), nan=0.0)

# =========================================================================
# 8. Render Plots & Synchronized Animation
# =========================================================================
print("Rendering Ray Tracing Plot...")
fig1, ax = plt.subplots(figsize=(9, 5), facecolor='w')
ax.set_title('Acoustic Lens & Virtual Source Ray Geometry', fontsize=13, pad=10)
ax.set_xlabel('Propagation Axis Z (m)')
ax.set_ylabel('Lateral Axis X (m)')
ax.grid(True, linestyle=':', alpha=0.6)
ax.axis('equal')

# Optical Axis Reference
ax.axhline(0, color='gray', linestyle='-.', linewidth=0.8, alpha=0.5)

# Render Lens Polygon
y_pts = np.linspace(-D/2, D/2, 400)
h_trace = (-B + np.sqrt(B**2 - 4 * A * (y_pts**2 + C0))) / (2 * A) 
lens_x = np.concatenate(([0], h_trace, [0]))
lens_y = np.concatenate(([y_pts[0]], y_pts, [y_pts[-1]]))
poly = Polygon(np.column_stack([lens_x, lens_y]), facecolor=[0.85, 0.92, 1.0], 
               edgecolor='steelblue', linewidth=1.2, alpha=0.6, label='Lens')
ax.add_patch(poly)

# Active Aperture Transducer Bar
source_pos_m = source_x / 1000.0
ax.plot([source_pos_m, source_pos_m], [-D/2, D/2], color='lime', linewidth=3.0, 
        solid_capstyle='round', label='Aperture', zorder=5)

# Virtual Source Point
ax.plot(-f, 0, marker='*', markersize=11, color='crimson', markeredgecolor='black', 
        markeredgewidth=0.8, label='Virtual Source', zorder=6)

# Trace 5 Representative Rays (Center, Mid-edges, Outer-edges)
num_rays = 5
y_sample = np.linspace(-D/2 * 0.92, D/2 * 0.92, num_rays)

for i, yi in enumerate(y_sample):
    hi = (-B + np.sqrt(B**2 - 4 * A * (yi**2 + C0))) / (2 * A)
    
    phi_n = np.arctan2(-(-2 * yi / (2 * A * hi + B)), 1)
    phi_t = phi_n + np.arcsin((c2 / c1) * np.sin(-phi_n))
    
    x_end = hi + 0.05 * np.cos(phi_t)
    y_end = yi + 0.05 * np.sin(phi_t)

    # 1. Incident Ray inside lens + Transmitted Ray into medium (solid blue)
    ax.plot([source_pos_m, hi, x_end], [yi, yi, y_end], color='royalblue', 
            linewidth=1.3, label='Acoustic Rays' if i == 0 else "", zorder=4)
    
    # 2. Back-projected virtual ray line (thin dotted red)
    if np.abs(yi) > 1e-4:
        ax.plot([-f, hi], [0, yi], color='crimson', linestyle=':', 
                linewidth=1.0, alpha=0.7, label='Virtual Rays' if i == 0 else "", zorder=3)

ax.set_xlim([-f - 0.015, d + 0.055])
ax.set_ylim([-D/2 - 0.015, D/2 + 0.015])
ax.legend(loc='lower left', framealpha=0.9, fontsize=9)
plt.tight_layout()
plt.show(block=False)

# --- Side-by-Side Animation Function ---
def animate_side_by_side(u1, u2, v1, v2, mask1_overlay, mask1_outline, mask2_overlay):
    pulse_width, speed, freq = 80.0, 0.5, 6 * np.pi
    u1_T, u2_T = u1.T.astype(np.float32), u2.T.astype(np.float32)
    v1_T, v2_T = v1.T.astype(np.float32), v2.T.astype(np.float32)
    
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
    axes[0].set_xlabel('Z (Depth)'); axes[0].set_ylabel('X (Lateral)')
    im1 = axes[0].imshow(np.zeros_like(u1_T), vmin=-1, vmax=1, cmap='gray', aspect='equal', zorder=1)
    
    # Setup Right Panel
    axes[1].set_title('Standard Spherical Wave (Virtual Focus)', fontsize=14)
    axes[1].set_xlabel('Z (Depth)'); axes[1].set_ylabel('X (Lateral)')
    im2 = axes[1].imshow(np.zeros_like(u2_T), vmin=-1, vmax=1, cmap='gray', aspect='equal', zorder=1)

    # Lateral pixel range corresponding to the active aperture [-D/2, D/2]
    y_aperture_indices = np.where(valid_y_idx)[0]
    y_aperture_min, y_aperture_max = y_aperture_indices[0], y_aperture_indices[-1]

    # Draw pure matplotlib contours/masks
    if mask1_outline is not None:
        # Solid cyan line for the physical lens on left panel
        axes[0].contour(mask1_outline.T, levels=[0.5], colors='cyan', linewidths=1.5, zorder=3)

    # Apply overlays to both panels
    for ax_obj, mask in zip(axes, [mask1_overlay, mask2_overlay]):
        if mask is not None:
            # Create a transparent red overlay for the baffled/inactive regions
            mask_rgba = np.zeros((mask.shape[1], mask.shape[0], 4))
            mask_rgba[mask.T > 0.5] = [1, 0, 0, 0.2]
        ax_obj.imshow(mask_rgba, aspect='equal', zorder=2)
        ax_obj.contour(mask.T, levels=[0.5], colors='red', linewidths=1.5, zorder=3)

        # Explicit active transducer aperture bar on BOTH subplots
        ax_obj.plot([source_idx_x, source_idx_x], [y_aperture_min, y_aperture_max], 
                    color='lime', linewidth=3.5, solid_capstyle='round', zorder=10, clip_on=False)

    # Build legend proxy elements
    legend_elements = [
        Line2D([0], [0], color='lime', lw=3, label='Aperture'),
        Line2D([0], [0], color='cyan', lw=1.5, label='Lens'),
        # Line2D([0], [0], color='red', lw=1.5, label='Baffle Boundary')
    ]
    axes[0].legend(handles=legend_elements, loc='lower right')
    axes[1].legend(handles=[legend_elements[0]], loc='lower right')

    def compute_frame(u_s, v_s, t):
        diff = u_s - t
        valid = np.abs(diff) <= (pulse_width / 2.0)
        img = np.zeros_like(u_s, dtype=np.float32)
        if np.any(valid):
            d_val = diff[valid]
            base_pulse = 0.5 * (1.0 + np.cos(2 * np.pi * d_val / pulse_width)) * np.cos(freq * d_val / pulse_width)
            img[valid] = v_s[valid] * base_pulse
        return img

    # Start the timeline early enough to catch the waves originating
    time_steps = np.arange(min_t - pulse_width/2, max_t + pulse_width, speed)
    
    def update(frame_idx):
        t = time_steps[frame_idx]
        im1.set_data(compute_frame(u1_safe, v1_T, t))
        im2.set_data(compute_frame(u2_safe, v2_T, t))
        return [im1, im2]
        
    print("Animating...")
    # blit=False is required so matplotlib respects zorder and draws the images under the background contours
    anim = animation.FuncAnimation(fig2, update, frames=len(time_steps), interval=1000/60, blit=False)
    plt.tight_layout()
    plt.show()
    return anim

# Execute the combined animation
anim = animate_side_by_side(u_scaled, u_virt_scaled, v_norm, v_virt_norm, aperture_mask, lens_mask, aperture_mask_virt)