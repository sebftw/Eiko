# To run this script you must install eiko with the optional example dependencies.
# This is done using "pip install eiko[examples]".

import sys

# Check if the optional dependencies are available
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
except ImportError:
    print("\n[Eiko] Error: This script requires 'matplotlib' for visualization.", file=sys.stderr)
    sys.exit(1)

import torch
import numpy as np

# Import the 3D Eiko solver
from eiko import eiko3d

# =========================================================================
# CONFIGURATION
# =========================================================================
# Select the planes to animate in the 2xN grid. 
# Available options: 'lateral_depth', 'elevation_depth', 'c_scan'
cross_planes_to_plot = ['lateral_depth', 'elevation_depth', 'c_scan']

# Toggle apodization window along the lateral axis (True = Tukey, False = Rectangular / all ones)
apod_tukey_x = True

# =========================================================================
# ACOUSTIC LENS GENERATOR (3D CYLINDRICAL)
# =========================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Lens and Medium Parameters
c1 = 3000.0  
c2 = 1500.0  
f  = 0.031   
d  = 0.04    
D  = 0.08    
elevation_steering_angle_deg = 15.0  # Steer angle on the flat elevation axis

# 2. 3D Grid Parameters (0.5 mm resolution for manageable 3D runtimes)
dx = 0.5e-3 
dy = 0.5e-3 
dz = 0.5e-3
x_domain = 0.1
y_domain = 0.12
z_domain = 0.12
msfm = True

Nx = int(round(x_domain / dx))
Ny = int(round(y_domain / dy))
Nz = int(round(z_domain / dz))

x_vec = np.arange(Nx) * dx - 0.02 
y_vec = np.arange(Ny) * dy - y_domain / 2
z_vec = np.arange(Nz) * dz - z_domain / 2
X, Y, Z = np.meshgrid(x_vec, y_vec, z_vec, indexing='ij')

# Physical coordinate vectors in millimeters for plotting
x_mm = x_vec * 1e3
y_mm = y_vec * 1e3
z_mm = z_vec * 1e3

# 3. Mathematical Surface Evaluation (Cylindrical Extrusion along Z)
n = c2 / c1            
L = d * (1 - n) + f     
A = 1 - n**2
B = 2 * (f - n * L)
C0 = f**2 - L**2        

h_grid = np.zeros((Nx, Ny, Nz))
valid_Y = np.abs(Y) <= D / 2

# Curve is evaluated purely along the Lateral (Y) axis
C_valid = Y[valid_Y]**2 + C0
Delta = B**2 - 4 * A * C_valid
h_grid[valid_Y] = (-B + np.sqrt(Delta)) / (2 * A)

# 4. Discretization 
lens_mask = valid_Y & (X >= 0) & (X <= h_grid)
sound_speed_map = c2 * np.ones((Nx, Ny, Nz))
sound_speed_map[lens_mask] = c1

# -------------------------------------------------------------------------
# APODIZATION HELPER FUNCTIONS
# -------------------------------------------------------------------------
def tukey_window(x, alpha=0.15):
    r = np.abs(x)
    w = np.zeros_like(r)
    flat = r <= (1.0 - alpha)
    taper = (r > (1.0 - alpha)) & (r <= 1.0)
    w[flat] = 1.0
    w[taper] = 0.5 * (1.0 + np.cos(np.pi * (r[taper] - (1.0 - alpha)) / alpha))
    return w

# 5. Compute Plane Wave (u)
source_x = -0.0 # mm
source_idx_x = np.where(x_vec <= source_x / 1000.0)[0][-1] - 1
start_x_idx = np.argmin(np.abs(x_vec - 0.0))

slowness = (1.0 / sound_speed_map).astype(np.float32)
u_init = np.full((Nx, Ny, Nz), np.inf, dtype=np.float32)

valid_y_mask = np.abs(y_vec) <= D / 2
valid_z_mask = np.abs(z_vec) <= D / 2
valid_aperture = np.outer(valid_y_mask, valid_z_mask)
y_in = y_vec[valid_y_mask]
z_in = z_vec[valid_z_mask]

# Target steering angle in medium c2
theta_target = np.radians(elevation_steering_angle_deg)

# Steeper internal angle required inside the lens (c1) by Snell's law
theta_lens = np.arcsin((c1 / c2) * np.sin(theta_target))

# Calculate slab refraction elevation shift (in mm) along the Z steering axis
lens_shift_z = d * (np.tan(theta_lens) - np.tan(theta_target)) * 1e3

# Phase delays along the elevation axis (Z)
z_delays_lens_1d = z_in * np.sin(theta_lens) / c1
z_delays_lens_2d = np.tile(z_delays_lens_1d, (len(y_in), 1))
z_delays_lens_2d -= np.min(z_delays_lens_2d)

u_init_slice = np.full((Ny, Nz), np.inf, dtype=np.float32)
u_init_slice[np.ix_(valid_y_mask, valid_z_mask)] = z_delays_lens_2d
u_init[source_idx_x, :, :] = u_init_slice

slowness[:source_idx_x, :, :] = np.inf

# Generate 2D separable apodization with toggle for X/Lateral axis
if apod_tukey_x:
    tw_y = tukey_window(y_in / (D / 2.0), alpha=0.15)
else:
    tw_y = np.ones_like(y_in)

tw_z = tukey_window(z_in / (D / 2.0), alpha=0.15)
tw_2d = np.outer(tw_y, tw_z)

v_lens_2d = np.zeros((Ny, Nz), dtype=np.float32)
v_lens_2d[np.ix_(valid_y_mask, valid_z_mask)] = tw_2d

v_init = np.zeros_like(slowness)
v_init[source_idx_x, :, :] = v_lens_2d
v_init[source_idx_x + 1, :, :] = v_lens_2d

print(f"Computing 3D steered plane wave passing through lens... (This may take a moment)")
with torch.no_grad():
    u_t, v_t = eiko3d(torch.tensor(u_init, device=device), 
                      torch.tensor(slowness, device=device), 
                      v_init=torch.tensor(v_init, device=device), 
                      dx=dx, msfm=msfm)
u = u_t.cpu().numpy()
v = v_t.cpu().numpy()

# 6. Compute Virtual Focus (u_virt) - Steered Cylindrical Wave
slowness_virt = (1.0 / c2) * np.ones((Nx, Ny, Nz), dtype=np.float32)
u_init_virt = np.full((Nx, Ny, Nz), np.inf, dtype=np.float32)

x_src = x_vec[source_idx_x]
dist_to_vf_lat = np.sqrt((x_src - (-f))**2 + y_in**2)
dist_2d = np.tile(dist_to_vf_lat[:, None], (1, len(z_in)))

focal_delays_2d = dist_2d / c2
z_delays_virt_1d = z_in * np.sin(theta_target) / c2
z_delays_virt_2d = np.tile(z_delays_virt_1d, (len(y_in), 1))

combined_delays = focal_delays_2d + z_delays_virt_2d
combined_delays -= np.min(combined_delays)

u_virt_slice = np.full((Ny, Nz), np.inf, dtype=np.float32)
u_virt_slice[np.ix_(valid_y_mask, valid_z_mask)] = combined_delays
u_init_virt[source_idx_x, :, :] = u_virt_slice

slowness_virt[:source_idx_x, :, :] = np.inf

h_in = (-B + np.sqrt(B**2 - 4 * A * (y_in**2 + C0))) / (2 * A)
sin_theta_lens_lat = y_in / np.sqrt((h_in + f)**2 + y_in**2)
sin_theta_virt_lat = y_in / np.sqrt((x_src + f)**2 + y_in**2)

tw_virt_y = np.interp(sin_theta_virt_lat, sin_theta_lens_lat, tw_y, left=0.0, right=0.0)
tw_virt_2d = np.outer(tw_virt_y, tw_z)

v_virt_2d = np.zeros((Ny, Nz), dtype=np.float32)
v_virt_2d[np.ix_(valid_y_mask, valid_z_mask)] = tw_virt_2d

v_init_virt = np.zeros_like(slowness_virt)
v_init_virt[source_idx_x, :, :] = v_virt_2d
v_init_virt[source_idx_x + 1, :, :] = v_virt_2d

print("Computing 3D steered virtual focus (cylindrical wave) synthesis...")
with torch.no_grad():
    u_virt_t, v_virt_t = eiko3d(torch.tensor(u_init_virt, device=device), 
                                torch.tensor(slowness_virt, device=device), 
                                v_init=torch.tensor(v_init_virt, device=device), 
                                dx=dx, msfm=msfm)
u_virt = u_virt_t.cpu().numpy()
v_virt = v_virt_t.cpu().numpy()

# =========================================================================
# 7. Synchronize the Wavefronts (Global 3D Shift)
# =========================================================================
target_depth = 50/1000
target_depth_idx = np.argmin(np.abs(x_vec - target_depth))
center_y_idx = np.argmin(np.abs(y_vec))
center_z_idx = np.argmin(np.abs(z_vec))

t_target_lens = u[target_depth_idx, center_y_idx, center_z_idx]
t_target_virt = u_virt[target_depth_idx, center_y_idx, center_z_idx]

u_shifted = u.copy()
u_shifted[np.isfinite(u_shifted)] -= t_target_lens

u_virt_shifted = u_virt.copy()
u_virt_shifted[np.isfinite(u_virt_shifted)] -= t_target_virt

u_scaled = u_shifted * c2 / dx
u_virt_scaled = u_virt_shifted * c2 / dx

max_v1 = np.nanmax(v[np.isfinite(v)]) if np.any(np.isfinite(v)) else 1.0
max_v2 = np.nanmax(v_virt[np.isfinite(v_virt)]) if np.any(np.isfinite(v_virt)) else 1.0
v_norm = np.nan_to_num(v / (max_v1 if max_v1 > 0 else 1.0), nan=0.0)
v_virt_norm = np.nan_to_num(v_virt / (max_v2 if max_v2 > 0 else 1.0), nan=0.0)

# =========================================================================
# 8. Render Transposed Synchronized Animation
# =========================================================================
def animate_selected_planes(cross_planes):
    pulse_width, speed, freq = 80.0, 0.5, 6 * np.pi
    
    valid_u = u_scaled[np.isfinite(u_scaled)]
    valid_u_virt = u_virt_scaled[np.isfinite(u_virt_scaled)]
    
    min_t = min(np.min(valid_u), np.min(valid_u_virt))
    max_t = max(np.max(valid_u), np.max(valid_u_virt))
    
    num_cols = len(cross_planes)
    # Row 0: Virtual Focus (Top), Row 1: Physical Lens (Bottom)
    fig, axes = plt.subplots(2, num_cols, figsize=(6 * num_cols, 8), facecolor='white', squeeze=False)
        
    fig.suptitle(f'Lens Design (Steered {elevation_steering_angle_deg}°)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    render_targets = []
    
    extent_lat_depth     = [x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]]
    extent_elev_depth    = [x_mm[0], x_mm[-1], z_mm[0], z_mm[-1]]
    extent_elev_lens     = [x_mm[0], x_mm[-1], z_mm[0] + lens_shift_z, z_mm[-1] + lens_shift_z]
    extent_cscan         = [y_mm[0], y_mm[-1], z_mm[0], z_mm[-1]]
    extent_cscan_lens    = [y_mm[0], y_mm[-1], z_mm[0] + lens_shift_z, z_mm[-1] + lens_shift_z]
    
    for col, plane in enumerate(cross_planes):
        ax_virt = axes[0, col] # Row 0: Virtual Focus Reference (Top)
        ax_lens = axes[1, col] # Row 1: Physical Lens (Bottom)
        
        if col == 0:
            ax_virt.annotate("Virtual Focus", xy=(0, 0.5), xycoords='axes fraction', 
                             xytext=(-80, 0), textcoords='offset points',
                             va='center', ha='center', rotation=90, fontsize=14, fontweight='bold')
            ax_lens.annotate("Physical Lens", xy=(0, 0.5), xycoords='axes fraction', 
                             xytext=(-80, 0), textcoords='offset points',
                             va='center', ha='center', rotation=90, fontsize=14, fontweight='bold')
        
        if plane == 'lateral_depth':
            u_top = u_virt_scaled[:, :, center_z_idx].T.astype(np.float32)
            u_bot = u_scaled[:, :, center_z_idx].T.astype(np.float32)
            v_top = v_virt_norm[:, :, center_z_idx].T.astype(np.float32)
            v_bot = v_norm[:, :, center_z_idx].T.astype(np.float32)
            
            u_top_safe = np.copy(u_top)
            u_bot_safe = np.copy(u_bot)
            u_top_safe[np.isinf(u_top_safe)] = max_t + pulse_width * 2
            u_bot_safe[np.isinf(u_bot_safe)] = max_t + pulse_width * 2
            
            ax_virt.set_title('XZ-slice', fontsize=13, fontweight='bold', pad=10)
            
            ax_virt.tick_params(labelbottom=False)
            ax_lens.set_xlabel('Depth - Z [mm]')
            for ax in (ax_virt, ax_lens):
                ax.set_ylabel('Lateral - X [mm]')
                
            im_virt = ax_virt.imshow(np.zeros_like(u_top), vmin=-1, vmax=1, cmap='gray', 
                                     aspect='equal', extent=extent_lat_depth, origin='lower', zorder=1)
            im_lens = ax_lens.imshow(np.zeros_like(u_bot), vmin=-1, vmax=1, cmap='gray', 
                                     aspect='equal', extent=extent_lat_depth, origin='lower', zorder=1)
            
            l_mask = lens_mask[:, :, center_z_idx].T
            ax_lens.contour(x_mm, y_mm, l_mask, levels=[0.5], colors='cyan', linewidths=1.5, zorder=3)
            
            b_mask = np.zeros((Nx, Ny))
            b_mask[:source_idx_x, :] = 1.0
            b_mask = b_mask.T
            
            for ax in (ax_virt, ax_lens):
                b_rgba = np.zeros((b_mask.shape[0], b_mask.shape[1], 4))
                b_rgba[b_mask > 0.5] = [1, 0, 0, 0.2]
                ax.imshow(b_rgba, aspect='equal', extent=extent_lat_depth, origin='lower', zorder=2)
                ax.contour(x_mm, y_mm, b_mask, levels=[0.5], colors='red', linewidths=1.5, zorder=3)
                ax.plot([source_x, source_x], [-D/2 * 1e3, D/2 * 1e3], color='lime', linewidth=3.5, solid_capstyle='round', zorder=10)
                
            render_targets.append((im_virt, im_lens, u_top_safe, u_bot_safe, v_top, v_bot))
            
        elif plane == 'elevation_depth':
            u_top = u_virt_scaled[:, center_y_idx, :].T.astype(np.float32)
            u_bot = u_scaled[:, center_y_idx, :].T.astype(np.float32)
            v_top = v_virt_norm[:, center_y_idx, :].T.astype(np.float32)
            v_bot = v_norm[:, center_y_idx, :].T.astype(np.float32)
            
            u_top_safe = np.copy(u_top)
            u_bot_safe = np.copy(u_bot)
            u_top_safe[np.isinf(u_top_safe)] = max_t + pulse_width * 2
            u_bot_safe[np.isinf(u_bot_safe)] = max_t + pulse_width * 2
            
            ax_virt.set_title('YZ-slice', fontsize=13, fontweight='bold', pad=10)
            
            ax_virt.tick_params(labelbottom=False)
            ax_lens.set_xlabel('Depth - Z [mm]')
            for ax in (ax_virt, ax_lens):
                ax.set_ylabel('Elevation - Y [mm]')
                
            im_virt = ax_virt.imshow(np.zeros_like(u_top), vmin=-1, vmax=1, cmap='gray', 
                                     aspect='equal', extent=extent_elev_depth, origin='lower', zorder=1)
            # Physical lens view limits shifted via ylims/extent to reflect slab refraction
            im_lens = ax_lens.imshow(np.zeros_like(u_bot), vmin=-1, vmax=1, cmap='gray', 
                                     aspect='equal', extent=extent_elev_lens, origin='lower', zorder=1)
            
            l_mask = lens_mask[:, center_y_idx, :].T
            ax_lens.contour(x_mm, z_mm + lens_shift_z, l_mask, levels=[0.5], colors='cyan', linewidths=1.5, zorder=3)
            
            b_mask = np.zeros((Nx, Nz))
            b_mask[:source_idx_x, :] = 1.0
            b_mask = b_mask.T
            
            # Top (Virtual Focus)
            b_rgba_top = np.zeros((b_mask.shape[0], b_mask.shape[1], 4))
            b_rgba_top[b_mask > 0.5] = [1, 0, 0, 0.2]
            ax_virt.imshow(b_rgba_top, aspect='equal', extent=extent_elev_depth, origin='lower', zorder=2)
            ax_virt.contour(x_mm, z_mm, b_mask, levels=[0.5], colors='red', linewidths=1.5, zorder=3)
            ax_virt.plot([source_x, source_x], [-D/2 * 1e3, D/2 * 1e3], color='lime', linewidth=3.5, solid_capstyle='round', zorder=10)

            # Bottom (Physical Lens - shifted via extent/ylim)
            b_rgba_bot = np.zeros((b_mask.shape[0], b_mask.shape[1], 4))
            b_rgba_bot[b_mask > 0.5] = [1, 0, 0, 0.2]
            ax_lens.imshow(b_rgba_bot, aspect='equal', extent=extent_elev_lens, origin='lower', zorder=2)
            ax_lens.contour(x_mm, z_mm + lens_shift_z, b_mask, levels=[0.5], colors='red', linewidths=1.5, zorder=3)
            ax_lens.plot([source_x, source_x], [-D/2 * 1e3 + lens_shift_z, D/2 * 1e3 + lens_shift_z], color='lime', linewidth=3.5, solid_capstyle='round', zorder=10)
                
            render_targets.append((im_virt, im_lens, u_top_safe, u_bot_safe, v_top, v_bot))
            
        elif plane == 'c_scan':
            u_top = u_virt_scaled[target_depth_idx, :, :].T.astype(np.float32)
            u_bot = u_scaled[target_depth_idx, :, :].T.astype(np.float32)
            v_top = v_virt_norm[target_depth_idx, :, :].T.astype(np.float32)
            v_bot = v_norm[target_depth_idx, :, :].T.astype(np.float32)
            
            u_top_safe = np.copy(u_top)
            u_bot_safe = np.copy(u_bot)
            u_top_safe[np.isinf(u_top_safe)] = max_t + pulse_width * 2
            u_bot_safe[np.isinf(u_bot_safe)] = max_t + pulse_width * 2
            
            ax_virt.set_title(f'C-Scan (Depth = {x_vec[target_depth_idx]*1000:.1f} mm)', fontsize=13, fontweight='bold', pad=10)
            
            ax_virt.tick_params(labelbottom=False)
            ax_lens.set_xlabel('Lateral - X [mm]')
            for ax in (ax_virt, ax_lens):
                ax.set_ylabel('Elevation - Y [mm]')
                
            im_virt = ax_virt.imshow(np.zeros_like(u_top), vmin=-1, vmax=1, cmap='gray', 
                                     aspect='equal', extent=extent_cscan, origin='lower', zorder=1)
            # Physical lens C-scan shown at shifted extent to reflect true slab refraction coordinates
            im_lens = ax_lens.imshow(np.zeros_like(u_bot), vmin=-1, vmax=1, cmap='gray', 
                                     aspect='equal', extent=extent_cscan_lens, origin='lower', zorder=1)
            
            aperture_extent_mm = D * 1e3
            
            # Virtual Focus Aperture & Center
            rect_top = Rectangle((-aperture_extent_mm / 2, -aperture_extent_mm / 2), 
                                 aperture_extent_mm, aperture_extent_mm,
                                 color='lime', fill=False, linewidth=1.5, linestyle='--', zorder=10, alpha=0.7)
            ax_virt.add_patch(rect_top)
            ax_virt.axvline(0, color='yellow', linestyle=':', alpha=0.4, zorder=4)
            ax_virt.axhline(0, color='yellow', linestyle=':', alpha=0.4, zorder=4)

            # Physical Lens Aperture & Center (shifted by lens_shift_z)
            rect_bot = Rectangle((-aperture_extent_mm / 2, -aperture_extent_mm / 2 + lens_shift_z), 
                                 aperture_extent_mm, aperture_extent_mm,
                                 color='lime', fill=False, linewidth=1.5, linestyle='--', zorder=10, alpha=0.7)
            ax_lens.add_patch(rect_bot)
            ax_lens.axvline(0, color='yellow', linestyle=':', alpha=0.4, zorder=4)
            ax_lens.axhline(lens_shift_z, color='yellow', linestyle=':', alpha=0.4, zorder=5)
                
            render_targets.append((im_virt, im_lens, u_top_safe, u_bot_safe, v_top, v_bot))

    legend_elements = [
        Line2D([0], [0], color='lime', lw=3, label='Aperture'),
        Line2D([0], [0], color='cyan', lw=1.5, label='Lens'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11, bbox_to_anchor=(0.5, 0.02))
    
    plt.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.15, hspace=0.15, wspace=0.25)
            
    def compute_frame(u_s, v_s, t):
        diff = u_s - t
        valid = np.abs(diff) <= (pulse_width / 2.0)
        img = np.zeros_like(u_s, dtype=np.float32)
        if np.any(valid):
            d_val = diff[valid]
            base_pulse = 0.5 * (1.0 + np.cos(2 * np.pi * d_val / pulse_width)) * np.cos(freq * d_val / pulse_width)
            img[valid] = v_s[valid] * base_pulse
        return img

    time_steps = np.arange(min_t - pulse_width/2, max_t + pulse_width, speed)
    
    def update(frame_idx):
        t = time_steps[frame_idx]
        ret = []
        for im_virt, im_lens, u_top_s, u_bot_s, v_top, v_bot in render_targets:
            im_virt.set_data(compute_frame(u_top_s, v_top, t))
            im_lens.set_data(compute_frame(u_bot_s, v_bot, t))
            ret.extend([im_virt, im_lens])
        return ret
        
    print(f"Animating {len(cross_planes)} cross-planes...")
    anim = animation.FuncAnimation(fig, update, frames=len(time_steps), interval=1000/60, blit=False)
    plt.show()
    return anim

# Execute the 3D animation sequence based on configuration
anim = animate_selected_planes(cross_planes_to_plot)