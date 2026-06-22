import sys
import os
import math

# =========================================================================
# Visualization Dependency & OS-Specific Diagnostics
# =========================================================================
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, PillowWriter
    
    if matplotlib.get_backend().lower() == 'agg':
        raise RuntimeError("Matplotlib is using the non-interactive 'Agg' backend.")
        
except (ImportError, RuntimeError) as e:
    print("\n[Eiko] Error: Interactive visualization could not be initialized.", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    print("\n--- How to Fix This ---", file=sys.stderr)
    
    if sys.platform == 'win32':
        print("Command: pip install PyQt5\n", file=sys.stderr)
    elif sys.platform.startswith('linux'):
        is_wsl = False
        try:
            with open('/proc/version', 'r') as f:
                if 'microsoft' in f.read().lower():
                    is_wsl = True
        except FileNotFoundError:
            pass
            
        if is_wsl:
            print("Detected OS: Windows Subsystem for Linux (WSL)", file=sys.stderr)
            print("Fix: Run this script from the native Windows Command Prompt instead.\n", file=sys.stderr)
        else:
            print("Command: sudo apt-get install python3-tk\n", file=sys.stderr)
    else:
        print("Command: pip install PyQt5\n", file=sys.stderr)
    sys.exit(1)

import torch
import numpy as np
from eiko import eiko

# =========================================================================
# 1. Setup Domain and Coordinates
# =========================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = 201
dx = 0.0005         # Grid spacing in meters (0.5 mm)
msfm = True         # Multi-stencil fast marching
c_bg = 1540.0       # Background speed of sound

# Create spatial coordinate grids centered at 0
offset = (N // 2) * dx
x_coords = torch.arange(N, dtype=torch.float32, device=device) * dx - offset
y_coords = torch.arange(N, dtype=torch.float32, device=device) * dx - offset

# Y (rows) and X (columns) grids
Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')

# Pre-compute numpy versions for Matplotlib to save overhead in loops
X_np = X.cpu().numpy()
Y_np = Y.cpu().numpy()

# =========================================================================
# 2. Define Slowness Field & Masks
# =========================================================================
def get_U_mask(X, Y):
    xc = 0.033
    yc = -0.025
    r_outer = 0.006
    thickness = 0.002
    r_inner = r_outer - thickness
    arm_length = 0.015
    
    # PyTorch logical masks
    arc_mask = ((X - xc)**2 + (Y - yc)**2 <= r_outer**2) & \
               ((X - xc)**2 + (Y - yc)**2 >= r_inner**2) & \
               (Y <= yc)
               
    left_arm_mask = (X >= xc - r_outer) & (X <= xc - r_inner) & \
                    (Y > yc) & (Y <= yc + arm_length)
                    
    right_arm_mask = (X >= xc + r_inner) & (X <= xc + r_outer) & \
                     (Y > yc) & (Y <= yc + arm_length)
                     
    return arc_mask | left_arm_mask | right_arm_mask

c_field = torch.full((N, N), c_bg, dtype=torch.float32, device=device)

# Example 1: High-speed circular anomaly
radius = 0.015
circ_mask = (X - 0.0)**2 + (Y - 0.01)**2 <= radius**2
c_field[circ_mask] = 3000.0

# Example 2: U-shaped impenetrable region (speed = 0)
u_mask = get_U_mask(X, Y)
c_field[u_mask] = 0.0

# Convert speed map to slowness map (1/c) with safety for division by zero
f = torch.where(c_field > 0, 1.0 / c_field, torch.tensor(float('inf'), dtype=torch.float32, device=device))

# Calculate worst-case max time to fix the colorbar
max_dist = math.sqrt(2) * (N * dx)
min_speed = c_field[c_field > 0].min().item()
global_max_time_ms = (max_dist / min_speed) * 1000

# =========================================================================
# 3. Interactive Visualization Setup
# =========================================================================
print(f"Initializing Interactive EIKO solver on {device}...")

extent = [x_coords[0].item()*1000, x_coords[-1].item()*1000, 
          y_coords[0].item()*1000, y_coords[-1].item()*1000]

fig, ax = plt.subplots(figsize=(8, 7), facecolor='white')
fig.suptitle('Interactive Eikonal Solver\n(Inside = Point Source | Outside = Plane Wave)', 
             fontsize=14, fontweight='bold')

# Run an initial solve at the center
u_init = torch.full((N, N), float('inf'), dtype=torch.float32, device=device)
center_idx = N // 2
u_init[center_idx, center_idx] = 0.0
with torch.no_grad():
    u_initial = eiko(u_init, f, dx=dx, msfm=msfm)
u_ms = u_initial.cpu().numpy() * 1000

# Plot the Time-of-Flight map with fixed limits
im = ax.imshow(u_ms, extent=extent, cmap='viridis', origin='lower', vmin=0, vmax=global_max_time_ms)
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Time of Flight (ms)')

# Overlay the speed map as faint contours
ax.contour(X_np*1000, Y_np*1000, c_field.cpu().numpy(), levels=[1540, 3000], 
           colors='white', alpha=0.4, linewidths=1.5, linestyles='dashed')

# Overlay the wavefront contours
contour_levels = np.linspace(0, global_max_time_ms, 7)
wavefront_contours = ax.contour(X_np*1000, Y_np*1000, u_ms, levels=contour_levels, 
                                colors='k', linewidths=0.5)

# Text box for state
info_text = ax.text(0.03, 0.96, f"Source: Point at (0.0, 0.0) mm\nLocal Speed: {c_bg:.0f} m/s", 
                    transform=ax.transAxes, fontsize=11, color='white', verticalalignment='top',
                    bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.4'))

# =========================================================================
# 4. Update Logic & Callbacks
# =========================================================================
def update_wave_source(x_m, y_m):
    global wavefront_contours
    
    x_min, x_max = extent[0], extent[1]
    y_min, y_max = extent[2], extent[3]
    is_inside = (x_min <= x_m <= x_max) and (y_min <= y_m <= y_max)
    
    u_init_new = torch.full((N, N), float('inf'), dtype=torch.float32, device=device)
    
    if is_inside:
        idx_x = max(0, min(N - 1, int(round((x_m / 1000.0 + offset) / dx))))
        idx_y = max(0, min(N - 1, int(round((y_m / 1000.0 + offset) / dx))))
        
        c_local = c_field[idx_y, idx_x].item()
        u_init_new[idx_y, idx_x] = 0.0
        
        status_str = f"Source: Point at ({x_m:.1f}, {y_m:.1f}) mm\nLocal Speed: {c_local:.0f} m/s"
    else:
        theta = np.arctan2(y_m, x_m)
        d_x = float(-np.cos(theta))
        d_y = float(-np.sin(theta))
        
        T = (X * d_x + Y * d_y) / c_bg
        T = T - T.min()
        
        if d_x > 0: u_init_new[:, 0]  = T[:, 0]
        if d_x < 0: u_init_new[:, -1] = T[:, -1]
        if d_y > 0: u_init_new[0, :]  = T[0, :]
        if d_y < 0: u_init_new[-1, :] = T[-1, :]
            
        deg = np.degrees(theta)
        status_str = f"Source: Plane Wave from {deg:.1f}°\nLocal Speed: N/A"

    info_text.set_text(status_str)

    with torch.no_grad():
        u_numerical = eiko(u_init_new, f, dx=dx, msfm=msfm)
    
    u_ms_new = u_numerical.cpu().numpy() * 1000
    im.set_data(u_ms_new)
    
    # Remove old contours and draw new ones
    try:
        wavefront_contours.remove()
    except AttributeError:
        for coll in wavefront_contours.collections:
            coll.remove()
    wavefront_contours = ax.contour(X_np*1000, Y_np*1000, u_ms_new, levels=contour_levels, 
                                    colors='k', linewidths=0.5)
    
    fig.canvas.draw_idle()

def on_mouse_move(event):
    if event.x is None or event.y is None:
        return
    inv = ax.transData.inverted()
    x_m, y_m = inv.transform((event.x, event.y))
    update_wave_source(x_m, y_m)

def on_key_press(event):
    global cid_mouse
    if event.key == 'r':
        # Unbind mouse to prevent interference during recording
        fig.canvas.mpl_disconnect(cid_mouse)
        info_text.set_bbox(dict(facecolor='darkred', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.4'))
        
        # Decide format based on FFMpeg availability
        if FFMpegWriter.isAvailable():
            writer = FFMpegWriter(fps=60)
            ext = "mp4"
        else:
            writer = PillowWriter(fps=60)
            ext = "gif"
            
        video_filename = f"eikonal_smooth_path.{ext}"
        print(f"Recording smooth trajectory to {video_filename}...")
        
        num_frames = 60 * 3
        radius_mm = 20.0
        center_x, center_y = 0.0, -5.0
        
        angles = np.linspace(0, 2*np.pi, num_frames + 1)[:-1]
        x_circle = center_x + radius_mm * -np.cos(angles)
        y_circle = center_y + radius_mm * np.sin(angles)
        
        with writer.saving(fig, video_filename, dpi=100):
            for x, y in zip(x_circle, y_circle):
                update_wave_source(x, y)
                fig.canvas.draw()
                writer.grab_frame()
                
        print("Recording Complete!")
        
        # Restore state
        info_text.set_bbox(dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.4'))
        cid_mouse = fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

# Register callbacks
cid_mouse = fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
fig.canvas.mpl_connect('key_press_event', on_key_press)

plt.tight_layout()
plt.show()
