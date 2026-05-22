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
from eiko import eiko

# =========================================================================
# 1. Setup Domain and Coordinates
# =========================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = 101
dx = 0.001          # Grid spacing in meters (1 mm)
msfm = True         # Multi-stencil fast marching
c_bg = 1540.0       # Background speed of sound

# Create spatial coordinate grids centered at 0offset = (N // 2) * dx
x_coords = torch.arange(N, dtype=torch.float32, device=device) * dx - offset
y_coords = torch.arange(N, dtype=torch.float32, device=device) * dx - offset

# Y (rows) and X (columns) grids
Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')

# =========================================================================
# 2. Define Slowness Field (Customizable)
# =========================================================================
# --- DEFINE YOUR SLOWNESS FIELD HERE ---
c_field = torch.full((N, N), c_bg, dtype=torch.float32, device=device)

# Example:
# A high-speed circular anomaly (e.g., a lens or bone at 6000 m/s)
radius = 0.015
mask = (X - 0.0)**2 + (Y - 0.01)**2 <= radius**2
c_field[mask] = 3000.0  

# Convert speed map to slowness map (1/c)
f = 1.0 / c_field
# ---------------------------------------

# Initialize u_init with infinity at unknown points
u_init = torch.full((N, N), float('inf'), dtype=torch.float32, device=device)

# =========================================================================
# 3. Interactive Visualization Setup
# =========================================================================
print(f"Initializing Interactive EIKO solver on {device}...")

# Set extent for real-world axis scales in mm
extent = [x_coords[0].item()*1000, x_coords[-1].item()*1000, 
          y_coords[0].item()*1000, y_coords[-1].item()*1000]

fig, ax = plt.subplots(figsize=(8, 7), facecolor='white')
fig.suptitle('Interactive Eikonal Solver\n(Inside plot = Point Source | Outside plot = Plane Wave)', 
             fontsize=14, fontweight='bold')

# Run an initial solve at the center so the plot isn't empty
center_idx = N // 2
u_init[center_idx, center_idx] = 0.0
with torch.no_grad():
    u_initial = eiko(u_init, f, dx=dx, msfm=msfm)

# Plot the Time-of-Flight map
im = ax.imshow(u_initial.cpu().numpy() * 1000, extent=extent, cmap='viridis', origin='lower')
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Time of Flight (ms)')

# Overlay the speed map as faint contours
ax.contour(X.cpu().numpy()*1000, Y.cpu().numpy()*1000, c_field.cpu().numpy() * 1000, 
           colors='white', alpha=0.4, linewidths=1.5, linestyles='dashed')

# Add a text box to display the current state
info_text = ax.text(0.03, 0.96, "Source: Point at (0.0, 0.0) mm", 
                    transform=ax.transAxes, fontsize=11, color='white',
                    verticalalignment='top',
                    bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.4'))

# =========================================================================
# 4. Interactive Mouse Callback
# =========================================================================
def on_mouse_move(event):
    # Ignore if mouse leaves the figure window entirely
    if event.x is None or event.y is None:
        return
    
    # Use Matplotlib's coordinate transforms to get data coordinates (in mm)
    # This works even when the mouse is outside the axes!
    inv = ax.transData.inverted()
    x_m, y_m = inv.transform((event.x, event.y))
    
    # Check if the mouse is physically inside the plot domain
    x_min, x_max = extent[0], extent[1]
    y_min, y_max = extent[2], extent[3]
    is_inside = (x_min <= x_m <= x_max) and (y_min <= y_m <= y_max)
    
    # Reset initialization array
    u_init.fill_(float('inf'))
    
    if is_inside:
        # --- POINT SOURCE MODE ---
        # Convert mm to grid indices
        idx_x = int(round((x_m / 1000.0 + offset) / dx))
        idx_y = int(round((y_m / 1000.0 + offset) / dx))
        idx_x = max(0, min(N - 1, idx_x))
        idx_y = max(0, min(N - 1, idx_y))
        
        u_init[idx_y, idx_x] = 0.0
        info_text.set_text(f"Source: Point at ({x_m:.1f}, {y_m:.1f}) mm")
        
    else:
        # --- PLANE WAVE MODE ---
        # Angle from center to the mouse cursor
        theta = np.arctan2(y_m, x_m)
        
        # Propagation direction (pointing inwards from the mouse)
        d_x = float(-np.cos(theta))
        d_y = float(-np.sin(theta))
        
        # Calculate the theoretical time-of-flight for a plane wave over the whole grid
        T = (X * d_x + Y * d_y) / c_bg
        
        # Shift it so the earliest time entering the domain is exactly 0
        T = T - T.min()
        
        # Apply the plane wave only to the 'inflow' boundaries
        if d_x > 0: u_init[:, 0]  = T[:, 0]   # Enters from left
        if d_x < 0: u_init[:, -1] = T[:, -1]  # Enters from right
        if d_y > 0: u_init[0, :]  = T[0, :]   # Enters from bottom
        if d_y < 0: u_init[-1, :] = T[-1, :]  # Enters from top
            
        deg = np.degrees(theta)
        info_text.set_text(f"Source: Plane Wave from {deg:.1f}°")
    
    # Run solver
    with torch.no_grad():
        u_numerical = eiko(u_init, f, dx=dx, msfm=msfm)
        
    u_cpu = u_numerical.cpu().numpy()
    
    # Update image data
    im.set_data(u_cpu * 1000)
    
    # Update colorbar limits dynamically
    valid_max = u_cpu[u_cpu != float('inf')].max() * 1000
    im.set_clim(vmin=0, vmax=valid_max)
    
    # Request a redraw
    fig.canvas.draw_idle()

# Register the mouse motion event
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

plt.tight_layout()
plt.show()