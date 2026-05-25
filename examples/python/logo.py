# To run this script you must install eiko with the optional example dependencies.
# This is done using "pip install eiko[examples]".

import sys
import os
import platform
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

# Import your solver
from eiko import eiko

def get_system_font(size):
    """Automatically detects the OS and returns a large, bold system font."""
    system = platform.system()
    
    if system == "Windows":
        # Standard Windows bold font
        font_paths = ["C:\\Windows\\Fonts\\arialbd.ttf"]
    elif system == "Darwin":  # macOS
        font_paths = ["/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial Bold.ttf"]
    else:  # Linux
        # Common Ubuntu/Linux font paths
        font_paths = [
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
        ]
        
    # Find the first valid font path on this system
    valid_font = next((f for f in font_paths if os.path.exists(f)), None)
    
    try:
        if valid_font:
            return ImageFont.truetype(valid_font, size)
        else:
            raise IOError("No matching fonts found.")
    except IOError:
        print("Warning: Could not load OS font. Falling back to default (text will be tiny).", file=sys.stderr)
        return ImageFont.load_default()

# --- 1. Setup Canvas using PIL ---
canvasHeight = 200
canvasWidth = 600

# Create a black canvas
img = Image.new('L', (canvasWidth, canvasHeight), color=0)
draw = ImageDraw.Draw(img)

# Load OS-specific font at size 200
font = get_system_font(200)

# Anchor='mm' handles centering automatically; just give it the middle coordinates
center_x = canvasWidth / 2
center_y = canvasHeight / 2
draw.text((center_x, center_y), "Eiko", fill=255, font=font, anchor='mm')

# Convert to NumPy array
canvas = np.array(img, dtype=np.float32)

# --- 2. Slowness Map & Eikonal Solver ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# f acts as slowness: 1 in the background, 3 inside the text
f = (canvas / 255.0) * 2.0 + 1.0

u_init = np.full((canvasHeight, canvasWidth), np.inf, dtype=np.float32)
u_init[canvasHeight // 2, 0] = 0.0 # Start from middle-left edge

u_init_t = torch.tensor(u_init, device=device)
f_t = torch.tensor(f, device=device)

print("Running Eikonal Solver...")
with torch.no_grad():
    u_t = eiko(u_init_t, f_t, dx=1.0, msfm=True)

# Pull back to CPU to establish bounds
u = u_t.cpu().numpy()
valid_u = u[np.isfinite(u)]
maxv = np.max(valid_u)

# --- 3. Animation Setup ---
pulse_width = 80.0
freq = 3 * np.pi

# Pad infinites so they don't break the animation math
u_safe = np.copy(u)
u_safe[np.isinf(u_safe)] = maxv + pulse_width

# Setup figure
fig, ax = plt.subplots(figsize=(10, 3.5), facecolor='white')
fig.suptitle('TIME OF FLIGHT CALCULATOR', fontsize=16, fontweight='bold', fontname='monospace')
ax.axis('off')

# inverted gray colormap matches MATLAB's 1-gray
im = ax.imshow(np.zeros_like(u), vmin=0, vmax=1.0, cmap='gray_r', aspect='equal')
plt.tight_layout()

# --- 4. Pure PyTorch Animation Engine ---
# Keep state variables on GPU
u_safe_gpu = torch.tensor(u_safe, device=device)
text_mask = f > 1.0
text_base = gaussian_filter(text_mask.astype(np.float32), sigma=0.5) * 0.95
text_base_gpu = torch.tensor(text_base, device=device)

current_wave = torch.zeros_like(u_safe_gpu)
wave_history = torch.zeros_like(u_safe_gpu)
trail_decay = 0.0
dt = 0.5
time_steps = np.arange(0, maxv + pulse_width, dt)

# Precompute the global minimum using the exact same math
temp_x = np.linspace(-pulse_width/2, pulse_width/2, 400)
temp_window = np.hanning(400)
temp_y = np.cos(freq * temp_x / pulse_width) * temp_window
global_my = np.min(temp_y)

def compute_frame(t):
    global current_wave, wave_history
    
    diff_t = u_safe_gpu - t
    active_mask = torch.abs(diff_t) <= (pulse_width / 2.0)
    
    raw_wave = torch.zeros_like(u_safe_gpu)
    
    if torch.any(active_mask):
        d_val = diff_t[active_mask]
        
        # Analytical Gabor patch
        window = 0.5 * (1.0 + torch.cos(2 * np.pi * d_val / pulse_width))
        carrier = torch.cos(freq * d_val / pulse_width)
        packet = window * carrier
        
        # Apply the exact global normalization
        packet = (packet + global_my) / (1.0 + global_my)
        
        raw_wave[active_mask] = packet
        
    # Track history and blend
    wave_history = torch.maximum(wave_history, raw_wave)
    current_wave = torch.maximum(raw_wave, current_wave * trail_decay)
    
    visible_text = text_base_gpu * wave_history
    final_img = torch.maximum(current_wave, visible_text)
    
    return final_img.cpu().numpy()

def update(frame_idx):
    t = time_steps[frame_idx]
    im.set_data(compute_frame(t))
    return [im]

print("Rendering Animation...")
anim = animation.FuncAnimation(fig, update, frames=len(time_steps), interval=1000/60, blit=True)
plt.show()