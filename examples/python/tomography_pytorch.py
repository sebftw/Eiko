# To run this script you must install eiko with the optional example dependencies.
# This is done using "pip install eiko[examples]".

import sys
import time

# Check if the optional dependency is available
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("\n[Eiko] Error: This example script requires 'matplotlib' for visualization.", file=sys.stderr)
    print("Please install it by running: pip install \"eiko[examples]\"\n", file=sys.stderr)
    sys.exit(1)

import torch
import numpy as np

# Import Eiko
from eiko import eiko

# =========================================================================
# 1. Setup Grid and True Slowness Model
# =========================================================================
# Force execution on the GPU to interface with the custom CUDA solver
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    raise RuntimeError("CUDA device required for fim_cuda solver.")

N = 64
dx = 1.0
msfm = False

# Create true slowness (f) natively on the GPU
true_f = torch.ones((N, N), dtype=torch.float32, device=device)
true_f[24:40, 24:40] = 2.0  # Python is 0-indexed (MATLAB 25:40 -> 24:40)

# =========================================================================
# 2. Setup Sources (Batched 2D: [Batch, H, W])
# =========================================================================
src_coords = torch.tensor([
    [4, 4],    # Top-Left
    [4, 59],   # Top-Right
    [59, 4],   # Bottom-Left
    [59, 59]   # Bottom-Right
], device=device)

# PyTorch requires batch first: [Batch, H, W]
u_init = torch.full((4, N, N), 1e6, dtype=torch.float32, device=device)

# Seed the sources with 0.0
for b in range(4):
    u_init[b, src_coords[b, 0], src_coords[b, 1]] = 0.0

# =========================================================================
# 3. Generate "Measured" Data (Target Traveltimes)
# =========================================================================
print('Generating target traveltime data...')
with torch.no_grad():
    T_measured = eiko(u_init, true_f, dx=dx, msfm=msfm)

# =========================================================================
# 4. Optimization Setup
# =========================================================================
# Initial guess: A completely homogeneous field of 1.0. (m = ln(f))
m_guess = torch.zeros((N, N), dtype=torch.float32, device=device, requires_grad=True)

learning_rate = 0.04
num_iters = 100

# Preallocate history lists (keeping tensors on GPU during loop to avoid syncs)
loss_history = []
f_history = []

# Initialize standard PyTorch ADAM optimizer
optimizer = torch.optim.Adam([m_guess], lr=learning_rate)

# =========================================================================
# 5. Pure Computation Loop (No Graphics, No D2H Transfers)
# =========================================================================
print('Starting ADAM inversion (Pure GPU Computation)...')

# Force the CPU to wait for the GPU's asynchronous queue to clear before starting the clock
torch.cuda.synchronize()
opt_timer_start = time.perf_counter()

for iter in range(num_iters):
    
    optimizer.zero_grad()
    
    # Forward Pass: Enforce positivity by mapping latent to physical
    f_pred = torch.exp(m_guess)
    T_pred = eiko(u_init, f_pred, dx=dx, msfm=msfm)
    
    # Compute Mean Squared Error Loss
    loss = 0.5 * torch.mean((T_pred - T_measured)**2)
    
    # Backward Pass
    loss.backward()
    
    # Mask out gradients at the source locations
    with torch.no_grad():
        m_guess.grad[src_coords[:, 0], src_coords[:, 1]] = 0.0
        
    optimizer.step()
    
    # Track loss and physical field (clone keeps a copy, detach removes autograd tape)
    # We leave them as GPU tensors here to avoid stalling the CUDA queue.
    loss_history.append(loss.detach())
    f_history.append(f_pred.detach().clone())

# Force the CPU to wait for the GPU to actually finish computing before stopping the clock
torch.cuda.synchronize()
elapsed_time = time.perf_counter() - opt_timer_start

print(f"Math finished in {elapsed_time:.4f} seconds ({num_iters/elapsed_time:.1f} iterations/sec).")

# =========================================================================
# 6. Visualization Loop
# =========================================================================
print('Rendering visualization...')

# Bring histories back to CPU RAM once, in a single block transfer
with torch.no_grad():
    loss_history_cpu = torch.stack(loss_history).cpu().numpy()
    f_history_cpu = torch.stack(f_history).cpu().numpy()
    true_f_cpu = true_f.cpu().numpy()

# Setup Figure
mseplot = False
cols = 3 if mseplot else 2
fig, axes = plt.subplots(1, cols, figsize=(15 if mseplot else 10, 5), facecolor='white')
if cols == 2:
    axes = list(axes) + [None] # Pad if missing mseplot axis

fig.suptitle(f'Traveltime Tomography Inversion\nFinished in {elapsed_time:.2f} seconds ({num_iters} iterations).', fontsize=14, fontweight='bold')

plt.ion() # Turn on interactive mode for animation

for iter in range(num_iters):
    current_f = f_history_cpu[iter]
    
    if iter == 0:
        # Axes 1: True Model (Converted to Sound Speed)
        im1 = axes[0].imshow(1540.0 / true_f_cpu, cmap='viridis_r', vmin=1540/2, vmax=1540/1)
        axes[0].set_title('True Sound Speed', fontsize=12)
        axes[0].axis('off')
        
        # Axes 2: Recovered Model
        im2 = axes[1].imshow(1540.0 / current_f, cmap='viridis_r', vmin=1540/2, vmax=1540/1)
        title2 = axes[1].set_title(f'Recovered Sound Speed (Iter {iter+1})', fontsize=12)
        axes[1].axis('off')
        
        # Add a single colorbar for the recovered plot
        cb = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cb.set_label('Sound Speed [m/s]')
        
        if mseplot:
            # Axes 3: Loss Curve
            line3, = axes[2].plot(range(1, num_iters + 1), loss_history_cpu, linewidth=2.5, color='#0072BD')
            axes[2].set_xlim([1, num_iters])
            axes[2].set_title('Optimization Loss', fontsize=12)
            axes[2].set_xlabel('Iteration')
            axes[2].set_ylabel('MSE Loss')
            axes[2].grid(True)
            
        plt.tight_layout()
    else:
        # Update plot data efficiently (matching MATLAB plt2.CData = ...)
        title2.set_text(f'Recovered Sound Speed (Iter {iter+1})')
        im2.set_data(1540.0 / current_f)
    
    # Animate as fast as matplotlib can render
    plt.pause(0.01)

plt.ioff()
print('Visualization complete!')
plt.show()