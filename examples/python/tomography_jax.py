# To run this script you must install eiko with the optional example dependencies.
# This is done using "pip install eiko[examples]".

import sys
import time
from functools import partial

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
# Guard against missing dependencies
# =========================================================================
missing_deps = []
err = None
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    missing_deps.append("matplotlib")
    err = e

try:
    import jax
    import jax.numpy as jnp
    import optax
except ImportError as e:
    missing_deps.append("jax/optax")
    err = e

if missing_deps:
    raise ImportError(
        f"\n\n[Eiko] This example script cannot run because the following components are missing: {', '.join(missing_deps)}.\n"
        f" Please install the required environment via: pip install \"eiko[examples]\"\n"
    ) from err

# Import Eiko
from eiko import eiko

# =========================================================================
# 1. Setup Grid and True Slowness Model
# =========================================================================
N = 64 
dx = 1.0
msfm = False

# Create true slowness (f): Background of 1.0, with a slow anomaly of 2.0
true_f = np.ones((N, N), dtype=np.float32)
true_f[24:40, 24:40] = 2.0  
true_f = jnp.array(true_f, dtype=jnp.float32)

# =========================================================================
# 2. Setup Sources (Batched 2D: [Batch, H, W])
# =========================================================================
u_init = np.full((4, N, N), np.inf, dtype=np.float32)
src_coords = jnp.array([
    [4, 4],    # Top-Left 
    [4, 59],   # Top-Right
    [59, 4],   # Bottom-Left
    [59, 59]   # Bottom-Right
])

# Seed the sources with 0.0
for b, (r, c) in enumerate(src_coords):
    u_init[b, r, c] = 0.0
u_init = jnp.array(u_init, dtype=jnp.float32)

# =========================================================================
# 3. Generate "Measured" Data (Target Traveltimes)
# =========================================================================
print('Generating target traveltime data...')
T_measured = eiko(u_init, true_f, dx=dx, msfm=msfm)

# =========================================================================
# 4. Optimization Setup
# =========================================================================
f_guess = np.ones((N, N), dtype=np.float32)
m_guess = jnp.log(jnp.array(f_guess, dtype=jnp.float32))

learning_rate = 0.04
num_iters = 100

optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(m_guess)

# Preallocate histories 
loss_history = []
f_history = []

# =========================================================================
# 5. Loss Function and Update Step
# =========================================================================
def eikonal_inversion_loss(m_latent, u_init_batch, T_target, dx, msfm):
    f_pred = jnp.exp(m_latent)
    T_pred = eiko(u_init_batch, f_pred, dx=dx, msfm=msfm)
    mse_diff = T_pred - T_target
    return 0.5 * jnp.mean(mse_diff**2) 

@partial(jax.jit, static_argnames=("dx", "msfm"))
def update_step(m_latent, state, u_init_batch, T_target, src_idx, dx, msfm):
    loss, grad_m = jax.value_and_grad(eikonal_inversion_loss)(m_latent, u_init_batch, T_target, dx, msfm)
    
    rows = src_idx[:, 0]
    cols = src_idx[:, 1]
    grad_m_masked = grad_m.at[rows, cols].set(0.0)
    
    updates, new_state = optimizer.update(grad_m_masked, state)
    new_m = optax.apply_updates(m_latent, updates)
    
    return new_m, new_state, loss

# =========================================================================
# 6. Pure Computation Loop (No Graphics, No D2H Transfers)
# =========================================================================
print('JIT Compiling and warming up XLA backend...')
# Dummy run to compile the graph so it isn't included in our timing benchmark
dummy_m, _, dummy_loss = update_step(m_guess, opt_state, u_init, T_measured, src_coords, dx, msfm)
jax.block_until_ready(dummy_loss)

print('Starting ADAM inversion (Pure GPU Computation)...')

# Start the clock only after JAX async queues are cleared
opt_timer_start = time.perf_counter()

for iter in range(num_iters):
    print(iter)
    m_guess, opt_state, current_loss = update_step(
        m_guess, opt_state, u_init, T_measured, src_coords, dx, msfm
    )
    # Append JAX DeviceArrays directly to keep them on the GPU
    loss_history.append(current_loss)
    f_history.append(jnp.exp(m_guess))

# Force the CPU to wait for the GPU to actually finish computing
jax.block_until_ready(f_history[-1])
elapsed_time = time.perf_counter() - opt_timer_start

print(f"Math finished in {elapsed_time:.4f} seconds ({num_iters/elapsed_time:.1f} iterations/sec).")

# =========================================================================
# 7. Visualization Loop
# =========================================================================
print('Rendering visualization...')

# Bring histories back to CPU RAM natively
f_history_cpu = np.array(f_history)
loss_history_cpu = np.array(loss_history)
true_f_cpu = np.array(true_f)

mseplot = False
cols = 3 if mseplot else 2
fig, axes = plt.subplots(1, cols, figsize=(15 if mseplot else 10, 5), facecolor='white')
if cols == 2:
    axes = list(axes) + [None] 

fig.suptitle(f'Traveltime Tomography Inversion\nFinished in {elapsed_time:.2f} seconds ({num_iters} iterations).', fontsize=14, fontweight='bold')
plt.ion() 

for iter in range(num_iters):
    current_f = f_history_cpu[iter]
    
    if iter == 0:
        im1 = axes[0].imshow(1540.0 / true_f_cpu, cmap='viridis_r', vmin=1540/2, vmax=1540/1)
        axes[0].set_title('True Sound Speed', fontsize=12)
        axes[0].axis('off')
        
        im2 = axes[1].imshow(1540.0 / current_f, cmap='viridis_r', vmin=1540/2, vmax=1540/1)
        title2 = axes[1].set_title(f'Recovered Sound Speed (Iter {iter+1})', fontsize=12)
        axes[1].axis('off')
        
        cb = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cb.set_label('Sound Speed [m/s]')
        
        if mseplot:
            line3, = axes[2].plot(range(1, num_iters + 1), loss_history_cpu, linewidth=2.5, color='#0072BD')
            axes[2].set_xlim([1, num_iters])
            axes[2].set_title('Optimization Loss', fontsize=12)
            axes[2].set_xlabel('Iteration')
            axes[2].set_ylabel('MSE Loss')
            axes[2].grid(True)
            
        plt.tight_layout()
    else:
        title2.set_text(f'Recovered Sound Speed (Iter {iter+1})')
        im2.set_data(1540.0 / current_f)
    
    plt.pause(0.01)

plt.ioff()
print('Visualization complete!')
plt.show()