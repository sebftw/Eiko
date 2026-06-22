import time
import torch
import cupy as cp
import cupyx.scipy.ndimage
import skfmm
import numpy as np
import matplotlib.pyplot as plt

import fteikpy
import pykonal
from eiko import eiko

MSFM = False  # Whether to enable MSFM for eiko (increases accuracy).

# ================================================
# fteikpy API Detection (Runs once)
# ================================================
try:
    _FTEIK_SOLVER_2D = fteikpy.Eikonal2D
    try:
        _FTEIK_SOLVER_2D(np.ones((2, 2), dtype=np.float64), gridsize=(1.0, 1.0))
        _FTEIK_KWARGS_2D = {"gridsize": (1.0, 1.0)}
    except TypeError:
        _FTEIK_KWARGS_2D = {"grid_spacing": (1.0, 1.0)}
except AttributeError:
    _FTEIK_SOLVER_2D = None

# ==========================================
# JAX 2D Gauss-Seidel Algorithm (JAX-GS)
# ==========================================
try:
    import jax
    import jax.numpy as jnp
    _JAX_PLATFORM = jax.devices()[0].platform
    HAS_JAX_GPU = (_JAX_PLATFORM in ['gpu', 'cuda', 'rocm'])
except (ImportError, Exception):
    HAS_JAX_GPU = False

if HAS_JAX_GPU:
    @jax.jit
    def _jax_red_black_eikonal(f: jax.Array, src_idx: tuple) -> jax.Array:
        """
        Solves the 2D Eikonal equation |grad(phi)| = f using a parallelized 
        Red-Black Gauss-Seidel relaxation scheme.
        
        To maintain high GPU utilization, the grid is split into an alternating
        checkerboard pattern. Since the 4-point first-order upwind stencil only
        depends on orthogonal neighbors, all 'Red' nodes can be updated completely
        in parallel using existing 'Black' states, and vice versa. This breaks the
        sequential loop dependency while maintaining the faster convergence rates
        inherent to Gauss-Seidel updates.

        Parameters
        ----------
        f : jax.Array
            A 2D single-precision float32 array of shape (N, N) representing 
            the slowness field (reciprocal of velocity) across the domain grid.
        src_idx : tuple of int
            A tuple containing the (i, j) coordinates of the point source initialization.

        Returns
        -------
        jax.Array
            A 2D single-precision float32 array of shape (N, N) containing the
            computed traveltime/distance field map.
        """
        n = f.shape[0]
        
        # Red-Black updates propagate wavefront information twice as fast per iteration
        # as standard Jacobi. A value of ~1.2 * N provides sufficient headroom for a 
        # wavefront to clear the full diagonal of a homogeneous 2D Cartesian grid.
        max_iters = int(n * 1.2) 
        
        # Initialize the solution grid with an approximation of infinity (1e6)
        phi_init = jnp.full((n, n), 1e6, dtype=jnp.float32)
        # The source location serves as the Dirichlet boundary condition (phi = 0)
        phi_init = phi_init.at[src_idx[0], src_idx[1]].set(0.0)
        
        # Pre-compute static boolean index masks for the checkerboard layout.
        # Nodes where (i + j) is even are marked as Red; odd nodes are Black.
        i_indices, j_indices = jnp.ogrid[:n, :n]
        red_mask = ((i_indices + j_indices) % 2 == 0)
        black_mask = ~red_mask

        def update_half_grid(phi: jax.Array, target_mask: jax.Array) -> jax.Array:
            """
            Applies a vectorized upwind Eikonal stencil calculation across the grid,
            selectively writing updates only to nodes matching the target boolean mask.
            """
            # Vectorized Neighbor Extraction via Array Shifting:
            # Instead of pixel-by-pixel loops, we shift the entire grid in 4 directions.
            # Boundaries are padded with 1e6 to act as absorbing/infinite walls.
            p_up = jnp.pad(phi[:-1, :], ((1, 0), (0, 0)), constant_values=1e6)
            p_dn = jnp.pad(phi[1:, :], ((0, 1), (0, 0)), constant_values=1e6)
            p_lt = jnp.pad(phi[:, :-1], ((0, 0), (1, 0)), constant_values=1e6)
            p_rt = jnp.pad(phi[:, 1:], ((0, 0), (0, 1)), constant_values=1e6)
            
            # Upwind Selectors: Choose the minimum neighboring value along each axis.
            # This models characteristic characteristics pointing backward along ray paths.
            phi_x = jnp.minimum(p_up, p_dn)
            phi_y = jnp.minimum(p_lt, p_rt)
            
            # Stencil Mathematics (Solving the Upwind Godunov Scheme):
            # We solve the quadratic equation: (phi - phi_x)^2 + (phi - phi_y)^2 = f^2
            diff = jnp.abs(phi_x - phi_y)
            
            # Condition 1: Wavefront propagates primarily along one dominant axis.
            # If the absolute difference is greater than the slowness cost 'f', 
            # the characteristic line falls outside the local coordinate quadrant.
            phi_new_1 = jnp.minimum(phi_x, phi_y) + f
            
            # Condition 2: Wavefront propagates diagonally across both axes.
            # jnp.maximum handles precision underflows, preventing negative values inside the sqrt.
            discriminant = jnp.maximum(2.0 * f**2 - diff**2, 0.0)
            phi_new_2 = 0.5 * (phi_x + phi_y + jnp.sqrt(discriminant))
            
            # Blend the two solution modes depending on the local gradient condition
            phi_step = jnp.where(diff < f, phi_new_2, phi_new_1)
            
            # Maintain the causality/monotonicity constraint (values should only decrease)
            phi_step = jnp.minimum(phi, phi_step)
            
            # Merge step: Commit updates for active mask elements, keep old states for others
            return jnp.where(target_mask, phi_step, phi)

        def body_fn(step: int, phi: jax.Array) -> jax.Array:
            """
            Main loop body executing one full Gauss-Seidel relaxation pass.
            """
            # Pass A: Update Red nodes using the most up-to-date Black states
            phi = update_half_grid(phi, red_mask)
            phi = phi.at[src_idx[0], src_idx[1]].set(0.0) # Clamp source to 0
            
            # Pass B: Update Black nodes using the *newly computed* Red states
            phi = update_half_grid(phi, black_mask)
            return phi.at[src_idx[0], src_idx[1]].set(0.0) # Clamp source to 0
            
        # lax.fori_loop prevents loop unrolling, creating a highly compact XLA execution graph
        return jax.lax.fori_loop(0, max_iters, body_fn, phi_init)


    def run_jax_fsm_batched_wrapper(f_batch: jax.Array, src_coords: tuple) -> jax.Array:
        #Uses jax.vmap to map the underlying 2D solver across the 0-axis of the input 
        #batch, pushing the execution into a single unified data-parallel GPU kernel.
        batched_solver = jax.vmap(_jax_red_black_eikonal, in_axes=(0, None))
        return batched_solver(f_batch, src_coords)

    _run_jax_fsm_batched = run_jax_fsm_batched_wrapper


# ==========================================
# Data Generation (2D)
# ==========================================
def create_batched_data_2d(batch_size=16, size=256):
    print(f"Generating batch of {batch_size} grids ({size}x{size})...")
    
    # GPU Native (float32)
    f_tensor = torch.ones((batch_size, 1, size, size), dtype=torch.float32, device='cuda')
    u_init = torch.full((batch_size, 1, size, size), 1e6, dtype=torch.float32, device='cuda')
    u_init[:, 0, size//2, size//2] = 0.0
    
    cp_list = []
    np_list = []
    vel_list = []
    
    jax_f_batch = jnp.ones((batch_size, size, size), dtype=jnp.float32) if HAS_JAX_GPU else None
    jax_src = (size // 2, size // 2)
    
    for _ in range(batch_size):
        # CuPy (float32)
        mask_cp = cp.ones((size, size), dtype=cp.float32)
        mask_cp[size//2, size//2] = 0.0
        cp_list.append(mask_cp)
        
        # scikit-fmm (float64)
        mask_sk = np.ones((size, size), dtype=np.float64)
        mask_sk[size//2, size//2] = -1.0
        np_list.append(mask_sk)
        
        # fteikpy & pykonal (float64)
        vel_np = np.ones((size, size), dtype=np.float64)
        vel_list.append(vel_np)
    
    # ==========================================
    # Generate Pre-initialized Data for eiko
    # ==========================================
    # Start with an empty grid of infinities
    u_init_pre = torch.full((BATCH_SIZE, 1, GRID_SIZE, GRID_SIZE), float('inf'), dtype=torch.float32, device='cuda')
    
    # Calculate the exact geometric distance from the center
    src_idx = GRID_SIZE // 2
    y_idx, x_idx = torch.meshgrid(torch.arange(GRID_SIZE, device='cuda'), torch.arange(GRID_SIZE, device='cuda'), indexing='ij')
    dist = torch.sqrt((x_idx - src_idx).float()**2 + (y_idx - src_idx).float()**2)
    
    # Create a 15-pixel radius patch
    patch_mask = dist <= 15.0
    
    # Inject the analytical times. 
    u_init_pre[:, 0, patch_mask] = dist[patch_mask] * f_tensor[0, 0, size//2, size//2]
    
    return u_init, u_init_pre, f_tensor, cp_list, np_list, vel_list, jax_f_batch, jax_src

# ==========================================
# Benchmark Logic
# ==========================================
def benchmark(name, batch_size, target_func, *args, runs=50, backend='pytorch', error_log=None):
    is_gpu = backend in ['pytorch', 'cupy', 'jax']
    
    # --- WARMUP ---
    try:
        for _ in range(3):
            _ = target_func(*args)
    except Exception as e:
        print(f"{name:>20} | {'FAILED TO RUN':>12}")
        if error_log is not None:
            error_log.append((name, str(e)))
        return None
    
    if is_gpu:
        torch.cuda.synchronize()
    
    # --- TIMING LOOP ---
    start_time = time.perf_counter()
    
    if is_gpu:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(runs):
            res = target_func(*args)
        
        if backend == 'jax':
            res.block_until_ready()
        end_event.record()
        
        torch.cuda.synchronize()
        total_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = total_time_ms / runs
    else:
        start_time = time.perf_counter()
        for _ in range(runs):
            _ = target_func(*args)
        end_time = time.perf_counter()
        avg_time_ms = ((end_time - start_time) / runs) * 1000
    
    fps = (batch_size * 1000) / avg_time_ms
    print(f"{name:>20} | {avg_time_ms:>8.3f} ms/batch | {fps:>10.1f} frames/s")
    
    # RETURN TUPLE MODIFIED HERE
    return avg_time_ms, fps


# ==========================================
# 2D Solver Wrappers
# ==========================================
def run_eiko_batched_2d(u_init, f):
    return eiko(u_init.squeeze(1), f.squeeze(1), msfm=MSFM)

def run_cupy_looped_2d(cp_data_list):
    return [cupyx.scipy.ndimage.distance_transform_edt(t) for t in cp_data_list]

def run_skfmm_looped_2d(np_data_list):
    return [skfmm.distance(t) for t in np_data_list]

def run_fteikpy_looped_2d(vel_data_list, size):
    if _FTEIK_SOLVER_2D is None:
        raise RuntimeError("fteikpy 2D module unavailable")
    
    results = []
    source_coords = (float(size // 2), float(size // 2))
    for v in vel_data_list:
        solver = _FTEIK_SOLVER_2D(v, **_FTEIK_KWARGS_2D)
        results.append(solver.solve(source_coords))
        
    return results

def run_pykonal_looped_2d(vel_data_list, size):
    results = []
    src = size // 2
    for v in vel_data_list:
        solver = pykonal.EikonalSolver(coord_sys="cartesian")
        solver.velocity.min_coords = 0, 0, 0
        solver.velocity.node_intervals = 1, 1, 1
        solver.velocity.npts = size, size, 1 
        solver.velocity.values = v.reshape(size, size, 1)
        
        solver.traveltime.values[src, src, 0] = 0.0
        solver.unknown[src, src, 0] = False
        solver.trial.push(src, src, 0)
        
        solver.solve()
        results.append(solver.traveltime.values[:, :, 0])
    return results

def run_jax_wrapper(f_batch, src_coords):
    return _run_jax_fsm_batched(f_batch, src_coords)

# ==========================================
# Main Execution & Plotting
# ==========================================
if __name__ == "__main__":
    BATCH_SIZE = 32
    GRID_SIZE = 512
    NUM_RUNS = 20
    NUM_RUNS_CPU = 3
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "No CUDA GPU Detected"
    
    u_init, u_pre_init, f_tensor, cp_list, np_list, vel_list, jax_f, jax_src = create_batched_data_2d(BATCH_SIZE, GRID_SIZE)
    
    print(f"GPU Device: {gpu_name}")
    print("-" * 65)
    print(f"2D Benchmarking with Batch: {BATCH_SIZE}, Grid: {GRID_SIZE}^2, Runs: {NUM_RUNS} ({NUM_RUNS_CPU} on CPU)")
    print("-" * 65)
    
    error_log = []
    fps_results = {}
    
    # GPU Benchmarks
    res = benchmark("eiko", BATCH_SIZE, run_eiko_batched_2d, u_init, f_tensor, runs=NUM_RUNS, backend='pytorch', error_log=error_log)
    if res: fps_results["eiko"] = res[1]
    
    res = benchmark("CuPy EDT (f=1 only)", BATCH_SIZE, run_cupy_looped_2d, cp_list, runs=NUM_RUNS, backend='cupy', error_log=error_log)
    if res: fps_results["CuPy EDT (f=1 only)"] = res[1]
    
    if HAS_JAX_GPU:
        res = benchmark("JAX GS", BATCH_SIZE, run_jax_wrapper, jax_f, jax_src, runs=NUM_RUNS, backend='jax', error_log=error_log)
        if res: fps_results["JAX GS"] = res[1]
    else:
        print(f"{'JAX GS':>20} | {'SKIPPED (No JAX with GPU support was detected)':>31}")
    
    print("-" * 65)
    # CPU Benchmarks
    res = benchmark("scikit-fmm", BATCH_SIZE, run_skfmm_looped_2d, np_list, runs=NUM_RUNS_CPU, backend='cpu', error_log=error_log)
    if res: fps_results["scikit-fmm"] = res[1]
        
    res = benchmark("fteikpy", BATCH_SIZE, run_fteikpy_looped_2d, vel_list, GRID_SIZE, runs=NUM_RUNS_CPU, backend='cpu', error_log=error_log)
    if res: fps_results["fteikpy"] = res[1]
        
    res = benchmark("pykonal", BATCH_SIZE, run_pykonal_looped_2d, vel_list, GRID_SIZE, runs=NUM_RUNS_CPU, backend='cpu', error_log=error_log)
    if res: fps_results["pykonal"] = res[1]
    print("-" * 65)
    
    if error_log:
        print("\n" + "=" * 65)
        print("ERROR LOG")
        print("=" * 65)
        for name, err_msg in error_log:
            print(f"[{name}]\n  {err_msg}\n")
        print("=" * 65)

    # ==========================================
    # Results Evaluation & Plotting
    # ==========================================
    print("\nCollecting solver outputs for error evaluation...")
    outputs = {}
    
    try: outputs["eiko"] = run_eiko_batched_2d(u_init, f_tensor)[0].cpu().numpy()
    except Exception: pass
    
    try: outputs["eiko (w. pre-init.)"] = run_eiko_batched_2d(u_pre_init, f_tensor)[0].cpu().numpy()
    except Exception: pass
    
    # Comparing accuracy against CuPy EDT is not fair, since EDT does not support variable speed of sound.
    # try: outputs["CuPy EDT"] = run_cupy_looped_2d(cp_list)[0].get()
    # except Exception: pass

    if HAS_JAX_GPU:
        try: outputs["JAX GS"] = np.array(run_jax_wrapper(jax_f, jax_src)[0])
        except Exception: pass

    try: outputs["scikit-fmm"] = run_skfmm_looped_2d(np_list)[0]
    except Exception: pass

    try:
        out_ft = run_fteikpy_looped_2d(vel_list[:1], GRID_SIZE)[0]
        ft_grid = out_ft.grid if hasattr(out_ft, 'grid') else np.array(out_ft)
        # Slicing the (N+1, N+1) node grid back down to match the (N, N) cell grid
        outputs["fteikpy"] = ft_grid[:GRID_SIZE, :GRID_SIZE]
    except Exception: pass

    try: outputs["pykonal"] = run_pykonal_looped_2d(vel_list[:1], GRID_SIZE)[0]
    except Exception: pass

    # Ground truth (Analytical solution for f=1 point source)
    src_idx = GRID_SIZE // 2
    X, Y = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE), indexing='ij')
    truth = np.sqrt((X - src_idx)**2 + (Y - src_idx)**2)

    # ------------------
    # PLOT 1: FPS BAR CHART
    # ------------------
    if fps_results:
        # 1. Sort data by performance so the chart tells a clear story
        sorted_results = sorted(fps_results.items(), key=lambda x: x[1])
        names, vals = zip(*sorted_results)
        
        # 2. Modern styling configuration
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['text.color'] = '#2c3e50'
        plt.rcParams['axes.labelcolor'] = '#2c3e50'
        plt.rcParams['xtick.color'] = '#7f8c8d'
        plt.rcParams['ytick.color'] = '#2c3e50'

        fig, ax = plt.subplots(figsize=(10, 5.5))
        
        # 3. Use horizontal bars with a single cohesive, professional color
        # (Multiple colors usually imply different data categories, which isn't the case here)
        bars = ax.barh(names, vals, color='#2980b9', edgecolor='none', height=0.6)
        
        # 4. Clean up the frame (despining)
        for spine in ['top', 'right', 'bottom']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color('#bdc3c7')
        
        # 5. Add a subtle background grid for context
        ax.grid(axis='x', linestyle='--', alpha=0.5, color='#bdc3c7')
        ax.set_axisbelow(True) # Ensure grid sits behind the bars
        
        # Labels and Titles
        ax.set_xlabel('Processing Rate (Frames per Second)', fontsize=11, fontweight='bold', labelpad=10)
        ax.set_title('Eikonal Solver Speed Comparison', fontsize=14, fontweight='bold', pad=20, loc='left')
        
        # 6. Clean, readable inline value labels
        for bar in bars:
            xval = bar.get_width()
            ax.text(
                xval + (max(vals) * 0.01), # Tiny offset to place text neatly past the bar end
                bar.get_y() + bar.get_height()/2, 
                f'{xval:,.0f}', 
                ha='left', 
                va='center', 
                fontsize=10, 
                fontweight='semibold',
                color='#2c3e50'
            )
            
        plt.tight_layout()
        plt.savefig('fps_comparison.png', dpi=300) # 300 DPI for a crisp export

    # ------------------
    # PLOT 2: ERROR HEATMAPS
    # ------------------
    if outputs:
        cols = 3
        rows = (len(outputs) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = np.atleast_1d(axes).flatten()
        
        for idx, (name, out_arr) in enumerate(outputs.items()):
            # Calculate absolute error against truth
            error = np.abs(out_arr - truth)
            
            im = axes[idx].imshow(error, cmap='magma', origin='upper')
            axes[idx].set_title(f"{name} (Absolute Error)")
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04, label="Error (Distance)")
            
        # Clean up empty subplots
        for i in range(len(outputs), len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        plt.savefig('error_comparison.png', dpi=300)
        plt.show()
