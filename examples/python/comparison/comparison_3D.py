import time
import torch
from eiko import eiko3d

import drjit as dr
import fastsweep
import piefs
import cupy as cp
import cupyx.scipy.ndimage
import skfmm
import numpy as np
import fteikpy
import pykonal

# ==========================================
# Optional Library Gating (JAX)
# ==========================================
try:
    import jax
    import jax.numpy as jnp
    _JAX_PLATFORM = jax.devices()[0].platform
    HAS_JAX_GPU = (_JAX_PLATFORM in ['cuda', 'gpu', 'rocm'])
except (ImportError, Exception):
    HAS_JAX_GPU = False

# ==========================================
# fteikpy API Detection (Runs once)
# ==========================================
if hasattr(fteikpy, "Eikonal3D"):
    _FTEIK_SOLVER = fteikpy.Eikonal3D
    _FTEIK_KWARGS = {"gridsize": (1.0, 1.0, 1.0)}
else:
    _FTEIK_SOLVER = fteikpy.Eikonal
    _FTEIK_KWARGS = {"grid_spacing": (1.0, 1.0, 1.0)}

# ==========================================
# JAX 3D Red-Black Gauss-Seidel Eikonal
# ==========================================
@jax.jit
def _jax_red_black_eikonal_3d_opt(f, src_idx):
    n = f.shape[0]
    
    # Cap iterations to a sensible limit for Fast Sweeping (e.g., 20)
    max_iters = 20  
    tol = 1e-4      # Convergence tolerance
    
    phi_init = jnp.full((n, n, n), 1e6, dtype=jnp.float32)
    phi_init = phi_init.at[src_idx[0], src_idx[1], src_idx[2]].set(0.0)
    
    # 3D Checkerboard mask
    i_idx, j_idx, k_idx = jnp.ogrid[:n, :n, :n]
    red_mask = ((i_idx + j_idx + k_idx) % 2 == 0)
    black_mask = ~red_mask

    def update_half_grid(phi, target_mask):
        # Pad ONCE and use zero-copy slices instead of 6 separate pad operations
        phi_pad = jnp.pad(phi, 1, constant_values=1e6)
        
        p_up = phi_pad[:-2, 1:-1, 1:-1]
        p_dn = phi_pad[2:, 1:-1, 1:-1]
        p_lt = phi_pad[1:-1, :-2, 1:-1]
        p_rt = phi_pad[1:-1, 2:, 1:-1]
        p_ft = phi_pad[1:-1, 1:-1, :-2]
        p_bk = phi_pad[1:-1, 1:-1, 2:]
        
        # Upwind differences
        v1 = jnp.minimum(p_up, p_dn)
        v2 = jnp.minimum(p_lt, p_rt)
        v3 = jnp.minimum(p_ft, p_bk)
        
        # Sort values: a <= b <= c
        a = jnp.minimum(v1, jnp.minimum(v2, v3))
        c = jnp.maximum(v1, jnp.maximum(v2, v3))
        b = (v1 + v2 + v3) - a - c
        
        # 1D, 2D, and 3D Godunov upwind solutions
        u1 = a + f
        
        discriminant_2d = jnp.maximum(2.0 * f**2 - (a - b)**2, 0.0)
        u2 = 0.5 * (a + b + jnp.sqrt(discriminant_2d))
        
        sum_v = a + b + c
        sum_sq = a**2 + b**2 + c**2
        discriminant_3d = jnp.maximum(4.0 * sum_v**2 - 12.0 * (sum_sq - f**2), 0.0)
        u3 = (2.0 * sum_v + jnp.sqrt(discriminant_3d)) / 6.0
        
        # Select correct dimensionality solution
        phi_new = jnp.where(u1 <= b, u1, jnp.where(u2 <= c, u2, u3))
        phi_step = jnp.minimum(phi, phi_new)
        
        return jnp.where(target_mask, phi_step, phi)

    # While_loop for early exit when the grid stops changing
    def cond_fn(state):
        step, _, max_diff = state
        return (step < max_iters) & (max_diff > tol)

    def body_fn(state):
        step, phi, _ = state
        phi_old = phi
        
        phi = update_half_grid(phi, red_mask)
        phi = phi.at[src_idx[0], src_idx[1], src_idx[2]].set(0.0)
        
        phi = update_half_grid(phi, black_mask)
        phi = phi.at[src_idx[0], src_idx[1], src_idx[2]].set(0.0)
        
        # Calculate maximum change across the grid
        max_diff = jnp.max(jnp.abs(phi - phi_old))
        
        return step + 1, phi, max_diff
        
    # Initial state: (step, phi, max_diff)
    initial_state = (0, phi_init, 1e6)
    
    # Run loop
    _, final_phi, _ = jax.lax.while_loop(cond_fn, body_fn, initial_state)
    
    return final_phi

_run_jax_fsm_batched = jax.vmap(_jax_red_black_eikonal_3d_opt, in_axes=(0, None))

# ==========================================
# Hybrid Precision Data Generation (3D)
# ==========================================
def create_batched_data_3d(batch_size=16, size=64):
    print(f"Generating batch of {batch_size} grids ({size}x{size}x{size})...")
    
    f_tensor = torch.ones((batch_size, 1, size, size, size), dtype=torch.float32, device='cuda')
    u_init = torch.full((batch_size, 1, size, size, size), 1e6, dtype=torch.float32, device='cuda')
    u_init[:, 0, size//2, size//2, size//2] = 0.0
    
    dr_list = []
    cp_list = []
    np_list = []
    vel_list = []
    
    jax_f_batch = jnp.ones((batch_size, size, size, size), dtype=jnp.float32) if HAS_JAX_GPU else None
    jax_src = (size // 2, size // 2, size // 2)
    
    for _ in range(batch_size):
        # Dr.Jit (float32)
        mask_np_dr = np.zeros((size, size, size), dtype=np.float32)
        mask_np_dr[size//2, size//2, size//2] = 1.0
        dr_t = dr.cuda.TensorXf(mask_np_dr)
        dr_t = dr.reshape(dr.cuda.TensorXf, dr_t, (size, size, size))
        dr_list.append(dr_t)
        
        # CuPy (float32)
        mask_cp = cp.ones((size, size, size), dtype=cp.float32)
        mask_cp[size//2, size//2, size//2] = 0.0
        cp_list.append(mask_cp)
        
        # scikit-fmm (float64)
        mask_sk = np.ones((size, size, size), dtype=np.float64)
        mask_sk[size//2, size//2, size//2] = -1.0
        np_list.append(mask_sk)

        # fteikpy & pykonal (float64)
        vel_np = np.ones((size, size, size), dtype=np.float64)
        vel_list.append(vel_np)
      
    return u_init, f_tensor, dr_list, cp_list, np_list, vel_list, jax_f_batch, jax_src

# ==========================================
# Benchmark Logic with CUDA Events
# ==========================================
def benchmark(name, batch_size, target_func, *args, runs=50, backend='pytorch', error_log=None):
    is_gpu = backend in ['pytorch', 'cupy', 'jax', 'drjit']
    
    # --- WARMUP ---
    try:
        for _ in range(min(3, runs)):
            res = target_func(*args)
            if backend == 'drjit':
                for r in res: dr.eval(r)
            elif backend == 'jax':
                res.block_until_ready()
                
        if backend == 'drjit':
            dr.sync_thread()
        if is_gpu:
            torch.cuda.synchronize()
    except Exception as e:
        print(f"{name:>20} | {'FAILED TO RUN':>12}")  
        if error_log is not None:
            error_log.append((name, str(e)))
        return None

    # --- TIMING LOOP ---
    if is_gpu:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(runs):
            res = target_func(*args)
            if backend == 'drjit':
                # Queue evaluation asynchronously
                for r in res: dr.eval(r) 
        
        # Enforce host barriers before recording the end event for JIT pipelines
        if backend == 'drjit':
            dr.sync_thread()
        elif backend == 'jax':
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
    return avg_time_ms

# ==========================================
# 3D Solver Wrappers
# ==========================================
def run_eiko_batched(u_init, f):
    return eiko3d(u_init, f)

def run_piefs_batched(u_init):
    return piefs.solve_eikonal(u_init, eps=1e-5, use_triton=True)

def run_fastsweep_looped(dr_data_list):
    return [fastsweep.redistance(t) for t in dr_data_list]

def run_cupy_looped(cp_data_list):
    return [cupyx.scipy.ndimage.distance_transform_edt(t) for t in cp_data_list]

def run_skfmm_looped(np_data_list):
    return [skfmm.distance(t) for t in np_data_list]

def run_fteikpy_looped(vel_data_list, size):
    results = []
    source_coords = (float(size // 2), float(size // 2), float(size // 2))
    for v in vel_data_list:
        solver = _FTEIK_SOLVER(v, **_FTEIK_KWARGS)
        results.append(solver.solve(source_coords))
    return results

def run_pykonal_looped(vel_data_list, size):
    results = []
    src = size // 2
    for v in vel_data_list:
        solver = pykonal.EikonalSolver(coord_sys="cartesian")
        solver.velocity.min_coords = 0, 0, 0
        solver.velocity.node_intervals = 1, 1, 1
        solver.velocity.npts = size, size, size
        solver.velocity.values = v
        solver.traveltime.values[src, src, src] = 0.0
        solver.unknown[src, src, src] = False
        solver.trial.push(src, src, src)
        solver.solve()
        results.append(solver.traveltime.values)
    return results

def run_jax_fsm_batched_wrapper(f_batch, src_coords):
    return _run_jax_fsm_batched(f_batch, src_coords)

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    BATCH_SIZE = 32
    GRID_SIZE = 64    
    NUM_RUNS = 10 
    NUM_RUNS_CPU = 2
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "No CUDA GPU Detected"
    
    data = create_batched_data_3d(BATCH_SIZE, GRID_SIZE)
    u_init, f_tensor, dr_list, cp_list, np_list, vel_list, jax_f, jax_src = data
    
    print(f"\nGPU Device: {gpu_name}")
    print("-" * 65)
    print(f"3D Benchmarking with Batch: {BATCH_SIZE}, Grid: {GRID_SIZE}^3, Runs: {NUM_RUNS} ({NUM_RUNS_CPU} on CPU)")
    print("-" * 65)
    
    error_log = []
    
    # GPU Benchmarks
    benchmark("eiko", BATCH_SIZE, run_eiko_batched, u_init, f_tensor, runs=NUM_RUNS, backend='pytorch', error_log=error_log)
    benchmark("CuPy EDT (f=1 only)", BATCH_SIZE, run_cupy_looped, cp_list, runs=NUM_RUNS, backend='cupy', error_log=error_log)
    if HAS_JAX_GPU:
        benchmark("JAX GS (Red-Black)", BATCH_SIZE, run_jax_fsm_batched_wrapper, jax_f, jax_src, runs=NUM_RUNS, backend='jax', error_log=error_log)
    else:
        print(f"{'JAX GS (Red-Black)':>20} | {'SKIPPED (No JAX GPU)':>31}")
    benchmark("fastsweep", BATCH_SIZE, run_fastsweep_looped, dr_list, runs=NUM_RUNS, backend='drjit', error_log=error_log)
    benchmark("piefs (Triton)", BATCH_SIZE, run_piefs_batched, u_init, runs=NUM_RUNS, backend='pytorch', error_log=error_log)
    print("-" * 65)
    # CPU Benchmarks
    benchmark("scikit-fmm (CPU)", BATCH_SIZE, run_skfmm_looped, np_list, runs=NUM_RUNS_CPU, backend='cpu', error_log=error_log)
    benchmark("fteikpy (CPU)", BATCH_SIZE, run_fteikpy_looped, vel_list, GRID_SIZE, runs=NUM_RUNS_CPU, backend='cpu', error_log=error_log)
    benchmark("pykonal (CPU)", BATCH_SIZE, run_pykonal_looped, vel_list, GRID_SIZE, runs=NUM_RUNS_CPU, backend='cpu', error_log=error_log)
    print("-" * 65)
    
    if error_log:
        print("\n" + "=" * 65)
        print("ERROR LOG")
        print("=" * 65)
        for name, err_msg in error_log:
            print(f"[{name}]")
            print(f"  {err_msg}\n")
        print("=" * 65)