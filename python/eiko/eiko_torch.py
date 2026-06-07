import os
import sys

try:
    import torch
    # Immediately trap CPU-only installations before doing anything else
    if torch.version.cuda is None:
        raise RuntimeError(
            "\n" + "="*75 + "\n"
            "[Eiko] ERROR: CPU-only PyTorch installation detected.\n"
            + "="*75 + "\n"
            "You have a CPU-only version of PyTorch.\n"
            "Eiko strictly requires a GPU-enabled (CUDA) version of PyTorch.\n\n"
            "HOW TO FIX:\n"
            "1. Uninstall your current version:  pip uninstall torch\n"
            "2. Get the correct GPU command at:  👉 https://pytorch.org/get-started/\n"
            + "="*75 + "\n"
        )
except ImportError as e:
    raise ImportError(
        "\n" + "="*65 + "\n"
        f"[Eiko] PyTorch bindings require 'torch' to be installed.\n"
        + "="*65 + "\n"
        "Eiko requires a GPU-enabled version of PyTorch.\n"
        "A standard 'pip install torch' installs a CPU-only version. \n\n"
        "To get the correct GPU (CUDA) installation, visit:\n"
        "👉 https://pytorch.org/get-started/\n"
        + "="*65 + "\n"
    ) from e

from torch.utils.cpp_extension import load
from eiko.build_config import CXX_ARGS, NVCC_ARGS, EXTRA_INCLUDE_PATHS, BIN_CACHE_DIR
from eiko import SRC_DIR, __version__

try:
    # --------------------------------------------------------------------
    #  The Fastest Path (Cached Import)
    # --------------------------------------------------------------------
    import eiko_torch_impl as _fim_cuda_impl

except ImportError:
    # --------------------------------------------------------------------
    # Runtime Download Fallback
    # --------------------------------------------------------------------
    from eiko.bootstrap import fetch_precompiled_wheel
    is_loaded = False

    torch_v = torch.__version__
    cuda_v = torch.version.cuda or "cpu."
    if cuda_v != "cpu":
        cuda_v = f"cu{cuda_v.replace('.', '')}"

    if fetch_precompiled_wheel(__version__, torch_v, cuda_v, BIN_CACHE_DIR, target_impl="eiko_torch_impl"):
        # Force Python to rescan sys.path directories, so it sees the new file
        import importlib
        importlib.invalidate_caches()
        try:
            import eiko_torch_impl as _fim_cuda_impl
            is_loaded = True
        except ImportError as e:
            print(f"[Eiko] Downloaded PyTorch binary failed to load ({e}).")
            
            # Debugging helper: Print exactly what is inside the directory
            found_files = os.listdir(BIN_CACHE_DIR) if os.path.exists(BIN_CACHE_DIR) else []
            # print(f"[Eiko] Debug - Files currently in cache ({BIN_CACHE_DIR}): {found_files}")
            print(f"[Eiko] Falling back to JIT compilation.")

    # --------------------------------------------------------------------
    # Final JIT Compilation Fallback (With User-Friendly Error Catching)
    # --------------------------------------------------------------------
    if not is_loaded:
        print("[Eiko] JIT Compiling CUDA kernels for your GPU... (This may take a minute)")
        sys.stdout.flush()

        torch_source = os.path.join(SRC_DIR, 'bindings', 'torch_bindings.cu')
        
        try:
            _fim_cuda_impl = load(
                name="eiko_torch_impl",
                sources=[torch_source],
                extra_cflags=CXX_ARGS,
                extra_cuda_cflags=NVCC_ARGS,
                extra_include_paths=EXTRA_INCLUDE_PATHS,
                verbose=False,
                build_directory=BIN_CACHE_DIR
            )
            print("[Eiko] Compilation complete. Congratulations, you are now ready to use Eiko! :)")
            
        except Exception as e:
            error_msg = str(e).lower()
            
            print("\n" + "="*75)
            print("[Eiko] FATAL ERROR: Local C++/CUDA compilation failed.")
            print("="*75)
            print("1. We could not find a compatible precompiled wheel for your exact system.")
            print("2. We attempted to compile the extension from source, but it failed.\n")
            
            # --- DIAGNOSIS ROUTINES ---
            # 1. MSVC Missing (Windows)
            if sys.platform == "win32" and ("cl.exe" in error_msg or "['where', 'cl']" in error_msg):
                print("DIAGNOSIS: Microsoft Visual Studio C++ compiler ('cl.exe') was not found.")
                print("FIX: 1) Install the 'Desktop development with C++' workload via the Visual Studio Installer.")
                print("     2) Ensure you run Python inside the 'x64 Native Tools Command Prompt for VS'.")
            
            # 2. GCC/G++ Missing (Linux)
            elif sys.platform != "win32" and ("['which', 'c++']" in error_msg or "['which', 'g++']" in error_msg or "gcc" in error_msg):
                print("DIAGNOSIS: A C++ host compiler (like GCC or G++) was not found.")
                print("FIX: Install build tools on your system (e.g., run 'sudo apt install build-essential').")
                
            # 3. NVCC Missing (Cross-Platform)
            elif "['which', 'nvcc']" in error_msg or "command 'nvcc' failed" in error_msg or "executable 'nvcc' not found" in error_msg:
                print("DIAGNOSIS: The NVIDIA CUDA compiler ('nvcc') was not found on your system path.")
                print("FIX: Ensure the NVIDIA CUDA Toolkit is installed and 'nvcc' is in your system PATH.")
                print("🔗 Download CUDA here: https://developer.nvidia.com/cuda-downloads")
            # 4. CUDA Architecture or Runtime Mismatch (e.g., your recent sm_120 error)
            elif "sm_" in error_msg or "compute_" in error_msg or "compatibility" in error_msg or "undefined symbol" in error_msg:
                print("DIAGNOSIS: Hardware/Software architecture compatibility mismatch.")
                print("The compilation failed because your PyTorch or CUDA driver version doesn't align with your GPU capability.")
                print("\nCOMPILER OUTPUT SNIPPET:")
                print(str(e))
            # 5. Generic Compilation Failure
            else:
                print("COMPILER OUTPUT:")
                print(str(e))
                
            # --- FASTEST FIX ADVICE ---
            print("\n" + "-"*75)
            print("FASTEST FIX: UPDATE PYTORCH")
            print("Eiko provides precompiled wheels for the newest PyTorch releases.")
            print(f"You are currently running PyTorch {torch_v}. Updating PyTorch to the latest")
            print("version will likely bypass this compilation step entirely.")
            print("👉 https://pytorch.org/get-started/")
                
            # --- GITHUB ISSUE TEMPLATE ---
            import platform
            py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
            os_name = platform.system()
            print("\n" + "-"*75)
            print("STILL STUCK? REQUEST A PRECOMPILED WHEEL:")
            print("Open an issue here: 🔗 https://github.com/sebftw/Eiko/issues")
            print("Please copy and paste the following system information into your issue description:\n")
            print(f"OS:      {os_name}")
            print(f"Python:  {py_ver}")
            print(f"PyTorch: {torch_v}")
            print(f"CUDA:    {cuda_v}")
            print("\n" + "="*75 + "\n")
            
            # Suppress the massive traceback and raise a clean error
            raise RuntimeError("Eiko initialization failed due to missing C++ build tools.") from None

#------------------------------------------------------------------------
# Global Solver Cache
# ------------------------------------------------------------------------
# Instantiates solvers once to persist memory and CUDA graphs across calls.
# Key format: (is_3d, is_backward, msfm, has_v, gated_x)
_solvers = {}

def _get_solver(is_3d: bool, is_backward: bool, msfm: bool, has_v: bool, gated_x: bool):
    """Fetches a compiled solver from the cache, instantiating it if necessary."""
    key = (is_3d, is_backward, msfm, has_v, gated_x)
    if key not in _solvers:
        _solvers[key] = _fim_cuda_impl.BatchedFIMSolver(
            is_3d=is_3d, 
            is_backward=is_backward, 
            msfm=msfm, 
            has_v=has_v,
            gated_x=gated_x
        )
    return _solvers[key]


# ------------------------------------------------------------------------
# Autograd Function
# ------------------------------------------------------------------------
class EikonalSolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u_init, f, v_init=None, dx=1.0, msfm=False, gated=False, is_3d=False):
        # ----------------------------------------------------------------
        # Enforce memory continuity and float32 data types here in Python 
        # to prevent the C++ backend from throwing TORCH_CHECK exceptions.
        # ----------------------------------------------------------------
        u_init = u_init.contiguous().float()
        f = f.contiguous().float()
        has_v = (v_init is not None)
        
        if has_v:
            v_init = v_init.contiguous().float()

        # Note that f must be non-negative. It should be positive if v is given. 
        # If it is negative, it will just be clamped to zero during the calculation, 
        # but then the gradients wont be correct.

        # Safely extract scalar value if dx is passed as a PyTorch Tensor.
        dx_val = dx.item() if isinstance(dx, torch.Tensor) else float(dx)
        
        # Clone to prevent the C++ kernel from modifying the input in-place.
        u_out = u_init.clone()
        v_out = v_init.clone() if has_v else None

        # Retrieve the correct cached forward solver.
        solver = _get_solver(is_3d=is_3d, is_backward=False, msfm=msfm, has_v=has_v, gated_x=gated)
        
        # Execute (msfm and has_v are omitted because they are now baked into the solver state).
        solver.solve(u=u_out, f=f, v=v_out, tof=None, dx=dx_val)

        # Save context for the backward pass.
        ctx.save_for_backward(u_out, f)
        ctx.dx = dx_val
        ctx.msfm = msfm
        ctx.is_3d = is_3d
        ctx.gated = gated
        ctx.has_v = has_v
        
        # Dynamically return based on input, mimicking MATLAB's varargout behavior.
        if has_v:
            return u_out, v_out
        return u_out

    @staticmethod
    def backward(ctx, *grad_outputs):
        # We use *grad_outputs to handle whether forward returned 1 or 2 tensors.
        grad_u = grad_outputs[0]
        
        u_out, f = ctx.saved_tensors
        
        # Ensure the incoming upstream gradient meets memory standards
        grad_u = grad_u.contiguous().float()

        # Initialize all return gradients to None
        grad_u_init = None
        grad_f = None
        grad_dx = None

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            # Calculate the gradients.
            lambda_adj = torch.zeros_like(u_out)
            
            # Execute the C++ backward adjoint kernel.
            # The backward pass adjoint calculation does not use the velocity field 'v'.
            # Therefore, we fetch a backward solver where has_v is explicitly False.
            solver = _get_solver(is_3d=ctx.is_3d, is_backward=True, msfm=ctx.msfm, has_v=False, gated_x=ctx.gated)
            
            # C++ correctly handles 'tof' memory mapping logic internally now
            solver.solve(u=lambda_adj, f=grad_u, v=None, tof=u_out, dx=ctx.dx)

            # Extract u_init gradient if requested.
            if ctx.needs_input_grad[0]:
                grad_u_init = lambda_adj  # Gradient w.r.t initial travel times 'u'.

            # Extract and reduce f gradient if requested.
            if ctx.needs_input_grad[1]:
                grad_f = lambda_adj * f * ctx.dx * ctx.dx  # Gradient w.r.t the slowness field 'f'.
                
                # If f was broadcasted, we must sum the gradient across the batch 
                # dimension to restore its original shape for the autograd engine.
                if f.dim() == lambda_adj.dim() - 1:
                    # f was completely missing the batch dimension, e.g., (H, W)
                    grad_f = grad_f.sum(dim=0)
                elif f.dim() == lambda_adj.dim() and f.size(0) == 1 and lambda_adj.size(0) > 1:
                    # f had a singleton batch dimension, e.g., (1, H, W)
                    grad_f = grad_f.sum(dim=0, keepdim=True)

        # Analytical dx Gradient via Euler's Homogeneity Theorem.
        if ctx.needs_input_grad[3]:
            grad_dx = torch.sum(grad_u * u_out) / ctx.dx
        
        # Return gradients corresponding to the 7 forward inputs: 
        # (u_init, f, v_init, dx, msfm, gated, is_3d)
        # We return None for non-differentiable or static structural arguments.
        return grad_u_init, grad_f, None, grad_dx, None, None, None

# ------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------
def eiko2d(u_init, f, v_init=None, dx=1.0, msfm=False, gated=False):
    if u_init.dim() > 3:
        raise ValueError(f"eiko2d expects a 2D or batched 2D grid (max 3 dims), got {u_init.dim()} dims.")
        
    return EikonalSolver.apply(u_init, f, v_init, dx, msfm, gated, False)

def eiko3d(u_init, f, v_init=None, dx=1.0, msfm=False, gated=False):
    if u_init.dim() < 3:
        raise ValueError(f"eiko3d expects a 3D or batched 3D grid (min 3 dims), got {u_init.dim()} dims.")
        
    return EikonalSolver.apply(u_init, f, v_init, dx, msfm, gated, True)

# Generic alias mapped to eiko2d for convenience
eiko = eiko2d
