import os
import sys
import torch

from torch.utils.cpp_extension import load
from eiko.build_config import CXX_ARGS, NVCC_ARGS, EXTRA_INCLUDE_PATHS, BIN_CACHE_DIR
from eiko import SRC_DIR, __version__

# Inject the shared cache directory into sys.path immediately
if BIN_CACHE_DIR not in sys.path:
    sys.path.insert(0, BIN_CACHE_DIR)

try:
    # --------------------------------------------------------------------
    # 1. The Fastest Path (Cached Import)
    # --------------------------------------------------------------------
    import eiko_torch_impl as _fim_cuda_impl

except ImportError:
    # --------------------------------------------------------------------
    # 2. Runtime Download Fallback
    # --------------------------------------------------------------------
    from eiko.bootstrap import fetch_precompiled_wheel
    is_loaded = False

    torch_v = torch.__version__
    cuda_v = torch.version.cuda or "cpu"

    # Normalize CUDA version string to match your registry format (e.g., "cu121")
    if cuda_v != "cpu":
        cuda_v = f"cu{cuda_v.replace('.', '')}"

    if fetch_precompiled_wheel(__version__, torch_v, cuda_v, BIN_CACHE_DIR, target_impl="eiko_torch_impl"):
        try:
            import eiko_torch_impl as _fim_cuda_impl
            is_loaded = True
        except ImportError as e:
            print(f"[Eiko] Downloaded PyTorch binary failed to load natively ({e}).")
            print(f"[Eiko] Falling back to local compilation.")

    # --------------------------------------------------------------------
    # 3. Final JIT Compilation Fallback
    # --------------------------------------------------------------------
    if not is_loaded:
        print("[Eiko] First-time PyTorch initialization: JIT Compiling CUDA kernels for your GPU... (This may take a minute)")
        sys.stdout.flush()

        torch_source = os.path.join(SRC_DIR, 'bindings', 'torch_bindings.cu')
        
        # We pass build_directory=BIN_CACHE_DIR to force PyTorch's compiler 
        # to drop the output .so/.pyd right next to the JAX ones.
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
