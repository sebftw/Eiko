import os
import sys
import math
import struct
import shutil
import sysconfig
import subprocess
from functools import partial

try:
    import jax
    import pybind11
    import jaxlib
    
    # Immediately trap CPU-only installations or missing GPU hardware
    if jax.default_backend() == "cpu":
        raise RuntimeError(
            "\n" + "="*75 + "\n"
            "[Eiko] ERROR: CPU-only JAX or missing GPU detected.\n"
            + "="*75 + "\n"
            "You have JAX installed, but it is currently defaulting to the CPU backend.\n"
            "Eiko strictly requires a GPU-enabled version of JAX to run.\n\n"
            "HOW TO FIX:\n"
            "1. If you have the CPU-only version, uninstall it first:\n"
            "   👉 pip uninstall jax jaxlib\n"
            "2. Install the CUDA-enabled version of JAX (e.g., for CUDA 12):\n"
            "   👉 pip install -U \"jax[cuda12]\"\n"
            "3. Ensure your NVIDIA drivers are correctly installed and visible to Python.\n\n"
            "For full installation details, visit:\n"
            "https://jax.readthedocs.io/en/latest/installation.html\n"
            + "="*75 + "\n"
        )
except ImportError as e:
    # Extract the specific package name that triggered the ImportError
    missing_pkg = getattr(e, 'name', 'a required dependency')
    
    raise ImportError(
        f"\n[Eiko] ERROR: Failed to import '{missing_pkg}'.\n"
        f"JAX bindings require 'jax', 'jaxlib', and 'pybind11' to be installed.\n"
        f"Please install via: pip install \"eiko[jax]\"\n"
    ) from e

import jax.numpy as jnp
from jax import core
from jax.interpreters import batching, xla, mlir

if not hasattr(core, "Primitive"):
    from jax._src.core import Primitive
    core.Primitive = Primitive

try:
    from jaxlib import xla_client
except ImportError:
    from jax.lib import xla_client

# Import our centralized configuration
from eiko.build_config import CXX_ARGS, NVCC_ARGS, EXTRA_INCLUDE_PATHS, BIN_CACHE_DIR
from eiko import SRC_DIR, __version__

try:
    # 1. Try loading the AOT compiled version from the pip-installed wheel
    from eiko import eiko_jax_impl as _fim_jax_impl
except ImportError:
    try:
        # 2. Try loading the JIT compiled version from the user's cache dir
        import eiko_jax_impl as _fim_jax_impl
    except ImportError:
        # Ensure the shared cache directory exists
        os.makedirs(BIN_CACHE_DIR, exist_ok=True)
        
        # --------------------------------------------------------------------
        # 3. Runtime Download Fallback
        # --------------------------------------------------------------------
        from eiko.bootstrap import fetch_precompiled_wheel
        is_loaded = False
        
        # Download into BIN_CACHE_DIR so both backends share the same folder
        if fetch_precompiled_wheel(__version__, torch_version=None, cuda_version=None, target_dir=BIN_CACHE_DIR, target_impl="eiko_jax_impl"):
            try:
                import eiko_jax_impl as _fim_jax_impl
                is_loaded = True
            except ImportError as e:
                print(f"[Eiko] Downloaded JAX binary failed to load natively ({e}).")
                print(f"[Eiko] Falling back to local compilation.")

        # --------------------------------------------------------------------
        # 4. Pure JIT Compilation Fallback via NVCC
        # --------------------------------------------------------------------
        if not is_loaded:
            print("[Eiko] Precompiled binary not found. Compiling kernels via nvcc... (This might take a minute)")
            sys.stdout.flush()

            import sysconfig
            import uuid # Added for atomic naming
            import platform
            import subprocess
            
            jax_source = os.path.join(SRC_DIR, 'bindings', 'jax_bindings.cu')
            
            
            # Grab the version-specific extension suffix
            ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")

            # Fallback just in case sysconfig returns None (rare, but safe practice)
            if not ext_suffix:
                ext_suffix = ".pyd" if os.name == "nt" else ".so"
            
            final_output_lib = os.path.join(BIN_CACHE_DIR, f"eiko_jax_impl{ext_suffix}")
            tmp_suffix = f".{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
            tmp_output_lib = final_output_lib + tmp_suffix
            
            includes = EXTRA_INCLUDE_PATHS + [pybind11.get_include(), sysconfig.get_path("include")]
            include_flags = [f"-I{path}" for path in includes if os.path.exists(path)]
            
            # Try to find it in the current environment PATH
            nvcc_path = shutil.which("nvcc")

            # Fallback to the standard Ubuntu/WSL CUDA path if missing
            if nvcc_path is None:
                fallback_path = "/usr/local/cuda/bin/nvcc"
                if os.path.exists(fallback_path):
                    nvcc_path = fallback_path
                else:
                    raise FileNotFoundError(
                        "nvcc not found in PATH, and standard /usr/local/cuda/bin/nvcc does not exist. "
                        "Please ensure CUDA toolkit is installed."
                    )
            
            # Point nvcc output directly to the temporary file
            cmd = [nvcc_path, "-shared", "-std=c++17", jax_source, "-o", tmp_output_lib]
            cmd += NVCC_ARGS
            # ENFORCE -fPIC for shared libraries
            cxx_args_pic = CXX_ARGS + ["-fPIC"] if "-fPIC" not in CXX_ARGS and os.name != "nt" else CXX_ARGS
            cmd += [f"-Xcompiler={arg}" for arg in cxx_args_pic]
            cmd += include_flags

            try:
                # Compile atomically to the unique temp file
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Atomically swap it into the active path
                os.replace(tmp_output_lib, final_output_lib)
                
                # Flush Python's directory and import caches
                import importlib
                importlib.invalidate_caches()
                
                import eiko_jax_impl as _fim_jax_impl
                print("[Eiko] Compilation complete. JAX bindings are ready! :)")
                
            except Exception as e:
                # Clean up the temporary file if compilation crashed
                if os.path.exists(tmp_output_lib):
                    os.remove(tmp_output_lib)
                    
                # Extract output depending on whether the process failed to start or failed to compile
                if isinstance(e, subprocess.CalledProcessError):
                    error_msg = e.stderr.decode('utf-8', errors='replace').lower()
                    raw_error = e.stderr.decode('utf-8', errors='replace')
                else:
                    error_msg = str(e).lower()
                    raw_error = str(e)

                # System diagnostics for the GitHub template
                py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
                os_name = platform.system()
                try:
                    import jax
                    j_ver = jax.__version__
                except ImportError:
                    j_ver = "unknown"
                    
                print("\n" + "="*75)
                print("[Eiko] FATAL ERROR: Local C++/CUDA compilation for JAX failed.")
                print("="*75)
                print("1. We could not find a compatible precompiled wheel for your exact system.")
                print("2. We attempted to compile the JAX extension via nvcc, but it failed.\n")
                
                # --- DIAGNOSIS ROUTINES ---
                # 1. NVCC Missing (Caught via FileNotFoundError usually)
                if isinstance(e, FileNotFoundError) or "nvcc" in error_msg:
                    print("DIAGNOSIS: The NVIDIA CUDA Toolkit compiler ('nvcc') was not found.")
                    print("FIX: Ensure the CUDA Toolkit is installed and 'nvcc' is in your system PATH.")
                    print("🔗 Download CUDA: https://developer.nvidia.com/cuda-downloads")
                    
                # 2. MSVC Missing (Windows)
                elif sys.platform == "win32" and ("cl.exe" in error_msg or "compiler" in error_msg):
                    print("DIAGNOSIS: Microsoft Visual Studio C++ compiler ('cl.exe') was not found by nvcc.")
                    print("FIX: 1) Install the 'Desktop development with C++' workload via the Visual Studio Installer.")
                    print("     2) Ensure you run Python inside the 'x64 Native Tools Command Prompt for VS'.")
                
                # 3. GCC/G++ Missing (Linux)
                elif sys.platform != "win32" and ("g++" in error_msg or "gcc" in error_msg or "c++" in error_msg):
                    print("DIAGNOSIS: A C++ host compiler (like GCC or G++) was not found by nvcc.")
                    print("FIX: Install build tools on your system (e.g., run 'sudo apt install build-essential').")
                    
                # 4. Generic Compilation Failure
                else:
                    print("COMPILER OUTPUT:")
                    print(raw_error)
                    
                # --- GITHUB ISSUE TEMPLATE ---
                py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
                os_name = platform.system()
                j_ver = jax.__version__
                print("\n" + "-"*75)
                print("STILL STUCK? REQUEST A PRECOMPILED WHEEL:")
                print("Open an issue here: 🔗 https://github.com/sebftw/Eiko/issues")
                print("Please copy and paste the following system information into your issue description:\n")
                print("```text")
                print(f"OS:      {os_name}")
                print(f"Python:  {py_ver}")
                print(f"JAX:     {j_ver}")
                print("```")
                print("="*75 + "\n")
                
                # Suppress the massive traceback and raise a clean error
                raise RuntimeError("Eiko JAX initialization failed due to missing C++ build tools.") from None

# ------------------------------------------------------------------------
# 5. XLA Custom Call Registration
# ------------------------------------------------------------------------
for name, target in _fim_jax_impl.registrations().items():
    xla_client.register_custom_call_target(name, target, platform="gpu")

# =========================================================
# 1. JAX PRIMITIVE DEFINITION & MLIR LOWERING
# =========================================================
_fim_prim = core.Primitive("jax_fim_solve")
_fim_prim.multiple_results = False
_fim_prim.def_impl(partial(xla.apply_primitive, _fim_prim))

def _fim_abstract_eval(*args, opaque_data, out_shape, out_dtype):
    return core.ShapedArray(out_shape, out_dtype)

_fim_prim.def_abstract_eval(_fim_abstract_eval)

def _build_custom_call_agnostic(call_target_name, result_types, operands, 
                                operand_layouts, result_layouts, backend_config):
    """
    A custom call builder that checks for legacy wrappers 
    and falls back to raw MLIR node generation for modern JAX.
    """
    # 1. Try mid-era JAX wrapper (JAX >= 0.4.15 and < 0.4.30)
    try:
        from jaxlib.hlo_helpers import custom_call
        return custom_call(
            call_target_name,
            result_types=result_types,
            operands=operands,
            operand_layouts=operand_layouts,
            result_layouts=result_layouts,
            backend_config=backend_config
        )
    except ImportError:
        pass
        
    # 2. Try legacy JAX wrapper (JAX < 0.4.15)
    try:
        from jax.interpreters.mlir import custom_call
        return custom_call(
            call_target_name,
            result_types=result_types,
            operands=operands,
            operand_layouts=operand_layouts,
            result_layouts=result_layouts,
            backend_config=backend_config
        )
    except (ImportError, AttributeError):
        pass
        
    # 3. Modern JAX (>= 0.4.30) where helpers are completely removed.
    import jaxlib.mlir.ir as ir
    try:
        from jaxlib.mlir.dialects import mhlo as hlo
    except ImportError:
        from jaxlib.mlir.dialects import hlo

    def _layout_attr(layouts):
        if layouts is None: 
            return None
        import numpy as np
        attr_list = []
        for layout in layouts:
            arr = np.array(layout, dtype=np.int64)
            
            # Define the element type as MLIR's 'index' type.
            index_type = ir.IndexType.get()
            
            # Define the 1D tensor shape explicitly using the index type.
            tensor_type = ir.RankedTensorType.get(arr.shape, index_type)
            
            # Create the attribute with the enforced tensor type.
            attr = ir.DenseIntElementsAttr.get(arr, type=tensor_type)
            attr_list.append(attr)
            
        return ir.ArrayAttr.get(attr_list)

    kwargs = {
        "call_target_name": ir.StringAttr.get(call_target_name),
        "has_side_effect": ir.BoolAttr.get(False),
        "api_version": ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 2),
        "called_computations": ir.ArrayAttr.get([]),
    }
    
    if backend_config is not None:
        kwargs["backend_config"] = ir.StringAttr.get(backend_config)

    if operand_layouts is not None:
        kwargs["operand_layouts"] = _layout_attr(operand_layouts)
        
    if result_layouts is not None:
        kwargs["result_layouts"] = _layout_attr(result_layouts)

    return hlo.CustomCallOp(result_types, operands, **kwargs)

# MLIR lowering rule: The bridge between JAX's Python graph and XLA C++.
def _fim_lowering(ctx, *args, opaque_data, out_shape, out_dtype):
    import jaxlib.mlir.ir as ir
    
    # Convert numpy dtype to MLIR IR type.
    tensor_type = ir.RankedTensorType.get(
        out_shape, mlir.dtype_to_ir_type(out_dtype)
    )
    
    operand_layouts = [tuple(range(arg.type.rank)[::-1]) for arg in args]
    result_layouts = [tuple(range(len(out_shape))[::-1])]
        
    # MLIR custom call builder.
    call = _build_custom_call_agnostic(
        "jax_fim_solve",
        result_types=[tensor_type],
        operands=args,
        operand_layouts=operand_layouts,
        result_layouts=result_layouts,
        backend_config=opaque_data
    )
    return call.results


mlir.register_lowering(_fim_prim, _fim_lowering, platform="gpu")

# =========================================================
# BATCHING RULE
# =========================================================
def _fim_batch_rule(batched_args, batch_dims, *, opaque_data, out_shape, out_dtype):
    # Unpack original configuration.
    unpacked = list(struct.unpack('=iiiifiiiiiii', opaque_data))
    
    # batched_args[0] is u_init, batched_args[1] is f
    u_init_arg = batched_args[0]
    f_arg = batched_args[1]
    
    bdim = batch_dims[0] if batch_dims[0] is not None else 0
    new_batch_axis_size = u_init_arg.shape[bdim]
    
    # Update the batch_size (index 3 in our struct)
    unpacked[3] = unpacked[3] * new_batch_axis_size
    
    # Dynamically recalculate broadcast_f (index 9)
    # The new u_init will have new_out_shape, which is 1 dimension higher.
    new_u_ndim = len(out_shape) + 1  
    
    if f_arg.ndim == new_u_ndim - 1:
        new_broadcast_f = 1
    else:
        new_broadcast_f = int(f_arg.shape[0] == 1 and unpacked[3] > 1)
        
    unpacked[9] = new_broadcast_f
    
    # Pack the updated config
    new_opaque_data = struct.pack('=iiiifiiiiiii', *unpacked)
    
    # Push the batch dimension to axis 0 for all batched arguments.
    aligned_args = [
        batching.moveaxis(arg, d, 0) if d is not None else arg 
        for arg, d in zip(batched_args, batch_dims)
    ]
    
    # Prepend the new batch dimension to the output shape.
    new_out_shape = (new_batch_axis_size,) + tuple(out_shape)
    
    out = _fim_prim.bind(
        *aligned_args, 
        opaque_data=new_opaque_data, 
        out_shape=new_out_shape, 
        out_dtype=out_dtype
    )
    # Tell JAX that the batched dimension of the output is at axis 0.
    return out, 0


batching.primitive_batchers[_fim_prim] = _fim_batch_rule

# =========================================================
# 2. BASE XLA CUSTOM CALL
# =========================================================
def _fim_custom_call(u_init, f, v, dx, msfm, is_3d, gated_x, is_backward, tof=None):
    u_init = jnp.asarray(u_init)
    f = jnp.asarray(f)
    
    has_v = v is not None
    has_tof = tof is not None
    operands = [u_init, f]

    if has_v:
        v = jnp.asarray(v)
        operands.append(v)
        
    if is_backward and has_tof:
        tof = jnp.asarray(tof)
        operands.append(tof)
    
    # Calculate the flattened batch size using math.prod
    if is_3d:
        depth, height, width = u_init.shape[-3:]
        # Multiplies all dimensions before the last 3. If empty, math.prod returns 1.
        batch_size = math.prod(u_init.shape[:-3]) 
    else:
        depth = 1
        height, width = u_init.shape[-2:]
        # Multiplies all dimensions before the last 2. If empty, math.prod returns 1.
        batch_size = math.prod(u_init.shape[:-2])
    
    # Check broadcasting against the flattened batch logic
    if f.ndim == u_init.ndim - 1:
        broadcast_f = True
    else:
        # If f has leading dimensions, calculate its flattened batch size too
        f_batch_size = math.prod(f.shape[:-3]) if is_3d else math.prod(f.shape[:-2])
        broadcast_f = (f_batch_size == 1 and batch_size > 1)
    
    opaque_data = struct.pack(
        '=iiiifiiiiiii', 
        width, height, depth, batch_size, float(dx), 
        int(is_3d), int(is_backward), int(msfm), int(has_v), 
        int(broadcast_f), int(gated_x), int(has_tof)
    )

    return _fim_prim.bind(
        *operands,
        opaque_data=opaque_data,
        out_shape=u_init.shape,
        out_dtype=u_init.dtype
    )

# =========================================================
# 3. VJP DEFINITION (Autograd support)
# =========================================================
@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5, 6))
def _solve_eikonal_base(u_init, f, v, dx, msfm, is_3d, gated_x):
    if is_3d is None:
        is_3d = u_init.ndim >= 4 and u_init.shape[-3] > 1
        
    return _fim_custom_call(u_init, f, v, dx, msfm, is_3d, gated_x, is_backward=False)

def solve_eikonal_fwd(u_init, f, v, dx, msfm, is_3d, gated_x):
    if is_3d is None:
        is_3d = u_init.ndim >= 4 and u_init.shape[-3] > 1
        
    u_out = _fim_custom_call(u_init, f, v, dx, msfm, is_3d, gated_x, is_backward=False)
    
    # Identify and cache the source nodes for the backward pass
    u_init_inf_mask = jnp.isinf(u_init)

    # Pack the mask into the residual tuple
    res = (u_out, f, u_init_inf_mask)
    return u_out, res

def solve_eikonal_bwd(v, dx, msfm, is_3d, gated_x, res, grad_u):
    u_out, f, u_init_inf_mask = res
    lambda_init = jnp.zeros_like(u_out)
    
    lambda_adj = _fim_custom_call(
        lambda_init, grad_u, None, dx, msfm, is_3d, gated_x, 
        is_backward=True, tof=u_out
    )
    
    grad_u_init = jnp.where(u_init_inf_mask, 0.0, lambda_adj)
    grad_f = lambda_adj * f * (dx * dx)
    grad_f = jnp.where(~u_init_inf_mask, 0.0, grad_f)
    
    if f.ndim == lambda_adj.ndim - 1:
        grad_f = jnp.sum(grad_f, axis=0)
    elif f.ndim == lambda_adj.ndim and f.shape[0] == 1 and lambda_adj.shape[0] > 1:
        grad_f = jnp.sum(grad_f, axis=0, keepdims=True)
    
    return grad_u_init, grad_f

_solve_eikonal_base.defvjp(solve_eikonal_fwd, solve_eikonal_bwd)

# =========================================================
# 4. PUBLIC API
# =========================================================
def eiko2d(u_init, f, v_init=None, dx=1.0, msfm=False, gated=False):
    if u_init.ndim > 3:
        raise ValueError(f"eiko2d expects a 2D or batched 2D grid (max 3 dims), got {u_init.ndim} dims.")
    
    out = _solve_eikonal_base(u_init, f, v_init, dx, msfm, False, gated)
    if v_init is not None:
        return out, v_init
    return out

def eiko3d(u_init, f, v_init=None, dx=1.0, msfm=False, gated=False):
    if u_init.ndim < 3:
        raise ValueError(f"eiko3d expects a 3D or batched 3D grid (min 3 dims), got {u_init.ndim} dims.")
        
    out = _solve_eikonal_base(u_init, f, v_init, dx, msfm, True, gated)
    if v_init is not None:
        return out, v_init
    return out

eiko = eiko2d
