import os
import sys
import struct
from functools import partial
from torch.utils.cpp_extension import load, _get_build_directory

try:
    import jax
    import pybind11
    import jaxlib
except ImportError as e:
    raise ImportError(
        f"\n[Eiko] JAX bindings require 'jax', 'jaxlib', and 'pybind11' to be installed.\n"
        f" Please install the required environment via: pip install \"eiko[jax]\"\n"
    ) from e

import jax.numpy as jnp
from jax import core
from jax.interpreters import batching
from jax.interpreters import xla
from jax.interpreters import mlir

# Robust Primitive resolution for modern JAX.
if not hasattr(core, "Primitive"):
    from jax._src.core import Primitive
    core.Primitive = Primitive

# Version-agnostic xla_client import.
try:
    from jaxlib import xla_client
except ImportError:
    from jax.lib import xla_client

# Version-agnostic custom_call import.
#try:
#    from jaxlib.hlo_helpers import custom_call
#except ImportError:
#    from jax.interpreters.mlir import custom_call

from eiko import SRC_DIR, CXX_ARGS, NVCC_ARGS, EXTRA_INCLUDE_PATHS

# ---------------------------------------------------------
# JIT COMPILATION & LOADING
# ---------------------------------------------------------
try:
    # 1. Attempt to import the pre-compiled binary 
    from eiko import eiko_jax_impl as _fim_jax_impl
    
except ImportError:
    # 2. Fallback to JIT compilation if the pre-compiled binary is missing
    build_dir = _get_build_directory('eiko_jax_impl', verbose=False)
    is_cached = os.path.exists(build_dir) and len(os.listdir(build_dir)) > 0

    if not is_cached:
        print("[Eiko] First-time JAX initialization: JIT Compiling CUDA kernels for your GPU... (This may take a minute)")
        sys.stdout.flush()

    jax_source = os.path.join(SRC_DIR, 'bindings', 'jax_bindings.cu')
    jax_includes = EXTRA_INCLUDE_PATHS + [pybind11.get_include()]

    _fim_jax_impl = load(
        name="eiko_jax_impl",
        sources=[jax_source],
        extra_cflags=CXX_ARGS,
        extra_cuda_cflags=NVCC_ARGS,
        extra_include_paths=jax_includes,
        verbose=False
    )
    
    if not is_cached:
        print("Congratulations, you are now ready to use Eiko with JAX! :)")

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
    
    # Find the new batch size. batched_args[0] is u_init.
    bdim = batch_dims[0] if batch_dims[0] is not None else 0
    new_batch_axis_size = batched_args[0].shape[bdim]
    
    # Update the batch_size (index 3 in our struct)
    unpacked[3] = unpacked[3] * new_batch_axis_size
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

    batch_size = u_init.shape[0]
    if is_3d:
        depth, height, width = u_init.shape[-3:]
    else:
        depth = 1
        height, width = u_init.shape[-2:]
    
    if f.ndim == u_init.ndim - 1:
        broadcast_f = True
    else:
        broadcast_f = (f.shape[0] == 1 and batch_size > 1)
    
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
    
    res = (u_out, f)
    return u_out, res

def solve_eikonal_bwd(v, dx, msfm, is_3d, gated_x, res, grad_u):
    u_out, f = res
    lambda_init = jnp.zeros_like(u_out)
    
    lambda_adj = _fim_custom_call(
        lambda_init, grad_u, None, dx, msfm, is_3d, gated_x, 
        is_backward=True, tof=u_out
    )
    
    grad_u_init = lambda_adj
    grad_f = lambda_adj * f * dx * dx
    
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