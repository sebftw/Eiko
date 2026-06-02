import os
import sys

# ---------------------------------------------------------
# PATH RESOLUTION
# ---------------------------------------------------------
# Resolve paths once for the entire package.

# Base directory where this file lives (Eiko/python/eiko/ or .../site-packages/eiko/).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Option A: Path when the package is installed via pip (src is inside eiko/).
SRC_DIR_INSTALLED = os.path.join(BASE_DIR, 'src')

# Option B: Path when running locally during development (src is sibling to python/).
SRC_DIR_DEV = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'src'))

# Dynamically choose the folder that actually contains your files
if os.path.exists(SRC_DIR_INSTALLED):
    SRC_DIR = SRC_DIR_INSTALLED
elif os.path.exists(SRC_DIR_DEV):
    SRC_DIR = SRC_DIR_DEV
else:
    raise FileNotFoundError(
        f"Could not find Eiko C++/CUDA source directory. Tried:\n"
        f"1. {SRC_DIR_INSTALLED}\n"
        f"2. {SRC_DIR_DEV}"
    )

EXTRA_INCLUDE_PATHS = [SRC_DIR]

try:
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME:
        cccl_base = os.path.join(CUDA_HOME, 'include', 'cccl')
        if os.path.exists(cccl_base):
            EXTRA_INCLUDE_PATHS.extend([
                cccl_base,
                os.path.join(cccl_base, 'thrust'),
                os.path.join(cccl_base, 'libcudacxx', 'include'),
                os.path.join(cccl_base, 'cub'),
            ])
except ImportError:
    # Let individual wrappers handle the missing torch error cleanly
    pass

# ---------------------------------------------------------
# PUBLIC API EXPORTS
# ---------------------------------------------------------

def _is_jax_array(obj):
    """Safely checks if an object is a JAX array without importing JAX."""
    return type(obj).__module__.startswith('jax')

def eiko2d(u_init, f, v_init=None, dx=1.0, msfm=False, gated=False):
    """
    EIKO2D Computes the shortest time-of-flight in an arbitrary 2D medium.

    Calculates the time-of-flight (u), given a slowness map (f = 1/c), and
    initial conditions (u_init, initialized as infinity at unknown points).

    EXAMPLE USAGES:
        u = eiko2d(u_init, f)                                 # (Standard usage).
        u = eiko2d(u_init, f, dx=0.5, msfm=True)              # (Named arguments).
        u, v_out = eiko2d(u_init, f, v_init=advection_field)  # (Advection).

    REQUIRED INPUTS:
        u_init  - Initial conditions (known arrival times/delays).
                  Shape: (H, W) for a single image, or (B, H, W) for a batch.
        f       - Slowness. Can be (H, W) [broadcast to batch] or (B, H, W).

    OPTIONAL INPUTS:
        dx      - Input grid spacing. Default: 1.0.
        v_init  - The initial advection field. Same size as u. Default: None (not used).
        msfm    - Whether to enable Multi-Stencil Fast Marching (MSFM).
                  Reduces bias along diagonal directions. Default: False.
        gated   - Whether to enforce positive propagation along the first 
                  data dimension. Speeds up computations. It is valid when 
                  time only increases when moving axially. Default: False.

    OUTPUTS:
        u       - Computed arrival time (time-of-flight) map. Shape matches u_init.
        v       - Output advection vectors (returned only if v_init was supplied).
    """
    if _is_jax_array(u_init):
        from .eiko_jax import eiko2d as jax_eiko2d
        return jax_eiko2d(u_init, f, v_init, dx, msfm, gated)
    else:
        from .eiko_torch import eiko2d as pt_eiko2d
        return pt_eiko2d(u_init, f, v_init, dx, msfm, gated)

def eiko3d(u_init, f, v_init=None, dx=1.0, msfm=False, gated=False):
    """
    EIKO3D Computes the shortest time-of-flight in an arbitrary 3D medium.

    Calculates the time-of-flight (u), given a slowness map (f = 1/c), and
    initial conditions (u_init, initialized as infinity at unknown points).

    EXAMPLE USAGES:
        u = eiko3d(u_init, f)                                 # (Standard usage).
        u = eiko3d(u_init, f, dx=0.5, msfm=True)              # (Named arguments).
        u, v_out = eiko3d(u_init, f, v_init=advection_field)  # (Advection).

    REQUIRED INPUTS:
        u_init  - Initial conditions (known arrival times/delays).
                  Shape: (D, H, W) for a single volume, or (B, D, H, W) for a batch.
        f       - Slowness. Can be (D, H, W) [broadcast to batch] or (B, D, H, W).

    OPTIONAL INPUTS:
        dx      - Input grid spacing. Default: 1.0.
        v_init  - The initial advection field. Same size as u. Default: None (not used).
        msfm    - Whether to enable Multi-Stencil Fast Marching (MSFM).
                  Reduces bias along diagonal directions. Default: False.
        gated   - Whether to enforce positive propagation along the first 
                  data dimension. Speeds up computations. It is valid when 
                  time only increases when moving axially. Default: False.

    OUTPUTS:
        u       - Computed arrival time (time-of-flight) map. Shape matches u_init.
        v       - Output advection vectors (returned only if v_init was supplied).
    """
    if _is_jax_array(u_init):
        from .eiko_jax import eiko3d as jax_eiko3d
        return jax_eiko3d(u_init, f, v_init, dx, msfm, gated)
    else:
        from .eiko_torch import eiko3d as pt_eiko3d
        return pt_eiko3d(u_init, f, v_init, dx, msfm, gated)

eiko = eiko2d

from .animate_eikonal import animate_eikonal

# Define what is imported when a user runs `from eiko import *`.
__all__ = ['eiko', 'eiko3d', 'animate_eikonal']

# Define the version number, so it is easily accessible.
try:
    # from importlib.metadata import version, PackageNotFoundError
    __version__ =  "0.8.5" # version("eiko")
except PackageNotFoundError:
    # This acts as a fallback when the package is being built
    # or imported from source without being installed first.
    __version__ = "unknown"

