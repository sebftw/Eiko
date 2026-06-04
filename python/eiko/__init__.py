import os
import sys
import shutil
import tempfile

# ---------------------------------------------------------
# PACKAGE METADATA & VERSION
# ---------------------------------------------------------
# This acts as the single source of truth for the package version.
# It is placed at the very top so internal submodules can safely import it.
__version__ = "0.8.5"

# ---------------------------------------------------------
# PATH & ENVIRONMENT INITIALIZATION
# ---------------------------------------------------------
from eiko.build_config import SRC_DIR, BIN_CACHE_DIR

# Inject the binary cache directory into sys.path globally.
# This ensures both Torch and JAX submodules can instantly resolve 
# precompiled or JIT-compiled binaries via native import statements.
if BIN_CACHE_DIR not in sys.path:
    sys.path.insert(0, BIN_CACHE_DIR)

# ---------------------------------------------------------
# INTERNAL UTILITIES
# ---------------------------------------------------------
def _is_jax_array(obj):
    """Safely checks if an object is a JAX array without importing JAX."""
    return type(obj).__module__.startswith('jax')

# ---------------------------------------------------------
# DYNAMIC ROUTING API
# ---------------------------------------------------------
def eiko2d(u_init, f, v_init=None, dx=1.0, msfm=False, gated=False):
    """
    EIKO2D Computes the shortest time-of-flight in an arbitrary 2D medium.

    Calculates the time-of-flight (u), given a slowness map (f = 1/c), and
    initial conditions (u_init, initialized as infinity at unknown points).
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
    """
    if _is_jax_array(u_init):
        from .eiko_jax import eiko3d as jax_eiko3d
        return jax_eiko3d(u_init, f, v_init, dx, msfm, gated)
    else:
        from .eiko_torch import eiko3d as pt_eiko3d
        return pt_eiko3d(u_init, f, v_init, dx, msfm, gated)

# Alias default 2D solver
eiko = eiko2d

# ---------------------------------------------------------
# CACHE MANAGEMENT UTILITIES
# ---------------------------------------------------------
def clear_binary_cache():
    """
    Removes all precompiled and JIT-compiled binaries from the persistent local cache.
    Useful if you switch CUDA drivers or encounter corrupted compilation states.
    """
    if os.path.exists(BIN_CACHE_DIR):
        shutil.rmtree(BIN_CACHE_DIR)
        print(f"[Eiko] Cleared binary cache at: {BIN_CACHE_DIR}")
    else:
        print("[Eiko] Binary cache is already empty.")

def clear_download_cache():
    """
    Removes any cached wheel archives from the temporary download directory.
    """
    eiko_tmp_dir = os.path.join(tempfile.gettempdir(), "eiko_cache")
    if os.path.exists(eiko_tmp_dir):
        shutil.rmtree(eiko_tmp_dir)
        print(f"[Eiko] Cleared downloaded wheels at: {eiko_tmp_dir}")
    else:
        print("[Eiko] Download cache is already empty.")

def clear_cache():
    """
    Completely resets the Eiko installation footprint by wiping both 
    downloaded wheels and compiled binaries.
    """
    clear_binary_cache()
    clear_download_cache()
    print("[Eiko] All caches cleared. The next import will trigger a fresh download or JIT compilation.")

# ---------------------------------------------------------
# MODULE IMPORTS & EXPORTS
# ---------------------------------------------------------
from .animate_eikonal import animate_eikonal

# Add the new utility functions to your public API exports
__all__ = [
    'eiko', 
    'eiko3d', 
    'animate_eikonal', 
    '__version__',
    'clear_binary_cache',
    'clear_download_cache',
    'clear_cache'
]
