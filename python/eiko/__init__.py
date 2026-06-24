import os
import sys
import errno
import shutil
import tempfile
from pathlib import Path

# ---------------------------------------------------------
# PACKAGE METADATA & VERSION
# ---------------------------------------------------------
# This acts as the single source of truth for the package version.
# It is placed at the very top so internal submodules can safely import it.
__version__ = "0.8.7"

# ---------------------------------------------------------
# PATH & ENVIRONMENT INITIALIZATION
# ---------------------------------------------------------
from eiko.build_config import SRC_DIR, BIN_CACHE_DIR

# Inject the binary cache directory into sys.path globally.
# This ensures both Torch and JAX submodules can instantly resolve 
# precompiled or JIT-compiled binaries via native import statements.
if BIN_CACHE_DIR not in sys.path:
    sys.path.insert(0, BIN_CACHE_DIR)

# Windows Python 3.8+ requires explicit DLL directory registration
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(BIN_CACHE_DIR)
    except Exception:
        pass

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
        import torch
        return pt_eiko2d(torch.as_tensor(u_init), torch.as_tensor(f), v_init, dx, msfm, gated)

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
        import torch
        return pt_eiko3d(torch.as_tensor(u_init), torch.as_tensor(f), v_init, dx, msfm, gated)

# Alias default 2D solver
eiko = eiko2d

# ---------------------------------------------------------
# CACHE MANAGEMENT UTILITIES
# ---------------------------------------------------------
def is_dir_empty(path_str):
    path = Path(path_str)
    try:
        return not any(path.iterdir())
    except FileNotFoundError:
        # Returns True if the directory doesn't exist
        return True
    except NotADirectoryError:
        # Optional: Handles cases where the path points to a file instead of a folder
        print(f"[Eiko] Warning: {path_str} is a file, not a directory.")
        return False

def clear_binary_cache():
    """
    Removes all precompiled and JIT-compiled binaries from the persistent local cache.
    Useful if you switch CUDA drivers, encounter corrupted states, or just want to re-trigger JIT-compilation.
    """
    if not is_dir_empty(BIN_CACHE_DIR):
        print(f"[Eiko] Clearing binary cache at: {BIN_CACHE_DIR}")
        files_in_use = False

        # Error codes representing a locked, busy, or permission-denied state
        # EACCES: Permission denied (Windows locked files, Linux strict permissions)
        # EBUSY: Device or resource busy (NFS loaded libraries)
        # ENOTEMPTY: Directory not empty (if a locked file prevented a sub-folder deletion)
        locked_errnos = {errno.EACCES, errno.EBUSY, errno.ENOTEMPTY}
        
        for item in os.listdir(BIN_CACHE_DIR):
            item_path = os.path.join(BIN_CACHE_DIR, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except OSError as e:
                if e.errno in locked_errnos:
                    files_in_use = True
                elif e.errno == errno.ENOENT:
                    # FileNotFoundError: Another process probably already deleted it. Safely ignore.
                    pass
                else:
                    # An unexpected OS error occurred (e.g., read-only filesystem, I/O error)
                    # Re-raise it so it doesn't fail silently.
                    raise
                
        if files_in_use:
            print("\n" + "-"*65)
            print("[Eiko] WARNING: Some cached files are currently in use.")
            print("To fully clear the cache, restart your Python/Jupyter session")
            print("and run this command *before* calling any Eiko solvers.")
            print("-"*65 + "\n")

def clear_download_cache():
    """
    Removes any cached wheel archives from the temporary download directory.
    """
    eiko_tmp_dir = os.path.join(tempfile.gettempdir(), "eiko_cache")
    if not is_dir_empty(eiko_tmp_dir):
        print(f"[Eiko] Clearing downloaded wheels at: {eiko_tmp_dir}")
        shutil.rmtree(eiko_tmp_dir)

def clear_caches():
    """
    Completely resets the Eiko installation footprint by wiping both 
    downloaded wheels and compiled binaries.
    """
    clear_download_cache()
    clear_binary_cache()


# ---------------------------------------------------------
# MODULE IMPORTS & EXPORTS
# ---------------------------------------------------------
from .animate_eikonal import animate_eikonal

# Add these functions and fields to the public API exports
__all__ = [
    'eiko', 
    'eiko3d', 
    'animate_eikonal', 
    '__version__',
    'clear_binary_cache',
    'clear_download_cache',
    'clear_caches'
]
