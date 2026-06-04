import os
import sys
from pathlib import Path

# ------------------------------------------------------------------------
# 1. Base Compiler Arguments
# ------------------------------------------------------------------------
NVCC_ARGS = ["-O3", "--use_fast_math"]

if sys.platform == "win32":
    # Microsoft Visual C++ uses /O2 for maximum speed
    CXX_ARGS = ["/O2"]
else:
    # GCC/Clang uses -O3
    CXX_ARGS = ["-O3"]

# ------------------------------------------------------------------------
# 2. Shared Binary Cache Directory
# ------------------------------------------------------------------------
if sys.platform == "win32":
    _cache_base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
else:
    _cache_base = Path.home() / ".cache"

BIN_CACHE_DIR = str(_cache_base / "eiko" / "binaries")

# ------------------------------------------------------------------------
# 3. Source Path Resolution
# ------------------------------------------------------------------------
# Base directory where this configuration file lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Option A: Path when the package is installed via pip (src is inside eiko/).
SRC_DIR_INSTALLED = os.path.join(BASE_DIR, 'src')

# Option B: Path when running locally during development (src is sibling to python/).
SRC_DIR_DEV = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'src'))

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

# ------------------------------------------------------------------------
# 4. CUDA Toolkit & CCCL Include Resolution
# ------------------------------------------------------------------------
cuda_home = None

# Attempt 1: Check standard environment variables (helps pure JAX/nvcc users)
if "CUDA_HOME" in os.environ:
    cuda_home = os.environ["CUDA_HOME"]
elif "CUDA_PATH" in os.environ:
    cuda_home = os.environ["CUDA_PATH"]

# Attempt 2: Fall back to PyTorch's internal resolution machinery if available
if not cuda_home:
    try:
        from torch.utils.cpp_extension import CUDA_HOME
        cuda_home = CUDA_HOME
    except ImportError:
        pass

# If found, add NVIDIA's Core Compute Libraries (Thrust, CUB, libcudacxx)
if cuda_home and os.path.exists(cuda_home):
    cccl_base = os.path.join(cuda_home, 'include', 'cccl')
    if os.path.exists(cccl_base):
        EXTRA_INCLUDE_PATHS.extend([
            cccl_base,
            os.path.join(cccl_base, 'thrust'),
            os.path.join(cccl_base, 'libcudacxx', 'include'),
            os.path.join(cccl_base, 'cub'),
        ])

# ------------------------------------------------------------------------
# 5. OS & Release Specific Flags
# ------------------------------------------------------------------------
local_version = os.environ.get("EIKO_LOCAL_VERSION", "")
IS_RELEASE_BUILD = bool(local_version)

if sys.platform == "win32":
    CXX_ARGS.extend([
        '/permissive-', '/EHsc', '/DTHRUST_IGNORE_CUB_VERSION_CHECK', 
        '/DTHRUST_FORCE_COMPATIBILITY', '/Zc:preprocessor', '/DNOMINMAX', '/wd3189'
    ])
    NVCC_ARGS.extend([
        '-allow-unsupported-compiler', 
        '-Xcompiler', '/Zc:preprocessor', 
        '-Xcompiler', '/wd3189', 
        '-Xcompiler', '/permissive-',
        '-Xcompiler', '/EHsc',
        '-DTHRUST_IGNORE_CUB_VERSION_CHECK', 
        '-DTHRUST_FORCE_COMPATIBILITY',
        '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH', 
        '-DNOMINMAX'
    ])
else:
    # Safely detect the ABI flag from the active PyTorch installation
    try:
        import torch
        abi_val = "1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"
    except ImportError:
        # Fallback to "0" (or "1" depending on your project's primary target) if torch isn't installed yet
        abi_val = "0"

    CXX_ARGS.extend([f'-D_GLIBCXX_USE_CXX11_ABI={abi_val}'])
    NVCC_ARGS.extend(['-Xcompiler', f'-D_GLIBCXX_USE_CXX11_ABI={abi_val}'])

if IS_RELEASE_BUILD:
    NVCC_ARGS.append('-arch=all-major')
else:
    NVCC_ARGS.append('-arch=native')
