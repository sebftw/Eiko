# build_release.py
import os
import sys

# Check if we are building a release wheel
IS_RELEASE_BUILD = os.environ.get("EIKO_RELEASE_BUILD", "0") == "1"

if sys.platform == "win32":
    CXX_ARGS = ['/permissive-', '/EHsc', '/DTHRUST_IGNORE_CUB_VERSION_CHECK', '/DTHRUST_FORCE_COMPATIBILITY',
                '/Zc:preprocessor', '/DNOMINMAX', '/wd3189']
    NVCC_ARGS = [
        '-allow-unsupported-compiler', 
        '-Xcompiler', '/Zc:preprocessor', 
        '-Xcompiler', '/wd3189', 
        '-Xcompiler', '/permissive-',
        '-Xcompiler', '/EHsc',
        '-DTHRUST_IGNORE_CUB_VERSION_CHECK', 
        '-DTHRUST_FORCE_COMPATIBILITY',
        '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH', 
        '-DNOMINMAX', 
        '--use_fast_math'
    ]
else:
    # Safely detect the ABI flag from the active PyTorch installation
    try:
        import torch
        abi_val = "1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"
    except ImportError:
        # Fallback to "0" (or "1" depending on your project's primary target) if torch isn't installed yet
        abi_val = "0"

    CXX_ARGS = [f'-D_GLIBCXX_USE_CXX11_ABI={abi_val}',
                '-O3']
    NVCC_ARGS = ['-Xcompiler', f'-D_GLIBCXX_USE_CXX11_ABI={abi_val}',
                 '--use_fast_math']

if IS_RELEASE_BUILD:
    NVCC_ARGS.append('-arch=all-major')
else:
    NVCC_ARGS.append('-arch=native')
EXTRA_INCLUDE_PATHS = [os.path.abspath("src")]
